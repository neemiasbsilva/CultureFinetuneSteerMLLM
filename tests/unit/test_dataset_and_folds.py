"""Pin the Stage-1 data contract that every training run silently depends on.

The SFT loaders and the k-fold splitter sit upstream of every reported number.
A JSONL loader that swallows a truncated line, a merge that quietly writes zero
records when an input path moves, an ``images`` column that disappears because
of the order records happen to sit in, or a fold split that leaks validation
rows into training all produce a run that trains happily and reports a score
that means something else.  None of those raise on their own, so they are
asserted here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data.dataset import CultureVLMDataset, load_hf_dataset, merge_wvs_and_visual
from src.data.make_folds import (
    check_class_balance,
    create_folds,
    load_and_validate,
    log_to_mlflow,
    save_folds,
)

_Record = dict[str, Any]


def _chat_record(text: str, images: list[str] | None = None) -> _Record:
    record: _Record = {
        "condition": "wvs_cultural",
        "culture": "arabic",
        "image_id": text,
        "messages": [
            {"role": "user", "content": "Describe the scene."},
            {"role": "assistant", "content": text},
        ],
    }
    if images is not None:
        record["images"] = images
    return record


def _write_jsonl(path: Path, records: list[_Record], *, blank_lines: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in records:
        if blank_lines:
            lines.append("")
            lines.append("   ")
        lines.append(json.dumps(record, ensure_ascii=False))
    if blank_lines:
        lines.append("")
    path.write_text("\n".join(lines) + "\n")
    return path


def _agreement_frame(n_per_class: int = 4) -> pd.DataFrame:
    rows = []
    for sentiment in range(5):
        for offset in range(n_per_class):
            rows.append({"id": f"img_{sentiment}_{offset}", "sentiment": sentiment})
    return pd.DataFrame(rows)


def _fold_frame(n_per_class: int = 20) -> pd.DataFrame:
    frame = _agreement_frame(n_per_class).rename(columns={"id": "image_id"})
    frame["image_path"] = frame["image_id"].apply(lambda image_id: f"/images/{image_id}.jpg")
    return frame


def _silence_class_balance_report(monkeypatch: pytest.MonkeyPatch) -> None:
    def _noop(folds: list[tuple[Any, Any]], labels: pd.Series, n_folds: int) -> None:
        return None

    monkeypatch.setattr("src.data.make_folds.check_class_balance", _noop)


def test_dataset_skips_blank_lines_and_reports_only_real_examples(tmp_path: Path) -> None:
    records = [_chat_record("one"), _chat_record("two"), _chat_record("three")]
    path = _write_jsonl(tmp_path / "train.jsonl", records, blank_lines=True)

    dataset = CultureVLMDataset(path)

    assert len(dataset) == 3
    assert dataset[0] == records[0]
    assert dataset[2]["messages"][1]["content"] == "three"
    assert repr(dataset) == f"CultureVLMDataset(path={path}, n=3)"


def test_dataset_fails_loudly_when_the_training_jsonl_is_absent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"absent\.jsonl"):
        CultureVLMDataset(tmp_path / "absent.jsonl")


@pytest.mark.parametrize(
    ("write_wvs", "write_visual", "expected"),
    [(True, True, 5), (False, True, 2), (True, False, 3), (False, False, 0)],
)
def test_merge_tolerates_missing_inputs_and_returns_the_written_record_count(
    tmp_path: Path, write_wvs: bool, write_visual: bool, expected: int
) -> None:
    wvs_path = tmp_path / "wvs.jsonl"
    visual_path = tmp_path / "visual.jsonl"
    if write_wvs:
        _write_jsonl(wvs_path, [_chat_record(f"wvs_{i}") for i in range(3)])
    if write_visual:
        _write_jsonl(
            visual_path,
            [_chat_record(f"vis_{i}", images=[f"/images/{i}.jpg"]) for i in range(2)],
        )
    output_path = tmp_path / "merged.jsonl"

    written = merge_wvs_and_visual(wvs_path, visual_path, output_path)

    assert written == expected
    assert len(CultureVLMDataset(output_path)) == expected


def test_merge_creates_the_output_parent_directory_and_keeps_wvs_examples_first(
    tmp_path: Path,
) -> None:
    wvs_path = _write_jsonl(tmp_path / "wvs.jsonl", [_chat_record("wvs_0")])
    visual_path = _write_jsonl(
        tmp_path / "visual.jsonl", [_chat_record("vis_0", images=["/images/0.jpg"])]
    )
    output_path = tmp_path / "deep" / "nested" / "merged.jsonl"

    written = merge_wvs_and_visual(wvs_path, visual_path, output_path)

    assert written == 2
    assert output_path.parent.is_dir()
    merged = CultureVLMDataset(output_path)
    assert [record["image_id"] for record in merged.records] == ["wvs_0", "vis_0"]
    assert "images" not in merged[0]
    assert merged[1]["images"] == ["/images/0.jpg"]


def test_hf_dataset_exposes_messages_and_drops_bookkeeping_fields(tmp_path: Path) -> None:
    records = [_chat_record("one"), _chat_record("two")]
    path = _write_jsonl(tmp_path / "wvs.jsonl", records, blank_lines=True)

    dataset = load_hf_dataset(path)

    assert dataset.column_names == ["messages"]
    assert len(dataset) == 2
    assert dataset[0]["messages"] == records[0]["messages"]


def test_hf_dataset_ignores_the_text_field_argument_and_always_returns_messages(
    tmp_path: Path,
) -> None:
    path = _write_jsonl(tmp_path / "wvs.jsonl", [_chat_record("one")])

    assert load_hf_dataset(path, text_field="text").column_names == ["messages"]
    assert load_hf_dataset(path, text_field="prompt").column_names == ["messages"]


def test_hf_dataset_adds_a_decodable_images_column_only_for_visual_records(
    tmp_path: Path,
) -> None:
    text_only = load_hf_dataset(_write_jsonl(tmp_path / "text.jsonl", [_chat_record("one")]))
    empty_images = load_hf_dataset(
        _write_jsonl(tmp_path / "empty.jsonl", [_chat_record("two", images=[])])
    )
    visual = load_hf_dataset(
        _write_jsonl(
            tmp_path / "visual.jsonl",
            [_chat_record("three", images=[str(tmp_path / "three.jpg")])],
        )
    )

    assert "images" not in text_only.column_names
    assert "images" not in empty_images.column_names
    assert visual.column_names == ["messages", "images"]
    assert "Image" in str(visual.features["images"])


@pytest.mark.parametrize(
    ("image_record_first", "expected_columns"),
    [(True, ["messages", "images"]), (False, ["messages"])],
)
def test_hf_dataset_keeps_the_images_column_only_when_the_first_record_carries_images(
    tmp_path: Path, image_record_first: bool, expected_columns: list[str]
) -> None:
    text_record = _chat_record("text_only")
    visual_record = _chat_record("visual", images=[str(tmp_path / "visual.jpg")])
    ordered = [visual_record, text_record] if image_record_first else [text_record, visual_record]
    path = _write_jsonl(tmp_path / "mixed.jsonl", ordered)

    dataset = load_hf_dataset(path)

    assert len(dataset) == 2
    assert dataset.column_names == expected_columns


def test_merged_split_loaded_as_hf_dataset_loses_every_visual_image(tmp_path: Path) -> None:
    wvs_path = _write_jsonl(tmp_path / "wvs.jsonl", [_chat_record("wvs_0")])
    visual_path = _write_jsonl(
        tmp_path / "visual.jsonl",
        [_chat_record("vis_0", images=[str(tmp_path / "vis_0.jpg")])],
    )
    merged_path = tmp_path / "merged.jsonl"
    assert merge_wvs_and_visual(wvs_path, visual_path, merged_path) == 2
    assert any("images" in record for record in CultureVLMDataset(merged_path).records)

    dataset = load_hf_dataset(merged_path)

    assert dataset.column_names == ["messages"]


def test_load_and_validate_drops_rows_whose_image_is_absent_from_disk(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    frame = pd.DataFrame(
        [
            {"id": "kept_a", "sentiment": 0},
            {"id": "gone_a", "sentiment": 1},
            {"id": "kept_b", "sentiment": 2},
            {"id": "gone_b", "sentiment": 3},
        ]
    )
    agreement_csv = tmp_path / "agreement.csv"
    frame.to_csv(agreement_csv)
    for image_id in ("kept_a", "kept_b"):
        (images_dir / f"{image_id}.jpg").touch()

    validated = load_and_validate(str(agreement_csv), str(images_dir))

    assert list(validated["image_id"]) == ["kept_a", "kept_b"]
    assert list(validated.index) == [0, 1]
    assert list(validated["image_path"]) == [
        str(images_dir / "kept_a.jpg"),
        str(images_dir / "kept_b.jpg"),
    ]
    assert "id" not in validated.columns


def test_load_and_validate_keeps_every_row_when_all_images_are_present(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    frame = _agreement_frame(n_per_class=2)
    agreement_csv = tmp_path / "agreement.csv"
    frame.to_csv(agreement_csv)
    for image_id in frame["id"]:
        (images_dir / f"{image_id}.jpg").touch()

    validated = load_and_validate(str(agreement_csv), str(images_dir))

    assert len(validated) == len(frame)
    assert all(Path(path).exists() for path in validated["image_path"])


def test_load_and_validate_requires_the_agreement_csv_to_carry_a_leading_index_column(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "kept_a.jpg").touch()
    agreement_csv = tmp_path / "no_index.csv"
    pd.DataFrame([{"id": "kept_a", "sentiment": 0}]).to_csv(agreement_csv, index=False)

    with pytest.raises(KeyError, match="image_id"):
        load_and_validate(str(agreement_csv), str(images_dir))


@pytest.mark.parametrize(
    ("val_indices", "expect_warning"),
    [([0, 4, 8, 12, 16], False), (list(range(10, 20)), True)],
)
def test_class_balance_report_warns_only_when_a_fold_share_drifts_past_five_points(
    val_indices: list[int], expect_warning: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = _fold_frame(n_per_class=4)

    check_class_balance([([], val_indices)], frame["sentiment"], 1)

    printed = capsys.readouterr().out
    assert "Class distribution per fold" in printed
    assert ("WARNING fold 1" in printed) is expect_warning


@pytest.mark.parametrize("n_folds", [3, 5])
def test_validation_folds_are_disjoint_and_together_cover_every_row(
    n_folds: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _silence_class_balance_report(monkeypatch)
    frame = _fold_frame()

    folds = create_folds(frame, n_folds=n_folds, seed=42)

    assert len(folds) == n_folds
    seen: list[str] = []
    for train_df, val_df in folds:
        train_ids = set(train_df["image_id"])
        val_ids = set(val_df["image_id"])
        assert train_ids.isdisjoint(val_ids)
        assert train_ids | val_ids == set(frame["image_id"])
        assert len(train_df) + len(val_df) == len(frame)
        seen.extend(val_df["image_id"])
    assert len(seen) == len(frame)
    assert set(seen) == set(frame["image_id"])


def test_folds_are_stratified_so_every_class_appears_in_every_validation_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence_class_balance_report(monkeypatch)
    frame = _fold_frame()

    folds = create_folds(frame, n_folds=5, seed=42)

    for _, val_df in folds:
        assert set(val_df["sentiment"]) == set(frame["sentiment"])
        assert len(val_df) == pytest.approx(len(frame) / 5, abs=1)


def test_fold_assignment_is_reproducible_for_a_fixed_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence_class_balance_report(monkeypatch)
    frame = _fold_frame()

    first = create_folds(frame, n_folds=5, seed=42)
    second = create_folds(frame, n_folds=5, seed=42)
    other_seed = create_folds(frame, n_folds=5, seed=7)

    assert [list(val["image_id"]) for _, val in first] == [
        list(val["image_id"]) for _, val in second
    ]
    assert [list(val["image_id"]) for _, val in first] != [
        list(val["image_id"]) for _, val in other_seed
    ]


def test_save_folds_writes_one_train_and_val_csv_per_one_indexed_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.data.make_folds.OUTPUT_DIR", tmp_path / "folds")
    monkeypatch.setattr("src.data.make_folds.ALL_TRAIN_CSV", tmp_path / "all_train.csv")
    _silence_class_balance_report(monkeypatch)
    frame = _fold_frame()
    folds = create_folds(frame, n_folds=5, seed=42)
    output_dir = tmp_path / "folds"

    save_folds(folds, output_dir)

    written = sorted(path.name for path in output_dir.iterdir())
    assert written == sorted(
        [f"fold_{index}_{split}.csv" for index in range(1, 6) for split in ("train", "val")]
    )
    reloaded = pd.read_csv(output_dir / "fold_1_val.csv")
    assert list(reloaded["image_id"]) == list(folds[0][1]["image_id"])
    assert list(reloaded.columns) == ["image_id", "sentiment", "image_path"]
    assert not (tmp_path / "all_train.csv").exists()


def test_mlflow_logging_failure_never_aborts_fold_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _explode(uri: str) -> None:
        calls.append(uri)
        raise RuntimeError("tracking server unreachable")

    monkeypatch.setattr("src.data.make_folds.mlflow.set_tracking_uri", _explode)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

    log_to_mlflow(_fold_frame(), 5)

    assert calls == ["http://127.0.0.1:5000"]
