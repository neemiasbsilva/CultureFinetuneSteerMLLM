"""Pin the contract between the annotation directory layout and the analysis frame."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.data_loading import load_annotations, load_failures, parse_failure_rate


def _record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "image_id": "img",
        "predicted_sentiment": 3,
        "ground_truth_sentiment": 2,
        "caption": "a busy street market",
        "justification": "people are shopping together",
        "predicted_perceptions": ["market", "crowd"],
    }
    record.update(overrides)
    return record


def _write_annotations(
    root: Path,
    parts: tuple[str, ...],
    records: list[dict[str, object]],
    filename: str = "annotations.jsonl",
) -> Path:
    directory = root.joinpath(*parts)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def _build_tree(root: Path) -> Path:
    _write_annotations(root, ("qwen3_vl_8b", "english", "wvs_cultural"), [_record(image_id="1")])
    _write_annotations(root, ("qwen3_vl_8b", "english", "inference_only"), [_record(image_id="2")])
    _write_annotations(root, ("qwen3_vl_8b", "arabic", "wvs_cultural"), [_record(image_id="3")])
    _write_annotations(root, ("gemma4_e2b", "english", "wvs_cultural"), [_record(image_id="4")])
    return root


def test_directory_layout_supplies_model_culture_and_condition_for_bare_records(
    tmp_path: Path,
) -> None:
    _write_annotations(
        tmp_path,
        ("qwen3_vl_8b", "english", "inference_only"),
        [_record(image_id="1"), _record(image_id="2")],
    )
    frame = load_annotations(annotations_dir=tmp_path)
    assert len(frame) == 2
    assert set(frame["model_name"]) == {"qwen3_vl_8b"}
    assert set(frame["culture"]) == {"english"}
    assert set(frame["condition"]) == {"inference_only"}


def test_a_layout_without_a_condition_directory_leaves_the_condition_empty(
    tmp_path: Path,
) -> None:
    _write_annotations(tmp_path, ("qwen3_vl_8b", "english"), [_record()])
    row = load_annotations(annotations_dir=tmp_path).iloc[0]
    assert row["model_name"] == "qwen3_vl_8b"
    assert row["culture"] == "english"
    assert row["condition"] == ""
    assert row["profile"] == "english__qwen3_vl_8b__"


def test_a_file_shallower_than_the_expected_layout_infers_no_model_or_culture(
    tmp_path: Path,
) -> None:
    _write_annotations(tmp_path, ("english",), [_record()])
    row = load_annotations(annotations_dir=tmp_path).iloc[0]
    assert row["model_name"] == ""
    assert row["culture"] == ""
    assert row["profile"] == "____"


def test_record_fields_take_precedence_over_the_directory_layout(tmp_path: Path) -> None:
    _write_annotations(
        tmp_path,
        ("qwen3_vl_8b", "english", "inference_only"),
        [_record(model_name="gemma4_e2b", culture="arabic", condition="wvs_cultural")],
    )
    row = load_annotations(annotations_dir=tmp_path).iloc[0]
    assert row["model_name"] == "gemma4_e2b"
    assert row["culture"] == "arabic"
    assert row["condition"] == "wvs_cultural"


@pytest.mark.parametrize(
    ("directory_condition", "record_condition"),
    [
        ("cultural", None),
        ("wvs_cultural", "cultural"),
        ("cultural", "  CULTURAL  "),
        ("wvs_cultural", None),
    ],
)
def test_legacy_cultural_naming_is_normalized_to_wvs_cultural(
    tmp_path: Path,
    directory_condition: str,
    record_condition: str | None,
) -> None:
    overrides: dict[str, object] = {}
    if record_condition is not None:
        overrides["condition"] = record_condition
    _write_annotations(
        tmp_path,
        ("qwen3_vl_8b", "english", directory_condition),
        [_record(**overrides)],
    )
    frame = load_annotations(annotations_dir=tmp_path)
    assert set(frame["condition"]) == {"wvs_cultural"}
    assert set(frame["profile"]) == {"english__qwen3_vl_8b__wvs_cultural"}


def test_profile_carries_the_condition_so_conditions_never_share_a_group_key(
    tmp_path: Path,
) -> None:
    _write_annotations(tmp_path, ("qwen3_vl_8b", "english", "cultural"), [_record(image_id="1")])
    _write_annotations(
        tmp_path,
        ("qwen3_vl_8b", "english", "inference_only"),
        [_record(image_id="1")],
    )
    frame = load_annotations(annotations_dir=tmp_path)
    assert set(frame["legacy_profile"]) == {"english__qwen3_vl_8b"}
    assert set(frame["profile"]) == {
        "english__qwen3_vl_8b__wvs_cultural",
        "english__qwen3_vl_8b__inference_only",
    }
    assert frame["profile"].nunique() == 2
    assert set(frame["profile_run"]) == {
        "english__qwen3_vl_8b__wvs_cultural__r01",
        "english__qwen3_vl_8b__inference_only__r01",
    }


def test_annotation_run_id_omits_the_condition_so_only_profile_separates_the_arms(
    tmp_path: Path,
) -> None:
    _write_annotations(tmp_path, ("qwen3_vl_8b", "english", "cultural"), [_record(image_id="1")])
    _write_annotations(
        tmp_path,
        ("qwen3_vl_8b", "english", "inference_only"),
        [_record(image_id="1")],
    )
    frame = load_annotations(annotations_dir=tmp_path)
    assert set(frame["annotation_run_id"]) == {"qwen3_vl_8b_english_r01"}
    assert frame["profile"].nunique() == 2


def test_missing_run_index_defaults_to_the_first_run_with_composed_identifiers(
    tmp_path: Path,
) -> None:
    _write_annotations(tmp_path, ("qwen3_vl_8b", "english", "wvs_cultural"), [_record()])
    row = load_annotations(annotations_dir=tmp_path).iloc[0]
    assert row["run_index"] == 1
    assert row["n_runs"] == 1
    assert row["run_id"] == "r01"
    assert row["annotation_run_id"] == "qwen3_vl_8b_english_r01"


@pytest.mark.parametrize(("run_index", "expected_run_id"), [(1, "r01"), (5, "r05"), (12, "r12")])
def test_explicit_run_index_is_zero_padded_into_the_run_id(
    tmp_path: Path,
    run_index: int,
    expected_run_id: str,
) -> None:
    _write_annotations(
        tmp_path,
        ("qwen3_vl_8b", "english", "wvs_cultural"),
        [_record(run_index=run_index)],
    )
    row = load_annotations(annotations_dir=tmp_path).iloc[0]
    assert row["run_id"] == expected_run_id
    assert row["profile_run"] == f"english__qwen3_vl_8b__wvs_cultural__{expected_run_id}"


def test_explicit_run_identifiers_are_never_overwritten_by_the_composed_defaults(
    tmp_path: Path,
) -> None:
    _write_annotations(
        tmp_path,
        ("qwen3_vl_8b", "english", "wvs_cultural"),
        [_record(run_index=3, n_runs=5, run_id="seed-42", annotation_run_id="paper-run-7")],
    )
    row = load_annotations(annotations_dir=tmp_path).iloc[0]
    assert row["run_id"] == "seed-42"
    assert row["n_runs"] == 5
    assert row["annotation_run_id"] == "paper-run-7"


@pytest.mark.parametrize("bad_sentiment", ["not_a_number", "", None, "positive"])
def test_non_numeric_sentiments_are_coerced_to_the_missing_sentinel(
    tmp_path: Path,
    bad_sentiment: object,
) -> None:
    _write_annotations(
        tmp_path,
        ("qwen3_vl_8b", "english", "wvs_cultural"),
        [
            _record(image_id="1", predicted_sentiment=bad_sentiment),
            _record(image_id="2", predicted_sentiment=4),
        ],
    )
    frame = load_annotations(annotations_dir=tmp_path).set_index("image_id")
    assert frame.loc["1", "predicted_sentiment"] == -1
    assert frame.loc["1", "predicted_sentiment_label"] == "unknown"
    assert frame.loc["2", "predicted_sentiment"] == 4
    assert frame.loc["2", "predicted_sentiment_label"] == "positive"


def test_sentiment_fields_absent_from_a_record_become_the_missing_sentinel(
    tmp_path: Path,
) -> None:
    complete = _record(image_id="1")
    partial = _record(image_id="2")
    del partial["ground_truth_sentiment"]
    _write_annotations(tmp_path, ("qwen3_vl_8b", "english", "wvs_cultural"), [complete, partial])
    frame = load_annotations(annotations_dir=tmp_path).set_index("image_id")
    assert frame.loc["2", "ground_truth_sentiment"] == -1
    assert frame.loc["2", "ground_truth_label"] == "unknown"
    assert frame.loc["1", "ground_truth_sentiment"] == 2
    assert frame.loc["1", "ground_truth_label"] == "neutral"


def test_sentiment_coercion_requires_at_least_one_record_to_carry_the_field(
    tmp_path: Path,
) -> None:
    bare = _record()
    del bare["predicted_sentiment"]
    del bare["ground_truth_sentiment"]
    _write_annotations(tmp_path, ("qwen3_vl_8b", "english", "wvs_cultural"), [bare])
    with pytest.raises(KeyError, match="predicted_sentiment"):
        load_annotations(annotations_dir=tmp_path)


@pytest.mark.parametrize("create_root", [True, False])
def test_an_empty_or_missing_root_returns_an_empty_frame_instead_of_raising(
    tmp_path: Path,
    create_root: bool,
) -> None:
    root = tmp_path / "annotations"
    if create_root:
        root.mkdir()
    frame = load_annotations(annotations_dir=root)
    failures = load_failures(annotations_dir=root)
    assert isinstance(frame, pd.DataFrame)
    assert frame.empty
    assert list(frame.columns) == []
    assert failures.empty


def test_a_root_holding_only_blank_lines_returns_an_empty_frame(tmp_path: Path) -> None:
    directory = tmp_path / "qwen3_vl_8b" / "english" / "wvs_cultural"
    directory.mkdir(parents=True)
    (directory / "annotations.jsonl").write_text("\n   \n\n")
    assert load_annotations(annotations_dir=tmp_path).empty


@pytest.mark.parametrize(
    ("filters", "expected_images"),
    [
        ({}, {"1", "2", "3", "4"}),
        ({"model_names": ["qwen3_vl_8b"]}, {"1", "2", "3"}),
        ({"cultures": ["english"]}, {"1", "2", "4"}),
        ({"conditions": ["wvs_cultural"]}, {"1", "3", "4"}),
        ({"conditions": ["cultural"]}, {"1", "3", "4"}),
        ({"conditions": ["inference_only"]}, {"2"}),
        ({"model_names": ["qwen3_vl_8b"], "conditions": ["wvs_cultural"]}, {"1", "3"}),
        ({"model_names": ["gemma4_e2b"], "cultures": ["arabic"]}, set()),
        ({"model_names": []}, set()),
    ],
)
def test_filter_arguments_restrict_the_loaded_rows(
    tmp_path: Path,
    filters: dict[str, list[str]],
    expected_images: set[str],
) -> None:
    frame = load_annotations(annotations_dir=_build_tree(tmp_path), **filters)
    assert set(frame["image_id"]) == expected_images
    assert len(frame) == len(expected_images)


def test_a_frame_emptied_by_filters_keeps_only_the_raw_record_columns(tmp_path: Path) -> None:
    frame = load_annotations(annotations_dir=_build_tree(tmp_path), cultures=["korean"])
    assert frame.empty
    assert "image_id" in frame.columns
    assert "profile" not in frame.columns
    assert "run_id" not in frame.columns


def test_sharded_annotation_files_are_loaded_alongside_the_canonical_file(
    tmp_path: Path,
) -> None:
    parts = ("qwen3_vl_8b", "english", "wvs_cultural")
    _write_annotations(tmp_path, parts, [_record(image_id="1")])
    _write_annotations(tmp_path, parts, [_record(image_id="2")], filename="annotations_002.jsonl")
    frame = load_annotations(annotations_dir=tmp_path)
    assert set(frame["image_id"]) == {"1", "2"}
    assert len(frame) == 2


def test_text_features_count_words_and_perception_tags(tmp_path: Path) -> None:
    sparse = _record(image_id="2", caption=None, justification=None)
    del sparse["predicted_perceptions"]
    _write_annotations(
        tmp_path,
        ("qwen3_vl_8b", "english", "wvs_cultural"),
        [_record(image_id="1"), sparse],
    )
    frame = load_annotations(annotations_dir=tmp_path).set_index("image_id")
    assert frame.loc["1", "caption_len"] == 4
    assert frame.loc["1", "justification_len"] == 4
    assert frame.loc["1", "n_perceptions"] == 2
    assert frame.loc["2", "caption_len"] == 0
    assert frame.loc["2", "justification_len"] == 0
    assert frame.loc["2", "n_perceptions"] == 0


def test_failure_records_are_normalized_and_filtered_like_annotations(tmp_path: Path) -> None:
    directory = tmp_path / "qwen3_vl_8b" / "english" / "cultural"
    directory.mkdir(parents=True)
    rows = [
        {
            "model_name": "qwen3_vl_8b",
            "culture": "english",
            "condition": "cultural",
            "image_id": "1",
        },
        {
            "model_name": "gemma4_e2b",
            "culture": "english",
            "condition": "inference_only",
            "image_id": "2",
        },
    ]
    with (directory / "failures.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    failures = load_failures(annotations_dir=tmp_path)
    assert set(failures["condition"]) == {"wvs_cultural", "inference_only"}
    restricted = load_failures(annotations_dir=tmp_path, conditions=["cultural"])
    assert set(restricted["image_id"]) == {"1"}


@pytest.mark.parametrize(
    ("n_success", "n_failure", "expected"),
    [(0, 0, 0.0), (10, 0, 0.0), (9, 1, 0.1), (0, 4, 1.0), (1, 3, 0.75)],
)
def test_parse_failure_rate_is_failures_over_all_attempts_and_guards_empty_input(
    n_success: int,
    n_failure: int,
    expected: float,
) -> None:
    successes = pd.DataFrame({"image_id": [str(i) for i in range(n_success)]})
    failures = pd.DataFrame({"image_id": [str(i) for i in range(n_failure)]})
    assert parse_failure_rate(successes, failures) == pytest.approx(expected)


def test_parse_failure_rate_accepts_the_columnless_empty_frames_the_loaders_return() -> None:
    assert parse_failure_rate(pd.DataFrame(), pd.DataFrame()) == pytest.approx(0.0)
