"""Pin the held-out scoring that decides whether the distributional adapter beat its base."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pytest
import torch

from src.data.distributional_data import PromptEncoder, parse_record
from src.evaluation.distributional_eval import (
    BLOCK_ORDER,
    METRICS,
    cao_emd,
    cohort_of,
    kl_gains,
    log_to_mlflow,
    output_dir_for,
    render_table,
    score_records,
    summarize,
    write_report,
)
from src.training.train_distributional import ordinal_emd

RecordFactory = Callable[..., dict[str, Any]]
TokenizerFactory = Callable[..., Any]
ModelFactory = Callable[..., Any]


def test_cao_emd_is_permutation_invariant_where_the_ordinal_distance_is_not() -> None:
    predicted = np.array([0.2, 0.8])
    target = np.array([0.8, 0.2])
    valid = np.array([[True, True]])

    assert cao_emd(predicted, target) == pytest.approx(0.0)
    assert ordinal_emd(predicted[None], target[None], valid).tolist() == pytest.approx([0.6])


def test_cao_emd_reproduces_the_reference_formula_on_a_hand_computed_pair() -> None:
    assert cao_emd(np.array([0.5, 0.5]), np.array([0.0, 1.0])) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("tag", "cohort"),
    [
        ("test_1", "test_1"),
        ("test_2", "new_country"),
        ("test_3", "new_country"),
        ("test_5", "new_country"),
        ("test_6", "new_country"),
        ("test_4", "new_both"),
        ("test_7", "new_both"),
    ],
)
def test_cohort_of_maps_every_reference_test_tag(tag: str, cohort: str) -> None:
    assert cohort_of(tag) == cohort


def test_cohort_of_rejects_tags_outside_the_test_split() -> None:
    with pytest.raises(ValueError, match="unknown reference test tag 'train'"):
        cohort_of("train")


def _row(arm_kl: float, *, record_id: str = "1", cohort: str = "test_1") -> dict[str, Any]:
    return {
        "id": record_id,
        "country": "Germany",
        "data_type": "test_1",
        "cohort": cohort,
        "options": 2,
        "kl": arm_kl,
        "emd": 0.1,
        "cao_emd": 0.05,
        "entropy": 0.5,
        "option_mass": 0.9,
    }


def test_summarize_groups_by_cohort_and_evaluation_item_for_both_arms() -> None:
    rows_by_arm = {
        "base": [_row(1.0), _row(3.0, record_id="46", cohort="new_country")],
        "adapter": [_row(0.5), _row(1.0, record_id="46", cohort="new_country")],
    }

    summary = summarize(rows_by_arm)

    assert list(summary) == ["all", "test_1", "new_country", "item_46"]
    assert summary["all"]["base"]["kl"] == pytest.approx(2.0)
    assert summary["all"]["adapter"]["kl"] == pytest.approx(0.75)
    assert summary["all"]["base"]["n"] == 2
    assert summary["item_46"]["adapter"] == {
        "kl": 1.0,
        "emd": 0.1,
        "cao_emd": 0.05,
        "entropy": 0.5,
        "option_mass": 0.9,
        "n": 1.0,
    }
    assert kl_gains(summary) == pytest.approx(
        {"all": 1.25, "test_1": 0.5, "new_country": 2.0, "item_46": 2.0}
    )
    assert set(BLOCK_ORDER) >= set(summary)


def test_render_table_has_one_row_per_block_and_arm() -> None:
    summary = summarize({"base": [_row(1.0)], "adapter": [_row(0.5)]})

    table = render_table(summary)

    assert table.row_count == 4
    assert [column.header for column in table.columns] == ["block", "arm", "n", *METRICS]


def test_the_report_lands_under_the_model_directory_with_meta_summary_and_records(
    tmp_path: Path,
) -> None:
    rows_by_arm = {"base": [_row(1.0)], "adapter": [_row(0.5)]}
    summary = summarize(rows_by_arm)
    directory = output_dir_for("qwen3_vl_2b", tmp_path)

    report_path, records_path = write_report(directory, summary, rows_by_arm, {"model": "x"})
    report = json.loads(report_path.read_text())

    assert directory == tmp_path / "qwen3_vl_2b"
    assert output_dir_for("qwen3_vl_2b") == Path("outputs/evaluation/distributional/qwen3_vl_2b")
    assert report["meta"] == {"model": "x"}
    assert report["summary"]["all"]["adapter"]["kl"] == pytest.approx(0.5)
    assert report["kl_gain_base_minus_adapter"]["all"] == pytest.approx(0.5)
    lines = records_path.read_text().splitlines()
    assert lines[0] == "arm,id,country,data_type,cohort,options," + ",".join(METRICS)
    assert len(lines) == 3


def test_score_records_reports_per_record_metrics_from_one_forward_pass(
    survey_tokenizer: TokenizerFactory,
    raw_survey_record: RecordFactory,
    logits_model: ModelFactory,
) -> None:
    encoder = PromptEncoder.build(survey_tokenizer(), max_length=256)
    records = [
        parse_record(
            raw_survey_record(
                record_id="46",
                country="Mexico",
                options=("(A)Yes", "(B)No"),
                dist={"A": 25.0, "B": 75.0},
                data_type="test_5",
            )
        ),
        parse_record(
            raw_survey_record(
                record_id="240",
                country="Mexico",
                options=("(A)Left", "(B)2", "(C)Right"),
                dist={"A": 100.0, "B": 0.0, "C": 0.0},
                data_type="test_7",
            )
        ),
    ]
    encoded = [encoder.encode(record) for record in records]
    width = max(len(row["input_ids"]) for row in encoded)
    vocab = max(encoder.letter_ids.values()) + 1
    logits = torch.zeros(2, width, vocab)
    logits[1, len(encoded[1]["input_ids"]) - 1, encoder.letter_ids["A"]] = math.log(2.0)
    model = logits_model(logits)

    rows = score_records(model, encoder, records, batch_size=8, device="cpu")

    assert [row["cohort"] for row in rows] == ["new_country", "new_both"]
    assert rows[0]["kl"] == pytest.approx(0.25 * math.log(0.25 / 0.5) + 0.75 * math.log(0.75 / 0.5))
    assert rows[0]["emd"] == pytest.approx(0.25)
    assert rows[1]["kl"] == pytest.approx(math.log(1 / 0.5))
    assert rows[1]["option_mass"] == pytest.approx(4 / (vocab + 1))
    assert set(model.calls[0]) == {"input_ids", "attention_mask", "use_cache", "logits_to_keep"}
    assert model.training is False


class _FakeRun:
    def __init__(self, run_id: str, log: list[tuple[str, str, float]]) -> None:
        self.run_id = run_id
        self.log = log

    def __enter__(self) -> _FakeRun:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_log_to_mlflow_appends_the_means_to_the_training_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mlflow_run_id.txt").write_text("run42")
    opened: list[str] = []
    logged: list[tuple[str, float]] = []

    def fake_start_run(run_id: str) -> _FakeRun:
        opened.append(run_id)
        return _FakeRun(run_id, [])

    monkeypatch.setattr(mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(mlflow, "log_metric", lambda key, value: logged.append((key, value)))
    summary = summarize({"base": [_row(1.0)], "adapter": [_row(0.5)]})

    assert log_to_mlflow(str(tmp_path), summary) is True
    assert opened == ["run42"]
    assert ("heldout_adapter_kl_all", 0.5) in logged
    assert ("heldout_base_kl_test_1", 1.0) in logged
    assert all(not key.endswith("cao_emd_all") for key, _ in logged)


def test_log_to_mlflow_skips_a_checkpoint_without_a_run_id(tmp_path: Path) -> None:
    assert log_to_mlflow(str(tmp_path), {}) is False
