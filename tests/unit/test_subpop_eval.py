"""Pin the SubPOP-Eval scoring that decides whether the SubPOP adapter beat its base."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import pytest
import torch

from src.data.subpop_data import STEERING_FILENAME, SubpopEncoder, load_steering, parse_record
from src.evaluation.subpop_eval import (
    HIGH_RELEVANCE_QKEYS,
    METRICS,
    blocks_of,
    evaluate,
    gains,
    log_to_mlflow,
    output_dir_for,
    render_table,
    score_records,
    subpop_wd,
    summarize,
    write_report,
)

RecordFactory = Callable[..., dict[str, Any]]
TokenizerFactory = Callable[..., Any]
ModelFactory = Callable[..., Any]


def test_the_distance_reproduces_the_upstream_code_on_its_two_documented_cases() -> None:
    assert subpop_wd([0.2, 0.3, 0.5], [0.1, 0.5, 0.4], [1.0, 2.0, 1.5]) == pytest.approx(0.15)
    assert subpop_wd([0.3, 0.4, 0.3], [0.2, 0.5, 0.3], [1.0, 2.0, -1.0]) == pytest.approx(0.1 / 0.7)


def test_the_distance_is_zero_on_a_match_one_at_opposite_ends_and_nan_without_an_order() -> None:
    assert subpop_wd([0.2, 0.8], [0.2, 0.8], [1.0, 2.0]) == pytest.approx(0.0)
    assert subpop_wd([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert math.isnan(subpop_wd([0.2, 0.8], [0.3, 0.7], [1.0, 1.0]))
    assert math.isnan(subpop_wd([0.2, 0.8], [0.3, 0.7], [1.0, -1.0]))


def test_the_distance_orders_the_options_by_their_level_not_their_position() -> None:
    in_order = subpop_wd([0.6, 0.3, 0.1], [0.1, 0.3, 0.6], [1.0, 2.0, 3.0])
    shuffled = subpop_wd([0.1, 0.6, 0.3], [0.6, 0.1, 0.3], [3.0, 1.0, 2.0])

    assert shuffled == pytest.approx(in_order)


def test_a_mass_free_side_falls_back_to_uniform_over_the_ordinal_options() -> None:
    assert subpop_wd([0.0, 0.0, 1.0], [0.5, 0.5, 0.0], [1.0, 2.0, -1.0]) == pytest.approx(0.0)


def _row(wd: float, *, qkey: str = "natspac", attribute: str = "SEX") -> dict[str, Any]:
    return {
        "qkey": qkey,
        "attribute": attribute,
        "group": "Female",
        "options": 3,
        "wd": wd,
        "kl": wd * 2,
        "entropy": 1.0,
        "option_mass": 0.9,
    }


def test_blocks_follow_the_attribute_the_upstream_exclusions_and_the_paper_items() -> None:
    assert HIGH_RELEVANCE_QKEYS == ("marital", "satjob", "attend", "relpersn", "othlang")
    assert blocks_of(_row(0.1)) == ["all", "attribute_SEX", "upstream"]
    assert blocks_of(_row(0.1, qkey="attend")) == ["all", "attribute_SEX", "item_attend"]
    assert blocks_of(_row(0.1, qkey="happy")) == [
        "all",
        "attribute_SEX",
        "upstream",
        "item_happy",
    ]


def test_summarize_orders_the_blocks_and_ignores_records_without_a_distance() -> None:
    rows_by_arm = {
        "base": [
            _row(0.4),
            _row(float("nan")),
            _row(0.2, qkey="attend", attribute="RACE"),
            _row(0.6, qkey="trust"),
        ],
        "adapter": [
            _row(0.2),
            _row(float("nan")),
            _row(0.1, qkey="attend", attribute="RACE"),
            _row(0.3, qkey="trust"),
        ],
    }

    summary = summarize(rows_by_arm)

    assert list(summary) == [
        "all",
        "upstream",
        "attribute_RACE",
        "attribute_SEX",
        "item_trust",
        "item_attend",
    ]
    assert summary["all"]["base"]["wd"] == pytest.approx(0.4)
    assert summary["all"]["base"]["n"] == 4.0
    assert summary["upstream"]["adapter"]["wd"] == pytest.approx(0.25)
    assert summary["upstream"]["adapter"]["n"] == 3.0
    assert gains(summary, "wd")["all"] == pytest.approx(0.2)
    assert gains(summary, "kl")["item_trust"] == pytest.approx(0.6)
    assert render_table(summary).row_count == 12


def test_the_report_writes_valid_json_even_when_a_block_has_no_distance(tmp_path: Path) -> None:
    rows_by_arm = {"base": [_row(float("nan"))], "adapter": [_row(float("nan"))]}
    summary = summarize(rows_by_arm)
    directory = output_dir_for("qwen3_vl_2b", tmp_path)

    report_path, records_path = write_report(directory, summary, rows_by_arm, {"model": "x"})
    report = json.loads(report_path.read_text())

    assert directory == tmp_path / "qwen3_vl_2b"
    assert output_dir_for("qwen3_vl_2b") == Path("outputs/evaluation/subpop/qwen3_vl_2b")
    assert "NaN" not in report_path.read_text()
    assert report["meta"] == {"model": "x"}
    assert report["summary"]["all"]["adapter"]["wd"] is None
    assert report["summary"]["all"]["adapter"]["kl"] is None
    assert set(report) == {
        "meta",
        "summary",
        "wd_gain_base_minus_adapter",
        "kl_gain_base_minus_adapter",
    }
    lines = records_path.read_text().splitlines()
    assert lines[0] == "arm,qkey,attribute,group,options," + ",".join(METRICS)
    assert len(lines) == 3


def _encoder(
    tmp_path: Path, steering: list[dict[str, str]], survey_tokenizer: TokenizerFactory
) -> SubpopEncoder:
    (tmp_path / STEERING_FILENAME).write_text(json.dumps(steering), encoding="utf-8")
    return SubpopEncoder.build(survey_tokenizer(), load_steering(tmp_path), max_length=256)


def test_score_records_drops_the_refusal_letter_before_it_renormalises(
    tmp_path: Path,
    survey_tokenizer: TokenizerFactory,
    raw_subpop_record: RecordFactory,
    subpop_steering_entries: list[dict[str, str]],
    logits_model: ModelFactory,
) -> None:
    encoder = _encoder(tmp_path, subpop_steering_entries, survey_tokenizer)
    records = [
        parse_record(
            raw_subpop_record(
                qkey="trust",
                attribute="SEX",
                group="Female",
                options=("Can trust", "Cannot trust", "Refused"),
                responses=(0.25, 0.75),
                ordinal=(1.0, 2.0),
            )
        ),
        parse_record(
            raw_subpop_record(
                qkey="natspac",
                options=("Too little", "About right", "Too much", "Refused"),
                responses=(1.0, 0.0, 0.0),
                ordinal=(1.0, 2.0, 3.0),
            )
        ),
    ]
    encoded = [encoder.encode(record) for record in records]
    width = max(len(row["input_ids"]) for row in encoded)
    vocab = max(encoder.letter_ids.values()) + 1
    logits = torch.zeros(2, width, vocab)
    logits[0, len(encoded[0]["input_ids"]) - 1, encoder.letter_ids[" C"]] = 50.0
    logits[1, len(encoded[1]["input_ids"]) - 1, encoder.letter_ids[" A"]] = math.log(2.0)
    model = logits_model(logits)

    rows = score_records(model, encoder, records, batch_size=8, device="cpu")

    assert [row["qkey"] for row in rows] == ["trust", "natspac"]
    assert [row["options"] for row in rows] == [2, 3]
    assert rows[0]["kl"] == pytest.approx(0.25 * math.log(0.25 / 0.5) + 0.75 * math.log(0.75 / 0.5))
    assert rows[0]["wd"] == pytest.approx(0.25)
    assert rows[0]["option_mass"] == pytest.approx(1.0)
    assert rows[1]["kl"] == pytest.approx(math.log(1 / 0.5))
    assert rows[1]["wd"] == pytest.approx((0.5 + 0.25) / 2)
    assert rows[1]["entropy"] == pytest.approx(-(0.5 * math.log(0.5) + 2 * 0.25 * math.log(0.25)))
    assert set(model.calls[0]) == {"input_ids", "attention_mask", "use_cache", "logits_to_keep"}
    assert model.training is False


class _FakeRun:
    def __enter__(self) -> _FakeRun:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_log_to_mlflow_appends_the_finite_means_to_the_training_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mlflow_run_id.txt").write_text("run42")
    opened: list[str] = []
    logged: list[tuple[str, float]] = []

    def fake_start_run(run_id: str) -> _FakeRun:
        opened.append(run_id)
        return _FakeRun()

    monkeypatch.setattr(mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(mlflow, "log_metric", lambda key, value: logged.append((key, value)))
    summary = summarize(
        {
            "base": [_row(0.4), _row(float("nan"), attribute="RACE")],
            "adapter": [_row(0.2), _row(float("nan"), attribute="RACE")],
        }
    )

    assert log_to_mlflow(str(tmp_path), summary) is True
    assert opened == ["run42"]
    assert ("heldout_adapter_wd_all", 0.2) in logged
    assert ("heldout_base_wd_attribute_SEX", 0.4) in logged
    assert all(not key.startswith("heldout_base_wd_attribute_RACE") for key, _ in logged)
    assert log_to_mlflow(str(tmp_path / "missing"), summary) is False


def test_the_script_refuses_a_config_written_for_another_track() -> None:
    cfg = {"model": {}, "training": {}, "data": {"format": "distributional"}}

    with pytest.raises(ValueError, match=r"data\.format must be 'subpop'"):
        evaluate(cfg, "qwen3_vl_2b")
