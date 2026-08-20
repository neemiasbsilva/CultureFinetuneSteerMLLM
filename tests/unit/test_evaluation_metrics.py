"""Pin the per-fold training metric contract.

Every fold score that reaches the Holm-Bonferroni comparison is produced here, so a
silent change is a silent change to the paper.  The three failure modes this file
guards are: the -1 failed-parse sentinel being scored as a real class instead of
being dropped, a fold with no parseable prediction raising instead of degrading to a
neutral record, and the fold aggregator switching between population and sample
standard deviation or quietly folding non-float fields into the mean.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
from pandas.errors import EmptyDataError

from src.evaluation.metrics import (
    aggregate_fold_metrics,
    build_fold_metrics_csv,
    compute_classification_metrics,
)

DEGRADED_KEYS = {
    "f1_macro",
    "f1_weighted",
    "f1_per_class",
    "mae",
    "qwk",
    "accuracy",
    "n",
}


def _fold(f1_macro: float, mae: float) -> dict[str, object]:
    return {
        "f1_macro": f1_macro,
        "f1_weighted": f1_macro,
        "f1_per_class": [f1_macro, f1_macro],
        "mae": mae,
        "qwk": 0.5,
        "accuracy": f1_macro,
        "n": 10,
    }


def test_failed_parse_sentinels_are_dropped_before_any_score_is_computed() -> None:
    with_sentinels = compute_classification_metrics(
        [0, 1, 2, 3, 4, -1, 2],
        [0, 1, 2, 2, 4, 3, -1],
    )
    without_sentinels = compute_classification_metrics([0, 1, 2, 3, 4], [0, 1, 2, 2, 4])
    assert with_sentinels == without_sentinels
    assert with_sentinels["n"] == 5


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([-1, 1, 2], [0, 1, 2]),
        ([0, 1, 2], [-1, 1, 2]),
        ([-1, 1, 2], [-1, 1, 2]),
    ],
)
def test_a_pair_is_dropped_when_either_side_carries_the_sentinel(
    y_true: list[int], y_pred: list[int]
) -> None:
    assert compute_classification_metrics(y_true, y_pred)["n"] == 2


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        ([], []),
        ([-1, -1], [3, 4]),
        ([3, 4], [-1, -1]),
        ([-1, -1, -1], [-1, -1, -1]),
    ],
)
def test_a_fold_with_no_valid_pair_degrades_instead_of_raising(
    y_true: list[int], y_pred: list[int]
) -> None:
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics == {
        "f1_macro": 0.0,
        "f1_weighted": 0.0,
        "f1_per_class": [],
        "mae": 5.0,
        "qwk": 0.0,
        "accuracy": 0.0,
        "n": 0,
    }


def test_degraded_and_scored_folds_expose_an_identical_key_set() -> None:
    scored = compute_classification_metrics([0, 1, 2, 3, 4], [0, 1, 2, 2, 4])
    degraded = compute_classification_metrics([-1], [-1])
    assert set(scored) == set(degraded) == DEGRADED_KEYS


def test_a_perfect_prediction_scores_one_and_zero_error() -> None:
    metrics = compute_classification_metrics([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert metrics["f1_weighted"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["qwk"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["f1_per_class"] == [1.0] * 5
    assert metrics["n"] == 5


def test_the_degraded_mae_sentinel_exceeds_any_error_reachable_on_a_five_point_scale() -> None:
    worst_real = compute_classification_metrics([0, 0, 0], [4, 4, 4])
    assert worst_real["mae"] == pytest.approx(4.0)
    assert compute_classification_metrics([-1], [-1])["mae"] > worst_real["mae"]


def test_every_scalar_score_is_rounded_to_six_decimals() -> None:
    metrics = compute_classification_metrics([0, 1, 2, 3, 4], [0, 1, 2, 2, 4])
    scalars = {k: v for k, v in metrics.items() if isinstance(v, float)}
    assert scalars == {k: round(v, 6) for k, v in scalars.items()}
    assert metrics["f1_macro"] == pytest.approx(0.733333)
    assert metrics["mae"] == pytest.approx(0.2)
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert all(v == round(v, 6) for v in metrics["f1_per_class"])
    assert metrics["f1_per_class"][2] == pytest.approx(0.666667)


def test_per_class_f1_has_one_entry_for_every_label_observed_after_filtering() -> None:
    metrics = compute_classification_metrics([0, 1, 1, -1, 4], [0, 1, 1, 2, 4])
    assert len(metrics["f1_per_class"]) == 3
    assert metrics["n"] == 4
    assert metrics["f1_per_class"] == [1.0, 1.0, 1.0]


def test_macro_f1_never_exceeds_weighted_f1_on_an_imbalanced_fold() -> None:
    metrics = compute_classification_metrics([0, 0, 0, 0, 1], [0, 0, 0, 0, 0])
    assert metrics["f1_macro"] <= metrics["f1_weighted"] + 1e-12


def test_aggregation_of_no_folds_returns_an_empty_mapping() -> None:
    assert aggregate_fold_metrics([]) == {}


def test_aggregation_emits_mean_and_std_only_for_float_valued_keys() -> None:
    aggregated = aggregate_fold_metrics([_fold(0.6, 1.0), _fold(0.8, 2.0)])
    assert set(aggregated) == {
        "f1_macro_mean",
        "f1_macro_std",
        "f1_weighted_mean",
        "f1_weighted_std",
        "mae_mean",
        "mae_std",
        "qwk_mean",
        "qwk_std",
        "accuracy_mean",
        "accuracy_std",
    }
    assert not any(key.startswith("f1_per_class") for key in aggregated)
    assert not any(key.startswith("n_") for key in aggregated)


def test_aggregation_uses_the_sample_standard_deviation() -> None:
    aggregated = aggregate_fold_metrics([_fold(0.6, 1.0), _fold(0.8, 2.0)])
    assert aggregated["f1_macro_mean"] == pytest.approx(0.7)
    assert aggregated["f1_macro_std"] == pytest.approx(math.sqrt(0.02), abs=1e-6)
    assert aggregated["f1_macro_std"] != pytest.approx(0.1, abs=1e-6)
    assert aggregated["mae_mean"] == pytest.approx(1.5)
    assert aggregated["mae_std"] == pytest.approx(math.sqrt(0.5), abs=1e-6)


def test_aggregated_moments_are_rounded_to_six_decimals() -> None:
    aggregated = aggregate_fold_metrics([_fold(1 / 3, 1.0), _fold(2 / 3, 2.0), _fold(0.1, 3.0)])
    assert aggregated == {k: round(v, 6) for k, v in aggregated.items()}


def test_a_constant_metric_across_folds_aggregates_to_zero_spread() -> None:
    aggregated = aggregate_fold_metrics([_fold(0.5, 1.0), _fold(0.5, 1.0), _fold(0.5, 1.0)])
    assert aggregated["f1_macro_mean"] == pytest.approx(0.5)
    assert aggregated["f1_macro_std"] == pytest.approx(0.0)
    assert aggregated["qwk_std"] == pytest.approx(0.0)


@pytest.mark.filterwarnings("ignore")
def test_aggregating_a_single_fold_currently_yields_nan_standard_deviations() -> None:
    """Characterization: ddof=1 on one observation is NaN. Pins today's behaviour."""
    aggregated = aggregate_fold_metrics([_fold(0.6, 1.0)])
    assert aggregated["f1_macro_mean"] == pytest.approx(0.6)
    assert math.isnan(aggregated["f1_macro_std"])


@pytest.mark.filterwarnings("ignore")
def test_a_unanimous_fold_currently_scores_qwk_as_nan_and_poisons_the_mean() -> None:
    """Characterization: np.mean propagates the NaN. Pins today's behaviour."""
    unanimous = compute_classification_metrics([3, 3, 3], [3, 3, 3])
    assert unanimous["accuracy"] == pytest.approx(1.0)
    assert math.isnan(unanimous["qwk"])
    mixed = compute_classification_metrics([0, 1, 2, 3], [0, 1, 2, 3])
    assert math.isnan(aggregate_fold_metrics([unanimous, mixed])["qwk_mean"])


def test_fold_metrics_csv_round_trips_through_pandas(tmp_path: Path) -> None:
    records = [
        {
            "model_name": "qwen3_vl_8b",
            "culture": "arabic",
            "condition": "wvs_cultural",
            "fold": fold,
            "f1_macro": 0.5 + fold / 100,
            "f1_weighted": 0.6 + fold / 100,
            "mae": 1.0 - fold / 100,
            "qwk": 0.4 + fold / 100,
            "accuracy": 0.7 + fold / 100,
        }
        for fold in (1, 2, 3)
    ]
    output_path = tmp_path / "fold_metrics.csv"
    frame = build_fold_metrics_csv(records, str(output_path))
    assert output_path.exists()
    reloaded = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(frame, reloaded)
    assert list(reloaded.columns) == list(records[0])
    assert len(reloaded) == 3


def test_fold_metrics_csv_writes_without_a_pandas_index_column(tmp_path: Path) -> None:
    output_path = tmp_path / "fold_metrics.csv"
    build_fold_metrics_csv([{"fold": 1, "f1_macro": 0.5}], str(output_path))
    header = output_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "fold,f1_macro"


def test_an_empty_result_list_writes_a_headerless_file_pandas_cannot_reread(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "fold_metrics.csv"
    frame = build_fold_metrics_csv([], str(output_path))
    assert frame.empty
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").strip() == ""
    with pytest.raises(EmptyDataError):
        pd.read_csv(output_path)
