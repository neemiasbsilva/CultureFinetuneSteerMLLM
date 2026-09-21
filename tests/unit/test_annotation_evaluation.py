"""Pin the Stage-4 evaluation contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.convergence import compute_condition_pair_agreement
from src.analysis.similarity import compute_culture_sim_matrix
from src.evaluation.annotation_eval import (
    METRIC_NAMES,
    _corrected_resampled_t,
    _holm_adjust,
    _metrics,
    aggregate_repetitions,
    assert_matched_generation_seeds,
    attempted_panel_violations,
    build_common_condition_panel,
    build_holm_comparisons,
    holm_bonferroni,
    incomplete_pass_groups,
    mode_with_median_tiebreak,
)


def test_mode_tie_breaks_by_median_then_lower() -> None:
    assert mode_with_median_tiebreak([0, 0, 4, 4]) == 0
    assert mode_with_median_tiebreak([0, 0, 3, 3, 4]) == 3
    assert mode_with_median_tiebreak([1, 2, 2, 3, 3]) == 2


def test_repetition_aggregation_ignores_failed_parses_and_records_coverage() -> None:
    attempts = pd.DataFrame(
        [
            {
                "model_name": "m",
                "culture": "arabic",
                "condition": "cultural",
                "image_id": "1",
                "run_index": 1,
                "predicted_sentiment": 4,
                "parse_valid": True,
                "caption": "four",
                "predicted_perceptions": ["a"],
            },
            {
                "model_name": "m",
                "culture": "arabic",
                "condition": "wvs_cultural",
                "image_id": "1",
                "run_index": 2,
                "predicted_sentiment": 2,
                "parse_valid": True,
                "caption": "two",
                "predicted_perceptions": ["b"],
            },
            {
                "model_name": "m",
                "culture": "arabic",
                "condition": "wvs_cultural",
                "image_id": "1",
                "run_index": 3,
                "predicted_sentiment": -1,
                "parse_valid": False,
                "caption": "bad",
                "predicted_perceptions": [],
            },
        ]
    )
    result = aggregate_repetitions(attempts, {"1": 3})
    assert len(result) == 1
    row = result.iloc[0]
    assert row["condition"] == "wvs_cultural"
    assert row["predicted_sentiment"] == 2
    assert row["n_attempts"] == 3
    assert row["n_valid_runs"] == 2
    assert row["parse_coverage"] == 2 / 3


def test_five_pass_gate_and_matched_seed_validation() -> None:
    rows = []
    for condition in ("wvs_cultural", "inference_only"):
        for run_index in range(1, 6):
            rows.append(
                {
                    "model_name": "m",
                    "culture": "english",
                    "condition": condition,
                    "image_id": "1",
                    "run_index": run_index,
                    "seed": 100 + run_index,
                }
            )
    frame = pd.DataFrame(rows)
    assert incomplete_pass_groups(frame, 5).empty
    assert_matched_generation_seeds(frame)

    incomplete = incomplete_pass_groups(frame[frame["run_index"] != 5], 5)
    assert len(incomplete) == 2
    broken = frame.copy()
    broken.loc[broken.index[-1], "seed"] = 999
    with pytest.raises(ValueError, match="unmatched generation seeds"):
        assert_matched_generation_seeds(broken)


def test_attempted_panel_requires_identical_condition_images() -> None:
    rows = []
    for condition, culture, images in [
        ("inference_only", "inference_only", ["1", "2"]),
        ("wvs_cultural", "english", ["1"]),
    ]:
        for image_id in images:
            rows.append(
                {
                    "model_name": "m",
                    "culture": culture,
                    "condition": condition,
                    "image_id": image_id,
                }
            )
    violations = attempted_panel_violations(
        pd.DataFrame(rows),
        ["inference_only", "wvs_cultural"],
        expected_cultures=["english"],
    )
    assert not violations.empty
    assert "wvs_cultural" in set(violations["condition"])


def test_common_panel_reuses_base_and_intersects_valid_images() -> None:
    rows = []
    for condition, culture, images in [
        ("wvs_cultural", "arabic", ["1", "2"]),
        ("inference_only", "inference_only", ["1"]),
    ]:
        for image in images:
            rows.append(
                {
                    "model_name": "m",
                    "culture": culture,
                    "condition": condition,
                    "image_id": image,
                    "predicted_sentiment": 2,
                    "ground_truth_sentiment": 2,
                    "n_attempts": 5,
                    "n_valid_runs": 5,
                    "caption": "scene",
                    "predicted_perceptions": [],
                }
            )
    panel = build_common_condition_panel(
        pd.DataFrame(rows),
        ["inference_only", "wvs_cultural"],
    )
    assert set(panel["image_id"]) == {"1"}
    assert set(panel["condition"]) == {"inference_only", "wvs_cultural"}
    assert set(panel["culture"]) == {"arabic"}


def test_metrics_include_primary_and_all_secondary_metrics() -> None:
    metrics = _metrics(np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 2, 4]))
    assert set(metrics) == set(METRIC_NAMES)
    assert metrics["f1_macro"] <= metrics["f1_weighted"] + 1e-12
    assert metrics["accuracy"] == 0.8
    assert metrics["mae"] == 0.2
    assert metrics["kappa_quadratic"] >= metrics["kappa_linear"]


def test_holm_adjustment_is_monotone_in_raw_p_order() -> None:
    raw = np.array([0.01, 0.011, 0.5, 0.04])
    adjusted = _holm_adjust(raw)
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert adjusted[order][0] == 0.04
    assert adjusted[order][1] == 0.04


def test_corrected_resampled_fold_test_uses_overlap_variance() -> None:
    a = [0.90, 0.83, 0.88, 0.79, 0.86]
    b = [0.74, 0.76, 0.78, 0.72, 0.77]
    statistic, p_value = _corrected_resampled_t(a, b)
    differences = np.asarray(a) - np.asarray(b)
    expected_se = np.sqrt((1 / 5 + 1 / 4) * np.var(differences, ddof=1))
    assert np.isclose(statistic, differences.mean() / expected_se)
    assert 0 <= p_value <= 1


def test_condition_agreement_requires_semantic_caption_embeddings() -> None:
    rows = pd.DataFrame(
        [
            {
                "model_name": "m",
                "culture": "english",
                "condition": "wvs_cultural",
                "image_id": "1",
                "predicted_sentiment": 3,
                "predicted_perceptions": ["street"],
                "caption": "A street",
            },
            {
                "model_name": "m",
                "culture": "english",
                "condition": "inference_only",
                "image_id": "1",
                "predicted_sentiment": 3,
                "predicted_perceptions": ["street"],
                "caption": "An avenue",
            },
        ]
    )
    with pytest.raises(ValueError, match="caption_embeddings are required"):
        compute_condition_pair_agreement(rows)
    result = compute_condition_pair_agreement(
        rows,
        caption_embeddings=np.asarray([[1.0, 0.0], [0.8, 0.6]]),
        condition_pairs={"wvs_vs_base": ("wvs_cultural", "inference_only")},
        n_bootstrap=20,
        seed=7,
    )
    assert np.isclose(result.iloc[0]["caption_cosine"], 0.8)
    assert result.iloc[0]["tag_jaccard"] == 1.0
    assert result.iloc[0]["sentiment_agreement"] == 1.0
    assert result.iloc[0]["caption_cosine_lo"] == result.iloc[0]["caption_cosine_hi"]
    assert result.iloc[0]["tag_jaccard_lo"] == 1.0


def test_legacy_similarity_matrix_refuses_to_mix_conditions() -> None:
    frame = pd.DataFrame(
        {
            "culture": ["english", "english"],
            "condition": ["wvs_cultural", "inference_only"],
        }
    )
    with pytest.raises(ValueError, match="multiple annotation conditions"):
        compute_culture_sim_matrix(frame, np.eye(2))


def test_fold_comparisons_and_holm_are_separated_by_planned_family() -> None:
    rows = []
    values = {
        "wvs_cultural": [0.9, 0.8, 0.9],
        "inference_only": [0.5, 0.4, 0.5],
    }
    for condition, scores in values.items():
        for fold, score in enumerate(scores, start=1):
            rows.append(
                {
                    "model_name": "m",
                    "culture": "arabic",
                    "condition": condition,
                    "fold": fold,
                    "f1_macro": score,
                }
            )
    comparisons = build_holm_comparisons(pd.DataFrame(rows))
    assert {item["comparison_family"] for item in comparisons} == {"wvs_vs_base"}
    result = holm_bonferroni(comparisons)
    assert set(result["comparison_family"]) == {"wvs_vs_base"}
    assert np.allclose(result["p_raw"], result["p_adjusted"])
