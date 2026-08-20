"""Pin the Holm-Bonferroni comparison family that decides which results get a star.

This module is the only thing standing between a fold-metrics CSV and a significance
claim in the paper, and every way it can break is silent.  A pair that differs on two
dimensions slipping into the family inflates the correction factor for everyone else.
A lost cumulative maximum lets a weaker comparison receive a smaller adjusted p-value
than a stronger one.  A change in how unequal fold lists are truncated re-pairs the
observations underneath a paired test.  A table that stops filtering or stops bolding
publishes the wrong rows.  None of those raise; they just change the numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest
from scipy.stats import ttest_rel

from src.evaluation.statistical_tests import (
    apply_holm_bonferroni,
    generate_latex_table,
    load_fold_metrics,
    run_all_comparisons,
)


def _write_fold_metrics(tmp_path: Path, rows: list[dict[str, object]]) -> str:
    path = tmp_path / "fold_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _three_dimension_groups() -> dict[tuple, list]:
    return {
        ("qwen", "arabic", "cultural"): [0.72, 0.74, 0.71, 0.73, 0.75],
        ("qwen", "arabic", "baseline"): [0.61, 0.60, 0.58, 0.63, 0.59],
        ("phi", "arabic", "cultural"): [0.66, 0.69, 0.64, 0.70, 0.67],
        ("qwen", "german", "cultural"): [0.55, 0.58, 0.54, 0.57, 0.56],
        ("phi", "german", "baseline"): [0.40, 0.44, 0.41, 0.43, 0.42],
    }


def _p_value_family(raw_p_values: Sequence[float]) -> list[dict]:
    return [{"label": f"c{index}", "p_raw": value} for index, value in enumerate(raw_p_values)]


def _table_frame(significance: Sequence[bool]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_a": f"m{index}",
                "culture_a": "arabic",
                "condition_a": "cultural",
                "model_b": f"m{index}",
                "culture_b": "arabic",
                "condition_b": "baseline",
                "n_folds": 5,
                "t_stat": 3.5 + index,
                "p_raw": 0.001 * (index + 1),
                "p_adj": 0.002 * (index + 1),
                "significant": flag,
            }
            for index, flag in enumerate(significance)
        ]
    )


def _body_rows(table: str) -> list[str]:
    lines = table.splitlines()
    return lines[lines.index(r"\midrule") + 1 : lines.index(r"\bottomrule")]


@pytest.mark.parametrize("metric", ["f1_macro", "mae"])
def test_load_fold_metrics_groups_fold_rows_by_model_culture_and_condition(
    tmp_path: Path, metric: str
) -> None:
    path = _write_fold_metrics(
        tmp_path,
        [
            {
                "model_name": "qwen",
                "culture": "arabic",
                "condition": "cultural",
                "fold": 1,
                "f1_macro": 0.71,
                "mae": 0.40,
            },
            {
                "model_name": "qwen",
                "culture": "arabic",
                "condition": "cultural",
                "fold": 2,
                "f1_macro": 0.75,
                "mae": 0.38,
            },
            {
                "model_name": "qwen",
                "culture": "arabic",
                "condition": "baseline",
                "fold": 1,
                "f1_macro": 0.60,
                "mae": 0.52,
            },
        ],
    )
    expected = {
        "f1_macro": {
            ("qwen", "arabic", "cultural"): [0.71, 0.75],
            ("qwen", "arabic", "baseline"): [0.60],
        },
        "mae": {
            ("qwen", "arabic", "cultural"): [0.40, 0.38],
            ("qwen", "arabic", "baseline"): [0.52],
        },
    }
    assert load_fold_metrics(path, metric) == expected[metric]


def test_load_fold_metrics_fails_loudly_when_the_requested_metric_column_is_absent(
    tmp_path: Path,
) -> None:
    path = _write_fold_metrics(
        tmp_path,
        [
            {
                "model_name": "qwen",
                "culture": "arabic",
                "condition": "cultural",
                "fold": 1,
                "f1_macro": 0.71,
            },
        ],
    )
    with pytest.raises(KeyError, match="qwk"):
        load_fold_metrics(path, "qwk")


def test_run_all_comparisons_compares_only_pairs_differing_on_exactly_one_dimension() -> None:
    comparisons = run_all_comparisons(_three_dimension_groups())
    pairs = {
        (
            c["model_a"],
            c["culture_a"],
            c["condition_a"],
            c["model_b"],
            c["culture_b"],
            c["condition_b"],
        )
        for c in comparisons
    }
    assert pairs == {
        ("qwen", "arabic", "cultural", "qwen", "arabic", "baseline"),
        ("qwen", "arabic", "cultural", "phi", "arabic", "cultural"),
        ("qwen", "arabic", "cultural", "qwen", "german", "cultural"),
    }


def test_run_all_comparisons_labels_each_row_by_the_dimension_that_differs() -> None:
    comparisons = run_all_comparisons(_three_dimension_groups())
    labels = {
        (
            c["model_a"],
            c["culture_a"],
            c["condition_a"],
            c["model_b"],
            c["culture_b"],
            c["condition_b"],
        ): c["comparison_type"]
        for c in comparisons
    }
    assert labels == {
        ("qwen", "arabic", "cultural", "qwen", "arabic", "baseline"): "cultural_vs_baseline",
        ("qwen", "arabic", "cultural", "phi", "arabic", "cultural"): "cross_architecture",
        ("qwen", "arabic", "cultural", "qwen", "german", "cultural"): "cross_culture",
    }


def test_run_all_comparisons_returns_an_empty_family_for_an_empty_group_map() -> None:
    assert run_all_comparisons({}) == []


def test_run_all_comparisons_leaves_the_multiplicity_correction_to_the_holm_step() -> None:
    comparisons = run_all_comparisons(_three_dimension_groups(), alpha=0.5)
    assert comparisons
    assert all("p_adj" not in c and "significant" not in c for c in comparisons)


def test_run_all_comparisons_reports_the_paired_t_statistic_of_the_two_fold_vectors() -> None:
    groups = {
        ("qwen", "arabic", "cultural"): [0.72, 0.74, 0.71, 0.73, 0.75],
        ("qwen", "arabic", "baseline"): [0.61, 0.60, 0.58, 0.63, 0.59],
    }
    expected = ttest_rel(
        groups[("qwen", "arabic", "cultural")],
        groups[("qwen", "arabic", "baseline")],
    )
    comparison = run_all_comparisons(groups)[0]
    assert comparison["n_folds"] == 5
    assert comparison["t_stat"] == pytest.approx(float(expected.statistic), abs=5e-5)
    assert comparison["p_raw"] == pytest.approx(float(expected.pvalue))
    assert comparison["delta_mean"] == pytest.approx(0.128)


def test_run_all_comparisons_signs_delta_mean_as_group_a_minus_group_b() -> None:
    groups = {
        ("qwen", "arabic", "baseline"): [0.61, 0.60, 0.58, 0.63, 0.59],
        ("qwen", "arabic", "cultural"): [0.72, 0.74, 0.71, 0.73, 0.75],
    }
    comparison = run_all_comparisons(groups)[0]
    assert comparison["condition_a"] == "baseline"
    assert comparison["delta_mean"] == pytest.approx(-0.128)


def test_run_all_comparisons_skips_groups_with_fewer_than_two_folds() -> None:
    groups = {
        ("qwen", "arabic", "cultural"): [0.70],
        ("qwen", "arabic", "baseline"): [0.51, 0.60, 0.55],
        ("phi", "arabic", "baseline"): [0.62, 0.58, 0.61],
    }
    comparisons = run_all_comparisons(groups)
    assert len(comparisons) == 1
    assert comparisons[0]["comparison_type"] == "cross_architecture"
    assert comparisons[0]["model_a"] == "qwen"


def test_run_all_comparisons_truncates_unequal_fold_vectors_to_the_shorter_length() -> None:
    groups = {
        ("qwen", "arabic", "cultural"): [0.80, 0.82, 0.84, 0.90],
        ("qwen", "arabic", "baseline"): [0.60, 0.65, 0.70],
    }
    comparison = run_all_comparisons(groups)[0]
    already_truncated = run_all_comparisons({key: value[:3] for key, value in groups.items()})[0]
    assert comparison["n_folds"] == 3
    assert comparison["delta_mean"] == pytest.approx(0.17)
    assert comparison["t_stat"] == already_truncated["t_stat"]
    assert comparison["p_raw"] == already_truncated["p_raw"]


def test_holm_adjusted_p_values_are_monotone_nondecreasing_in_raw_p_order() -> None:
    result = apply_holm_bonferroni(_p_value_family([0.01, 0.011, 0.5, 0.04]))
    adjusted = [c["p_adj"] for c in result]
    assert [c["p_raw"] for c in result] == [0.01, 0.011, 0.04, 0.5]
    assert adjusted == sorted(adjusted)
    assert adjusted == [
        pytest.approx(0.04),
        pytest.approx(0.04),
        pytest.approx(0.08),
        pytest.approx(0.5),
    ]


def test_holm_lifts_a_step_down_product_that_would_undercut_a_stronger_comparison() -> None:
    result = apply_holm_bonferroni(_p_value_family([0.01, 0.011]))
    assert result[1]["p_adj"] == pytest.approx(0.02)
    assert result[0]["p_adj"] == pytest.approx(0.02)
    assert result[1]["p_adj"] > result[1]["p_raw"] * 1.0


@pytest.mark.parametrize(
    ("raw_p_values", "expected_adjusted"),
    [
        ([0.03], [0.03]),
        ([0.2, 0.9], [0.4, 0.9]),
        ([0.4, 0.45], [0.8, 0.8]),
        ([0.5, 0.6, 0.7], [1.0, 1.0, 1.0]),
        ([0.9, 0.95, 0.99, 1.0], [1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_holm_adjusted_p_values_are_capped_at_one(
    raw_p_values: list[float], expected_adjusted: list[float]
) -> None:
    result = apply_holm_bonferroni(_p_value_family(raw_p_values))
    adjusted = [c["p_adj"] for c in result]
    assert adjusted == [pytest.approx(value) for value in expected_adjusted]
    assert all(value <= 1.0 for value in adjusted)


def test_holm_sorts_the_returned_family_by_ascending_raw_p_value() -> None:
    result = apply_holm_bonferroni(_p_value_family([0.5, 0.01, 0.2]))
    assert [c["label"] for c in result] == ["c1", "c2", "c0"]


def test_holm_returns_an_empty_family_untouched() -> None:
    empty: list[dict] = []
    assert apply_holm_bonferroni(empty) is empty


def test_holm_annotates_the_callers_comparison_dicts_in_place() -> None:
    original = _p_value_family([0.5, 0.01])
    result = apply_holm_bonferroni(original)
    assert result is not original
    assert [c["label"] for c in original] == ["c0", "c1"]
    assert all("p_adj" in c and "significant" in c for c in original)
    assert original[1]["p_adj"] == result[0]["p_adj"]


def test_holm_rounds_adjusted_p_values_to_six_decimal_places() -> None:
    result = apply_holm_bonferroni(_p_value_family([0.0000123456789, 0.9]))
    assert result[0]["p_adj"] == 0.000025


def test_holm_significance_uses_a_strict_threshold_against_alpha() -> None:
    result = apply_holm_bonferroni(_p_value_family([0.025, 0.9]))
    assert result[0]["p_adj"] == pytest.approx(0.05)
    assert result[0]["significant"] is False
    assert result[1]["significant"] is False


def test_holm_significance_flags_respond_to_the_alpha_argument() -> None:
    result = apply_holm_bonferroni(_p_value_family([0.025, 0.9]), alpha=0.10)
    assert result[0]["significant"] is True
    assert result[1]["significant"] is False


@pytest.mark.parametrize("significant_only", [False, True])
def test_latex_table_returns_its_placeholder_comment_for_an_empty_frame(
    significant_only: bool,
) -> None:
    empty = _table_frame([True]).iloc[0:0]
    assert generate_latex_table(empty, significant_only=significant_only) == (
        "% No comparisons to display\n"
    )


def test_latex_table_returns_its_placeholder_when_the_filter_removes_every_row() -> None:
    frame = _table_frame([False, False])
    assert generate_latex_table(frame, significant_only=True) == ("% No comparisons to display\n")
    assert len(_body_rows(generate_latex_table(frame, significant_only=False))) == 2


def test_latex_table_drops_non_significant_rows_when_significant_only_is_set() -> None:
    frame = _table_frame([True, False, True])
    rows = _body_rows(generate_latex_table(frame, significant_only=True))
    assert len(rows) == 2
    assert all(row.startswith(r"\textbf{") for row in rows)
    assert "m1 &" not in "\n".join(rows)


def test_latex_table_bolds_only_the_rows_that_survive_the_correction() -> None:
    rows = _body_rows(generate_latex_table(_table_frame([True, False])))
    assert rows[0].startswith(r"\textbf{")
    assert rows[0].endswith(r"} \\")
    assert r"\textbf" not in rows[1]
    assert rows[1].endswith(r"0.0040 \\")


def test_latex_table_renders_one_body_row_per_comparison_with_fixed_precision() -> None:
    row = _body_rows(generate_latex_table(_table_frame([False])))[0]
    expected = (
        r"m0 & arabic & cultural & m0 & arabic & baseline & 5 & 3.500 "
        r"& 0.0010 & 0.0020 \\"
    )
    assert row == expected


def test_latex_table_wraps_the_body_in_a_single_booktabs_table_environment() -> None:
    table = generate_latex_table(_table_frame([True, False]))
    assert table.startswith(r"\begin{table*}[ht]")
    assert table.endswith(r"\end{table*}")
    assert table.count(r"\toprule") == 1
    assert table.count(r"\bottomrule") == 1
    assert r"\label{tab:holm_ttests}" in table


def test_fold_metrics_csv_flows_through_correction_into_a_bold_latex_row(
    tmp_path: Path,
) -> None:
    cultural = [0.72, 0.74, 0.71, 0.73, 0.75]
    baseline = [0.61, 0.60, 0.58, 0.63, 0.59]
    rows: list[dict[str, object]] = []
    paired = zip(cultural, baseline, strict=True)
    for fold, (cultural_score, baseline_score) in enumerate(paired, start=1):
        rows.append(
            {
                "model_name": "qwen",
                "culture": "arabic",
                "condition": "cultural",
                "fold": fold,
                "f1_macro": cultural_score,
            }
        )
        rows.append(
            {
                "model_name": "qwen",
                "culture": "arabic",
                "condition": "baseline",
                "fold": fold,
                "f1_macro": baseline_score,
            }
        )
    groups = load_fold_metrics(_write_fold_metrics(tmp_path, rows), "f1_macro")
    comparisons = apply_holm_bonferroni(run_all_comparisons(groups))
    assert len(comparisons) == 1
    assert comparisons[0]["comparison_type"] == "cultural_vs_baseline"
    assert comparisons[0]["p_adj"] == pytest.approx(comparisons[0]["p_raw"], abs=1e-6)
    assert comparisons[0]["significant"] is True
    body = _body_rows(generate_latex_table(pd.DataFrame(comparisons), significant_only=True))
    assert len(body) == 1
    assert body[0].startswith(r"\textbf{qwen & arabic & cultural &")
