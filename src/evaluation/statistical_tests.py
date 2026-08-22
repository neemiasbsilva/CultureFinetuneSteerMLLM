"""
Holm-Bonferroni corrected paired t-tests across culture × model × condition.

Extends ../multimodal-LLMs-see-sentiment/scripts/statistical_tests.py
with the additional comparison dimensions required by CultureVLM.

Comparison dimensions (162 total at full scale):
  Dim 1: cultural_model vs. baseline_model  (9 cultures × 3 architectures = 27)
  Dim 2: cross-architecture within cultural  (C(3,2) × 9 cultures = 27)
  Dim 3: cross-culture within architecture   (C(9,2) × 3 architectures = 108)

Primary metric: fold-wise val macro-F1 (5 observations per comparison).

Usage:
    uv run python src/evaluation/statistical_tests.py \
        --fold-metrics outputs/training/fold_metrics.csv \
        --metric f1_macro \
        --output outputs/training/stats/holm_ttests

    # Significant comparisons only:
    uv run python src/evaluation/statistical_tests.py \
        --fold-metrics outputs/training/fold_metrics.csv \
        --significant-only
"""

import argparse
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from src.data import cultures

CULTURES = cultures.CULTURES
MODELS = ["qwen3_5_2b", "phi4", "gemma4_e2b"]
N_FOLDS = 5
ALPHA = 0.05


def load_fold_metrics(path: str, metric: str) -> dict[tuple[str, str, str], list[float]]:
    """Load per-fold metric values grouped by (model_name, culture, condition).

    Args:
        path (str): Path to fold_metrics.csv.
        metric (str): Column name to extract (e.g., "f1_macro").

    Returns:
        dict: {(model_name, culture, condition): [fold_1_value, ..., fold_N_value]}
    """
    df = pd.read_csv(path)
    groups: dict[tuple[str, str, str], list[float]] = {}
    for _, row in df.iterrows():
        key = (str(row["model_name"]), str(row["culture"]), str(row["condition"]))
        groups.setdefault(key, []).append(float(row[metric]))
    return groups


def run_all_comparisons(
    groups: dict[tuple[str, str, str], list[float]], alpha: float = ALPHA
) -> list[dict[str, Any]]:
    """Run all pairwise t-tests and collect raw p-values.

    Only pairs that differ on exactly one dimension (model, culture, or
    condition) are compared; pairs differing on more than one dimension are
    skipped because their difference cannot be attributed to a single factor.
    """
    comparisons: list[dict[str, Any]] = []
    keys = list(groups.keys())

    for k1, k2 in itertools.combinations(keys, 2):
        model1, culture1, cond1 = k1
        model2, culture2, cond2 = k2

        same_model = model1 == model2
        same_culture = culture1 == culture2
        same_cond = cond1 == cond2

        if sum([not same_model, not same_culture, not same_cond]) != 1:
            continue

        v1 = groups[k1]
        v2 = groups[k2]
        if len(v1) < 2 or len(v2) < 2:
            continue

        n = min(len(v1), len(v2))
        t_stat, p_val = ttest_rel(v1[:n], v2[:n])
        delta_mean = float(np.mean(np.array(v1[:n]) - np.array(v2[:n])))

        comparisons.append(
            {
                "model_a": model1,
                "culture_a": culture1,
                "condition_a": cond1,
                "model_b": model2,
                "culture_b": culture2,
                "condition_b": cond2,
                "n_folds": n,
                "t_stat": round(t_stat, 4),
                "p_raw": p_val,
                "delta_mean": round(delta_mean, 6),
                "comparison_type": (
                    "cultural_vs_baseline"
                    if not same_cond
                    else ("cross_architecture" if not same_model else "cross_culture")
                ),
            }
        )

    return comparisons


def apply_holm_bonferroni(
    comparisons: list[dict[str, Any]], alpha: float = ALPHA
) -> list[dict[str, Any]]:
    """Apply Holm-Bonferroni correction in-place; returns sorted list.

    Holm adjusted p-values are the cumulative maximum of the step-down
    products.  Omitting that monotonicity step can make a less significant
    comparison receive a smaller adjusted p-value than an earlier one.
    """
    if not comparisons:
        return comparisons
    comparisons = sorted(comparisons, key=lambda x: x["p_raw"])
    n = len(comparisons)
    running_max = 0.0
    for rank, comp in enumerate(comparisons):
        p_adj = float(comp["p_raw"]) * (n - rank)
        running_max = max(running_max, p_adj)
        comp["p_adj"] = round(min(running_max, 1.0), 6)
        comp["significant"] = comp["p_adj"] < alpha
    return comparisons


def generate_latex_table(
    df: pd.DataFrame,
    significant_only: bool = False,
    alpha: float = ALPHA,
) -> str:
    if significant_only:
        df = df[df["significant"]]
    if df.empty:
        return "% No comparisons to display\n"

    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\caption{Pairwise model comparisons via two-sided paired $t$-tests. "
        r"P-values are Holm--Bonferroni adjusted ($\alpha=0.05$). "
        r"Bold rows are significant after correction.}",
        r"\label{tab:holm_ttests}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllllrrrr}",
        r"\toprule",
        r"Model A & Culture A & Cond A & Model B & Culture B & Cond B & "
        r"$n$ & $t$ & $p_{\text{raw}}$ & $p_{\text{adj}}$ \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        bold = r"\textbf{" if row["significant"] else ""
        bold_end = r"}" if row["significant"] else ""
        lines.append(
            f"{bold}{row['model_a']} & {row['culture_a']} & {row['condition_a']} & "
            f"{row['model_b']} & {row['culture_b']} & {row['condition_b']} & "
            f"{row['n_folds']} & {row['t_stat']:.3f} & "
            f"{row['p_raw']:.4f} & {row['p_adj']:.4f}{bold_end} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Holm-Bonferroni corrected t-tests for CultureVLM."
    )
    parser.add_argument("--fold-metrics", default="outputs/training/fold_metrics.csv")
    parser.add_argument(
        "--metric", default="f1_macro", choices=["f1_macro", "f1_weighted", "mae", "qwk"]
    )
    parser.add_argument("--output", default="outputs/training/stats/holm_ttests")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--significant-only", action="store_true")
    args = parser.parse_args()

    print(f"Loading fold metrics from {args.fold_metrics}...")
    groups = load_fold_metrics(args.fold_metrics, args.metric)
    print(f"Found {len(groups)} (model, culture, condition) groups.")

    comparisons = run_all_comparisons(groups, alpha=args.alpha)
    print(f"Running {len(comparisons)} pairwise comparisons...")
    comparisons = apply_holm_bonferroni(comparisons, alpha=args.alpha)

    df = pd.DataFrame(comparisons)
    n_sig = df["significant"].sum() if "significant" in df.columns else 0
    print(f"Significant after correction: {n_sig}/{len(df)}")

    out_prefix = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_prefix) + ".csv", index=False)
    print(f"Saved CSV: {out_prefix}.csv")

    tex = generate_latex_table(df, significant_only=args.significant_only, alpha=args.alpha)
    with open(str(out_prefix) + ".tex", "w") as f:
        f.write(tex)
    print(f"Saved LaTeX: {out_prefix}.tex")


if __name__ == "__main__":
    main()
