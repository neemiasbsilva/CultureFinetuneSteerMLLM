"""
Stage 4 evaluation: annotation quality vs. σ₃P₅ ground truth.

Follows the mllm-persona-evaluation methodology (notebooks/03_perceptsent_cv_agreement.ipynb):
  - Load annotation JSONL outputs (from Stage 3)
  - Load σ₃P₅ ground-truth labels (agreement_p5_sigma3.csv)
  - Evaluate per (culture × model): accuracy, macro-F1, Cohen's kappa (linear-weighted), MAE
  - Image-bootstrap 95% CI (1,000 resamples)
  - Cross-fold comparison via Holm-Bonferroni corrected paired t-tests

Usage:
    uv run python src/evaluation/annotation_eval.py
    uv run python src/evaluation/annotation_eval.py --conditions cultural baseline inference_only
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from scipy.stats import ttest_rel
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)

load_dotenv()
console = Console()

ANNOTATIONS_DIR = Path("outputs/annotations")
AGREEMENT_CSV = os.getenv(
    "AGREEMENT_CSV",
    "../multimodal-LLMs-see-sentiment/data/agreement_p5_sigma3.csv",
)
FOLDS_DIR = Path("data/folds")
OUTPUT_DIR = Path("outputs/evaluation")
N_BOOTSTRAP = 1_000
ALPHA = 0.05


def load_ground_truth() -> dict[str, int]:
    df = pd.read_csv(AGREEMENT_CSV, index_col=0)
    df = df.rename(columns={"id": "image_id"})
    return dict(zip(df["image_id"].astype(str), df["sentiment"].astype(int)))


def load_annotations(conditions: list[str] | None = None) -> pd.DataFrame:
    records = []
    for model_dir in sorted(ANNOTATIONS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for culture_dir in sorted(model_dir.iterdir()):
            if not culture_dir.is_dir():
                continue
            ann_file = culture_dir / "annotations.jsonl"
            if not ann_file.exists():
                continue
            with open(ann_file) as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
    df = pd.DataFrame(records)
    if df.empty:
        return df
    if conditions:
        df = df[df["condition"].isin(conditions)]
    return df


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa_linear": cohen_kappa_score(y_true, y_pred, weights="linear"),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = N_BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Returns {metric: (mean, ci_low, ci_high)} via image bootstrap."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_images = len(y_true)
    boot = {k: [] for k in ("accuracy", "f1_macro", "kappa_linear", "mae")}
    for _ in range(n):
        idx = rng.integers(0, n_images, size=n_images)
        m = _metrics(y_true[idx], y_pred[idx])
        for k, v in m.items():
            boot[k].append(v)
    result = {}
    for k, vals in boot.items():
        arr = np.array(vals)
        result[k] = (float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return result


def evaluate_per_fold(
    ann_df: pd.DataFrame,
    gt: dict[str, int],
) -> pd.DataFrame:
    """Compute per-fold accuracy for each (culture × model × condition).
    Uses the pre-built image folds from make_folds.py."""
    rows = []
    fold_files = sorted(FOLDS_DIR.glob("fold_*_val.csv"))
    if not fold_files:
        console.print("[yellow]No fold files found — skipping per-fold evaluation.[/yellow]")
        return pd.DataFrame()

    for fold_path in fold_files:
        fold_n = int(fold_path.stem.split("_")[1])
        fold_ids = set(pd.read_csv(fold_path)["image_id"].astype(str).tolist())
        fold_ann = ann_df[ann_df["image_id"].isin(fold_ids)].copy()

        for (culture, model_name, condition), grp in fold_ann.groupby(["culture", "model_name", "condition"]):
            y_true = np.array([gt[img] for img in grp["image_id"] if img in gt])
            y_pred = np.array([grp.loc[grp["image_id"] == img, "predicted_sentiment"].iloc[0]
                               for img in grp["image_id"] if img in gt])
            if len(y_true) == 0:
                continue
            m = _metrics(y_true, y_pred)
            rows.append({
                "fold": fold_n,
                "culture": culture,
                "model_name": model_name,
                "condition": condition,
                "n_images": len(y_true),
                **m,
            })
    return pd.DataFrame(rows)


def holm_bonferroni(comparisons: list[dict], alpha: float = ALPHA) -> pd.DataFrame:
    """Apply Holm-Bonferroni correction to a list of paired t-test comparisons."""
    results = []
    for c in comparisons:
        a, b = np.array(c["vals_a"]), np.array(c["vals_b"])
        if len(a) < 2 or len(b) < 2:
            continue
        stat, p = ttest_rel(a, b)
        results.append({**c, "t_stat": stat, "p_raw": p})
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).sort_values("p_raw").reset_index(drop=True)
    m = len(df)
    df["p_adjusted"] = (df["p_raw"] * (m - df.index)).clip(upper=1.0)
    df["significant"] = df["p_adjusted"] < alpha
    return df[["label_a", "label_b", "metric", "t_stat", "p_raw", "p_adjusted", "significant"]]


def build_holm_comparisons(fold_df: pd.DataFrame, metric: str = "f1_macro") -> list[dict]:
    """Build trained-adapter comparisons plus raw base-model references."""
    if fold_df.empty:
        return []

    def _fold_vals(sub):
        return sub.groupby("fold")[metric].mean().sort_index().tolist()

    trained_df = fold_df[fold_df["condition"].isin(["cultural", "baseline"])]
    cultures = sorted(trained_df["culture"].unique())
    models = sorted(fold_df["model_name"].unique())
    comparisons = []

    # Dim 1: cultural vs. baseline per (culture × model)
    for culture in cultures:
        for model in models:
            a = fold_df[(fold_df.culture == culture) & (fold_df.model_name == model) & (fold_df.condition == "cultural")]
            b = fold_df[(fold_df.culture == culture) & (fold_df.model_name == model) & (fold_df.condition == "baseline")]
            if a.empty or b.empty:
                continue
            comparisons.append({
                "label_a": f"{model}_{culture}_cultural",
                "label_b": f"{model}_{culture}_baseline",
                "metric": metric,
                "vals_a": _fold_vals(a),
                "vals_b": _fold_vals(b),
            })

    # Dim 2: trained adapters vs. raw base model (no LoRA adapter)
    for culture in cultures:
        for model in models:
            raw = fold_df[
                (fold_df.model_name == model)
                & (fold_df.condition == "inference_only")
            ]
            if raw.empty:
                continue
            for condition in ("cultural", "baseline"):
                trained = fold_df[
                    (fold_df.culture == culture)
                    & (fold_df.model_name == model)
                    & (fold_df.condition == condition)
                ]
                if trained.empty:
                    continue
                comparisons.append({
                    "label_a": f"{model}_{culture}_{condition}",
                    "label_b": f"{model}_base_inference_only",
                    "metric": metric,
                    "vals_a": _fold_vals(trained),
                    "vals_b": _fold_vals(raw),
                })

    # Dim 3: cross-architecture within cultural
    from itertools import combinations
    for culture in cultures:
        for m1, m2 in combinations(models, 2):
            a = fold_df[(fold_df.culture == culture) & (fold_df.model_name == m1) & (fold_df.condition == "cultural")]
            b = fold_df[(fold_df.culture == culture) & (fold_df.model_name == m2) & (fold_df.condition == "cultural")]
            if a.empty or b.empty:
                continue
            comparisons.append({
                "label_a": f"{m1}_{culture}_cultural",
                "label_b": f"{m2}_{culture}_cultural",
                "metric": metric,
                "vals_a": _fold_vals(a),
                "vals_b": _fold_vals(b),
            })

    # Dim 4: cross-culture within architecture
    for model in models:
        for c1, c2 in combinations(cultures, 2):
            a = fold_df[(fold_df.model_name == model) & (fold_df.culture == c1) & (fold_df.condition == "cultural")]
            b = fold_df[(fold_df.model_name == model) & (fold_df.culture == c2) & (fold_df.condition == "cultural")]
            if a.empty or b.empty:
                continue
            comparisons.append({
                "label_a": f"{model}_{c1}_cultural",
                "label_b": f"{model}_{c2}_cultural",
                "metric": metric,
                "vals_a": _fold_vals(a),
                "vals_b": _fold_vals(b),
            })

    return comparisons


def print_summary_table(boot_results: pd.DataFrame) -> None:
    table = Table(title="Annotation Evaluation — Mean ± 95% CI (bootstrap)")
    for col in ["condition", "culture", "model_name", "n_images",
                "accuracy", "f1_macro", "kappa_linear", "mae"]:
        table.add_column(col, justify="right" if col in ("n_images",) else "left")
    for _, row in boot_results.iterrows():
        table.add_row(
            row["condition"], row["culture"], row["model_name"], str(int(row["n_images"])),
            f"{row['accuracy_mean']:.3f} [{row['accuracy_lo']:.3f}, {row['accuracy_hi']:.3f}]",
            f"{row['f1_macro_mean']:.3f} [{row['f1_macro_lo']:.3f}, {row['f1_macro_hi']:.3f}]",
            f"{row['kappa_linear_mean']:.3f} [{row['kappa_linear_lo']:.3f}, {row['kappa_linear_hi']:.3f}]",
            f"{row['mae_mean']:.3f} [{row['mae_lo']:.3f}, {row['mae_hi']:.3f}]",
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate annotation quality vs. σ₃P₅ ground truth.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["cultural", "baseline", "inference_only"],
    )
    parser.add_argument("--metric", default="f1_macro", help="Metric for Holm-Bonferroni comparison")
    args = parser.parse_args()

    console.rule("[bold blue]CultureVLM — Annotation Evaluation (Stage 4)[/bold blue]")

    gt = load_ground_truth()
    console.print(f"Ground truth: {len(gt)} images\n")

    ann_df = load_annotations(conditions=args.conditions)
    if ann_df.empty:
        console.print("[red]No annotation files found in outputs/annotations/. Run Stage 3 first.[/red]")
        return
    console.print(f"Loaded {len(ann_df):,} annotations across {ann_df['culture'].nunique()} cultures, "
                  f"{ann_df['model_name'].nunique()} models\n")

    ann_df["image_id"] = ann_df["image_id"].astype(str)
    ann_df = ann_df[ann_df["image_id"].isin(gt)]

    # Bootstrap metrics per (culture × model × condition)
    rng = np.random.default_rng(42)
    boot_rows = []
    for (culture, model_name, condition), grp in ann_df.groupby(["culture", "model_name", "condition"]):
        y_true = np.array([gt[img] for img in grp["image_id"]])
        y_pred = grp["predicted_sentiment"].fillna(2).astype(int).to_numpy()
        boot = bootstrap_metrics(y_true, y_pred, rng=rng)
        row: dict = {"culture": culture, "model_name": model_name, "condition": condition,
                     "n_images": len(y_true)}
        for metric, (mean, lo, hi) in boot.items():
            row[f"{metric}_mean"] = mean
            row[f"{metric}_lo"] = lo
            row[f"{metric}_hi"] = hi
        boot_rows.append(row)

    boot_df = pd.DataFrame(boot_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    boot_df.to_csv(OUTPUT_DIR / "annotation_metrics_bootstrap.csv", index=False)
    console.print(f"Bootstrap metrics → {OUTPUT_DIR}/annotation_metrics_bootstrap.csv\n")
    print_summary_table(boot_df)

    # Per-fold evaluation for Holm-Bonferroni
    fold_df = evaluate_per_fold(ann_df, gt)
    if not fold_df.empty:
        fold_df.to_csv(OUTPUT_DIR / "annotation_metrics_per_fold.csv", index=False)
        console.print(f"\nPer-fold metrics → {OUTPUT_DIR}/annotation_metrics_per_fold.csv")

        comparisons = build_holm_comparisons(fold_df, metric=args.metric)
        console.print(f"Running Holm-Bonferroni on {len(comparisons)} comparisons ({args.metric})...")
        holm_df = holm_bonferroni(comparisons)
        if not holm_df.empty:
            holm_df.to_csv(OUTPUT_DIR / f"holm_bonferroni_{args.metric}.csv", index=False)
            n_sig = holm_df["significant"].sum()
            console.print(f"Significant after correction: {n_sig}/{len(holm_df)} (α={ALPHA})")
            console.print(f"Results → {OUTPUT_DIR}/holm_bonferroni_{args.metric}.csv")

    console.print("\n[bold green]✓ Evaluation complete[/bold green]")


if __name__ == "__main__":
    main()
