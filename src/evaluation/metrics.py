"""
Training evaluation metrics for CultureVLM fine-tuning.

Computes per-fold classification quality (macro-F1, MAE, QWK) used as
input to the Holm-Bonferroni statistical comparison across conditions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)


def compute_classification_metrics(
    y_true: list[int], y_pred: list[int]
) -> dict[str, float]:
    """Primary metrics used in Holm-Bonferroni comparisons."""
    valid = [(t, p) for t, p in zip(y_true, y_pred) if t >= 0 and p >= 0]
    if not valid:
        return {
            "f1_macro": 0.0, "f1_weighted": 0.0,
            "f1_per_class": [], "mae": 5.0, "qwk": 0.0, "accuracy": 0.0,
        }
    yt, yp = zip(*valid)
    return {
        "f1_macro":    round(float(f1_score(yt, yp, average="macro", zero_division=0)), 6),
        "f1_weighted": round(float(f1_score(yt, yp, average="weighted", zero_division=0)), 6),
        "f1_per_class": [round(float(v), 6) for v in f1_score(yt, yp, average=None, zero_division=0)],
        "mae":         round(float(mean_absolute_error(yt, yp)), 6),
        "qwk":         round(float(cohen_kappa_score(yt, yp, weights="quadratic")), 6),
        "accuracy":    round(float(accuracy_score(yt, yp)), 6),
        "n":           len(yt),
    }


def aggregate_fold_metrics(fold_records: list[dict]) -> dict[str, float]:
    """
    Aggregate per-fold metric records into mean ± std.
    Input: list of dicts from compute_classification_metrics (one per fold).
    """
    if not fold_records:
        return {}
    keys = [k for k in fold_records[0] if isinstance(fold_records[0][k], float)]
    result = {}
    for k in keys:
        vals = [r[k] for r in fold_records if k in r]
        result[f"{k}_mean"] = round(float(np.mean(vals)), 6)
        result[f"{k}_std"] = round(float(np.std(vals, ddof=1)), 6)
    return result


def build_fold_metrics_csv(
    results: list[dict], output_path: str
) -> pd.DataFrame:
    """
    Build the fold_metrics.csv consumed by statistical_tests.py.

    Expected result schema per entry:
        {model_name, culture, condition, fold, f1_macro, f1_weighted, mae, qwk, accuracy}
    """
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    return df
