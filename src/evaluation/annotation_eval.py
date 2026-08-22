"""Stage-4 evaluation of matched base/WVS annotation runs.

Repeated generations are reduced to one prediction per image using a
deterministic mode rule.  Every condition is then evaluated on the same valid
image set, and uncertainty/comparisons resample paired images rather than raw
generation rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from scipy.stats import t as student_t
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)

from src.analysis.convergence import compute_condition_pair_agreement
from src.annotation.conditions import CONDITIONS, normalize_condition
from src.data import cultures

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
METRIC_NAMES = (
    "f1_macro",
    "f1_weighted",
    "accuracy",
    "mae",
    "kappa_linear",
    "kappa_quadratic",
)
COMPARISON_FAMILIES: dict[str, tuple[str, str]] = {
    "wvs_vs_base": ("wvs_cultural", "inference_only"),
}
DEFAULT_CAPTION_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STUDY_CULTURES = cultures.CULTURES
SHARED_MODELS = ("gemma4_e2b", "gemma4_e4b", "qwen3_vl_8b", "gemma4_31b")


def load_ground_truth(path: str | Path | None = None) -> dict[str, int]:
    """Load the sigma3-P5 reference label keyed by string image ID."""

    df = pd.read_csv(path or AGREEMENT_CSV, index_col=0)
    df = df.rename(columns={"id": "image_id"})
    return dict(zip(df["image_id"].astype(str), df["sentiment"].astype(int), strict=True))


def _annotation_files(root: Path, include_failures: bool) -> list[tuple[Path, bool]]:
    if not root.exists():
        return []
    result: list[tuple[Path, bool]] = []
    for path in sorted(root.rglob("*.jsonl")):
        name = path.name.lower()
        is_failure = "failure" in name
        if name == "annotations.jsonl" or name.startswith("annotations_"):
            result.append((path, False))
        elif include_failures and (name == "annotation_failures.jsonl" or is_failure):
            result.append((path, True))
    return result


def _path_metadata(path: Path, root: Path) -> dict[str, str]:
    """Best-effort metadata for legacy and condition-subdirectory layouts."""

    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return {}
    metadata: dict[str, str] = {}
    if len(parts) >= 3:
        metadata["model_name"] = parts[0]
        metadata["culture"] = parts[1]
    if len(parts) >= 4:
        candidate = normalize_condition(parts[2], strict=False)
        if candidate in CONDITIONS:
            metadata["condition"] = candidate
    return metadata


def load_annotations(
    conditions: Sequence[str] | None = None,
    annotations_dir: Path | str = ANNOTATIONS_DIR,
    *,
    include_failures: bool = True,
) -> pd.DataFrame:
    """Load annotation attempts from legacy or condition-aware layouts.

    A later successful record wins over an earlier failure with the same
    annotation ID.  Legacy ``condition='cultural'`` rows are normalized to
    ``wvs_cultural`` at the boundary.
    """

    root = Path(annotations_dir)
    by_id: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for path, source_failure in _annotation_files(root, include_failures):
        defaults = _path_metadata(path, root)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    warnings.warn(
                        f"Skipping malformed JSONL at {path}:{line_number}: {exc}",
                        stacklevel=2,
                    )
                    continue
                for key, value in defaults.items():
                    record.setdefault(key, value)
                record["condition"] = normalize_condition(
                    record.get("condition", defaults.get("condition", "")),
                    strict=False,
                )
                prediction = pd.to_numeric(
                    pd.Series([record.get("predicted_sentiment")]), errors="coerce"
                ).iloc[0]
                inferred_valid = (
                    not source_failure
                    and not record.get("error")
                    and pd.notna(prediction)
                    and 0 <= int(prediction) <= 4
                )
                record["parse_valid"] = bool(record.get("parse_valid", inferred_valid))
                record["_source_failure"] = source_failure
                annotation_id = record.get("annotation_id")
                if not annotation_id:
                    anonymous.append(record)
                    continue
                key = str(annotation_id)
                previous = by_id.get(key)
                if previous is None or (
                    record["parse_valid"] and not previous.get("parse_valid", False)
                ):
                    by_id[key] = record

    df = pd.DataFrame([*by_id.values(), *anonymous])
    if df.empty:
        return df
    for column in ("model_name", "culture", "condition", "image_id"):
        if column not in df:
            df[column] = ""
        df[column] = df[column].astype(str)
    df["condition"] = df["condition"].map(lambda value: normalize_condition(value, strict=False))
    df["predicted_sentiment"] = pd.to_numeric(
        cast("pd.Series[Any]", df.get("predicted_sentiment")), errors="coerce"
    )
    if "run_index" not in df:
        df["run_index"] = 1
    df["run_index"] = pd.to_numeric(df["run_index"], errors="coerce").fillna(1).astype(int)
    df["parse_valid"] = df["parse_valid"].fillna(False).astype(bool) & df[
        "predicted_sentiment"
    ].between(0, 4, inclusive="both")
    if conditions:
        wanted = {normalize_condition(value, strict=False) for value in conditions}
        df = df[df["condition"].isin(wanted)]
    return df.reset_index(drop=True)


def mode_with_median_tiebreak(values: Iterable[int | float]) -> int:
    """Mode; ties go closest to the sample median, then to the lower label."""

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("cannot aggregate an empty prediction set")
    labels, counts = np.unique(array.astype(int), return_counts=True)
    candidates = labels[counts == counts.max()]
    median = float(np.median(array))
    return int(min(candidates, key=lambda label: (abs(float(label) - median), int(label))))


def aggregate_repetitions(
    annotations: pd.DataFrame,
    gt: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Reduce repeated passes to one deterministic row per image/condition."""

    if annotations.empty:
        return pd.DataFrame()
    df = annotations.copy()
    df["condition"] = df["condition"].map(lambda value: normalize_condition(value, strict=False))
    df["image_id"] = df["image_id"].astype(str)
    df["predicted_sentiment"] = pd.to_numeric(df["predicted_sentiment"], errors="coerce")
    valid_mask = df.get("parse_valid", pd.Series(True, index=df.index)).astype(bool)
    valid_mask &= df["predicted_sentiment"].between(0, 4, inclusive="both")
    df["_valid"] = valid_mask

    group_columns = ["model_name", "culture", "condition", "image_id"]
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_columns, dropna=False, sort=True):
        valid = group[group["_valid"]].sort_values("run_index", kind="stable")
        prediction: float = np.nan
        representative = group.sort_values("run_index", kind="stable").iloc[0]
        if not valid.empty:
            prediction = mode_with_median_tiebreak(valid["predicted_sentiment"])
            candidates = valid[valid["predicted_sentiment"].astype(int) == int(prediction)]
            representative = candidates.iloc[0]
        image_id = str(keys[3])
        truth = (
            gt.get(image_id)
            if gt is not None
            else representative.get("ground_truth_sentiment", np.nan)
        )
        row = {
            "model_name": str(keys[0]),
            "culture": str(keys[1]),
            "condition": str(keys[2]),
            "image_id": image_id,
            "ground_truth_sentiment": truth,
            "predicted_sentiment": prediction,
            "n_attempts": len(group),
            "n_valid_runs": len(valid),
            "parse_coverage": float(len(valid) / len(group)),
            "selected_run_index": int(representative.get("run_index", 1)),
            "caption": representative.get("caption", "") or "",
            "justification": representative.get("justification", "") or "",
            "predicted_perceptions": representative.get("predicted_perceptions", []),
            "checkpoint_identity": representative.get("checkpoint_identity"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def incomplete_pass_groups(
    annotations: pd.DataFrame,
    expected_runs: int = 5,
) -> pd.DataFrame:
    """List model/culture/condition/images missing any planned pass index."""
    expected = set(range(1, expected_runs + 1))
    rows: list[dict[str, object]] = []
    keys = ["model_name", "culture", "condition", "image_id"]
    for values, group in annotations.groupby(keys, sort=True, dropna=False):
        observed = set(pd.to_numeric(group["run_index"], errors="coerce").dropna().astype(int))
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        if missing or unexpected:
            rows.append(
                {
                    **dict(zip(keys, values, strict=True)),
                    "observed_runs": sorted(observed),
                    "missing_runs": missing,
                    "unexpected_runs": unexpected,
                }
            )
    return pd.DataFrame(rows)


def assert_matched_generation_seeds(annotations: pd.DataFrame) -> None:
    """Fail if a model/image/pass used different planned seeds by condition."""
    if "seed" not in annotations or annotations["seed"].isna().all():
        return
    work = annotations.dropna(subset=["seed"]).copy()
    mismatch = work.groupby(["model_name", "image_id", "run_index"], sort=True)["seed"].nunique()
    mismatch = mismatch[mismatch > 1]
    if not mismatch.empty:
        raise ValueError(
            f"Found {len(mismatch):,} model/image/pass groups with unmatched generation seeds"
        )


def attempted_panel_violations(
    annotations: pd.DataFrame,
    conditions: Sequence[str],
    expected_cultures: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Report condition arms that did not attempt the identical image set."""
    wanted = [normalize_condition(value) for value in conditions]
    trained = [value for value in wanted if value != "inference_only"]
    rows: list[dict[str, object]] = []
    for model_name, model in annotations.groupby("model_name", sort=True):
        base_images = set(model[model["condition"] == "inference_only"]["image_id"].astype(str))
        culture_values = (
            sorted(str(value) for value in expected_cultures)
            if expected_cultures is not None
            else sorted(model[model["condition"].isin(trained)]["culture"].astype(str).unique())
        )
        for culture in culture_values:
            culture_rows = model[model["culture"] == culture]
            arm_sets: dict[str, set[Any]] = {
                condition: set(
                    culture_rows[culture_rows["condition"] == condition]["image_id"].astype(str)
                )
                for condition in trained
            }
            if "inference_only" in wanted:
                arm_sets["inference_only"] = base_images
            union = set().union(*arm_sets.values()) if arm_sets else set()
            for condition, images in arm_sets.items():
                missing = union - images
                extra = images - (set.intersection(*arm_sets.values()) if arm_sets else set())
                if missing or extra:
                    rows.append(
                        {
                            "model_name": model_name,
                            "culture": culture,
                            "condition": condition,
                            "n_attempted_images": len(images),
                            "n_union_images": len(union),
                            "n_missing_from_union": len(missing),
                        }
                    )
    return pd.DataFrame(rows)


def restrict_to_common_valid_images(
    panel: pd.DataFrame,
    conditions: Sequence[str],
) -> pd.DataFrame:
    """Keep the intersection of valid image IDs within every model/culture."""

    if panel.empty:
        return panel.copy()
    wanted = [normalize_condition(value) for value in conditions]
    result: list[pd.DataFrame] = []
    for _, group in panel.groupby(["model_name", "culture"], sort=True):
        image_sets: list[set[str]] = []
        complete = True
        for condition in wanted:
            condition_rows = group[
                (group["condition"] == condition)
                & pd.to_numeric(group["predicted_sentiment"], errors="coerce").between(0, 4)
            ]
            if condition_rows.empty:
                complete = False
                break
            image_sets.append(set(condition_rows["image_id"].astype(str)))
        if not complete:
            continue
        common = set.intersection(*image_sets)
        if common:
            result.append(
                group[group["condition"].isin(wanted) & group["image_id"].astype(str).isin(common)]
            )
    if not result:
        return panel.iloc[0:0].copy()
    return pd.concat(result, ignore_index=True)


def build_common_condition_panel(
    aggregated: pd.DataFrame,
    conditions: Sequence[str],
) -> pd.DataFrame:
    """Replicate the culture-neutral base rows, then form matched panels.

    Older annotation files may store duplicate base runs under culture
    directories, so the canonical culture-neutral location is preferred
    whenever it is present.
    """

    if aggregated.empty:
        return aggregated.copy()
    wanted = [normalize_condition(value) for value in conditions]
    trained_conditions = [value for value in wanted if value != "inference_only"]
    trained = aggregated[aggregated["condition"].isin(trained_conditions)].copy()
    if "inference_only" not in wanted:
        return restrict_to_common_valid_images(trained, wanted)

    base = aggregated[aggregated["condition"] == "inference_only"].copy()
    rows: list[pd.DataFrame] = []
    for (model_name, culture), culture_rows in trained.groupby(
        ["model_name", "culture"], sort=True
    ):
        model_base = base[base["model_name"] == model_name].copy()
        if model_base.empty:
            continue
        neutral = model_base[model_base["culture"] == "inference_only"]
        if not neutral.empty:
            model_base = neutral
        model_base = model_base.sort_values("culture", kind="stable").drop_duplicates(
            "image_id", keep="first"
        )
        model_base["culture"] = cast("str", culture)
        rows.extend([culture_rows, model_base])
    if not rows:
        return aggregated.iloc[0:0].copy()
    expanded = pd.concat(rows, ignore_index=True)
    return restrict_to_common_valid_images(expanded, wanted)


def _safe_kappa(y_true: np.ndarray, y_pred: np.ndarray, weights: str) -> float:
    if np.array_equal(y_true, y_pred):
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = float(cohen_kappa_score(y_true, y_pred, weights=weights))
    return value if np.isfinite(value) else 0.0


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """All primary/secondary ordinal classification metrics."""

    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    truth = truth[valid].astype(int)
    prediction = prediction[valid].astype(int)
    if truth.size == 0:
        return {name: float("nan") for name in METRIC_NAMES}
    return {
        "f1_macro": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "kappa_linear": _safe_kappa(truth, prediction, "linear"),
        "kappa_quadratic": _safe_kappa(truth, prediction, "quadratic"),
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = N_BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Return ``metric -> (bootstrap mean, CI low, CI high)`` by image."""

    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    if len(truth) == 0:
        return {name: (float("nan"),) * 3 for name in METRIC_NAMES}
    rng = rng or np.random.default_rng(42)
    values: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for _ in range(n):
        indices = rng.integers(0, len(truth), size=len(truth))
        sample = _metrics(truth[indices], prediction[indices])
        for name in METRIC_NAMES:
            values[name].append(sample[name])
    return {
        name: (
            float(np.nanmean(samples)),
            float(np.nanpercentile(samples, 2.5)),
            float(np.nanpercentile(samples, 97.5)),
        )
        for name, samples in values.items()
    }


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\0".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**32)


def encode_caption_embeddings(
    panel: pd.DataFrame,
    *,
    model_name: str = DEFAULT_CAPTION_EMBEDDING_MODEL,
    cache_dir: Path | str,
    device: str | None = None,
    batch_size: int = 128,
) -> np.ndarray:
    """Encode row-aligned captions with a content-addressed semantic cache."""
    digest = hashlib.sha256(model_name.encode("utf-8"))
    identity_columns = [
        "model_name",
        "culture",
        "condition",
        "image_id",
        "caption",
    ]
    for row in panel[identity_columns].itertuples(index=False, name=None):
        digest.update(json.dumps(row, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\0")
    cache_path = Path(cache_dir) / f"caption_{digest.hexdigest()[:20]}.npy"
    if device is None:
        from src.utils.device import get_device

        device = get_device()
    from src.analysis.embeddings import EmbeddingManager

    manager = EmbeddingManager(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )
    return manager.encode_column(panel, "caption", cache_path=cache_path)


def evaluate_metrics(
    panel: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 42,
) -> pd.DataFrame:
    """Metrics and image-bootstrap CIs for every matched condition."""

    rows: list[dict[str, Any]] = []
    for (model, culture, condition), group in panel.groupby(
        ["model_name", "culture", "condition"], sort=True
    ):
        group = group.sort_values("image_id", kind="stable")
        truth = group["ground_truth_sentiment"].to_numpy(dtype=int)
        prediction = group["predicted_sentiment"].to_numpy(dtype=int)
        point = _metrics(truth, prediction)
        boot = bootstrap_metrics(
            truth,
            prediction,
            n=n_bootstrap,
            rng=np.random.default_rng(_stable_seed(seed, model, culture)),
        )
        row: dict[str, object] = {
            "model_name": model,
            "culture": culture,
            "condition": condition,
            "n_images": len(group),
            "n_attempts": int(group["n_attempts"].sum()),
            "n_valid_runs": int(group["n_valid_runs"].sum()),
            "parse_coverage": float(group["n_valid_runs"].sum() / group["n_attempts"].sum()),
        }
        for name in METRIC_NAMES:
            mean, low, high = boot[name]
            row[name] = point[name]
            row[f"{name}_mean"] = mean
            row[f"{name}_lo"] = low
            row[f"{name}_hi"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_deltas(
    panel: pd.DataFrame,
    n: int = N_BOOTSTRAP,
    seed: int = 42,
) -> pd.DataFrame:
    """Paired image-bootstrap CIs for the three planned condition contrasts."""

    rows: list[dict[str, Any]] = []
    for (model, culture), group in panel.groupby(["model_name", "culture"], sort=True):
        truth_by_image = group.drop_duplicates("image_id").set_index("image_id")[
            "ground_truth_sentiment"
        ]
        predictions = group.pivot(
            index="image_id", columns="condition", values="predicted_sentiment"
        )
        for family, (condition_a, condition_b) in COMPARISON_FAMILIES.items():
            if condition_a not in predictions or condition_b not in predictions:
                continue
            pair = predictions[[condition_a, condition_b]].dropna().sort_index()
            if pair.empty:
                continue
            truth = truth_by_image.loc[pair.index].to_numpy(dtype=int)
            pred_a = pair[condition_a].to_numpy(dtype=int)
            pred_b = pair[condition_b].to_numpy(dtype=int)
            point_a = _metrics(truth, pred_a)
            point_b = _metrics(truth, pred_b)
            rng = np.random.default_rng(_stable_seed(seed, model, culture, family))
            samples: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
            for _ in range(n):
                indices = rng.integers(0, len(pair), size=len(pair))
                metrics_a = _metrics(truth[indices], pred_a[indices])
                metrics_b = _metrics(truth[indices], pred_b[indices])
                for name in METRIC_NAMES:
                    samples[name].append(metrics_a[name] - metrics_b[name])
            for name in METRIC_NAMES:
                values = np.asarray(samples[name], dtype=float)
                rows.append(
                    {
                        "model_name": model,
                        "culture": culture,
                        "comparison_family": family,
                        "condition_a": condition_a,
                        "condition_b": condition_b,
                        "metric": name,
                        "n_images": len(pair),
                        "estimate_a": point_a[name],
                        "estimate_b": point_b[name],
                        "delta": point_a[name] - point_b[name],
                        "delta_bootstrap_mean": float(np.nanmean(values)),
                        "delta_lo": float(np.nanpercentile(values, 2.5)),
                        "delta_hi": float(np.nanpercentile(values, 97.5)),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_per_fold(
    ann_df: pd.DataFrame,
    gt: dict[str, int],
    folds_dir: Path | str = FOLDS_DIR,
) -> pd.DataFrame:
    """Compute fold metrics from already aggregated/common image rows."""

    fold_files = sorted(Path(folds_dir).glob("fold_*_val.csv"))
    if not fold_files:
        console.print("[yellow]No fold files found — skipping paired fold tests.[/yellow]")
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for fold_path in fold_files:
        fold_n = int(fold_path.stem.split("_")[1])
        fold_ids = set(pd.read_csv(fold_path)["image_id"].astype(str))
        fold = ann_df[ann_df["image_id"].astype(str).isin(fold_ids)]
        for (culture, model, condition), group in fold.groupby(
            ["culture", "model_name", "condition"], sort=True
        ):
            group = group.drop_duplicates("image_id").sort_values("image_id")
            valid_ids = [image for image in group["image_id"].astype(str) if image in gt]
            if not valid_ids:
                continue
            indexed = group.assign(image_id=group["image_id"].astype(str)).set_index("image_id")
            truth = np.asarray([gt[image] for image in valid_ids], dtype=int)
            prediction = indexed.loc[valid_ids, "predicted_sentiment"].to_numpy(dtype=int)
            rows.append(
                {
                    "fold": fold_n,
                    "culture": culture,
                    "model_name": model,
                    "condition": condition,
                    "n_images": len(valid_ids),
                    **_metrics(truth, prediction),
                }
            )
    return pd.DataFrame(rows)


def build_holm_comparisons(
    fold_df: pd.DataFrame,
    metric: str = "f1_macro",
) -> list[dict[str, Any]]:
    """Build only the three preregistered, fold-paired condition contrasts."""

    if fold_df.empty:
        return []
    comparisons: list[dict[str, Any]] = []
    for (model, culture), group in fold_df.groupby(["model_name", "culture"], sort=True):
        for family, (condition_a, condition_b) in COMPARISON_FAMILIES.items():
            a = group[group["condition"] == condition_a][["fold", metric]]
            b = group[group["condition"] == condition_b][["fold", metric]]
            paired = a.merge(b, on="fold", suffixes=("_a", "_b")).sort_values("fold")
            if len(paired) < 2:
                continue
            comparisons.append(
                {
                    "label_a": f"{model}_{culture}_{condition_a}",
                    "label_b": f"{model}_{culture}_{condition_b}",
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "comparison_family": family,
                    "metric": metric,
                    "folds": paired["fold"].astype(int).tolist(),
                    "vals_a": paired[f"{metric}_a"].astype(float).tolist(),
                    "vals_b": paired[f"{metric}_b"].astype(float).tolist(),
                }
            )
    return comparisons


def _corrected_resampled_t(
    values_a: Sequence[float],
    values_b: Sequence[float],
    test_train_ratio: float | None = None,
) -> tuple[float, float]:
    """Nadeau-Bengio corrected paired test over cross-validation folds.

    The variance multiplier is ``1/n_folds + n_test/n_train``.  For disjoint
    k-fold validation partitions the default ratio is ``1/(k - 1)``.  This is
    deliberately more conservative than treating fold scores as independent.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 2:
        return float("nan"), float("nan")
    differences = a - b
    if np.allclose(differences, 0):
        return 0.0, 1.0
    ratio = 1.0 / (len(differences) - 1) if test_train_ratio is None else float(test_train_ratio)
    if ratio < 0:
        raise ValueError("test_train_ratio must be non-negative")
    variance = float(np.var(differences, ddof=1))
    standard_error = float(np.sqrt((1.0 / len(differences) + ratio) * variance))
    if standard_error == 0:
        return float("inf") * np.sign(float(np.mean(differences))), 0.0
    statistic = float(np.mean(differences) / standard_error)
    p_value = float(2.0 * student_t.sf(abs(statistic), df=len(differences) - 1))
    return statistic, p_value


def _holm_adjust(raw_p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Valid step-down Holm adjustment with monotone adjusted p-values."""

    raw = np.asarray(raw_p_values, dtype=float)
    raw = np.where(np.isfinite(raw), raw, 1.0)
    order = np.argsort(raw, kind="stable")
    sorted_raw = raw[order]
    scaled = sorted_raw * (len(raw) - np.arange(len(raw)))
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def holm_bonferroni(
    comparisons: list[dict[str, Any]],
    alpha: float = ALPHA,
    test_train_ratio: float | None = None,
) -> pd.DataFrame:
    """Corrected paired fold tests with Holm applied separately per family."""

    tested: list[dict[str, Any]] = []
    for comparison in comparisons:
        statistic, p_value = _corrected_resampled_t(
            comparison["vals_a"],
            comparison["vals_b"],
            test_train_ratio=test_train_ratio,
        )
        if not np.isfinite(p_value):
            continue
        tested.append(
            {
                **comparison,
                "n_folds": len(comparison["vals_a"]),
                "paired_test": "nadeau_bengio_corrected_resampled_t",
                "test_train_ratio": (
                    float(test_train_ratio)
                    if test_train_ratio is not None
                    else 1.0 / (len(comparison["vals_a"]) - 1)
                ),
                "t_stat": statistic,
                "p_raw": p_value,
                "delta_mean": float(
                    np.mean(np.asarray(comparison["vals_a"]) - np.asarray(comparison["vals_b"]))
                ),
            }
        )
    if not tested:
        return pd.DataFrame()
    frame = pd.DataFrame(tested)
    if "comparison_family" not in frame:
        frame["comparison_family"] = "all"
    adjusted_parts: list[pd.DataFrame] = []
    for _, family in frame.groupby("comparison_family", sort=False):
        family = family.copy()
        family["p_adjusted"] = _holm_adjust(family["p_raw"].to_numpy())
        family["significant"] = family["p_adjusted"] <= alpha
        adjusted_parts.append(family)
    result = pd.concat(adjusted_parts, ignore_index=True)
    return result.sort_values(["comparison_family", "p_raw"], kind="stable").reset_index(drop=True)


def parse_coverage_table(attempts: pd.DataFrame) -> pd.DataFrame:
    """Attempt-level parse coverage, including groups with zero valid parses."""

    if attempts.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in attempts.groupby(["model_name", "culture", "condition"], sort=True):
        valid = int(group["parse_valid"].sum())
        rows.append(
            {
                "model_name": keys[0],
                "culture": keys[1],
                "condition": keys[2],
                "n_attempts": len(group),
                "n_valid": valid,
                "parse_coverage": valid / len(group),
            }
        )
    return pd.DataFrame(rows)


def print_summary_table(results: pd.DataFrame) -> None:
    table = Table(title="Annotation Evaluation — matched images, bootstrap 95% CI")
    columns = [
        "condition",
        "culture",
        "model_name",
        "n_images",
        "parse_coverage",
        "f1_macro",
        "f1_weighted",
        "accuracy",
        "mae",
        "kappa_linear",
        "kappa_quadratic",
    ]
    for column in columns:
        table.add_column(column)
    for _, row in results.iterrows():
        formatted = [
            str(row["condition"]),
            str(row["culture"]),
            str(row["model_name"]),
            str(int(row["n_images"])),
            f"{row['parse_coverage']:.3f}",
        ]
        for metric in METRIC_NAMES:
            formatted.append(
                f"{row[metric]:.3f} [{row[f'{metric}_lo']:.3f}, {row[f'{metric}_hi']:.3f}]"
            )
        table.add_row(*formatted)
    console.print(table)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"

    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return "NA" if not np.isfinite(value) else f"{float(value):.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(render(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def write_evaluation_report(
    path: Path,
    *,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    content_agreement: pd.DataFrame,
    coverage: pd.DataFrame,
    holm: pd.DataFrame,
    caption_embedding_model: str,
    n_bootstrap: int,
    seed: int,
) -> None:
    """Write a human-readable index over the complete machine-readable CSVs."""
    primary_deltas = cast(
        "pd.DataFrame",
        (
            deltas[
                (deltas.get("comparison_family") == "wvs_vs_base")
                & (deltas.get("metric") == "f1_macro")
            ]
            if not deltas.empty
            else deltas
        ),
    )
    sections = [
        "# Matched Annotation Evaluation Report",
        "",
        (
            f"All conditions were aggregated over repeated passes, restricted to their "
            f"common successfully parsed image set, and evaluated with {n_bootstrap:,} "
            f"paired image-bootstrap samples (seed {seed}). Caption cosine uses "
            f"`{caption_embedding_model}` semantic embeddings."
        ),
        "",
        "## Condition metrics",
        "",
        _markdown_table(
            metrics,
            [
                "model_name",
                "culture",
                "condition",
                "n_images",
                "parse_coverage",
                "f1_macro",
                "f1_macro_lo",
                "f1_macro_hi",
                "f1_weighted",
                "accuracy",
                "mae",
                "kappa_linear",
                "kappa_quadratic",
            ],
        ),
        "",
        "## Primary paired WVS vs base macro-F1 delta",
        "",
        _markdown_table(
            primary_deltas,
            [
                "model_name",
                "culture",
                "n_images",
                "estimate_a",
                "estimate_b",
                "delta",
                "delta_lo",
                "delta_hi",
            ],
        ),
        "",
        "## Caption, tag, and sentiment agreement",
        "",
        _markdown_table(
            content_agreement,
            [
                "model_name",
                "culture",
                "comparison_family",
                "n_images",
                "caption_cosine",
                "caption_cosine_lo",
                "caption_cosine_hi",
                "tag_jaccard",
                "tag_jaccard_lo",
                "tag_jaccard_hi",
                "sentiment_agreement",
                "sentiment_agreement_lo",
                "sentiment_agreement_hi",
            ],
        ),
        "",
        "## Parse coverage",
        "",
        _markdown_table(
            coverage,
            [
                "model_name",
                "culture",
                "condition",
                "n_attempts",
                "n_valid",
                "parse_coverage",
            ],
        ),
        "",
        "## Corrected paired fold tests with family-wise Holm adjustment",
        "",
        _markdown_table(
            holm,
            [
                "comparison_family",
                "label_a",
                "label_b",
                "n_folds",
                "delta_mean",
                "t_stat",
                "p_raw",
                "p_adjusted",
                "significant",
            ],
        )
        if not holm.empty
        else "_No fold tests were available._",
        "",
        "Complete bootstrap distributions and per-fold values are in the adjacent CSV files.",
        "",
    ]
    path.write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate matched base/WVS annotations against sigma3-P5."
    )
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--metric", choices=METRIC_NAMES, default="f1_macro")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test-train-ratio",
        type=float,
        help=(
            "n_test/n_train correction for paired fold tests; defaults to "
            "1/(n_folds-1) for k-fold validation"
        ),
    )
    parser.add_argument("--annotations-dir", type=Path, default=ANNOTATIONS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--caption-embedding-model",
        default=os.getenv("CAPTION_EMBEDDING_MODEL", DEFAULT_CAPTION_EMBEDDING_MODEL),
    )
    parser.add_argument("--embedding-device")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=int(os.getenv("ANNOTATION_EXPECTED_RUNS", "5")),
    )
    parser.add_argument(
        "--expected-cultures",
        nargs="+",
        default=os.getenv("CULTURES", " ".join(STUDY_CULTURES)).split(),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=os.getenv("MODELS", " ".join(SHARED_MODELS)).split(),
    )
    args = parser.parse_args()
    if args.n_bootstrap < 1:
        parser.error("--n-bootstrap must be >= 1")
    if args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be >= 1")
    if args.expected_runs < 1:
        parser.error("--expected-runs must be >= 1")
    if args.test_train_ratio is not None and args.test_train_ratio < 0:
        parser.error("--test-train-ratio must be non-negative")

    requested = [normalize_condition(value) for value in args.conditions]
    console.rule("[bold blue]CultureVLM — Matched Annotation Evaluation[/bold blue]")
    gt = load_ground_truth()
    attempts = load_annotations(
        requested, annotations_dir=args.annotations_dir, include_failures=True
    )
    if not attempts.empty:
        attempts = attempts[attempts["model_name"].isin(args.models)].copy()
    if attempts.empty:
        console.print("[red]No annotation attempts found. Run Stage 3 first.[/red]")
        raise SystemExit(1)
    missing_models = sorted(set(args.models) - set(attempts["model_name"].astype(str)))
    if missing_models:
        console.print(
            "[red]Missing annotation attempts for expected models: "
            + ", ".join(missing_models)
            + "[/red]"
        )
        raise SystemExit(1)
    attempts = attempts[attempts["image_id"].isin(gt)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    incomplete = incomplete_pass_groups(attempts, args.expected_runs)
    if not incomplete.empty:
        incomplete.to_csv(args.output_dir / "annotation_incomplete_passes.csv", index=False)
        console.print(
            f"[red]{len(incomplete):,} image/condition groups do not contain exactly "
            f"passes 1..{args.expected_runs}; refusing an unmatched evaluation.[/red]"
        )
        raise SystemExit(1)
    panel_violations = attempted_panel_violations(
        attempts, requested, expected_cultures=args.expected_cultures
    )
    if not panel_violations.empty:
        panel_violations.to_csv(
            args.output_dir / "annotation_attempted_panel_violations.csv", index=False
        )
        console.print(
            f"[red]{len(panel_violations):,} model/culture/condition arms did not "
            "attempt an identical image set; refusing evaluation.[/red]"
        )
        raise SystemExit(1)
    try:
        assert_matched_generation_seeds(attempts)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc
    aggregated = aggregate_repetitions(attempts, gt)
    panel = build_common_condition_panel(aggregated, requested)
    if panel.empty:
        console.print(
            "[red]No common valid image set across the requested conditions. "
            "Check trained adapters and parse coverage.[/red]"
        )
        raise SystemExit(1)

    aggregated.to_csv(args.output_dir / "annotation_aggregated_per_image.csv", index=False)
    panel.to_csv(args.output_dir / "annotation_common_panel.csv", index=False)
    coverage = parse_coverage_table(attempts)
    coverage.to_csv(args.output_dir / "annotation_parse_coverage.csv", index=False)

    metrics = evaluate_metrics(panel, n_bootstrap=args.n_bootstrap, seed=args.seed)
    metrics.to_csv(args.output_dir / "annotation_metrics_bootstrap.csv", index=False)
    print_summary_table(metrics)

    deltas = paired_bootstrap_deltas(panel, n=args.n_bootstrap, seed=args.seed)
    deltas.to_csv(args.output_dir / "annotation_paired_bootstrap_deltas.csv", index=False)

    caption_embeddings = encode_caption_embeddings(
        panel,
        model_name=args.caption_embedding_model,
        cache_dir=args.output_dir / "embedding_cache",
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
    )
    content_agreement = compute_condition_pair_agreement(
        panel,
        caption_embeddings=caption_embeddings,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    content_agreement.to_csv(args.output_dir / "annotation_condition_agreement.csv", index=False)

    fold_metrics = evaluate_per_fold(panel, gt)
    holm = pd.DataFrame()
    if not fold_metrics.empty:
        fold_metrics.to_csv(args.output_dir / "annotation_metrics_per_fold.csv", index=False)
        comparisons = build_holm_comparisons(fold_metrics, metric=args.metric)
        holm = holm_bonferroni(comparisons, test_train_ratio=args.test_train_ratio)
        if not holm.empty:
            holm.to_csv(args.output_dir / f"holm_bonferroni_{args.metric}.csv", index=False)

    write_evaluation_report(
        args.output_dir / "annotation_report.md",
        metrics=metrics,
        deltas=deltas,
        content_agreement=content_agreement,
        coverage=coverage,
        holm=holm,
        caption_embedding_model=args.caption_embedding_model,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    (args.output_dir / "evaluation_provenance.json").write_text(
        json.dumps(
            {
                "conditions": requested,
                "caption_embedding_model": args.caption_embedding_model,
                "n_bootstrap": args.n_bootstrap,
                "expected_runs": args.expected_runs,
                "expected_cultures": args.expected_cultures,
                "models": args.models,
                "seed": args.seed,
                "fold_metric": args.metric,
                "test_train_ratio": args.test_train_ratio,
                "paired_fold_test": "nadeau_bengio_corrected_resampled_t",
                "common_panel_rows": len(panel),
                "common_panel_unique_images": int(panel["image_id"].nunique()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    console.print(f"[bold green]Evaluation complete:[/bold green] {args.output_dir}")


if __name__ == "__main__":
    main()
