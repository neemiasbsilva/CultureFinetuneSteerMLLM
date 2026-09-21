"""Cosine similarity analysis for cultural annotation outputs."""

from typing import cast

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist, squareform


def _require_single_condition(df: pd.DataFrame, operation: str) -> None:
    if "condition" in df.columns and df["condition"].dropna().nunique() > 1:
        raise ValueError(
            f"{operation} received multiple annotation conditions; filter to one "
            "condition or compute one matrix per condition"
        )


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cast("np.ndarray", 1.0 - cdist(a, b, metric="cosine"))


def compute_per_image_similarity(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    group_col: str = "culture",
    context_cols: list[str] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df["_emb_idx"] = np.arange(len(df))

    if context_cols is None:
        context_cols = ["condition"] if "condition" in df.columns else []
    group_keys = [*context_cols, "image_id"]
    rows = []
    for keys, grp in df.groupby(group_keys):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        metadata = dict(zip(group_keys, key_values, strict=True))
        idxs = cast("np.ndarray", grp["_emb_idx"].values)
        if len(idxs) < 2:
            continue
        embs = embeddings[idxs]
        sim_matrix = cosine_similarity_matrix(embs, embs)
        upper = sim_matrix[np.triu_indices(len(idxs), k=1)]
        rows.append(
            {
                **metadata,
                "sim_mean": float(np.mean(upper)),
                "sim_std": float(np.std(upper)),
                "n_annotations": len(idxs),
            }
        )

    return pd.DataFrame(rows)


def compute_within_cross_similarity(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    group_col: str = "culture",
    modality: str = "caption",
) -> pd.DataFrame:
    df = df.copy()
    df["_emb_idx"] = np.arange(len(df))
    context_cols = ["condition"] if "condition" in df.columns else []
    group_keys = [*context_cols, "image_id"]
    rows = []
    for keys, img_grp in df.groupby(group_keys):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        metadata = dict(zip(group_keys, key_values, strict=True))
        idxs = cast("np.ndarray", img_grp["_emb_idx"].values)
        embs = embeddings[idxs]
        img_grp = img_grp.reset_index(drop=True)

        within_sims, cross_sims = [], []
        for i in range(len(img_grp)):
            for j in range(i + 1, len(img_grp)):
                sim = float(cosine_similarity_matrix(embs[i : i + 1], embs[j : j + 1])[0, 0])
                if img_grp.iloc[i][group_col] == img_grp.iloc[j][group_col]:
                    within_sims.append(sim)
                else:
                    cross_sims.append(sim)

        rows.append(
            {
                **metadata,
                "dimension": group_col,
                "modality": modality,
                "within_mean": float(np.mean(within_sims)) if within_sims else np.nan,
                "cross_mean": float(np.mean(cross_sims)) if cross_sims else np.nan,
                "n_within_pairs": len(within_sims),
                "n_cross_pairs": len(cross_sims),
            }
        )

    return pd.DataFrame(rows)


def compute_culture_sim_matrix(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    group_col: str = "culture",
) -> tuple[np.ndarray, list[str]]:
    _require_single_condition(df, "compute_culture_sim_matrix")
    groups = sorted(df[group_col].unique())
    df = df.copy()
    df["_emb_idx"] = np.arange(len(df))

    group_mean_embs = {}
    for grp in groups:
        idxs = df[df[group_col] == grp]["_emb_idx"].values
        group_mean_embs[grp] = embeddings[idxs].mean(axis=0, keepdims=True)

    n = len(groups)
    matrix = np.zeros((n, n))
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            matrix[i, j] = float(
                cosine_similarity_matrix(group_mean_embs[gi], group_mean_embs[gj])[0, 0]
            )

    return matrix, groups


def compute_image_conditioned_culture_sim(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    group_col: str = "culture",
) -> tuple[np.ndarray, list[str]]:
    _require_single_condition(df, "compute_image_conditioned_culture_sim")
    groups = sorted(df[group_col].unique())
    df = df.copy()
    df["_emb_idx"] = np.arange(len(df))

    n = len(groups)
    group_to_idx = {g: i for i, g in enumerate(groups)}

    accum = np.zeros((n, n))
    counts = np.zeros((n, n))

    for _image_id, img_grp in df.groupby("image_id"):
        img_grp = img_grp.reset_index(drop=True)
        for grp in groups:
            grp_rows = img_grp[img_grp[group_col] == grp]
            if len(grp_rows) == 0:
                continue
            gi = group_to_idx[grp]
            grp_embs = embeddings[grp_rows["_emb_idx"].values]
            mean_emb_i = grp_embs.mean(axis=0, keepdims=True)

            for other_grp in groups:
                other_rows = img_grp[img_grp[group_col] == other_grp]
                if len(other_rows) == 0:
                    continue
                gj = group_to_idx[other_grp]
                other_embs = embeddings[other_rows["_emb_idx"].values]
                mean_emb_j = other_embs.mean(axis=0, keepdims=True)
                sim = float(cosine_similarity_matrix(mean_emb_i, mean_emb_j)[0, 0])
                accum[gi, gj] += sim
                counts[gi, gj] += 1

    with np.errstate(invalid="ignore"):
        matrix = np.where(counts > 0, accum / counts, 0.0)

    return matrix, groups


def cluster_order(matrix: np.ndarray) -> np.ndarray:
    dist = 1.0 - matrix
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, None)
    condensed = squareform(dist)
    Z = linkage(condensed, method="ward")
    dn = dendrogram(Z, no_plot=True)
    return np.array(dn["leaves"])
