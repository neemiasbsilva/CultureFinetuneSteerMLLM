"""
Cosine similarity analysis for cultural annotation outputs.

Adapts analyzing-persona-effects-mllm/src/similarity.py:
  - "demographic dimension" → "culture" dimension
  - 24×24 profile matrix → 10×10 (culture × model or culture-only)
  - Adds image-conditioned cross-culture similarity matrix.

Key functions:
    compute_per_image_similarity()         — mean cosine sim per image across culture models
    compute_within_cross_similarity()      — within-culture vs. cross-culture per image
    compute_culture_sim_matrix()           — N×N cross-culture similarity heatmap
    compute_image_conditioned_culture_sim()— controls for visual content
    cluster_order()                        — Ward hierarchical for heatmap ordering
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between two embedding matrices."""
    return 1.0 - cdist(a, b, metric="cosine")


def compute_per_image_similarity(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    group_col: str = "culture",
) -> pd.DataFrame:
    """
    Per-image mean cosine similarity across all culture models.
    Returns DataFrame indexed by image_id with columns:
        caption_sim_mean, caption_sim_std, n_annotations
    """
    df = df.copy()
    df["_emb_idx"] = np.arange(len(df))

    rows = []
    for image_id, grp in df.groupby("image_id"):
        idxs = grp["_emb_idx"].values
        if len(idxs) < 2:
            continue
        embs = embeddings[idxs]
        sim_matrix = cosine_similarity_matrix(embs, embs)
        # Upper triangle (excluding diagonal)
        upper = sim_matrix[np.triu_indices(len(idxs), k=1)]
        rows.append({
            "image_id": image_id,
            "sim_mean": float(np.mean(upper)),
            "sim_std": float(np.std(upper)),
            "n_annotations": len(idxs),
        })

    return pd.DataFrame(rows)


def compute_within_cross_similarity(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    group_col: str = "culture",
    modality: str = "caption",
) -> pd.DataFrame:
    """
    For each image: compute mean cosine similarity among annotations that SHARE
    a culture value (within) vs. those that DIFFER (cross).

    Mirrors mllm-persona-evaluation's within/cross demographic analysis.
    """
    df = df.copy()
    df["_emb_idx"] = np.arange(len(df))
    groups = df[group_col].unique()

    rows = []
    for image_id, img_grp in df.groupby("image_id"):
        idxs = img_grp["_emb_idx"].values
        embs = embeddings[idxs]
        img_grp = img_grp.reset_index(drop=True)

        within_sims, cross_sims = [], []
        for i in range(len(img_grp)):
            for j in range(i + 1, len(img_grp)):
                sim = float(cosine_similarity_matrix(
                    embs[i:i+1], embs[j:j+1]
                )[0, 0])
                if img_grp.iloc[i][group_col] == img_grp.iloc[j][group_col]:
                    within_sims.append(sim)
                else:
                    cross_sims.append(sim)

        rows.append({
            "image_id": image_id,
            "dimension": group_col,
            "modality": modality,
            "within_mean": float(np.mean(within_sims)) if within_sims else np.nan,
            "cross_mean": float(np.mean(cross_sims)) if cross_sims else np.nan,
            "n_within_pairs": len(within_sims),
            "n_cross_pairs": len(cross_sims),
        })

    return pd.DataFrame(rows)


def compute_culture_sim_matrix(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    group_col: str = "culture",
) -> tuple[np.ndarray, list[str]]:
    """
    Compute mean cosine similarity between every pair of culture groups.
    Returns (N×N matrix, list of group labels).

    Mirrors the 24×24 profile similarity matrix from analyzing-persona-effects-mllm.
    """
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
            matrix[i, j] = float(cosine_similarity_matrix(
                group_mean_embs[gi], group_mean_embs[gj]
            )[0, 0])

    return matrix, groups


def compute_image_conditioned_culture_sim(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    group_col: str = "culture",
) -> tuple[np.ndarray, list[str]]:
    """
    Image-conditioned cross-culture similarity matrix.
    For each image, compute per-group mean embeddings, then average the pairwise
    cosine similarities across images. Controls for visual content variation.

    Returns (N×N matrix, list of group labels).
    """
    groups = sorted(df[group_col].unique())
    df = df.copy()
    df["_emb_idx"] = np.arange(len(df))

    n = len(groups)
    group_to_idx = {g: i for i, g in enumerate(groups)}

    accum = np.zeros((n, n))
    counts = np.zeros((n, n))

    for image_id, img_grp in df.groupby("image_id"):
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
    """
    Ward hierarchical clustering on 1-cosine distance for heatmap ordering.
    Returns permutation indices.
    """
    dist = 1.0 - matrix
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, None)
    condensed = squareform(dist)
    Z = linkage(condensed, method="ward")
    dn = dendrogram(Z, no_plot=True)
    return np.array(dn["leaves"])
