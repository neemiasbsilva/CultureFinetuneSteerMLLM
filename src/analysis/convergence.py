"""
Per-image convergence metrics across cultural annotation outputs.

Adapted from analyzing-persona-effects-mllm/src/convergence.py.
Computes three complementary dimensions per image:
  1. caption_sim       — mean cosine similarity of captions across culture models
  2. sentiment_agreement — proportion of culture-model pairs with identical prediction
  3. label_jaccard     — mean pairwise Jaccard of predicted_perceptions tag sets

Usage:
    from src.analysis.convergence import compute_convergence
    conv_df = compute_convergence(df, caption_embeddings)
    conv_df.to_csv("outputs/analysis/convergence_all_dimensions.csv", index=False)
"""

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.similarity import compute_per_image_similarity

DEFAULT_CONDITION_PAIRS = {
    "wvs_vs_base": ("wvs_cultural", "inference_only"),
}


def jaccard(set_a: set[str], set_b: set[str]) -> float:
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def compute_sentiment_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-image proportion of (culture_model) pairs sharing the same
    predicted_sentiment label.
    """
    group_keys = (["condition"] if "condition" in df.columns else []) + ["image_id"]
    rows = []
    for keys, grp in df.groupby(group_keys):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        metadata = dict(zip(group_keys, key_values, strict=True))
        labels = grp["predicted_sentiment"].tolist()
        if len(labels) < 2:
            rows.append(
                {
                    **metadata,
                    "sentiment_agreement": 1.0,
                    "majority_sentiment": labels[0] if labels else -1,
                    "n_models": len(labels),
                }
            )
            continue

        total_pairs = 0
        agree_pairs = 0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                total_pairs += 1
                if labels[i] == labels[j]:
                    agree_pairs += 1

        majority = grp["predicted_sentiment"].mode().iloc[0]
        rows.append(
            {
                **metadata,
                "sentiment_agreement": agree_pairs / total_pairs,
                "majority_sentiment": int(majority),
                "n_models": len(labels),
            }
        )

    return pd.DataFrame(rows)


def compute_label_jaccard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-image mean pairwise Jaccard similarity of predicted_perceptions tag sets.
    """
    group_keys = (["condition"] if "condition" in df.columns else []) + ["image_id"]
    rows = []
    for keys, grp in df.groupby(group_keys):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        metadata = dict(zip(group_keys, key_values, strict=True))
        tag_sets = [
            set(str(t).lower() for t in tags) if isinstance(tags, list) else set()
            for tags in grp["predicted_perceptions"]
        ]
        if len(tag_sets) < 2:
            rows.append({**metadata, "label_jaccard": 1.0})
            continue

        sims = []
        for i in range(len(tag_sets)):
            for j in range(i + 1, len(tag_sets)):
                sims.append(jaccard(tag_sets[i], tag_sets[j]))

        rows.append({**metadata, "label_jaccard": float(np.mean(sims))})

    return pd.DataFrame(rows)


def compute_convergence(
    df: pd.DataFrame,
    caption_embeddings: np.ndarray,
    justification_embeddings: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Merge all convergence dimensions into one DataFrame per image.

    Columns:
        image_id, caption_sim, sentiment_agreement, majority_sentiment,
        label_jaccard, n_models [, justification_sim]
    """
    caption_sim_df = compute_per_image_similarity(df, caption_embeddings, group_col="culture")
    caption_sim_df = caption_sim_df.rename(
        columns={"sim_mean": "caption_sim", "sim_std": "caption_sim_std"}
    )

    sent_df = compute_sentiment_agreement(df)
    jaccard_df = compute_label_jaccard(df)

    merge_keys = ["image_id"]
    if "condition" in caption_sim_df.columns or "condition" in sent_df.columns:
        merge_keys.insert(0, "condition")
    merged = caption_sim_df.merge(sent_df, on=merge_keys, how="outer")
    merged = merged.merge(jaccard_df, on=merge_keys, how="outer")

    if justification_embeddings is not None:
        just_sim_df = compute_per_image_similarity(
            df, justification_embeddings, group_col="culture"
        )
        just_sim_df = just_sim_df.rename(columns={"sim_mean": "justification_sim"})
        keep = [*merge_keys, "justification_sim"]
        just_sim_df = just_sim_df[keep]
        merged = merged.merge(just_sim_df, on=merge_keys, how="outer")

    return merged.reset_index(drop=True)


def compute_condition_pair_agreement(
    df: pd.DataFrame,
    caption_embeddings: np.ndarray | None = None,
    condition_pairs: dict[str, tuple[str, str]] | None = None,
    *,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare matched per-image outputs across experimental conditions.

    ``df`` should contain one aggregated row per model/culture/condition/image.
    ``caption_embeddings`` must be row-aligned semantic embeddings.  Requiring
    them avoids silently substituting lexical TF-IDF similarity for the
    preregistered caption-cosine metric.
    """

    if caption_embeddings is None:
        raise ValueError("caption_embeddings are required for semantic caption cosine similarity")
    condition_pairs = condition_pairs or DEFAULT_CONDITION_PAIRS
    work = df.reset_index(drop=True).copy()
    if len(caption_embeddings) != len(work):
        raise ValueError("caption_embeddings must contain exactly one row per annotation row")
    work["_embedding_idx"] = np.arange(len(work))
    rows: list[dict[str, Any]] = []
    for (model_name, culture), group in work.groupby(["model_name", "culture"]):
        for family, (condition_a, condition_b) in condition_pairs.items():
            columns = [
                "image_id",
                "predicted_sentiment",
                "predicted_perceptions",
                "caption",
                "_embedding_idx",
            ]
            a = group[group["condition"] == condition_a][columns]
            b = group[group["condition"] == condition_b][columns]
            paired = a.merge(b, on="image_id", suffixes=("_a", "_b"))
            if paired.empty:
                continue
            tag_scores = [
                jaccard(
                    {str(tag).lower() for tag in (tags_a if isinstance(tags_a, list) else [])},
                    {str(tag).lower() for tag in (tags_b if isinstance(tags_b, list) else [])},
                )
                for tags_a, tags_b in zip(
                    paired["predicted_perceptions_a"],
                    paired["predicted_perceptions_b"],
                    strict=True,
                )
            ]
            idx_a = paired["_embedding_idx_a"].to_numpy(dtype=int)
            idx_b = paired["_embedding_idx_b"].to_numpy(dtype=int)
            emb_a = caption_embeddings[idx_a]
            emb_b = caption_embeddings[idx_b]
            denominator = np.linalg.norm(emb_a, axis=1) * np.linalg.norm(emb_b, axis=1)
            cosine = np.divide(
                np.sum(emb_a * emb_b, axis=1),
                denominator,
                out=np.zeros(len(paired), dtype=float),
                where=denominator > 0,
            )
            scores: dict[str, Any] = {
                "caption_cosine": np.asarray(cosine, dtype=float),
                "tag_jaccard": np.asarray(tag_scores, dtype=float),
                "sentiment_agreement": np.asarray(
                    paired["predicted_sentiment_a"] == paired["predicted_sentiment_b"],
                    dtype=float,
                ),
            }
            row = {
                "model_name": model_name,
                "culture": culture,
                "comparison_family": family,
                "condition_a": condition_a,
                "condition_b": condition_b,
                "n_images": len(paired),
                **{name: float(np.mean(values)) for name, values in scores.items()},
            }
            if n_bootstrap > 0:
                payload = f"{seed}\0{model_name}\0{culture}\0{family}".encode()
                group_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
                rng = np.random.default_rng(group_seed)
                samples: dict[str, list[float]] = {name: [] for name in scores}
                for _ in range(n_bootstrap):
                    indices = rng.integers(0, len(paired), size=len(paired))
                    for name, values in scores.items():
                        samples[name].append(float(np.mean(values[indices])))
                for name, values in samples.items():
                    row[f"{name}_lo"] = float(np.percentile(values, 2.5))
                    row[f"{name}_hi"] = float(np.percentile(values, 97.5))
            rows.append(row)
    return pd.DataFrame(rows)
