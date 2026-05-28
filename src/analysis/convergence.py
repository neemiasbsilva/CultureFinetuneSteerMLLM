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

import numpy as np
import pandas as pd

from src.analysis.similarity import compute_per_image_similarity


def jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def compute_sentiment_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-image proportion of (culture_model) pairs sharing the same
    predicted_sentiment label.
    """
    rows = []
    for image_id, grp in df.groupby("image_id"):
        labels = grp["predicted_sentiment"].tolist()
        if len(labels) < 2:
            rows.append({
                "image_id": image_id,
                "sentiment_agreement": 1.0,
                "majority_sentiment": labels[0] if labels else -1,
                "n_models": len(labels),
            })
            continue

        total_pairs = 0
        agree_pairs = 0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                total_pairs += 1
                if labels[i] == labels[j]:
                    agree_pairs += 1

        majority = grp["predicted_sentiment"].mode().iloc[0]
        rows.append({
            "image_id": image_id,
            "sentiment_agreement": agree_pairs / total_pairs,
            "majority_sentiment": int(majority),
            "n_models": len(labels),
        })

    return pd.DataFrame(rows)


def compute_label_jaccard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-image mean pairwise Jaccard similarity of predicted_perceptions tag sets.
    """
    rows = []
    for image_id, grp in df.groupby("image_id"):
        tag_sets = [
            set(str(t).lower() for t in tags) if isinstance(tags, list) else set()
            for tags in grp["predicted_perceptions"]
        ]
        if len(tag_sets) < 2:
            rows.append({"image_id": image_id, "label_jaccard": 1.0})
            continue

        sims = []
        for i in range(len(tag_sets)):
            for j in range(i + 1, len(tag_sets)):
                sims.append(jaccard(tag_sets[i], tag_sets[j]))

        rows.append({"image_id": image_id, "label_jaccard": float(np.mean(sims))})

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
    caption_sim_df = caption_sim_df.rename(columns={"sim_mean": "caption_sim", "sim_std": "caption_sim_std"})

    sent_df = compute_sentiment_agreement(df)
    jaccard_df = compute_label_jaccard(df)

    merged = caption_sim_df.merge(sent_df, on="image_id", how="outer")
    merged = merged.merge(jaccard_df, on="image_id", how="outer")

    if justification_embeddings is not None:
        just_sim_df = compute_per_image_similarity(df, justification_embeddings, group_col="culture")
        just_sim_df = just_sim_df.rename(columns={"sim_mean": "justification_sim"})[["image_id", "justification_sim"]]
        merged = merged.merge(just_sim_df, on="image_id", how="outer")

    return merged.reset_index(drop=True)
