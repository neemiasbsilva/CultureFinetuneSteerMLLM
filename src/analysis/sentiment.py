"""
Sentiment accuracy analysis: cultural VLM predictions vs. σ₃P₅ ground truth.

Adapted from analyzing-persona-effects-mllm/src/sentiment.py.
Adds cross-validation with VADER and RoBERTa on justification text to
triangulate the VLM's cultural sentiment signal.

Outputs:
    outputs/analysis/human_annotator_agreement.csv
"""

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SENTIMENT_INT_TO_LABEL = {
    0: "negative",
    1: "slightly_negative",
    2: "neutral",
    3: "slightly_positive",
    4: "positive",
}

VADER_TO_P5 = {
    "pos": 4,
    "neg": 0,
    "neu": 2,
}

ROBERTA_TO_P5 = {
    "positive": 4,
    "negative": 0,
    "neutral": 2,
    "label_0": 0,
    "label_1": 2,
    "label_2": 4,
}


def _vader_score(text: str, analyzer: SentimentIntensityAnalyzer) -> int:
    scores = analyzer.polarity_scores(str(text))
    compound = scores["compound"]
    if compound >= 0.05:
        return 4
    elif compound <= -0.05:
        return 0
    return 2


def _roberta_score(text: str, clf: Any) -> int:
    try:
        result = clf(str(text)[:512], truncation=True)[0]
        label = result["label"].lower()
        for k, v in ROBERTA_TO_P5.items():
            if k in label:
                return v
    except Exception:
        pass
    return 2


def compute_accuracy_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """Classification metrics for one (culture, model) prediction set."""
    valid = [(t, p) for t, p in zip(y_true, y_pred, strict=False) if t >= 0 and p >= 0]
    if not valid:
        return {}
    yt, yp = zip(*valid, strict=True)
    return {
        "accuracy": round(accuracy_score(yt, yp), 4),
        "f1_macro": round(f1_score(yt, yp, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(yt, yp, average="weighted", zero_division=0), 4),
        "kappa_linear": round(cohen_kappa_score(yt, yp, weights="linear"), 4),
        "kappa_quadratic": round(cohen_kappa_score(yt, yp, weights="quadratic"), 4),
        "mae": round(mean_absolute_error(yt, yp), 4),
        "n": len(yt),
    }


def evaluate_all_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate each condition-aware (culture, model_name) prediction set.
    """
    rows = []
    group_columns = ["culture", "model_name"]
    if "condition" in df.columns:
        group_columns.append("condition")
    for keys, grp in df.groupby(group_columns):
        y_true = grp["ground_truth_sentiment"].tolist()
        y_pred = grp["predicted_sentiment"].tolist()
        metrics = compute_accuracy_metrics(y_true, y_pred)
        if len(group_columns) == 3:
            culture, model_name, condition = keys
            metrics.update(
                {
                    "culture": culture,
                    "model_name": model_name,
                    "condition": condition,
                }
            )
        else:
            culture, model_name = keys
            metrics.update({"culture": culture, "model_name": model_name})
        rows.append(metrics)
    return pd.DataFrame(rows)


def add_vader_predictions(df: pd.DataFrame, text_col: str = "justification") -> pd.DataFrame:
    """Append VADER-based sentiment predictions as a new column."""
    analyzer = SentimentIntensityAnalyzer()
    df = df.copy()
    df["vader_sentiment"] = df[text_col].apply(lambda t: _vader_score(t, analyzer))
    return df


def add_roberta_predictions(
    df: pd.DataFrame,
    text_col: str = "justification",
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    device: str = "mps",
) -> pd.DataFrame:
    """Append RoBERTa-based sentiment predictions as a new column."""
    clf = pipeline(
        "text-classification",
        model=model_name,
        device=0 if device == "cuda" else (0 if device == "mps" else -1),
        truncation=True,
        max_length=512,
    )
    df = df.copy()
    df["roberta_sentiment"] = df[text_col].apply(lambda t: _roberta_score(t, clf))
    return df
