"""
Load and structure cultural annotation outputs.

Adapts analyzing-persona-effects-mllm/src/data_loading.py:
  - "persona_id" → "culture"
  - "profile"    → "(culture, model_name)"
  - Demographic columns replaced by culture + model_name columns.
"""

import json
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("outputs/annotations")

CULTURES = [
    "arabic", "bengali", "chinese", "english", "german",
    "korean", "portuguese", "spanish", "turkish", "baseline",
]
MODEL_NAMES = ["qwen_vl", "phi4", "gemma4"]

SENTIMENT_INT_TO_LABEL = {
    0: "negative", 1: "slightly_negative", 2: "neutral",
    3: "slightly_positive", 4: "positive",
}


def load_annotations(
    model_names: list[str] | None = None,
    cultures: list[str] | None = None,
    annotations_dir: Path | str = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Load all JSONL annotation files into a single DataFrame.

    Returns columns:
        annotation_id, image_id, culture, model_name, condition,
        annotation_run_id, run_id, run_index, n_runs,
        ground_truth_sentiment, predicted_sentiment, predicted_sentiment_label,
        predicted_perceptions (list), caption, justification,
        parse_retries, error, timestamp_utc,
        ground_truth_label, profile (culture__model_name),
        profile_run (culture__model_name__run_id)
    """
    annotations_dir = Path(annotations_dir)
    model_names = model_names or MODEL_NAMES
    cultures = cultures or CULTURES

    records = []
    for model_name in model_names:
        for culture in cultures:
            jsonl_path = annotations_dir / model_name / culture / "annotations.jsonl"
            if not jsonl_path.exists():
                continue
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        records.append(rec)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Normalise types
    df["predicted_sentiment"] = pd.to_numeric(df["predicted_sentiment"], errors="coerce").fillna(-1).astype(int)
    df["ground_truth_sentiment"] = pd.to_numeric(df["ground_truth_sentiment"], errors="coerce").fillna(-1).astype(int)
    if "run_index" not in df.columns:
        df["run_index"] = 1
    else:
        df["run_index"] = pd.to_numeric(df["run_index"], errors="coerce").fillna(1).astype(int)
    if "n_runs" not in df.columns:
        df["n_runs"] = 1
    else:
        df["n_runs"] = pd.to_numeric(df["n_runs"], errors="coerce").fillna(1).astype(int)
    if "run_id" not in df.columns:
        df["run_id"] = df["run_index"].apply(lambda i: f"r{int(i):02d}")
    else:
        df["run_id"] = df["run_id"].fillna(
            df["run_index"].apply(lambda i: f"r{int(i):02d}")
        )
    if "annotation_run_id" not in df.columns:
        df["annotation_run_id"] = (
            df["model_name"] + "_" + df["culture"] + "_" + df["run_id"]
        )
    else:
        df["annotation_run_id"] = df["annotation_run_id"].fillna(
            df["model_name"] + "_" + df["culture"] + "_" + df["run_id"]
        )

    # Derived columns
    df["ground_truth_label"] = df["ground_truth_sentiment"].map(SENTIMENT_INT_TO_LABEL).fillna("unknown")
    df["predicted_sentiment_label"] = df["predicted_sentiment"].map(SENTIMENT_INT_TO_LABEL).fillna("unknown")
    df["profile"] = df["culture"] + "__" + df["model_name"]
    df["profile_run"] = df["profile"] + "__" + df["run_id"]

    # Word counts
    df["caption_len"] = df["caption"].fillna("").apply(lambda x: len(x.split()))
    df["justification_len"] = df["justification"].fillna("").apply(lambda x: len(x.split()))
    df["n_perceptions"] = df["predicted_perceptions"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    return df.reset_index(drop=True)


def load_failures(
    model_names: list[str] | None = None,
    cultures: list[str] | None = None,
    annotations_dir: Path | str = OUTPUT_DIR,
) -> pd.DataFrame:
    """Load annotation failure records for quality analysis."""
    annotations_dir = Path(annotations_dir)
    model_names = model_names or MODEL_NAMES
    cultures = cultures or CULTURES

    records = []
    for model_name in model_names or MODEL_NAMES:
        for culture in cultures or CULTURES:
            jsonl_path = annotations_dir / model_name / culture / "annotation_failures.jsonl"
            if not jsonl_path.exists():
                continue
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

    return pd.DataFrame(records) if records else pd.DataFrame()


def parse_failure_rate(df: pd.DataFrame, failures: pd.DataFrame) -> float:
    total = len(df) + len(failures)
    return len(failures) / total if total > 0 else 0.0
