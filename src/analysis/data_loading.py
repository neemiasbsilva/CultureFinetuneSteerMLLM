"""Load and structure cultural annotation outputs."""

import json
from pathlib import Path

import pandas as pd

from src.annotation.conditions import normalize_condition
from src.data import cultures

OUTPUT_DIR = Path("outputs/annotations")

CULTURES = [*cultures.CULTURES, cultures.INFERENCE_ONLY_CULTURE]
MODEL_NAMES = [
    "qwen3_5_2b",
    "phi4",
    "gemma4_e2b",
    "gemma4_e4b",
    "gemma4_31b",
    "qwen3_vl_2b",
    "qwen3_vl_8b",
    "qwen3_27b",
    "muse_glimmer_30b",
]

SENTIMENT_INT_TO_LABEL = {
    0: "negative",
    1: "slightly_negative",
    2: "neutral",
    3: "slightly_positive",
    4: "positive",
}


def load_annotations(
    model_names: list[str] | None = None,
    cultures: list[str] | None = None,
    annotations_dir: Path | str = OUTPUT_DIR,
    conditions: list[str] | None = None,
) -> pd.DataFrame:
    annotations_dir = Path(annotations_dir)
    records = []
    paths = sorted(annotations_dir.rglob("annotations.jsonl"))
    paths += sorted(annotations_dir.rglob("annotations_*.jsonl"))
    for jsonl_path in dict.fromkeys(paths):
        relative = jsonl_path.relative_to(annotations_dir).parts
        defaults = {
            "model_name": relative[0] if len(relative) >= 3 else "",
            "culture": relative[1] if len(relative) >= 3 else "",
        }
        if len(relative) >= 4:
            defaults["condition"] = normalize_condition(relative[2], strict=False)
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    for key, value in defaults.items():
                        rec.setdefault(key, value)
                    records.append(rec)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    for column in ("model_name", "culture", "condition"):
        if column not in df:
            df[column] = ""
    df["condition"] = df["condition"].map(lambda value: normalize_condition(value, strict=False))
    if model_names is not None:
        df = df[df["model_name"].isin(model_names)]
    if cultures is not None:
        df = df[df["culture"].isin(cultures)]
    if conditions is not None:
        wanted = {normalize_condition(value, strict=False) for value in conditions}
        df = df[df["condition"].isin(wanted)]
    if df.empty:
        return df.reset_index(drop=True)

    df["predicted_sentiment"] = (
        pd.to_numeric(df["predicted_sentiment"], errors="coerce").fillna(-1).astype(int)
    )
    df["ground_truth_sentiment"] = (
        pd.to_numeric(df["ground_truth_sentiment"], errors="coerce").fillna(-1).astype(int)
    )
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
        df["run_id"] = df["run_id"].fillna(df["run_index"].apply(lambda i: f"r{int(i):02d}"))
    if "annotation_run_id" not in df.columns:
        df["annotation_run_id"] = df["model_name"] + "_" + df["culture"] + "_" + df["run_id"]
    else:
        df["annotation_run_id"] = df["annotation_run_id"].fillna(
            df["model_name"] + "_" + df["culture"] + "_" + df["run_id"]
        )

    df["ground_truth_label"] = (
        df["ground_truth_sentiment"].map(SENTIMENT_INT_TO_LABEL).fillna("unknown")
    )
    df["predicted_sentiment_label"] = (
        df["predicted_sentiment"].map(SENTIMENT_INT_TO_LABEL).fillna("unknown")
    )
    df["legacy_profile"] = df["culture"] + "__" + df["model_name"]
    df["profile"] = df["legacy_profile"] + "__" + df["condition"]
    df["profile_run"] = df["profile"] + "__" + df["run_id"]

    if "caption" not in df:
        df["caption"] = ""
    if "justification" not in df:
        df["justification"] = ""
    if "predicted_perceptions" not in df:
        df["predicted_perceptions"] = [[] for _ in range(len(df))]
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
    conditions: list[str] | None = None,
) -> pd.DataFrame:
    annotations_dir = Path(annotations_dir)
    records = []
    for jsonl_path in sorted(annotations_dir.rglob("*failure*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "condition" in df:
        df["condition"] = df["condition"].map(
            lambda value: normalize_condition(value, strict=False)
        )
    if model_names is not None:
        df = df[df["model_name"].isin(model_names)]
    if cultures is not None:
        df = df[df["culture"].isin(cultures)]
    if conditions is not None and "condition" in df:
        wanted = {normalize_condition(value, strict=False) for value in conditions}
        df = df[df["condition"].isin(wanted)]
    return df.reset_index(drop=True)


def parse_failure_rate(df: pd.DataFrame, failures: pd.DataFrame) -> float:
    total = len(df) + len(failures)
    return len(failures) / total if total > 0 else 0.0
