"""
Batch annotation orchestrator.

Runs trained (culture, condition, model_name) adapters and the raw
inference-only base MLLM reference against the full σ₃P₅ image set.
Each image can be annotated across multiple independent passes so answer
variance can be analyzed.

Usage:
    uv run python src/annotation/pipeline.py \
        --culture arabic \
        --model-name qwen_vl \
        --condition cultural

    uv run python src/annotation/pipeline.py \
        --culture arabic \
        --model-name qwen_vl \
        --condition cultural \
        --limit 10 \
        --n-runs 5     # smoke test, repeated annotations

    # Raw base-model inference, no LoRA adapter loaded
    uv run python src/annotation/pipeline.py \
        --culture inference_only \
        --model-name qwen_vl \
        --condition inference_only

    # All cultures × all models × trained conditions, plus raw base-model inference
    uv run python src/annotation/pipeline.py --all
"""

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.annotation.config import (
    ANNOTATION_SYSTEM_PROMPT,
    SENTIMENT_INT_TO_LABEL,
    AnnotatorSettings,
)
from src.annotation.graph import build_annotation_graph
from src.annotation.nodes.image_loader import ImageCache

load_dotenv()
console = Console()

AGREEMENT_CSV = os.getenv(
    "AGREEMENT_CSV",
    "../multimodal-LLMs-see-sentiment/data/agreement_p5_sigma3.csv",
)
OUTPUT_DIR = Path("outputs/annotations")
CHECKPOINTS_DIR = Path("checkpoints")
IMAGES_DIR = os.getenv("PERCEPTSENT_IMAGES_DIR", "../perceptsent/images")

CULTURES = [
    "arabic", "bengali", "chinese", "english", "german",
    "korean", "portuguese", "spanish", "turkish",
]
INFERENCE_ONLY_CULTURE = "inference_only"
CONDITIONS = ["cultural", "baseline", "inference_only"]
MODEL_NAMES = [
    "qwen3_5_2b", "phi4", "gemma4_e2b",    # original Mac/MLX models
    "gemma4_e4b", "gemma4_31b",            # Gemma-4 (HF backend)
    "qwen3_vl_8b", "qwen3_27b",             # Qwen3-VL / Qwen3 (HF backend)
]


def load_image_list(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(AGREEMENT_CSV, index_col=0)
    df = df.rename(columns={"id": "image_id"})
    images_path = Path(IMAGES_DIR)
    df["image_path"] = df["image_id"].apply(lambda x: str(images_path / f"{x}.jpg"))
    df = df[df["image_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    if limit:
        df = df.head(limit)
    return df


def load_existing_annotation_ids(out_path: Path) -> set[str]:
    """Resume support: skip already-completed run/image annotations."""
    if not out_path.exists():
        return set()
    ids = set()
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                annotation_id = rec.get("annotation_id")
                if annotation_id:
                    ids.add(str(annotation_id))
    return ids


def build_run_id(model_name: str, culture: str, run_index: int) -> str:
    """Return the independent annotation-pass ID for this culture/model run."""
    return f"{model_name}_{culture}_r{run_index:02d}"


def build_annotation_id(
    model_name: str,
    culture: str,
    condition: str,
    image_id: str,
    run_index: int,
) -> str:
    """Mirror mllm-persona-evaluation's p<run>_img_<image>_<condition> IDs."""
    run_id = build_run_id(model_name, culture, run_index)
    return f"p{run_id}_img_{image_id}_{condition}"


def build_annotation_record(
    state: dict,
    model_name: str,
    culture: str,
    condition: str,
) -> dict:
    """Flatten a completed LangGraph state into a serialisable annotation record.

    Args:
        state (dict): Final graph state after all nodes have run.
        model_name (str): Model identifier (e.g., "qwen3_5_2b").
        culture (str): Culture name (e.g., "arabic").
        condition (str): Annotation condition ("cultural", "baseline", "inference_only").

    Returns:
        dict: Flat record suitable for JSONL serialisation.
    """
    parsed = state.get("parsed") or {}
    sentiment_int = parsed.get("sentiment", -1)
    run_index = int(state["run_index"])
    annotation_run_id = build_run_id(model_name, culture, run_index)
    return {
        "annotation_id": build_annotation_id(
            model_name=model_name,
            culture=culture,
            condition=condition,
            image_id=state["image_id"],
            run_index=run_index,
        ),
        "annotation_run_id": annotation_run_id,
        "run_id": state["run_id"],
        "run_index": run_index,
        "n_runs": int(state["n_runs"]),
        "image_id": state["image_id"],
        "culture": culture,
        "model_name": model_name,
        "condition": condition,
        "ground_truth_sentiment": state["ground_truth_sentiment"],
        "predicted_sentiment": sentiment_int,
        "predicted_sentiment_label": SENTIMENT_INT_TO_LABEL.get(sentiment_int, "unknown"),
        "predicted_perceptions": parsed.get("tags", []),
        "caption": parsed.get("caption", ""),
        "justification": parsed.get("justification", ""),
        "parse_retries": state.get("parse_retries", 0),
        "error": state.get("error"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


async def run_pipeline(
    culture: str,
    condition: str,
    model_name: str,
    df: pd.DataFrame,
    settings: AnnotatorSettings,
    semaphore: asyncio.Semaphore,
    out_path: Path,
    failures_path: Path,
    n_runs: int,
) -> dict:
    graph = build_annotation_graph(settings)

    image_ids = df["image_id"].tolist()
    cache = ImageCache(image_ids, settings.images_dir)
    console.print(
        f"  [green]{culture}/{condition}/{model_name}[/green]: {len(cache)} images cached"
    )

    existing = load_existing_annotation_ids(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    success, failed, skipped = 0, 0, 0

    with open(out_path, "a") as out_f, open(failures_path, "a") as fail_f:
        for run_index in range(1, n_runs + 1):
            run_id = f"r{run_index:02d}"
            console.print(
                f"  [cyan]{culture}/{condition}/{model_name}[/cyan]: run {run_index}/{n_runs}"
            )

            for _, row in df.iterrows():
                image_id = str(row["image_id"])
                annotation_id = build_annotation_id(
                    model_name=model_name,
                    culture=culture,
                    condition=condition,
                    image_id=image_id,
                    run_index=run_index,
                )
                if annotation_id in existing:
                    skipped += 1
                    continue

                initial_state = {
                    "image_id": image_id,
                    "image_path": str(row["image_path"]),
                    "image_b64": cache.get(image_id),
                    "culture": culture,
                    "condition": condition,
                    "model_name": model_name,
                    "run_id": run_id,
                    "run_index": run_index,
                    "n_runs": n_runs,
                    "system_prompt": ANNOTATION_SYSTEM_PROMPT,
                    "raw_output": None,
                    "parsed": None,
                    "parse_retries": 0,
                    "error": None,
                    "ground_truth_sentiment": int(row["sentiment"]),
                }

                async with semaphore:
                    final_state = await graph.ainvoke(initial_state)

                record = build_annotation_record(
                    final_state,
                    model_name,
                    culture,
                    condition,
                )

                if final_state.get("error") or final_state.get("parsed") is None:
                    fail_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    failed += 1
                else:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    existing.add(annotation_id)
                    success += 1

    return {
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "culture": culture,
        "condition": condition,
        "model": model_name,
        "n_runs": n_runs,
    }


def run_single(
    culture: str,
    model_name: str,
    condition: str = "cultural",
    limit: int | None = None,
    n_runs: int = 5,
) -> None:
    if condition == "inference_only" and culture != INFERENCE_ONLY_CULTURE:
        console.print(
            "[yellow]Inference-only runs do not use culture-specific weights; "
            f"recording under culture='{INFERENCE_ONLY_CULTURE}'.[/yellow]"
        )
        culture = INFERENCE_ONLY_CULTURE

    settings = AnnotatorSettings()
    df = load_image_list(limit=limit)
    console.print(
        f"Annotating {len(df)} images × {n_runs} runs: "
        f"[bold]{culture}[/bold] / [bold]{condition}[/bold] / [bold]{model_name}[/bold]"
    )

    out_dir = OUTPUT_DIR / model_name / culture
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "annotations.jsonl"
    failures_path = out_dir / "annotation_failures.jsonl"

    # Caps concurrent graph.ainvoke calls to avoid OOM and API rate limits.
    semaphore = asyncio.Semaphore(settings.max_concurrent)

    result = asyncio.run(
        run_pipeline(
            culture,
            condition,
            model_name,
            df,
            settings,
            semaphore,
            out_path,
            failures_path,
            n_runs=n_runs,
        )
    )
    console.print(
        f"[green]✓ Done[/green] — success: {result['success']}, "
        f"failed: {result['failed']}, skipped: {result['skipped']} | output: {out_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cultural VLM annotation pipeline.")
    parser.add_argument("--culture", choices=CULTURES + [INFERENCE_ONLY_CULTURE])
    parser.add_argument("--model-name", choices=MODEL_NAMES)
    parser.add_argument(
        "--condition",
        choices=CONDITIONS,
        default="cultural",
        help=(
            "Annotation condition: cultural LoRA adapter, neutral-WVS baseline "
            "LoRA adapter, or inference_only for the raw base MLLM with no adapter."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all trained culture-condition pairs plus one inference-only run per model",
    )
    parser.add_argument("--limit", type=int, help="Cap images for testing")
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Number of independent annotation passes per image (default: 5). "
            "Each pass uses a distinct run ID (r01, r02, ...) and appends to "
            "the same JSONL with resumable annotation IDs."
        ),
    )
    args = parser.parse_args()
    if args.n_runs < 1:
        parser.error("--n-runs must be >= 1")

    console.rule("[bold blue]CultureVLM Annotation Pipeline[/bold blue]")

    if args.all:
        for model_name in MODEL_NAMES:
            for culture in CULTURES:
                for condition in ("cultural", "baseline"):
                    run_single(
                        culture,
                        model_name,
                        condition=condition,
                        limit=args.limit,
                        n_runs=args.n_runs,
                    )
            run_single(
                INFERENCE_ONLY_CULTURE,
                model_name,
                condition="inference_only",
                limit=args.limit,
                n_runs=args.n_runs,
            )
    elif args.culture and args.model_name:
        run_single(
            args.culture,
            args.model_name,
            condition=args.condition,
            limit=args.limit,
            n_runs=args.n_runs,
        )
    else:
        parser.error("Specify --culture and --model-name, or --all")


if __name__ == "__main__":
    main()
