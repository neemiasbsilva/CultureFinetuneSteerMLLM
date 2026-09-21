"""Batch inference for trained culture VLMs."""

import json
import re
import subprocess
from typing import Any

import pandas as pd
from rich.console import Console

console = Console()

ANNOTATION_SYSTEM_PROMPT = (
    "You are an expert urban visual sentiment analyst. "
    "Analyze each image and return your assessment in JSON format. "
    "Be precise and consistent."
)

USER_PROMPT = (
    "Analyze this urban scene and provide your sentiment assessment. "
    "Respond only with this exact JSON:\n"
    '{"sentiment": <0-4>, "caption": "<one sentence>", '
    '"justification": "<2-3 sentences>", "tags": ["<tag1>", ...]}\n'
    "Scale: 0=negative, 1=slightly negative, 2=neutral, 3=slightly positive, 4=positive."
)


def _parse_sentiment(raw: str) -> int | None:
    try:
        obj = json.loads(raw)
        s = obj.get("sentiment")
        if isinstance(s, int) and 0 <= s <= 4:
            return s
        if isinstance(s, str) and s.isdigit():
            return int(s)
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r'"sentiment"\s*:\s*(\d)', raw)
    if m:
        return int(m.group(1))
    return None


def batch_generate_mlx(
    model_id: str,
    adapter_path: str,
    val_csv: str,
    images_dir: str,
) -> list[dict[str, Any]]:
    val_df = pd.read_csv(val_csv)
    results: list[dict[str, Any]] = []

    for _, row in val_df.iterrows():
        image_id = str(row["image_id"])
        ground_truth = int(row["sentiment"])

        prompt = f"{ANNOTATION_SYSTEM_PROMPT}\n\n{USER_PROMPT}"
        cmd = [
            "mlx_lm.generate",
            "--model",
            model_id,
            "--adapter-path",
            adapter_path,
            "--prompt",
            prompt,
            "--max-tokens",
            "256",
            "--temp",
            "0.0",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
            raw = result.stdout.strip()
            predicted = _parse_sentiment(raw)
        except (subprocess.TimeoutExpired, OSError):
            predicted = None

        results.append(
            {
                "image_id": image_id,
                "ground_truth": ground_truth,
                "predicted": predicted if predicted is not None else 2,
            }
        )

    n_parsed = sum(1 for r in results if r["predicted"] != 2 or r["ground_truth"] == 2)
    console.print(f"  Inference: {len(results)} images, {n_parsed} successfully parsed")
    return results
