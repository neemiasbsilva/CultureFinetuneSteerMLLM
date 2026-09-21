"""Annotation pipeline configuration."""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ANNOTATION_SYSTEM_PROMPT = (
    "You are a visual annotation assistant. "
    "Assess only the image shown and return the requested JSON. "
    "Do not assume a viewer identity, location, culture, or demographic context."
)

ANNOTATION_USER_PROMPT = (
    "Analyze this urban scene and provide your sentiment assessment. "
    "Respond only with this exact JSON:\n"
    '{"sentiment": <0-4>, "caption": "<one sentence>", '
    '"justification": "<2-3 sentences>", "tags": ["<tag1>", ...]}\n'
    "Scale: 0=negative, 1=slightly_negative, 2=neutral, 3=slightly_positive, 4=positive."
)

SENTIMENT_LABEL_MAP = {
    "negative": 0,
    "very negative": 0,
    "slightly_negative": 1,
    "slightly negative": 1,
    "neutral": 2,
    "slightly_positive": 3,
    "slightly positive": 3,
    "positive": 4,
    "very positive": 4,
}

SENTIMENT_INT_TO_LABEL = {
    0: "negative",
    1: "slightly_negative",
    2: "neutral",
    3: "slightly_positive",
    4: "positive",
}


def annotation_seed(
    base_seed: int,
    model_name: str,
    image_id: str,
    run_index: int,
) -> int:
    payload = f"{int(base_seed)}\0{model_name}\0{image_id}\0{int(run_index)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


@dataclass
class AnnotatorSettings:
    max_concurrent: int = field(
        default_factory=lambda: int(os.getenv("ANNOTATION_MAX_CONCURRENT", "1"))
    )
    timeout_seconds: int = 360
    max_retries: int = 3
    temperature: float = 0.1

    images_dir: str = field(
        default_factory=lambda: os.getenv("PERCEPTSENT_IMAGES_DIR", "../perceptsent/images")
    )
    output_dir: Path = Path("outputs/annotations")
    checkpoints_dir: Path = Path("checkpoints")

    model_id_map: dict[str, str] = field(
        default_factory=lambda: {
            "qwen3_5_2b": "mlx-community/Qwen3.5-2B-4bit",
        }
    )
    hf_model_id_map: dict[str, str] = field(
        default_factory=lambda: {
            "qwen3_5_2b": "Qwen/Qwen3.5-2B",
            "gemma4_e2b": "google/gemma-4-E2B-it",
            "phi4": "microsoft/Phi-4-multimodal-instruct",
            "gemma4_e4b": "google/gemma-4-E4B-it",
            "gemma4_31b": "google/gemma-4-31B-it",
            "qwen3_vl_2b": "Qwen/Qwen3-VL-2B-Thinking",
            "qwen3_vl_8b": "Qwen/Qwen3-VL-8B-Thinking",
            "qwen3_27b": "Qwen/Qwen3.6-27B",
            "muse_glimmer_30b": "meta-models/Muse-Glimmer-30B",
        }
    )

    seed: int = 42
