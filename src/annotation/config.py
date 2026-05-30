"""Annotation pipeline configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Standard annotation prompt — no persona, no culture re-injection.
# The cultural bias is already in the model weights from fine-tuning.
ANNOTATION_SYSTEM_PROMPT = (
    "You are an expert urban visual sentiment analyst. "
    "Analyze each image and return your assessment as JSON. "
    "Be precise and consistent."
)

ANNOTATION_USER_PROMPT = (
    "Analyze this urban scene and provide your sentiment assessment. "
    "Respond only with this exact JSON:\n"
    '{"sentiment": <0-4>, "caption": "<one sentence>", '
    '"justification": "<2-3 sentences>", "tags": ["<tag1>", ...]}\n'
    "Scale: 0=negative, 1=slightly_negative, 2=neutral, 3=slightly_positive, 4=positive."
)

SENTIMENT_LABEL_MAP = {
    "negative": 0, "very negative": 0,
    "slightly_negative": 1, "slightly negative": 1,
    "neutral": 2,
    "slightly_positive": 3, "slightly positive": 3,
    "positive": 4, "very positive": 4,
}

SENTIMENT_INT_TO_LABEL = {
    0: "negative", 1: "slightly_negative", 2: "neutral",
    3: "slightly_positive", 4: "positive",
}


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

    # MLX-community model IDs for mlx_vlm inference (Apple Silicon only)
    model_id_map: dict = field(default_factory=lambda: {
        "qwen3_5_2b": "mlx-community/Qwen3.5-2B-4bit",
        "gemma4_e2b": "mlx-community/gemma-4-e2b-it-4bit",
    })
    # HF model IDs for models that use the HF backend (CUDA or MPS)
    hf_model_id_map: dict = field(default_factory=lambda: {
        "phi4":         "microsoft/Phi-4-multimodal-instruct",
        "gemma4_e4b":   "google/gemma-4-E4B-it",
        "gemma4_31b":   "google/gemma-4-31B-it",
        "qwen3_vl_8b":  "Qwen/Qwen3-VL-8B-Thinking-FP8",
        "qwen3_vl_30b": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    })

    seed: int = 42
