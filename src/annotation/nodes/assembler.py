"""
Prompt assembler node.

Unlike mllm-persona-evaluation which injects demographic personas here,
we use a NEUTRAL system prompt for ALL culture models. The cultural bias
is already encoded in the model weights through fine-tuning.
"""

from typing import Any

from src.annotation.config import ANNOTATION_SYSTEM_PROMPT, ANNOTATION_USER_PROMPT
from src.annotation.state import CulturalAnnotationState


def assembler_node(state: CulturalAnnotationState) -> CulturalAnnotationState:
    """Set the system prompt — identical for all culture models (no persona injection)."""
    return {**state, "system_prompt": ANNOTATION_SYSTEM_PROMPT}


def build_messages(state: CulturalAnnotationState) -> list[dict[str, Any]]:
    """
    Construct the multimodal message list for the VLM API call.
    Image is passed as base64 data URL.
    """
    image_b64 = state.get("image_b64", "")
    return [
        {"role": "system", "content": state["system_prompt"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": ANNOTATION_USER_PROMPT},
            ],
        },
    ]
