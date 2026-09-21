"""Prompt assembler node."""

from typing import Any

from src.annotation.config import ANNOTATION_SYSTEM_PROMPT, ANNOTATION_USER_PROMPT
from src.annotation.state import CulturalAnnotationState


def assembler_node(state: CulturalAnnotationState) -> CulturalAnnotationState:
    return {**state, "system_prompt": ANNOTATION_SYSTEM_PROMPT}


def build_messages(state: CulturalAnnotationState) -> list[dict[str, Any]]:
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
