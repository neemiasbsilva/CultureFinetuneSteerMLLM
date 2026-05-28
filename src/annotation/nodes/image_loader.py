"""
Image loader node for the LangGraph annotation pipeline.

Reads a JPEG from disk and base64-encodes it for multimodal API calls.
Adapted directly from mllm-persona-evaluation/src/annotator/nodes/image_loader.py.
"""

import base64
from pathlib import Path

from src.annotation.state import CulturalAnnotationState


def image_loader_node(state: CulturalAnnotationState) -> CulturalAnnotationState:
    """Load image from disk and encode as base64."""
    image_path = Path(state["image_path"])

    if not image_path.exists():
        return {**state, "error": f"Image not found: {image_path}"}

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {**state, "image_b64": image_b64, "error": None}


class ImageCache:
    """
    Pre-load all images into memory before a batch run to eliminate N-fold
    disk I/O during annotation — same optimisation as mllm-persona-evaluation.
    """

    def __init__(self, image_ids: list[str], images_dir: str) -> None:
        self._cache: dict[str, str] = {}
        images_path = Path(images_dir)
        missing = 0
        for image_id in image_ids:
            path = images_path / f"{image_id}.jpg"
            if path.exists():
                with open(path, "rb") as f:
                    self._cache[image_id] = base64.b64encode(f.read()).decode("utf-8")
            else:
                missing += 1
        if missing:
            print(f"[ImageCache] Warning: {missing}/{len(image_ids)} images not found.")

    def get(self, image_id: str) -> str | None:
        return self._cache.get(image_id)

    def __len__(self) -> int:
        return len(self._cache)
