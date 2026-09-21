"""CulturalAnnotationState — LangGraph state schema for cultural VLM annotation."""

from typing import Any, TypedDict


class CulturalAnnotationState(TypedDict):
    image_id: str
    image_path: str
    image_b64: str | None
    culture: str
    condition: str
    model_name: str
    run_id: str
    run_index: int
    n_runs: int
    seed: int
    generation_seed: int
    system_prompt: str
    checkpoint_identity: str | None
    checkpoint_path: str | None
    adapter_backend: str | None
    raw_output: str | None
    parsed: dict[str, Any] | None
    parse_strategy: str
    parse_retries: int
    error: str | None
    ground_truth_sentiment: int
