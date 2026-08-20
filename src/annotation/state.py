"""CulturalAnnotationState — LangGraph state schema for cultural VLM annotation."""

from typing import Any, TypedDict


class CulturalAnnotationState(TypedDict):
    """State passed between the nodes of the cultural annotation graph.

    Attributes:
        image_b64: Base64 image payload, populated by the ``image_loader`` node.
        culture: ``"arabic"`` | ... | ``"turkish"`` | ``"inference_only"``.
        condition: Canonical registry condition.
        model_name: ``"qwen3_5_2b"`` | ``"phi4"`` | ``"gemma4_e2b"`` | ``"gemma4_e4b"`` | ...
        run_id: ``"r01"`` | ``"r02"`` | ...
        run_index: 1-indexed independent annotation pass.
        n_runs: Requested number of passes.
        seed: Stable model/image/run seed, shared across conditions.
        generation_seed: Actual seed used for the final generation attempt.
        system_prompt: Neutral annotation prompt, with no persona injection.
        checkpoint_identity: Base model or exact adapter artifact identity.
        adapter_backend: ``"mlx"`` | ``"hf"``.
        raw_output: Raw LLM text response.
        parsed: ``{sentiment, caption, justification, tags}``.
        parse_strategy: ``direct_json`` | ``extracted_json`` | ``invalid`` | ...
        ground_truth_sentiment: σ₃P₅ label (0–4).
    """

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
