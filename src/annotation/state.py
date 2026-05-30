"""CulturalAnnotationState — LangGraph state schema for cultural VLM annotation."""

from typing import Optional, TypedDict


class CulturalAnnotationState(TypedDict):
    image_id: str
    image_path: str
    image_b64: Optional[str]            # populated by image_loader node
    culture: str                        # "arabic" | ... | "turkish" | "inference_only"
    condition: str                      # "cultural" | "baseline" | "inference_only"
    model_name: str                     # "qwen3_5_2b" | "phi4" | "gemma4_e2b" | "gemma4_e4b" | ...
    run_id: str                         # "r01" | "r02" | ...
    run_index: int                      # 1-indexed independent annotation pass
    n_runs: int                         # requested number of passes
    system_prompt: str                  # neutral annotation prompt (NO persona injection)
    raw_output: Optional[str]           # raw LLM text response
    parsed: Optional[dict]              # {sentiment, caption, justification, tags}
    parse_retries: int
    error: Optional[str]
    ground_truth_sentiment: int         # σ₃P₅ label (0–4)
