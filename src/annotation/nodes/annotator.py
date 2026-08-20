"""
Vision annotator node — runs either a fine-tuned cultural/baseline VLM
adapter or the raw base VLM via mlx_vlm (Apple Silicon) or HuggingFace
(CUDA/MPS) and parses the structured JSON output.

Backend selection:
  - mlx_vlm (subprocess): models listed in settings.model_id_map
  - HuggingFace (in-process): models listed in settings.hf_model_id_map

Device for HF inference is auto-detected: cuda > mps > cpu.
"""

import hashlib
import json
import random
import re
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import jsonschema
import numpy as np

from src.annotation.conditions import checkpoint_directory, get_condition_spec, normalize_condition
from src.annotation.config import (
    ANNOTATION_USER_PROMPT,
    SENTIMENT_LABEL_MAP,
    AnnotatorSettings,
)
from src.annotation.state import CulturalAnnotationState

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["sentiment", "caption", "justification", "tags"],
    "properties": {
        "sentiment": {"type": "integer", "minimum": 0, "maximum": 4},
        "caption": {"type": "string", "minLength": 5},
        "justification": {"type": "string", "minLength": 10},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
}

CHECKPOINTS_DIR = Path("checkpoints")
ADAPTER_ARTIFACTS = {
    "mlx": "adapters.safetensors",
    "hf": "adapter_model.safetensors",
}
_GENERATION_SEED_LOCK = threading.Lock()
_ARTIFACT_DIGEST_CACHE: dict[Path, tuple[int, int, str]] = {}


def _file_sha256(path: Path) -> str:
    stat = path.stat()
    cached = _ARTIFACT_DIGEST_CACHE.get(path)
    signature = (stat.st_mtime_ns, stat.st_size)
    if cached and cached[:2] == signature:
        return cached[2]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _ARTIFACT_DIGEST_CACHE[path] = (*signature, value)
    return value


class AdapterCheckpointError(RuntimeError):
    """A trained condition cannot be run with its requested adapter."""


def _mlx_available() -> bool:
    """True on Apple Silicon machines with mlx installed."""
    try:
        import mlx  # noqa: F401

        return True
    except ImportError:
        return False


def _adapter_path(
    culture: str,
    model_name: str,
    condition: str,
    *,
    checkpoints_dir: str | Path | None = None,
    backend: str | None = None,
) -> Path | None:
    """Return a validated adapter directory, never a raw-model fallback.

    ``inference_only`` is the sole condition allowed to return ``None``.  A
    trained condition raises before model generation if its directory is
    missing or contains an artifact for the other inference backend.
    """

    canonical = normalize_condition(condition)
    spec = get_condition_spec(canonical)
    if not spec.trained:
        return None

    root = CHECKPOINTS_DIR if checkpoints_dir is None else Path(checkpoints_dir)
    path = checkpoint_directory(root, culture, model_name, canonical)
    assert path is not None
    if not path.is_dir():
        raise AdapterCheckpointError(f"Missing adapter for trained condition '{canonical}': {path}")

    if backend is not None:
        if backend not in ADAPTER_ARTIFACTS:
            raise ValueError(f"Unknown inference backend: {backend!r}")
        artifact = path / ADAPTER_ARTIFACTS[backend]
        if not artifact.is_file():
            found = [name for name in ADAPTER_ARTIFACTS.values() if (path / name).is_file()]
            detail = f"; found {', '.join(found)}" if found else ""
            raise AdapterCheckpointError(
                f"Adapter '{canonical}' at {path} is incompatible with the "
                f"{backend} backend: expected {artifact.name}{detail}"
            )
    elif not any((path / name).is_file() for name in ADAPTER_ARTIFACTS.values()):
        expected = " or ".join(ADAPTER_ARTIFACTS.values())
        raise AdapterCheckpointError(
            f"Adapter '{canonical}' at {path} has no supported weights; expected {expected}"
        )
    return path


def _normalise_parsed_object(obj: object) -> dict[str, Any]:
    """Normalize recognized label strings without inventing unknown values."""

    if not isinstance(obj, dict):
        raise TypeError("annotation response must be a JSON object")
    result = dict(obj)
    sentiment = result.get("sentiment")
    if isinstance(sentiment, str):
        key = sentiment.lower().strip()
        if key not in SENTIMENT_LABEL_MAP:
            raise ValueError(f"unknown sentiment label: {sentiment!r}")
        result["sentiment"] = SENTIMENT_LABEL_MAP[key]
    jsonschema.validate(result, OUTPUT_SCHEMA)
    return result


def _parse_output_with_strategy(raw: str) -> tuple[dict[str, Any] | None, str]:
    """Parse only schema-valid JSON and report how it was recovered.

    A generation that yields no schema-valid object is a failure: in particular, a
    neutral label/caption is never fabricated from unstructured text.
    """

    if not raw or not raw.strip():
        return None, "empty"

    raw_clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

    try:
        return _normalise_parsed_object(json.loads(raw_clean)), "direct_json"
    except Exception:
        pass

    match = re.search(r"\{.*?\}", raw_clean, re.DOTALL)
    if match:
        try:
            return _normalise_parsed_object(json.loads(match.group())), "extracted_json"
        except Exception:
            pass

    return None, "invalid"


def _parse_output(raw: str) -> dict[str, Any] | None:
    """Backward-compatible parsed-value-only wrapper."""

    parsed, _ = _parse_output_with_strategy(raw)
    return parsed


def _run_mlx_vlm_generate(
    model_id: str,
    adapter_path: Path | None,
    image_path: str,
    system_prompt: str,
    user_prompt: str,
    settings: AnnotatorSettings,
    seed: int | None = None,
) -> str:
    """Call mlx_vlm generate as subprocess and return the generated text."""
    cmd = [
        sys.executable,
        "-m",
        "mlx_vlm",
        "generate",
        "--model",
        model_id,
        "--image",
        image_path,
        "--prompt",
        user_prompt,
        "--system",
        system_prompt,
        "--max-tokens",
        str(512),
        "--temperature",
        str(settings.temperature),
        "--skip-special-tokens",
    ]
    if seed is not None:
        cmd += ["--seed", str(int(seed))]
    if adapter_path is not None:
        cmd += ["--adapter-path", str(adapter_path)]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=settings.timeout_seconds,
        cwd=Path.cwd(),
        check=True,
    )
    stdout = result.stdout

    parts = stdout.split("==========")
    if len(parts) >= 3:
        block = parts[1]
        if "</think>" in block:
            response = block.split("</think>", 1)[1].strip()
        elif "<|im_start|>assistant" in block:
            response = block.split("<|im_start|>assistant", 1)[1].strip()
        else:
            lines = block.strip().split("\n")
            response_lines = []
            skip = True
            for line in lines:
                if skip and (
                    line.startswith("Files:") or line.startswith("Prompt:") or not line.strip()
                ):
                    continue
                skip = False
                response_lines.append(line)
            response = "\n".join(response_lines)
        return response.strip()

    return stdout


def _vlm_model_id(model_name: str, settings: AnnotatorSettings) -> str:
    return settings.model_id_map.get(model_name, f"mlx-community/{model_name}-4bit")


def _hf_model_id(model_name: str, settings: AnnotatorSettings) -> str:
    return settings.hf_model_id_map.get(model_name, f"microsoft/{model_name}")


def _build_messages(
    model_name: str,
    image: Any,
    system_prompt: str,
    user_prompt: str,
) -> list[dict[str, Any]]:
    """Build HF chat messages in the format expected by each model family.

    Qwen3-VL takes a structured content list holding the image dict; Gemma-4 takes an
    inline image placeholder and no system role; Phi-4 and the legacy models take an
    image token inside the content string.

    Args:
        model_name: Registered model name whose family selects the message layout.
        image: PIL.Image to annotate.
        system_prompt: Cultural/baseline system prompt, dropped for families without a
            system role.
        user_prompt: Annotation instruction shown to the model.

    Returns:
        Chat messages ready for the processor's chat template.
    """
    if model_name.startswith("qwen3_vl"):
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
    elif model_name.startswith("gemma4"):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
    else:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<|image_1|>\n{user_prompt}"},
        ]


_HF_MODEL_CACHE: dict[tuple[str, str | None], tuple[Any, Any, str]] = {}


def _load_hf_model(model_id: str, adapter_path: Path | None) -> tuple[Any, Any, str]:
    """Load (and cache) the HF processor/model for this (model_id, adapter_path) pair.

    A pipeline invocation runs thousands of images/runs against the same
    model+adapter — reloading full weights from disk on every single call
    made larger architectures (e.g. gemma4_31b) unable to make meaningful
    progress within a run.

    The base model is built with the same loader as training, so an adapter is always
    evaluated on the base it was fitted to.
    """
    cache_key = (model_id, str(adapter_path) if adapter_path else None)
    cached = _HF_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from src.utils.device import get_device
    from src.utils.model_loading import build_base_model

    device = get_device()

    processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        model_id, trust_remote_code=True
    )
    model = build_base_model(
        model_id,
        auto_class=AutoModelForImageTextToText,
        quantization=None,
        dtype=torch.bfloat16,
        device=device,
    )

    if adapter_path is not None:
        try:
            model = PeftModel.from_pretrained(model, str(adapter_path))
        except Exception as exc:
            raise AdapterCheckpointError(
                f"Adapter at {adapter_path} is incompatible with base model {model_id}: {exc}"
            ) from exc
    model.eval()

    _HF_MODEL_CACHE[cache_key] = (processor, model, device)
    return _HF_MODEL_CACHE[cache_key]


def _run_hf_generate(
    model_name: str,
    model_id: str,
    adapter_path: Path | None,
    image_path: str,
    system_prompt: str,
    user_prompt: str,
    settings: AnnotatorSettings,
    seed: int | None = None,
) -> str:
    """HF inference for multimodal models — supports CUDA, MPS, and CPU.

    torch/transformers sampling draws on process-global RNG state, so the small
    seed-and-generate section is serialized under a lock: otherwise
    ANNOTATION_MAX_CONCURRENT lets matched passes race one another.
    """
    import torch
    from PIL import Image

    processor, model, device = _load_hf_model(model_id, adapter_path)

    image = Image.open(image_path).convert("RGB")
    messages = _build_messages(model_name, image, system_prompt, user_prompt)

    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt").to(device)
    else:
        text = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text, images=[image], return_tensors="pt").to(device)

    with _GENERATION_SEED_LOCK:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=settings.temperature,
                do_sample=settings.temperature > 0,
            )
    gen_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
    return cast(str, processor.batch_decode(gen_ids, skip_special_tokens=True)[0])


def _inference_backend(model_name: str, settings: AnnotatorSettings) -> str:
    """Select the backend once, using the same policy as model generation."""

    if model_name in settings.model_id_map and _mlx_available():
        return "mlx"
    if model_name in settings.hf_model_id_map:
        return "hf"
    raise RuntimeError(
        f"No usable backend for model '{model_name}' (mlx unavailable, no HF id registered)"
    )


def validate_inference_assets(
    culture: str,
    model_name: str,
    condition: str,
    settings: AnnotatorSettings,
) -> tuple[str, Path | None, str]:
    """Preflight backend and adapter, returning a stable checkpoint identity."""

    canonical = normalize_condition(condition)
    backend = _inference_backend(model_name, settings)
    adapter_path = _adapter_path(
        culture,
        model_name,
        canonical,
        checkpoints_dir=settings.checkpoints_dir,
        backend=backend,
    )
    model_id = (
        _vlm_model_id(model_name, settings)
        if backend == "mlx"
        else _hf_model_id(model_name, settings)
    )
    if adapter_path is None:
        identity = f"base::{model_id}"
    else:
        artifact = ADAPTER_ARTIFACTS[backend]
        artifact_path = adapter_path / artifact
        identity = (
            f"adapter::{adapter_path.as_posix()}::{artifact}::sha256:{_file_sha256(artifact_path)}"
        )
    return backend, adapter_path, identity


def make_annotator_node(
    settings: AnnotatorSettings,
) -> Callable[[CulturalAnnotationState], CulturalAnnotationState]:
    """Factory — returns a LangGraph node function bound to settings."""

    def annotator_node(state: CulturalAnnotationState) -> CulturalAnnotationState:
        """Annotate one image, retrying only stochastic generation failures.

        ``AdapterCheckpointError`` is re-raised rather than retried: loading incompatible
        adapter tensors is a configuration failure, not a stochastic generation failure.
        """
        if state.get("error"):
            return state

        culture = state["culture"]
        condition = normalize_condition(state["condition"])
        model_name = state["model_name"]
        image_path = state["image_path"]
        system_prompt = state["system_prompt"]
        seed = int(state.get("seed", settings.seed))
        backend, adapter_path, checkpoint_identity = validate_inference_assets(
            culture, model_name, condition, settings
        )
        use_mlx = backend == "mlx"

        retries = 0
        raw_output = ""
        parse_strategy = "not_attempted"

        while retries <= settings.max_retries:
            attempt_seed = (seed + retries * 1_000_003) % (2**31 - 1)
            try:
                if not use_mlx:
                    raw_output = _run_hf_generate(
                        model_name=model_name,
                        model_id=_hf_model_id(model_name, settings),
                        adapter_path=adapter_path,
                        image_path=image_path,
                        system_prompt=system_prompt,
                        user_prompt=ANNOTATION_USER_PROMPT,
                        settings=settings,
                        seed=attempt_seed,
                    )
                else:
                    raw_output = _run_mlx_vlm_generate(
                        model_id=_vlm_model_id(model_name, settings),
                        adapter_path=adapter_path,
                        image_path=image_path,
                        system_prompt=system_prompt,
                        user_prompt=ANNOTATION_USER_PROMPT,
                        settings=settings,
                        seed=attempt_seed,
                    )
                parsed, parse_strategy = _parse_output_with_strategy(raw_output)
                if parsed:
                    return {
                        **state,
                        "condition": condition,
                        "seed": seed,
                        "generation_seed": attempt_seed,
                        "checkpoint_identity": checkpoint_identity,
                        "checkpoint_path": str(adapter_path) if adapter_path else None,
                        "adapter_backend": backend,
                        "raw_output": raw_output,
                        "parsed": parsed,
                        "parse_strategy": parse_strategy,
                        "parse_retries": retries,
                        "error": None,
                    }
            except AdapterCheckpointError:
                raise
            except subprocess.TimeoutExpired:
                raw_output = f"ERROR: timeout after {settings.timeout_seconds}s"
                parse_strategy = "generation_timeout"
            except subprocess.CalledProcessError as exc:
                if adapter_path is not None:
                    raise AdapterCheckpointError(
                        f"Adapter generation failed for {adapter_path}: {exc.stderr or exc}"
                    ) from exc
                raw_output = f"ERROR: {exc}: {exc.stderr}"
                parse_strategy = "generation_error"
            except Exception as exc:
                raw_output = f"ERROR: {exc}"
                parse_strategy = "generation_error"

            retries += 1

        return {
            **state,
            "condition": condition,
            "seed": seed,
            "generation_seed": (seed + max(0, retries - 1) * 1_000_003) % (2**31 - 1),
            "checkpoint_identity": checkpoint_identity,
            "checkpoint_path": str(adapter_path) if adapter_path else None,
            "adapter_backend": backend,
            "raw_output": raw_output,
            "parsed": None,
            "parse_strategy": parse_strategy,
            "parse_retries": retries,
            "error": f"Parse exhausted after {settings.max_retries} retries",
        }

    return annotator_node
