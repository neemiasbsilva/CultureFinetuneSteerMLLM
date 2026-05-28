"""
Vision annotator node — runs either a fine-tuned cultural/baseline VLM
adapter or the raw base VLM via mlx_vlm (Apple Silicon) or HuggingFace
(CUDA/MPS) and parses the structured JSON output.

Backend selection:
  - mlx_vlm (subprocess): models listed in settings.model_id_map
  - HuggingFace (in-process): models listed in settings.hf_model_id_map

Device for HF inference is auto-detected: cuda > mps > cpu.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import jsonschema

from src.annotation.config import (
    ANNOTATION_USER_PROMPT,
    SENTIMENT_INT_TO_LABEL,
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


def _adapter_path(culture: str, model_name: str, condition: str) -> Path | None:
    """Return adapter directory for this run, or None for raw base-model inference."""
    if condition == "inference_only":
        return None
    p = CHECKPOINTS_DIR / culture / model_name / condition
    if (p / "adapters.safetensors").exists() or (p / "adapter_model.safetensors").exists():
        return p
    return None


def _parse_output(raw: str) -> dict | None:
    """Three-tier JSON extraction."""
    raw_clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

    # Tier 1: direct parse
    try:
        obj = json.loads(raw_clean)
        jsonschema.validate(obj, OUTPUT_SCHEMA)
        return obj
    except Exception:
        pass

    # Tier 2: extract first {...} block
    match = re.search(r"\{.*?\}", raw_clean, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj.get("sentiment"), str):
                obj["sentiment"] = SENTIMENT_LABEL_MAP.get(
                    obj["sentiment"].lower().strip(), 2
                )
            jsonschema.validate(obj, OUTPUT_SCHEMA)
            return obj
        except Exception:
            pass

    # Tier 3: keyword heuristic
    sentiment_int = 2
    for label, val in SENTIMENT_LABEL_MAP.items():
        if label in raw.lower():
            sentiment_int = val
            break
    return {
        "sentiment": sentiment_int,
        "caption": raw[:120].strip(),
        "justification": raw[:300].strip(),
        "tags": [],
    }


def _run_mlx_vlm_generate(
    model_id: str,
    adapter_path: Path | None,
    image_path: str,
    system_prompt: str,
    user_prompt: str,
    settings: AnnotatorSettings,
) -> str:
    """Call mlx_vlm generate as subprocess and return the generated text."""
    cmd = [
        sys.executable, "-m", "mlx_vlm", "generate",
        "--model", model_id,
        "--image", image_path,
        "--prompt", user_prompt,
        "--system", system_prompt,
        "--max-tokens", str(512),
        "--temperature", str(settings.temperature),
        "--skip-special-tokens",
    ]
    if adapter_path is not None:
        cmd += ["--adapter-path", str(adapter_path)]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=settings.timeout_seconds,
        cwd=Path.cwd(),
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
                if skip and (line.startswith("Files:") or line.startswith("Prompt:") or not line.strip()):
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
    image,  # PIL.Image
    system_prompt: str,
    user_prompt: str,
) -> list:
    """Build HF chat messages in the format expected by each model family."""
    if model_name.startswith("qwen3_vl"):
        # Qwen3-VL: structured content list with image dict
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
        # Gemma-4: inline image placeholder, no system role
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
        # Phi-4 (and legacy models): image token in content string
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<|image_1|>\n{user_prompt}"},
        ]


def _run_hf_generate(
    model_name: str,
    model_id: str,
    adapter_path: Path | None,
    image_path: str,
    system_prompt: str,
    user_prompt: str,
    settings: AnnotatorSettings,
) -> str:
    """HF inference for multimodal models — supports CUDA, MPS, and CPU."""
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    from src.utils.device import get_device
    device = get_device()

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)

    if adapter_path is not None and (adapter_path / "adapter_model.safetensors").exists():
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    messages = _build_messages(model_name, image, system_prompt, user_prompt)

    # Use apply_chat_template if the processor supports it (most modern VLMs do)
    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text, images=[image], return_tensors="pt").to(device)
    else:
        text = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text, images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=settings.temperature,
            do_sample=settings.temperature > 0,
        )
    gen_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0]


def make_annotator_node(settings: AnnotatorSettings):
    """Factory — returns a LangGraph node function bound to settings."""

    def annotator_node(state: CulturalAnnotationState) -> CulturalAnnotationState:
        if state.get("error"):
            return state

        culture = state["culture"]
        condition = state["condition"]
        model_name = state["model_name"]
        image_path = state["image_path"]
        system_prompt = state["system_prompt"]
        adapter_path = _adapter_path(culture, model_name, condition)

        retries = 0
        raw_output = ""

        while retries <= settings.max_retries:
            try:
                if model_name in settings.hf_model_id_map:
                    raw_output = _run_hf_generate(
                        model_name=model_name,
                        model_id=_hf_model_id(model_name, settings),
                        adapter_path=adapter_path,
                        image_path=image_path,
                        system_prompt=system_prompt,
                        user_prompt=ANNOTATION_USER_PROMPT,
                        settings=settings,
                    )
                else:
                    raw_output = _run_mlx_vlm_generate(
                        model_id=_vlm_model_id(model_name, settings),
                        adapter_path=adapter_path,
                        image_path=image_path,
                        system_prompt=system_prompt,
                        user_prompt=ANNOTATION_USER_PROMPT,
                        settings=settings,
                    )
                parsed = _parse_output(raw_output)
                if parsed:
                    return {
                        **state,
                        "raw_output": raw_output,
                        "parsed": parsed,
                        "parse_retries": retries,
                        "error": None,
                    }
            except subprocess.TimeoutExpired:
                raw_output = f"ERROR: timeout after {settings.timeout_seconds}s"
            except Exception as exc:
                raw_output = f"ERROR: {exc}"

            retries += 1

        return {
            **state,
            "raw_output": raw_output,
            "parsed": None,
            "parse_retries": retries,
            "error": f"Parse exhausted after {settings.max_retries} retries",
        }

    return annotator_node
