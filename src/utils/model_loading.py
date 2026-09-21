"""Base-model loading — one place that decides how weights reach the GPU."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

QUANTIZATION_STRATEGIES: tuple[str | None, ...] = (None, "4bit")

MODALITIES: tuple[str, ...] = ("vision_text", "text")

DEFAULT_MODALITY = "vision_text"

_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class ModelConfigError(ValueError):
    pass


def resolve_quantization(
    model_cfg: dict[str, Any], train_cfg: dict[str, Any] | None = None
) -> str | None:
    declared = (train_cfg or {}).get("quantization", model_cfg.get("quantization"))
    if declared in (None, "", "none"):
        return None
    if declared not in QUANTIZATION_STRATEGIES:
        known = ", ".join(repr(s) for s in QUANTIZATION_STRATEGIES if s)
        raise ModelConfigError(
            f"Unknown quantization strategy {declared!r}. Expected one of: {known}, "
            "or omit the key for an immediate load."
        )
    return declared


def resolve_dtype(model_cfg: dict[str, Any]) -> torch.dtype:
    declared = model_cfg.get("dtype")
    if declared == "auto":
        raise ModelConfigError(
            'dtype: "auto" is not accepted. It resolves against the checkpoint\'s '
            "own metadata, which is absent on composite (vision+text) configs — so "
            "it reads as 'let HF work it out' while doing nothing. Declare the "
            'compute dtype explicitly (dtype: "bfloat16").'
        )
    if declared is None:
        raise ModelConfigError(
            'model.dtype is required. Declare it explicitly, e.g. dtype: "bfloat16".'
        )
    if declared not in _DTYPES:
        known = ", ".join(sorted(_DTYPES))
        raise ModelConfigError(f"Unknown dtype {declared!r}. Expected one of: {known}.")
    return _DTYPES[declared]


def resolve_modality(model_cfg: dict[str, Any]) -> str:
    declared = model_cfg.get("modality", DEFAULT_MODALITY)
    if declared not in MODALITIES:
        known = ", ".join(repr(m) for m in MODALITIES)
        raise ModelConfigError(
            f"Unknown modality {declared!r}. Expected one of: {known}, or omit the "
            f"key for {DEFAULT_MODALITY!r}."
        )
    return str(declared)


def auto_class_for_modality(modality: str) -> Any:
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    if modality == "text":
        return AutoModelForCausalLM
    if modality == "vision_text":
        return AutoModelForImageTextToText
    raise ModelConfigError(f"Unknown modality {modality!r}.")


def _checkpoint_config(model_id: str) -> Any | None:
    try:
        return AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        return None


def checkpoint_is_composite(model_id: str) -> bool | None:
    config = _checkpoint_config(model_id)
    if config is None:
        return None
    return getattr(config, "vision_config", None) is not None


def validate_modality(model_id: str, model_cfg: dict[str, Any]) -> str:
    modality = resolve_modality(model_cfg)
    if modality == "text" and checkpoint_is_composite(model_id) is True:
        raise ModelConfigError(
            f"'{model_id}' carries a vision_config, so modality: \"text\" would load "
            "its language path alone and train an adapter against a model that is "
            'not the released one. Declare modality: "vision_text", or point '
            "`model.id` at a text-only release."
        )
    return modality


def checkpoint_quant_method(model_id: str) -> str | None:
    config = _checkpoint_config(model_id)
    if config is None:
        return None
    embedded = getattr(config, "quantization_config", None)
    if not embedded:
        return None
    if isinstance(embedded, dict):
        return embedded.get("quant_method")
    return getattr(embedded, "quant_method", None)


def validate_base_model_config(
    model_id: str, model_cfg: dict[str, Any], train_cfg: dict[str, Any] | None = None
) -> str | None:
    resolve_dtype(model_cfg)
    strategy = resolve_quantization(model_cfg, train_cfg)
    embedded = checkpoint_quant_method(model_id)

    if embedded is None:
        return strategy

    raise ModelConfigError(
        f"'{model_id}' is pre-quantized with quant_method={embedded!r}. Loading it "
        "under any strategy here would drop or misread the quantization metadata "
        "that lives alongside the weights. This repository trains only from "
        "unquantized checkpoints — point `model.id` at a BF16 release, with "
        'quantization: "4bit" if the full weights do not fit.'
    )


def load_processor(model_id: str, modality: str, *, trust_remote_code: bool = True) -> Any:
    from transformers import AutoProcessor, AutoTokenizer

    if modality == "text":
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        model_id, trust_remote_code=trust_remote_code
    )


def processor_tokenizer(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def apply_chat_template_file(processor: Any, template_path: str | Path) -> str:
    path = Path(template_path)
    if not path.is_file():
        raise ModelConfigError(
            f"model.chat_template points at {path}, which does not exist. Paths are "
            "resolved from the repository root."
        )
    template = path.read_text()
    if not template.strip():
        raise ModelConfigError(f"model.chat_template at {path} is empty.")
    tokenizer = processor_tokenizer(processor)
    tokenizer.chat_template = template
    if processor is not tokenizer:
        processor.chat_template = template
    return template


def configured_chat_template(model_name: str, config_dir: str | Path = "configs") -> Path | None:
    import yaml

    config_path = Path(config_dir) / f"{model_name}.yaml"
    if not config_path.is_file():
        return None
    config = yaml.safe_load(config_path.read_text()) or {}
    template = (config.get("model") or {}).get("chat_template")
    return Path(template) if template else None


def require_chat_template(processor: Any, model_id: str) -> None:
    tokenizer = processor_tokenizer(processor)
    if getattr(tokenizer, "chat_template", None) or getattr(processor, "chat_template", None):
        return
    raise ModelConfigError(
        f"'{model_id}' ships no chat template, so the WVS chat records cannot be "
        "rendered into training text. Point `model.chat_template` at a Jinja file "
        "(see configs/templates/), or use an instruction-tuned release of this base."
    )


def build_base_model(
    model_id: str,
    *,
    auto_class: Any,
    quantization: str | None,
    dtype: torch.dtype,
    device: str,
    attn_implementation: str | None = None,
    trust_remote_code: bool = True,
) -> Any:
    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    if quantization == "4bit" and device == "cuda":
        from transformers import BitsAndBytesConfig

        return auto_class.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            ),
            device_map={"": device},
            **kwargs,
        )

    model = auto_class.from_pretrained(
        model_id,
        dtype=dtype,
        device_map={"": device} if device == "cuda" else None,
        **kwargs,
    )
    return model if device == "cuda" else model.to(device)
