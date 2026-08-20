"""Base-model loading — one place that decides how weights reach the GPU.

Training and annotation used to load bases independently, and they disagreed;
that disagreement once trained nine adapters against a corrupt forward pass.
So: name the strategy explicitly in the config, validate it against the
checkpoint before any weights are read, and never pass a None quantization
config — `from_pretrained` applies that kwarg to the config object, overwriting
the checkpoint's own `quantization_config`.

Strategies (config key `training.quantization`, falling back to
`model.quantization`):

    None      immediate load at the declared dtype.
    "4bit"    QLoRA via bitsandbytes NF4 — for BF16 checkpoints too large to
              hold whole (gemma4_31b, qwen3_27b).

Pre-quantized checkpoints are refused outright: this repository trains only
from unquantized releases.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig

QUANTIZATION_STRATEGIES: tuple[str | None, ...] = (None, "4bit")

_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class ModelConfigError(ValueError):
    """A model config cannot be loaded safely as written."""


def resolve_quantization(
    model_cfg: dict[str, Any], train_cfg: dict[str, Any] | None = None
) -> str | None:
    """Return the declared quantization strategy.

    `training.quantization` is where `"4bit"` has always lived; `model.quantization`
    is accepted so inference-side configs need no `training` section.

    Args:
        model_cfg (dict): `model` section of the config.
        train_cfg (dict | None): `training` section, if the config has one.

    Returns:
        str | None: One of QUANTIZATION_STRATEGIES.

    Raises:
        ModelConfigError: If the declared value is not a known strategy.
    """
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
    """Map the config's `dtype` string onto a torch dtype.

    Args:
        model_cfg (dict): `model` section of the config.

    Returns:
        torch.dtype: The declared compute dtype.

    Raises:
        ModelConfigError: If `dtype` is missing, "auto", or unrecognised.
    """
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


def checkpoint_quant_method(model_id: str) -> str | None:
    """Return the `quant_method` baked into a checkpoint, if it is pre-quantized.

    Reads config metadata only — no weights are downloaded or allocated. A config
    that cannot be read at all — offline, gated, or not yet downloaded — also yields
    None, since validation cannot speak to a checkpoint it cannot see.

    Args:
        model_id (str): HF repo id or local path.

    Returns:
        str | None: e.g. "fp8", or None for an unquantized checkpoint.
    """
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
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
    """Fail before loading if the config cannot load the checkpoint faithfully.

    Catches the two ways this has gone wrong: an unusable `dtype`, and a
    pre-quantized checkpoint, which no declared strategy can load faithfully.

    Args:
        model_id (str): HF repo id or local path.
        model_cfg (dict): `model` section of the config.
        train_cfg (dict | None): `training` section, if any.

    Returns:
        str | None: The validated quantization strategy.

    Raises:
        ModelConfigError: If the combination would load something other than what
            the config describes.
    """
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
    """Load a base model onto `device` under the given quantization strategy.

    Args:
        model_id (str): HF repo id or local path.
        auto_class (Any): The `Auto*` class to load with, e.g.
            `AutoModelForImageTextToText`.
        quantization (str | None): A value from QUANTIZATION_STRATEGIES.
        dtype (torch.dtype): Compute dtype.
        device (str): "cuda", "mps", or "cpu".
        attn_implementation (str | None): Passed through when set.
        trust_remote_code (bool): Passed through to `from_pretrained`.

    Returns:
        Any: The loaded model, on `device`.
    """
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
