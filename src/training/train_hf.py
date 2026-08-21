"""
HuggingFace LoRA fine-tuning — works on CUDA (Linux/RTX) and MPS (Apple Silicon).

Training input: WVS cultural Q&A text by default, or any chat-format JSONL
(including visual SFT with image-path messages) via a config-level
`data: {train_jsonl, val_jsonl}` section.
Device is selected automatically: cuda > mps > cpu.
How the base weights are loaded is declared by `training.quantization` and
`model.modality`, and enforced by src/utils/model_loading.py: "4bit" for QLoRA
via bitsandbytes, omitted for an immediate load; "text" for a text-only base
such as llama3_2_3b, "vision_text" (the default) for a composite one. The config
is validated against the checkpoint's own metadata before any weights are read,
and the loaded base must beat uniform guessing on a real batch before the first
optimizer step.

Training automatically resumes from the latest checkpoint if one exists,
and continues logging to the same MLflow run.

Usage:
    uv run python src/training/train_hf.py \\
        --config configs/phi4.yaml \\
        --culture arabic

    uv run python src/training/train_hf.py \\
        --config configs/gemma4_31b.yaml \\
        --culture arabic \\
        --debug
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, cast

import mlflow
import torch
import yaml
from dotenv import load_dotenv
from peft import LoraConfig, TaskType
from rich.console import Console
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer  # type: ignore[attr-defined]

from src.data.dataset import load_hf_dataset
from src.training.early_stopping import EarlyStopping
from src.utils.device import get_device
from src.utils.model_loading import (
    apply_chat_template_file,
    auto_class_for_modality,
    build_base_model,
    load_processor,
    processor_tokenizer,
    require_chat_template,
    resolve_dtype,
    validate_base_model_config,
    validate_modality,
)

load_dotenv()
console = Console()

PROCESSED_DIR = Path("data/processed")
CHECKPOINTS_DIR = Path("checkpoints")

VISION_TEXT_EXCLUDE_MODULES = ".*(vision_tower|audio_tower).*"


def default_exclude_modules(modality: str) -> str | None:
    """Keep LoRA on the language path, which a text-only base is entirely made of.

    A text-only checkpoint has no tower to exclude, and PEFT warns when an
    exclusion pattern matches nothing — noise that reads like a misconfigured
    adapter every run. Composite configs may still name their own pattern:
    the families disagree on what the tower is called.
    """
    return None if modality == "text" else VISION_TEXT_EXCLUDE_MODULES


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML training config.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Parsed configuration.
    """
    with open(config_path) as f:
        return cast(dict[str, Any], yaml.safe_load(f))


def get_model_key(cfg: dict[str, Any], config_path: str | Path) -> str:
    """Stable experiment/checkpoint identity, independent of config filename."""
    return str(cfg.get("model", {}).get("key") or Path(config_path).stem)


def checkpoint_output_dir(culture: str, model_name: str, condition: str) -> Path:
    return CHECKPOINTS_DIR / culture / model_name / condition


def _apply_soft_token_budget(image_processor: Any, max_pixels: int) -> dict[str, object] | None:
    """Map a pixel ceiling onto token-budget processors (e.g. Gemma-4).

    These resize each image so its patch count stays within
    max_soft_tokens * pooling_kernel_size**2, so the pixel ceiling translates
    to a soft-token count via (patch_size * pooling_kernel_size)**2 px/token.
    The processor only accepts a fixed menu of soft-token counts, so the
    largest one that fits the pixel budget is used, or the minimum supported
    count when none fit.
    """
    max_soft_tokens = getattr(image_processor, "max_soft_tokens", None)
    patch_size = getattr(image_processor, "patch_size", None)
    pooling_kernel_size = getattr(image_processor, "pooling_kernel_size", None)
    if not (max_soft_tokens and patch_size and pooling_kernel_size):
        return None

    pixels_per_token = (int(patch_size) * int(pooling_kernel_size)) ** 2
    budget_tokens = max(1, max_pixels // pixels_per_token)
    supported = getattr(
        sys.modules.get(type(image_processor).__module__), "_SUPPORTED_SOFT_TOKENS", None
    )
    if supported:
        fitting = [tokens for tokens in supported if tokens <= budget_tokens]
        budget_tokens = max(fitting) if fitting else min(supported)
    image_processor.max_soft_tokens = budget_tokens
    return {
        "max_soft_tokens": budget_tokens,
        "effective_max_pixels": budget_tokens * pixels_per_token,
    }


def _apply_image_token_budget(image_processor: Any, max_pixels: int) -> dict[str, object] | None:
    """Map a pixel ceiling onto merged-token processors (e.g. Muse Glimmer).

    These carry no `size` at all — each image is resized so its merged-patch
    grid stays within max_image_tokens, so the pixel ceiling translates to a
    token count via (patch_size * merge_size)**2 px/token.
    """
    max_image_tokens = getattr(image_processor, "max_image_tokens", None)
    patch_size = getattr(image_processor, "patch_size", None)
    merge_size = getattr(image_processor, "merge_size", None)
    if not (max_image_tokens and patch_size and merge_size):
        return None

    pixels_per_token = (int(patch_size) * int(merge_size)) ** 2
    budget_tokens = max(1, max_pixels // pixels_per_token)
    image_processor.max_image_tokens = budget_tokens
    return {
        "max_image_tokens": budget_tokens,
        "effective_max_pixels": budget_tokens * pixels_per_token,
    }


def _size_get(size: Any, key: str) -> Any:
    """Read a field from either a plain dict or a transformers SizeDict."""
    if isinstance(size, dict):
        return size.get(key)
    return getattr(size, key, None)


def _size_set(size: Any, key: str, value: Any) -> None:
    if isinstance(size, dict):
        size[key] = value
    else:
        setattr(size, key, value)


def configure_processor_image_budget(processor: Any, max_pixels: int | None) -> dict[str, object]:
    """Apply the configured image budget across Qwen/Gemma processor shapes.

    Fixed-resolution processors expose size.height/width; their aspect ratio is
    preserved while the area is reduced to the configured ceiling. Qwen-VL-family
    processors instead store pixel-AREA budgets (min_pixels / max_pixels) in
    size.shortest_edge / size.longest_edge.
    """
    if max_pixels is None:
        return {}
    max_pixels = int(max_pixels)
    if max_pixels < 1:
        raise ValueError("model.image_max_pixels must be >= 1")
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("Active VLM processor has no image_processor")

    applied: dict[str, object] = {"max_pixels": max_pixels}
    handled = False
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_pixels
        handled = True
    size = getattr(image_processor, "size", None)
    height, width = _size_get(size, "height"), _size_get(size, "width")
    if height and width:
        height, width = int(height), int(width)
        scale = min(1.0, (max_pixels / max(1, height * width)) ** 0.5)
        _size_set(size, "height", max(1, int(height * scale)))
        _size_set(size, "width", max(1, int(width * scale)))
        applied["size"] = {
            "height": _size_get(size, "height"),
            "width": _size_get(size, "width"),
        }
        handled = True
    elif _size_get(size, "longest_edge") and hasattr(image_processor, "merge_size"):
        _size_set(size, "longest_edge", max_pixels)
        shortest_edge = _size_get(size, "shortest_edge")
        if shortest_edge and int(shortest_edge) > max_pixels:
            _size_set(size, "shortest_edge", max_pixels)
        applied["size"] = {
            "shortest_edge": _size_get(size, "shortest_edge"),
            "longest_edge": max_pixels,
        }
        handled = True
    else:
        token_budget = _apply_soft_token_budget(image_processor, max_pixels)
        if token_budget is None:
            token_budget = _apply_image_token_budget(image_processor, max_pixels)
        if token_budget is not None:
            applied.update(token_budget)
            handled = True
    if not handled:
        raise ValueError(
            "Cannot apply model.image_max_pixels to this image processor; "
            "configure a supported max_pixels, height/width, pixel-area "
            "longest_edge, max_soft_tokens, or max_image_tokens processor"
        )
    return applied


def get_wvs_paths(culture: str) -> tuple[Path, Path | None]:
    """Return (train_jsonl, val_jsonl) using last 10% of WVS cultural data as val."""
    src = PROCESSED_DIR / culture / "wvs_cultural_anchoring.jsonl"
    if not src.exists():
        raise FileNotFoundError(
            f"WVS data not found: {src}\n"
            f"Run: uv run python src/data/culture_training_data.py --culture {culture}"
        )
    with open(src) as f:
        lines = [line for line in f if line.strip()]
    n_val = max(1, len(lines) // 10)
    train_lines, val_lines = lines[:-n_val], lines[-n_val:]
    train_path = PROCESSED_DIR / culture / "wvs_cultural_train.jsonl"
    val_path = PROCESSED_DIR / culture / "wvs_cultural_val.jsonl"
    with open(train_path, "w") as f:
        f.writelines(train_lines)
    with open(val_path, "w") as f:
        f.writelines(val_lines)
    console.print(f"  WVS split: {len(train_lines)} train / {len(val_lines)} val")
    return train_path, val_path


def get_data_paths(cfg: dict[str, Any], culture: str) -> tuple[Path, Path | None]:
    """Return (train_jsonl, val_jsonl): explicit `data:` section if present, else WVS split."""
    data_cfg = cfg.get("data")
    if not data_cfg:
        return get_wvs_paths(culture)
    train = Path(data_cfg["train_jsonl"])
    if not train.exists():
        raise FileNotFoundError(f"Training data not found: {train}")
    val = Path(data_cfg["val_jsonl"]) if data_cfg.get("val_jsonl") else None
    return train, (val if val and val.exists() else None)


def load_training_datasets(cfg: dict[str, Any], culture: str) -> tuple[Any, Any, dict[str, int]]:
    """Load the WVS JSONL training and validation splits."""
    data_cfg = cfg.get("data") or {}
    data_format = str(data_cfg.get("format", "jsonl")).lower()
    if data_format != "jsonl":
        raise ValueError(f"Unsupported data.format: {data_format!r}")
    train_jsonl, val_jsonl = get_data_paths(cfg, culture)
    train_dataset = load_hf_dataset(train_jsonl)
    val_dataset = load_hf_dataset(val_jsonl) if val_jsonl and val_jsonl.exists() else None
    return (
        train_dataset,
        val_dataset,
        {
            "train_rows": len(train_dataset),
            "val_rows": len(val_dataset) if val_dataset is not None else 0,
        },
    )


def _find_last_checkpoint(output_dir: str) -> str | None:
    """Return the path of the latest HuggingFace checkpoint, or None."""
    checkpoints = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


_MLFLOW_RUN_ID_FILE = "mlflow_run_id.txt"
_DONE_SENTINEL = "TRAINING_DONE"


def _is_training_complete(output_dir: str, run_name: str = "", experiment_name: str = "") -> bool:
    """Return True if training is done.

    Checks the local sentinel file first (fast path), then falls back to
    querying MLflow for a FINISHED run with the given name — needed for runs
    that completed before the sentinel system was introduced.
    If MLflow confirms completion, the sentinel is written so future checks
    are instant.
    """
    if (Path(output_dir) / _DONE_SENTINEL).exists():
        return True
    if run_name and experiment_name:
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            exp = client.get_experiment_by_name(experiment_name)
            if exp:
                hits = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string=f"run_name = '{run_name}' and attributes.status = 'FINISHED'",
                    max_results=1,
                )
                if hits:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    _mark_training_complete(output_dir)
                    return True
        except Exception:
            pass
    return False


def _mark_training_complete(output_dir: str) -> None:
    (Path(output_dir) / _DONE_SENTINEL).touch()


def _load_mlflow_run_id(output_dir: str) -> str | None:
    path = Path(output_dir) / _MLFLOW_RUN_ID_FILE
    if path.exists():
        return path.read_text().strip() or None
    return None


def _save_mlflow_run_id(output_dir: str, run_id: str) -> None:
    (Path(output_dir) / _MLFLOW_RUN_ID_FILE).write_text(run_id)


def _resume_mlflow_run(output_dir: str, run_name: str) -> tuple[str | None, bool]:
    """Return (run_id_to_use, is_new).

    Always reuses the saved run — resetting it to RUNNING if needed — so that
    every restart continues the same MLflow trace regardless of whether the
    previous session ended via Ctrl+C (FAILED) or a crash (RUNNING).
    A new run is only created when no saved run_id exists or the ID is gone
    from the server.
    """
    from mlflow.tracking import MlflowClient

    saved_run_id = _load_mlflow_run_id(output_dir)
    if saved_run_id:
        try:
            client = MlflowClient()
            status = client.get_run(saved_run_id).info.status
            if status != "RUNNING":
                client.update_run(saved_run_id, status="RUNNING")
            console.print(f"  Continuing MLflow run [bold]{saved_run_id}[/bold] (was {status})")
            return saved_run_id, False
        except Exception as exc:
            console.print(
                f"  [yellow]Cannot reopen MLflow run {saved_run_id} ({exc}) — "
                "starting new run[/yellow]"
            )
    return None, True


def preprocess_logits_for_metrics(logits: Any, labels: Any) -> torch.Tensor:
    """Reduce [B, T, V] logits to per-token predictive entropy [B, T].

    Storing full-vocab logits across eval batches OOMs; entropy is computed
    per micro-batch and only the [B, T] result is accumulated. The float32
    upcast keeps log_softmax numerically stable for bf16 models.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    logp = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def compute_metrics_fn(eval_preds: Any) -> dict[str, float]:
    """Mean next-token predictive entropy (nats) over supervised tokens.

    Positions with label -100 (prompt, image, and padding tokens) are
    excluded. Logits at position t predict token t+1, hence the one-step
    shift before masking.

    Args:
        eval_preds: (per_token_entropy, labels) tuple from the Trainer,
            both [N, T] arrays.

    Returns:
        dict: {"entropy": float} — logged by the Trainer as eval_entropy.
    """
    entropies, labels = eval_preds
    ent = entropies[:, :-1]
    lab = labels[:, 1:]
    mask = lab != -100
    return {"entropy": float(ent[mask].mean())}


def _check_vram_headroom(model_cfg: dict[str, Any], quantization: str | None, device: str) -> None:
    """Fail fast with a clear message if the model cannot fit in GPU VRAM.

    Checks free VRAM as well as total: this box has one GPU shared with the
    machine-bias-reproduction runs, so a model that fits the card in principle
    can still OOM a third of the way through weight loading.

    The checkpoint size is estimated from the local HuggingFace cache; a model
    that is not cached yet is left alone so the download and load proceed
    naturally. The 3 GB of headroom covers the LoRA parameters, activations,
    and optimizer state, and the 0.25 ratio for "4bit" reflects NF4 compressing
    16-bit weights to 4-bit.

    Args:
        model_cfg: model section of the YAML config.
        quantization: declared quantization strategy, or None for immediate load.
        device: resolved device string ("cuda", "mps", "cpu").
    """
    if device != "cuda":
        return

    import torch
    from huggingface_hub import scan_cache_dir

    model_id = model_cfg["id"]

    try:
        cache = scan_cache_dir()
        repo = next((r for r in cache.repos if r.repo_id == model_id), None)
        checkpoint_bytes = repo.size_on_disk if repo else 0
    except Exception:
        checkpoint_bytes = 0

    if checkpoint_bytes == 0:
        return

    checkpoint_gb = checkpoint_bytes / 1024**3
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / 1024**3
    total_gb = total_bytes / 1024**3

    headroom_gb = 3.0
    ratios: dict[str | None, float] = {"4bit": 0.25}
    effective_gb = checkpoint_gb * ratios.get(quantization, 1.0)
    required_gb = effective_gb + headroom_gb

    def _fail(limit_name: str, limit_gb: float, advice: str) -> None:
        raise RuntimeError(
            f"\nModel '{model_id}' cannot fit on this GPU ({limit_name}).\n"
            f"  Checkpoint : {checkpoint_gb:.1f} GB on disk\n"
            f"  Effective  : {effective_gb:.1f} GB resident "
            f"(quantization: {quantization or 'none'})\n"
            f"  Headroom   : {headroom_gb:.1f} GB (LoRA + activations)\n"
            f"  Required   : {required_gb:.1f} GB\n"
            f"  GPU total  : {total_gb:.1f} GB\n"
            f"  Free now   : {free_gb:.1f} GB\n"
            f"\n{advice}"
        )

    if required_gb > total_gb:
        _fail(
            "exceeds the card",
            total_gb,
            "Add 'quantization: \"4bit\"' to the training section to cut the weight "
            "footprint to a quarter — but only for a checkpoint that is not already "
            "pre-quantized, since quantizers cannot be stacked. Otherwise this model "
            "does not fit on this hardware.",
        )

    if required_gb > free_gb:
        _fail(
            "fits the card, but not right now",
            free_gb,
            f"{total_gb - free_gb:.1f} GB is held by another process — most likely a "
            "machine-bias-reproduction run on the same GPU. Check with `nvidia-smi` "
            "and start again once it drains.",
        )


def _assert_base_is_sane(trainer: Any, model: Any, processor: Any) -> tuple[float, float]:
    """Verify the loaded base beats uniform guessing before spending a run on it.

    A pretrained model scoring no better than chance on its own training
    distribution has been loaded wrong, and no amount of LoRA will recover it —
    the adapter just grows to compensate for a broken forward pass; a corrupt
    load once drove nine adapters to a mean update norm of 16,977 where healthy
    runs sit near 0.5. Cheaper to learn that in one forward pass than in 3,900
    steps.

    Args:
        trainer: The constructed SFTTrainer, used for its real collated batch.
        model: The PEFT-wrapped model to score.
        processor: Processor, for the tokenizer vocabulary as a fallback.

    Returns:
        tuple[float, float]: (loss, loss / ln(vocab_size)).

    Raises:
        RuntimeError: If the base does no better than uniform guessing.
    """
    import math

    text_config = model.config.get_text_config()
    vocab_size = getattr(text_config, "vocab_size", None) or len(processor_tokenizer(processor))
    guess_loss = math.log(vocab_size)

    batch = next(iter(trainer.get_train_dataloader()))
    batch = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            loss = float(model(**batch).loss)
    finally:
        model.train(was_training)

    ratio = loss / guess_loss
    if ratio >= 1.0:
        raise RuntimeError(
            f"\nThe base model scores worse than uniform guessing — it is not loaded "
            f"correctly.\n"
            f"  Forward loss : {loss:.4f}\n"
            f"  ln(vocab)    : {guess_loss:.4f}  (vocab_size {vocab_size})\n"
            f"  Ratio        : {ratio:.2f}  (healthy runs land at 0.35-0.85)\n"
            f"\nTraining from here produces an adapter that only makes sense on top of "
            "this broken load, and is unusable anywhere else. Check `model.dtype` and "
            "`training.quantization` against the checkpoint — a load that drops or "
            "misreads the metadata stored alongside the weights leaves values orders "
            "of magnitude off."
        )
    if ratio >= 0.9:
        console.print(
            f"[yellow]  Base health: loss {loss:.4f} vs ln(vocab) {guess_loss:.4f} "
            f"(ratio {ratio:.2f}) — close to chance, verify the load[/yellow]"
        )
    else:
        console.print(
            f"[green]  Base health: loss {loss:.4f} vs ln(vocab) {guess_loss:.4f} "
            f"(ratio {ratio:.2f})[/green]"
        )
    return loss, ratio


def train(cfg: dict[str, Any], model_name: str, culture: str, debug: bool) -> None:
    """Run LoRA fine-tuning for one (model, culture) pair.

    Resumes from the latest checkpoint and reuses the existing MLflow run
    if both exist. Marks completion via a sentinel file to allow idempotent
    reruns from the orchestration script. The tracking URI is set before the
    completion check so that the MLflow query behind it works.

    The base weights are placed on a single GPU by the device_map={"": device}
    inside build_base_model, which avoids pre-computing a map from parameter
    counts: balanced_low_0 over-estimates and spills layers to the meta device,
    which breaks LoRA backprop.

    Args:
        cfg (dict): Parsed YAML config (model, lora, training, mlflow sections).
        model_name (str): Config stem used as a directory name and run tag.
        culture (str): Target culture (e.g., "arabic").
        debug (bool): If True, run only 1 epoch for fast iteration.
    """
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    condition = cfg.get("condition", "cultural")
    output_dir = str(checkpoint_output_dir(culture, model_name, condition))
    run_name = f"{model_name}_{culture}_{condition}"

    tracking_uri = cfg.get("mlflow", {}).get("tracking_uri", "http://127.0.0.1:5000")
    experiment_name = cfg["mlflow"]["experiment"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name

    if _is_training_complete(output_dir, run_name, experiment_name):
        console.print(f"[dim]Skipping {model_name}/{culture} — already FINISHED in MLflow[/dim]")
        return

    device = get_device()
    console.print(f"Device    : [bold]{device}[/bold]")
    console.print(f"Loading model [bold]{model_cfg['id']}[/bold]...")

    modality = validate_modality(model_cfg["id"], model_cfg)
    quantization = validate_base_model_config(model_cfg["id"], model_cfg, train_cfg)
    torch_dtype = resolve_dtype(model_cfg)
    console.print(f"  Modality: {modality}")
    if quantization:
        console.print(f"  [yellow]Quantization: {quantization}[/yellow]")

    try:
        import flash_attn  # type: ignore[import-not-found] # noqa: F401

        attn_impl = "flash_attention_2" if device == "cuda" else "eager"
    except ImportError:
        attn_impl = "eager"

    _check_vram_headroom(model_cfg, quantization, device)
    model = build_base_model(
        model_cfg["id"],
        auto_class=auto_class_for_modality(modality),
        quantization=quantization,
        dtype=torch_dtype,
        device=device,
        attn_implementation=attn_impl,
    )

    meta_params = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if meta_params:
        raise RuntimeError(
            f"{len(meta_params)} model parameter(s) landed on the meta device "
            "(CPU-offloaded). LoRA training requires all base parameters on GPU. "
            "Add 'quantization: 4bit' to your config to reduce memory, or use a "
            "GPU with enough VRAM to hold the full model."
        )

    model.enable_input_require_grads()
    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    processor = load_processor(model_cfg["id"], modality)
    if modality == "text":
        if model_cfg.get("image_max_pixels") is not None:
            raise ValueError(
                'model.image_max_pixels is set on a modality: "text" config, where no '
                "image ever reaches the model. Remove the key so the config describes "
                "what is actually trained."
            )
    else:
        image_budget = configure_processor_image_budget(
            processor, model_cfg.get("image_max_pixels")
        )
        if image_budget:
            console.print(f"  Image processor budget: {image_budget}")

    if model_cfg.get("chat_template"):
        apply_chat_template_file(processor, model_cfg["chat_template"])
        console.print(f"  Chat template: {model_cfg['chat_template']}")
    require_chat_template(processor, model_cfg["id"])
    processor_tokenizer(processor).model_max_length = train_cfg["max_seq_len"]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        exclude_modules=lora_cfg.get("exclude_modules", default_exclude_modules(modality)),
        bias="none",
    )

    train_dataset, val_dataset, data_stats = load_training_datasets(cfg, culture)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    last_checkpoint = _find_last_checkpoint(output_dir)
    if last_checkpoint:
        console.print(f"  Resuming from checkpoint [bold]{last_checkpoint}[/bold]")

    resume_run_id, is_new = _resume_mlflow_run(output_dir, run_name)
    with mlflow.start_run(
        run_id=resume_run_id, run_name=run_name if is_new else None
    ) as active_run:
        if is_new:
            _save_mlflow_run_id(output_dir, active_run.info.run_id)
            console.print(f"  New MLflow run [bold]{active_run.info.run_id}[/bold]")

        mlflow.log_params({f"data_{key}": value for key, value in data_stats.items()})

        use_bf16 = device in ("cuda", "mps")

        training_args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=train_cfg["batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation"],
            num_train_epochs=1 if debug else train_cfg["max_epochs"],
            learning_rate=train_cfg["learning_rate"],
            lr_scheduler_type=train_cfg.get("lr_schedule", "cosine"),
            warmup_steps=train_cfg["warmup_steps"],
            bf16=use_bf16,
            fp16=False,
            optim="adamw_torch",
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=train_cfg["eval_steps"] if val_dataset else None,
            save_strategy="steps",
            save_steps=train_cfg["save_steps"],
            load_best_model_at_end=bool(val_dataset),
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False,
            report_to="mlflow",
            logging_steps=train_cfg["logging_steps"],
            max_grad_norm=train_cfg["max_grad_norm"],
            run_name=run_name,
            max_length=train_cfg["max_seq_len"],
            assistant_only_loss=bool(train_cfg.get("assistant_only_loss", False)),
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=peft_config,
            processing_class=processor,
            compute_metrics=compute_metrics_fn if val_dataset else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics if val_dataset else None,
        )

        stopper = EarlyStopping(
            patience=train_cfg["early_stopping_patience"],
            min_epochs=train_cfg.get("early_stopping_min_epochs", 10),
            monitor="loss",
        )

        class _EarlyStopCallback(TrainerCallback):
            """Trainer callback that triggers early stopping on validation loss."""

            def on_evaluate(  # type: ignore[override]
                self,
                args: Any,
                state: Any,
                control: Any,
                metrics: Any,
                **kwargs: Any,
            ) -> None:
                val_loss = metrics.get("eval_loss")
                if val_loss is None:
                    return
                epoch = int(state.epoch or 0)
                if stopper.step(val_loss, epoch=epoch):
                    control.should_training_stop = True
                    console.print(
                        f"[yellow]Early stopping at step {state.global_step} "
                        f"(best loss={stopper.best:.4f} @ epoch {stopper.best_epoch})[/yellow]"
                    )
                if stopper.improved:
                    entropy = metrics.get("eval_entropy", float("nan"))
                    console.print(
                        f"[green]New best val loss: {val_loss:.4f} "
                        f"(entropy {entropy:.3f} nats) at epoch {epoch}[/green]"
                    )

        trainer.add_callback(_EarlyStopCallback())

        base_loss, base_ratio = _assert_base_is_sane(trainer, model, processor)
        mlflow.log_metric("base_health_loss", base_loss)
        mlflow.log_metric("base_health_loss_over_guess", base_ratio)

        trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
        _mark_training_complete(output_dir)
        console.print(f"[bold green]✓ Saved to {output_dir}[/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HuggingFace LoRA training for culture-mllm (CUDA + MPS)."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--culture", required=True)
    parser.add_argument("--debug", action="store_true", help="1 epoch only")
    args = parser.parse_args()

    console.rule("[bold blue]culture-mllm Hugging Face LoRA Training[/bold blue]")
    cfg = load_config(args.config)
    model_name = get_model_key(cfg, args.config)

    console.print(f"Model     : {cfg['model']['id']}")
    console.print(f"Culture   : {args.culture}")
    console.print(f"Debug     : {args.debug}\n")

    train(cfg, model_name, args.culture, args.debug)


if __name__ == "__main__":
    main()
