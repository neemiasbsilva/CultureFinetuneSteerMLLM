"""The parts of a LoRA run that both training tracks share, relocated from train_hf.py."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlflow
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from rich.console import Console
from transformers import TrainerCallback

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

console = Console()

CHECKPOINTS_DIR = Path("checkpoints")

VISION_TEXT_EXCLUDE_MODULES = ".*(vision_tower|audio_tower).*"

MLFLOW_RUN_ID_FILE = "mlflow_run_id.txt"
DONE_SENTINEL = "TRAINING_DONE"


def default_exclude_modules(modality: str) -> str | None:
    return None if modality == "text" else VISION_TEXT_EXCLUDE_MODULES


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        return cast(dict[str, Any], yaml.safe_load(f))


def get_model_key(cfg: dict[str, Any], config_path: str | Path) -> str:
    return str(cfg.get("model", {}).get("key") or Path(config_path).stem)


def checkpoint_output_dir(culture: str, model_name: str, condition: str) -> Path:
    return CHECKPOINTS_DIR / culture / model_name / condition


def _apply_soft_token_budget(image_processor: Any, max_pixels: int) -> dict[str, object] | None:
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
    if isinstance(size, dict):
        return size.get(key)
    return getattr(size, key, None)


def _size_set(size: Any, key: str, value: Any) -> None:
    if isinstance(size, dict):
        size[key] = value
    else:
        setattr(size, key, value)


def configure_processor_image_budget(processor: Any, max_pixels: int | None) -> dict[str, object]:
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


def find_last_checkpoint(output_dir: str) -> str | None:
    checkpoints = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def is_training_complete(output_dir: str, run_name: str = "", experiment_name: str = "") -> bool:
    if (Path(output_dir) / DONE_SENTINEL).exists():
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
                    mark_training_complete(output_dir)
                    return True
        except Exception:
            pass
    return False


def mark_training_complete(output_dir: str) -> None:
    (Path(output_dir) / DONE_SENTINEL).touch()


def load_mlflow_run_id(output_dir: str) -> str | None:
    path = Path(output_dir) / MLFLOW_RUN_ID_FILE
    if path.exists():
        return path.read_text().strip() or None
    return None


def save_mlflow_run_id(output_dir: str, run_id: str) -> None:
    (Path(output_dir) / MLFLOW_RUN_ID_FILE).write_text(run_id)


def resume_mlflow_run(output_dir: str, run_name: str) -> tuple[str | None, bool]:
    from mlflow.tracking import MlflowClient

    saved_run_id = load_mlflow_run_id(output_dir)
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


def check_vram_headroom(model_cfg: dict[str, Any], quantization: str | None, device: str) -> None:
    if device != "cuda":
        return

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


@dataclass(frozen=True)
class Backbone:
    model: Any
    processor: Any
    modality: str
    quantization: str | None
    device: str


@dataclass(frozen=True)
class ResumeState:
    last_checkpoint: str | None
    run_id: str | None
    is_new: bool


def configure_mlflow(cfg: dict[str, Any]) -> tuple[str, str]:
    tracking_uri = cfg.get("mlflow", {}).get("tracking_uri", "http://127.0.0.1:5000")
    experiment_name = cfg["mlflow"]["experiment"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    return str(tracking_uri), str(experiment_name)


def load_backbone(model_cfg: dict[str, Any], train_cfg: dict[str, Any]) -> Backbone:
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

    check_vram_headroom(model_cfg, quantization, device)
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

    return Backbone(
        model=model,
        processor=processor,
        modality=modality,
        quantization=quantization,
        device=device,
    )


def build_lora_config(lora_cfg: dict[str, Any], modality: str) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        exclude_modules=lora_cfg.get("exclude_modules", default_exclude_modules(modality)),
        bias="none",
    )


def wrap_lora(
    model: Any,
    lora_config: LoraConfig,
    *,
    quantization: str | None,
    gradient_checkpointing: bool,
) -> Any:
    peft_model = get_peft_model(model, lora_config)
    if gradient_checkpointing:
        peft_model.enable_input_require_grads()
    quantized = (
        quantization is not None
        or getattr(peft_model, "is_loaded_in_4bit", False)
        or getattr(peft_model, "is_loaded_in_8bit", False)
    )
    if quantized:
        for param in peft_model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.bfloat16)
    return peft_model


def resume_state(output_dir: str, run_name: str) -> ResumeState:
    last_checkpoint = find_last_checkpoint(output_dir)
    if last_checkpoint:
        console.print(f"  Resuming from checkpoint [bold]{last_checkpoint}[/bold]")
    run_id, is_new = resume_mlflow_run(output_dir, run_name)
    return ResumeState(last_checkpoint=last_checkpoint, run_id=run_id, is_new=is_new)


def trainer_arguments(
    train_cfg: dict[str, Any],
    *,
    output_dir: str,
    run_name: str,
    has_validation: bool,
    debug: bool,
    use_bf16: bool,
) -> dict[str, Any]:
    return {
        "output_dir": output_dir,
        "per_device_train_batch_size": train_cfg["batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation"],
        "num_train_epochs": 1 if debug else train_cfg["max_epochs"],
        "learning_rate": train_cfg["learning_rate"],
        "lr_scheduler_type": train_cfg.get("lr_schedule", "cosine"),
        "warmup_steps": train_cfg["warmup_steps"],
        "bf16": use_bf16,
        "fp16": False,
        "optim": "adamw_torch",
        "eval_strategy": "steps" if has_validation else "no",
        "eval_steps": train_cfg["eval_steps"] if has_validation else None,
        "save_strategy": "steps",
        "save_steps": train_cfg["save_steps"],
        "save_total_limit": train_cfg.get("save_total_limit"),
        "load_best_model_at_end": has_validation,
        "metric_for_best_model": "eval_loss" if has_validation else None,
        "greater_is_better": False,
        "report_to": "mlflow",
        "logging_steps": train_cfg["logging_steps"],
        "max_grad_norm": train_cfg["max_grad_norm"],
        "run_name": run_name,
    }


class EarlyStopCallback(TrainerCallback):
    def __init__(self, stopper: EarlyStopping) -> None:
        super().__init__()
        self.stopper = stopper

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
        if self.stopper.step(val_loss, epoch=epoch):
            control.should_training_stop = True
            console.print(
                f"[yellow]Early stopping at step {state.global_step} "
                f"(best loss={self.stopper.best:.4f} @ epoch {self.stopper.best_epoch})[/yellow]"
            )
        if self.stopper.improved:
            entropy = metrics.get("eval_entropy", float("nan"))
            console.print(
                f"[green]New best val loss: {val_loss:.4f} "
                f"(entropy {entropy:.3f} nats) at epoch {epoch}[/green]"
            )


def early_stopping_callback(train_cfg: dict[str, Any]) -> EarlyStopCallback:
    stopper = EarlyStopping(
        patience=train_cfg["early_stopping_patience"],
        min_epochs=train_cfg.get("early_stopping_min_epochs", 10),
        monitor="loss",
    )
    return EarlyStopCallback(stopper)


def finalize_checkpoint(trainer: Any, processor: Any, output_dir: str) -> None:
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    mark_training_complete(output_dir)
