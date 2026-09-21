"""HuggingFace LoRA fine-tuning on WVS cultural Q&A text, one adapter per (model, culture)."""

import argparse
from pathlib import Path
from typing import Any

import mlflow
import torch
from dotenv import load_dotenv
from rich.console import Console
from trl import SFTConfig, SFTTrainer  # type: ignore[attr-defined]

from src.data.dataset import load_hf_dataset
from src.training.common import (
    build_lora_config,
    checkpoint_output_dir,
    configure_mlflow,
    default_exclude_modules,
    early_stopping_callback,
    finalize_checkpoint,
    get_model_key,
    is_training_complete,
    load_backbone,
    load_config,
    resume_state,
    save_mlflow_run_id,
    trainer_arguments,
)
from src.utils.model_loading import processor_tokenizer

__all__ = [
    "checkpoint_output_dir",
    "compute_metrics_fn",
    "default_exclude_modules",
    "get_data_paths",
    "get_model_key",
    "get_wvs_paths",
    "load_config",
    "load_training_datasets",
    "main",
    "preprocess_logits_for_metrics",
    "train",
]

load_dotenv()
console = Console()

PROCESSED_DIR = Path("data/processed")


def get_wvs_paths(culture: str) -> tuple[Path, Path | None]:
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
    data_cfg = cfg.get("data")
    if not data_cfg:
        return get_wvs_paths(culture)
    train = Path(data_cfg["train_jsonl"])
    if not train.exists():
        raise FileNotFoundError(f"Training data not found: {train}")
    val = Path(data_cfg["val_jsonl"]) if data_cfg.get("val_jsonl") else None
    return train, (val if val and val.exists() else None)


def load_training_datasets(cfg: dict[str, Any], culture: str) -> tuple[Any, Any, dict[str, int]]:
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


def preprocess_logits_for_metrics(logits: Any, labels: Any) -> torch.Tensor:
    if isinstance(logits, tuple):
        logits = logits[0]
    logp = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def compute_metrics_fn(eval_preds: Any) -> dict[str, float]:
    entropies, labels = eval_preds
    ent = entropies[:, :-1]
    lab = labels[:, 1:]
    mask = lab != -100
    return {"entropy": float(ent[mask].mean())}


def _assert_base_is_sane(trainer: Any, model: Any, processor: Any) -> tuple[float, float]:
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
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    condition = cfg.get("condition", "cultural")
    output_dir = str(checkpoint_output_dir(culture, model_name, condition))
    run_name = f"{model_name}_{culture}_{condition}"

    _, experiment_name = configure_mlflow(cfg)

    if is_training_complete(output_dir, run_name, experiment_name):
        console.print(f"[dim]Skipping {model_name}/{culture} — already FINISHED in MLflow[/dim]")
        return

    backbone = load_backbone(model_cfg, train_cfg)
    model, processor = backbone.model, backbone.processor
    peft_config = build_lora_config(lora_cfg, backbone.modality)

    train_dataset, val_dataset, data_stats = load_training_datasets(cfg, culture)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    state = resume_state(output_dir, run_name)
    with mlflow.start_run(
        run_id=state.run_id, run_name=run_name if state.is_new else None
    ) as active_run:
        if state.is_new:
            save_mlflow_run_id(output_dir, active_run.info.run_id)
            console.print(f"  New MLflow run [bold]{active_run.info.run_id}[/bold]")

        mlflow.log_params({f"data_{key}": value for key, value in data_stats.items()})

        training_args = SFTConfig(
            **trainer_arguments(
                train_cfg,
                output_dir=output_dir,
                run_name=run_name,
                has_validation=bool(val_dataset),
                debug=debug,
                use_bf16=backbone.device in ("cuda", "mps"),
            ),
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

        trainer.add_callback(early_stopping_callback(train_cfg))

        base_loss, base_ratio = _assert_base_is_sane(trainer, model, processor)
        mlflow.log_metric("base_health_loss", base_loss)
        mlflow.log_metric("base_health_loss_over_guess", base_ratio)

        trainer.train(resume_from_checkpoint=state.last_checkpoint)
        finalize_checkpoint(trainer, processor, output_dir)
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
