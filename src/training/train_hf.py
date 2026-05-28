"""
HuggingFace LoRA fine-tuning — works on CUDA (Linux/RTX) and MPS (Apple Silicon).

Training input: WVS cultural Q&A text only — no images, no visual SFT.
Device is selected automatically: cuda > mps > cpu.
Large models (e.g. gemma4_31b) use 4-bit QLoRA when quantization: "4bit"
is set in the config and a CUDA device is available.

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
from pathlib import Path

import mlflow
import torch
import yaml
from dotenv import load_dotenv
from peft import LoraConfig, TaskType
from rich.console import Console
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

from src.data.dataset import load_hf_dataset
from src.training.early_stopping import EarlyStopping
from src.utils.device import get_device

load_dotenv()
console = Console()

PROCESSED_DIR = Path("data/processed")
CHECKPOINTS_DIR = Path("checkpoints")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_wvs_paths(culture: str) -> tuple[Path, Path | None]:
    """Return (train_jsonl, val_jsonl) using last 10% of WVS cultural data as val."""
    src = PROCESSED_DIR / culture / "wvs_cultural_anchoring.jsonl"
    if not src.exists():
        raise FileNotFoundError(
            f"WVS data not found: {src}\n"
            f"Run: uv run python src/data/culture_training_data.py --culture {culture}"
        )
    with open(src) as f:
        lines = [l for l in f if l.strip()]
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


def _find_last_checkpoint(output_dir: str) -> str | None:
    """Return the path of the latest HuggingFace checkpoint, or None."""
    checkpoints = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


_MLFLOW_RUN_ID_FILE = "mlflow_run_id.txt"


def _load_mlflow_run_id(output_dir: str) -> str | None:
    path = Path(output_dir) / _MLFLOW_RUN_ID_FILE
    if path.exists():
        return path.read_text().strip() or None
    return None


def _save_mlflow_run_id(output_dir: str, run_id: str) -> None:
    (Path(output_dir) / _MLFLOW_RUN_ID_FILE).write_text(run_id)


def _resume_mlflow_run(output_dir: str, run_name: str):
    """Return (run_id_to_use, is_new) after ensuring the saved run is resumable.

    When a run ends (FINISHED/FAILED/KILLED), MLflow refuses to re-enter it
    via start_run(run_id=...) and silently creates a new one instead.
    We fix this by explicitly resetting the status to RUNNING via the client
    before handing the run_id back to start_run.
    """
    from mlflow.tracking import MlflowClient

    saved_run_id = _load_mlflow_run_id(output_dir)
    if saved_run_id:
        try:
            client = MlflowClient()
            run = client.get_run(saved_run_id)
            if run.info.status != "RUNNING":
                client.update_run(saved_run_id, status="RUNNING")
            console.print(f"  Continuing MLflow run [bold]{saved_run_id}[/bold]")
            return saved_run_id, False
        except Exception as exc:
            console.print(f"  [yellow]Cannot resume MLflow run {saved_run_id} ({exc}) — starting new run[/yellow]")
    return None, True


def preprocess_logits_for_metrics(logits, labels):
    # Reduce full vocab logits to predicted token IDs before accumulation to avoid OOM
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(-1)


def compute_metrics_fn(eval_preds):
    preds, labels = eval_preds
    preds = preds.flatten()
    labels = labels.flatten()
    mask = labels != -100
    f1 = f1_score(labels[mask], preds[mask], average="macro", zero_division=0)
    return {"eval_f1_macro": f1}


def _build_quantization_config(train_cfg: dict, device: str):
    """Return BitsAndBytesConfig for QLoRA when requested, else None."""
    if train_cfg.get("quantization") != "4bit" or device != "cuda":
        return None
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def train(cfg: dict, model_name: str, culture: str, debug: bool) -> None:
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    device = get_device()
    console.print(f"Device    : [bold]{device}[/bold]")
    console.print(f"Loading model [bold]{model_cfg['id']}[/bold]...")

    quant_cfg = _build_quantization_config(train_cfg, device)
    if quant_cfg is not None:
        console.print("  [yellow]4-bit QLoRA active (bitsandbytes)[/yellow]")

    # flash_attention_2 on CUDA if installed, eager otherwise
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2" if device == "cuda" else "eager"
    except ImportError:
        attn_impl = "eager"

    dtype_str = model_cfg.get("dtype", "bfloat16")
    torch_dtype = torch.bfloat16 if dtype_str != "auto" else "auto"

    model = AutoModelForImageTextToText.from_pretrained(
        model_cfg["id"],
        quantization_config=quant_cfg,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    # Explicit placement only when not using device_map="auto"
    if device != "cuda":
        model = model.to(device)

    model.enable_input_require_grads()
    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    processor = AutoProcessor.from_pretrained(model_cfg["id"], trust_remote_code=True)
    processor.tokenizer.model_max_length = train_cfg["max_seq_len"]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        exclude_modules=lora_cfg.get("exclude_modules", ".*(vision_tower|audio_tower).*"),
        bias="none",
    )

    train_jsonl, val_jsonl = get_wvs_paths(culture)
    train_dataset = load_hf_dataset(train_jsonl)
    val_dataset = load_hf_dataset(val_jsonl) if val_jsonl and val_jsonl.exists() else None

    output_dir = str(CHECKPOINTS_DIR / culture / model_name / "cultural")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    last_checkpoint = _find_last_checkpoint(output_dir)
    if last_checkpoint:
        console.print(f"  Resuming from checkpoint [bold]{last_checkpoint}[/bold]")

    tracking_uri = cfg.get("mlflow", {}).get("tracking_uri", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg["mlflow"]["experiment"]

    run_name = f"{model_name}_{culture}_cultural"
    resume_run_id, is_new = _resume_mlflow_run(output_dir, run_name)
    with mlflow.start_run(run_id=resume_run_id, run_name=run_name if is_new else None) as active_run:
        if is_new:
            _save_mlflow_run_id(output_dir, active_run.info.run_id)
            console.print(f"  New MLflow run [bold]{active_run.info.run_id}[/bold]")

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
            metric_for_best_model="eval_f1_macro" if val_dataset else None,
            greater_is_better=True,
            report_to="mlflow",
            logging_steps=train_cfg["logging_steps"],
            max_grad_norm=train_cfg["max_grad_norm"],
            run_name=run_name,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=peft_config,
            compute_metrics=compute_metrics_fn if val_dataset else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics if val_dataset else None,
        )

        stopper = EarlyStopping(
            patience=train_cfg["early_stopping_patience"],
            min_epochs=train_cfg.get("early_stopping_min_epochs", 10),
            monitor="f1",
        )

        class _EarlyStopCallback(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics, **kwargs):
                val_f1 = metrics.get("eval_f1_macro", 0.0)
                epoch = int(state.epoch or 0)
                if stopper.step(val_f1, epoch=epoch):
                    control.should_training_stop = True
                    console.print(
                        f"[yellow]Early stopping at step {state.global_step} "
                        f"(best F1={stopper.best:.4f} @ epoch {stopper.best_epoch})[/yellow]"
                    )
                if stopper.improved:
                    console.print(f"[green]New best val F1: {val_f1:.4f} at epoch {epoch}[/green]")

        trainer.add_callback(_EarlyStopCallback())
        trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
        console.print(f"[bold green]✓ Saved to {output_dir}[/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="HuggingFace LoRA training for culture-mllm (CUDA + MPS).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--culture", required=True)
    parser.add_argument("--debug", action="store_true", help="1 epoch only")
    args = parser.parse_args()

    console.rule("[bold blue]culture-mllm HF Training — WVS text only[/bold blue]")
    cfg = load_config(args.config)
    model_name = Path(args.config).stem

    console.print(f"Model     : {cfg['model']['id']}")
    console.print(f"Culture   : {args.culture}")
    console.print(f"Debug     : {args.debug}\n")

    train(cfg, model_name, args.culture, args.debug)


if __name__ == "__main__":
    main()
