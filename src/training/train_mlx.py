"""
MLX LoRA fine-tuning for culture-mllm (primary backend for Apple Silicon).

Training input: WVS cultural Q&A text only (from CultureLLM).
No images, no visual SFT — cultural knowledge is baked into the text path of the VLM.

Conditions:
  cultural  — WVS examples with culture-specific system prompt  (wvs_cultural_anchoring.jsonl)
  baseline  — same WVS examples with neutral system prompt      (wvs_baseline_anchoring.jsonl)

The fine-tuned adapters are used for image annotation in Stage 3.
Annotation outputs are evaluated against σ₃P₅ labels in Stage 4.

NOTE: MLX is Apple Silicon only. On CUDA machines, use train_hf.py instead,
or let scripts/02_train_culture_models.sh handle the fallback automatically.

Usage:
    uv run python src/training/train_mlx.py \\
        --config configs/qwen3_5_2b.yaml \\
        --culture arabic \\
        --condition cultural

    # Smoke test (10 steps):
    uv run python src/training/train_mlx.py \\
        --config configs/qwen3_5_2b.yaml \\
        --culture arabic \\
        --condition cultural \\
        --debug
"""

import argparse
import os
import subprocess
import threading
import re
from pathlib import Path

import mlflow
import yaml
from dotenv import load_dotenv
from rich.console import Console

from src.training.early_stopping import EarlyStopping

load_dotenv()
console = Console()

PROCESSED_DIR = Path("data/processed")
CHECKPOINTS_DIR = Path("checkpoints")


def _mlx_available() -> bool:
    """Return True if the mlx package is importable (Apple Silicon only).

    Returns:
        bool: True if MLX is available, False otherwise.
    """
    try:
        import mlx  # noqa: F401
        return True
    except ImportError:
        return False


def load_config(config_path: str) -> dict:
    """Load a YAML training config.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Parsed configuration.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_wvs_paths(culture: str, condition: str) -> tuple[Path, Path | None]:
    """Split WVS anchoring data into train/val files (90/10 split).

    Args:
        culture (str): Culture name (e.g., "arabic").
        condition (str): "cultural" or "baseline" — selects the source JSONL.

    Returns:
        tuple: (train_jsonl, val_jsonl) Path objects.

    Raises:
        FileNotFoundError: If the source WVS JSONL does not exist.
    """
    culture_dir = PROCESSED_DIR / culture
    filename = "wvs_cultural_anchoring.jsonl" if condition == "cultural" else "wvs_baseline_anchoring.jsonl"
    src = culture_dir / filename

    if not src.exists():
        raise FileNotFoundError(
            f"WVS data not found: {src}\n"
            f"Run: uv run python src/data/culture_training_data.py --culture {culture}"
        )

    import json
    with open(src) as f:
        lines = [l for l in f if l.strip()]

    n_val = max(1, len(lines) // 10)
    train_lines, val_lines = lines[:-n_val], lines[-n_val:]

    suffix = "cultural" if condition == "cultural" else "baseline"
    train_path = culture_dir / f"wvs_{suffix}_train.jsonl"
    val_path = culture_dir / f"wvs_{suffix}_val.jsonl"

    with open(train_path, "w") as f:
        f.writelines(train_lines)
    with open(val_path, "w") as f:
        f.writelines(val_lines)

    console.print(
        f"  WVS split: {len(train_lines)} train / {len(val_lines)} val "
        f"({filename})"
    )
    return train_path, val_path


def _prepare_data_dir(train_jsonl: Path, val_jsonl: Path | None) -> Path:
    """Create the mlx_lm data directory with symlinks to train/valid JSONL files.

    mlx_lm.lora requires a directory with train.jsonl and valid.jsonl rather
    than explicit file paths.

    Args:
        train_jsonl (Path): Training data JSONL.
        val_jsonl (Path | None): Validation data JSONL, or None to skip valid.jsonl.

    Returns:
        Path: The prepared data directory.
    """
    data_dir = train_jsonl.parent / f"mlx_data_{train_jsonl.stem}"
    data_dir.mkdir(exist_ok=True)
    for link_name, src in [("train.jsonl", train_jsonl), ("valid.jsonl", val_jsonl)]:
        link = data_dir / link_name
        if link.exists() or link.is_symlink():
            link.unlink()
        if src and src.exists():
            link.symlink_to(src.resolve())
    return data_dir


def run_mlx_lora(
    cfg: dict,
    train_jsonl: Path,
    val_jsonl: Path | None,
    checkpoint_dir: Path,
    debug: bool = False,
    mlflow_run_id: str | None = None,
) -> None:
    """Launch mlx_lm.lora as a subprocess and stream metrics to MLflow.

    Parses stdout with regex to extract train/val loss and logs each step via
    the MLflow client. Early stopping is evaluated on every val loss line.
    A background thread handles the stdout stream so proc.wait() doesn't block.

    Args:
        cfg (dict): Parsed YAML config.
        train_jsonl (Path): Training data path.
        val_jsonl (Path | None): Validation data path, or None to skip validation.
        checkpoint_dir (Path): Directory where adapters are saved.
        debug (bool, optional): If True, run only 10 iterations. Defaults to False.
        mlflow_run_id (str | None, optional): Active MLflow run ID for metric logging.

    Raises:
        RuntimeError: If mlx_lm.lora exits with a non-zero, non-SIGTERM code.
    """
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_dir = _prepare_data_dir(train_jsonl, val_jsonl)

    n_lines = sum(1 for _ in open(train_jsonl))
    iters_per_epoch = max(n_lines // train_cfg["batch_size"], 1)
    max_iters = iters_per_epoch * train_cfg.get("max_epochs", 250)

    cmd = [
        "mlx_lm.lora",
        "--model", model_cfg["mlx_id"],
        "--train",
        "--data", str(data_dir),
        "--num-layers", str(lora_cfg["layers"]),
        "--batch-size", str(train_cfg["batch_size"]),
        "--grad-accumulation-steps", str(train_cfg["gradient_accumulation"]),
        "--grad-checkpoint",
        "--learning-rate", str(train_cfg["learning_rate"]),
        "--iters", "10" if debug else str(max_iters),
        "--steps-per-report", str(train_cfg["logging_steps"]),
        "--steps-per-eval", str(train_cfg["eval_steps"]),
        "--save-every", str(train_cfg["save_steps"]),
        "--adapter-path", str(checkpoint_dir),
        "--max-seq-length", str(train_cfg["max_seq_len"]),
        "--seed", "42",
    ]
    if val_jsonl and val_jsonl.exists():
        cmd += ["--val-batches", "50"]

    console.print(f"[blue]Running MLX LoRA:[/blue]\n  {' '.join(cmd)}\n")

    # Matches mlx_lm.lora stdout: "Iter N: Train loss X.XX, It/sec Y.YY"
    _train_re = re.compile(r"Iter (\d+): Train loss ([\d.]+).*?It/sec ([\d.]+)")
    # Matches mlx_lm.lora stdout: "Iter N: Val loss X.XX"
    _val_re = re.compile(r"Iter (\d+): Val loss ([\d.]+)")
    mlflow_client = mlflow.MlflowClient() if mlflow_run_id else None

    stopper = EarlyStopping(
        patience=train_cfg.get("early_stopping_patience", 15),
        min_epochs=train_cfg.get("early_stopping_min_epochs", 10),
        monitor="loss",
    )
    stopped_early = False

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    def _stream():
        nonlocal stopped_early
        for line in proc.stdout:
            console.print(line, end="")
            if not mlflow_client and not stopper:
                continue

            m = _train_re.search(line)
            if m and mlflow_client:
                step = int(m.group(1))
                try:
                    mlflow_client.log_metric(mlflow_run_id, "train_loss", float(m.group(2)), step=step)
                    mlflow_client.log_metric(mlflow_run_id, "it_per_sec", float(m.group(3)), step=step)
                except Exception:
                    pass
                continue

            m = _val_re.search(line)
            if m:
                step = int(m.group(1))
                val_loss = float(m.group(2))
                if mlflow_client:
                    try:
                        mlflow_client.log_metric(mlflow_run_id, "val_loss", val_loss, step=step)
                    except Exception:
                        pass

                if not debug:
                    epoch = step // iters_per_epoch
                    if stopper.step(val_loss, epoch=epoch):
                        console.print(
                            f"[yellow]MLX early stop at step {step} "
                            f"(best_loss={stopper.best:.4f} @ epoch {stopper.best_epoch})[/yellow]"
                        )
                        stopped_early = True
                        proc.terminate()
                        return
                    if stopper.improved:
                        console.print(f"[green]New best val loss: {val_loss:.4f} at step {step}[/green]")

    t = threading.Thread(target=_stream, daemon=True)
    t.start()
    proc.wait()
    t.join()

    # returncode -15 is SIGTERM from our own proc.terminate() — not an error
    if proc.returncode not in (0, -15):
        raise RuntimeError(f"mlx_lm.lora exited with code {proc.returncode}")

    if mlflow_client and stopped_early:
        try:
            mlflow_client.log_param(mlflow_run_id, "early_stopped", "true")
            mlflow_client.log_metric(mlflow_run_id, "best_val_loss", stopper.best)
        except Exception:
            pass


def _patch_adapter_config_for_vlm(checkpoint_dir: Path, cfg: dict) -> None:
    """Rewrite adapter_config.json so mlx_vlm can load it.

    mlx_lm.lora writes a verbose training config. mlx_vlm.apply_lora_layers
    passes the entire file as **kwargs to get_peft_model — it must contain
    ONLY {rank, alpha, dropout}. The original training config is preserved
    as mlx_lm_training_config.json.
    """
    import json
    for config_path in sorted(checkpoint_dir.glob("**/adapter_config.json")):
        with open(config_path) as f:
            full_cfg = json.load(f)

        backup_path = config_path.parent / "mlx_lm_training_config.json"
        if not backup_path.exists():
            with open(backup_path, "w") as f:
                json.dump(full_cfg, f, indent=2)

        lora_p = full_cfg.get("lora_parameters", {})
        rank = lora_p.get("rank", cfg["lora"]["r"])
        scale = lora_p.get("scale", float(cfg["lora"]["alpha"]) / rank)
        vlm_cfg = {
            "rank": rank,
            "alpha": int(scale * rank),
            "dropout": lora_p.get("dropout", cfg["lora"]["dropout"]),
        }
        with open(config_path, "w") as f:
            json.dump(vlm_cfg, f, indent=2)


def main() -> None:
    """Entry point: parse CLI args, set up MLflow, and run MLX LoRA training."""
    if not _mlx_available():
        raise RuntimeError(
            "MLX is not available on this machine.\n"
            "On CUDA machines, use train_hf.py instead, or let "
            "scripts/02_train_culture_models.sh handle the fallback automatically."
        )

    parser = argparse.ArgumentParser(description="MLX LoRA fine-tuning for culture-mllm (WVS text only).")
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    parser.add_argument("--culture", required=True, help="Culture name (e.g. arabic)")
    parser.add_argument("--condition", choices=["cultural", "baseline"], default="cultural")
    parser.add_argument("--debug", action="store_true", help="Run only 10 steps (smoke test)")
    args = parser.parse_args()

    console.rule("[bold blue]culture-mllm MLX Training — WVS text only[/bold blue]")
    cfg = load_config(args.config)
    model_name = Path(args.config).stem

    console.print(f"Config    : {args.config}")
    console.print(f"Model     : {cfg['model']['id']}")
    console.print(f"Culture   : {args.culture}")
    console.print(f"Condition : {args.condition}")
    console.print(f"Debug     : {args.debug}\n")

    train_jsonl, val_jsonl = get_wvs_paths(args.culture, args.condition)
    checkpoint_dir = CHECKPOINTS_DIR / args.culture / model_name / args.condition

    run_name = f"{model_name}_{args.culture}_{args.condition}"
    tracking_uri = cfg.get("mlflow", {}).get("tracking_uri", "http://127.0.0.1:5000")

    active = None
    mlflow_client = None
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        active = mlflow.start_run(run_name=run_name)
        run_id = active.info.run_id
        mlflow_client = mlflow.MlflowClient()
        mlflow_client.set_tag(run_id, "culture", args.culture)
        mlflow_client.set_tag(run_id, "condition", args.condition)
        mlflow_client.set_tag(run_id, "model_name", model_name)
        mlflow_client.set_tag(run_id, "debug", str(args.debug))
        mlflow_client.log_param(run_id, "model_id", cfg["model"]["id"])
        mlflow_client.log_param(run_id, "mlx_id", cfg["model"]["mlx_id"])
        mlflow_client.log_param(run_id, "lora_r", cfg["lora"]["r"])
        mlflow_client.log_param(run_id, "lora_layers", cfg["lora"]["layers"])
        mlflow_client.log_param(run_id, "learning_rate", cfg["training"]["learning_rate"])
        mlflow_client.log_param(run_id, "culture", args.culture)
        mlflow_client.log_param(run_id, "condition", args.condition)
        mlflow_client.log_param(run_id, "training_data", "wvs_text_only")
    except Exception as exc:
        console.print(f"[yellow]MLflow init skipped: {exc}[/yellow]")
        run_id = None

    try:
        run_mlx_lora(cfg, train_jsonl, val_jsonl, checkpoint_dir, debug=args.debug, mlflow_run_id=run_id)
    finally:
        if active and mlflow_client and run_id:
            try:
                mlflow_client.log_param(run_id, "checkpoint_dir", str(checkpoint_dir))
                mlflow_client.set_tag(run_id, "status", "complete")
                mlflow.end_run()
            except Exception:
                pass

    _patch_adapter_config_for_vlm(checkpoint_dir, cfg)
    console.print(f"\n[bold green]✓ Checkpoint saved to {checkpoint_dir}[/bold green]")
    if active:
        exp_id = active.info.experiment_id
        console.print(f"[green]MLflow:[/green] {tracking_uri}/#/experiments/{exp_id}/runs/{run_id}")


if __name__ == "__main__":
    main()
