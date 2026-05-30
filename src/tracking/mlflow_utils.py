"""MLflow experiment tracking utilities for CultureVLM."""

import os
from contextlib import contextmanager

import mlflow


def setup_mlflow(tracking_uri: str | None = None, experiment: str | None = None) -> None:
    """Configure MLflow tracking URI and experiment, falling back to env vars.

    Args:
        tracking_uri (str | None, optional): MLflow server URI. Defaults to MLFLOW_TRACKING_URI env var.
        experiment (str | None, optional): Experiment name. Defaults to MLFLOW_EXPERIMENT_TRAINING env var.
    """
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    exp = experiment or os.getenv("MLFLOW_EXPERIMENT_TRAINING", "culture_mllm_training")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)


@contextmanager
def training_run(model_name: str, culture: str, condition: str, fold: int | str):
    """Context manager for a single (model, culture, condition, fold) training run.

    Args:
        model_name (str): Model identifier.
        culture (str): Culture name.
        condition (str): Training condition ("cultural" or "baseline").
        fold (int | str): Fold index or label.

    Yields:
        mlflow.ActiveRun: The active MLflow run.
    """
    run_name = f"{model_name}_{culture}_{condition}_fold{fold}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "model": model_name,
            "culture": culture,
            "condition": condition,
            "fold": str(fold),
            "stage": "training",
        })
        yield run


def log_training_params(cfg: dict, culture: str, condition: str, fold: int) -> None:
    mlflow.log_params({
        "model_id": cfg["model"]["id"],
        "backend": cfg["model"]["backend"],
        "lora_r": cfg["lora"]["r"],
        "lora_alpha": cfg["lora"]["alpha"],
        "learning_rate": cfg["training"]["learning_rate"],
        "batch_size": cfg["training"]["batch_size"],
        "gradient_accumulation": cfg["training"]["gradient_accumulation"],
        "warmup_steps": cfg["training"]["warmup_steps"],
        "early_stopping_patience": cfg["training"]["early_stopping_patience"],
        "culture": culture,
        "condition": condition,
        "fold": fold,
    })


def log_training_step(step: int, train_loss: float, val_loss: float | None = None,
                      val_f1: float | None = None, val_mae: float | None = None) -> None:
    metrics = {"train_loss": train_loss}
    if val_loss is not None:
        metrics["val_loss"] = val_loss
    if val_f1 is not None:
        metrics["val_f1_macro"] = val_f1
    if val_mae is not None:
        metrics["val_mae"] = val_mae
    mlflow.log_metrics(metrics, step=step)


def log_eval_results(metrics: dict, culture: str | None = None) -> None:
    prefix = f"{culture}/" if culture else ""
    mlflow.log_metrics({f"{prefix}{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})


def log_artifacts(paths: list[str]) -> None:
    for path in paths:
        try:
            mlflow.log_artifact(path)
        except Exception as exc:
            print(f"[mlflow] Could not log artifact {path}: {exc}")
