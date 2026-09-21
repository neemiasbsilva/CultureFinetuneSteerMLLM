"""First-token KL LoRA trainer for country-level WVS response distributions (Cao et al. 2025)."""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
import torch
from datasets import Dataset
from numpy.typing import NDArray
from rich.console import Console
from scipy.special import logsumexp, xlogy
from torch.optim.lr_scheduler import LambdaLR
from transformers import EvalPrediction, Trainer, TrainingArguments

from src.data.distributional_data import (
    ANSWER_PREFIX,
    CONDITION,
    DATA_FORMAT,
    EVALUATION_ITEMS,
    PSEUDO_CULTURE,
    RAW_DIR,
    SOURCE_COMMIT,
    DistributionalCollator,
    PromptEncoder,
    count_items,
    filter_evaluation_items,
    load_split,
    pad_token_id_for,
)
from src.training.common import (
    build_lora_config,
    checkpoint_output_dir,
    configure_mlflow,
    early_stopping_callback,
    finalize_checkpoint,
    get_model_key,
    is_training_complete,
    load_backbone,
    load_config,
    resume_state,
    save_mlflow_run_id,
    trainer_arguments,
    wrap_lora,
)
from src.utils.model_loading import processor_tokenizer

console = Console()

MIN_BASE_OPTION_MASS = 0.01
WARN_BASE_OPTION_MASS = 0.10
STEP_EPOCH_SCHEDULE = "step_epoch"
DEBUG_TRAIN_ROWS = 128
DEBUG_VALID_ROWS = 64
DEBUG_EVAL_STEPS = 8
DEBUG_CONDITION_SUFFIX = "_debug"

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def answer_positions(answer_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.unique(answer_index)
    return positions, torch.searchsorted(positions, answer_index)


def forward_answer_logits(model: Any, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    positions, columns = answer_positions(batch["answer_index"])
    try:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            logits_to_keep=positions,
        )
    except TypeError as exc:
        if "logits_to_keep" not in str(exc):
            raise
        raise RuntimeError(
            "This backbone's forward does not accept `logits_to_keep`, so its "
            "language-model head cannot be restricted to the answer positions. "
            "Full-vocabulary logits for one batch cost gigabytes once autocast "
            "upcasts them, so the objective is not runnable on it as written."
        ) from exc
    rows = torch.arange(outputs.logits.shape[0], device=outputs.logits.device)
    return cast(torch.Tensor, outputs.logits[rows, columns].float())


def gather_option_logits(answer: torch.Tensor, option_token_ids: torch.Tensor) -> torch.Tensor:
    return answer.gather(1, option_token_ids)


def option_log_probs(option_logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    masked = option_logits.masked_fill(~valid, float("-inf"))
    return option_logits - torch.logsumexp(masked, dim=-1, keepdim=True)


def kl_terms(option_logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    valid = target_dist >= 0
    target = target_dist.clamp_min(0.0)
    log_predicted = option_log_probs(option_logits, valid)
    return (torch.xlogy(target, target) - target * log_predicted).sum(dim=-1)


def option_count(target_dist: torch.Tensor) -> torch.Tensor:
    return (target_dist >= 0).sum(dim=-1).clamp_min(1)


def kl_loss(option_logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    per_record = kl_terms(option_logits, target_dist) / option_count(target_dist).to(
        option_logits.dtype
    )
    return per_record.mean()


def exact_kl(option_logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    return kl_terms(option_logits, target_dist).mean()


def uniform_kl(target_dist: torch.Tensor) -> torch.Tensor:
    target = target_dist.clamp_min(0.0)
    entropy = -torch.xlogy(target, target).sum(dim=-1)
    return torch.log(option_count(target_dist).float()) - entropy


def option_mass(
    answer: torch.Tensor, option_token_ids: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    probabilities = torch.softmax(answer, dim=-1)
    return (probabilities.gather(1, option_token_ids) * valid).sum(dim=-1)


def predicted_distribution(option_logits: FloatArray, valid: BoolArray) -> FloatArray:
    logits = np.where(valid, option_logits, -np.inf)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    weights = np.where(valid, np.exp(shifted), 0.0)
    return np.asarray(weights / weights.sum(axis=-1, keepdims=True), dtype=np.float64)


def option_entropy(predicted: FloatArray, valid: BoolArray) -> FloatArray:
    return np.asarray(-np.where(valid, xlogy(predicted, predicted), 0.0).sum(axis=-1))


def ordinal_emd(predicted: FloatArray, target: FloatArray, valid: BoolArray) -> FloatArray:
    predicted_mass = np.where(valid, predicted, 0.0)
    target_mass = np.where(valid, target, 0.0)
    gaps = np.abs(np.cumsum(predicted_mass, axis=-1) - np.cumsum(target_mass, axis=-1))
    cuts = np.maximum(valid.sum(axis=-1) - 1, 1)
    return np.asarray(np.where(valid, gaps, 0.0).sum(axis=-1) / cuts)


def exact_kl_array(option_logits: FloatArray, target: FloatArray, valid: BoolArray) -> FloatArray:
    logits = np.where(valid, option_logits, -np.inf)
    log_predicted = np.where(valid, option_logits - logsumexp(logits, axis=-1, keepdims=True), 0.0)
    target_mass = np.where(valid, target, 0.0)
    return np.asarray((xlogy(target_mass, target_mass) - target_mass * log_predicted).sum(axis=-1))


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    predictions = eval_pred.predictions
    if not isinstance(predictions, tuple):
        raise ValueError("expected (option_logits, option_mass) predictions from prediction_step")
    parts = cast(tuple[Any, ...], predictions)
    if len(parts) != 2:
        raise ValueError("expected (option_logits, option_mass) predictions from prediction_step")
    option_logits = np.asarray(parts[0], dtype=np.float64)
    mass = np.asarray(parts[1], dtype=np.float64)
    target = np.asarray(eval_pred.label_ids, dtype=np.float64)
    valid = target >= 0
    predicted = predicted_distribution(option_logits, valid)
    return {
        "kl": float(exact_kl_array(option_logits, target, valid).mean()),
        "entropy": float(option_entropy(predicted, valid).mean()),
        "emd": float(ordinal_emd(predicted, target, valid).mean()),
        "option_mass": float(mass.mean()),
    }


def step_epoch_factor(gamma: float, steps_per_epoch: int) -> Callable[[int], float]:
    def factor(step: int) -> float:
        return float(gamma ** (step // steps_per_epoch))

    return factor


def score_batch(model: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
    answer = forward_answer_logits(model, batch)
    option_logits = gather_option_logits(answer, batch["option_token_ids"])
    target = batch["target_dist"]
    loss = kl_loss(option_logits, target)
    with torch.no_grad():
        mass = option_mass(answer.detach(), batch["option_token_ids"], target >= 0)
    return loss, {"option_logits": option_logits.detach(), "option_mass": mass}


class DistributionalTrainer(Trainer):
    def __init__(
        self, *args: Any, steps_per_epoch: int, lr_gamma: float | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.steps_per_epoch = steps_per_epoch
        self.lr_gamma = lr_gamma

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer | None = None
    ) -> torch.optim.lr_scheduler.LRScheduler:
        if self.lr_gamma is None:
            return super().create_scheduler(num_training_steps, optimizer)
        if self.lr_scheduler is None:
            target = optimizer if optimizer is not None else self.optimizer
            if target is None:
                raise RuntimeError("the optimizer must be created before its scheduler")
            self.lr_scheduler = LambdaLR(
                target, step_epoch_factor(self.lr_gamma, self.steps_per_epoch)
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        loss, extras = score_batch(model, inputs)
        return (loss, extras) if return_outputs else loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, Any, torch.Tensor | None]:
        prepared = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, extras = score_batch(model, prepared)
        if prediction_loss_only:
            return loss.detach(), None, None
        return (
            loss.detach(),
            (extras["option_logits"], extras["option_mass"]),
            prepared["target_dist"],
        )


def assert_base_elicits_options(trainer: Any, model: Any) -> tuple[float, float, float]:
    batch = next(iter(trainer.get_train_dataloader()))
    batch = {key: value.to(model.device) for key, value in batch.items()}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            answer = forward_answer_logits(model, batch)
    finally:
        model.train(was_training)

    option_logits = gather_option_logits(answer, batch["option_token_ids"])
    target = batch["target_dist"]
    kl = float(kl_loss(option_logits, target))
    uniform = float((uniform_kl(target) / option_count(target).float()).mean())
    ratio = kl / max(uniform, 1e-12)
    mass = float(option_mass(answer, batch["option_token_ids"], target >= 0).mean())

    if mass < MIN_BASE_OPTION_MASS:
        raise RuntimeError(
            f"\nThe base puts only {mass:.4f} of its next-token mass on the option letters "
            f"after {ANSWER_PREFIX!r} — the letters are not what it emits there.\n"
            "Check the option token ids and the prompt format before training; an "
            "adapter trained from here would learn a format the model never produces."
        )
    colour = "yellow" if mass < WARN_BASE_OPTION_MASS else "green"
    console.print(
        f"[{colour}]  Base health: KL {kl:.4f} vs uniform {uniform:.4f} (ratio {ratio:.2f}), "
        f"option mass {mass:.3f}[/{colour}]"
    )
    return kl, ratio, mass


def load_training_splits(
    encoder: PromptEncoder,
    *,
    raw_dir: Path = RAW_DIR,
    exclude_items: bool,
    debug: bool,
) -> tuple[Dataset, Dataset, dict[str, Any]]:
    train_records = load_split("train", raw_dir)
    valid_records = load_split("valid", raw_dir)
    dropped = 0
    if exclude_items:
        train_records, dropped_train = filter_evaluation_items(train_records)
        valid_records, dropped_valid = filter_evaluation_items(valid_records)
        dropped = sum(dropped_train.values()) + sum(dropped_valid.values())
    if debug:
        train_records = train_records[:DEBUG_TRAIN_ROWS]
        valid_records = valid_records[:DEBUG_VALID_ROWS]
    in_train = count_items(train_records)
    stats: dict[str, Any] = {
        "train_rows": len(train_records),
        "val_rows": len(valid_records),
        "exclude_evaluation_items": exclude_items,
        "excluded_items": ",".join(EVALUATION_ITEMS) if exclude_items else "",
        "dropped_rows": dropped,
        "evaluation_item_rows_in_train": ",".join(
            f"{item}={count}" for item, count in in_train.items()
        ),
        "source_commit": SOURCE_COMMIT,
    }
    return encoder.dataset(train_records), encoder.dataset(valid_records), stats


def train(cfg: dict[str, Any], model_name: str, debug: bool) -> None:
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg.get("data") or {}
    data_format = str(data_cfg.get("format", "")).lower()
    if data_format != DATA_FORMAT:
        raise ValueError(
            f"data.format must be {DATA_FORMAT!r} for this trainer, got {data_format!r}"
        )

    condition = str(cfg.get("condition", CONDITION))
    if debug:
        condition = condition + DEBUG_CONDITION_SUFFIX
    output_dir = str(checkpoint_output_dir(PSEUDO_CULTURE, model_name, condition))
    run_name = f"{model_name}_{PSEUDO_CULTURE}_{condition}"

    _, experiment_name = configure_mlflow(cfg)

    if is_training_complete(output_dir, run_name, experiment_name):
        console.print(f"[dim]Skipping {run_name} — already FINISHED in MLflow[/dim]")
        return

    backbone = load_backbone(model_cfg, train_cfg)
    tokenizer = processor_tokenizer(backbone.processor)
    encoder = PromptEncoder.build(tokenizer, max_length=int(train_cfg["max_seq_len"]))
    console.print(f"  Option tokens: {encoder.letter_ids}")

    train_dataset, val_dataset, data_stats = load_training_splits(
        encoder,
        exclude_items=bool(data_cfg.get("exclude_evaluation_items", False)),
        debug=debug,
    )
    console.print(f"  Records: {data_stats['train_rows']} train / {data_stats['val_rows']} valid")

    model = wrap_lora(
        backbone.model,
        build_lora_config(lora_cfg, backbone.modality),
        quantization=backbone.quantization,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", False)),
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    state = resume_state(output_dir, run_name)
    with mlflow.start_run(
        run_id=state.run_id, run_name=run_name if state.is_new else None
    ) as active_run:
        if state.is_new:
            save_mlflow_run_id(output_dir, active_run.info.run_id)
            console.print(f"  New MLflow run [bold]{active_run.info.run_id}[/bold]")

        mlflow.log_params({f"data_{key}": value for key, value in data_stats.items()})
        mlflow.log_params(
            {
                "prompt_answer_prefix": ANSWER_PREFIX,
                "option_letters": "".join(encoder.letter_ids),
                "loss": "first_token_kl_over_options_div_k",
            }
        )

        kwargs = trainer_arguments(
            train_cfg,
            output_dir=output_dir,
            run_name=run_name,
            has_validation=True,
            debug=debug,
            use_bf16=backbone.device in ("cuda", "mps"),
        )
        lr_gamma: float | None = None
        if str(train_cfg.get("lr_schedule", "")) == STEP_EPOCH_SCHEDULE:
            kwargs["lr_scheduler_type"] = "constant"
            lr_gamma = float(train_cfg["lr_gamma"])
        if debug:
            kwargs["eval_steps"] = DEBUG_EVAL_STEPS
            kwargs["save_steps"] = DEBUG_EVAL_STEPS
        training_args = TrainingArguments(
            **kwargs, remove_unused_columns=False, label_names=["target_dist"]
        )

        effective_batch = int(train_cfg["batch_size"]) * int(train_cfg["gradient_accumulation"])
        trainer = DistributionalTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DistributionalCollator(pad_token_id_for(tokenizer)),
            processing_class=backbone.processor,
            compute_metrics=compute_metrics,
            steps_per_epoch=math.ceil(len(train_dataset) / effective_batch),
            lr_gamma=lr_gamma,
        )
        trainer.add_callback(early_stopping_callback(train_cfg))

        kl, ratio, mass = assert_base_elicits_options(trainer, model)
        mlflow.log_metric("base_health_kl", kl)
        mlflow.log_metric("base_health_kl_over_uniform", ratio)
        mlflow.log_metric("base_health_option_mass", mass)

        trainer.train(resume_from_checkpoint=state.last_checkpoint)
        finalize_checkpoint(trainer, backbone.processor, output_dir)
        console.print(f"[bold green]✓ Saved to {output_dir}[/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="First-token KL LoRA training on country-level WVS response distributions."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--debug", action="store_true", help="one epoch over a small slice, into a debug leaf"
    )
    args = parser.parse_args()

    console.rule("[bold blue]culture-mllm Distributional LoRA Training[/bold blue]")
    cfg = load_config(args.config)
    model_name = get_model_key(cfg, args.config)

    console.print(f"Model     : {cfg['model']['id']}")
    console.print(f"Culture   : {PSEUDO_CULTURE} (46 countries through the prompt)")
    console.print(f"Debug     : {args.debug}\n")

    train(cfg, model_name, args.debug)


if __name__ == "__main__":
    main()
