"""First-token forward-KL LoRA trainer for US subpopulation distributions (Suh et al. 2025)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
from datasets import Dataset
from rich.console import Console
from transformers import EvalPrediction, TrainingArguments

from src.data.distributional_data import DistributionalCollator, pad_token_id_for
from src.data.subpop_data import (
    ANSWER_STUB,
    CONDITION,
    DATA_FORMAT,
    DATASET_REVISION,
    MAX_OPTIONS,
    PSEUDO_CULTURE,
    RAW_DIR,
    SOURCE_COMMIT,
    SPLIT_SEED,
    STEERING_TYPE,
    VAL_RATIO,
    SubpopEncoder,
    filter_related_items,
    load_records,
    load_steering,
    related_qkeys,
    split_by_question,
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
from src.training.train_distributional import (
    DEBUG_CONDITION_SUFFIX,
    DEBUG_EVAL_STEPS,
    DEBUG_TRAIN_ROWS,
    DEBUG_VALID_ROWS,
    DistributionalTrainer,
    assert_base_elicits_options,
    exact_kl,
    exact_kl_array,
    option_entropy,
    predicted_distribution,
)
from src.utils.model_loading import processor_tokenizer

console = Console()

LOSS_NAME = "first_token_forward_kl_over_ten_letters"


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
        "option_mass": float(mass.mean()),
    }


def load_training_splits(
    encoder: SubpopEncoder,
    *,
    raw_dir: Path = RAW_DIR,
    val_ratio: float = VAL_RATIO,
    seed: int = SPLIT_SEED,
    exclude_related: bool,
    debug: bool,
) -> tuple[Dataset, Dataset, dict[str, Any]]:
    records = load_records("train", raw_dir)
    dropped = 0
    if exclude_related:
        records, dropped = filter_related_items(records)
    train_records, valid_records = split_by_question(records, val_ratio=val_ratio, seed=seed)
    if debug:
        train_records = train_records[:DEBUG_TRAIN_ROWS]
        valid_records = valid_records[:DEBUG_VALID_ROWS]
    in_train = related_qkeys(train_records)
    stats: dict[str, Any] = {
        "train_rows": len(train_records),
        "val_rows": len(valid_records),
        "train_questions": len({record.qkey for record in train_records}),
        "val_questions": len({record.qkey for record in valid_records}),
        "val_ratio": val_ratio,
        "split_seed": seed,
        "exclude_related_items": exclude_related,
        "dropped_rows": dropped,
        "related_questions_in_train": ",".join(
            f"{label}={len(qkeys)}" for label, qkeys in in_train.items()
        ),
        "dataset_revision": DATASET_REVISION,
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
    encoder = SubpopEncoder.build(
        tokenizer, load_steering(RAW_DIR), max_length=int(train_cfg["max_seq_len"])
    )
    console.print(f"  Option tokens: {dict(encoder.letter_ids)}")

    train_dataset, val_dataset, data_stats = load_training_splits(
        encoder,
        val_ratio=float(data_cfg.get("val_ratio", VAL_RATIO)),
        seed=int(data_cfg.get("split_seed", SPLIT_SEED)),
        exclude_related=bool(data_cfg.get("exclude_related_items", False)),
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
                "prompt_answer_stub": ANSWER_STUB,
                "option_letters": "".join(letter.strip() for letter in encoder.letter_ids),
                "steering": STEERING_TYPE,
                "loss": LOSS_NAME,
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
            data_collator=DistributionalCollator(
                pad_token_id_for(tokenizer), max_options=MAX_OPTIONS
            ),
            processing_class=backbone.processor,
            compute_metrics=compute_metrics,
            steps_per_epoch=math.ceil(len(train_dataset) / effective_batch),
            loss_fn=exact_kl,
        )
        trainer.add_callback(early_stopping_callback(train_cfg))

        kl, ratio, mass = assert_base_elicits_options(trainer, model, stub=ANSWER_STUB)
        mlflow.log_metric("base_health_kl", kl)
        mlflow.log_metric("base_health_kl_over_uniform", ratio)
        mlflow.log_metric("base_health_option_mass", mass)

        trainer.train(resume_from_checkpoint=state.last_checkpoint)
        finalize_checkpoint(trainer, backbone.processor, output_dir)
        console.print(f"[bold green]✓ Saved to {output_dir}[/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="First-token forward-KL LoRA training on SubPOP subpopulation distributions."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--debug", action="store_true", help="one epoch over a small slice, into a debug leaf"
    )
    args = parser.parse_args()

    console.rule("[bold blue]culture-mllm SubPOP LoRA Training[/bold blue]")
    cfg = load_config(args.config)
    model_name = get_model_key(cfg, args.config)

    console.print(f"Model     : {cfg['model']['id']}")
    console.print(f"Culture   : {PSEUDO_CULTURE} (22 US subpopulations through the prompt)")
    console.print(f"Debug     : {args.debug}\n")

    train(cfg, model_name, args.debug)


if __name__ == "__main__":
    main()
