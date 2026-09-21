"""Held-out fit of the global distributional adapter against its base on Cao et al.'s test split."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from scipy.stats import wasserstein_distance

from src.data.distributional_data import (
    CONDITION,
    DATA_FORMAT,
    EVALUATION_ITEMS,
    PSEUDO_CULTURE,
    RAW_DIR,
    DistributionalCollator,
    PromptEncoder,
    SurveyRecord,
    dedupe_records,
    load_split,
    pad_token_id_for,
)
from src.training.common import (
    DONE_SENTINEL,
    checkpoint_output_dir,
    configure_mlflow,
    get_model_key,
    load_backbone,
    load_config,
    load_mlflow_run_id,
)
from src.training.train_distributional import (
    DEBUG_CONDITION_SUFFIX,
    FloatArray,
    exact_kl_array,
    forward_answer_logits,
    gather_option_logits,
    option_entropy,
    option_mass,
    ordinal_emd,
    predicted_distribution,
)
from src.utils.model_loading import processor_tokenizer

console = Console()

OUTPUT_ROOT = Path("outputs/evaluation/distributional")
REPORT_FILENAME = "heldout.json"
RECORDS_FILENAME = "records.csv"
COHORTS: dict[str, tuple[str, ...]] = {
    "test_1": ("test_1",),
    "new_country": ("test_2", "test_3", "test_5", "test_6"),
    "new_both": ("test_4", "test_7"),
}
BLOCK_ORDER: tuple[str, ...] = (
    "all",
    *COHORTS,
    *(f"item_{item}" for item in EVALUATION_ITEMS),
)
ARMS: tuple[str, ...] = ("base", "adapter")
METRICS: tuple[str, ...] = ("kl", "emd", "cao_emd", "entropy", "option_mass")
LOGGED_METRICS: tuple[str, ...] = ("kl", "emd", "entropy", "option_mass")
DEFAULT_BATCH_SIZE = 16
DEBUG_LIMIT = 64


def cohort_of(data_type: str) -> str:
    for name, tags in COHORTS.items():
        if data_type in tags:
            return name
    raise ValueError(f"unknown reference test tag {data_type!r}; expected one of {COHORTS}")


def cao_emd(predicted: FloatArray, target: FloatArray) -> float:
    return float(wasserstein_distance(target, predicted))


def score_records(
    model: Any,
    encoder: PromptEncoder,
    records: Sequence[SurveyRecord],
    *,
    batch_size: int,
    device: str,
) -> list[dict[str, Any]]:
    collator = DistributionalCollator(pad_token_id_for(encoder.tokenizer))
    rows: list[dict[str, Any]] = []
    model.eval()
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        batch = collator([encoder.encode(record) for record in chunk])
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            answer = forward_answer_logits(model, batch)
        option_logits = gather_option_logits(answer, batch["option_token_ids"])
        target = batch["target_dist"]
        mass = option_mass(answer, batch["option_token_ids"], target >= 0)

        logits_np = option_logits.cpu().numpy().astype(np.float64)
        target_np = target.cpu().numpy().astype(np.float64)
        valid = target_np >= 0
        predicted = predicted_distribution(logits_np, valid)
        kl = exact_kl_array(logits_np, target_np, valid)
        emd = ordinal_emd(predicted, target_np, valid)
        entropy = option_entropy(predicted, valid)
        for index, record in enumerate(chunk):
            width = len(record.letters)
            rows.append(
                {
                    "id": record.id,
                    "country": record.country,
                    "data_type": record.data_type,
                    "cohort": cohort_of(record.data_type),
                    "options": width,
                    "kl": float(kl[index]),
                    "emd": float(emd[index]),
                    "cao_emd": cao_emd(predicted[index, :width], target_np[index, :width]),
                    "entropy": float(entropy[index]),
                    "option_mass": float(mass[index].item()),
                }
            )
    return rows


def summarize(
    rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for arm, rows in rows_by_arm.items():
        for row in rows:
            blocks = ["all", str(row["cohort"])]
            if row["id"] in EVALUATION_ITEMS:
                blocks.append(f"item_{row['id']}")
            for block in blocks:
                grouped.setdefault(block, {}).setdefault(arm, []).append(row)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for block in BLOCK_ORDER:
        if block not in grouped:
            continue
        summary[block] = {}
        for arm, rows in grouped[block].items():
            means = {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}
            means["n"] = float(len(rows))
            summary[block][arm] = means
    return summary


def kl_gains(summary: Mapping[str, Mapping[str, Mapping[str, float]]]) -> dict[str, float]:
    return {
        block: arms["base"]["kl"] - arms["adapter"]["kl"]
        for block, arms in summary.items()
        if "base" in arms and "adapter" in arms
    }


def render_table(summary: Mapping[str, Mapping[str, Mapping[str, float]]]) -> Table:
    table = Table(title="Held-out fit on the reference test split")
    table.add_column("block")
    table.add_column("arm")
    table.add_column("n", justify="right")
    for metric in METRICS:
        table.add_column(metric, justify="right")
    for block, arms in summary.items():
        for arm in ARMS:
            if arm not in arms:
                continue
            values = arms[arm]
            table.add_row(
                block,
                arm,
                str(int(values["n"])),
                *(f"{values[metric]:.4f}" for metric in METRICS),
            )
    return table


def output_dir_for(model_name: str, root: Path = OUTPUT_ROOT) -> Path:
    return root / model_name


def write_report(
    directory: Path,
    summary: Mapping[str, Mapping[str, Mapping[str, float]]],
    rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    meta: Mapping[str, Any],
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / REPORT_FILENAME
    report = {
        "meta": dict(meta),
        "summary": summary,
        "kl_gain_base_minus_adapter": kl_gains(summary),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    records_path = directory / RECORDS_FILENAME
    columns = ["arm", "id", "country", "data_type", "cohort", "options", *METRICS]
    with open(records_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for arm, rows in rows_by_arm.items():
            for row in rows:
                writer.writerow({"arm": arm, **{column: row[column] for column in columns[1:]}})
    return report_path, records_path


def log_to_mlflow(
    checkpoint_dir: str, summary: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> bool:
    run_id = load_mlflow_run_id(checkpoint_dir)
    if run_id is None:
        return False
    with mlflow.start_run(run_id=run_id):
        for block, arms in summary.items():
            for arm, values in arms.items():
                for metric in LOGGED_METRICS:
                    mlflow.log_metric(f"heldout_{arm}_{metric}_{block}", values[metric])
    return True


def evaluate(
    cfg: dict[str, Any],
    model_name: str,
    *,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_root: Path = OUTPUT_ROOT,
    log_mlflow: bool = True,
    debug: bool = False,
) -> dict[str, Any]:
    data_cfg = cfg.get("data") or {}
    data_format = str(data_cfg.get("format", "")).lower()
    if data_format != DATA_FORMAT:
        raise ValueError(
            f"data.format must be {DATA_FORMAT!r} for this script, got {data_format!r}"
        )
    condition = str(cfg.get("condition", CONDITION))
    if debug:
        condition = condition + DEBUG_CONDITION_SUFFIX
        limit = DEBUG_LIMIT if limit is None else limit
    checkpoint = checkpoint_output_dir(PSEUDO_CULTURE, model_name, condition)
    if not (checkpoint / DONE_SENTINEL).is_file():
        raise FileNotFoundError(f"{checkpoint} has no {DONE_SENTINEL}; train the adapter first")
    if log_mlflow:
        configure_mlflow(cfg)

    backbone = load_backbone(cfg["model"], cfg["training"])
    tokenizer = processor_tokenizer(backbone.processor)
    encoder = PromptEncoder.build(tokenizer, max_length=int(cfg["training"]["max_seq_len"]))
    records = dedupe_records(load_split("test", RAW_DIR))
    if limit is not None:
        records = records[:limit]
    console.print(f"  Scoring {len(records)} held-out records with {checkpoint}")

    model = PeftModel.from_pretrained(backbone.model, str(checkpoint))
    model.eval()
    with model.disable_adapter():
        base_rows = score_records(
            model, encoder, records, batch_size=batch_size, device=backbone.device
        )
    adapter_rows = score_records(
        model, encoder, records, batch_size=batch_size, device=backbone.device
    )
    rows_by_arm = {"base": base_rows, "adapter": adapter_rows}
    summary = summarize(rows_by_arm)

    meta = {
        "model": model_name,
        "base_model_id": cfg["model"]["id"],
        "checkpoint": str(checkpoint),
        "condition": condition,
        "split": "test",
        "records": len(records),
        "deduplicated_by": "country, id",
        "limit": limit,
        "cohorts": COHORTS,
        "evaluation_items": EVALUATION_ITEMS,
    }
    directory = output_dir_for(model_name + (DEBUG_CONDITION_SUFFIX if debug else ""), output_root)
    report_path, _ = write_report(directory, summary, rows_by_arm, meta)
    console.print(render_table(summary))
    if log_mlflow and log_to_mlflow(str(checkpoint), summary):
        console.print("  Held-out means appended to the training run")
    console.print(f"[bold green]✓ Report at {report_path}[/bold green]")
    return {"meta": meta, "summary": summary, "report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Held-out fit of the distributional adapter against its base."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--debug", action="store_true", help="score the _debug checkpoint leaf")
    args = parser.parse_args()

    console.rule("[bold blue]culture-mllm Distributional Held-out Evaluation[/bold blue]")
    cfg = load_config(args.config)
    model_name = get_model_key(cfg, args.config)
    evaluate(
        cfg,
        model_name,
        limit=args.limit,
        batch_size=args.batch_size,
        output_root=args.output_root,
        log_mlflow=not args.no_mlflow,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
