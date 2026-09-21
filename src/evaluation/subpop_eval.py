"""Held-out fit of the SubPOP adapter against its base on SubPOP-Eval (GSS 2022)."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from peft import PeftModel
from rich.console import Console
from rich.table import Table

from src.data.distributional_data import DistributionalCollator, pad_token_id_for
from src.data.subpop_data import (
    CONDITION,
    DATA_FORMAT,
    DATASET_REVISION,
    MAX_OPTIONS,
    PSEUDO_CULTURE,
    RAW_DIR,
    SubpopEncoder,
    SubpopRecord,
    load_records,
    load_steering,
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
    exact_kl_array,
    forward_answer_logits,
    gather_option_logits,
    option_entropy,
    option_mass,
    predicted_distribution,
)
from src.utils.model_loading import processor_tokenizer

console = Console()

OUTPUT_ROOT = Path("outputs/evaluation/subpop")
REPORT_FILENAME = "heldout.json"
RECORDS_FILENAME = "records.csv"
HIGH_RELEVANCE_QKEYS: tuple[str, ...] = ("marital", "satjob", "attend", "relpersn", "othlang")
ITEM_QKEYS: dict[str, str] = {
    "happy": "happiness",
    "trust": "social trust",
    "attend": "religious attendance",
}
ARMS: tuple[str, ...] = ("base", "adapter")
METRICS: tuple[str, ...] = ("wd", "kl", "entropy", "option_mass")
GAIN_METRICS: tuple[str, ...] = ("wd", "kl")
DEFAULT_BATCH_SIZE = 16
DEBUG_LIMIT = 64

Summary = dict[str, dict[str, dict[str, float]]]


def subpop_wd(
    predicted: Sequence[float], target: Sequence[float], ordinal: Sequence[float]
) -> float:
    if max(ordinal) == min(ordinal):
        return float("nan")
    ranked = sorted(zip(ordinal, target, predicted, strict=True))
    kept = [entry for entry in ranked if entry[0] >= 0] or ranked
    levels = [entry[0] for entry in kept]
    span = max(levels) - min(levels)
    if span <= 0:
        return float("nan")
    cdf_target = np.cumsum(_unit([entry[1] for entry in kept]))
    cdf_predicted = np.cumsum(_unit([entry[2] for entry in kept]))
    gaps = np.abs(cdf_target - cdf_predicted)[:-1]
    return float((gaps * np.diff(levels)).sum() / span)


def _unit(values: Sequence[float]) -> list[float]:
    total = sum(values)
    if total == 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def score_records(
    model: Any,
    encoder: SubpopEncoder,
    records: Sequence[SubpopRecord],
    *,
    batch_size: int,
    device: str,
) -> list[dict[str, Any]]:
    collator = DistributionalCollator(pad_token_id_for(encoder.tokenizer), max_options=MAX_OPTIONS)
    rows: list[dict[str, Any]] = []
    model.eval()
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        batch = collator([encoder.encode(record) for record in chunk])
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            answer = forward_answer_logits(model, batch)
        option_logits = gather_option_logits(answer, batch["option_token_ids"])
        mass = option_mass(answer, batch["option_token_ids"], batch["target_dist"] >= 0)

        logits_np = option_logits.cpu().numpy().astype(np.float64)
        widths = np.asarray([len(record.responses) for record in chunk])
        valid = np.arange(MAX_OPTIONS)[None, :] < widths[:, None]
        target_np = np.zeros((len(chunk), MAX_OPTIONS), dtype=np.float64)
        for index, record in enumerate(chunk):
            target_np[index, : len(record.responses)] = record.responses
        predicted = predicted_distribution(logits_np, valid)
        kl = exact_kl_array(logits_np, target_np, valid)
        entropy = option_entropy(predicted, valid)
        for index, record in enumerate(chunk):
            width = int(widths[index])
            rows.append(
                {
                    "qkey": record.qkey,
                    "attribute": record.attribute,
                    "group": record.group,
                    "options": width,
                    "wd": subpop_wd(
                        predicted[index, :width].tolist(), record.responses, record.ordinal
                    ),
                    "kl": float(kl[index]),
                    "entropy": float(entropy[index]),
                    "option_mass": float(mass[index].item()),
                }
            )
    return rows


def blocks_of(row: Mapping[str, Any]) -> list[str]:
    blocks = ["all", f"attribute_{row['attribute']}"]
    if row["qkey"] not in HIGH_RELEVANCE_QKEYS:
        blocks.append("upstream")
    if row["qkey"] in ITEM_QKEYS:
        blocks.append(f"item_{row['qkey']}")
    return blocks


def block_order(blocks: Sequence[str]) -> list[str]:
    attributes = sorted(block for block in blocks if block.startswith("attribute_"))
    items = [f"item_{qkey}" for qkey in ITEM_QKEYS]
    ordered = ["all", "upstream", *attributes, *items]
    return [block for block in ordered if block in blocks]


def _finite_mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def summarize(rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> Summary:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for arm, rows in rows_by_arm.items():
        for row in rows:
            for block in blocks_of(row):
                grouped.setdefault(block, {}).setdefault(arm, []).append(row)
    summary: Summary = {}
    for block in block_order(list(grouped)):
        summary[block] = {}
        for arm, rows in grouped[block].items():
            means = {metric: _finite_mean([row[metric] for row in rows]) for metric in METRICS}
            means["n"] = float(len(rows))
            summary[block][arm] = means
    return summary


def gains(
    summary: Mapping[str, Mapping[str, Mapping[str, float]]], metric: str
) -> dict[str, float]:
    return {
        block: arms["base"][metric] - arms["adapter"][metric]
        for block, arms in summary.items()
        if "base" in arms and "adapter" in arms
    }


def render_table(summary: Mapping[str, Mapping[str, Mapping[str, float]]]) -> Table:
    table = Table(title="Held-out fit on SubPOP-Eval")
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


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
        **{f"{metric}_gain_base_minus_adapter": gains(summary, metric) for metric in GAIN_METRICS},
    }
    report_path.write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    records_path = directory / RECORDS_FILENAME
    columns = ["arm", "qkey", "attribute", "group", "options", *METRICS]
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
                for metric in METRICS:
                    if math.isfinite(values[metric]):
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
    encoder = SubpopEncoder.build(
        tokenizer, load_steering(RAW_DIR), max_length=int(cfg["training"]["max_seq_len"])
    )
    records = load_records("eval", RAW_DIR)
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
        "split": "eval",
        "dataset_revision": DATASET_REVISION,
        "records": len(records),
        "limit": limit,
        "refusal_option": "dropped before renormalising, as upstream",
        "high_relevance_qkeys": list(HIGH_RELEVANCE_QKEYS),
        "item_qkeys": ITEM_QKEYS,
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
        description="Held-out fit of the SubPOP adapter against its base on SubPOP-Eval."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--debug", action="store_true", help="score the _debug checkpoint leaf")
    args = parser.parse_args()

    console.rule("[bold blue]culture-mllm SubPOP Held-out Evaluation[/bold blue]")
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
