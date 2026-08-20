#!/usr/bin/env python3
"""Generate the training/validation loss curve figure from MLflow metrics.

Reads actual loss history for the Arabic cultural Qwen3.5-2B run
(run 8ec81ee5) and saves a publication-ready PDF to
outputs/figures/loss_curve_arabic_cultural.pdf.

Usage:
    uv run python scripts/generate_loss_curve.py
    uv run python scripts/generate_loss_curve.py --run-id <full-run-id> --out <path>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mlflow

REPO_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_URI = f"sqlite:///{REPO_ROOT}/mlflow.db"

TRAIN_COLOR = "#FD4F4F"
VAL_COLOR = "#5BAD75"

EPOCH_STEPS = [1890, 3780]
TOTAL_STEPS = 5670


def fetch_metrics(run_id: str) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Fetch train and validation loss histories from MLflow.

    Args:
        run_id (str): MLflow run ID.

    Returns:
        tuple: (train_steps, train_values, val_steps, val_values), each a list.
    """
    client = mlflow.MlflowClient(MLFLOW_URI)
    tl = sorted(client.get_metric_history(run_id, "train_loss"), key=lambda m: m.step)
    vl = sorted(client.get_metric_history(run_id, "val_loss"), key=lambda m: m.step)
    return (
        [m.step for m in tl],
        [m.value for m in tl],
        [m.step for m in vl],
        [m.value for m in vl],
    )


def find_run_id(culture: str, condition: str, model: str) -> str:
    """Look up an MLflow run ID by culture, condition, and model tags.

    Args:
        culture (str): Culture tag value (e.g., "arabic").
        condition (str): Condition tag value (e.g., "cultural").
        model (str): Model name tag value (e.g., "qwen3_5_2b").

    Returns:
        str: The run ID of the first matching run.

    Raises:
        ValueError: If no matching run is found.
    """
    client = mlflow.MlflowClient(MLFLOW_URI)
    exps = client.search_experiments()
    exp_id = next(e.experiment_id for e in exps if "culturevlm" in e.name)
    runs = client.search_runs(
        exp_id,
        filter_string=(
            f"tags.culture = '{culture}' AND "
            f"tags.condition = '{condition}' AND "
            f"tags.model_name = '{model}'"
        ),
    )
    if not runs:
        raise ValueError(f"No run found for {culture}/{condition}/{model}")
    return cast(str, runs[0].info.run_id)


def plot(
    train_steps: list[Any],
    train_vals: list[Any],
    val_steps: list[Any],
    val_vals: list[Any],
    out_path: Path,
    culture: str = "Arabic",
    condition: str = "cultural",
) -> None:
    """Render and save a publication-ready loss-curve PDF.

    Axis limits are set before the annotations are drawn so that the final limits are
    already in effect when annotation and epoch-label positions are computed.

    Args:
        train_steps (list): Step indices for training loss.
        train_vals (list): Training loss values.
        val_steps (list): Step indices for validation loss.
        val_vals (list): Validation loss values.
        out_path (Path): Destination PDF path.
        culture (str, optional): Culture label for the title. Defaults to "Arabic".
        condition (str, optional): Condition label for the title. Defaults to "cultural".
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.0))

    ax.plot(
        train_steps,
        train_vals,
        color=TRAIN_COLOR,
        linewidth=1.3,
        label="Train loss (every 25 steps)",
    )
    ax.plot(
        val_steps,
        val_vals,
        color=VAL_COLOR,
        linewidth=1.3,
        linestyle="--",
        marker="o",
        markersize=2.0,
        markevery=5,
        label="Val loss (every 100 steps)",
    )

    ax.set_xlim(0, TOTAL_STEPS + 50)
    ymax = min(4.0, max(max(train_vals), max(val_vals)) * 1.15)
    ax.set_ylim(0, ymax)

    for step in EPOCH_STEPS:
        ax.axvline(step, color="black", linewidth=0.7, linestyle="--", alpha=0.45)
    ax.text(EPOCH_STEPS[0] + 30, ymax * 0.97, "Epoch 2", fontsize=6.5, color="#555555", va="top")
    ax.text(EPOCH_STEPS[1] + 30, ymax * 0.97, "Epoch 3", fontsize=6.5, color="#555555", va="top")

    final_tl = train_vals[-1]
    final_vl = val_vals[-1]
    bbox = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", linewidth=0.6)
    ax.annotate(
        f"TL={final_tl:.3f}",
        xy=(train_steps[-1], final_tl),
        xytext=(-48, 6),
        textcoords="offset points",
        fontsize=6.5,
        color="#444444",
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6),
        bbox=bbox,
    )
    ax.annotate(
        f"VL={final_vl:.3f}",
        xy=(val_steps[-1], final_vl),
        xytext=(6, -14),
        textcoords="offset points",
        fontsize=6.5,
        color="#444444",
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6),
        bbox=bbox,
    )

    spike_y = min(ymax * 0.94, 3.6)
    ax.annotate(
        "spike\nclipped",
        xy=(200, spike_y),
        xytext=(370, spike_y - 0.3),
        fontsize=5.5,
        color="#888888",
        ha="left",
        arrowprops=dict(arrowstyle="-|>", color="#aaaaaa", lw=0.5),
    )
    ax.set_xlabel("Training Step", fontsize=8)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1000))
    ax.grid(axis="both", linestyle=":", linewidth=0.5, color="#cccccc")
    ax.legend(fontsize=7, framealpha=0.9, loc="upper right")

    title = f"Qwen3.5-2B — {culture.capitalize()} {condition}"
    ax.set_title(title, fontsize=8, pad=4)

    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate loss-curve PDF from MLflow")
    parser.add_argument(
        "--run-id", default=None, help="Full MLflow run ID (auto-detected if omitted)"
    )
    parser.add_argument("--culture", default="arabic")
    parser.add_argument("--condition", default="cultural")
    parser.add_argument("--model", default="qwen3_5_2b")
    parser.add_argument(
        "--out",
        default=None,
        help="Output PDF path (default: outputs/figures/loss_curve_<culture>_<condition>.pdf)",
    )
    args = parser.parse_args()

    run_id = args.run_id or find_run_id(args.culture, args.condition, args.model)
    print(f"Using run_id: {run_id}")

    ts, tv, vs, vv = fetch_metrics(run_id)
    print(f"  train points: {len(ts)}  val points: {len(vs)}")

    out = (
        Path(args.out)
        if args.out
        else (REPO_ROOT / "outputs" / "figures" / f"loss_curve_{args.culture}_{args.condition}.pdf")
    )
    plot(ts, tv, vs, vv, out, culture=args.culture, condition=args.condition)


if __name__ == "__main__":
    main()
