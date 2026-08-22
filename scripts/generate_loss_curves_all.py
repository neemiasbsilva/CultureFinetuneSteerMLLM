#!/usr/bin/env python3
"""Generate loss-curve figures for all Qwen3.5-2B fine-tuning runs.

Produces two outputs:
  outputs/figures/loss_curves_all.pdf
    — three-column grid, one panel per culture, cultural (solid) and baseline
      (dashed) overlaid for direct comparison.

  outputs/figures/loss_curve_<culture>_<condition>.pdf  (two per culture)
    — Individual panels, one per (culture, condition) pair.

Usage:
    uv run python scripts/generate_loss_curves_all.py
    uv run python scripts/generate_loss_curves_all.py --no-individual
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mlflow
from matplotlib.axes import Axes

from src.data.cultures import CULTURES

REPO_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_URI = f"sqlite:///{REPO_ROOT}/mlflow.db"
FIG_DIR = REPO_ROOT / "outputs" / "figures"

GRID_COLS = 3
EPOCH_FRACTIONS = (1 / 3, 2 / 3)

C_CULTURAL = "#E05252"
C_BASELINE = "#5B8FD4"


def _fetch(client: mlflow.MlflowClient, run_id: str, key: str) -> list[Any]:
    """Fetch a metric history from MLflow, sorted by step.

    Args:
        client (mlflow.MlflowClient): Active MLflow client.
        run_id (str): MLflow run ID.
        key (str): Metric key (e.g., "train_loss").

    Returns:
        list: Metric objects sorted by step.
    """
    return sorted(client.get_metric_history(run_id, key), key=lambda m: m.step)


def load_all_runs() -> dict[tuple[str, str], dict[str, Any]]:
    """Load all Qwen3.5-2B fine-tuning run metrics from MLflow.

    Returns:
        dict: Mapping of (culture, condition) to a dict with keys
            "ts", "tv" (train steps/values) and "vs", "vv" (val steps/values).
    """
    client = mlflow.MlflowClient(MLFLOW_URI)
    exps = client.search_experiments()
    exp_id = next(e.experiment_id for e in exps if "culturevlm" in e.name)
    rows = client.search_runs(exp_id)

    data: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        tags = r.data.tags
        culture = tags.get("culture")
        condition = tags.get("condition")
        if not culture or not condition:
            continue
        tl = _fetch(client, r.info.run_id, "train_loss")
        vl = _fetch(client, r.info.run_id, "val_loss")
        data[(culture, condition)] = {
            "ts": [m.step for m in tl],
            "tv": [m.value for m in tl],
            "vs": [m.step for m in vl],
            "vv": [m.value for m in vl],
            "model": tags.get("model_name", "unknown"),
        }
    return data


def _draw_panel(
    ax: Axes,
    ts: list[Any],
    tv: list[Any],
    vs: list[Any],
    vv: list[Any],
    color: str,
    condition_label: str,
    alpha: float = 1.0,
    linestyle: str = "-",
) -> None:
    ax.plot(
        ts,
        tv,
        color=color,
        linewidth=0.9,
        linestyle=linestyle,
        alpha=alpha,
        label=f"Train ({condition_label})",
    )
    ax.plot(
        vs,
        vv,
        color=color,
        linewidth=0.9,
        linestyle="--",
        marker="o",
        markersize=1.4,
        markevery=8,
        alpha=alpha * 0.85,
        label=f"Val ({condition_label})",
    )


def _run_end(data: dict[tuple[str, str], dict[str, Any]], culture: str | None = None) -> float:
    """Return the last training step reached, over one culture or over every run.

    Cultures no longer share a step count: a partition built from a single country
    holds half the examples of a pooled one, so its epochs fall at different steps.
    Epoch markers are placed from each culture's own run length rather than from a
    constant.
    """
    ends = [
        d["ts"][-1]
        for (run_culture, _), d in data.items()
        if d["ts"] and (culture is None or run_culture == culture)
    ]
    return float(max(ends)) if ends else 0.0


def _finalise_panel(
    ax: Axes,
    x_limit: float,
    epoch_steps: list[float],
    show_epoch_labels: bool = False,
) -> None:
    """Apply shared axis limits, epoch markers, and grid to a panel.

    Args:
        ax (Axes): The axes to finalise.
        x_limit (float): Upper bound of the step axis, shared across panels.
        epoch_steps (list[float]): Step positions of this run's epoch boundaries.
        show_epoch_labels (bool, optional): Whether to annotate epoch boundaries. Defaults to False.
    """
    ymax = min(4.0, ax.get_ylim()[1])
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, x_limit + 60)

    for step in epoch_steps:
        ax.axvline(step, color="black", linewidth=0.5, linestyle=":", alpha=0.4)
    if show_epoch_labels:
        for step, label in zip(epoch_steps, ("E2", "E3"), strict=False):
            ax.text(step + 40, ymax * 0.96, label, fontsize=5, color="#777777", va="top")

    ax.tick_params(labelsize=6)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2000))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.grid(axis="both", linestyle=":", linewidth=0.4, color="#dddddd")


def make_combined(data: dict[tuple[str, str], dict[str, Any]], out_path: Path) -> None:
    """Render a grid comparing cultural vs. baseline conditions per culture.

    Args:
        data (dict): Run data as returned by load_all_runs().
        out_path (Path): Destination PDF path.
    """
    n_rows = -(-len(CULTURES) // GRID_COLS)
    x_limit = _run_end(data)
    fig, axes = plt.subplots(
        n_rows, GRID_COLS, figsize=(7.2, 1.87 * n_rows), sharey=False, sharex=True
    )
    fig.subplots_adjust(hspace=0.42, wspace=0.28, left=0.07, right=0.98, top=0.91, bottom=0.09)

    for ax in axes.flat[len(CULTURES) :]:
        ax.set_visible(False)

    for idx, culture in enumerate(CULTURES):
        ax = axes[idx // GRID_COLS][idx % GRID_COLS]
        cul = data.get((culture, "cultural"))
        bas = data.get((culture, "baseline"))
        epoch_steps = [f * _run_end(data, culture) for f in EPOCH_FRACTIONS]

        if cul:
            _draw_panel(ax, cul["ts"], cul["tv"], cul["vs"], cul["vv"], C_CULTURAL, "cultural")
        if bas:
            _draw_panel(
                ax,
                bas["ts"],
                bas["tv"],
                bas["vs"],
                bas["vv"],
                C_BASELINE,
                "baseline",
                linestyle="--",
            )

        show_elabels = idx in (0,)
        _finalise_panel(ax, x_limit, epoch_steps, show_epoch_labels=show_elabels)
        ax.text(
            0.03,
            0.97,
            culture.capitalize(),
            transform=ax.transAxes,
            fontsize=7,
            fontweight="bold",
            color="#333333",
            va="top",
            ha="left",
        )

        for d, color, yoff in [(cul, C_CULTURAL, +5), (bas, C_BASELINE, -9)]:
            if d:
                tl_f = d["tv"][-1]
                ax.annotate(
                    f"{tl_f:.2f}",
                    xy=(d["ts"][-1], tl_f),
                    xytext=(-22, yoff),
                    textcoords="offset points",
                    fontsize=5,
                    color=color,
                    alpha=0.9,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.4, alpha=0.6),
                )

    fig.text(0.5, 0.01, "Training Step", ha="center", fontsize=8)
    fig.text(0.01, 0.5, "Cross-Entropy Loss", va="center", rotation="vertical", fontsize=8)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=C_CULTURAL, lw=1.2, label="Cultural — train"),
        Line2D([0], [0], color=C_CULTURAL, lw=1.2, linestyle="--", label="Cultural — val"),
        Line2D([0], [0], color=C_BASELINE, lw=1.2, label="Baseline — train"),
        Line2D([0], [0], color=C_BASELINE, lw=1.2, linestyle="--", label="Baseline — val"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        fontsize=6.5,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.0,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined: {out_path}")


def make_individual(data: dict[tuple[str, str], dict[str, Any]], fig_dir: Path) -> None:
    """Save one PDF per (culture, condition) run.

    Args:
        data (dict): Run data as returned by load_all_runs().
        fig_dir (Path): Directory where individual PDFs are written.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    for (culture, condition), d in data.items():
        fig, ax = plt.subplots(figsize=(4.0, 2.4))
        color = C_CULTURAL if condition == "cultural" else C_BASELINE
        model = d.get("model", "qwen3_5_2b")
        _draw_panel(ax, d["ts"], d["tv"], d["vs"], d["vv"], color, condition)
        run_end = float(d["ts"][-1]) if d["ts"] else 0.0
        _finalise_panel(ax, run_end, [f * run_end for f in EPOCH_FRACTIONS], show_epoch_labels=True)

        bbox = dict(
            boxstyle="round,pad=0.15", facecolor="white", edgecolor="#cccccc", linewidth=0.5
        )
        ax.annotate(
            f"TL={d['tv'][-1]:.3f}",
            xy=(d["ts"][-1], d["tv"][-1]),
            xytext=(-42, 6),
            textcoords="offset points",
            fontsize=6,
            color="#333333",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
            bbox=bbox,
        )
        ax.annotate(
            f"VL={d['vv'][-1]:.3f}",
            xy=(d["vs"][-1], d["vv"][-1]),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=6,
            color="#333333",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
            bbox=bbox,
        )

        ax.set_xlabel("Training Step", fontsize=7)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=7)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.9)
        fig.tight_layout(pad=0.4)

        out = fig_dir / f"loss_curve_{model}_{culture}_{condition}.pdf"
        fig.savefig(out, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  {out.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-individual", action="store_true", help="Skip generating individual per-run PDFs"
    )
    args = parser.parse_args()

    print("Loading MLflow data…")
    data = load_all_runs()
    print(f"  {len(data)} runs loaded")

    make_combined(data, FIG_DIR / "loss_curves_qwen3_5_2b_all.pdf")

    if not args.no_individual:
        print("Generating individual panels…")
        make_individual(data, FIG_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
