#!/usr/bin/env python3
"""Generate loss-curve figures for all Qwen3.5-2B fine-tuning runs.

Produces two outputs:
  outputs/figures/loss_curves_all.pdf
    — 3×3 grid, one panel per culture, cultural (solid) and baseline
      (dashed) overlaid for direct comparison.

  outputs/figures/loss_curve_<culture>_<condition>.pdf  (18 files)
    — Individual panels, one per (culture, condition) pair.

Usage:
    uv run python scripts/generate_loss_curves_all.py
    uv run python scripts/generate_loss_curves_all.py --no-individual
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mlflow

REPO_ROOT  = Path(__file__).resolve().parents[1]
MLFLOW_URI = f"sqlite:///{REPO_ROOT}/mlflow.db"
FIG_DIR    = REPO_ROOT / "outputs" / "figures"

CULTURES = [
    "arabic", "bengali", "chinese",
    "english", "german", "korean",
    "portuguese", "spanish", "turkish",
]
EPOCH_STEPS = [1890, 3780]
TOTAL_STEPS = 5670

# Palette
C_CULTURAL = "#E05252"   # warm red  — cultural condition
C_BASELINE = "#5B8FD4"   # steel blue — baseline condition


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

def _fetch(client: mlflow.MlflowClient, run_id: str, key: str):
    return sorted(client.get_metric_history(run_id, key), key=lambda m: m.step)


def load_all_runs() -> dict:
    """Return {(culture, condition): {"ts": [...], "tv": [...], "vs": [...], "vv": [...]}}"""
    client = mlflow.MlflowClient(MLFLOW_URI)
    exps   = client.search_experiments()
    exp_id = next(e.experiment_id for e in exps if "culturevlm" in e.name)
    rows   = client.search_runs(exp_id)

    data = {}
    for r in rows:
        tags      = r.data.tags
        culture   = tags.get("culture")
        condition = tags.get("condition")
        if not culture or not condition:
            continue
        tl = _fetch(client, r.info.run_id, "train_loss")
        vl = _fetch(client, r.info.run_id, "val_loss")
        data[(culture, condition)] = {
            "ts": [m.step  for m in tl], "tv": [m.value for m in tl],
            "vs": [m.step  for m in vl], "vv": [m.value for m in vl],
            "model": tags.get("model_name", "unknown"),
        }
    return data


# ---------------------------------------------------------------------------
# Shared panel helper
# ---------------------------------------------------------------------------

def _draw_panel(
    ax: plt.Axes,
    ts, tv, vs, vv,
    color: str,
    condition_label: str,
    alpha: float = 1.0,
    linestyle: str = "-",
) -> None:
    ax.plot(ts, tv, color=color, linewidth=0.9, linestyle=linestyle,
            alpha=alpha, label=f"Train ({condition_label})")
    ax.plot(vs, vv, color=color, linewidth=0.9, linestyle="--",
            marker="o", markersize=1.4, markevery=8,
            alpha=alpha * 0.85, label=f"Val ({condition_label})")


def _finalise_panel(ax: plt.Axes, show_epoch_labels: bool = False) -> None:
    ymax = min(4.0, ax.get_ylim()[1])
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, TOTAL_STEPS + 60)

    for step in EPOCH_STEPS:
        ax.axvline(step, color="black", linewidth=0.5, linestyle=":", alpha=0.4)
    if show_epoch_labels:
        ax.text(EPOCH_STEPS[0] + 40, ymax * 0.96, "E2",
                fontsize=5, color="#777777", va="top")
        ax.text(EPOCH_STEPS[1] + 40, ymax * 0.96, "E3",
                fontsize=5, color="#777777", va="top")

    ax.tick_params(labelsize=6)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2000))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.grid(axis="both", linestyle=":", linewidth=0.4, color="#dddddd")


# ---------------------------------------------------------------------------
# Combined 3×3 grid
# ---------------------------------------------------------------------------

def make_combined(data: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.6), sharey=False, sharex=True)
    fig.subplots_adjust(hspace=0.42, wspace=0.28,
                        left=0.07, right=0.98, top=0.91, bottom=0.09)

    for idx, culture in enumerate(CULTURES):
        ax  = axes[idx // 3][idx % 3]
        cul = data.get((culture, "cultural"))
        bas = data.get((culture, "baseline"))

        if cul:
            _draw_panel(ax, cul["ts"], cul["tv"], cul["vs"], cul["vv"],
                        C_CULTURAL, "cultural")
        if bas:
            _draw_panel(ax, bas["ts"], bas["tv"], bas["vs"], bas["vv"],
                        C_BASELINE, "baseline", linestyle="--")

        show_elabels = idx in (0,)   # epoch labels only on first panel
        _finalise_panel(ax, show_epoch_labels=show_elabels)
        ax.text(0.03, 0.97, culture.capitalize(), transform=ax.transAxes,
                fontsize=7, fontweight="bold", color="#333333",
                va="top", ha="left")

        # final-value annotations
        for d, color, yoff in [(cul, C_CULTURAL, +5), (bas, C_BASELINE, -9)]:
            if d:
                tl_f = d["tv"][-1]
                ax.annotate(f"{tl_f:.2f}",
                            xy=(d["ts"][-1], tl_f),
                            xytext=(-22, yoff), textcoords="offset points",
                            fontsize=5, color=color, alpha=0.9,
                            arrowprops=dict(arrowstyle="-", color=color,
                                            lw=0.4, alpha=0.6))

    # shared axis labels
    fig.text(0.5, 0.01, "Training Step", ha="center", fontsize=8)
    fig.text(0.01, 0.5, "Cross-Entropy Loss", va="center",
             rotation="vertical", fontsize=8)

    # legend (top, shared)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=C_CULTURAL, lw=1.2, label="Cultural — train"),
        Line2D([0], [0], color=C_CULTURAL, lw=1.2, linestyle="--", label="Cultural — val"),
        Line2D([0], [0], color=C_BASELINE, lw=1.2, label="Baseline — train"),
        Line2D([0], [0], color=C_BASELINE, lw=1.2, linestyle="--", label="Baseline — val"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               fontsize=6.5, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.995), columnspacing=1.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined: {out_path}")


# ---------------------------------------------------------------------------
# Individual panels
# ---------------------------------------------------------------------------

def make_individual(data: dict, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for (culture, condition), d in data.items():
        fig, ax = plt.subplots(figsize=(4.0, 2.4))
        color     = C_CULTURAL if condition == "cultural" else C_BASELINE
        model     = d.get("model", "qwen_vl")
        _draw_panel(ax, d["ts"], d["tv"], d["vs"], d["vv"], color, condition)
        _finalise_panel(ax, show_epoch_labels=True)

        # final-value labels
        bbox = dict(boxstyle="round,pad=0.15", facecolor="white",
                    edgecolor="#cccccc", linewidth=0.5)
        ax.annotate(f"TL={d['tv'][-1]:.3f}",
                    xy=(d["ts"][-1], d["tv"][-1]),
                    xytext=(-42, 6), textcoords="offset points",
                    fontsize=6, color="#333333",
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
                    bbox=bbox)
        ax.annotate(f"VL={d['vv'][-1]:.3f}",
                    xy=(d["vs"][-1], d["vv"][-1]),
                    xytext=(5, -12), textcoords="offset points",
                    fontsize=6, color="#333333",
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
                    bbox=bbox)

        ax.set_xlabel("Training Step", fontsize=7)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=7)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.9)
        fig.tight_layout(pad=0.4)

        out = fig_dir / f"loss_curve_{model}_{culture}_{condition}.pdf"
        fig.savefig(out, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  {out.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-individual", action="store_true",
                        help="Skip generating individual per-run PDFs")
    args = parser.parse_args()

    print("Loading MLflow data…")
    data = load_all_runs()
    print(f"  {len(data)} runs loaded")

    make_combined(data, FIG_DIR / "loss_curves_qwen_vl_all.pdf")

    if not args.no_individual:
        print("Generating individual panels…")
        make_individual(data, FIG_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
