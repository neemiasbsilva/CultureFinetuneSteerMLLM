"""Generate summary CSV and LaTeX tables from analysis outputs."""

from pathlib import Path
from typing import cast

import pandas as pd


def make_accuracy_table(agreement_csv: str, output_tex: str) -> str:
    """LaTeX table of macro-F1 / kappa per condition/culture/model."""
    df = pd.read_csv(agreement_csv)
    if "condition" not in df:
        df["condition"] = "legacy_unspecified"
    df = df.sort_values(["model_name", "culture", "condition"])

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Cultural VLM accuracy vs.\ $\sigma_3 P_5$ human agreement labels.}",
        r"\label{tab:accuracy}",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Model & Culture & Condition & Acc & F1-macro & "
        r"$\kappa_\text{lin}$ & $\kappa_\text{qw}$ \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row.get('model_name', '')} & {row.get('culture', '')} & "
            f"{row.get('condition', '')} & "
            f"{row.get('accuracy', 0):.3f} & {row.get('f1_macro', 0):.3f} & "
            f"{row.get('kappa_linear', 0):.3f} & {row.get('kappa_quadratic', 0):.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    Path(output_tex).parent.mkdir(parents=True, exist_ok=True)
    with open(output_tex, "w") as f:
        f.write(tex)
    return tex


def make_convergence_summary(convergence_csv: str, output_tex: str) -> str:
    """LaTeX table of convergence metrics without pooling conditions."""
    df = pd.read_csv(convergence_csv)
    if "condition" not in df:
        df["condition"] = "legacy_unspecified"
    metrics = ["caption_sim", "sentiment_agreement", "label_jaccard"]
    summary_rows: list[tuple[str, str, float, float]] = []
    for condition, group in df.groupby("condition", sort=True):
        summary = group[metrics].agg(["mean", "std"]).T
        for metric, values in summary.iterrows():
            summary_rows.append(
                (str(condition), cast(str, metric), float(values["mean"]), float(values["std"]))
            )

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Mean per-image convergence metrics across cultural models.}",
        r"\label{tab:convergence}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Condition & Metric & Mean & Std \\",
        r"\midrule",
    ]
    for condition, metric, mean, std in summary_rows:
        lines.append(
            f"{condition} & {metric.replace('_', ' ').title()} & {mean:.3f} & {std:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    Path(output_tex).parent.mkdir(parents=True, exist_ok=True)
    with open(output_tex, "w") as f:
        f.write(tex)
    return tex
