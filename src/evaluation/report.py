"""Generate summary CSV and LaTeX tables from analysis outputs."""

from pathlib import Path

import numpy as np
import pandas as pd


def make_accuracy_table(agreement_csv: str, output_tex: str) -> str:
    """LaTeX table of macro-F1 / kappa per (culture × model)."""
    df = pd.read_csv(agreement_csv)
    df = df.sort_values(["model_name", "culture"])

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Cultural VLM accuracy vs.\ $\sigma_3 P_5$ human agreement labels.}",
        r"\label{tab:accuracy}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Culture & Acc & F1-macro & $\kappa_\text{lin}$ & $\kappa_\text{qw}$ \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row.get('model_name','')} & {row.get('culture','')} & "
            f"{row.get('accuracy',0):.3f} & {row.get('f1_macro',0):.3f} & "
            f"{row.get('kappa_linear',0):.3f} & {row.get('kappa_quadratic',0):.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    Path(output_tex).parent.mkdir(parents=True, exist_ok=True)
    with open(output_tex, "w") as f:
        f.write(tex)
    return tex


def make_convergence_summary(convergence_csv: str, output_tex: str) -> str:
    """LaTeX table of mean convergence metrics per model."""
    df = pd.read_csv(convergence_csv)
    summary = df[["caption_sim", "sentiment_agreement", "label_jaccard"]].agg(
        ["mean", "std"]
    ).T.round(3)
    summary.columns = ["mean", "std"]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Mean per-image convergence metrics across cultural models.}",
        r"\label{tab:convergence}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Metric & Mean & Std \\",
        r"\midrule",
    ]
    for metric, row in summary.iterrows():
        lines.append(f"{metric.replace('_', ' ').title()} & {row['mean']:.3f} & {row['std']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    Path(output_tex).parent.mkdir(parents=True, exist_ok=True)
    with open(output_tex, "w") as f:
        f.write(tex)
    return tex
