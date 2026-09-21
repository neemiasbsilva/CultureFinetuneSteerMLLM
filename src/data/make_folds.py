"""Stratified 5-fold creation from the σ₃P₅ agreement dataset."""

import argparse
import os
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import StratifiedKFold

load_dotenv()
console = Console()

AGREEMENT_CSV = os.getenv(
    "AGREEMENT_CSV",
    "../multimodal-LLMs-see-sentiment/data/agreement_p5_sigma3.csv",
)
IMAGES_DIR = os.getenv("PERCEPTSENT_IMAGES_DIR", "../perceptsent/images")
OUTPUT_DIR = Path("data/folds")
ALL_TRAIN_CSV = Path("data/all_train.csv")

SENTIMENT_LABELS: dict[Hashable, str] = {
    0: "negative",
    1: "slightly_negative",
    2: "neutral",
    3: "slightly_positive",
    4: "positive",
}


def load_and_validate(agreement_csv: str, images_dir: str) -> pd.DataFrame:
    df = pd.read_csv(agreement_csv, index_col=0)
    df = df.rename(columns={"id": "image_id"})

    total = len(df)
    images_path = Path(images_dir)
    df["image_path"] = df["image_id"].apply(lambda x: str(images_path / f"{x}.jpg"))
    exists = df["image_path"].apply(lambda p: Path(p).exists())
    missing = (~exists).sum()

    if missing > 0:
        console.print(f"[yellow]Dropping {missing}/{total} images not found on disk.[/yellow]")
        df = df[exists].reset_index(drop=True)

    console.print(f"[green]Dataset: {len(df)} valid images (dropped {missing})[/green]")
    return df


def check_class_balance(folds: list[tuple[Any, Any]], labels: pd.Series, n_folds: int) -> None:
    table = Table(title="Class distribution per fold (val set)")
    table.add_column("Fold")
    for c in sorted(labels.unique()):
        table.add_column(f"class_{c}")
    table.add_column("total")

    for fold_idx, (_, val_idx) in enumerate(folds):
        val_labels = labels.iloc[val_idx]
        counts = val_labels.value_counts().sort_index()
        row = [str(fold_idx + 1)]
        for c in sorted(labels.unique()):
            row.append(str(counts.get(c, 0)))
        row.append(str(len(val_idx)))
        table.add_row(*row)

    console.print(table)

    global_dist = labels.value_counts(normalize=True).sort_index()
    for fold_idx, (_, val_idx) in enumerate(folds):
        fold_dist = labels.iloc[val_idx].value_counts(normalize=True).sort_index()
        for c in global_dist.index:
            delta = abs(global_dist[c] - fold_dist.get(c, 0.0))
            if delta > 0.05:
                console.print(
                    f"[red]WARNING fold {fold_idx + 1} class {c}: "
                    f"delta={delta:.3f} > 0.05 threshold[/red]"
                )


def create_folds(
    df: pd.DataFrame, n_folds: int = 5, seed: int = 42
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_splits = list(skf.split(df, df["sentiment"]))
    check_class_balance(fold_splits, df["sentiment"], n_folds)

    results = []
    for train_idx, val_idx in fold_splits:
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        results.append((train_df, val_df))

    return results


def save_folds(folds: list[tuple[pd.DataFrame, pd.DataFrame]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold_idx, (train_df, val_df) in enumerate(folds):
        train_df.to_csv(output_dir / f"fold_{fold_idx + 1}_train.csv", index=False)
        val_df.to_csv(output_dir / f"fold_{fold_idx + 1}_val.csv", index=False)
        console.print(f"  fold {fold_idx + 1}: {len(train_df)} train / {len(val_df)} val")


def log_to_mlflow(df: pd.DataFrame, n_folds: int) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_TRAINING", "culture_mllm_training"))
        with mlflow.start_run(run_name="data_fold_creation"):
            mlflow.log_param("n_images", len(df))
            mlflow.log_param("n_folds", n_folds)
            for c, count in df["sentiment"].value_counts().sort_index().items():
                mlflow.log_metric(f"class_{c}_count", count)
            mlflow.log_artifact(str(ALL_TRAIN_CSV))
    except Exception as exc:
        console.print(f"[yellow]MLflow logging skipped: {exc}[/yellow]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create stratified k-folds for CultureVLM training."
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agreement-csv", default=AGREEMENT_CSV)
    parser.add_argument("--images-dir", default=IMAGES_DIR)
    args = parser.parse_args()

    console.rule("[bold blue]CultureVLM — K-Fold Creation[/bold blue]")

    df = load_and_validate(args.agreement_csv, args.images_dir)

    ALL_TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ALL_TRAIN_CSV, index=False)
    console.print(f"[green]Saved all_train.csv: {len(df)} rows → {ALL_TRAIN_CSV}[/green]")

    console.print("\nGlobal sentiment distribution:")
    for c, count in df["sentiment"].value_counts().sort_index().items():
        console.print(f"  {c} ({SENTIMENT_LABELS[c]}): {count} ({count / len(df) * 100:.1f}%)")

    folds = create_folds(df, n_folds=args.n_folds, seed=args.seed)
    save_folds(folds, OUTPUT_DIR)
    log_to_mlflow(df, args.n_folds)

    console.print(f"\n[bold green]✓ {args.n_folds} folds saved to {OUTPUT_DIR}/[/bold green]")


if __name__ == "__main__":
    main()
