#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
01_prepare_data.sh — Stage 1, data preparation.

Builds, in order:
  stratified image folds   consumed by Stage 4 evaluation against the sigma3-P5 labels
  WVS cultural anchoring   the fine-tuning input for Stage 2, cultural and baseline variants

Fine-tuning is text-only WVS question/answer material; no images or visual SFT
are built here. Images enter only at annotation (Stage 3) and evaluation (Stage 4).

ENV overrides:
  FOLDS=5    number of stratified folds
  SEED=42    fold assignment seed

Usage:
  ./scripts/01_prepare_data.sh
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

FOLDS="${FOLDS:-5}"
SEED="${SEED:-42}"

echo "=== CultureVLM: Data Preparation ==="

echo "[1/2] Creating stratified image folds (for Stage 4 evaluation)..."
uv run python src/data/make_folds.py --n-folds "$FOLDS" --seed "$SEED"

echo "[2/2] Loading WVS cultural anchoring data (cultural + baseline variants)..."
uv run python src/data/culture_training_data.py --all

echo "=== Data preparation complete ==="
echo ""
echo "Next: bash scripts/02_train_culture_models.sh"
