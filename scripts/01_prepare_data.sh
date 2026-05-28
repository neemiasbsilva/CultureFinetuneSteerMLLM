#!/usr/bin/env bash
# Stage 1 — Data preparation
#
# Steps:
#   1. Create stratified image folds (used in Stage 4 evaluation against σ₃P₅ labels)
#   2. Load WVS cultural anchoring text data (fine-tuning input for Stage 2)
#
# Note: Fine-tuning is text-only (WVS Q&A). No images or visual SFT are built here.
# Images are used only during annotation (Stage 3) and evaluation (Stage 4).
set -euo pipefail

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
