#!/usr/bin/env bash
# Stage 2 — WVS text-only LoRA fine-tuning
#
# For each architecture, trains:
#   - 9 cultural models  (WVS Q&A with culture-specific system prompt)
#   - 9 baseline models  (same WVS Q&A with neutral system prompt)
#
# Annotation in Stage 3 uses these adapters with a neutral inference prompt.
# Evaluation in Stage 4 compares annotation quality against σ₃P₅ labels.
#
# Backend selection:
#   - "mlx"  → train_mlx.py  (Apple Silicon only)
#   - "hf"   → train_hf.py   (CUDA or MPS, auto-detected)
# MLX configs automatically fall back to train_hf.py on CUDA machines.
#
# ENV overrides:
#   MODELS="qwen_vl gemma4 phi4 gemma4_e4b gemma4_31b qwen3_vl_8b qwen3_vl_30b"
#   CULTURES="arabic chinese english"     — subset of cultures
#   CONDITIONS="cultural baseline"        — which conditions to run

set -euo pipefail

MODELS="${MODELS:-gemma4 phi4 gemma4_e4b gemma4_31b qwen3_vl_8b qwen3_vl_30b}"
CULTURES="${CULTURES:-arabic bengali chinese english german korean portuguese spanish turkish}"
CONDITIONS="${CONDITIONS:-cultural baseline}"

echo "=== culture-mllm: WVS Fine-Tuning ==="
echo "Models     : $MODELS"
echo "Cultures   : $CULTURES"
echo "Conditions : $CONDITIONS"

# Detect if MLX is available (Apple Silicon only)
MLX_AVAILABLE=$(uv run python -c "import mlx" 2>/dev/null && echo "yes" || echo "no")
echo "MLX        : $MLX_AVAILABLE"
echo ""

for MODEL in $MODELS; do
  CONFIG="configs/${MODEL}.yaml"
  if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG — skipping"
    continue
  fi

  BACKEND=$(uv run python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model']['backend'])")

  if [ "$BACKEND" = "mlx" ] && [ "$MLX_AVAILABLE" = "no" ]; then
    TRAIN_CMD="uv run python src/training/train_hf.py"
    echo "--- Model: $MODEL (backend: mlx → falling back to HF+CUDA) ---"
  elif [ "$BACKEND" = "mlx" ]; then
    TRAIN_CMD="uv run python src/training/train_mlx.py"
    echo "--- Model: $MODEL (backend: mlx) ---"
  else
    TRAIN_CMD="uv run python src/training/train_hf.py"
    echo "--- Model: $MODEL (backend: hf) ---"
  fi

  for CULTURE in $CULTURES; do
    for CONDITION in $CONDITIONS; do
      echo "  Training: $MODEL / $CULTURE / $CONDITION"
      $TRAIN_CMD --config "$CONFIG" --culture "$CULTURE" --condition "$CONDITION"
    done
  done
  echo ""
done

echo "=== Fine-tuning complete ==="
echo "Next: bash scripts/03_run_annotation.sh"
