#!/usr/bin/env bash
# Stage 2 — WVS text-only LoRA fine-tuning (cultural condition only)
#
# For each architecture, trains one cultural model per culture using
# WVS Q&A with a culture-specific system prompt.
#
# Resume behaviour:
#   - If training was completed for a model/culture (TRAINING_DONE sentinel
#     exists), that combination is skipped automatically.
#   - If training was interrupted (Ctrl+C → MLflow run marked FAILED), the
#     script resumes from the last HF checkpoint and opens a fresh MLflow run.
#   - If the process was killed (SIGKILL → MLflow run left as RUNNING), the
#     same MLflow run is reopened and training continues from the checkpoint.
#
# Backend selection:
#   - "mlx"  → train_mlx.py  (Apple Silicon only)
#   - "hf"   → train_hf.py   (CUDA or MPS, auto-detected)
# MLX configs automatically fall back to train_hf.py on CUDA machines.
#
# ENV overrides:
#   MODELS="gemma4_e2b phi4 ..."      — which architectures to train
#   CULTURES="arabic chinese english" — subset of cultures

set -euo pipefail

MODELS="${MODELS:-gemma4_e2b gemma4_31b gemma4_e4b qwen3_5_2b qwen3_vl_30b qwen3_vl_8b phi4}"
CULTURES="${CULTURES:-arabic bengali chinese english german korean portuguese spanish turkish}"

# On Ctrl+C: let the Python subprocess handle MLflow cleanup (marks run FAILED),
# then exit with the standard interrupt code so the user can re-run the script.
trap 'echo ""; echo "Interrupted. Re-run this script to resume from the last checkpoint."; exit 130' INT

echo "=== culture-mllm: WVS Fine-Tuning ==="
echo "Models     : $MODELS"
echo "Cultures   : $CULTURES"

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
    DONE_FILE="checkpoints/${CULTURE}/${MODEL}/cultural/TRAINING_DONE"
    if [ -f "$DONE_FILE" ]; then
      echo "  Skipping $MODEL / $CULTURE — already complete"
      continue
    fi
    echo "  Training: $MODEL / $CULTURE / cultural"
    $TRAIN_CMD --config "$CONFIG" --culture "$CULTURE"
  done
  echo ""
done

echo "=== Fine-tuning complete ==="
echo "Next: bash scripts/03_run_annotation.sh"
