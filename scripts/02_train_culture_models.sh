#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
02_train_culture_models.sh — Stage 2, WVS text-only LoRA fine-tuning.

For each architecture, trains one cultural model per culture from WVS
question/answer material under a culture-specific system prompt.

Resume behaviour:
  TRAINING_DONE sentinel present   the model/culture pair is skipped
  interrupted (MLflow run FAILED)  resumes from the last HF checkpoint, opens a fresh run
  killed (MLflow run RUNNING)      the same run is reopened and training continues

Backend selection reads model.backend from the config:
  mlx   train_mlx.py, Apple Silicon only; falls back to train_hf.py on CUDA machines
  hf    train_hf.py, CUDA or MPS, auto-detected

On Ctrl+C the Python subprocess marks its MLflow run FAILED before this script
exits 130, so a re-run resumes rather than starting over.

ENV overrides:
  MODELS="gemma4_e2b phi4 ..."       which architectures to train
  CULTURES="arabic chinese english"  subset of cultures

Usage:
  ./scripts/02_train_culture_models.sh
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

MODELS="${MODELS:-gemma4_e2b gemma4_31b gemma4_e4b qwen3_5_2b qwen3_27b qwen3_vl_8b}"
CULTURES="${CULTURES:-arabic bengali chinese english german korean portuguese spanish spanish-mx turkish}"

trap 'echo ""; echo "Interrupted. Re-run this script to resume from the last checkpoint."; exit 130' INT

echo "=== culture-mllm: WVS Fine-Tuning ==="
echo "Models     : $MODELS"
echo "Cultures   : $CULTURES"

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
