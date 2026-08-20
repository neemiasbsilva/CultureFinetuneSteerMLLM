#!/usr/bin/env bash
# Stage 3 — matched annotation: base model and WVS cultural adapters
#
# Annotates ALL 3,565 σ₃P₅ images. For each architecture (smallest → largest):
#   1. inference_only — raw base model, no LoRA adapter (no-training reference)
#   2. wvs_cultural   — culture-specific WVS adapter (legacy cultural/ dir)
# Both conditions share the same N_RUNS / LIMIT so results are directly
# comparable. No persona injection — cultural bias lives in adapter weights.
#
# ENV overrides:
#   MODELS="qwen3_5_2b gemma4_e2b ..."    — architectures (order preserved)
#   CULTURES="arabic chinese ..."         — subset for trained conditions
#   CONDITIONS="inference_only wvs_cultural"
#   LIMIT=10                              — cap images (for smoke testing)
#   N_RUNS=5                              — independent passes per image

set -euo pipefail

# Small → large: fast models produce results first; OOM-prone giants run last.
MODELS="${MODELS:-gemma4_e2b gemma4_e4b qwen3_vl_8b gemma4_31b}"

CULTURES="${CULTURES:-arabic bengali chinese english german korean portuguese spanish turkish}"
CONDITIONS="${CONDITIONS:-inference_only wvs_cultural}"
LIMIT="${LIMIT:-}"
N_RUNS="${N_RUNS:-5}"

echo "=== CultureVLM: Annotation Pipeline ==="
echo "  Models    : $MODELS"
echo "  Conditions: $CONDITIONS"
echo "  Runs/image: $N_RUNS"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
  LIMIT_ARG="--limit $LIMIT"
  echo "  [SMOKE TEST] Limit: $LIMIT images"
fi

for MODEL in $MODELS; do
  for CONDITION in $CONDITIONS; do
    if [ "$CONDITION" = "inference_only" ]; then
      echo "  Annotating: $MODEL / inference_only (raw base model)"
      uv run python src/annotation/pipeline.py \
        --culture inference_only \
        --condition inference_only \
        --model-name "$MODEL" \
        --n-runs "$N_RUNS" \
        $LIMIT_ARG
    else
      for CULTURE in $CULTURES; do
        echo "  Annotating: $MODEL / $CULTURE / $CONDITION"
        uv run python src/annotation/pipeline.py \
          --culture "$CULTURE" \
          --condition "$CONDITION" \
          --model-name "$MODEL" \
          --n-runs "$N_RUNS" \
          $LIMIT_ARG
      done
    fi
  done
done

echo ""
echo "=== Annotation complete ==="
echo "Outputs in: outputs/annotations/"
