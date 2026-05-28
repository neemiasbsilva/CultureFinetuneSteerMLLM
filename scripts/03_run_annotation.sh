#!/usr/bin/env bash
# Stage 3 — Cultural VLM annotation
#
# Runs annotation on ALL 3,565 σ₃P₅ images using the final trained models.
# No persona injection — cultural bias is in model weights.
#
# ENV overrides:
#   MODELS="qwen_vl phi4 gemma4"   — architectures to annotate with
#   CULTURES="arabic chinese ..."  — subset of cultures
#   CONDITIONS="cultural baseline" — trained adapter conditions to run
#   LIMIT=10                       — cap images (for smoke testing)
#   N_RUNS=5                       — independent passes per image for variance

set -euo pipefail

# MODELS="${MODELS:-qwen_vl gemma4 phi4}"
MODELS="${MODELS:-qwen_vl}"
CULTURES="${CULTURES:-arabic bengali chinese english german korean portuguese spanish turkish}"
CONDITIONS="${CONDITIONS:-cultural baseline}"
LIMIT="${LIMIT:-}"
N_RUNS="${N_RUNS:-3}"

echo "=== CultureVLM: Annotation Pipeline ==="
echo "  Trained conditions: $CONDITIONS"
echo "  Independent runs per image: $N_RUNS"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
  LIMIT_ARG="--limit $LIMIT"
  echo "  [SMOKE TEST] Limit: $LIMIT images"
fi

for MODEL in $MODELS; do
  for CULTURE in $CULTURES; do
    for CONDITION in $CONDITIONS; do
      echo "  Annotating: $MODEL / $CULTURE / $CONDITION"
      uv run python src/annotation/pipeline.py \
        --culture "$CULTURE" \
        --condition "$CONDITION" \
        --model-name "$MODEL" \
        --n-runs "$N_RUNS" \
        $LIMIT_ARG
    done
  done
done

echo ""
echo "=== Annotation complete ==="
echo "Outputs in: outputs/annotations/"
