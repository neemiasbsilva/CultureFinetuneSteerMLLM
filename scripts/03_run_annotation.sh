#!/usr/bin/env bash
# Stage 3 — Cultural VLM annotation
#
# Runs annotation on ALL 3,565 σ₃P₅ images using the cultural-condition adapters.
# No persona injection — cultural bias is in model weights.
#
# ENV overrides:
#   MODELS="qwen3_5_2b phi4 gemma4_e2b"   — architectures to annotate with
#   CULTURES="arabic chinese ..."  — subset of cultures
#   LIMIT=10                       — cap images (for smoke testing)
#   N_RUNS=5                       — independent passes per image for variance

set -euo pipefail

# MODELS="${MODELS:-qwen3_5_2b gemma4_e2b phi4 gemma4_e4b gemma4_31b qwen3_vl_8b qwen3_vl_30b}"
MODELS="${MODELS:-qwen3_5_2b gemma4_e2b phi4 gemma4_e4b gemma4_31b qwen3_vl_8b qwen3_vl_30b}"
CULTURES="${CULTURES:-arabic bengali chinese english german korean portuguese spanish turkish}"
LIMIT="${LIMIT:-}"
N_RUNS="${N_RUNS:-3}"

echo "=== CultureVLM: Annotation Pipeline ==="
echo "  Condition: cultural"
echo "  Independent runs per image: $N_RUNS"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
  LIMIT_ARG="--limit $LIMIT"
  echo "  [SMOKE TEST] Limit: $LIMIT images"
fi

for MODEL in $MODELS; do
  for CULTURE in $CULTURES; do
    echo "  Annotating: $MODEL / $CULTURE / cultural"
    uv run python src/annotation/pipeline.py \
      --culture "$CULTURE" \
      --condition cultural \
      --model-name "$MODEL" \
      --n-runs "$N_RUNS" \
      $LIMIT_ARG
  done
done

echo ""
echo "=== Annotation complete ==="
echo "Outputs in: outputs/annotations/"
