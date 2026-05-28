#!/usr/bin/env bash
# Stage 3b — Raw base-MLLM annotation, no fine-tuning adapter
#
# Runs the same neutral annotation prompt on the unadapted base VLMs.
# This condition is the no-training reference needed to separate:
#   1) base multimodal sentiment ability,
#   2) generic WVS fine-tuning effects, and
#   3) culture-specific identity-framing effects.
#
# ENV overrides:
#   MODELS="qwen_vl phi4 gemma4"   — architectures to annotate with
#   LIMIT=10                       — cap images (for smoke testing)
#   N_RUNS=5                       — independent passes per image for variance

set -euo pipefail

# MODELS="${MODELS:-qwen_vl gemma4 phi4}"
MODELS="${MODELS:-qwen_vl}"
LIMIT="${LIMIT:-}"
N_RUNS="${N_RUNS:-3}"

echo "=== CultureVLM: Inference-Only Annotation Pipeline ==="
echo "  Condition: inference_only (no LoRA adapter loaded)"
echo "  Independent runs per image: $N_RUNS"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
  LIMIT_ARG="--limit $LIMIT"
  echo "  [SMOKE TEST] Limit: $LIMIT images"
fi

for MODEL in $MODELS; do
  echo "  Annotating raw base model: $MODEL"
  uv run python src/annotation/pipeline.py \
    --culture inference_only \
    --condition inference_only \
    --model-name "$MODEL" \
    --n-runs "$N_RUNS" \
    $LIMIT_ARG
done

echo ""
echo "=== Inference-only annotation complete ==="
echo "Outputs in: outputs/annotations/{model}/inference_only/"
