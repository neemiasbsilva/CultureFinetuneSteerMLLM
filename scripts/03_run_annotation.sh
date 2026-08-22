#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
03_run_annotation.sh — Stage 3, matched annotation over all 3,565 sigma3-P5 images.

Two conditions per architecture:
  inference_only   raw base model, no LoRA adapter — the no-training reference
  wvs_cultural     the culture-specific WVS adapter, stored in the legacy cultural/ dir

Both conditions share the same N_RUNS and LIMIT, so their results are directly
comparable. No persona is injected at inference; cultural bias lives in the
adapter weights alone.

The default MODELS order runs smallest to largest, so fast models produce results
first and the OOM-prone giants run last.

ENV overrides:
  MODELS="qwen3_5_2b gemma4_e2b ..."          architectures, order preserved
  CULTURES="arabic chinese ..."               subset for the trained conditions
  CONDITIONS="inference_only wvs_cultural"
  LIMIT=10                                    cap images, for smoke testing
  N_RUNS=5                                    independent passes per image

Usage:
  ./scripts/03_run_annotation.sh
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

MODELS="${MODELS:-gemma4_e2b gemma4_e4b qwen3_vl_8b gemma4_31b}"
CULTURES="${CULTURES:-arabic bengali chinese english german korean portuguese spanish spanish-mx turkish}"
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
