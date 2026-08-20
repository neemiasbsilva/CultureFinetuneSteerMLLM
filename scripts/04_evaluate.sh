#!/usr/bin/env bash
# Stage 4 — matched annotation evaluation against sigma3-P5 ground truth.
#
# Aggregates repeated runs per image, restricts every comparison to a common
# valid image set, and writes paired bootstrap and fold/Holm results.
#
# ENV overrides: CONDITIONS, N_BOOTSTRAP, SEED, METRIC.

set -euo pipefail

CONDITIONS="${CONDITIONS:-inference_only wvs_cultural}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
SEED="${SEED:-42}"
METRIC="${METRIC:-f1_macro}"
TEST_TRAIN_RATIO="${TEST_TRAIN_RATIO:-}"
CAPTION_EMBEDDING_MODEL="${CAPTION_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-}"
EXPECTED_RUNS="${EXPECTED_RUNS:-5}"
CULTURES="${CULTURES:-arabic bengali chinese english german korean portuguese spanish turkish}"
MODELS="${MODELS:-gemma4_e2b gemma4_e4b qwen3_vl_8b gemma4_31b}"

echo "=== CultureVLM: Matched Annotation Evaluation ==="
echo "  Conditions : $CONDITIONS"
echo "  Bootstrap  : $N_BOOTSTRAP"
echo "  Fold metric: $METRIC"

ARGS=(
  --conditions $CONDITIONS
  --n-bootstrap "$N_BOOTSTRAP"
  --seed "$SEED"
  --metric "$METRIC"
  --caption-embedding-model "$CAPTION_EMBEDDING_MODEL"
  --expected-runs "$EXPECTED_RUNS"
  --expected-cultures $CULTURES
  --models $MODELS
)
if [ -n "$TEST_TRAIN_RATIO" ]; then
  ARGS+=(--test-train-ratio "$TEST_TRAIN_RATIO")
fi
if [ -n "$EMBEDDING_DEVICE" ]; then
  ARGS+=(--embedding-device "$EMBEDDING_DEVICE")
fi

uv run python src/evaluation/annotation_eval.py \
  "${ARGS[@]}"

echo ""
echo "=== Evaluation complete ==="
echo "Results in: outputs/evaluation/"
