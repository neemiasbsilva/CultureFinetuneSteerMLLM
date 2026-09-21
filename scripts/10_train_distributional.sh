#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
10_train_distributional.sh — Stage 10, distributional LoRA fine-tuning (Cao et al. 2025).

One adapter per architecture, conditioned on 46 countries through the prompt and
trained to match country-level WVS-7 response distributions with a first-token
KL loss under the reference setup: prompt, loss, LoRA, optimiser, schedule and
data protocol. Steps, run in order unless STEPS narrows them:
  prepare  copy the SimLLMCultureDist splits under data/raw/distributional and
           write provenance.json (source commit, checksums, counts, attribution)
  train    one adapter per model into checkpoints/global/<model>/distributional
  eval     score the held-out test split, base against adapter

Resume behaviour matches stage 02: TRAINING_DONE skips a finished model, an
interrupted run resumes from its last checkpoint into the same MLflow run.

ENV overrides:
  MODELS="qwen3_vl_2b ..."          architectures; each needs configs/<model>_distributional.yaml
  SOURCE_DIR=../SimLLMCultureDist   the reference clone the splits are copied from
  STEPS="prepare train eval"        subset of steps
  DEBUG=1                           one epoch over a small slice into a *_debug checkpoint leaf
  FORCE_EVAL=1                      re-run eval even if heldout.json exists

Usage:
  ./scripts/10_train_distributional.sh
  MODELS="qwen3_vl_2b" STEPS="train" ./scripts/10_train_distributional.sh
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

require_uv

MODELS="${MODELS:-qwen3_vl_2b}"
SOURCE_DIR="${SOURCE_DIR:-../SimLLMCultureDist}"
STEPS="${STEPS:-prepare train eval}"
DEBUG="${DEBUG:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

# This card is shared with the sibling repo's inference runs, so the allocator
# keeps refilling a fragmented pool. Expandable segments give the blocks back.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

trap 'echo ""; echo "Interrupted. Re-run this script to resume from the last checkpoint."; exit 130' INT

has_step() { [[ " $STEPS " == *" $1 "* ]]; }

log "=== culture-mllm: Distributional Fine-Tuning (Cao et al. 2025) ==="
log "Models : $MODELS"
log "Steps  : $STEPS"
log "Debug  : $DEBUG"

if has_step prepare; then
  uv run python src/data/distributional_data.py prepare --source-dir "$SOURCE_DIR"
fi

DEBUG_FLAG=()
if [ "$DEBUG" = "1" ]; then
  DEBUG_FLAG=(--debug)
fi

for MODEL in $MODELS; do
  CONFIG="configs/${MODEL}_distributional.yaml"
  if [ ! -f "$CONFIG" ]; then
    log "Config not found: $CONFIG — skipping"
    continue
  fi

  if has_step train; then
    DONE_FILE="checkpoints/global/${MODEL}/distributional/TRAINING_DONE"
    if [ -f "$DONE_FILE" ] && [ "$DEBUG" != "1" ]; then
      log "Skipping $MODEL — already complete"
    else
      log "--- Training: $MODEL / global / distributional ---"
      uv run python src/training/train_distributional.py --config "$CONFIG" "${DEBUG_FLAG[@]}"
    fi
  fi

  if has_step eval; then
    REPORT="outputs/evaluation/distributional/${MODEL}/heldout.json"
    if [ -f "$REPORT" ] && [ "$FORCE_EVAL" != "1" ]; then
      log "Skipping eval for $MODEL — $REPORT exists (FORCE_EVAL=1 to redo)"
    else
      log "--- Held-out evaluation: $MODEL ---"
      uv run python src/evaluation/distributional_eval.py --config "$CONFIG" "${DEBUG_FLAG[@]}"
    fi
  fi
done

log "=== Distributional track complete ==="
echo "Next: CULTURES=global MODELS=\"$MODELS\" PROBLEM=distributional ./scripts/09_push_hub.sh push"
