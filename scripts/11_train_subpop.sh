#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
11_train_subpop.sh — Stage 11, SubPOP LoRA fine-tuning (Suh et al. 2025).

One adapter per architecture, conditioned on 22 US subpopulations through a QA
steering prompt and trained to match SubPOP-Train response distributions with a
first-token forward-KL loss under the reference setup: prompt, loss, LoRA,
optimiser, schedule and by-question split. Steps, run in order unless STEPS
narrows them:
  prepare  fetch the gated jjssuh/subpop tables at the pinned revision, copy the
           steering prompts from the reference clone under data/raw/subpop and
           write provenance.json (revision, commit, checksums, counts, attribution)
  train    one adapter per model into checkpoints/subpop/<model>/subpop
  eval     score SubPOP-Eval, base against adapter

Resume behaviour matches stage 02: TRAINING_DONE skips a finished model, an
interrupted run resumes from its last checkpoint into the same MLflow run.

ENV overrides:
  MODELS="qwen3_vl_2b ..."   architectures; each needs configs/<model>_subpop.yaml
  SOURCE_DIR=../subpop       the reference clone the steering prompts are copied from
  STEPS="prepare train eval" subset of steps
  DEBUG=1                    one epoch over a small slice into a *_debug checkpoint leaf
  FORCE_EVAL=1               re-run eval even if heldout.json exists

Usage:
  ./scripts/11_train_subpop.sh
  MODELS="qwen3_vl_2b" STEPS="train" ./scripts/11_train_subpop.sh
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

require_uv

MODELS="${MODELS:-qwen3_vl_2b}"
SOURCE_DIR="${SOURCE_DIR:-../subpop}"
STEPS="${STEPS:-prepare train eval}"
DEBUG="${DEBUG:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

# This card is shared with the sibling repo's inference runs, so the allocator
# keeps refilling a fragmented pool. Expandable segments give the blocks back.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

trap 'echo ""; echo "Interrupted. Re-run this script to resume from the last checkpoint."; exit 130' INT

has_step() { [[ " $STEPS " == *" $1 "* ]]; }

log "=== culture-mllm: SubPOP Fine-Tuning (Suh et al. 2025) ==="
log "Models : $MODELS"
log "Steps  : $STEPS"
log "Debug  : $DEBUG"

if has_step prepare; then
  uv run python src/data/subpop_data.py prepare --source-dir "$SOURCE_DIR"
fi

DEBUG_FLAG=()
LEAF="subpop"
REPORT_SUFFIX=""
if [ "$DEBUG" = "1" ]; then
  DEBUG_FLAG=(--debug)
  LEAF="subpop_debug"
  REPORT_SUFFIX="_debug"
fi

for MODEL in $MODELS; do
  CONFIG="configs/${MODEL}_subpop.yaml"
  if [ ! -f "$CONFIG" ]; then
    log "Config not found: $CONFIG — skipping"
    continue
  fi

  if has_step train; then
    DONE_FILE="checkpoints/subpop/${MODEL}/subpop/TRAINING_DONE"
    if [ -f "$DONE_FILE" ] && [ "$DEBUG" != "1" ]; then
      log "Skipping $MODEL — already complete"
    else
      log "--- Training: $MODEL / subpop / $LEAF ---"
      uv run python src/training/train_subpop.py --config "$CONFIG" "${DEBUG_FLAG[@]}"
    fi
  fi

  if has_step eval; then
    REPORT="outputs/evaluation/subpop/${MODEL}${REPORT_SUFFIX}/heldout.json"
    if [ -f "$REPORT" ] && [ "$FORCE_EVAL" != "1" ]; then
      log "Skipping eval for $MODEL — $REPORT exists (FORCE_EVAL=1 to redo)"
    else
      log "--- Held-out evaluation: $MODEL ---"
      uv run python src/evaluation/subpop_eval.py --config "$CONFIG" "${DEBUG_FLAG[@]}"
    fi
  fi
done

log "=== SubPOP track complete ==="
echo "Next: stage checkpoints/subpop/<model>/subpop as the 'subpop' arm in ../machine-bias-reproduction"
