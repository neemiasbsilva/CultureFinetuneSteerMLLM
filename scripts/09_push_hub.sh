#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
09_push_hub.sh — Stage 9, publish completed LoRA checkpoints to the
Hugging Face Hub collection (Neemias/Culture-Steering-MLLM-Collection).

Only checkpoints with a TRAINING_DONE sentinel are published, and only the
files a PeftModel.from_pretrained() call needs — training artifacts
(checkpoint-*/, training_args.bin, mlflow_run_id.txt) never leave the machine.
Each (culture, backbone, problem) triple lands under that same path prefix
inside the one collection repo.

ENV overrides:
  REPO_ID    Neemias/Culture-Steering-MLLM-Collection
  CULTURES   arabic bengali chinese english german korean portuguese spanish spanish-mx turkish
  MODELS     gemma4_e2b gemma4_e4b gemma4_31b qwen3_5_2b qwen3_27b qwen3_vl_8b
  PROBLEM    cultural
  DRY_RUN    1 to print the plan without uploading (default: 0)
  PRIVATE    1 to create the repo private (default: 0)

Usage:
  ./scripts/09_push_hub.sh status
  ./scripts/09_push_hub.sh push
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

COMMAND="${1:-status}"
REPO_ID="${REPO_ID:-Neemias/Culture-Steering-MLLM-Collection}"
CULTURES="${CULTURES:-arabic bengali chinese english german korean portuguese spanish spanish-mx turkish}"
MODELS="${MODELS:-gemma4_e2b gemma4_e4b gemma4_31b qwen3_5_2b qwen3_27b qwen3_vl_8b}"
PROBLEM="${PROBLEM:-cultural}"
DRY_RUN="${DRY_RUN:-0}"
PRIVATE="${PRIVATE:-0}"

echo "=== culture-mllm: Hugging Face Hub ($COMMAND) ==="
echo "  Repo     : $REPO_ID"
echo "  Cultures : $CULTURES"
echo "  Models   : $MODELS"
echo "  Problem  : $PROBLEM"

ARGS=(
  "$COMMAND"
  --repo-id "$REPO_ID"
  --cultures $CULTURES
  --models $MODELS
  --problem "$PROBLEM"
)

if [ "$COMMAND" = "push" ]; then
  if [ "$DRY_RUN" = "1" ]; then
    ARGS+=(--dry-run)
  fi
  if [ "$PRIVATE" = "1" ]; then
    ARGS+=(--private)
  fi
fi

uv run python src/hub/push.py "${ARGS[@]}"
