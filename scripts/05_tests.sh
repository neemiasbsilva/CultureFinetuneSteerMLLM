#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
05_tests.sh — Regression tests over the annotation and evaluation machinery.

tests/unit/  — the condition registry and checkpoint layout, the annotation
               graph wiring, repetition aggregation, the matched-seed gate,
               Holm adjustment, fold statistics, dataset and fold builders,
               early stopping, and base-model loading

Usage:
  ./scripts/05_tests.sh [pytest args...]

Example:
  ./scripts/05_tests.sh -k condition -vv
EOF
}


case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

require_uv
uv run pytest tests "$@"
