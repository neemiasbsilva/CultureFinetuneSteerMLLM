#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'EOF'
06_lint.sh — Static checks over the tracked Python sources.

Runs, in order:
  ruff format --check   formatting
  ruff check            lint rules E, F, I, UP, B, SIM, RUF
  mypy                  strict type checking

Only tracked files are checked, so untracked scratch work never fails the run.

Usage:
  ./scripts/06_lint.sh
EOF
}


case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

require_uv

status=0

log "ruff format --check"
tracked_python_files | xargs -0 uv run ruff format --check || status=1

log "ruff check"
tracked_python_files | xargs -0 uv run ruff check || status=1

log "mypy"
tracked_python_files | xargs -0 uv run mypy || status=1

if [ "$status" -ne 0 ]; then
    die "static checks failed"
fi

log "all static checks passed"
