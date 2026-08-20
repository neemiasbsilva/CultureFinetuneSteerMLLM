#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

die() { echo "ERROR: $*" >&2; exit 1; }

log() { echo "[$(date '+%H:%M:%S')] $*"; }

require_uv() {
    command -v uv >/dev/null 2>&1 \
        || die "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
}

tracked_python_files() {
    git ls-files -z '*.py' | while IFS= read -r -d '' path; do
        [ -f "$path" ] && printf '%s\0' "$path"
    done
}
