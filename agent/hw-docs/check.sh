#!/usr/bin/env bash
# check.sh — thin entrypoint; the linter itself is check.py (same dir).
# Used by CI and by hand:  ./check.sh [--offline]  (see check.py --help)
set -euo pipefail
PY=python3; command -v python3 >/dev/null 2>&1 || PY=python
exec "$PY" "$(dirname "$0")/check.py" "$@"
