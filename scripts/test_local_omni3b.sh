#!/usr/bin/env bash
set -euo pipefail
AMP_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$AMP_PROJECT_ROOT/scripts/python_env.sh"
cd -- "$AMP_PROJECT_ROOT"
exec "$AMP_PYTHON" -u -m amp.local_test "$@"
