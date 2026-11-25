#!/usr/bin/env bash
set -euo pipefail
AMP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AMP_ROOT"
# Optional local credentials file; .env is ignored by the repository.
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
source "$PWD/scripts/python_env.sh"
exec "$AMP_PYTHON" scripts/test_api_teacher.py "$@"
