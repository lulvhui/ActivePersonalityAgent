#!/usr/bin/env bash
# Source from project launchers. Never fall back to venv or system Python.
AMP_CONDA_ROOT="${AMP_CONDA_ROOT:-$HOME/miniconda3}"
AMP_PYTHON="${PYTHON_BIN:-$AMP_CONDA_ROOT/envs/active-personality/bin/python}"
if [[ ! -x "$AMP_PYTHON" ]]; then
    printf 'Conda Python missing: %s\nRun scripts/setup_local_omni3b.sh first.\n' "$AMP_PYTHON" >&2
    exit 1
fi
export PATH="$(dirname -- "$AMP_PYTHON"):$PATH"
export PYTHONNOUSERSITE=1
