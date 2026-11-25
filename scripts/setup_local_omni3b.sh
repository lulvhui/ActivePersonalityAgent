#!/usr/bin/env bash
# Conda bootstrap and dedicated project environment; base holds no project packages.
set -euo pipefail
AMP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
AMP_CONDA_ROOT="${AMP_CONDA_ROOT:-$HOME/miniconda3}"
AMP_ENV="$AMP_CONDA_ROOT/envs/active-personality"
AMP_DOWNLOADS="$AMP_ROOT/.downloads"
AMP_TORCH_INDEX="${AMP_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
mkdir -p "$AMP_DOWNLOADS" "$AMP_ROOT/runs/conda_setup"
exec > >(tee -a "$AMP_ROOT/runs/conda_setup/install.log") 2>&1
[[ "$(uname -m)" == x86_64 ]] || { echo 'Linux x86_64 required.'; exit 1; }
if [[ ! -x "$AMP_CONDA_ROOT/bin/conda" ]]; then
    [[ ! -e "$AMP_CONDA_ROOT" ]] || { echo "Existing non-Conda directory: $AMP_CONDA_ROOT"; exit 1; }
    AMP_INSTALLER=Miniconda3-latest-Linux-x86_64.sh
    AMP_URL=https://repo.anaconda.com/miniconda
    curl --fail --location --retry 3 --connect-timeout 20 --speed-limit 1024 --speed-time 30 --continue-at - -o "$AMP_DOWNLOADS/$AMP_INSTALLER" "$AMP_URL/$AMP_INSTALLER"
    curl --fail --location --retry 3 --connect-timeout 20 -o "$AMP_DOWNLOADS/miniconda-index.html" "$AMP_URL/"
    /usr/bin/python3 - "$AMP_DOWNLOADS" "$AMP_INSTALLER" <<'CHECKSUM'
import re
import sys
from pathlib import Path
folder, name = Path(sys.argv[1]), sys.argv[2]
html = (folder / "miniconda-index.html").read_text()
row = next(row for row in re.findall(r"<tr[\s>].*?</tr>", html, re.S) if f'"{name}"' in row)
checksum = re.search(r"\b[0-9a-f]{64}\b", row).group()
(folder / (name + ".sha256")).write_text(f"{checksum}  {name}\n")
CHECKSUM
    (cd "$AMP_DOWNLOADS" && sha256sum --check "$AMP_INSTALLER.sha256")
    bash "$AMP_DOWNLOADS/$AMP_INSTALLER" -b -p "$AMP_CONDA_ROOT"
fi
if [[ ! -f "$AMP_ENV/conda-meta/history" ]]; then
    "$AMP_CONDA_ROOT/bin/conda" create -y -p "$AMP_ENV" --override-channels -c conda-forge python=3.10 pip ffmpeg
fi
export PATH="$AMP_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$AMP_ROOT/cache/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
"$AMP_ENV/bin/python" -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url "$AMP_TORCH_INDEX"
"$AMP_ENV/bin/python" -m pip install -r "$AMP_ROOT/requirements-local-omni3b.txt"
"$AMP_ENV/bin/python" -m pip install -e "$AMP_ROOT[jev]"
"$AMP_ENV/bin/python" -m pip check
"$AMP_ENV/bin/python" -m pip freeze > "$AMP_ROOT/runs/conda_setup/packages.txt"
"$AMP_CONDA_ROOT/bin/conda" config --set auto_activate_base false
"$AMP_CONDA_ROOT/bin/conda" init bash
PYTHON_BIN="$AMP_ENV/bin/python" bash "$AMP_ROOT/scripts/test_local_omni3b.sh" --check-only
printf '\nReady: source %s/etc/profile.d/conda.sh && conda activate active-personality\n' "$AMP_CONDA_ROOT"
