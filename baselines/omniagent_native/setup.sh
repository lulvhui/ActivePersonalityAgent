#!/usr/bin/env bash
set -euo pipefail
cd /mnt/r6_hdd4t/lhlu/ActivePersonalityAgent
AMP_CONDA_ROOT="${AMP_CONDA_ROOT:-$HOME/miniconda3}"
AMP_NATIVE_ENV="$AMP_CONDA_ROOT/envs/omniagent-native"
upstream=/mnt/r6_hdd4t/lhlu/OmniAgent
if [[ ! -f "$AMP_NATIVE_ENV/conda-meta/history" ]]; then
  "$AMP_CONDA_ROOT/bin/conda" create -y -p "$AMP_NATIVE_ENV" --override-channels -c conda-forge python=3.10 pip
fi
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${AMP_NATIVE_PIP_CACHE:-$HOME/.cache/pip}"
"$AMP_NATIVE_ENV/bin/python" -m pip install 'setuptools<81' torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --extra-index-url https://download.pytorch.org/whl/cu126
"$AMP_NATIVE_ENV/bin/python" -m pip install -r baselines/omniagent_native/requirements-inference.txt
"$AMP_NATIVE_ENV/bin/python" -m pip install -e "$upstream/qwen-vl-utils" -e "$upstream/qwen-omni-utils"
"$AMP_NATIVE_ENV/bin/python" -m pip freeze > baselines/omniagent_native/installed-requirements.txt
