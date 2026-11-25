#!/usr/bin/env bash
set -euo pipefail
cd /mnt/r6_hdd4t/lhlu/ActivePersonalityAgent
source "$PWD/scripts/python_env.sh"
AMP_NATIVE_PYTHON="${AMP_NATIVE_PYTHON:-$AMP_CONDA_ROOT/envs/omniagent-native/bin/python}"
run_dir="${1:-runs/omniagent_expanded_20261117}"
export PATH="$PWD/.local-tools/ffmpeg/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
"$AMP_PYTHON" - <<'PY'
import subprocess
for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines():
    gpu,used=map(int,line.split(','))
    if gpu==0 and used>1536:raise RuntimeError(f'GPU 0 uses {used} MiB; refusing overlap')
PY
CUDA_VISIBLE_DEVICES=0 "$AMP_PYTHON" -u baselines/omniagent_native/run_ours.py \
  --manifest "$run_dir/data/manifest.jsonl" --model /mnt/r6_hdd4t/lhlu/models/OmniAgent-RL-7B \
  --output "$run_dir/omniagent_transfer" > "$run_dir/omniagent_transfer_parallel.log" 2>&1
