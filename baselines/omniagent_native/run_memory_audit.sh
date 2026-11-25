#!/usr/bin/env bash
set -euo pipefail
cd /mnt/r6_hdd4t/lhlu/ActivePersonalityAgent
source "$PWD/scripts/python_env.sh"
AMP_NATIVE_PYTHON="${AMP_NATIVE_PYTHON:-$AMP_CONDA_ROOT/envs/omniagent-native/bin/python}"
output=runs/memory_audit_20261120
mkdir -p "$output"
exec 9>/tmp/lhlu_omniagent_native_gpu.lock
flock -n 9 || { echo 'Another comparison holds the GPU lock.'; exit 1; }
export PATH="$PWD/.local-tools/ffmpeg/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
"$AMP_PYTHON" - <<'PY'
import subprocess
for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines():
    gpu,used=map(int,line.split(','))
    if gpu in (0,1) and used>1536:raise RuntimeError(f'GPU {gpu} is busy ({used} MiB); refusing overlapping loads')
PY
CUDA_VISIBLE_DEVICES=0 "$AMP_NATIVE_PYTHON" -u baselines/omniagent_native/audit_memory.py kernel \
  --output "$output/native_sdpa_kernel.json" > "$output/kernel.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$AMP_PYTHON" -u baselines/omniagent_native/audit_memory.py model \
  --model /mnt/r6_hdd4t/lhlu/models/Qwen2.5-Omni-7B --output "$output/base7b" > "$output/base7b.log" 2>&1 &
base_pid=$!
CUDA_VISIBLE_DEVICES=1 "$AMP_PYTHON" -u baselines/omniagent_native/audit_memory.py model \
  --model /mnt/r6_hdd4t/lhlu/models/OmniAgent-RL-7B --output "$output/omni7b" > "$output/omni7b.log" 2>&1 &
omni_pid=$!
status=0
wait "$base_pid" || status=1
wait "$omni_pid" || status=1
exit "$status"
