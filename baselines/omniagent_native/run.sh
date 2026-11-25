#!/usr/bin/env bash
set -euo pipefail
cd /mnt/r6_hdd4t/lhlu/ActivePersonalityAgent
source "$PWD/scripts/python_env.sh"
AMP_NATIVE_PYTHON="${AMP_NATIVE_PYTHON:-$AMP_CONDA_ROOT/envs/omniagent-native/bin/python}"
run_dir="${1:-runs/omniagent_native/comparison_v1}"
mkdir -p "$run_dir"
exec 9>/tmp/lhlu_omniagent_native_gpu.lock
flock -n 9 || { echo 'Another owned native comparison holds the GPU lock.'; exit 1; }
export PATH=/mnt/r6_hdd4t/lhlu/ActivePersonalityAgent/.local-tools/ffmpeg/bin:$PATH
export PYTHONPATH=/mnt/r6_hdd4t/lhlu/OmniAgent:/mnt/r6_hdd4t/lhlu/OmniAgent/demo:/mnt/r6_hdd4t/lhlu/ActivePersonalityAgent${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export CPATH=/mnt/r6_hdd4t/lhlu/ActivePersonalityAgent/.local-tools/python3.10-headers/usr/include/python3.10:/mnt/r6_hdd4t/lhlu/ActivePersonalityAgent/.local-tools/python3.10-headers/usr/include
export DASHSCOPE_API_KEY='' OSS_ACCESS_KEY=''
"$AMP_PYTHON" baselines/omniagent_native/task.py
"$AMP_PYTHON" - <<'PY'
import subprocess
lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
for line in lines:
    gpu,used=map(int,line.split(','))
    if gpu in (0,1) and used>1536:raise RuntimeError(f'GPU {gpu} already has {used} MiB allocated; refusing overlapping model loads')
print('Both GPUs are free for sequential model tests',flush=True)
PY
CUDA_VISIBLE_DEVICES=0,1 "$AMP_NATIVE_PYTHON" -u \
  baselines/omniagent_native/run_native.py --output "$run_dir/native" > "$run_dir/native.log" 2>&1
if [[ "${2:-}" == "--native-only" ]]; then
  exit 0
fi
# Run sequentially so two model engines cannot overlap on the same GPU.
sleep 3
CUDA_VISIBLE_DEVICES=1 "$AMP_PYTHON" -u baselines/omniagent_native/run_ours.py \
  --output "$run_dir/ours" > "$run_dir/ours.log" 2>&1
"$AMP_PYTHON" baselines/omniagent_native/compare_results.py --run-dir "$run_dir"
