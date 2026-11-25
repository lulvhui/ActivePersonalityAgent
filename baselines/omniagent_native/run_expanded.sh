#!/usr/bin/env bash
set -euo pipefail
cd /mnt/r6_hdd4t/lhlu/ActivePersonalityAgent
source "$PWD/scripts/python_env.sh"
AMP_NATIVE_PYTHON="${AMP_NATIVE_PYTHON:-$AMP_CONDA_ROOT/envs/omniagent-native/bin/python}"
run_dir="${1:-runs/omniagent_expanded_20261117}"
mkdir -p "$run_dir"
exec 9>/tmp/lhlu_omniagent_native_gpu.lock
flock -n 9 || { echo 'Another comparison holds the GPU lock.'; exit 1; }
export PATH="$PWD/.local-tools/ffmpeg/bin:$PATH"
export PYTHONPATH=/mnt/r6_hdd4t/lhlu/OmniAgent:/mnt/r6_hdd4t/lhlu/OmniAgent/demo:$PWD${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export CPATH="$PWD/.local-tools/python3.10-headers/usr/include/python3.10:$PWD/.local-tools/python3.10-headers/usr/include"
export DASHSCOPE_API_KEY='' OSS_ACCESS_KEY=''
if [[ ! -f "$run_dir/data/manifest.jsonl" ]]; then
  "$AMP_PYTHON" baselines/omniagent_native/expanded.py prepare --output "$run_dir"
fi
"$AMP_PYTHON" - <<'PY'
import subprocess
for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines():
    gpu,used=map(int,line.split(','))
    if gpu in (0,1) and used>1536:raise RuntimeError(f'GPU {gpu} uses {used} MiB; refusing overlap')
PY
printf '%s\n' native_running > "$run_dir/status.txt"
CUDA_VISIBLE_DEVICES=0,1 "$AMP_PYTHON" -u baselines/omniagent_native/native_batches.py \
  --run-dir "$run_dir" > "$run_dir/native_batches.log" 2>&1
"$AMP_PYTHON" baselines/omniagent_native/expanded.py summarize --output "$run_dir" > "$run_dir/summary.log"
printf '%s\n' current_models_running > "$run_dir/status.txt"
CUDA_VISIBLE_DEVICES=0 "$AMP_PYTHON" -u baselines/omniagent_native/run_ours.py \
  --manifest "$run_dir/data/manifest.jsonl" --model /mnt/r6_hdd4t/lhlu/models/Qwen2.5-Omni-3B \
  --output "$run_dir/current_3b" > "$run_dir/current_3b.log" 2>&1 &
pid3=$!
CUDA_VISIBLE_DEVICES=1 "$AMP_PYTHON" -u baselines/omniagent_native/run_ours.py \
  --manifest "$run_dir/data/manifest.jsonl" --model /mnt/r6_hdd4t/lhlu/models/Qwen2.5-Omni-7B \
  --output "$run_dir/current_7b" > "$run_dir/current_7b.log" 2>&1 &
pid7=$!
status=0
wait "$pid3" || status=1
wait "$pid7" || status=1
if [[ "$status" != 0 ]]; then
  printf '%s\n' current_model_process_failed > "$run_dir/status.txt"
  exit "$status"
fi
printf '%s\n' transfer_running > "$run_dir/status.txt"
CUDA_VISIBLE_DEVICES=1 "$AMP_PYTHON" -u baselines/omniagent_native/run_ours.py \
  --manifest "$run_dir/data/manifest.jsonl" --model /mnt/r6_hdd4t/lhlu/models/OmniAgent-RL-7B \
  --output "$run_dir/omniagent_transfer" > "$run_dir/omniagent_transfer.log" 2>&1
"$AMP_PYTHON" baselines/omniagent_native/expanded.py summarize --output "$run_dir" > "$run_dir/summary.log"
printf '%s\n' complete > "$run_dir/status.txt"
