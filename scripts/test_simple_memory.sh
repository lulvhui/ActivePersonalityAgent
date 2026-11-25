#!/usr/bin/env bash
set -euo pipefail
cd /mnt/r6_hdd4t/lhlu/ActivePersonalityAgent
source "$PWD/scripts/python_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
out="${1:-runs/simple_memory_restore/local}"
if (( $# > 0 )); then shift; fi
for size in 3b 7b; do
  if [[ -e "$out/$size/summary.json" ]]; then echo 'Choose a fresh output directory.'; exit 1; fi
done
while read -r used; do
  if (( used > 1500 )); then echo 'GPUs busy; no model launched.'; exit 1; fi
done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
mkdir -p "$out"
CUDA_VISIBLE_DEVICES=0 "$AMP_PYTHON" -u scripts/test_simple_memory.py --model /mnt/r6_hdd4t/lhlu/models/Qwen2.5-Omni-3B --output "$out/3b" "$@" > "$out/3b.log" 2>&1 &
pid3=$!
CUDA_VISIBLE_DEVICES=1 "$AMP_PYTHON" -u scripts/test_simple_memory.py --model /mnt/r6_hdd4t/lhlu/models/Qwen2.5-Omni-7B --output "$out/7b" "$@" > "$out/7b.log" 2>&1 &
pid7=$!
status=0
wait "$pid3" || status=1
wait "$pid7" || status=1
exit "$status"
