#!/usr/bin/env bash
# Sample GPU utilization and memory every N seconds into a CSV until killed.
#
#   scripts/log_gpu.sh outputs/part3/logs/gpu.csv 10
#
# utilization.gpu is the % of the last sample period in which a kernel was running on the GPU.
# Low values during training usually mean the GPU is waiting on the dataloader.
set -euo pipefail

out="${1:?usage: log_gpu.sh OUT_CSV [INTERVAL_S]}"
interval="${2:-10}"
mkdir -p "$(dirname "$out")"

# -l loops forever; --format=csv writes a header row first.
exec nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,nounits \
  -l "$interval" > "$out"
