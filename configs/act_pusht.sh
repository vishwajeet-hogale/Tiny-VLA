#!/usr/bin/env bash
# Part 3: train ACT on PushT with lerobot-train (lerobot 0.6.1), on the 4GB RTX 3050.
#
# Run from the repo root with the venv active, inside tmux (it takes ~1.5-2 h):
#   tmux new -s act-train 'bash configs/act_pusht.sh'
#
# Also samples GPU utilization every 10 s into outputs/part3/logs/gpu.csv for the whole run.
set -euo pipefail

OUTPUT_DIR=outputs/part3/act_pusht
LOG_DIR=outputs/part3/logs
mkdir -p "$LOG_DIR"

# Fail loudly if there is no CUDA, instead of lerobot quietly training on the CPU.
python -c "from tiny_vla.device import get_device; get_device('cuda')"

# lerobot-train refuses to start if the output dir exists (unless resuming), so check up front.
if [ -e "$OUTPUT_DIR" ]; then
  echo "$OUTPUT_DIR already exists. Move it away, or resume with --resume=true." >&2
  exit 1
fi

# GPU sampler runs in the background and is stopped when this script exits, even on Ctrl-C.
scripts/log_gpu.sh "$LOG_DIR/gpu.csv" 10 &
GPU_LOGGER_PID=$!
trap 'kill $GPU_LOGGER_PID 2>/dev/null || true' EXIT

# chunk_size / n_action_steps:
#   ACT's defaults (100 / 100) come from the ACT paper's 50 fps arm, where 100 steps = 2 s.
#   PushT is 10 fps, so 100 steps would be 10 s, nearly a whole demo (median 12 s), executed
#   open loop. chunk_size=20 keeps the paper's 2 s prediction horizon at 10 fps. Executing
#   n_action_steps=10 (1 s) before re-planning keeps the policy reactive: PushT is contact-rich,
#   and after a push the block can end up far from where a 2 s old plan assumed. This is close
#   to the pretrained diffusion policy's 1.6 s horizon / 0.8 s executed.
#   n_action_steps only affects inference, so it can still be changed at eval time.
#
# batch_size=8: the probe showed 64 fits in VRAM (2150 of 4096 MB), but the first run at 64 with
#   2 dataloader workers crashed the whole WSL VM around step 700 from *system RAM* pressure
#   (the laptop has 7.3 GB; WSL gets 3.5 GB). 8 is ACT's own default and uses 1910 MB VRAM.
# num_workers=1, prefetch_factor=2: the real RAM cost is the worker processes, not the batch
#   (a sample is ~28 KB). Each spawned worker re-imports torch/lerobot and opens its own video
#   decoders, so fewer workers is what keeps RAM in bounds. Expect the GPU to wait on data.
# env_eval_freq=0: in lerobot 0.6.1 the flag is env_eval_freq (there is no eval_freq). 0 turns
#   off sim evaluation during training; scripts/eval_checkpoints.py evaluates checkpoints after.
# video_backend=pyav: torchcodec can't load its libraries on this machine.
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.chunk_size=20 \
  --policy.n_action_steps=10 \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht \
  --dataset.video_backend=pyav \
  --env.type=pusht \
  --batch_size=8 \
  --num_workers=1 \
  --prefetch_factor=2 \
  --steps=30000 \
  --save_freq=5000 \
  --env_eval_freq=0 \
  --log_freq=100 \
  --wandb.enable=false \
  --seed=1000 \
  --output_dir="$OUTPUT_DIR" \
  2>&1 | tee "$LOG_DIR/train.log"
