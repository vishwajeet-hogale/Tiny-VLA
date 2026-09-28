"""Part 3, step 3: find the largest ACT batch size that fits on this GPU.

For each batch size, runs 20 real training steps (fp16 AMP, like `lerobot-train --policy.use_amp=true`)
and records peak VRAM and training speed. Stops at the first out-of-memory error.

Three memory numbers are recorded, from smallest to largest:
  peak_allocated_mb    tensors PyTorch actually holds (torch.cuda.max_memory_allocated)
  peak_reserved_mb     memory PyTorch's caching allocator has claimed from the GPU
  peak_device_used_mb  everything in use on the GPU (total - free from cudaMemGetInfo):
                       the above plus the CUDA context, cuDNN workspaces, other processes and
                       (under WSL) the Windows desktop. This is the one that decides OOM.
A batch size is "safe" if peak_device_used_mb <= 85% of the GPU's total memory.

    python scripts/probe_batch_size.py
Writes outputs/part3/probe.csv.
"""

import argparse
import csv
import time
from pathlib import Path

import torch

from tiny_vla.act import build_act, make_act_dataset, make_act_preprocessor, make_optimizer, prepare_batch
from tiny_vla.device import get_device

OUT_FILE = Path("outputs/part3/probe.csv")
BATCH_SIZES = [8, 16, 32, 64]
N_STEPS = 20
WARMUP_STEPS = 5  # excluded from steps/sec: cuDNN autotuning and allocator growth happen here
HEADROOM = 0.15
GRAD_CLIP_NORM = 10.0  # lerobot-train's default (optimizer grad_clip_norm)


def device_used_mb(device: torch.device) -> float:
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / 2**20


def probe(batch_size: int, device: torch.device, chunk_size: int, num_workers: int, total_mb: float) -> dict:
    policy, meta = build_act(device, chunk_size=chunk_size, n_action_steps=chunk_size)
    preprocessor = make_act_preprocessor(policy, meta)
    optimizer = make_optimizer(policy)
    # GradScaler multiplies the loss before backward so small fp16 gradients don't round to zero,
    # then divides them back before the optimizer step. accelerate does the same with fp16.
    scaler = torch.amp.GradScaler("cuda")
    loader = torch.utils.data.DataLoader(
        make_act_dataset(policy, meta),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        # Same worker setup as lerobot-train: spawned workers, 4 batches prefetched per worker.
        multiprocessing_context="spawn" if num_workers > 0 else None,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    policy.train()
    torch.cuda.reset_peak_memory_stats(device)
    peak_device = 0.0
    batches = iter(loader)

    for step in range(N_STEPS):
        if step == WARMUP_STEPS:
            torch.cuda.synchronize(device)
            start = time.perf_counter()
        batch = prepare_batch(next(batches), preprocessor, meta.camera_keys)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, _ = policy.forward(batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        peak_device = max(peak_device, device_used_mb(device))

    torch.cuda.synchronize(device)
    steps_per_s = (N_STEPS - WARMUP_STEPS) / (time.perf_counter() - start)
    return {
        "batch_size": batch_size,
        "status": "ok",
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 2**20),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 2**20),
        "peak_device_used_mb": round(peak_device),
        "steps_per_s": round(steps_per_s, 2),
        "samples_per_s": round(steps_per_s * batch_size, 1),
        "safe_with_15pct_headroom": peak_device <= (1 - HEADROOM) * total_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk-size", type=int, default=20, help="must match configs/act_pusht.sh")
    parser.add_argument("--num-workers", type=int, default=2, help="must match configs/act_pusht.sh")
    args = parser.parse_args()
    device = get_device(args.device)
    total_mb = torch.cuda.get_device_properties(device).total_memory / 2**20

    print(f"GPU: {torch.cuda.get_device_name(device)}, {total_mb:.0f} MB total, "
          f"{device_used_mb(device):.0f} MB already in use before loading anything")
    print(f"ACT chunk_size={args.chunk_size}, AMP fp16, {N_STEPS} steps each, num_workers={args.num_workers}\n")

    rows = []
    for batch_size in BATCH_SIZES:
        try:
            row = probe(batch_size, device, args.chunk_size, args.num_workers, total_mb)
        except torch.OutOfMemoryError:
            row = {"batch_size": batch_size, "status": "OOM"}
        rows.append(row)
        print(row, flush=True)
        torch.cuda.empty_cache()  # return cached blocks so the next probe starts clean
        if row["status"] == "OOM":
            break

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    safe = [r["batch_size"] for r in rows if r.get("safe_with_15pct_headroom")]
    print(f"\nSaved {OUT_FILE}")
    print(f"Budget with {HEADROOM:.0%} headroom: {(1 - HEADROOM) * total_mb:.0f} MB of {total_mb:.0f} MB")
    print(f"Largest safe batch size: {max(safe) if safe else 'none'}")


# The guard matters: DataLoader workers are started with "spawn", which re-imports this file.
if __name__ == "__main__":
    main()
