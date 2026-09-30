"""Part 5: a plain, readable training loop for ACT on PushT (replaces lerobot-train).

It does what lerobot-train does for a single-GPU ACT run (see outputs/part5/HOW_LEROBOT_TRAINS.md),
written out step by step, plus three things lerobot-train doesn't do by default:
  - a 90/10 train/validation split of *episodes* with a fixed seed, and a validation loss
  - logging to a CSV and TensorBoard every --log-freq steps
  - saving the fp16 GradScaler and the data position in each checkpoint, for exact resuming

Checkpoints use lerobot's pretrained format, so tiny_vla.policy.load_pretrained, watch.py,
eval_checkpoints.py and lerobot-eval can all load them.

    python -m tiny_vla.train --output-dir outputs/part5/run
    python -m tiny_vla.train --output-dir outputs/part5/run --resume   # continue from the last checkpoint

CUDA only: it stops with an error if CUDA isn't available.
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.tensorboard import SummaryWriter

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from tiny_vla.device import get_device

REPO_ID = "lerobot/pusht"
VIDEO_BACKEND = "pyav"  # torchcodec can't load on the Linux box
CSV_FIELDS = [
    "step", "train_loss", "train_l1", "train_kld", "val_l1", "grad_norm", "lr", "steps_per_s",
    "data_s", "update_s", "skipped_steps", "gpu_mem_mb",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in --output-dir")
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=20)
    p.add_argument("--n-action-steps", type=int, default=10)
    p.add_argument("--save-freq", type=int, default=5_000)
    p.add_argument("--log-freq", type=int, default=1_000)
    p.add_argument("--val-fraction", type=float, default=0.1)
    # 0 = load data in the main process. On the WSL box (3.5 GB RAM, with VS Code holding ~0.9 GB of
    # swap) one extra worker process left too little RAM to write a checkpoint. Same batches either way.
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1000)
    return p.parse_args()


def set_seed(seed: int):
    """Seed every random number generator the run touches (same as lerobot's set_seed)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_episodes(n_episodes: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Shuffle episode indices with a fixed seed and hold out the first val_fraction of them.

    Splitting by episode (not by frame) matters: frames next to each other in one demo are nearly
    identical, so a frame-level split would put near-copies of every val frame in the train set
    and make the val loss look far better than it is.
    """
    order = np.random.default_rng(seed).permutation(n_episodes).tolist()
    n_val = round(n_episodes * val_fraction)  # 0 -> train on every episode, no validation
    return sorted(order[n_val:]), sorted(order[:n_val])


class ResumableBatchSampler:
    """Yields lists of dataset indices forever: a fresh shuffle every epoch, fully resumable.

    Epoch e's order is torch.randperm seeded with (seed + e), so the batch at any step can be
    recomputed from nothing but the step number. Resuming at step s starts exactly where the
    interrupted run would have continued. (lerobot's EpisodeAwareSampler uses the same idea.)
    Like lerobot (drop_last=False), the last batch of each epoch may be smaller.
    """

    def __init__(self, n_items: int, batch_size: int, seed: int, start_step: int = 0):
        self.n_items, self.batch_size, self.seed, self.start_step = n_items, batch_size, seed, start_step
        self.batches_per_epoch = math.ceil(n_items / batch_size)

    def __iter__(self):
        epoch, batch_in_epoch = divmod(self.start_step, self.batches_per_epoch)
        while True:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(self.n_items, generator=generator).tolist()
            for b in range(batch_in_epoch, self.batches_per_epoch):
                yield order[b * self.batch_size : (b + 1) * self.batch_size]
            epoch, batch_in_epoch = epoch + 1, 0


def prepare_batch(batch: dict, preprocessor, camera_keys: list[str]) -> dict:
    """uint8 images -> float 0..1, then the preprocessor: add batch dim, move to GPU, normalize."""
    for key in camera_keys:
        batch[key] = batch[key].float() / 255.0
    return preprocessor(batch)


@torch.no_grad()
def validation_l1(policy, val_loader, preprocessor, camera_keys) -> float:
    """Mean L1 action error on the validation episodes, measured the way the policy is *used*.

    policy.eval() turns off dropout and, for ACT, skips the CVAE encoder and uses z = 0 (the same
    thing that happens at inference). In that mode policy.forward returns the L1 loss only, so this
    is comparable with the train L1, not with the train total loss (L1 + kl_weight * KL).
    """
    policy.eval()
    total, count = 0.0, 0
    for batch in val_loader:
        batch = prepare_batch(batch, preprocessor, camera_keys)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, _ = policy.forward(batch)
        n = len(batch["action"])
        total, count = total + loss.item() * n, count + n
    policy.train()
    return total / count


def save_checkpoint(ckpt_dir: Path, policy, preprocessor, postprocessor, optimizer, scheduler, scaler,
                    step: int, split: dict, args):
    """Save a checkpoint that lerobot tools can load, plus everything needed to resume.

    pretrained_model/  config.json, model.safetensors, policy_{pre,post}processor*.json/.safetensors
                       (lerobot's format: load_pretrained, watch.py, lerobot-eval all read this)
    training_state.pt  optimizer, scheduler, GradScaler, step, RNG states, episode split, args
    """
    model_dir = ckpt_dir / "pretrained_model"
    policy.save_pretrained(model_dir)
    preprocessor.save_pretrained(model_dir)
    postprocessor.save_pretrained(model_dir)
    (model_dir / "tiny_train_args.json").write_text(json.dumps({**vars(args), "output_dir": str(args.output_dir)}, indent=2))
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            # lerobot-train does NOT save this. Without it, a resumed fp16 run restarts the loss
            # scale at 65536 and skips its first few steps while the scaler re-learns it.
            "scaler": scaler.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
            "split": split,
        },
        ckpt_dir / "training_state.pt",
    )


def latest_checkpoint(output_dir: Path) -> Path | None:
    ckpts = sorted(p for p in (output_dir / "checkpoints").glob("[0-9]*") if (p / "training_state.pt").exists())
    return ckpts[-1] if ckpts else None


def main():
    args = parse_args()
    device = get_device("cuda")  # CUDA or a loud error; no --device on purpose
    out = args.output_dir

    # ------------------------------------------------------------------------------------------
    # Seeding and GPU flags (lerobot_train.py: train(), lines 283-292).
    # cudnn.benchmark picks the fastest conv algorithms for our fixed image size; TF32 lets
    # float32 matmuls use tensor cores. Both trade bit-exact reproducibility for speed.
    # ------------------------------------------------------------------------------------------
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    resume_from = latest_checkpoint(out) if args.resume else None
    if args.resume and resume_from is None:
        raise SystemExit(f"--resume given but no checkpoint found under {out}/checkpoints")
    if not args.resume and (out / "checkpoints").exists():
        raise SystemExit(f"{out} already has checkpoints. Use --resume, or a new --output-dir.")
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------------------------
    # Episode split: 90% of episodes for training, 10% held out for validation.
    # On resume, reuse the saved split so the validation set never changes mid-run.
    # ------------------------------------------------------------------------------------------
    meta = LeRobotDatasetMetadata(REPO_ID)
    if resume_from is not None:
        state = torch.load(resume_from / "training_state.pt", weights_only=False)
        split = state["split"]
    else:
        train_eps, val_eps = split_episodes(meta.total_episodes, args.val_fraction, args.seed)
        split = {"train_episodes": train_eps, "val_episodes": val_eps}
    (out / "split.json").write_text(json.dumps(split))

    # ------------------------------------------------------------------------------------------
    # Policy config: ACT with our chunk settings and AMP. The config also knows which time offsets
    # it needs from the dataset: action_delta_indices = range(chunk_size).
    # ------------------------------------------------------------------------------------------
    config = ACTConfig(
        device=str(device), use_amp=True, chunk_size=args.chunk_size, n_action_steps=args.n_action_steps,
        push_to_hub=False,
    )

    # ------------------------------------------------------------------------------------------
    # Datasets (datasets/factory.py: make_dataset). One LeRobotDataset per split, each returning a
    # window of chunk_size future actions per frame. return_uint8 keeps images small in the
    # dataloader; they're converted to float on the GPU side (prepare_batch).
    # ------------------------------------------------------------------------------------------
    delta_timestamps = resolve_delta_timestamps(config, meta)
    def make_dataset(episodes):
        return LeRobotDataset(REPO_ID, episodes=episodes, delta_timestamps=delta_timestamps,
                              video_backend=VIDEO_BACKEND, return_uint8=True)
    train_ds = make_dataset(split["train_episodes"])
    val_ds = make_dataset(split["val_episodes"]) if split["val_episodes"] else None
    print(f"train: {len(split['train_episodes'])} episodes / {len(train_ds)} frames, "
          f"val: {len(split['val_episodes'])} episodes / {len(val_ds) if val_ds else 0} frames")

    # ------------------------------------------------------------------------------------------
    # Policy and processors (policies/factory.py: make_policy, make_pre_post_processors).
    # Normalization stats are the dataset's precomputed ones, over ALL episodes (lerobot does the
    # same), so the val episodes slightly influence the normalization. It's only a mean/std per
    # feature, so the leak is negligible, but it's there.
    # ------------------------------------------------------------------------------------------
    policy = make_policy(config, ds_meta=meta)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=meta.stats)

    # ------------------------------------------------------------------------------------------
    # Optimizer, scheduler and gradient clipping from the policy's own presets
    # (optim/factory.py: make_optimizer_and_scheduler). For ACT: AdamW(lr=1e-5, weight_decay=1e-4),
    # backbone in its own param group, grad_clip_norm=10, and no scheduler (constant lr).
    # ------------------------------------------------------------------------------------------
    optimizer_cfg = config.get_optimizer_preset()
    optimizer = optimizer_cfg.build(policy.get_optim_params())
    scheduler_cfg = config.get_scheduler_preset()
    scheduler = scheduler_cfg.build(optimizer, args.steps) if scheduler_cfg is not None else None
    grad_clip_norm = optimizer_cfg.grad_clip_norm

    # ------------------------------------------------------------------------------------------
    # fp16 AMP. lerobot-train gets this from accelerate (mixed_precision="fp16"); written out, it's
    # torch.autocast around the forward pass plus a GradScaler. fp16 has a tiny range, so small
    # gradients would round to zero: the scaler multiplies the loss by a big factor before
    # backward, divides the gradients back before clipping, and skips steps whose gradients
    # overflowed to inf, shrinking the factor when that happens.
    # ------------------------------------------------------------------------------------------
    scaler = torch.amp.GradScaler("cuda")

    # ------------------------------------------------------------------------------------------
    # Resume: restore weights, optimizer, scheduler, scaler, RNG, and the step counter.
    # ------------------------------------------------------------------------------------------
    step = 0
    if resume_from is not None:
        policy.load_state_dict(load_file(resume_from / "pretrained_model" / "model.safetensors"))
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        step = state["step"]
        print(f"Resumed from {resume_from} at step {step}")

    # ------------------------------------------------------------------------------------------
    # DataLoaders. Train: our resumable sampler, pin_memory, and --num-workers spawned workers
    # (default 0, see parse_args: each worker is a full extra Python process costing ~500 MB).
    # The sampler runs in the main process, so the batches are the same for any worker count.
    # Val: fixed order, loaded in the main process (it runs once per log window).
    # ------------------------------------------------------------------------------------------
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_sampler=ResumableBatchSampler(len(train_ds), args.batch_size, args.seed, start_step=step),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0) if val_ds else None
    train_batches = iter(train_loader)

    # ------------------------------------------------------------------------------------------
    # Logging: a CSV (appended to on resume) and TensorBoard (`tensorboard --logdir <output-dir>`).
    # ------------------------------------------------------------------------------------------
    csv_path = out / "metrics.csv"
    new_csv = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if new_csv:
        csv_writer.writeheader()
    tb = SummaryWriter(out / "tensorboard")

    window = {"loss": 0.0, "l1": 0.0, "kld": 0.0, "grad_norm": 0.0, "data_s": 0.0, "update_s": 0.0,
              "steps": 0, "skipped": 0}
    window_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    policy.train()
    print(f"Training steps {step} -> {args.steps}, batch {args.batch_size}")

    # ==========================================================================================
    # The training loop (lerobot_train.py lines 601-619 + update_policy()).
    # ==========================================================================================
    while step < args.steps:
        # 1. Next batch, onto the GPU and normalized.
        t0 = time.perf_counter()
        batch = prepare_batch(next(train_batches), preprocessor, train_ds.meta.camera_keys)
        t1 = time.perf_counter()

        # 2. Forward pass in fp16. For ACT, policy.forward runs the CVAE encoder (train mode),
        #    predicts the chunk, and returns L1 + kl_weight * KL, plus both parts in loss_dict.
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, loss_dict = policy.forward(batch)

        # 3. Backward on the scaled loss (gradients come out multiplied by the scale factor).
        scaler.scale(loss).backward()

        # 4. Unscale the gradients back to their true size, THEN clip. Clipping scaled gradients
        #    would compare against a threshold that's 65536x too small. The returned norm is
        #    measured before clipping (what lerobot logs as grdn).
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)

        # 5. Optimizer step via the scaler: skipped if any gradient is inf/NaN. Then let the scaler
        #    adjust its factor (shrink after an overflow, grow after 2000 clean steps).
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < scale_before  # the scale only shrinks when a step was skipped
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        step += 1

        # 6. Accumulate metrics for this log window. loss.item() syncs with the GPU, so update_s
        #    includes the real compute time.
        window["loss"] += loss.item()
        window["l1"] += loss_dict["l1_loss"]
        window["kld"] += loss_dict.get("kld_loss", 0.0)
        window["grad_norm"] += grad_norm.item() if torch.isfinite(grad_norm) else 0.0
        window["data_s"] += t1 - t0
        window["update_s"] += time.perf_counter() - t1
        window["steps"] += 1
        window["skipped"] += int(skipped)

        # 7. Every log_freq steps: validation loss, then one CSV row and TensorBoard scalars.
        if step % args.log_freq == 0 or step == args.steps:
            elapsed = time.perf_counter() - window_start  # training time only; val runs after this
            n = window["steps"]
            row = {
                "step": step,
                "train_loss": window["loss"] / n,
                "train_l1": window["l1"] / n,
                "train_kld": window["kld"] / n,
                "val_l1": validation_l1(policy, val_loader, preprocessor, val_ds.meta.camera_keys) if val_ds else float("nan"),
                "grad_norm": window["grad_norm"] / max(n - window["skipped"], 1),
                "lr": optimizer.param_groups[0]["lr"],
                "steps_per_s": n / elapsed,
                "data_s": window["data_s"] / n,
                "update_s": window["update_s"] / n,
                "skipped_steps": window["skipped"],
                "gpu_mem_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            }
            csv_writer.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()})
            csv_file.flush()
            for key, value in row.items():
                if key != "step":
                    tb.add_scalar(key, value, step)
            print(f"step {step:>6}  loss {row['train_loss']:.4f}  l1 {row['train_l1']:.4f}  "
                  f"val_l1 {row['val_l1']:.4f}  grdn {row['grad_norm']:.2f}  lr {row['lr']:.1e}  "
                  f"{row['steps_per_s']:.1f} steps/s  data {row['data_s'] * 1000:.0f} ms", flush=True)
            window = dict.fromkeys(window, 0)
            window["loss"] = window["l1"] = window["kld"] = window["grad_norm"] = 0.0
            window_start = time.perf_counter()

        # 8. Checkpoint every save_freq steps and at the end (like should_save_checkpoint).
        if step % args.save_freq == 0 or step == args.steps:
            ckpt_dir = out / "checkpoints" / f"{step:06d}"
            save_checkpoint(ckpt_dir, policy, preprocessor, postprocessor, optimizer, scheduler, scaler,
                            step, split, args)
            print(f"Saved {ckpt_dir}", flush=True)
            window_start = time.perf_counter()  # don't count checkpoint time as training time

    csv_file.close()
    tb.close()
    print("Done.")


# The guard matters: DataLoader workers are started with "spawn", which re-imports this module.
if __name__ == "__main__":
    main()
