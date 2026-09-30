"""Part 3, step 5: evaluate every saved ACT checkpoint with the Part 2 rollout loop.

For each checkpoint in outputs/part3/act_pusht/checkpoints/<step>/pretrained_model, runs the
same 20 seeded episodes. Then parses the training loss from the lerobot-train log, plots loss
and success rate against training step on one figure, and saves 2 videos from the best checkpoint.

    python scripts/eval_checkpoints.py
    python scripts/eval_checkpoints.py --run-dir outputs/part3/long/act_pusht \
        --train-log outputs/part3/long/logs/train.log --out-dir outputs/part3/long
Writes to --out-dir (default outputs/part3/):
    checkpoint_eval.csv           one row per checkpoint
    checkpoint_eval_episodes.csv  one row per episode
    loss_vs_success.png
    videos/best_<step>/seed_*.mp4
"""

import argparse
import csv
import json
import re

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tiny_vla.device import get_device
from tiny_vla.policy import load_pretrained
from tiny_vla.rollout import EpisodeResult, make_env, run_episode
from tiny_vla.video import record_episode

RUN_DIR = Path("outputs/part3/act_pusht")
TRAIN_LOG = Path("outputs/part3/logs/train.log")
OUT_DIR = Path("outputs/part3")
SEEDS = list(range(1000, 1020))  # 20 fixed seeds; the first 10 are the Part 2 seeds
N_VIDEOS = 2


def find_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """[(step, pretrained_model dir)] sorted by step. Skips the "last" symlink."""
    found = []
    for step_dir in (run_dir / "checkpoints").iterdir():
        if step_dir.name.isdigit() and (step_dir / "pretrained_model").is_dir():
            found.append((int(step_dir.name), step_dir / "pretrained_model"))
    return sorted(found)


def parse_train_log(log_path: Path, log_freq: int) -> dict[str, np.ndarray]:
    """Read the metrics lerobot-train prints every `log_freq` steps.

    A line looks like: "... step:2K smpl:16K ep:128 epch:0.62 loss:0.345 grdn:... lr:... updt_s:0.083 data_s:0.041"
    The printed step is rounded to whole thousands (format_big_number), so it can't be plotted
    directly. Instead the k-th metrics line (counting from 1) is step k * log_freq; that's exact
    for a run that wasn't resumed.
    """
    pattern = re.compile(r"step:\S+ smpl:\S+ ep:\S+ epch:\S+ loss:([0-9.]+).*?updt_s:([0-9.]+) data_s:([0-9.]+)")
    # Progress bars write "\r" without newlines, so split on both.
    text = log_path.read_text(errors="replace").replace("\r", "\n")
    rows = [tuple(float(v) for v in m.groups()) for m in map(pattern.search, text.splitlines()) if m]
    if not rows:
        raise SystemExit(f"No metrics lines found in {log_path}")
    loss, update_s, data_s = map(np.array, zip(*rows))
    return {"step": np.arange(1, len(rows) + 1) * log_freq, "loss": loss, "update_s": update_s, "data_s": data_s}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="torch device (default: cuda, error if unavailable)")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR, help="lerobot-train output_dir")
    parser.add_argument("--train-log", type=Path, default=TRAIN_LOG, help="stdout of that lerobot-train run")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="where CSVs, the plot and videos go")
    args = parser.parse_args()
    device = get_device(args.device)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = find_checkpoints(args.run_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints under {args.run_dir}/checkpoints")
    print(f"Checkpoints: {[step for step, _ in checkpoints]}, {len(SEEDS)} episodes each\n")

    env = make_env()
    summary_rows, episode_rows = [], []
    for step, path in checkpoints:
        policy, preprocessor, postprocessor = load_pretrained(path, device)
        results: list[EpisodeResult] = []
        for seed in SEEDS:
            r = run_episode(policy, env, preprocessor, postprocessor, seed)
            results.append(r)
            episode_rows.append({"step": step, **r.row()})
        summary_rows.append(
            {
                "step": step,
                "success_rate": np.mean([r.success for r in results]),
                "avg_max_coverage": np.mean([r.max_coverage for r in results]),
                "avg_episode_length": np.mean([r.steps for r in results]),
                "model_calls_per_episode": np.mean([r.chunk_predictions for r in results]),
                "wall_time_per_episode_s": np.mean([r.wall_time_s for r in results]),
                "peak_vram_mb": max(r.peak_vram_mb or 0 for r in results),
            }
        )
        row = summary_rows[-1]
        print(f"step {step:>6}: success {row['success_rate']:.0%}  max_cov {row['avg_max_coverage']:.2f}  "
              f"length {row['avg_episode_length']:.0f}  {row['wall_time_per_episode_s']:.1f}s/ep", flush=True)
        del policy
        torch.cuda.empty_cache()

    # Save CSVs.
    with open(out_dir / "checkpoint_eval.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in summary_rows)
    with open(out_dir / "checkpoint_eval_episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)

    # Training loss from the log. log_freq comes from the config lerobot-train saved with the run.
    train_config = json.loads((checkpoints[-1][1] / "train_config.json").read_text())
    log = parse_train_log(args.train_log, train_config["log_freq"])
    if log["step"][-1] != train_config["steps"]:
        print(f"WARNING: log ends at step {log['step'][-1]}, run was {train_config['steps']} steps "
              "(resumed or truncated log?), so loss steps may be off.")
    print(f"\nTrain log: {len(log['loss'])} metric lines. Mean time per step: "
          f"update {log['update_s'].mean() * 1000:.0f} ms, waiting for data {log['data_s'].mean() * 1000:.0f} ms")

    # One figure, two y-axes: loss on the left (log scale, it falls by orders of magnitude),
    # success rate on the right.
    fig, ax_loss = plt.subplots(figsize=(10, 5))
    ax_loss.plot(log["step"], log["loss"], color="tab:blue", alpha=0.35, linewidth=1, label="train loss (per log window)")
    window = 10  # smooth over 10 log windows (1000 steps) to see the trend
    if len(log["loss"]) >= window:
        smooth = np.convolve(log["loss"], np.ones(window) / window, mode="valid")
        ax_loss.plot(log["step"][window - 1 :], smooth, color="tab:blue", linewidth=2, label="train loss (smoothed)")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("train loss = L1 + kl_weight * KL (log scale)", color="tab:blue")
    ax_loss.tick_params(axis="y", labelcolor="tab:blue")

    ax_success = ax_loss.twinx()
    steps = [r["step"] for r in summary_rows]
    ax_success.plot(steps, [r["success_rate"] for r in summary_rows], color="tab:red", marker="o", linewidth=2,
                    label=f"success rate ({len(SEEDS)} episodes)")
    ax_success.plot(steps, [r["avg_max_coverage"] for r in summary_rows], color="tab:orange", marker="s",
                    linestyle="--", label="avg max coverage")
    ax_success.set_ylim(0, 1)
    ax_success.set_ylabel("success rate / coverage", color="tab:red")
    ax_success.tick_params(axis="y", labelcolor="tab:red")

    handles = ax_loss.get_legend_handles_labels()[0] + ax_success.get_legend_handles_labels()[0]
    ax_loss.legend(handles, [h.get_label() for h in handles], loc="center right")
    ax_loss.set_title("ACT on PushT: training loss vs rollout success")
    ax_loss.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_vs_success.png", dpi=120)
    print(f"Saved {out_dir}/loss_vs_success.png, checkpoint_eval.csv, checkpoint_eval_episodes.csv")

    # Videos from the best checkpoint: highest success, ties broken by coverage, then the later step.
    best = max(summary_rows, key=lambda r: (r["success_rate"], r["avg_max_coverage"], r["step"]))
    best_path = dict(checkpoints)[best["step"]]
    print(f"\nBest checkpoint: step {best['step']} ({best['success_rate']:.0%} success). Recording {N_VIDEOS} videos...")
    policy, preprocessor, postprocessor = load_pretrained(best_path, device)
    video_dir = out_dir / "videos" / f"best_{best['step']:06d}"
    # Prefer one success and one failure, so the videos show both what works and what doesn't.
    best_episodes = [e for e in episode_rows if e["step"] == best["step"]]
    successes = [e["seed"] for e in best_episodes if e["success"]]
    failures = [e["seed"] for e in best_episodes if not e["success"]]
    video_seeds = (successes[:1] + failures[:1]) if successes and failures else SEEDS[:N_VIDEOS]
    for seed in video_seeds:
        result, path = record_episode(policy, env, preprocessor, postprocessor, seed, video_dir)
        print(f"  seed {seed}: success={result.success} steps={result.steps} -> {path}")
    env.close()


if __name__ == "__main__":
    main()
