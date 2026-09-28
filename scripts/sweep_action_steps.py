"""Sweep n_action_steps: how many actions of each predicted chunk to execute before re-planning.

For each value, runs the same 10 seeded episodes and records success, coverage, episode length,
model calls, wall-clock time and peak VRAM. Writes:
    outputs/part2/sweep_episodes.csv   one row per episode
    outputs/part2/sweep.csv            one row per n_action_steps (averages)
    outputs/part2/sweep.png

Note: the policy predicts horizon=16 actions starting at the *previous* observation step
(n_obs_steps=2), so at most horizon - n_obs_steps + 1 = 15 of them are in the future.
n_action_steps=16 therefore still executes only 15 actions per chunk (see
DiffusionModel.generate_actions: actions[:, 1:1 + n_action_steps] on a length-16 chunk).

    python scripts/sweep_action_steps.py
    python scripts/sweep_action_steps.py --values 8 16 4 1   # cheap settings first
"""

import argparse
import csv
from dataclasses import asdict, fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tiny_vla.device import get_device
from tiny_vla.policy import load_policy
from tiny_vla.rollout import EpisodeResult, make_env, run_episode

OUT_DIR = Path("outputs/part2")
N_ACTION_STEPS = [1, 4, 8, 16]
SEEDS = list(range(1000, 1010))  # same seeds as `lerobot-eval --seed 1000 --eval.n_episodes 10`

parser = argparse.ArgumentParser()
parser.add_argument("--device", default=None, help="torch device (default: cuda, error if unavailable)")
parser.add_argument(
    "--values", type=int, nargs="+", default=N_ACTION_STEPS, help="n_action_steps values, run in this order"
)
args = parser.parse_args()
device = get_device(args.device)
OUT_DIR.mkdir(parents=True, exist_ok=True)
env = make_env()

episode_rows = []
summary_rows = []
for n in args.values:
    policy, preprocessor, postprocessor = load_policy(device, n_action_steps=n)
    results: list[EpisodeResult] = []
    for seed in SEEDS:
        r = run_episode(policy, env, preprocessor, postprocessor, seed)
        results.append(r)
        print(
            f"n_action_steps={n:<2} seed={seed} success={r.success!s:<5} steps={r.steps:<3} "
            f"max_cov={r.max_coverage:.2f} calls={r.chunk_predictions:<3} "
            f"denoise={r.denoise_passes:<5} time={r.wall_time_s:.1f}s vram={r.peak_vram_mb or 0:.0f}MB",
            flush=True,
        )
        row = {k: v for k, v in asdict(r).items() if k != "frames"}
        episode_rows.append({"n_action_steps": n, **row})

    vram = [r.peak_vram_mb for r in results if r.peak_vram_mb is not None]
    summary_rows.append(
        {
            "n_action_steps": n,
            "episodes": len(results),
            "success_rate": np.mean([r.success for r in results]),
            "avg_max_coverage": np.mean([r.max_coverage for r in results]),
            "avg_max_reward": np.mean([r.max_reward for r in results]),
            "avg_episode_length": np.mean([r.steps for r in results]),
            "model_calls_per_episode": np.mean([r.chunk_predictions for r in results]),
            "denoise_passes_per_episode": np.mean([r.denoise_passes for r in results]),
            "env_steps_per_model_call": sum(r.steps for r in results) / sum(r.chunk_predictions for r in results),
            "wall_time_per_episode_s": np.mean([r.wall_time_s for r in results]),
            "peak_vram_mb": max(vram) if vram else "",
        }
    )
    del policy  # free GPU memory before loading the next copy

# Save CSVs, sorted by n_action_steps regardless of the order the values ran in.
summary_rows.sort(key=lambda row: row["n_action_steps"])
episode_rows.sort(key=lambda row: (row["n_action_steps"], row["seed"]))
episode_fields = ["n_action_steps"] + [f.name for f in fields(EpisodeResult) if f.name != "frames"]
with open(OUT_DIR / "sweep_episodes.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=episode_fields)
    writer.writeheader()
    writer.writerows(episode_rows)
with open(OUT_DIR / "sweep.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
    writer.writeheader()
    for row in summary_rows:
        writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

print()
print(f"{'n':>3} {'success':>8} {'max_cov':>8} {'length':>7} {'calls/ep':>9} {'s/ep':>6} {'vram_MB':>8}")
for row in summary_rows:
    print(
        f"{row['n_action_steps']:>3} {row['success_rate']:>8.0%} {row['avg_max_coverage']:>8.2f} "
        f"{row['avg_episode_length']:>7.1f} {row['model_calls_per_episode']:>9.1f} "
        f"{row['wall_time_per_episode_s']:>6.1f} {row['peak_vram_mb']:>8.0f}"
    )

# Plot the four headline metrics against n_action_steps.
x = [row["n_action_steps"] for row in summary_rows]
panels = [
    ("success_rate", "success rate"),
    ("avg_max_coverage", "avg max coverage"),
    ("model_calls_per_episode", "model calls per episode"),
    ("wall_time_per_episode_s", "wall-clock s per episode"),
]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, (key, label) in zip(axes.flat, panels):
    ax.plot(x, [row[key] for row in summary_rows], marker="o")
    ax.set_xticks(x)
    ax.set_xlabel("n_action_steps")
    ax.set_title(label)
    ax.grid(alpha=0.3)
axes[0, 0].set_ylim(0, 1)
axes[0, 1].set_ylim(0, 1)
fig.suptitle(f"diffusion_pusht, {len(SEEDS)} episodes per setting (seeds {SEEDS[0]}-{SEEDS[-1]})")
fig.tight_layout()
fig.savefig(OUT_DIR / "sweep.png", dpi=120)
print(f"\nSaved {OUT_DIR}/sweep.csv, sweep_episodes.csv, sweep.png")
