"""Part 5, step 7: how wrong is each position of ACT's predicted action chunk?

For every frame of the validation episodes, predict a chunk the way the policy does at inference
(policy.eval(): no dropout, CVAE skipped, z = 0) and compare action k (k = 1..chunk_size) with
what the human actually did k-1 steps later. Errors are mean L1 in pixels of the 512x512
workspace (averaged over x and y), with padded positions past the episode end excluded.

As a reference, "hold position" predicts every future action as the agent's current position.
Its error grows only because the human keeps moving, so it shows how much harder the far future
is by itself.

    python scripts/chunk_error.py
    python scripts/chunk_error.py --checkpoint outputs/part5/run/checkpoints/030000/pretrained_model
Writes outputs/part5/chunk_error.png and chunk_error.csv.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from tiny_vla.device import get_device
from tiny_vla.policy import load_pretrained

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, default=Path("outputs/part5/run"))
parser.add_argument("--checkpoint", type=Path, default=None, help="default: the run's last checkpoint")
parser.add_argument("--out-dir", type=Path, default=Path("outputs/part5"))
args = parser.parse_args()
device = get_device("cuda")

checkpoint = args.checkpoint or sorted((args.run_dir / "checkpoints").glob("[0-9]*"))[-1] / "pretrained_model"
policy, preprocessor, postprocessor = load_pretrained(checkpoint, device)
chunk_size, n_action_steps = policy.config.chunk_size, policy.config.n_action_steps
val_episodes = json.loads((args.run_dir / "split.json").read_text())["val_episodes"]
print(f"checkpoint {checkpoint}\nval episodes: {len(val_episodes)}, chunk_size {chunk_size}")

meta = LeRobotDatasetMetadata("lerobot/pusht")
dataset = LeRobotDataset("lerobot/pusht", episodes=val_episodes, video_backend="pyav", return_uint8=True,
                         delta_timestamps=resolve_delta_timestamps(policy.config, meta))
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

error_sum = torch.zeros(chunk_size)  # summed |error| per chunk position, over valid frames
hold_sum = torch.zeros(chunk_size)
valid_count = torch.zeros(chunk_size)

with torch.no_grad():
    for batch in loader:
        true_actions = batch["action"].clone()  # (B, chunk, 2) in pixels, before normalization
        current_pos = batch["observation.state"].clone()  # (B, 2) agent position in pixels
        valid = ~batch["action_is_pad"]  # (B, chunk): False where the chunk runs past the episode end

        for key in meta.camera_keys:
            batch[key] = batch[key].float() / 255.0
        normalized = preprocessor(batch)
        predicted = postprocessor(policy.predict_action_chunk(normalized))  # (B, chunk, 2), pixels, CPU

        per_position = (predicted - true_actions).abs().mean(dim=-1)  # (B, chunk), mean over x and y
        hold = (current_pos[:, None, :] - true_actions).abs().mean(dim=-1)
        error_sum += (per_position * valid).sum(dim=0)
        hold_sum += (hold * valid).sum(dim=0)
        valid_count += valid.sum(dim=0)

error = (error_sum / valid_count).numpy()
hold_error = (hold_sum / valid_count).numpy()
positions = np.arange(1, chunk_size + 1)

print(f"\n{'k':>3} {'ACT error px':>13} {'hold-position px':>17} {'frames':>7}")
for k, e, h, n in zip(positions, error, hold_error, valid_count.numpy()):
    print(f"{k:>3} {e:>13.2f} {h:>17.2f} {int(n):>7}")
print(f"\naction 1: {error[0]:.1f} px, action {n_action_steps}: {error[n_action_steps - 1]:.1f} px, "
      f"action {chunk_size}: {error[-1]:.1f} px")

args.out_dir.mkdir(parents=True, exist_ok=True)
with open(args.out_dir / "chunk_error.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["position", "act_l1_px", "hold_position_l1_px", "frames"])
    writer.writerows(zip(positions, error.round(3), hold_error.round(3), valid_count.int().tolist()))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(positions, error, marker="o", color="tab:blue", label="ACT prediction (z = 0)")
ax.plot(positions, hold_error, marker=".", color="tab:gray", linestyle="--", label='"hold position" baseline')
ax.axvspan(0.5, n_action_steps + 0.5, color="tab:green", alpha=0.08,
           label=f"executed before re-planning (n_action_steps={n_action_steps})")
ax.set_xticks(positions)
ax.set_xlabel("position in the predicted chunk (1 = the action for the current step)")
ax.set_ylabel("mean L1 error (pixels, 512x512 workspace)")
ax.set_title(f"ACT chunk error by position on {len(val_episodes)} held-out episodes")
ax.set_ylim(bottom=0)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(args.out_dir / "chunk_error.png", dpi=120)
print(f"Saved {args.out_dir}/chunk_error.png, chunk_error.csv")
