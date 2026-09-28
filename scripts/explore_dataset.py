"""Part 1: explore the lerobot/pusht dataset. No training.

Prints dataset metadata and normalization stats, saves a few plots to outputs/part1/,
and shows how delta_timestamps changes the shape of a sample.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt
import torch

from lerobot.datasets import LeRobotDataset

REPO_ID = "lerobot/pusht"
OUT_DIR = Path("outputs/part1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# torchcodec fails to load on this machine, so use the pyav video decoder directly.
VIDEO_BACKEND = "pyav"


def episode_column(ds: LeRobotDataset, episode: int, key: str) -> torch.Tensor:
    """Return one column (e.g. "action") for every frame of an episode, shape (T, ...).

    Reads the parquet table directly, so no video frames are decoded.
    """
    ep = ds.meta.episodes[episode]
    rows = ds.hf_dataset[ep["dataset_from_index"] : ep["dataset_to_index"]]
    return torch.stack(rows[key])


# ---------------------------------------------------------------------------
# 1. Basic metadata
# ---------------------------------------------------------------------------
ds = LeRobotDataset(REPO_ID, video_backend=VIDEO_BACKEND)

print("=== 1. Metadata ===")
print(f"episodes:     {ds.num_episodes}")
print(f"total frames: {ds.num_frames}")
print(f"fps:          {ds.fps}")
print()
print(f"{'feature':<20} {'shape':<14} dtype")
for name, info in ds.features.items():
    print(f"{name:<20} {str(info['shape']):<14} {info['dtype']}")
print()

# ---------------------------------------------------------------------------
# 2. Normalization stats (computed over the whole dataset, stored in meta/stats.json)
# ---------------------------------------------------------------------------
print("=== 2. Normalization stats ===")
for key in ["observation.state", "action"]:
    print(key)
    for stat in ["min", "max", "mean", "std"]:
        print(f"  {stat:<5} {ds.meta.stats[key][stat]}")
print()

# ---------------------------------------------------------------------------
# 3. Grid of 8 frames from episode 0
# ---------------------------------------------------------------------------
ep0 = ds.meta.episodes[0]
ep0_start, ep0_len = ep0["dataset_from_index"], ep0["length"]

# 8 frame indices spread evenly from the first to the last frame of the episode.
frame_ids = torch.linspace(0, ep0_len - 1, 8).round().int().tolist()

fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
for ax, frame_id in zip(axes.flat, frame_ids):
    sample = ds[ep0_start + frame_id]
    image = sample["observation.image"].permute(1, 2, 0)  # (C, H, W) -> (H, W, C) for matplotlib
    action = sample["action"]
    ax.imshow(image)
    ax.set_title(f"frame {frame_id}\naction = ({action[0]:.0f}, {action[1]:.0f})", fontsize=10)
    ax.axis("off")
fig.suptitle("Episode 0")
fig.tight_layout()
fig.savefig(OUT_DIR / "episode0_frames.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 4a. State and action over time for episode 0
# ---------------------------------------------------------------------------
state = episode_column(ds, 0, "observation.state")  # (T, 2): agent (x, y)
action = episode_column(ds, 0, "action")  # (T, 2): target (x, y)
t = torch.arange(len(state)) / ds.fps  # seconds

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for i, axis_name in enumerate(["x", "y"]):
    axes[i].plot(t, state[:, i], label="state (agent position)")
    axes[i].plot(t, action[:, i], label="action (target position)", linestyle="--")
    axes[i].set_ylabel(axis_name)
    axes[i].legend(loc="upper right")
axes[1].set_xlabel("time (s)")
fig.suptitle("Episode 0: state vs action")
fig.tight_layout()
fig.savefig(OUT_DIR / "episode0_state_action.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 4b. 2D agent trajectories for 5 episodes
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
for episode in range(5):
    xy = episode_column(ds, episode, "observation.state")
    ax.plot(xy[:, 0], xy[:, 1], label=f"episode {episode}")
    ax.scatter(xy[0, 0], xy[0, 1], marker="o", s=30)  # start point
ax.set_xlim(0, 512)
ax.set_ylim(0, 512)
ax.invert_yaxis()  # PushT uses image coordinates: y grows downward
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Agent trajectories (dots = start)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "trajectories_5_episodes.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Histogram of episode lengths
# ---------------------------------------------------------------------------
lengths = ds.meta.episodes["length"]

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(lengths, bins=30)
ax.set_xlabel("episode length (frames)")
ax.set_ylabel("number of episodes")
ax.set_title(f"Episode lengths (min {min(lengths)}, max {max(lengths)}, mean {sum(lengths) / len(lengths):.0f})")
fig.tight_layout()
fig.savefig(OUT_DIR / "episode_lengths.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. delta_timestamps: ask each sample for a window of frames instead of one
# ---------------------------------------------------------------------------
# Offsets are in seconds relative to the current frame. At 10 fps one frame is 0.1 s.
frame_dt = 1 / ds.fps
delta_timestamps = {
    "observation.image": [-1 * frame_dt, 0.0],  # previous frame + current frame
    "observation.state": [-1 * frame_dt, 0.0],
    "action": [i * frame_dt for i in range(16)],  # current action + next 15
}
ds_windowed = LeRobotDataset(REPO_ID, delta_timestamps=delta_timestamps, video_backend=VIDEO_BACKEND)

SAMPLE_IDX = 0  # first frame of episode 0, so the "previous" observation doesn't exist
before = ds[SAMPLE_IDX]
after = ds_windowed[SAMPLE_IDX]

print("=== 6. Sample shapes, before vs after delta_timestamps ===")
print(f"{'key':<26} {'before':<22} after")
for key in after:
    if not isinstance(after[key], torch.Tensor):
        continue
    before_shape = str(tuple(before[key].shape)) if key in before else "-"
    print(f"{key:<26} {before_shape:<22} {tuple(after[key].shape)}")
print()
# The *_is_pad masks mark window positions that fall outside the episode. Those are
# filled by repeating the nearest real frame, and a policy should ignore them.
print("observation.state_is_pad:", after["observation.state_is_pad"].tolist())
print("action_is_pad:           ", after["action_is_pad"].tolist())
print()
print(f"Plots saved to {OUT_DIR}/")
