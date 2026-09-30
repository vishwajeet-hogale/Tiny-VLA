"""Part 6, B4: do the policies see more than one way to push the T?

For 6 starting seeds, draw on the first frame the action chunks each policy would predict there:
    blue    50 Diffusion samples (50 different starting noises)
    red     ACT with z = 0 (how it runs at test time)
    orange  50 ACT samples with z drawn from N(0, I) (what the CVAE latent can express)
    green   ACT trained without the VAE
    gray    the agent paths of the 3 demos whose first frame looks most like this start

Picking seeds: the dataset stores only the image and agent position, not the block pose, so there
is no direct way to find "the demos that started here". Instead, 60 candidate seeds are scored by
how bimodal the Diffusion samples are: 2-means on where each sampled path ends, and the score is
the distance between the two cluster centers in pixels (0 if either cluster has under 10% of the
samples). The 4 most bimodal seeds are kept, plus the 2 least bimodal as a contrast. The gray demo
paths come from demos whose agent started within 100 px of this seed's agent, ranked by first-frame
pixel distance: did humans go different ways from similar starts?

Numbers per seed (probe_stats.csv): the spread of each policy's 50 endpoints (mean distance to
their average, px), and for ACT how far the chunk moves when z is pushed to an extreme (3 in every
dimension). A near-zero spread with a clear extreme shift means the hook works but the decoder
ignores z at normal magnitudes.

    python scripts/multimodality_probe.py
    python scripts/multimodality_probe.py --diffusion outputs/models/diffusion_pusht   # the reference model
Writes outputs/part6/probe/seed_<seed>.png and probe_seeds.csv.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.datasets import LeRobotDataset
from tiny_vla.device import get_device
from tiny_vla.policy import load_pretrained
from tiny_vla.rollout import make_env, sim_obs_to_policy_obs

N_SAMPLES = 50
CANDIDATE_SEEDS = list(range(2000, 2060))
OUT_DIR = Path("outputs/part6/probe")

parser = argparse.ArgumentParser()
parser.add_argument("--diffusion", default="outputs/part6/colab/diffusion_pusht/checkpoints/050000/pretrained_model")
parser.add_argument("--act", default="outputs/part3/long/act_pusht/checkpoints/150000/pretrained_model")
parser.add_argument("--act-novae", default="outputs/part6/colab/act_novae_pusht/checkpoints/030000/pretrained_model")
args = parser.parse_args()
device = get_device("cuda")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def first_observation(env, seed):
    obs, _ = env.reset(seed=seed)
    return obs, env.render()  # policy observation (96x96) and a 384x384 frame to draw on


@torch.no_grad()
def diffusion_samples(policy, preprocessor, postprocessor, obs, n) -> np.ndarray:
    """n chunks from n different noises for the same observation: (n, future_steps, 2) pixels.

    At the first step the policy's 2-frame history is just the current frame twice (that's what
    populate_queues does on the first select_action). We call the model's own sampler directly,
    so every sample gets the full horizon, not only the n_action_steps it would execute.
    """
    batch = preprocessor(sim_obs_to_policy_obs(obs))
    n_obs = policy.config.n_obs_steps
    model_input = {
        "observation.state": batch["observation.state"][:, None].repeat(n, n_obs, 1),  # (n, 2 frames, 2)
        "observation.images": batch["observation.image"][:, None, None].repeat(n, n_obs, 1, 1, 1, 1),
    }
    global_cond = policy.diffusion._prepare_global_conditioning(model_input)
    chunk = policy.diffusion.conditional_sample(n, global_cond=global_cond)  # (n, horizon, 2), normalized
    future = chunk[:, n_obs - 1 :]  # the horizon starts at the oldest observation; keep now and later
    return postprocessor(future).numpy()


@torch.no_grad()
def act_chunks(policy, preprocessor, postprocessor, obs, n, sample_z: bool, z_value: float | None = None) -> np.ndarray:
    """ACT chunks: (n, chunk_size, 2) pixels. sample_z=False gives the test-time z = 0 chunk;
    z_value sets every dimension of z to that number instead (for the sensitivity check).

    In eval mode ACT always feeds z = 0 into encoder_latent_input_proj. A forward pre-hook on
    that layer swaps its input for samples from N(0, I), with no change to lerobot's code.
    """
    batch = preprocessor(sim_obs_to_policy_obs(obs))
    batch = {k: v.repeat(n, *[1] * (v.dim() - 1)) if torch.is_tensor(v) else v for k, v in batch.items()}
    handle = None
    if sample_z or z_value is not None:
        def replace_z(module, inputs):
            z = inputs[0]
            return (torch.full_like(z, z_value) if z_value is not None else torch.randn_like(z),)
        handle = policy.model.encoder_latent_input_proj.register_forward_pre_hook(replace_z)
    try:
        chunk = policy.predict_action_chunk(batch)
    finally:
        if handle is not None:
            handle.remove()
    return postprocessor(chunk).numpy()


def bimodality(endpoints: np.ndarray) -> float:
    """Distance in pixels between the 2 clusters of endpoints (0 = one blob or a tiny minority)."""
    rng = np.random.default_rng(0)
    centers = endpoints[rng.choice(len(endpoints), 2, replace=False)]
    for _ in range(20):  # 2-means
        labels = np.argmin(((endpoints[:, None] - centers[None]) ** 2).sum(-1), axis=1)
        if min((labels == 0).sum(), (labels == 1).sum()) == 0:
            return 0.0
        centers = np.array([endpoints[labels == k].mean(0) for k in range(2)])
    if min((labels == 0).mean(), (labels == 1).mean()) < 0.1:
        return 0.0
    return float(np.linalg.norm(centers[0] - centers[1]))


def spread(endpoints: np.ndarray) -> float:
    """Mean distance (px) of sampled endpoints from their average: 0 = every sample ends in the same place."""
    return float(np.linalg.norm(endpoints - endpoints.mean(0), axis=1).mean())


def load(path, name):
    if not (Path(path) / "config.json").exists():
        print(f"WARNING: no {name} at {path}; it will be left out of the probe")
        return None
    return load_pretrained(path, device)


env = make_env()
diffusion = load(args.diffusion, "Diffusion")
if diffusion is None:
    raise SystemExit("The probe needs a Diffusion policy (it picks the seeds). Pass --diffusion.")

# 1. Score the candidate seeds by how bimodal the Diffusion samples are.
torch.manual_seed(0)
scores = []
for seed in CANDIDATE_SEEDS:
    obs, _ = first_observation(env, seed)
    samples = diffusion_samples(*diffusion, obs, N_SAMPLES)
    scores.append((bimodality(samples[:, -1]), seed))
scores.sort(reverse=True)
chosen = [s for _, s in scores[:4]] + [s for _, s in scores[-2:]]
print("endpoint cluster gap in px (top 4 + bottom 2):", [(s, round(b)) for b, s in scores[:4] + scores[-2:]])
with open(OUT_DIR / "probe_seeds.csv", "w", newline="") as f:
    csv.writer(f).writerows([("seed", "diffusion_cluster_gap_px", "chosen")] +
                            [(s, round(b, 3), s in chosen) for b, s in scores])

# 2. First frames of every dataset episode, for the nearest-demo lookup (image distance).
dataset = LeRobotDataset("lerobot/pusht", video_backend="pyav")
episodes = dataset.meta.episodes
first_frames = torch.stack([dataset[i]["observation.image"] for i in episodes["dataset_from_index"]])  # (206,3,96,96)
states = torch.stack(list(dataset.hf_dataset["observation.state"]))  # all agent positions, (frames, 2)
start_states = states[torch.tensor(episodes["dataset_from_index"])]  # agent position at each demo's start

act = load(args.act, "ACT")
act_novae = load(args.act_novae, "ACT without VAE")

score_of = {seed: b for b, seed in scores}
all_stats = []
for seed in chosen:
    obs, frame = first_observation(env, seed)
    scale = frame.shape[0] / 512  # actions are in 512x512 sim coordinates
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(frame)

    image = torch.from_numpy(obs["pixels"]).permute(2, 0, 1).float() / 255
    distance = ((first_frames - image) ** 2).mean(dim=(1, 2, 3))
    agent_gap = (start_states - torch.from_numpy(obs["agent_pos"]).float()).norm(dim=1)
    distance[agent_gap > 100] = float("inf")  # only demos whose agent started nearby
    nearest = [ep for ep in distance.argsort()[:3].tolist() if torch.isfinite(distance[ep])]
    for rank, ep in enumerate(nearest):
        path = states[episodes["dataset_from_index"][ep] : episodes["dataset_to_index"][ep]][:40].numpy()
        ax.plot(path[:, 0] * scale, path[:, 1] * scale, color="0.35", linestyle="--", linewidth=1.3,
                label="3 most similar demos (first 4 s)" if rank == 0 else None)

    torch.manual_seed(seed)
    diff = diffusion_samples(*diffusion, obs, N_SAMPLES)
    stats = {"seed": seed, "diffusion_cluster_gap_px": score_of[seed], "diffusion_spread_px": spread(diff[:, -1])}
    for i, chunk in enumerate(diff):
        ax.plot(chunk[:, 0] * scale, chunk[:, 1] * scale, color="tab:blue", alpha=0.25, linewidth=1,
                label=f"Diffusion, {N_SAMPLES} noise samples" if i == 0 else None)
    if act is not None:
        act_z = act_chunks(*act, obs, N_SAMPLES, sample_z=True)
        act_zero = act_chunks(*act, obs, 1, sample_z=False)[0]
        act_extreme = act_chunks(*act, obs, 1, sample_z=False, z_value=3.0)[0]
        stats["act_z_spread_px"] = spread(act_z[:, -1])
        stats["act_z_extreme_shift_px"] = float(np.linalg.norm(act_extreme - act_zero, axis=1).mean())
        for i, chunk in enumerate(act_z):
            ax.plot(chunk[:, 0] * scale, chunk[:, 1] * scale, color="tab:orange", alpha=0.25, linewidth=1,
                    label=f"ACT, {N_SAMPLES} samples of z ~ N(0, I)" if i == 0 else None)
        ax.plot(act_zero[:, 0] * scale, act_zero[:, 1] * scale, color="tab:red", linewidth=3, label="ACT, z = 0 (test time)")
    if act_novae is not None:
        chunk = act_chunks(*act_novae, obs, 1, sample_z=False)[0]
        ax.plot(chunk[:, 0] * scale, chunk[:, 1] * scale, color="tab:green", linewidth=3, label="ACT without VAE")

    agent = obs["agent_pos"] * scale
    ax.scatter([agent[0]], [agent[1]], s=60, color="black", zorder=5, label="agent start")
    score = score_of[seed]
    ax.set_title(f"seed {seed}: Diffusion endpoint clusters {score:.0f} px apart")
    ax.axis("off")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"seed_{seed}.png", dpi=110)
    plt.close(fig)
    print(f"saved {OUT_DIR}/seed_{seed}.png  " + "  ".join(f"{k}={v:.1f}" for k, v in stats.items() if k != "seed"))
    all_stats.append(stats)
env.close()

with open(OUT_DIR / "probe_stats.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_stats[0]))
    writer.writeheader()
    writer.writerows(all_stats)
print(f"saved {OUT_DIR}/probe_stats.csv")
