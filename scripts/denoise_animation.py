"""Part 6, B6: watch one Diffusion Policy action chunk form, from pure noise to the final path.

The sampling loop below is DiffusionModel.conditional_sample (lerobot/policies/diffusion/
modeling_diffusion.py) written out, so the chunk can be saved after every denoising step:

    sample = pure Gaussian noise, shape (1, horizon, 2)
    for t in scheduler.timesteps:              # 99, 98, ..., 0 for DDPM with 100 steps
        predicted_noise = unet(sample, t, conditioned on the 2 observation frames)
        sample = scheduler.step(predicted_noise, t, sample)   # remove a little of the noise

Each GIF frame shows the chunk (in pixels, drawn over the start frame) after one step. The view is
larger than the 512x512 workspace, because in normalized units the noise starts at N(0, 1), which
puts most of the first points off the table. The dotted gray line is the final path, shown
throughout for reference.

    python scripts/denoise_animation.py --seed 2000
    python scripts/denoise_animation.py --diffusion outputs/models/diffusion_pusht --seed 2000
Writes outputs/part6/denoise_seed_<seed>.gif.
"""

import argparse
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from tiny_vla.device import get_device
from tiny_vla.policy import load_pretrained
from tiny_vla.rollout import make_env, sim_obs_to_policy_obs

parser = argparse.ArgumentParser()
parser.add_argument("--diffusion", default="outputs/part6/colab/diffusion_pusht/checkpoints/050000/pretrained_model")
parser.add_argument("--seed", type=int, default=2000)
args = parser.parse_args()
device = get_device("cuda")

policy, preprocessor, postprocessor = load_pretrained(args.diffusion, device)
model = policy.diffusion
env = make_env()
obs, _ = env.reset(seed=args.seed)
frame = env.render()

# Conditioning: the first observation, repeated to fill the 2-frame history (as on the first step).
batch = preprocessor(sim_obs_to_policy_obs(obs))
n_obs = policy.config.n_obs_steps
model_input = {
    "observation.state": batch["observation.state"][:, None].repeat(1, n_obs, 1),
    "observation.images": batch["observation.image"][:, None, None].repeat(1, n_obs, 1, 1, 1, 1),
}

torch.manual_seed(args.seed)
with torch.no_grad():
    global_cond = model._prepare_global_conditioning(model_input)
    sample = torch.randn(1, policy.config.horizon, policy.config.action_feature.shape[0], device=device)
    model.noise_scheduler.set_timesteps(model.num_inference_steps)
    history = [(None, sample.clone())]  # (timestep, chunk) after each step; None = the starting noise
    for t in model.noise_scheduler.timesteps:
        predicted_noise = model.unet(sample, torch.full((1,), t, dtype=torch.long, device=device), global_cond=global_cond)
        sample = model.noise_scheduler.step(predicted_noise, t, sample).prev_sample
        history.append((int(t), sample.clone()))

# Unnormalize every intermediate chunk to pixels; keep only now-and-future steps (see the probe).
paths = [postprocessor(chunk[:, n_obs - 1 :])[0].numpy() for _, chunk in history]
final = paths[-1]
n_steps = len(history) - 1
print(f"{n_steps} denoising steps ({policy.config.noise_scheduler_type}), horizon {len(final)} future actions")

gif_frames = []
for i, ((t, _), path) in enumerate(zip(history, paths)):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=80)
    ax.set_facecolor("0.93")
    ax.imshow(frame, extent=(0, 512, 512, 0))
    ax.plot(final[:, 0], final[:, 1], color="0.5", linestyle=":", linewidth=1.2)
    ax.plot(path[:, 0], path[:, 1], color="tab:blue", linewidth=1.5, alpha=0.8)
    ax.scatter(path[:, 0], path[:, 1], c=np.arange(len(path)), cmap="plasma", s=22, zorder=3)
    ax.scatter([path[0, 0]], [path[0, 1]], color="red", s=60, zorder=4, marker="x")  # the action it would execute now
    ax.set_xlim(-300, 812)
    ax.set_ylim(812, -300)  # image coordinates: y down
    ax.set_xticks([])
    ax.set_yticks([])
    label = "pure noise" if t is None else f"after step {i}/{n_steps} (t = {t})"
    ax.set_title(f"seed {args.seed}: {label}", fontsize=10)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    gif_frames.append(Image.open(buffer).convert("P", palette=Image.ADAPTIVE, colors=128))

durations = [600] + [80] * (len(gif_frames) - 2) + [2000]  # linger on the noise and on the result
out = Path("outputs/part6") / f"denoise_seed_{args.seed}.gif"
out.parent.mkdir(parents=True, exist_ok=True)
gif_frames[0].save(out, save_all=True, append_images=gif_frames[1:], duration=durations, loop=0)
print(f"Saved {out} ({len(gif_frames)} frames)")
env.close()
