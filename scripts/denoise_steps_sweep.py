"""Part 6, B5: how few denoising steps can Diffusion Policy get away with?

Evaluates one Diffusion checkpoint on 100 seeds for each setting below and plots success rate and
time per model call against the number of denoising steps. The weights never change; only the
sampler does.

    DDPM 100   the setting it was trained with (num_train_timesteps = 100)
    DDIM 25, 10, 5, 2   DDIM takes big deterministic jumps along the same noise schedule, so it can
               skip steps. lerobot 0.6.1 supports it (DiffusionConfig.noise_scheduler_type = "DDIM").
    DDPM 10    control: plain DDPM with 10 of its 100 steps, to show why the scheduler has to change

Time: the DDPM 100 setting alone takes ~2 h here (~70 s per episode); the rest ~1 h together.
Results are cached per setting in outputs/part6/denoise_sweep/, so it can be stopped and re-run.

    python scripts/denoise_steps_sweep.py
    python scripts/denoise_steps_sweep.py --diffusion outputs/models/diffusion_pusht
Writes outputs/part6/denoise_sweep.png and denoise_sweep.csv.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare import wilson_interval
from tiny_vla.device import get_device
from tiny_vla.evaluate import evaluate_policy_dir
from tiny_vla.rollout import make_env

SEEDS = list(range(2000, 2100))  # the first 100 of compare_models.py's seeds
SETTINGS = [("DDIM", 2), ("DDIM", 5), ("DDIM", 10), ("DDIM", 25), ("DDPM", 100), ("DDPM", 10)]
OUT_DIR = Path("outputs/part6")
CACHE_DIR = OUT_DIR / "denoise_sweep"

parser = argparse.ArgumentParser()
parser.add_argument("--diffusion", default="outputs/part6/colab/diffusion_pusht/checkpoints/050000/pretrained_model")
args = parser.parse_args()
device = get_device("cuda")
# One cache folder per checkpoint, so switching --diffusion never reuses another model's results.
CACHE_DIR = CACHE_DIR / args.diffusion.strip("/").replace("/", "__")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
env = make_env()

summary = []
for scheduler, n_steps in SETTINGS:
    cache = CACHE_DIR / f"{scheduler}_{n_steps}.csv"
    if cache.exists():
        rows = list(csv.DictReader(open(cache)))
        print(f"{scheduler} {n_steps:>3}: loaded from cache")
    else:
        print(f"{scheduler} {n_steps:>3}: evaluating {len(SEEDS)} seeds ...", flush=True)
        results = evaluate_policy_dir(args.diffusion, device, env, SEEDS,
                                      noise_scheduler_type=scheduler, num_inference_steps=n_steps)
        rows = [r.row() for r in results]
        with open(cache, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    successes = sum(r["success"] in (True, "True") for r in rows)
    lo, hi = wilson_interval(successes, len(rows))
    summary.append({
        "scheduler": scheduler, "steps": n_steps, "success_rate": successes / len(rows), "ci_lo": lo, "ci_hi": hi,
        "avg_max_coverage": np.mean([float(r["max_coverage"]) for r in rows]),
        "latency_ms": np.nanmean([float(r["mean_chunk_latency_ms"]) for r in rows]),
        "wall_time_per_episode_s": np.mean([float(r["wall_time_s"]) for r in rows]),
    })
    s = summary[-1]
    print(f"   success {s['success_rate']:.0%} ({lo:.0%}-{hi:.0%})  max_cov {s['avg_max_coverage']:.2f}  "
          f"{s['latency_ms']:.0f} ms per model call", flush=True)
env.close()

with open(OUT_DIR / "denoise_sweep.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in summary)

fig, (ax_s, ax_t) = plt.subplots(1, 2, figsize=(12, 4.8))
for scheduler, color, marker in [("DDIM", "tab:blue", "o"), ("DDPM", "tab:red", "s")]:
    rows = sorted((r for r in summary if r["scheduler"] == scheduler), key=lambda r: r["steps"])
    x = [r["steps"] for r in rows]
    y = np.array([r["success_rate"] for r in rows])
    err = np.array([[r["success_rate"] - r["ci_lo"] for r in rows], [r["ci_hi"] - r["success_rate"] for r in rows]])
    ax_s.errorbar(x, y, yerr=err, color=color, marker=marker, capsize=4, linestyle="-" if scheduler == "DDIM" else "none",
                  label=f"{scheduler} (95% CI)")
    ax_t.plot(x, [r["latency_ms"] for r in rows], color=color, marker=marker,
              linestyle="-" if scheduler == "DDIM" else "none", label=scheduler)
for ax in (ax_s, ax_t):
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 25, 100])
    ax.set_xticklabels(["2", "5", "10", "25", "100"])
    ax.set_xlabel("denoising steps per chunk (log scale)")
    ax.grid(alpha=0.3)
    ax.legend()
ax_s.set_ylim(0, 1)
ax_s.set_ylabel(f"success rate ({len(SEEDS)} seeds)")
ax_s.set_title("Quality")
ax_t.set_yscale("log")
ax_t.set_ylabel("time per model call (ms, log scale)")
ax_t.set_title("Speed")
fig.suptitle("Diffusion Policy on PushT: denoising steps vs success and latency")
fig.tight_layout()
fig.savefig(OUT_DIR / "denoise_sweep.png", dpi=120)
print(f"Saved {OUT_DIR}/denoise_sweep.png, denoise_sweep.csv")
