"""Part 5, step 5: plot my training loop's loss curve over the Part 3 lerobot-train curve.

Both runs use the same ACT config (chunk 20, batch 8, AMP, seed 1000, 30k steps). Two
differences are expected: my loop trains on 90% of the episodes (10% held out for validation) and
shuffles the data in a different order, so the curves should be close but not identical.

lerobot-train logged a loss averaged over every 100 steps; my loop logs one averaged over every
1000. To compare like with like, the lerobot numbers are averaged into the same 1000-step windows.

    python scripts/compare_to_lerobot_train.py
Writes outputs/part5/loss_vs_lerobot_train.png and prints the per-window gap.
"""

import argparse
import csv
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse the log parser from eval_checkpoints.py (lerobot rounds the printed step, see there).
_spec = importlib.util.spec_from_file_location("eval_checkpoints", Path(__file__).with_name("eval_checkpoints.py"))
_eval_checkpoints = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_checkpoints)

parser = argparse.ArgumentParser()
parser.add_argument("--mine", type=Path, default=Path("outputs/part5/run/metrics.csv"))
parser.add_argument("--lerobot-log", type=Path, default=Path("outputs/part3/logs/train.log"))
parser.add_argument("--out", type=Path, default=Path("outputs/part5/loss_vs_lerobot_train.png"))
args = parser.parse_args()

mine = list(csv.DictReader(open(args.mine)))
my_steps = np.array([int(r["step"]) for r in mine])
my_loss = np.array([float(r["train_loss"]) for r in mine])
window = int(my_steps[0])  # the log frequency of my run (first row is at step log_freq)

ref = _eval_checkpoints.parse_train_log(args.lerobot_log, log_freq=100)
# Average lerobot's 100-step windows into my windows: window k covers steps (k-1)*window .. k*window.
ref_loss = np.array([ref["loss"][(ref["step"] > s - window) & (ref["step"] <= s)].mean() for s in my_steps])

gap = (my_loss - ref_loss) / ref_loss
print(f"{'step':>6} {'mine':>8} {'lerobot':>8} {'gap':>7}")
for s, m, r, g in zip(my_steps, my_loss, ref_loss, gap):
    print(f"{s:>6} {m:>8.4f} {r:>8.4f} {g:>+7.1%}")
print(f"\nmedian gap {np.median(gap):+.1%}; after step 5000: {np.median(gap[my_steps > 5000]):+.1%} "
      f"(max |gap| {np.abs(gap[my_steps > 5000]).max():.1%})")

fig, (ax, ax_gap) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax.plot(ref["step"], ref["loss"], color="tab:gray", alpha=0.3, linewidth=1, label="lerobot-train (100-step windows)")
ax.plot(my_steps, ref_loss, color="tab:gray", marker="o", markersize=3, label="lerobot-train (1000-step windows)")
ax.plot(my_steps, my_loss, color="tab:blue", marker="o", markersize=3, label="my loop, tiny_vla.train (1000-step windows)")
ax.set_yscale("log")
ax.set_ylabel("train loss = L1 + 10 * KL (log scale)")
ax.set_title("ACT on PushT: my training loop vs lerobot-train, same config")
ax.legend()
ax.grid(alpha=0.3)
ax_gap.axhline(0, color="black", linewidth=0.8)
ax_gap.plot(my_steps, gap * 100, color="tab:blue", marker="o", markersize=3)
ax_gap.set_ylabel("mine vs lerobot (%)")
ax_gap.set_xlabel("training step")
ax_gap.grid(alpha=0.3)
fig.tight_layout()
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=120)
print(f"Saved {args.out}")
