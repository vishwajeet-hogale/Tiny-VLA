"""Part 5, step 6: evaluate every checkpoint of my training loop on 100 seeds.

Uses the same harness as Part 3 (tiny_vla.rollout via tiny_vla.evaluate). Plots train loss, val
loss and success rate against step on one figure, and reports how well each loss predicts success
(Spearman rank correlation across checkpoints).

    python scripts/eval_part5.py
Writes to outputs/part5/:
    checkpoint_eval.csv, checkpoint_eval_episodes.csv, loss_val_success.png
"""

import argparse
import csv

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tiny_vla.device import get_device
from tiny_vla.evaluate import evaluate_policy_dir, find_checkpoints, summarize
from tiny_vla.rollout import make_env

SEEDS = list(range(1000, 1100))  # 100 fixed seeds; the first 20 are the Part 3 eval seeds


def spearman(x, y) -> float:
    """Rank correlation: 1 = y always rises when x rises, -1 = always falls. Needs no scipy."""
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/part5/run"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/part5"))
    args = parser.parse_args()
    device = get_device("cuda")

    checkpoints = find_checkpoints(args.run_dir)
    print(f"Checkpoints: {[s for s, _ in checkpoints]}, {len(SEEDS)} episodes each")
    env = make_env()
    summary, episodes = [], []
    for step, path in checkpoints:
        results = evaluate_policy_dir(path, device, env, SEEDS)
        row = {"step": step, **summarize(results)}
        summary.append(row)
        episodes += [{"step": step, **r.row()} for r in results]
        print(f"step {step:>6}: success {row['success_rate']:.0%}  max_cov {row['avg_max_coverage']:.2f}", flush=True)
    env.close()

    with open(args.out_dir / "checkpoint_eval.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with open(args.out_dir / "checkpoint_eval_episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)

    # Losses from the run's metrics.csv, picked at the checkpoint steps for the correlations.
    metrics = list(csv.DictReader(open(args.run_dir / "metrics.csv")))
    m_step = np.array([int(r["step"]) for r in metrics])
    train_loss = np.array([float(r["train_loss"]) for r in metrics])
    train_l1 = np.array([float(r["train_l1"]) for r in metrics])
    val_l1 = np.array([float(r["val_l1"]) for r in metrics])
    at_ckpt = [int(np.where(m_step == row["step"])[0][0]) for row in summary]
    success = np.array([row["success_rate"] for row in summary])

    # Lower loss should mean higher success, so a useful loss has a strongly *negative* correlation.
    print("\nSpearman correlation with success across checkpoints (more negative = better predictor):")
    for name, values in [("train loss (L1+KL)", train_loss), ("train L1", train_l1), ("val L1", val_l1)]:
        print(f"  {name:<20} {spearman(values[at_ckpt], success):+.2f}")
    print(f"  (only {len(summary)} checkpoints, so these are rough)")

    fig, ax_loss = plt.subplots(figsize=(10, 5))
    ax_loss.plot(m_step, train_loss, color="tab:blue", marker=".", label="train loss (L1 + 10*KL)")
    ax_loss.plot(m_step, train_l1, color="tab:cyan", marker=".", linestyle="--", label="train L1 (CVAE z)")
    ax_loss.plot(m_step, val_l1, color="tab:purple", marker=".", label="val L1 (z = 0, held-out episodes)")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("loss (log scale)")
    ax_success = ax_loss.twinx()
    ax_success.plot([r["step"] for r in summary], success, color="tab:red", marker="o", linewidth=2,
                    label=f"success rate ({len(SEEDS)} seeds)")
    ax_success.set_ylim(0, 1)
    ax_success.set_ylabel("success rate", color="tab:red")
    handles = ax_loss.get_legend_handles_labels()[0] + ax_success.get_legend_handles_labels()[0]
    ax_loss.legend(handles, [h.get_label() for h in handles], loc="center right")
    ax_loss.set_title("ACT trained with my loop: train loss, val loss and rollout success")
    ax_loss.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "loss_val_success.png", dpi=120)
    print(f"Saved {args.out_dir}/loss_val_success.png, checkpoint_eval.csv, checkpoint_eval_episodes.csv")


if __name__ == "__main__":
    main()
