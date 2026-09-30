"""Part 6, B3: evaluate several PushT policies on the same 200 seeds and compare them.

Default models (a missing one is skipped with a warning):
    act_part3       my ACT from the Part 3 long run (batch 32, step 150k: best of that run)
    act_novae       ACT with use_vae=false, trained on Colab (notebooks/train_diffusion_colab.ipynb)
    diffusion_mine  my Diffusion Policy, 50k steps on Colab
    diffusion_ref   lerobot/diffusion_pusht, converted by scripts/prepare_policy.py
Override with --model name=path (repeatable).

For each model it reports:
    success rate with a Wilson 95% CI, average max coverage
    failure breakdown by how far a failed episode got (max coverage):
        near miss >= 0.85 | partial 0.30-0.85 | never got going < 0.30
        plus "stalled": failed episodes whose agent barely moved over their last 50 steps
    smoothness of the executed targets, in pixels per step:
        step: mean |a_t - a_(t-1)|, jerk: mean |a_t - 2a_(t-1) + a_(t-2)|
        boundary jump: mean |a_t - a_(t-1)| on the first step of each new chunk, divided by the
        same inside chunks (1.0 = re-planning is invisible; >1 = the path jumps when a new chunk starts)
    inference latency: mean GPU-synced time of a model call (ms), and model calls per episode
Then pairwise verdicts (McNemar, from compare.py) for every pair.

Diffusion is slow here: ~70 s per episode at batch size 1, so ~4 h per diffusion model for 200
seeds. Results are cached per model in outputs/part6/eval/, so the script can be stopped and
re-run: finished models are loaded from the cache.

    python scripts/compare_models.py
Writes outputs/part6/results.md and outputs/part6/eval/<model>_episodes.csv (+ _actions.npz).
"""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from compare import verdict, wilson_interval
from tiny_vla.device import get_device
from tiny_vla.evaluate import evaluate_policy_dir
from tiny_vla.rollout import make_env

SEEDS = list(range(2000, 2200))  # 200 seeds, disjoint from the 1000-1099 used for picking checkpoints
DEFAULT_MODELS = {
    "act_part3": "outputs/part3/long/act_pusht/checkpoints/150000/pretrained_model",
    "act_novae": "outputs/part6/colab/act_novae_pusht/checkpoints/030000/pretrained_model",
    "diffusion_mine": "outputs/part6/colab/diffusion_pusht/checkpoints/050000/pretrained_model",
    "diffusion_ref": "outputs/models/diffusion_pusht",
}
OUT_DIR = Path("outputs/part6")
CACHE_DIR = OUT_DIR / "eval"


def smoothness(actions: np.ndarray, n_action_steps: int) -> dict:
    """Step size, jerk, and the jump at chunk boundaries for one episode's executed targets."""
    if len(actions) < 3:
        return {"step_px": np.nan, "jerk_px": np.nan, "boundary_ratio": np.nan}
    step = np.linalg.norm(np.diff(actions, axis=0), axis=1)  # step[i] = |a_(i+1) - a_i|
    jerk = np.linalg.norm(np.diff(actions, n=2, axis=0), axis=1)
    # A new chunk starts at t = 0, n, 2n, ...; the jump into it is step[t - 1] for t = n, 2n, ...
    boundary = np.zeros(len(step), dtype=bool)
    boundary[n_action_steps - 1 :: n_action_steps] = True
    inside = step[~boundary].mean() if (~boundary).any() else np.nan
    ratio = step[boundary].mean() / inside if boundary.any() and inside > 0 else np.nan
    return {"step_px": step.mean(), "jerk_px": jerk.mean(), "boundary_ratio": ratio}


def evaluate_or_load(name: str, path: Path, device, env) -> list[dict]:
    """Per-episode rows for one model, from the cache if this model already ran."""
    csv_path, npz_path = CACHE_DIR / f"{name}_episodes.csv", CACHE_DIR / f"{name}_actions.npz"
    source = CACHE_DIR / f"{name}_source.txt"  # which checkpoint the cached results came from
    if csv_path.exists() and npz_path.exists() and source.exists() and source.read_text() == str(path):
        rows = list(csv.DictReader(open(csv_path)))
        if [int(r["seed"]) for r in rows] == SEEDS:
            print(f"{name}: loaded {len(rows)} cached episodes")
            actions = np.load(npz_path)
            for r in rows:
                r["actions"] = actions[str(r["seed"])]
            return rows
    # Read n_action_steps from the saved config without loading the model.
    n_action_steps = json.loads((path / "config.json").read_text())["n_action_steps"]
    print(f"{name}: evaluating {len(SEEDS)} seeds (n_action_steps={n_action_steps}) ...", flush=True)
    results = evaluate_policy_dir(path, device, env, SEEDS)
    rows = []
    for r in results:
        row = {**r.row(), **smoothness(r.actions, n_action_steps), "n_action_steps": n_action_steps}
        rows.append(row)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(npz_path, **{str(r.seed): r.actions for r in results})
    source.write_text(str(path))
    for row, r in zip(rows, results):
        row["actions"] = r.actions
    return rows


def as_bool(v) -> bool:
    return v is True or v in ("True", "true", "1")


def summarize(rows: list[dict]) -> dict:
    success = np.array([as_bool(r["success"]) for r in rows])
    cov = np.array([float(r["max_coverage"]) for r in rows])
    failed = ~success
    lo, hi = wilson_interval(int(success.sum()), len(rows))
    # "Stalled": in a failed episode, the target barely moved over the last 50 steps (< 1 px/step).
    stalled = sum(
        1 for r, f in zip(rows, failed)
        if f and len(r["actions"]) > 50 and np.linalg.norm(np.diff(r["actions"][-50:], axis=0), axis=1).mean() < 1.0
    )
    return {
        "success": success.mean(), "ci_lo": lo, "ci_hi": hi,
        "max_cov": cov.mean(),
        "fail_near": int((failed & (cov >= 0.85)).sum()),
        "fail_partial": int((failed & (cov >= 0.30) & (cov < 0.85)).sum()),
        "fail_none": int((failed & (cov < 0.30)).sum()),
        "fail_stalled": stalled,
        "step_px": np.nanmean([float(r["step_px"]) for r in rows]),
        "jerk_px": np.nanmean([float(r["jerk_px"]) for r in rows]),
        "boundary_ratio": np.nanmean([float(r["boundary_ratio"]) for r in rows]),
        "latency_ms": np.nanmean([float(r["mean_chunk_latency_ms"]) for r in rows]),
        "calls_per_ep": np.mean([float(r["chunk_predictions"]) for r in rows]),
        "n": len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", default=[], help="name=path (repeatable); replaces the defaults")
    args = parser.parse_args()
    models = dict(m.split("=", 1) for m in args.model) if args.model else DEFAULT_MODELS
    device = get_device("cuda")

    available = {}
    for name, path in models.items():
        if (Path(path) / "config.json").exists():
            available[name] = Path(path)
        else:
            print(f"WARNING: skipping {name}, no policy at {path}")
    if len(available) < 1:
        raise SystemExit("No models to evaluate.")

    env = make_env()
    episodes = {name: evaluate_or_load(name, path, device, env) for name, path in available.items()}
    env.close()
    summary = {name: summarize(rows) for name, rows in episodes.items()}

    lines = [f"# Part 6: PushT policies on the same {len(SEEDS)} seeds ({SEEDS[0]}-{SEEDS[-1]})", ""]
    lines += [
        "| model | success (95% CI) | avg max coverage | failures: near / partial / none (stalled) "
        "| step px | jerk px | chunk-boundary jump | model call ms | calls / ep |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summary.items():
        n_fail = s["fail_near"] + s["fail_partial"] + s["fail_none"]
        lines.append(
            f"| {name} | {s['success']:.0%} ({s['ci_lo']:.0%}-{s['ci_hi']:.0%}) | {s['max_cov']:.2f} "
            f"| {s['fail_near']} / {s['fail_partial']} / {s['fail_none']} of {n_fail} ({s['fail_stalled']}) "
            f"| {s['step_px']:.1f} | {s['jerk_px']:.1f} | {s['boundary_ratio']:.2f}x | {s['latency_ms']:.1f} "
            f"| {s['calls_per_ep']:.0f} |"
        )
    lines += ["", "## Pairwise verdicts (McNemar exact test on paired seeds, alpha = 0.05)", ""]
    success_by_model = {n: {int(r["seed"]): as_bool(r["success"]) for r in rows} for n, rows in episodes.items()}
    for a, b in itertools.combinations(success_by_model, 2):
        lines.append(f"- {verdict(a, success_by_model[a], b, success_by_model[b])}")
    text = "\n".join(lines) + "\n"
    print("\n" + text)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.md").write_text(text)
    print(f"Saved {OUT_DIR}/results.md")


if __name__ == "__main__":
    main()
