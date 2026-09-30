"""Evaluate saved policies with the hand-written rollout loop (tiny_vla.rollout)."""

from pathlib import Path

import numpy as np
import torch

from tiny_vla.policy import load_pretrained
from tiny_vla.rollout import EpisodeResult, run_episode


def find_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """[(step, pretrained_model dir)] for a lerobot-train or tiny_vla.train run, sorted by step."""
    found = []
    for step_dir in (Path(run_dir) / "checkpoints").iterdir():
        if step_dir.name.isdigit() and (step_dir / "pretrained_model").is_dir():
            found.append((int(step_dir.name), step_dir / "pretrained_model"))
    return sorted(found)


def evaluate_policy_dir(path: Path, device: torch.device, env, seeds: list[int], **load_kwargs) -> list[EpisodeResult]:
    """Load one policy folder and run one episode per seed. Frees the GPU memory afterwards."""
    policy, preprocessor, postprocessor = load_pretrained(path, device, **load_kwargs)
    results = [run_episode(policy, env, preprocessor, postprocessor, seed) for seed in seeds]
    del policy
    torch.cuda.empty_cache()
    return results


def summarize(results: list[EpisodeResult]) -> dict:
    """Average the per-episode numbers into one row."""
    return {
        "episodes": len(results),
        "success_rate": float(np.mean([r.success for r in results])),
        "avg_max_coverage": float(np.mean([r.max_coverage for r in results])),
        "avg_episode_length": float(np.mean([r.steps for r in results])),
        "model_calls_per_episode": float(np.mean([r.chunk_predictions for r in results])),
        "wall_time_per_episode_s": float(np.mean([r.wall_time_s for r in results])),
        "peak_vram_mb": float(max(r.peak_vram_mb or 0 for r in results)),
    }
