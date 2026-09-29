"""Draw the rollout overlay on rendered frames and save episodes as MP4s."""

from pathlib import Path

import cv2
import numpy as np

from lerobot.utils.io_utils import write_video
from tiny_vla.rollout import EpisodeResult, run_episode


def draw_overlay(frame: np.ndarray, info: dict) -> np.ndarray:
    """Return a copy of an RGB frame with the step/reward text and action-chunk dots drawn on it.

    The big red dot is this step's target; smaller dots are the queued future targets, fading
    from orange to yellow the further ahead they are.
    """
    frame = frame.copy()
    height, width = frame.shape[:2]

    # Actions are in the sim's 512x512 coordinates; the rendered frame is smaller, so scale.
    chunk = info["chunk"]
    if chunk is not None:
        for i, (x, y) in reversed(list(enumerate(chunk))):  # draw the current target last (on top)
            center = (int(x / 512 * width), int(y / 512 * height))
            if i == 0:
                cv2.circle(frame, center, 7, (220, 30, 30), -1)
            else:
                fade = i / max(len(chunk) - 1, 1)  # 0 = next action ... 1 = furthest ahead
                cv2.circle(frame, center, 4, (255, int(140 + 100 * fade), 0), -1)

    lines = [
        f"step {info['step']}",
        f"reward {info['reward']:.2f}  coverage {info['coverage']:.0%}",
    ]
    for row, text in enumerate(lines):
        origin = (8, 22 + 22 * row)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def record_episode(policy, env, preprocessor, postprocessor, seed: int, out_dir: Path) -> tuple[EpisodeResult, Path]:
    """Run one episode, drawing the overlay on every frame, and save it as an MP4 in out_dir."""
    frames = []

    def collect(frame, info) -> bool:
        frames.append(draw_overlay(frame, info))
        return False  # never stop early

    result = run_episode(policy, env, preprocessor, postprocessor, seed, render=True, on_step=collect)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"seed_{seed}_{'success' if result.success else 'fail'}.mp4"
    write_video(path, frames, fps=env.metadata["render_fps"])
    return result, path
