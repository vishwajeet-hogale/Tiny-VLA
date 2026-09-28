"""Watch the pretrained diffusion policy play PushT.

If a display is available ($DISPLAY is set) the rollout plays live in an OpenCV window:
    q  quit
    r  reset (start a new episode with the next seed)
Otherwise (or with --save) each episode is saved as an MP4 in outputs/part2/videos/.

Overlay: step count, reward and coverage, plus the action chunk the policy is currently
executing: the big red dot is this step's target, smaller dots are the queued future targets
(fading from orange to yellow the further ahead they are).

    python scripts/watch.py                      # live window, seeds 1000, 1001, ...
    python scripts/watch.py --save --episodes 3  # MP4s instead
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from lerobot.utils.io_utils import write_video
from tiny_vla.device import get_device
from tiny_vla.policy import load_policy
from tiny_vla.rollout import make_env, run_episode

VIDEO_DIR = Path("outputs/part2/videos")
WINDOW = "tiny-vla: PushT"


def draw_overlay(frame: np.ndarray, info: dict) -> np.ndarray:
    """Return a copy of an RGB frame with the text and action-chunk dots drawn on it."""
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


def watch_in_window(policy, env, preprocessor, postprocessor, seed: int, delay_ms: int):
    """Play episodes live until the user presses q or closes the window."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    quit_requested = False

    def show(frame, info) -> bool:
        nonlocal quit_requested
        cv2.imshow(WINDOW, cv2.cvtColor(draw_overlay(frame, info), cv2.COLOR_RGB2BGR))  # cv2 wants BGR
        key = cv2.waitKey(delay_ms) & 0xFF
        window_closed = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
        if key == ord("q") or window_closed:
            quit_requested = True
        return quit_requested or key == ord("r")  # True -> run_episode stops this episode

    while not quit_requested:
        result = run_episode(policy, env, preprocessor, postprocessor, seed, render=True, on_step=show)
        print(f"seed {seed}: success={result.success} steps={result.steps} max_coverage={result.max_coverage:.0%}")
        seed += 1
    cv2.destroyAllWindows()


def save_videos(policy, env, preprocessor, postprocessor, seed: int, episodes: int):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for seed in range(seed, seed + episodes):
        frames = []

        def collect(frame, info) -> bool:  # draw the overlay on each frame as it is rendered
            frames.append(draw_overlay(frame, info))
            return False

        result = run_episode(policy, env, preprocessor, postprocessor, seed, render=True, on_step=collect)
        path = VIDEO_DIR / f"seed_{seed}_{'success' if result.success else 'fail'}.mp4"
        write_video(path, frames, fps=env.metadata["render_fps"])
        print(f"seed {seed}: success={result.success} steps={result.steps} "
              f"max_coverage={result.max_coverage:.0%} -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None, help="torch device (default: cuda, error if unavailable)")
    parser.add_argument("--seed", type=int, default=1000, help="seed of the first episode")
    parser.add_argument("--save", action="store_true", help="save MP4s even if a display is available")
    parser.add_argument("--episodes", type=int, default=1, help="episodes to save in MP4 mode")
    parser.add_argument("--delay-ms", type=int, default=50, help="pause per frame in window mode")
    args = parser.parse_args()

    policy, preprocessor, postprocessor = load_policy(get_device(args.device))
    env = make_env()

    if os.environ.get("DISPLAY") and not args.save:
        print(f"DISPLAY={os.environ['DISPLAY']}: opening a window (q quit, r reset)")
        watch_in_window(policy, env, preprocessor, postprocessor, args.seed, args.delay_ms)
    else:
        print(f"No display (or --save): writing MP4s to {VIDEO_DIR}/")
        save_videos(policy, env, preprocessor, postprocessor, args.seed, args.episodes)
    env.close()


if __name__ == "__main__":
    main()
