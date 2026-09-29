"""Watch a policy play PushT (default: the pretrained diffusion policy from Part 2).

If a display is available ($DISPLAY is set) the rollout plays live in an OpenCV window:
    q  quit
    r  reset (start a new episode with the next seed)
Otherwise (or with --save) each episode is saved as an MP4 in outputs/part2/videos/.

Overlay: step count, reward and coverage, plus the action chunk the policy is currently
executing: the big red dot is this step's target, smaller dots are the queued future targets
(fading from orange to yellow the further ahead they are).

    python scripts/watch.py                      # live window, seeds 1000, 1001, ...
    python scripts/watch.py --save --episodes 3  # MP4s instead
    python scripts/watch.py --policy-path outputs/part3/act_pusht/checkpoints/030000/pretrained_model
"""

import argparse
import os
from pathlib import Path

import cv2

from tiny_vla.device import get_device
from tiny_vla.policy import POLICY_DIR, load_pretrained
from tiny_vla.rollout import make_env, run_episode
from tiny_vla.video import draw_overlay, record_episode

VIDEO_DIR = Path("outputs/part2/videos")
WINDOW = "tiny-vla: PushT"


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


def save_videos(policy, env, preprocessor, postprocessor, seed: int, episodes: int, out_dir: Path):
    for seed in range(seed, seed + episodes):
        result, path = record_episode(policy, env, preprocessor, postprocessor, seed, out_dir)
        print(f"seed {seed}: success={result.success} steps={result.steps} "
              f"max_coverage={result.max_coverage:.0%} -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None, help="torch device (default: cuda, error if unavailable)")
    parser.add_argument("--seed", type=int, default=1000, help="seed of the first episode")
    parser.add_argument("--save", action="store_true", help="save MP4s even if a display is available")
    parser.add_argument("--episodes", type=int, default=1, help="episodes to save in MP4 mode")
    parser.add_argument("--delay-ms", type=int, default=50, help="pause per frame in window mode")
    parser.add_argument("--policy-path", default=str(POLICY_DIR), help="policy folder (config + weights + processors)")
    parser.add_argument("--video-dir", default=str(VIDEO_DIR), help="where MP4s go in save mode")
    args = parser.parse_args()

    policy, preprocessor, postprocessor = load_pretrained(args.policy_path, get_device(args.device))
    env = make_env()

    if os.environ.get("DISPLAY") and not args.save:
        print(f"DISPLAY={os.environ['DISPLAY']}: opening a window (q quit, r reset)")
        watch_in_window(policy, env, preprocessor, postprocessor, args.seed, args.delay_ms)
    else:
        print(f"No display (or --save): writing MP4s to {args.video_dir}/")
        save_videos(policy, env, preprocessor, postprocessor, args.seed, args.episodes, Path(args.video_dir))
    env.close()


if __name__ == "__main__":
    main()
