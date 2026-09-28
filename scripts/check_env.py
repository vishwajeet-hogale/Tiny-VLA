"""Sanity-check the environment: print library versions and the best available device."""

import lerobot
import torch

# Importing gym_pusht registers the "gym_pusht/PushT-v0" environment with gymnasium.
import gym_pusht  # noqa: F401


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = pick_device()

print(f"lerobot version: {lerobot.__version__}")
print(f"torch version:   {torch.__version__}")
print(f"device:          {device}")
if device == "cuda":
    print(f"gpu:             {torch.cuda.get_device_name(0)}")
print("gym_pusht imported OK")
