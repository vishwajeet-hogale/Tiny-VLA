"""Load a saved policy (diffusion, ACT, ...) together with its pre/post processors."""

from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

# Written by scripts/prepare_policy.py (the Hub checkpoint needs converting for lerobot 0.6.1).
POLICY_DIR = Path("outputs/models/diffusion_pusht")


def load_pretrained(path: str | Path, device: torch.device, n_action_steps: int | None = None, **config_overrides):
    """Load any policy folder that has config.json, model.safetensors and processor files.

    Works for the converted diffusion checkpoint and for lerobot-train checkpoints
    (outputs/.../checkpoints/<step>/pretrained_model). Returns (policy, preprocessor, postprocessor).

    n_action_steps: how many actions from each predicted chunk to execute before predicting
    again. None keeps the checkpoint's value.
    config_overrides: any other config fields to change before the model is built, e.g.
    noise_scheduler_type="DDIM", num_inference_steps=10 for a diffusion policy.
    """
    path = Path(path)
    if not (path / "policy_preprocessor.json").exists():
        raise SystemExit(f"{path} has no policy_preprocessor.json (not a lerobot 0.6 policy folder).")

    # Edit the config before building the model: from_pretrained moves the weights to
    # config.device, which is whatever device the checkpoint was saved from.
    config = PreTrainedConfig.from_pretrained(path)  # reads "type" to pick DiffusionConfig, ACTConfig, ...
    config.device = str(device)
    if n_action_steps is not None:
        # n_action_steps is only read when the action queue is created (policy.reset()) and when
        # a predicted chunk is sliced, so it can differ from training.
        config.n_action_steps = n_action_steps
    for key, value in config_overrides.items():
        if not hasattr(config, key):
            raise ValueError(f"{type(config).__name__} has no field {key!r}")
        setattr(config, key, value)

    policy = get_policy_class(config.type).from_pretrained(path, config=config)
    policy.eval()

    # The saved preprocessor ends with "move to device"; point that step at our device.
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor


def load_policy(device: torch.device, n_action_steps: int | None = None):
    """The pretrained PushT diffusion policy (Part 2)."""
    if not POLICY_DIR.exists():
        raise SystemExit(f"{POLICY_DIR} not found. Run `python scripts/prepare_policy.py` first.")
    return load_pretrained(POLICY_DIR, device, n_action_steps)
