"""Load the pretrained PushT diffusion policy together with its pre/post processors."""

from pathlib import Path

import torch

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

# Written by scripts/prepare_policy.py (the Hub checkpoint needs converting for lerobot 0.6.1).
POLICY_DIR = Path("outputs/models/diffusion_pusht")


def load_policy(device: torch.device, n_action_steps: int | None = None):
    """Return (policy, preprocessor, postprocessor), ready for inference on `device`.

    n_action_steps: how many actions from each predicted chunk to execute before predicting
    again. None keeps the checkpoint's value (8).
    """
    if not (POLICY_DIR / "policy_preprocessor.json").exists():
        raise SystemExit(f"{POLICY_DIR} not found. Run `python scripts/prepare_policy.py` first.")

    # Edit the config before building the model: from_pretrained moves the weights to
    # config.device, which the checkpoint saved as "cuda".
    config = DiffusionConfig.from_pretrained(POLICY_DIR)
    config.device = str(device)
    if n_action_steps is not None:
        # n_action_steps is only read when the action queue is created (policy.reset()) and when
        # a predicted chunk is sliced (generate_actions()), so it can differ from training.
        config.n_action_steps = n_action_steps

    policy = DiffusionPolicy.from_pretrained(POLICY_DIR, config=config)
    policy.eval()

    # The saved preprocessor ends with "move to device"; point that step at our device.
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(POLICY_DIR),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor
