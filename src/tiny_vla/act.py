"""Build ACT for PushT the same way lerobot-train does (lerobot/scripts/lerobot_train.py).

Shared by the Part 3 scripts that poke at ACT directly: print its config, trace tensor
shapes through it, and probe batch sizes.
"""

import torch

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_policy, make_pre_post_processors

REPO_ID = "lerobot/pusht"
VIDEO_BACKEND = "pyav"  # torchcodec can't load on the Linux box


def build_act(device: torch.device, **config_overrides) -> tuple[ACTPolicy, LeRobotDatasetMetadata]:
    """Create an untrained ACT policy sized for PushT's features (96x96 image, 2D state/action).

    make_policy reads the input/output shapes from the dataset metadata; the ResNet18 backbone
    starts from ImageNet weights (config.pretrained_backbone_weights).
    """
    meta = LeRobotDatasetMetadata(REPO_ID)
    config = ACTConfig(device=str(device), **config_overrides)
    policy = make_policy(config, ds_meta=meta)
    return policy, meta


def make_act_dataset(policy: ACTPolicy, meta: LeRobotDatasetMetadata) -> LeRobotDataset:
    """The training dataset, with the delta_timestamps ACT asks for.

    ACT uses 1 observation (the current frame) and a chunk of `chunk_size` future actions, so
    resolve_delta_timestamps gives {"action": [0, 0.1, ..., (chunk_size - 1) / fps]}.
    """
    delta_timestamps = resolve_delta_timestamps(policy.config, meta)
    return LeRobotDataset(
        REPO_ID, delta_timestamps=delta_timestamps, video_backend=VIDEO_BACKEND, return_uint8=True
    )


def make_act_preprocessor(policy: ACTPolicy, meta: LeRobotDatasetMetadata):
    """Normalization pipeline built from the dataset stats (training has no saved processor yet)."""
    preprocessor, _ = make_pre_post_processors(policy.config, dataset_stats=meta.stats)
    return preprocessor


def prepare_batch(batch: dict, preprocessor, camera_keys: list[str]) -> dict:
    """uint8 images -> float 0..1, then normalize and move to the device (as lerobot-train does)."""
    for key in camera_keys:
        if batch[key].dtype == torch.uint8:
            batch[key] = batch[key].float() / 255.0
    return preprocessor(batch)


def make_optimizer(policy: ACTPolicy) -> torch.optim.Optimizer:
    """ACT's optimizer preset: AdamW, with the backbone in its own parameter group."""
    return torch.optim.AdamW(
        policy.get_optim_params(), lr=policy.config.optimizer_lr, weight_decay=policy.config.optimizer_weight_decay
    )
