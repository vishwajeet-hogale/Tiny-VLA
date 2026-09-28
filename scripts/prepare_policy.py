"""Convert the Hub checkpoint lerobot/diffusion_pusht into the format lerobot 0.6.1 expects.

Why this is needed: the checkpoint was saved by an older lerobot that kept normalization
stats *inside* the model weights (keys like "normalize_inputs.buffer_observation_state.min").
lerobot 0.6.1 moved normalization out of the model into separate pre/post processor pipelines
saved next to it (policy_preprocessor.json, policy_postprocessor.json). The Hub repo has no
processor files, and loading the old weights silently drops those stats keys.

This does what lerobot's own migrate_policy_normalization.py does (that script crashes in
0.6.1 while saving the config: it keeps crop_shape as a list and draccus can't encode it).

Run once:  python scripts/prepare_policy.py
Output:    outputs/models/diffusion_pusht/  (model + config + pre/post processors)
"""

from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

HUB_ID = "lerobot/diffusion_pusht"
OUT_DIR = Path("outputs/models/diffusion_pusht")

# The old normalization buffers are named "<prefix><feature with _ instead of .>.<stat>",
# e.g. "normalize_inputs.buffer_observation_state.min" -> stats["observation.state"]["min"].
STAT_PREFIXES = ["normalize_inputs.buffer_", "normalize_targets.buffer_", "unnormalize_outputs.buffer_"]

# 1. Read the raw weights file and pull out the normalization stats.
raw_weights = load_file(hf_hub_download(HUB_ID, "model.safetensors"))
stats: dict[str, dict] = {}
for key, tensor in raw_weights.items():
    for prefix in STAT_PREFIXES:
        if key.startswith(prefix):
            feature, stat = key[len(prefix) :].rsplit(".", 1)
            stats.setdefault(feature.replace("_", "."), {})[stat] = tensor
            break

for feature, feature_stats in stats.items():
    print(f"{feature:<20} {sorted(feature_stats)}")

# 2. Load the model. This parses config.json properly (crop_shape becomes a tuple) and loads every
#    weight except the old normalization buffers, which it reports as "unexpected keys".
policy = DiffusionPolicy.from_pretrained(HUB_ID)

# 3. Build fresh processors for this policy config, filled with the stats from step 1.
preprocessor, postprocessor = make_pre_post_processors(policy.config, dataset_stats=stats)

# 4. Save everything into one folder that make_pre_post_processors(pretrained_path=...) can load.
OUT_DIR.mkdir(parents=True, exist_ok=True)
policy.save_pretrained(OUT_DIR)
preprocessor.save_pretrained(OUT_DIR)
postprocessor.save_pretrained(OUT_DIR)
print(f"\nSaved to {OUT_DIR}:")
for path in sorted(OUT_DIR.iterdir()):
    print(f"  {path.name}")
