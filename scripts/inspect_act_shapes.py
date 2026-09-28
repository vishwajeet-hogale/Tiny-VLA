"""Part 3, step 2: trace tensor shapes through one ACT training forward pass.

Loads one batch of lerobot/pusht with ACT's delta_timestamps, registers forward hooks on
ACT's main submodules, runs policy.forward (the training pass that computes the loss), and
prints each module's input and output shapes in the order the modules actually ran.

Transformer modules in ACT use (sequence, batch, dim) layout, like torch.nn.MultiheadAttention's
default, so their shapes are listed as (S, B, D).

    python scripts/inspect_act_shapes.py              # default config: chunk_size=100
    python scripts/inspect_act_shapes.py --chunk-size 20
Writes outputs/part3/act_shapes.txt.
"""

import argparse
from pathlib import Path

import torch

from lerobot.datasets.factory import resolve_delta_timestamps
from tiny_vla.act import build_act, make_act_dataset, make_act_preprocessor, prepare_batch
from tiny_vla.device import get_device

OUT_FILE = Path("outputs/part3/act_shapes.txt")

parser = argparse.ArgumentParser()
parser.add_argument("--device", default=None)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--chunk-size", type=int, default=None, help="default: ACTConfig's own default")
args = parser.parse_args()

overrides = {} if args.chunk_size is None else {"chunk_size": args.chunk_size, "n_action_steps": args.chunk_size}
policy, meta = build_act(get_device(args.device), **overrides)
dataset = make_act_dataset(policy, meta)
preprocessor = make_act_preprocessor(policy, meta)

lines: list[str] = []


def log(text: str = ""):
    print(text)
    lines.append(text)


def shape_of(x) -> str:
    """Describe a tensor, or the tensors inside a tuple/list/dict, by shape."""
    if isinstance(x, torch.Tensor):
        return str(tuple(x.shape))
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {shape_of(v)}" for k, v in x.items()) + "}"
    if isinstance(x, (tuple, list)):
        return "(" + ", ".join(shape_of(v) for v in x) + ")"
    return type(x).__name__


# (module, label, what it does). Hooks fire in execution order, not the order listed here.
model = policy.model
watched = [
    (model.vae_encoder, "CVAE encoder", "reads [CLS, state, action chunk] and summarizes them in the CLS token"),
    (model.vae_encoder_latent_output_proj, "CVAE -> latent params", "CLS token -> [mu, log(sigma^2)], each latent_dim"),
    (model.encoder_latent_input_proj, "latent z -> token", "project the sampled z up to one dim_model token"),
    (model.backbone, "backbone (ResNet18)", "image -> 512-channel feature map, 32x downsampled"),
    (model.encoder_img_feat_input_proj, "image feature proj", "1x1 conv: 512 backbone channels -> dim_model"),
    (model.encoder, "transformer encoder", "self-attention over [z, state, image patches] tokens"),
    (model.decoder, "transformer decoder", "chunk_size queries cross-attend to the encoder output"),
    (model.action_head, "action head", "linear: dim_model -> action_dim for every chunk step"),
]
order = 0


def make_hook(label: str, description: str):
    def hook(module, inputs, output):
        global order
        order += 1
        log(f"{order}. {label}  ({description})")
        log(f"     in : {shape_of(inputs)}")
        log(f"     out: {shape_of(output)}")

    return hook


handles = [m.register_forward_hook(make_hook(label, desc)) for m, label, desc in watched]

loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
batch = next(iter(loader))

log(f"ACT chunk_size={policy.config.chunk_size}, dim_model={policy.config.dim_model}, "
    f"latent_dim={policy.config.latent_dim}, batch_size={args.batch_size}")
action_ts = resolve_delta_timestamps(policy.config, meta)["action"]  # the same call make_act_dataset uses
log(f"delta_timestamps['action'] = [{action_ts[0]:.1f}, {action_ts[1]:.1f}, ..., {action_ts[-1]:.1f}]  "
    f"({len(action_ts)} values, seconds from the current frame)")
log()
log("=== Batch fed to policy.forward (after preprocessing) ===")
batch = prepare_batch(batch, preprocessor, meta.camera_keys)
for key in ["observation.image", "observation.state", "action", "action_is_pad"]:
    log(f"  {key:<20} {shape_of(batch[key])}  {batch[key].dtype}")
log()
log("=== Forward pass, in execution order ===")

policy.train()  # training mode: the CVAE encoder only runs when training with actions in the batch
loss, loss_dict = policy.forward(batch)

log()
log(f"=== Loss ===\n  loss = l1 + kl_weight * kld = {loss.item():.4f}   {loss_dict}")

for h in handles:
    h.remove()
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUT_FILE.write_text("\n".join(lines) + "\n")
print(f"\nSaved {OUT_FILE}")
