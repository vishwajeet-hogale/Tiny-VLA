"""Part 3, step 1: print ACT's default config (from the installed lerobot) and its parameter count.

    python scripts/print_act_config.py
"""

from tiny_vla.act import build_act
from tiny_vla.device import get_device

policy, meta = build_act(get_device())
config = policy.config

fields = [
    "chunk_size",
    "n_action_steps",
    "n_obs_steps",
    "kl_weight",
    "use_vae",
    "latent_dim",
    "vision_backbone",
    "pretrained_backbone_weights",
    "dim_model",
    "n_heads",
    "dim_feedforward",
    "n_encoder_layers",
    "n_decoder_layers",
    "n_vae_encoder_layers",
    "dropout",
    "temporal_ensemble_coeff",
    "optimizer_lr",
    "optimizer_lr_backbone",
]
print("=== Default ACTConfig (lerobot 0.6.1) ===")
for name in fields:
    print(f"{name:<28} {getattr(config, name)}")
print()
print("input features: ", {k: v.shape for k, v in config.input_features.items()})
print("output features:", {k: v.shape for k, v in config.output_features.items()})
print()


def count(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


model = policy.model
parts = {
    "backbone (ResNet18)": model.backbone,
    "CVAE encoder": model.vae_encoder,
    "transformer encoder": model.encoder,
    "transformer decoder": model.decoder,
}
total = count(policy)
print(f"trainable parameters: {total:,} ({total / 1e6:.1f}M)")
for name, module in parts.items():
    n = count(module)
    print(f"  {name:<22} {n:>12,}  ({n / total:.0%})")
other = total - sum(count(m) for m in parts.values())
print(f"  {'other (projections, embeddings, action head)':<22} {other:>12,}  ({other / total:.0%})")
