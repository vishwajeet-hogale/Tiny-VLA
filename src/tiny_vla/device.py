"""Pick the torch device. Defaults to cuda and refuses to silently fall back to cpu."""

import torch


def get_device(requested: str | None = None) -> torch.device:
    """Return the device to run on.

    With no argument this is cuda. If cuda isn't available we stop with an error instead of
    quietly running on cpu (which is ~100x slower for diffusion and easy to miss).
    Pass e.g. "cpu", "mps" or "cuda:1" (the --device flag) to choose explicitly.
    """
    name = requested or "cuda"

    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available, refusing to fall back to CPU.\n"
            f"  torch.__version__   = {torch.__version__}\n"
            f"  torch.version.cuda  = {torch.version.cuda}  (None means this is a CPU-only torch build)\n"
            "Check `nvidia-smi`, or pass --device cpu / --device mps to run elsewhere on purpose."
        )
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but torch.backends.mps.is_available() is False.")

    return torch.device(name)
