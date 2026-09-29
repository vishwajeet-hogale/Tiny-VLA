# tiny-vla

## Goal
A learning project to understand robot learning with Hugging Face LeRobot, built in small parts
(starting with the PushT sim). The owner wants to understand every line: favor small, readable
scripts over clever abstractions, and explain non-obvious code.

## Pinned versions
- Python 3.12 managed with `uv` (`uv sync`, `uv run python ...`)
- `lerobot[pusht,diffusion,training]==0.6.1`, pinned exactly in `pyproject.toml`
- A uv override in `pyproject.toml` keeps a single OpenCV (the GUI build, 4.13.0.92); don't remove it

## Rule: check the installed source, don't assume
Always check the installed lerobot source for APIs rather than assuming from memory, since the
API changes between versions. The installed source is under
`.venv/lib/python3.12/site-packages/lerobot/`.

## Hardware
- Mac M3, 16GB unified memory (device: `mps`)
- Separate Linux box with a 4GB NVIDIA GPU (RTX 3050 Laptop, device: `cuda`); VRAM is tight, so keep batch sizes small
  - It runs as WSL2 with only ~3.5GB of system RAM. `lerobot-eval` must use `--eval.use_async_envs=false`
    (one process per env runs out of RAM) and a small `--eval.batch_size` like 2 (it buffers every
    rendered video frame in RAM)
  - torchcodec can't load here; use `video_backend="pyav"` for datasets
- Heavy training goes to Google Colab; local machines are for development, debugging, and small runs

Code should pick the device at runtime (cuda → mps → cpu), never hardcode it.

## Layout
- `src/tiny_vla/`: reusable code (importable as `tiny_vla`)
- `scripts/`: standalone runnable scripts
- `notebooks/`: exploration
- `configs/`: training/eval configs
- `outputs/`: checkpoints, logs, videos (gitignored)
