# tiny-vla

A learning project for robot learning with [Hugging Face LeRobot](https://github.com/huggingface/lerobot),
built in small parts, starting with the PushT simulation environment.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # creates .venv with Python 3.12 and installs lerobot[pusht]
uv run python scripts/check_env.py
```

## Layout

| Path             | Purpose                                              |
|------------------|------------------------------------------------------|
| `src/tiny_vla/`  | Reusable code, importable as `tiny_vla`              |
| `scripts/`       | Small standalone scripts you run directly            |
| `notebooks/`     | Exploratory Jupyter notebooks                        |
| `configs/`       | Training and eval configs                            |
| `outputs/`       | Checkpoints, logs, videos (gitignored)               |
