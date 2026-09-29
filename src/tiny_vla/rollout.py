"""A hand-written rollout loop for PushT policies (the pretrained diffusion policy, and ACT).

This is a single-env version of `rollout()` in lerobot/scripts/lerobot_eval.py, written out
step by step so every conversion between the simulator and the policy is visible.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch

import gym_pusht  # noqa: F401  (registers "gym_pusht/PushT-v0" with gymnasium)
from lerobot.envs.configs import PushtEnv


def make_env() -> gym.Env:
    """Create PushT with exactly the settings lerobot-eval uses (lerobot/envs/configs.py: PushtEnv).

    obs_type="pixels_agent_pos" -> observations are a 96x96 RGB image plus the agent's (x, y).
    max_episode_steps=300       -> gymnasium's TimeLimit truncates the episode after 300 steps.
    """
    cfg = PushtEnv()
    return gym.make(cfg.gym_id, disable_env_checker=cfg.disable_env_checker, **cfg.gym_kwargs)


def sim_obs_to_policy_obs(obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Map simulator observation keys/formats to the ones the policy was trained on.

    The sim and the dataset describe the same things with different names and layouts:
      sim "pixels"    (96, 96, 3) uint8 0..255, channels last
        -> "observation.image" (1, 3, 96, 96) float32 0..1, channels first
      sim "agent_pos" (2,) float64
        -> "observation.state" (1, 2) float32
    The leading 1 is the batch dimension: the policy always works on batches, here of size 1.
    This mirrors lerobot.envs.utils.preprocess_observation().
    """
    image = torch.from_numpy(obs["pixels"])  # (H, W, C) uint8
    image = image.permute(2, 0, 1)  # -> (C, H, W): torch conv layers want channels first
    image = image.float() / 255.0  # -> 0..1 floats, the range the dataset images were stored in
    state = torch.from_numpy(obs["agent_pos"]).float()
    return {
        "observation.image": image.unsqueeze(0),
        "observation.state": state.unsqueeze(0),
    }


@dataclass
class EpisodeResult:
    seed: int
    success: bool  # did coverage exceed 95% before the step limit?
    steps: int  # env steps taken
    max_coverage: float  # best fraction of the goal covered by the T block (0..1)
    max_reward: float  # best reward; PushT reward is min(coverage / 0.95, 1)
    sum_reward: float
    chunk_predictions: int  # times the policy ran the model to predict a new action chunk
    denoise_passes: int | None  # diffusion only: U-Net passes (num_inference_steps per chunk)
    wall_time_s: float
    peak_vram_mb: float | None  # None when not on cuda
    frames: list[np.ndarray] = field(default_factory=list, repr=False)  # filled when render=True


class _CallCounter:
    """Counts how many times an nn.Module's forward() runs, using a forward hook."""

    def __init__(self, module: torch.nn.Module):
        self.count = 0
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.count += 1

    def remove(self):
        self._handle.remove()


def _instrumented_modules(policy) -> tuple[torch.nn.Module, torch.nn.Module | None]:
    """(module that runs once per chunk prediction, module that runs once per denoising step).

    Diffusion: the image encoder runs once per chunk (on the 2 stacked frames) and the U-Net
               runs num_inference_steps times per chunk.
    ACT:       the whole ACT model runs once per chunk; there is no denoising loop.
    """
    if policy.name == "diffusion":
        return policy.diffusion.rgb_encoder, policy.diffusion.unet
    if policy.name == "act":
        return policy.model, None
    raise NotImplementedError(f"No instrumentation for policy type {policy.name!r}")


def _action_queue(policy):
    """The policy's private queue of not-yet-executed actions (each normalized, shape (batch, 2))."""
    if policy.name == "diffusion":
        return policy._queues["action"]  # lerobot/policies/diffusion/modeling_diffusion.py
    return policy._action_queue  # lerobot/policies/act/modeling_act.py


def planned_actions(policy, postprocessor) -> np.ndarray:
    """The actions still waiting in the policy's queue, in env coordinates, shape (n, 2).

    Reads a private attribute (see _action_queue). Queued actions are still normalized, so they
    go through the postprocessor (the same unnormalize step every executed action gets).
    """
    queue = _action_queue(policy)
    if len(queue) == 0:
        return np.zeros((0, 2))
    chunk = torch.stack(list(queue))  # (n, batch=1, 2)
    return postprocessor(chunk[:, 0]).numpy()  # (n, 2)


def run_episode(
    policy,
    env: gym.Env,
    preprocessor,
    postprocessor,
    seed: int,
    render: bool = False,
    on_step: Callable[[np.ndarray, dict], bool] | None = None,
) -> EpisodeResult:
    """Run one PushT episode with `policy` and return what happened.

    render:  if True, render a frame after reset and after every step. Frames are passed to
             `on_step(frame, step_info)` if given, otherwise stored in result.frames.
    on_step: optional callback, called after reset and after every step when render=True.
             Return True to stop the episode early (used by the viewer's q/r keys).
    """
    device = next(policy.parameters()).device
    on_cuda = device.type == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # Count model work with forward hooks, so the lerobot code doesn't need to be modified
    # (which modules are hooked depends on the policy type, see _instrumented_modules).
    chunk_module, denoise_module = _instrumented_modules(policy)
    chunk_counter = _CallCounter(chunk_module)
    denoise_counter = _CallCounter(denoise_module) if denoise_module is not None else None

    # Seed torch too: diffusion starts every chunk from random noise, so without this the same
    # env seed could give a different episode each run.
    torch.manual_seed(seed)

    # 1. Reset. Clears the policy's observation history and action queue from the last episode,
    #    and puts the env in the start state determined by `seed` (block pose + agent position).
    policy.reset()
    obs, info = env.reset(seed=seed)

    frames = []
    coverages, rewards = [0.0], [0.0]
    success = False
    steps = 0

    def emit_frame(chunk=None) -> bool:
        frame = env.render()
        step_info = {"step": steps, "reward": rewards[-1], "coverage": coverages[-1], "chunk": chunk}
        if on_step is not None:
            return bool(on_step(frame, step_info))
        frames.append(frame)
        return False

    stop = render and emit_frame()
    start = time.perf_counter()

    while not stop:
        # 2. Map sim observation keys and formats to the policy's (see sim_obs_to_policy_obs).
        batch = sim_obs_to_policy_obs(obs)

        # 3. Preprocess with the pipeline saved alongside the checkpoint:
        #    rename keys (none here) -> add batch dim (already there) -> move to GPU -> normalize.
        #    Normalizing maps state to [-1, 1] with the dataset min/max and standardizes the
        #    image with the dataset mean/std, so the model sees inputs just like in training.
        batch = preprocessor(batch)

        # 4. Choose an action. When the policy's action queue is empty it runs the model to predict
        #    a chunk of future actions and queues the first n_action_steps of them; otherwise it just
        #    pops the next queued action with no model call. (Diffusion also keeps the last 2
        #    observations and runs 100 denoising passes per chunk; ACT uses 1 observation and one
        #    forward pass.) inference_mode skips autograd bookkeeping.
        with torch.inference_mode():
            action = policy.select_action(batch)  # (1, 2), normalized

        # 5. Postprocess: undo the action normalization ([-1, 1] -> 0..512 pixel coords) and
        #    move back to the CPU.
        action = postprocessor(action)
        action_np = action[0].numpy()  # (2,): the env takes one unbatched (x, y) target

        # 6. Step the sim. The action is a target position; the env's PD controller pulls the
        #    agent toward it for 0.1 s. reward = min(coverage / 0.95, 1); terminated means the
        #    block covers >95% of the goal; truncated means the 300-step limit was hit.
        obs, reward, terminated, truncated, info = env.step(action_np)
        steps += 1
        rewards.append(float(reward))
        coverages.append(float(info["coverage"]))
        success = success or bool(info["is_success"])

        if render:
            chunk = np.concatenate([action_np[None], planned_actions(policy, postprocessor)])
            stop = emit_frame(chunk)
        if terminated or truncated:
            break

    wall_time = time.perf_counter() - start
    chunk_counter.remove()
    if denoise_counter is not None:
        denoise_counter.remove()

    return EpisodeResult(
        seed=seed,
        success=success,
        steps=steps,
        max_coverage=max(coverages),
        max_reward=max(rewards),
        sum_reward=sum(rewards),
        chunk_predictions=chunk_counter.count,
        denoise_passes=denoise_counter.count if denoise_counter is not None else None,
        wall_time_s=wall_time,
        peak_vram_mb=torch.cuda.max_memory_allocated(device) / 2**20 if on_cuda else None,
        frames=frames,
    )
