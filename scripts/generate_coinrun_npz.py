from __future__ import annotations

import argparse
from pathlib import Path

import gym
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, desc=None):
        return iterable if iterable is not None else _NoOpProgress()


class _NoOpProgress:
    def update(self, value: int) -> None:
        pass

    def close(self) -> None:
        pass


def make_env(num_envs: int):
    return gym.make(
        "procgen:procgen-coinrun-v0",
        num_envs=num_envs,
        distribution_mode="easy",
        use_backgrounds=False,
        restrict_themes=True,
    )


def unpack_reset(result):
    if isinstance(result, tuple):
        return result[0]
    return result


def unpack_step(result):
    if len(result) == 5:
        observations, rewards, terminated, truncated, infos = result
        return observations, rewards, np.logical_or(terminated, truncated), infos
    observations, rewards, dones, infos = result
    return observations, rewards, dones, infos


def rgb_from_observation(observation):
    if isinstance(observation, dict):
        return observation["rgb"]
    return observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/coinrun_20k.npz")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(args.num_envs)
    try:
        env.seed(args.seed)
    except AttributeError:
        pass
    observations = rgb_from_observation(unpack_reset(env.reset()))
    action_count = env.action_space.n

    frames = np.empty((args.steps, 64, 64, 3), dtype=np.uint8)
    actions = np.empty((args.steps,), dtype=np.int64)
    dones = np.empty((args.steps,), dtype=np.bool_)

    cursor = 0
    progress = tqdm(total=args.steps, desc="collecting")
    rng = np.random.default_rng(args.seed)

    while cursor < args.steps:
        action_batch = rng.integers(0, action_count, size=(args.num_envs,), dtype=np.int64)
        next_observations, _, done_batch, _ = unpack_step(env.step(action_batch))
        next_observations = rgb_from_observation(next_observations)

        for env_idx in range(args.num_envs):
            if cursor >= args.steps:
                break
            frames[cursor] = observations[env_idx]
            actions[cursor] = action_batch[env_idx]
            dones[cursor] = done_batch[env_idx]
            cursor += 1
            progress.update(1)

        observations = next_observations

    progress.close()
    env.close()
    np.savez_compressed(out, frames=frames, actions=actions, dones=dones, action_count=np.asarray(action_count))
    print(f"wrote {out} with {args.steps} transitions and {action_count} actions")


if __name__ == "__main__":
    main()
