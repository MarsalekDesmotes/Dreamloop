from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_v2 import dataset_manifest_hash
from src.toy_arena_v2 import (
    ACTION_COUNT,
    COLLISION_RADIUS,
    COLLISION_COOLDOWN,
    EVENT_EPISODE_END,
    EVENT_LOSE,
    EVENT_WIN,
    ENEMY_SPEED,
    GOAL_COINS,
    MAX_HEALTH,
    POLICY_NAMES,
    ToyArenaV2,
    choose_policy_action,
)


ARRAY_SPECS = {
    "frames": (np.uint8, (128, 128, 3)),
    "actions": (np.uint8, ()),
    "dones": (np.bool_, ()),
    "events": (np.uint8, ()),
    "episode_ids": (np.int32, ()),
    "player_pos": (np.float32, (2,)),
    "player_vel": (np.float32, (2,)),
    "coin_pos": (np.float32, (2,)),
    "coin_pad": (np.uint8, ()),
    "enemy_pos": (np.float32, (3, 2)),
    "enemy_vel": (np.float32, (3, 2)),
    "score": (np.int16, ()),
    "collision_cooldown": (np.uint8, ()),
    "health": (np.uint8, ()),
    "portal_unlocked": (np.bool_, ()),
    "game_status": (np.int8, ()),
}


def episode_policy_ids(episodes: int, seed: int) -> np.ndarray:
    fractions = np.asarray((0.35, 0.25, 0.20, 0.20), dtype=np.float64)
    counts = np.floor(fractions * episodes).astype(int)
    counts[0] += episodes - int(counts.sum())
    policies = np.concatenate([np.full(count, policy, dtype=np.uint8) for policy, count in enumerate(counts)])
    rng = np.random.default_rng(seed)
    rng.shuffle(policies)
    return policies


def episode_splits(episodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(episodes)
    train_count = int(episodes * 0.80)
    val_count = int(episodes * 0.10)
    splits = np.full(episodes, 2, dtype=np.uint8)
    splits[order[:train_count]] = 0
    splits[order[train_count : train_count + val_count]] = 1
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic Toy Arena V2 memmap data.")
    parser.add_argument("--out", default="data/toy_arena_v2_60k")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--terminal-hold", type=int, default=18)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.episodes < 10 or args.episode_length < 64:
        raise ValueError("use at least 10 episodes and 64 frames per episode")
    if args.size != 128:
        raise ValueError("V2 dataset schema is fixed at 128x128")
    if args.terminal_hold < 1:
        raise ValueError("terminal hold must be at least one frame")

    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    frame_count = args.episodes * args.episode_length
    specs = dict(ARRAY_SPECS)
    specs["frames"] = (np.uint8, (args.size, args.size, 3))
    arrays = {
        name: open_memmap(out / f"{name}.npy", mode="w+", dtype=dtype, shape=(frame_count, *tail))
        for name, (dtype, tail) in specs.items()
    }
    seeds = np.arange(args.episodes, dtype=np.uint32) + np.uint32(args.seed * 10_000)
    policies = episode_policy_ids(args.episodes, args.seed + 1)
    splits = episode_splits(args.episodes, args.seed + 2)
    np.save(out / "episode_seeds.npy", seeds)
    np.save(out / "episode_policies.npy", policies)
    np.save(out / "episode_splits.npy", splits)

    event_counts = {"coin": 0, "collision": 0, "dash": 0, "win": 0, "lose": 0}
    for episode_id in tqdm(range(args.episodes), desc="toy arena v2 episodes"):
        env = ToyArenaV2(seed=int(seeds[episode_id]), size=args.size)
        policy_rng = np.random.default_rng(int(seeds[episode_id]) ^ 0xA5A5A5A5)
        segment_id = 0
        terminal_frames = 0
        for local_step in range(args.episode_length):
            index = episode_id * args.episode_length + local_step
            snapshot = env.snapshot()
            arrays["frames"][index] = env.render()
            arrays["episode_ids"][index] = episode_id
            for name in (
                "player_pos",
                "player_vel",
                "coin_pos",
                "coin_pad",
                "enemy_pos",
                "enemy_vel",
                "score",
                "collision_cooldown",
                "health",
                "portal_unlocked",
                "game_status",
            ):
                arrays[name][index] = snapshot[name]

            is_last = local_step == args.episode_length - 1
            is_terminal = int(snapshot["game_status"]) != 0
            terminal_frames = terminal_frames + 1 if is_terminal else 0
            reset_after_frame = is_terminal and terminal_frames >= args.terminal_hold and not is_last

            if reset_after_frame:
                action = 0
                event = EVENT_EPISODE_END
            elif is_terminal:
                action = 0
                _, event = env.step(action)
            else:
                action = choose_policy_action(env, policy_rng, int(policies[episode_id]), local_step)
                _, event = env.step(action)
            arrays["actions"][index] = action
            if is_last and not (event & EVENT_EPISODE_END):
                event |= EVENT_EPISODE_END
            arrays["events"][index] = event
            arrays["dones"][index] = is_last or reset_after_frame
            event_counts["coin"] += int(bool(event & 1))
            event_counts["collision"] += int(bool(event & 2))
            event_counts["dash"] += int(bool(event & 4))
            event_counts["win"] += int(bool(event & EVENT_WIN))
            event_counts["lose"] += int(bool(event & EVENT_LOSE))
            if reset_after_frame:
                segment_id += 1
                reset_seed = int(seeds[episode_id]) + segment_id * 1_000_003
                env.reset(seed=reset_seed)
                terminal_frames = 0

    for array in arrays.values():
        array.flush()

    metadata = {
        "version": 12,
        "frame_count": frame_count,
        "episodes": args.episodes,
        "episode_length": args.episode_length,
        "size": args.size,
        "fps": args.fps,
        "action_count": ACTION_COUNT,
        "seed": args.seed,
        "split_names": ["train", "val", "test"],
        "policy_names": list(POLICY_NAMES),
        "event_bits": {
            "coin": 1,
            "collision": 2,
            "dash": 4,
            "episode_end": 8,
            "win": EVENT_WIN,
            "lose": EVENT_LOSE,
        },
        "event_counts": event_counts,
        "transition_contract": "frame[t] + action[t] -> frame[t+1] within an episode",
        "render_contract": "escape_protocol_v6_unobscured_terminal_scene",
        "dynamics_contract": "v12_observable_closing_contact_cooldown36",
        "max_health": MAX_HEALTH,
        "goal_coins": GOAL_COINS,
        "collision_radius": COLLISION_RADIUS,
        "collision_cooldown": COLLISION_COOLDOWN,
        "collision_rule": "damage while closing or within 15px core",
        "enemy_speed": ENEMY_SPEED,
        "terminal_hold": args.terminal_hold,
        "reset_contract": "done[t] marks that frame[t+1] starts a new deterministic round",
        "terminal_action_contract": "terminal hold frames use NOOP until deterministic reset",
    }
    metadata["manifest_hash"] = dataset_manifest_hash(metadata)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
