from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, desc=None):
        return iterable


ACTION_COUNT = 6
NOOP, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, DASH = range(ACTION_COUNT)


def rect(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    frame[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = np.asarray(color, dtype=np.uint8)


def disk(frame: np.ndarray, cx: float, cy: float, radius: float, color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    frame[mask] = np.asarray(color, dtype=np.uint8)


def ring(frame: np.ndarray, cx: float, cy: float, r0: float, r1: float, color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (r0 * r0 <= d2) & (d2 <= r1 * r1)
    frame[mask] = np.asarray(color, dtype=np.uint8)


def clamp_vec(pos: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(pos, lo, hi)


def choose_action(
    player: np.ndarray,
    gem: np.ndarray,
    enemies: np.ndarray,
    rng: np.random.Generator,
    policy: str = "chase",
) -> int:
    if policy == "random":
        return int(rng.integers(0, ACTION_COUNT))
    if policy == "mixed" and rng.random() < 0.45:
        return int(rng.integers(0, ACTION_COUNT))
    if policy == "event":
        roll = rng.random()
        if roll < 0.18:
            return int(rng.integers(0, ACTION_COUNT))
        if roll < 0.62:
            direction = gem - player
        else:
            direction = enemies[np.argmin(np.linalg.norm(enemies - player, axis=1))] - player
        if rng.random() < 0.18 and np.linalg.norm(direction) > 8.0:
            return DASH
        if abs(direction[0]) > abs(direction[1]):
            return MOVE_RIGHT if direction[0] > 0 else MOVE_LEFT
        return MOVE_DOWN if direction[1] > 0 else MOVE_UP

    if rng.random() < 0.12:
        return int(rng.integers(0, ACTION_COUNT))

    threat = player - enemies[np.argmin(np.linalg.norm(enemies - player, axis=1))]
    to_gem = gem - player
    direction = to_gem
    if np.linalg.norm(threat) < 22.0:
        direction = direction + 1.8 * threat

    if rng.random() < 0.08 and np.linalg.norm(direction) > 10.0:
        return DASH
    if abs(direction[0]) > abs(direction[1]):
        return MOVE_RIGHT if direction[0] > 0 else MOVE_LEFT
    return MOVE_DOWN if direction[1] > 0 else MOVE_UP


def draw_frame(
    size: int,
    player: np.ndarray,
    prev_player: np.ndarray,
    gem: np.ndarray,
    enemies: np.ndarray,
    action: int,
    score_flash: float,
) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :, :] = np.array([18, 24, 34], dtype=np.uint8)

    tile = max(8, size // 12)
    for y in range(0, size, tile):
        for x in range(0, size, tile):
            if (x // tile + y // tile) % 2 == 0:
                rect(frame, x, y, x + tile, y + tile, (22, 30, 43))

    margin = max(6, size // 18)
    rect(frame, 0, 0, size, margin, (70, 80, 102))
    rect(frame, 0, size - margin, size, size, (70, 80, 102))
    rect(frame, 0, 0, margin, size, (70, 80, 102))
    rect(frame, size - margin, 0, size, size, (70, 80, 102))

    action_colors = {
        NOOP: (95, 105, 120),
        MOVE_UP: (98, 179, 255),
        MOVE_DOWN: (111, 232, 173),
        MOVE_LEFT: (255, 202, 88),
        MOVE_RIGHT: (255, 128, 112),
        DASH: (213, 128, 255),
    }
    rect(frame, margin + 2, 3, size - margin - 2, margin - 2, action_colors[action])

    pulse = 1.0 + 0.18 * math.sin(score_flash)
    ring(frame, gem[0], gem[1], 5 * pulse, 8 * pulse, (255, 226, 92))
    disk(frame, gem[0], gem[1], 4 * pulse, (90, 255, 190))

    trail_steps = 4
    for i in range(trail_steps):
        t = (i + 1) / trail_steps
        p = prev_player * (1.0 - t) + player * t
        disk(frame, p[0], p[1], 5 + i, (34, 83, 150))

    for idx, enemy in enumerate(enemies):
        disk(frame, enemy[0], enemy[1], 7, (255, 86, 96))
        disk(frame, enemy[0] - 2, enemy[1] - 2, 2, (255, 190, 190))
        if idx == 0:
            ring(frame, enemy[0], enemy[1], 10, 11, (255, 144, 94))

    disk(frame, player[0], player[1], 8, (72, 145, 255))
    disk(frame, player[0] - 3, player[1] - 4, 3, (225, 242, 255))
    disk(frame, player[0] + 4, player[1] - 1, 2, (20, 35, 70))
    return frame


def random_pos(rng: np.random.Generator, size: int, margin: int) -> np.ndarray:
    return rng.uniform(margin, size - margin, size=2).astype(np.float32)


def nearby_pos(rng: np.random.Generator, center: np.ndarray, size: int, margin: int) -> np.ndarray:
    angle = rng.uniform(0, math.tau)
    radius = rng.uniform(16.0, 28.0)
    offset = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * radius
    return clamp_vec(center + offset, margin, size - margin).astype(np.float32)


def reset(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    margin = max(16, size // 8)
    player = random_pos(rng, size, margin)
    gem = random_pos(rng, size, margin)
    enemies = np.stack([random_pos(rng, size, margin) for _ in range(3)]).astype(np.float32)
    enemy_phase = rng.uniform(0, math.tau, size=3).astype(np.float32)
    enemy_radius = rng.uniform(size * 0.11, size * 0.22, size=3).astype(np.float32)
    return player, player.copy(), gem, enemies, np.stack([enemy_phase, enemy_radius], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/toy_arena_128_50k.npz")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--episode-len", type=int, default=600)
    parser.add_argument("--policy", choices=["chase", "mixed", "random", "event"], default="chase")
    parser.add_argument(
        "--event-respawn-rate",
        type=float,
        default=None,
        help="Chance to respawn a collected gem close to the player. Defaults to 0.70 for event policy, otherwise 0.",
    )
    args = parser.parse_args()

    if args.size < 64 or args.size % 4 != 0:
        raise ValueError("--size must be at least 64 and divisible by 4")

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    frames = np.empty((args.steps, args.size, args.size, 3), dtype=np.uint8)
    actions = np.empty((args.steps,), dtype=np.int64)
    dones = np.zeros((args.steps,), dtype=np.bool_)

    player, prev_player, gem, enemies, enemy_params = reset(rng, args.size)
    margin = max(14, args.size // 9)
    score_flash = 0.0
    event_respawn_rate = 0.70 if args.event_respawn_rate is None and args.policy == "event" else (args.event_respawn_rate or 0.0)
    coin_collects = 0
    enemy_contacts = 0

    for i in tqdm(range(args.steps), desc="toy arena"):
        action = choose_action(player, gem, enemies, rng, args.policy)
        frames[i] = draw_frame(args.size, player, prev_player, gem, enemies, action, score_flash)
        actions[i] = action

        prev_player = player.copy()
        velocity = np.array([0.0, 0.0], dtype=np.float32)
        if action == MOVE_UP:
            velocity[1] -= 2.3
        elif action == MOVE_DOWN:
            velocity[1] += 2.3
        elif action == MOVE_LEFT:
            velocity[0] -= 2.3
        elif action == MOVE_RIGHT:
            velocity[0] += 2.3
        elif action == DASH:
            direction = gem - player
            norm = np.linalg.norm(direction) + 1e-6
            velocity += direction / norm * 5.0
        player = clamp_vec(player + velocity, margin, args.size - margin)

        t = i * 0.045
        for j in range(len(enemies)):
            phase, radius = enemy_params[j]
            center = np.array(
                [
                    args.size * (0.33 + 0.18 * j),
                    args.size * (0.40 + 0.14 * ((j + 1) % 2)),
                ],
                dtype=np.float32,
            )
            enemies[j] = center + np.array([math.cos(t + phase), math.sin(t * 0.8 + phase)], dtype=np.float32) * radius

        if np.linalg.norm(player - gem) < 13.0:
            coin_collects += 1
            gem = nearby_pos(rng, player, args.size, margin) if rng.random() < event_respawn_rate else random_pos(rng, args.size, margin)
            score_flash = 0.0
        if np.min(np.linalg.norm(enemies - player, axis=1)) < 15.0:
            enemy_contacts += 1
        score_flash += 0.22

        if i > 0 and i % args.episode_len == 0:
            dones[i] = True
            player, prev_player, gem, enemies, enemy_params = reset(rng, args.size)

    np.savez_compressed(out, frames=frames, actions=actions, dones=dones, action_count=np.asarray(ACTION_COUNT))
    print(f"wrote {out} with {args.steps} frames at {args.size}x{args.size}")
    print(
        f"policy={args.policy} coin_collects={coin_collects} enemy_contacts={enemy_contacts} "
        f"event_respawn_rate={event_respawn_rate:.2f}"
    )


if __name__ == "__main__":
    main()
