from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, desc=None):
        return iterable


ACTION_COUNT = 4
NOOP, LEFT, RIGHT, JUMP = range(ACTION_COUNT)


def draw_frame(player_x: float, player_y: float, coin_x: float, obstacle_x: float) -> np.ndarray:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :, :] = np.array([135, 195, 235], dtype=np.uint8)
    frame[52:, :, :] = np.array([54, 130, 72], dtype=np.uint8)
    frame[55:, :, :] = np.array([84, 74, 54], dtype=np.uint8)

    ox = int(round(obstacle_x))
    if -5 < ox < 64:
        frame[44:55, max(0, ox - 3) : min(64, ox + 4), :] = np.array([190, 60, 45], dtype=np.uint8)

    cx = int(round(coin_x))
    if 2 <= cx < 62:
        yy, xx = np.ogrid[:64, :64]
        mask = (xx - cx) ** 2 + (yy - 33) ** 2 <= 16
        frame[mask] = np.array([245, 205, 55], dtype=np.uint8)

    px = int(round(player_x))
    py = int(round(player_y))
    frame[max(0, py - 7) : min(64, py + 1), max(0, px - 4) : min(64, px + 5), :] = np.array([50, 80, 210], dtype=np.uint8)
    frame[max(0, py - 11) : max(0, py - 7), max(0, px - 3) : min(64, px + 4), :] = np.array([235, 215, 180], dtype=np.uint8)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/toy_runner_20k.npz")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    frames = np.empty((args.steps, 64, 64, 3), dtype=np.uint8)
    actions = np.empty((args.steps,), dtype=np.int64)
    dones = np.zeros((args.steps,), dtype=np.bool_)

    player_x = 16.0
    player_y = 52.0
    vy = 0.0
    coin_x = 60.0
    obstacle_x = 42.0

    for i in tqdm(range(args.steps), desc="toy runner"):
        action = int(rng.choice([NOOP, LEFT, RIGHT, JUMP], p=[0.25, 0.2, 0.35, 0.2]))
        actions[i] = action
        frames[i] = draw_frame(player_x, player_y, coin_x, obstacle_x)

        if action == LEFT:
            player_x -= 1.2
        elif action == RIGHT:
            player_x += 1.5
        elif action == JUMP and player_y >= 52.0:
            vy = -4.2

        vy += 0.35
        player_y += vy
        if player_y > 52.0:
            player_y = 52.0
            vy = 0.0

        player_x = float(np.clip(player_x, 6.0, 58.0))
        obstacle_x -= 0.55
        coin_x -= 0.55
        if obstacle_x < -8:
            obstacle_x = 70.0 + float(rng.integers(0, 20))
        if coin_x < -8:
            coin_x = 75.0 + float(rng.integers(0, 25))

        if i > 0 and i % 500 == 0:
            dones[i] = True
            player_x = 16.0
            player_y = 52.0
            vy = 0.0
            coin_x = 60.0
            obstacle_x = 42.0

    np.savez_compressed(out, frames=frames, actions=actions, dones=dones, action_count=np.asarray(ACTION_COUNT))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
