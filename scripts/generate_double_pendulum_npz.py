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


ACTION_COUNT = 5
NOOP, KICK_LEFT, KICK_RIGHT, PUSH_IN, PUSH_OUT = range(ACTION_COUNT)


def line(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: np.ndarray, width: int = 1) -> None:
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        frame[max(0, y - width) : min(64, y + width + 1), max(0, x - width) : min(64, x + width + 1)] = color


def disk(frame: np.ndarray, cx: int, cy: int, radius: int, color: np.ndarray) -> None:
    yy, xx = np.ogrid[:64, :64]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    frame[mask] = color


def draw_frame(theta1: float, theta2: float, omega1: float, omega2: float, action: int) -> np.ndarray:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :, :] = np.array([8, 12, 20], dtype=np.uint8)

    origin = np.array([32.0, 13.0])
    l1 = 18.0
    l2 = 17.0
    p1 = origin + np.array([l1 * math.sin(theta1), l1 * math.cos(theta1)])
    p2 = p1 + np.array([l2 * math.sin(theta2), l2 * math.cos(theta2)])

    # Soft trace-like glow based on angular velocity.
    speed = min(1.0, (abs(omega1) + abs(omega2)) / 12.0)
    bg_boost = int(24 * speed)
    frame[:, :, 2] = np.clip(frame[:, :, 2] + bg_boost, 0, 255)

    if action != NOOP:
        colors = {
            KICK_LEFT: np.array([100, 165, 255], dtype=np.uint8),
            KICK_RIGHT: np.array([255, 140, 120], dtype=np.uint8),
            PUSH_IN: np.array([120, 255, 185], dtype=np.uint8),
            PUSH_OUT: np.array([245, 210, 90], dtype=np.uint8),
        }
        frame[2:6, 2:62] = colors[action]

    rod1 = np.array([110, 190, 255], dtype=np.uint8)
    rod2 = np.array([255, 205, 115], dtype=np.uint8)
    joint = np.array([245, 245, 250], dtype=np.uint8)
    bob1 = np.array([70, 145, 255], dtype=np.uint8)
    bob2 = np.array([255, 120, 90], dtype=np.uint8)

    x0, y0 = origin.astype(int)
    x1, y1 = np.round(p1).astype(int)
    x2, y2 = np.round(p2).astype(int)
    line(frame, x0, y0, x1, y1, rod1, width=1)
    line(frame, x1, y1, x2, y2, rod2, width=1)
    disk(frame, x0, y0, 2, joint)
    disk(frame, x1, y1, 3, bob1)
    disk(frame, x2, y2, 4, bob2)
    return frame


def step(theta1: float, theta2: float, omega1: float, omega2: float, action: int, dt: float) -> tuple[float, float, float, float]:
    # Standard equal-mass double pendulum dynamics, with small external action torques.
    g = 9.81
    m1 = m2 = 1.0
    l1 = l2 = 1.0
    delta = theta2 - theta1

    den1 = (m1 + m2) * l1 - m2 * l1 * math.cos(delta) * math.cos(delta)
    den2 = (l2 / l1) * den1
    alpha1 = (
        m2 * l1 * omega1 * omega1 * math.sin(delta) * math.cos(delta)
        + m2 * g * math.sin(theta2) * math.cos(delta)
        + m2 * l2 * omega2 * omega2 * math.sin(delta)
        - (m1 + m2) * g * math.sin(theta1)
    ) / den1
    alpha2 = (
        -m2 * l2 * omega2 * omega2 * math.sin(delta) * math.cos(delta)
        + (m1 + m2) * g * math.sin(theta1) * math.cos(delta)
        - (m1 + m2) * l1 * omega1 * omega1 * math.sin(delta)
        - (m1 + m2) * g * math.sin(theta2)
    ) / den2

    torque1 = 0.0
    torque2 = 0.0
    if action == KICK_LEFT:
        torque1 -= 4.0
    elif action == KICK_RIGHT:
        torque1 += 4.0
    elif action == PUSH_IN:
        torque2 -= 4.0
    elif action == PUSH_OUT:
        torque2 += 4.0

    omega1 = (omega1 + (alpha1 + torque1) * dt) * 0.999
    omega2 = (omega2 + (alpha2 + torque2) * dt) * 0.999
    theta1 += omega1 * dt
    theta2 += omega2 * dt
    return theta1, theta2, omega1, omega2


def reset_state(rng: np.random.Generator) -> tuple[float, float, float, float]:
    return (
        float(rng.uniform(-0.8, 0.8)),
        float(rng.uniform(0.6, 1.8)),
        float(rng.uniform(-0.15, 0.15)),
        float(rng.uniform(-0.15, 0.15)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/double_pendulum_50k.npz")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--episode-len", type=int, default=700)
    parser.add_argument("--dt", type=float, default=0.035)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    frames = np.empty((args.steps, 64, 64, 3), dtype=np.uint8)
    actions = np.empty((args.steps,), dtype=np.int64)
    dones = np.zeros((args.steps,), dtype=np.bool_)

    theta1, theta2, omega1, omega2 = reset_state(rng)
    for i in tqdm(range(args.steps), desc="double pendulum"):
        action = int(rng.choice([NOOP, KICK_LEFT, KICK_RIGHT, PUSH_IN, PUSH_OUT], p=[0.74, 0.065, 0.065, 0.065, 0.065]))
        frames[i] = draw_frame(theta1, theta2, omega1, omega2, action)
        actions[i] = action

        theta1, theta2, omega1, omega2 = step(theta1, theta2, omega1, omega2, action, args.dt)

        if i > 0 and i % args.episode_len == 0:
            dones[i] = True
            theta1, theta2, omega1, omega2 = reset_state(rng)

    np.savez_compressed(out, frames=frames, actions=actions, dones=dones, action_count=np.asarray(ACTION_COUNT))
    print(f"wrote {out} with {args.steps} frames")


if __name__ == "__main__":
    main()
