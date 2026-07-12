from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pygame
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_coinrun_npz
from src.model import ActionConditionedNextFrame
from scripts.generate_toy_arena_npz import clamp_vec, draw_frame, random_pos, reset


NOOP = 0
MOVE_UP = 1
MOVE_DOWN = 2
MOVE_LEFT = 3
MOVE_RIGHT = 4
DASH = 5

TOY_ARENA_PALETTE = np.asarray(
    [
        (18, 24, 34),
        (22, 30, 43),
        (70, 80, 102),
        (95, 105, 120),
        (98, 179, 255),
        (111, 232, 173),
        (255, 202, 88),
        (255, 128, 112),
        (213, 128, 255),
        (255, 226, 92),
        (90, 255, 190),
        (34, 83, 150),
        (255, 86, 96),
        (255, 190, 190),
        (255, 144, 94),
        (72, 145, 255),
        (225, 242, 255),
        (20, 35, 70),
    ],
    dtype=np.float32,
)


class ToyArenaRuntime:
    def __init__(self, size: int, seed: int = 1001):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.episode_step = 0
        self.reset()

    def reset(self) -> None:
        self.player, self.prev_player, self.gem, self.enemies, self.enemy_params = reset(self.rng, self.size)
        self.margin = max(14, self.size // 9)
        self.score_flash = 0.0
        self.episode_step = 0

    def current_frame(self, action: int = NOOP) -> np.ndarray:
        return draw_frame(
            self.size,
            self.player,
            self.prev_player,
            self.gem,
            self.enemies,
            action,
            self.score_flash,
        )

    def step(self, action: int) -> np.ndarray:
        self.prev_player = self.player.copy()
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
            direction = self.gem - self.player
            norm = np.linalg.norm(direction) + 1e-6
            velocity += direction / norm * 5.0
        self.player = clamp_vec(self.player + velocity, self.margin, self.size - self.margin)

        t = self.episode_step * 0.045
        for j in range(len(self.enemies)):
            phase, radius = self.enemy_params[j]
            center = np.array(
                [
                    self.size * (0.33 + 0.18 * j),
                    self.size * (0.40 + 0.14 * ((j + 1) % 2)),
                ],
                dtype=np.float32,
            )
            self.enemies[j] = center + np.array(
                [math.cos(t + phase), math.sin(t * 0.8 + phase)],
                dtype=np.float32,
            ) * radius

        if np.linalg.norm(self.player - self.gem) < 13.0:
            self.gem = random_pos(self.rng, self.size, self.margin)
            self.score_flash = 0.0
        self.score_flash += 0.22
        self.episode_step += 1
        return self.current_frame(action)

    def warmup(self, context: int) -> deque[np.ndarray]:
        history: deque[np.ndarray] = deque(maxlen=context)
        for _ in range(context):
            history.append(self.step(NOOP))
        return history


def frames_to_tensor(history: deque[np.ndarray], device: str) -> torch.Tensor:
    frames = np.stack(list(history)).astype(np.float32) / 255.0
    frames = np.transpose(frames, (0, 3, 1, 2)).reshape(1, frames.shape[0] * 3, frames.shape[1], frames.shape[2])
    return torch.from_numpy(frames).to(device)


def choose_action(keys: pygame.key.ScancodeWrapper) -> int:
    if keys[pygame.K_SPACE]:
        return DASH
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        return MOVE_UP
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        return MOVE_DOWN
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        return MOVE_LEFT
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        return MOVE_RIGHT
    return NOOP


def make_surface(frame: np.ndarray, scale: int) -> pygame.Surface:
    surface = pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))
    if scale != 1:
        width, height = surface.get_size()
        surface = pygame.transform.scale(surface, (width * scale, height * scale))
    return surface


def palette_snap(frame: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return frame
    pixels = frame.astype(np.float32)
    diff = pixels[:, :, None, :] - TOY_ARENA_PALETTE[None, None, :, :]
    nearest = TOY_ARENA_PALETTE[np.argmin(np.sum(diff * diff, axis=-1), axis=-1)]
    snapped = pixels * (1.0 - amount) + nearest * amount
    return np.clip(snapped, 0, 255).astype(np.uint8)


def foreground_mask(frame: np.ndarray, threshold: float = 0.18) -> np.ndarray:
    pixels = frame.astype(np.float32) / 255.0
    saturation = pixels.max(axis=-1, keepdims=True) - pixels.min(axis=-1, keepdims=True)
    brightness = pixels.mean(axis=-1, keepdims=True)
    return ((saturation > threshold) | (brightness > 0.48)).astype(np.float32)


def stabilize_prediction(
    pred: np.ndarray,
    previous: np.ndarray,
    blend: float,
    sharpen: float,
    palette_snap_amount: float = 0.0,
    foreground_persist: float = 0.0,
    foreground_threshold: float = 0.18,
) -> np.ndarray:
    if blend > 0:
        pred = pred.astype(np.float32) * (1.0 - blend) + previous.astype(np.float32) * blend
    if foreground_persist > 0:
        prev_pixels = previous.astype(np.float32)
        pred_pixels = pred.astype(np.float32)
        prev_mask = foreground_mask(previous, foreground_threshold)
        pred_mask = foreground_mask(np.clip(pred_pixels, 0, 255).astype(np.uint8), foreground_threshold)
        prev_brightness = prev_pixels.mean(axis=-1, keepdims=True)
        pred_brightness = pred_pixels.mean(axis=-1, keepdims=True)
        sudden_fade = (pred_brightness < prev_brightness * 0.82).astype(np.float32)
        protect = prev_mask * (1.0 - pred_mask * 0.55) * sudden_fade
        strength = np.clip(foreground_persist, 0.0, 0.85) * protect
        pred = pred_pixels * (1.0 - strength) + prev_pixels * strength
    if sharpen > 0:
        center = pred.mean(axis=(0, 1), keepdims=True)
        pred = center + (pred - center) * (1.0 + sharpen)
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    return palette_snap(pred, palette_snap_amount)


def scheduled_stabilize(start: float, end: float, ramp_steps: int, step: int) -> float:
    if ramp_steps <= 0:
        return end
    t = max(0.0, min(step / float(ramp_steps), 1.0))
    return start * (1.0 - t) + end * t


def draw_label(screen: pygame.Surface, font: pygame.font.Font, text: str) -> None:
    label = font.render(text, True, (230, 236, 246))
    shadow = font.render(text, True, (12, 16, 24))
    screen.blit(shadow, (13, 13))
    screen.blit(label, (12, 12))


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a tiny action-conditioned world model in real time.")
    parser.add_argument("--data", default="data/toy_arena_smoke.npz")
    parser.add_argument("--checkpoint", default="runs/improved/best.pt")
    parser.add_argument("--start", type=int, default=32)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--closed-loop", action="store_true", help="Feed model predictions back into the model. Drifts quickly.")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--stabilize", type=float, default=0.20, help="Blend closed-loop predictions with the previous frame.")
    parser.add_argument("--stabilize-end", type=float, default=0.35, help="Optional final blend value for a closed-loop ramp.")
    parser.add_argument("--stabilize-ramp-steps", type=int, default=120, help="Frames used to ramp from --stabilize to --stabilize-end.")
    parser.add_argument("--history-sharpen", type=float, default=0.0, help="Light contrast boost before feeding predictions back.")
    parser.add_argument("--palette-snap", type=float, default=0.0, help="Blend predictions toward the toy arena color palette.")
    parser.add_argument("--foreground-persist", type=float, default=0.12, help="Keep small bright/saturated objects from vanishing in one closed-loop step.")
    parser.add_argument("--foreground-threshold", type=float, default=0.18)
    args = parser.parse_args()

    data_path = Path(args.data)
    checkpoint_path = Path(args.checkpoint)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Download the Colab checkpoint to this path or pass --checkpoint with the .pt file."
        )

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    arrays = load_coinrun_npz(str(data_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    context = int(checkpoint["context"])
    action_count = int(checkpoint["action_count"])
    model = ActionConditionedNextFrame(action_count=action_count, context=context).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if len(arrays.frames) <= context:
        raise ValueError("Dataset must contain more frames than the model context.")

    size = int(arrays.frames.shape[1])
    runtime = ToyArenaRuntime(size=size, seed=args.seed)
    history = runtime.warmup(context)
    current = history[-1].copy()
    closed_loop = bool(args.closed_loop)
    closed_loop_steps = 0
    show_engine = False

    pygame.init()
    pygame.display.set_caption("MiniMIRA playable world model")
    height, width = current.shape[:2]
    screen = pygame.display.set_mode((width * args.scale, height * args.scale))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    running = True
    with torch.no_grad():
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        runtime.reset()
                        history = runtime.warmup(context)
                        current = history[-1].copy()
                        closed_loop_steps = 0
                    elif event.key == pygame.K_e:
                        show_engine = not show_engine
                    elif event.key == pygame.K_m:
                        closed_loop = not closed_loop
                        closed_loop_steps = 0

            action = choose_action(pygame.key.get_pressed())
            if show_engine:
                current = runtime.step(action)
                history.append(current)
            else:
                pred = model(frames_to_tensor(history, device), torch.tensor([action], dtype=torch.long, device=device))
                raw_current = np.clip(pred[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
                if closed_loop:
                    blend = scheduled_stabilize(
                        args.stabilize,
                        args.stabilize if args.stabilize_end is None else args.stabilize_end,
                        args.stabilize_ramp_steps,
                        closed_loop_steps,
                    )
                    current = stabilize_prediction(
                        raw_current,
                        history[-1],
                        blend=max(0.0, min(blend, 0.85)),
                        sharpen=max(0.0, min(args.history_sharpen, 1.0)),
                        palette_snap_amount=max(0.0, min(args.palette_snap, 1.0)),
                        foreground_persist=max(0.0, min(args.foreground_persist, 0.85)),
                        foreground_threshold=max(0.0, min(args.foreground_threshold, 1.0)),
                    )
                    history.append(current)
                    closed_loop_steps += 1
                else:
                    current = raw_current
                    history.append(runtime.step(action))
                    closed_loop_steps = 0

            screen.blit(make_surface(current, args.scale), (0, 0))
            mode = "engine" if show_engine else ("model closed-loop" if closed_loop else "model assisted")
            draw_label(screen, font, f"{mode} | WASD/arrows | Space dash | R reset | E engine | M closed-loop | Esc")
            pygame.display.flip()
            clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
