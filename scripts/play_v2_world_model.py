from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pygame
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_v2_world_model import initialize_context
from src.runtime_v2 import LatentWorldRuntime, RGBWorldRuntime, SemanticWorldRuntime
from src.toy_arena_v2 import DASH, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, NOOP, ToyArenaV2


ACTION_NAMES = ("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "DASH")


def keyboard_action(keys) -> int:
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


def surface(frame: np.ndarray, scale: int) -> pygame.Surface:
    value = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    return pygame.transform.scale(value, (frame.shape[1] * scale, frame.shape[0] * scale))


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the pure streaming Toy Arena V2 world model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", choices=("latent", "rgb", "semantic"), default="latent")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--collision-threshold", type=float, default=None)
    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    runtime_class = {
        "latent": LatentWorldRuntime,
        "rgb": RGBWorldRuntime,
        "semantic": SemanticWorldRuntime,
    }[args.model_type]
    runtime = (
        runtime_class(args.checkpoint, device, collision_threshold=args.collision_threshold)
        if args.model_type == "semantic"
        else runtime_class(args.checkpoint, device)
    )
    seed = args.seed

    def reset():
        engine = ToyArenaV2(seed=seed)
        frames, actions = initialize_context(engine, runtime.context)
        runtime.initialize(frames, actions)
        return engine, frames[-1]

    engine, current = reset()
    assisted = False
    pygame.init()
    pygame.display.set_caption("Dreamloop - Toy Arena")
    width = current.shape[1] * args.scale
    height = current.shape[0] * args.scale
    ui_height = 34
    screen = pygame.display.set_mode((width, height + ui_height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    running = True
    frame_count = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    seed += 1
                    engine, current = reset()
                elif event.key == pygame.K_e:
                    assisted = not assisted
                    if not assisted:
                        frames, actions = initialize_context(engine, runtime.context)
                        runtime.initialize(frames, actions)
                        current = frames[-1]

        action = keyboard_action(pygame.key.get_pressed())
        current = engine.step(action)[0] if assisted else runtime.step(action)
        screen.blit(surface(current, args.scale), (0, 0))
        pygame.draw.rect(screen, (12, 15, 21), (0, height, width, ui_height))
        mode = "ENGINE ASSISTED" if assisted else f"{args.model_type.upper()} PURE CLOSED LOOP"
        label = f"{mode}  ACTION {ACTION_NAMES[action]}  {clock.get_fps():4.1f} FPS  R reset  E engine  Esc"
        screen.blit(font.render(label, True, (232, 237, 245)), (8, height + 8))
        pygame.display.flip()
        clock.tick(args.fps)
        frame_count += 1
        if args.max_frames and frame_count >= args.max_frames:
            running = False
    pygame.quit()


if __name__ == "__main__":
    main()
