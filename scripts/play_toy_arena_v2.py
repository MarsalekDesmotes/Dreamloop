from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.toy_arena_v2 import (
    DASH,
    GAME_LOST,
    GAME_WON,
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
    NOOP,
    ToyArenaV2,
)


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


def scaled_surface(frame: np.ndarray, scale: int) -> pygame.Surface:
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    return pygame.transform.scale(surface, (frame.shape[1] * scale, frame.shape[0] * scale))


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the Toy Arena gameplay engine.")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--scale", type=int, default=5)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.display.set_caption("Toy Arena - Escape Protocol")
    env = ToyArenaV2(seed=args.seed)
    frame = env.render()
    width = frame.shape[1] * args.scale
    height = frame.shape[0] * args.scale
    footer_height = 34
    screen = pygame.display.set_mode((width, height + footer_height))
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
                    args.seed += 1
                    frame = env.reset(args.seed)

        action = keyboard_action(pygame.key.get_pressed())
        frame, _ = env.step(action)
        screen.blit(scaled_surface(frame, args.scale), (0, 0))
        pygame.draw.rect(screen, (12, 15, 21), (0, height, width, footer_height))
        if env.state.game_status == GAME_WON:
            label = "ESCAPED  R new run  Esc quit"
            color = (112, 255, 196)
        elif env.state.game_status == GAME_LOST:
            label = "RUN LOST  R retry  Esc quit"
            color = (255, 126, 136)
        else:
            remaining = max(0, 3 - env.state.score)
            label = f"ENERGY {env.state.score}/3  HP {env.state.health}/3  {remaining} TO PORTAL  R reset"
            color = (232, 237, 245)
        screen.blit(font.render(label, True, color), (8, height + 8))
        pygame.display.flip()
        clock.tick(args.fps)
        frame_count += 1
        if args.max_frames and frame_count >= args.max_frames:
            running = False
    pygame.quit()


if __name__ == "__main__":
    main()
