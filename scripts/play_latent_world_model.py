from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pygame
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.play_sequence_world_model import make_action_plan
from scripts.play_world_model import NOOP, ToyArenaRuntime, choose_action, draw_label, make_surface, stabilize_prediction
from scripts.train_latent_dynamics import decode_sequence, encode_sequence
from src.data import load_coinrun_npz
from src.model import ActionConditionedLatentDynamics, ToyArenaAutoencoder


def history_to_tensor(history: deque[np.ndarray], device: str) -> torch.Tensor:
    frames = np.stack(list(history)).astype(np.float32) / 255.0
    frames = np.transpose(frames, (0, 3, 1, 2))
    return torch.from_numpy(frames[None]).to(device)


def predict_chunk(
    autoencoder: ToyArenaAutoencoder,
    model: ActionConditionedLatentDynamics,
    history: deque[np.ndarray],
    action: int,
    previous_action: int,
    horizon: int,
    device: str,
    action_inertia: int,
    stabilize: float,
    foreground_persist: float,
) -> list[np.ndarray]:
    frames = history_to_tensor(history, device)
    context_latents = encode_sequence(autoencoder, frames)
    actions = make_action_plan(action, horizon, previous_action=previous_action, inertia=action_inertia).to(device)
    pred_latents = model(context_latents, actions)
    pred = decode_sequence(autoencoder, pred_latents)[0].permute(0, 2, 3, 1).detach().cpu().numpy()
    raw = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    out: list[np.ndarray] = []
    previous = history[-1]
    for frame in raw:
        stable = stabilize_prediction(frame, previous, blend=stabilize, sharpen=0.0, foreground_persist=foreground_persist)
        out.append(stable)
        previous = stable
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the latent MIRA-lite world model in chunks.")
    parser.add_argument("--data", default="data/toy_arena_mixed_event_128_16k.npz")
    parser.add_argument("--checkpoint", default="runs/latent_dynamics_c8_h8_e20_gpu/best.pt")
    parser.add_argument("--autoencoder", default=None)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--assisted", action="store_true")
    parser.add_argument("--chunk-stride", type=int, default=2)
    parser.add_argument("--action-inertia", type=int, default=1)
    parser.add_argument("--stabilize", type=float, default=0.08)
    parser.add_argument("--foreground-persist", type=float, default=0.12)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    data_path = Path(args.data)
    checkpoint_path = Path(args.checkpoint)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    arrays = load_coinrun_npz(str(data_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ae_path = args.autoencoder or checkpoint["autoencoder_checkpoint"]
    ae_checkpoint = torch.load(ae_path, map_location=device)
    context = int(checkpoint["context"])
    horizon = int(checkpoint["horizon"])
    latent_channels = int(checkpoint["latent_channels"])

    autoencoder = ToyArenaAutoencoder(latent_channels=latent_channels).to(device)
    autoencoder.load_state_dict(ae_checkpoint["model"])
    autoencoder.eval()
    model = ActionConditionedLatentDynamics(
        action_count=int(checkpoint["action_count"]),
        latent_channels=latent_channels,
        context=context,
        horizon=horizon,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    runtime = ToyArenaRuntime(size=int(arrays.frames.shape[1]), seed=args.seed)
    history = runtime.warmup(context)
    current = history[-1].copy()
    chunk: deque[np.ndarray] = deque()
    chunk_age = args.chunk_stride
    previous_action = NOOP
    assisted = bool(args.assisted)

    pygame.init()
    pygame.display.set_caption("MiniMIRA latent chunk world model")
    height, width = current.shape[:2]
    screen = pygame.display.set_mode((width * args.scale, height * args.scale))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    running = True
    frame_count = 0
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
                        chunk.clear()
                        chunk_age = args.chunk_stride
                        previous_action = NOOP
                    elif event.key == pygame.K_e:
                        assisted = not assisted
                        chunk.clear()
                        chunk_age = args.chunk_stride

            action = choose_action(pygame.key.get_pressed())
            if action != previous_action:
                chunk.clear()
                chunk_age = args.chunk_stride

            if assisted:
                current = runtime.step(action)
                history.append(current)
                chunk.clear()
            else:
                if not chunk or chunk_age >= max(1, args.chunk_stride):
                    chunk = deque(
                        predict_chunk(
                            autoencoder,
                            model,
                            history,
                            action,
                            previous_action=previous_action,
                            horizon=horizon,
                            device=device,
                            action_inertia=max(0, args.action_inertia),
                            stabilize=max(0.0, min(args.stabilize, 0.85)),
                            foreground_persist=max(0.0, min(args.foreground_persist, 0.85)),
                        )
                    )
                    chunk_age = 0
                current = chunk.popleft()
                history.append(current)
                chunk_age += 1

            previous_action = action
            screen.blit(make_surface(current, args.scale), (0, 0))
            mode = "engine assisted" if assisted else f"latent closed-loop {device}"
            draw_label(screen, font, f"{mode} | WASD/arrows | Space dash | R reset | E engine | Esc")
            pygame.display.flip()
            clock.tick(args.fps)
            frame_count += 1
            if args.max_frames > 0 and frame_count >= args.max_frames:
                running = False

    pygame.quit()


if __name__ == "__main__":
    main()
