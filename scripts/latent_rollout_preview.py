from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sequence_rollout_preview import foreground_mse, label_panel, mse, player_mse
from scripts.train_latent_dynamics import decode_sequence, encode_sequence
from src.data import load_coinrun_npz
from src.model import ActionConditionedLatentDynamics, ToyArenaAutoencoder


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    frames = frames.astype(np.float32) / 255.0
    frames = np.transpose(frames, (0, 3, 1, 2))
    return torch.from_numpy(frames[None])


def latent_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a - b).square()).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Render and score latent world model rollouts.")
    parser.add_argument("--data", default="data/toy_arena_mixed_event_128_16k.npz")
    parser.add_argument("--checkpoint", default="runs/latent_dynamics_c8_h8_e20_gpu/best.pt")
    parser.add_argument("--autoencoder", default=None, help="Optional override for the autoencoder checkpoint.")
    parser.add_argument("--out", default="runs/latent_dynamics_c8_h8_e20_gpu/latent_preview.mp4")
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--start", type=int, default=120)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    arrays = load_coinrun_npz(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ae_path = args.autoencoder or checkpoint["autoencoder_checkpoint"]
    ae_checkpoint = torch.load(ae_path, map_location="cpu")
    latent_channels = int(checkpoint["latent_channels"])
    context = int(checkpoint["context"])
    horizon = int(checkpoint["horizon"])

    autoencoder = ToyArenaAutoencoder(latent_channels=latent_channels)
    autoencoder.load_state_dict(ae_checkpoint["model"])
    autoencoder.eval()
    model = ActionConditionedLatentDynamics(
        action_count=int(checkpoint["action_count"]),
        latent_channels=latent_channels,
        context=context,
        horizon=horizon,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if args.start + context + args.chunks * horizon >= len(arrays.frames):
        raise ValueError("start/chunks/horizon exceed dataset length.")

    history_frames = arrays.frames[args.start : args.start + context].copy()
    history_latents = encode_sequence(autoencoder, frames_to_tensor(history_frames))[0]
    panels = []
    losses = []
    fg_losses = []
    player_losses = []
    latent_losses = []
    separator = np.full((history_frames.shape[1], 6, 3), 255, dtype=np.uint8)

    with torch.no_grad():
        for chunk in range(args.chunks):
            target_start = args.start + context + chunk * horizon
            target_end = target_start + horizon
            actions = torch.from_numpy(arrays.actions[target_start - 1 : target_end - 1].astype(np.int64))[None]
            target_frames = arrays.frames[target_start:target_end]
            target_tensor = frames_to_tensor(target_frames)
            target_latents = encode_sequence(autoencoder, target_tensor)[0]

            pred_latents = model(history_latents[None], actions)[0]
            pred = decode_sequence(autoencoder, pred_latents[None])[0].permute(0, 2, 3, 1).numpy()
            pred_u8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
            latent_losses.append(latent_mse(pred_latents, target_latents))

            for step in range(horizon):
                losses.append(mse(pred_u8[step], target_frames[step]))
                fg_losses.append(foreground_mse(pred_u8[step], target_frames[step]))
                player_losses.append(player_mse(pred_u8[step], target_frames[step]))
                panels.append(
                    np.concatenate(
                        [
                            label_panel(pred_u8[step], f"latent pred t+{chunk * horizon + step + 1}"),
                            separator,
                            label_panel(target_frames[step], "real future"),
                        ],
                        axis=1,
                    )
                )

            if args.closed_loop:
                history_latents = torch.cat([history_latents, pred_latents], dim=0)[-context:]
            else:
                history_frames = arrays.frames[target_end - context : target_end].copy()
                history_latents = encode_sequence(autoencoder, frames_to_tensor(history_frames))[0]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, panels, fps=args.fps) if out.suffix.lower() == ".mp4" else imageio.mimsave(out, panels, duration=1.0 / args.fps)
    metrics = {
        "checkpoint": args.checkpoint,
        "autoencoder": str(ae_path),
        "data": args.data,
        "context": context,
        "horizon": horizon,
        "chunks": args.chunks,
        "closed_loop": args.closed_loop,
        "mse": float(np.mean(losses)),
        "foreground_mse": float(np.mean(fg_losses)),
        "player_mse": float(np.mean(player_losses)),
        "latent_mse": float(np.mean(latent_losses)),
    }
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out}")
    if args.metrics_out:
        metrics_out = Path(args.metrics_out)
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"wrote {metrics_out}")


if __name__ == "__main__":
    main()
