from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_coinrun_npz
from src.model import ActionConditionedSequencePredictor


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    frames = frames.astype(np.float32) / 255.0
    frames = np.transpose(frames, (0, 3, 1, 2)).reshape(1, frames.shape[0] * 3, frames.shape[1], frames.shape[2])
    return torch.from_numpy(frames)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float32) / 255.0
    bf = b.astype(np.float32) / 255.0
    return float(np.mean((af - bf) ** 2))


def foreground_mse(a: np.ndarray, b: np.ndarray, threshold: float = 0.18) -> float:
    af = a.astype(np.float32) / 255.0
    bf = b.astype(np.float32) / 255.0
    saturation = bf.max(axis=2, keepdims=True) - bf.min(axis=2, keepdims=True)
    brightness = bf.mean(axis=2, keepdims=True)
    mask = ((saturation > threshold) | (brightness > 0.48)).astype(np.float32)
    return float(np.sum((af - bf) ** 2 * mask) / max(float(mask.sum() * 3), 1.0))


def player_mse(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float32) / 255.0
    bf = b.astype(np.float32) / 255.0
    red = bf[:, :, 0:1]
    green = bf[:, :, 1:2]
    blue = bf[:, :, 2:3]
    mask = ((blue > 0.45) & (green > 0.25) & (red < 0.55) & (blue - green > 0.15) & (blue - red > 0.20)).astype(
        np.float32
    )
    return float(np.sum((af - bf) ** 2 * mask) / max(float(mask.sum() * 3), 1.0))


def label_panel(frame: np.ndarray, label: str) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, frame.shape[1], 14), fill=(10, 12, 18))
    draw.text((4, 2), label, fill=(235, 240, 248), font=font)
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render and score a direct multi-frame sequence model.")
    parser.add_argument("--data", default="data/toy_arena_mixed_event_128_16k.npz")
    parser.add_argument("--checkpoint", default="runs/sequence_model/best.pt")
    parser.add_argument("--out", default="runs/sequence_model/sequence_preview.gif")
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--closed-loop", action="store_true", help="Feed predicted chunks back as context between chunks.")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    arrays = load_coinrun_npz(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    context = int(checkpoint["context"])
    horizon = int(checkpoint["horizon"])
    model = ActionConditionedSequencePredictor(
        action_count=int(checkpoint["action_count"]),
        context=context,
        horizon=horizon,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if args.start + context + args.chunks * horizon >= len(arrays.frames):
        raise ValueError("start/chunks/horizon exceed dataset length.")

    history = arrays.frames[args.start : args.start + context].copy()
    panels = []
    losses = []
    fg_losses = []
    player_losses = []
    separator = np.full((history.shape[1], 6, 3), 255, dtype=np.uint8)

    with torch.no_grad():
        for chunk in range(args.chunks):
            target_start = args.start + context + chunk * horizon
            target_end = target_start + horizon
            actions = torch.from_numpy(arrays.actions[target_start - 1 : target_end - 1].astype(np.int64))[None]
            pred = model(frames_to_tensor(history), actions)[0].permute(0, 2, 3, 1).numpy()
            pred_u8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
            real = arrays.frames[target_start:target_end]

            for step in range(horizon):
                losses.append(mse(pred_u8[step], real[step]))
                fg_losses.append(foreground_mse(pred_u8[step], real[step]))
                player_losses.append(player_mse(pred_u8[step], real[step]))
                panels.append(
                    np.concatenate(
                        [
                            label_panel(pred_u8[step], f"sequence pred t+{chunk * horizon + step + 1}"),
                            separator,
                            label_panel(real[step], "real future"),
                        ],
                        axis=1,
                    )
                )

            if args.closed_loop:
                history = np.concatenate([history, pred_u8], axis=0)[-context:]
            else:
                history = arrays.frames[target_end - context : target_end].copy()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".mp4":
        imageio.mimsave(out, panels, fps=args.fps)
    else:
        imageio.mimsave(out, panels, duration=1.0 / args.fps)
    metrics = {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "context": context,
        "horizon": horizon,
        "chunks": args.chunks,
        "closed_loop": args.closed_loop,
        "mse": float(np.mean(losses)),
        "foreground_mse": float(np.mean(fg_losses)),
        "player_mse": float(np.mean(player_losses)),
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
