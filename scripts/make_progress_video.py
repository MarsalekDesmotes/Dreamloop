from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_coinrun_npz
from src.model import ActionConditionedNextFrame


def to_tensor(frames: np.ndarray) -> torch.Tensor:
    frames = frames.astype(np.float32) / 255.0
    frames = np.transpose(frames, (0, 3, 1, 2)).reshape(1, frames.shape[0] * 3, frames.shape[1], frames.shape[2])
    return torch.from_numpy(frames)


def load_model(checkpoint_path: str) -> tuple[ActionConditionedNextFrame, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    context = int(checkpoint["context"])
    model = ActionConditionedNextFrame(action_count=int(checkpoint["action_count"]), context=context)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, context


def predictions(model: ActionConditionedNextFrame, context: int, frames: np.ndarray, actions: np.ndarray, start: int, steps: int) -> list[np.ndarray]:
    history = frames[start : start + context].copy()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for step in range(steps):
            action_idx = start + context + step - 1
            target_idx = start + context + step
            action = torch.tensor([int(actions[action_idx])], dtype=torch.long)
            pred = model(to_tensor(history), action)[0].permute(1, 2, 0).numpy()
            out.append(np.clip(pred * 255.0, 0, 255).astype(np.uint8))
            history = frames[target_idx - context + 1 : target_idx + 1]
    return out


def label_panel(frame: np.ndarray, label: str, scale: int) -> Image.Image:
    image = Image.fromarray(frame).resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.Resampling.NEAREST)
    header = 34
    canvas = Image.new("RGB", (image.width, image.height + header), (12, 15, 22))
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), label, fill=(230, 235, 245))
    return canvas


def make_compare(pred: np.ndarray, real: np.ndarray) -> np.ndarray:
    h = real.shape[0]
    separator = np.full((h, 4, 3), 255, dtype=np.uint8)
    return np.concatenate([pred, separator, real], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/toy_arena_128_50k.npz")
    parser.add_argument("--early", default="runs/toy_arena_post/best.pt")
    parser.add_argument("--improved", default="runs/toy_arena_post_20ep/best.pt")
    parser.add_argument("--out", default="runs/toy_arena_post_20ep/progress_4panel_30s.mp4")
    parser.add_argument("--gif-out", default="")
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()

    arrays = load_coinrun_npz(args.data)
    steps = args.seconds * args.fps

    early_model, early_context = load_model(args.early)
    improved_model, improved_context = load_model(args.improved)
    context = max(early_context, improved_context)
    if early_context != improved_context:
        raise ValueError("Early and improved checkpoints must use the same context")
    if args.start + context + steps >= len(arrays.frames):
        raise ValueError("Requested video runs past dataset length")

    early_preds = predictions(early_model, context, arrays.frames, arrays.actions, args.start, steps)
    improved_preds = predictions(improved_model, context, arrays.frames, arrays.actions, args.start, steps)
    real_frames = [arrays.frames[args.start + context + i] for i in range(steps)]

    composed: list[np.ndarray] = []
    for i in range(steps):
        engine = label_panel(real_frames[i], "01 ENGINE / GROUND TRUTH", args.scale)
        early = label_panel(early_preds[i], "02 EARLY MODEL", args.scale)
        improved = label_panel(improved_preds[i], "03 IMPROVED MODEL", args.scale)
        compare = label_panel(make_compare(improved_preds[i], real_frames[i]), "04 FINAL: PREDICTION vs ENGINE", args.scale)

        panel_w = max(engine.width, early.width, improved.width, compare.width)
        panel_h = max(engine.height, early.height, improved.height, compare.height)

        def pad(panel: Image.Image) -> Image.Image:
            padded = Image.new("RGB", (panel_w, panel_h), (12, 15, 22))
            padded.paste(panel, ((panel_w - panel.width) // 2, (panel_h - panel.height) // 2))
            return padded

        gap = 12
        frame = Image.new("RGB", (panel_w * 2 + gap, panel_h * 2 + gap), (8, 10, 14))
        frame.paste(pad(engine), (0, 0))
        frame.paste(pad(early), (panel_w + gap, 0))
        frame.paste(pad(improved), (0, panel_h + gap))
        frame.paste(pad(compare), (panel_w + gap, panel_h + gap))
        composed.append(np.asarray(frame))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, composed, fps=args.fps, macro_block_size=1)
    print(f"wrote {out}")

    if args.gif_out:
        gif_out = Path(args.gif_out)
        gif_out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(gif_out, composed, duration=1.0 / args.fps)
        print(f"wrote {gif_out}")


if __name__ == "__main__":
    main()
