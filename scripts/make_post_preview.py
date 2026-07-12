from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def add_header(frame: np.ndarray, scale: int, title: str) -> np.ndarray:
    image = Image.fromarray(frame).resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.Resampling.NEAREST)
    header_h = 54
    canvas = Image.new("RGB", (image.width, image.height + header_h), (12, 15, 22))
    canvas.paste(image, (0, header_h))
    draw = ImageDraw.Draw(canvas)

    left_w = (frame.shape[1] - 4) * scale // 2
    right_x = left_w + 4 * scale
    draw.text((18, 14), "MODEL PREDICTION", fill=(130, 190, 255))
    draw.text((right_x + 18, 14), "ENGINE / GROUND TRUTH", fill=(130, 255, 185))
    if title:
        draw.text((image.width // 2 - len(title) * 3, 34), title, fill=(230, 235, 245))
    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", required=True)
    parser.add_argument("--gif-out", default="runs/toy_arena_128/post_preview.gif")
    parser.add_argument("--mp4-out", default="runs/toy_arena_128/post_preview.mp4")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--title", default="Tiny action-conditioned world model")
    args = parser.parse_args()

    frames = imageio.mimread(args.input)
    processed = [add_header(np.asarray(frame[:, :, :3]), args.scale, args.title) for frame in frames]

    gif_out = Path(args.gif_out)
    gif_out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_out, processed, duration=1.0 / args.fps)
    print(f"wrote {gif_out}")

    mp4_out = Path(args.mp4_out)
    mp4_out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(mp4_out, processed, fps=args.fps, macro_block_size=1)
    print(f"wrote {mp4_out}")


if __name__ == "__main__":
    main()
