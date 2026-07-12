from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_coinrun_npz
from src.model import ActionConditionedNextFrame


def to_tensor(frames: np.ndarray) -> torch.Tensor:
    frames = frames.astype(np.float32) / 255.0
    frames = np.transpose(frames, (0, 3, 1, 2)).reshape(1, frames.shape[0] * 3, frames.shape[1], frames.shape[2])
    return torch.from_numpy(frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/coinrun_20k.npz")
    parser.add_argument("--checkpoint", default="runs/coinrun_next_frame/best.pt")
    parser.add_argument("--out", default="runs/coinrun_next_frame/rollout.gif")
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()

    arrays = load_coinrun_npz(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    context = int(checkpoint["context"])
    model = ActionConditionedNextFrame(action_count=int(checkpoint["action_count"]), context=context)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    history = arrays.frames[args.start : args.start + context].copy()
    panels = []
    with torch.no_grad():
        for step in range(args.steps):
            action_idx = args.start + context + step - 1
            target_idx = args.start + context + step
            action = torch.tensor([int(arrays.actions[action_idx])], dtype=torch.long)
            pred = model(to_tensor(history), action)[0].permute(1, 2, 0).numpy()
            pred_u8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
            real = arrays.frames[target_idx]

            separator = np.full((real.shape[0], 4, 3), 255, dtype=np.uint8)
            panels.append(np.concatenate([pred_u8, separator, real], axis=1))

            # Teacher-forced preview keeps the context on real frames; closed-loop comes next.
            history = arrays.frames[target_idx - context + 1 : target_idx + 1]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, panels, duration=0.12)
    print(f"wrote {out} (left=prediction, right=real)")


if __name__ == "__main__":
    main()
