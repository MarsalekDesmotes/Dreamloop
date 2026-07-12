from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_v2 import load_toy_arena_v2
from src.eval_v2 import decode_state_probe
from src.model_v2 import ArenaStateProbe
from src.runtime_v2 import LatentWorldRuntime, frames_tensor
from src.training_v2 import load_trusted_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Toy Arena V2 one-episode overfit gate.")
    parser.add_argument("--data", default="data/toy_arena_v2_60k")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--start", type=int, default=24)
    parser.add_argument("--out", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    arrays = load_toy_arena_v2(args.data)
    checkpoint = load_trusted_checkpoint(args.checkpoint, map_location="cpu")
    if checkpoint.get("episode_id") not in (None, args.episode_id):
        raise ValueError("checkpoint was trained for a different overfit episode")
    if int(arrays.episode_splits[args.episode_id]) != 0:
        raise ValueError("overfit evaluation requires a training episode")

    runtime = LatentWorldRuntime(args.checkpoint, device)
    context = runtime.context
    episode_length = int(arrays.metadata["episode_length"])
    episode_start = args.episode_id * episode_length
    first_target = episode_start + max(args.start, context)
    last_target = first_target + args.frames
    episode_end = episode_start + episode_length
    if last_target >= episode_end:
        raise ValueError("requested rollout extends beyond the episode")
    context_start = first_target - context
    context_frames = [np.asarray(frame) for frame in arrays.frames[context_start:first_target]]
    context_actions = [int(action) for action in arrays.actions[context_start : first_target - 1]]
    runtime.initialize(context_frames, context_actions)

    probe_checkpoint = load_trusted_checkpoint(args.probes, map_location=device)
    state_probe = ArenaStateProbe().to(device)
    state_probe.load_state_dict(probe_checkpoint["state_probe"])
    state_probe.eval()
    player_errors: list[float] = []
    coin_errors: list[float] = []
    video_frames: list[np.ndarray] = []
    for index in range(first_target, last_target):
        prediction = runtime.step(int(arrays.actions[index - 1]))
        if args.video:
            video_frames.append(np.concatenate([prediction, np.asarray(arrays.frames[index])], axis=1))
        tensor = frames_tensor([prediction], device)
        with torch.inference_mode():
            decoded = decode_state_probe(state_probe(tensor))
        player = torch.tensor(np.asarray(arrays.player_pos[index]), device=device)[None]
        coin = torch.tensor(np.asarray(arrays.coin_pos[index]), device=device)[None]
        player_errors.append(float(torch.linalg.vector_norm(decoded["player_pos"] - player, dim=1).item()))
        coin_errors.append(float(torch.linalg.vector_norm(decoded["coin_pos"] - coin, dim=1).item()))

    result = {
        "checkpoint": args.checkpoint,
        "episode_id": args.episode_id,
        "frames": args.frames,
        "player_centroid_error": float(np.mean(player_errors)),
        "coin_centroid_error": float(np.mean(coin_errors)),
        "player_p95_error": float(np.percentile(player_errors, 95)),
        "coin_p95_error": float(np.percentile(coin_errors, 95)),
        "worst_player_frames": [
            {"offset": int(index + 1), "error": float(player_errors[index])}
            for index in np.argsort(player_errors)[-5:][::-1]
        ],
        "prefix_errors": {
            str(prefix): {
                "player": float(np.mean(player_errors[:prefix])),
                "coin": float(np.mean(coin_errors[:prefix])),
            }
            for prefix in (1, 2, 4, 8, 16, 32, 64)
            if prefix <= args.frames
        },
    }
    result["gate_pass"] = bool(
        result["player_centroid_error"] <= 1.0 and result["coin_centroid_error"] <= 1.0
    )
    output = Path(args.out or Path(args.checkpoint).parent / "overfit_metrics.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.video:
        imageio.mimsave(args.video, video_frames, fps=int(arrays.metadata["fps"]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
