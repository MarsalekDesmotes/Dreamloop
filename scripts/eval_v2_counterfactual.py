from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_v2_world_model import initialize_context
from src.model_v2 import InverseDynamicsProbe
from src.runtime_v2 import LatentWorldRuntime, RGBWorldRuntime, frames_tensor
from src.toy_arena_v2 import ACTION_COUNT, ToyArenaV2
from src.training_v2 import load_trusted_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual action response test for Toy Arena V2.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--model-type", choices=("latent", "rgb"), default="latent")
    parser.add_argument("--contexts", type=int, default=24)
    parser.add_argument("--max-response-frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=8_000_003)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    runtime_class = LatentWorldRuntime if args.model_type == "latent" else RGBWorldRuntime
    runtime = runtime_class(args.checkpoint, device)
    probe_checkpoint = load_trusted_checkpoint(args.probes, map_location=device)
    inverse = InverseDynamicsProbe(ACTION_COUNT).to(device)
    inverse.load_state_dict(probe_checkpoint["inverse_probe"])
    inverse.eval()
    response_frames = []
    correct_transitions = 0
    total_transitions = 0

    for context_index in range(args.contexts):
        env = ToyArenaV2(args.seed + context_index * 1009)
        context_frames, context_actions = initialize_context(env, runtime.context)
        for action in range(ACTION_COUNT):
            runtime.initialize(context_frames, context_actions)
            previous_previous = context_frames[-2]
            previous = context_frames[-1]
            first_response = args.max_response_frames + 1
            for frame_index in range(1, args.max_response_frames + 1):
                prediction = runtime.step(action)
                with torch.inference_mode():
                    predicted_action = int(
                        inverse(
                            frames_tensor([previous_previous], device),
                            frames_tensor([previous], device),
                            frames_tensor([prediction], device),
                        ).argmax(dim=1).item()
                    )
                total_transitions += 1
                correct_transitions += int(predicted_action == action)
                if predicted_action == action and first_response > args.max_response_frames:
                    first_response = frame_index
                previous_previous = previous
                previous = prediction
            response_frames.append(first_response)

    p95 = float(np.percentile(response_frames, 95))
    metrics = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "contexts": args.contexts,
        "action_following": correct_transitions / total_transitions,
        "response_p95_frames": p95,
        "within_two_frames": float(np.mean(np.asarray(response_frames) <= 2)),
    }
    metrics["gate_pass"] = bool(metrics["action_following"] >= 0.90 and p95 <= 2.0)
    out = Path(args.out or Path(args.checkpoint).parent / f"{args.model_type}_counterfactual_metrics.json")
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
