from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_v2_world_model import initialize_context
from src.eval_v2 import decode_state_probe
from src.model_v2 import ArenaStateProbe
from src.runtime_v2 import LatentWorldRuntime, RGBWorldRuntime, frames_tensor
from src.toy_arena_v2 import EVENT_COIN, EVENT_COLLISION, POLICY_COIN, POLICY_COLLISION, ToyArenaV2, choose_policy_action
from src.training_v2 import load_trusted_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out Toy Arena V2 coin and collision events.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--model-type", choices=("latent", "rgb"), default="latent")
    parser.add_argument("--out", default=None)
    parser.add_argument("--target-events", type=int, default=200)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--seed", type=int, default=9_000_001)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    runtime_class = LatentWorldRuntime if args.model_type == "latent" else RGBWorldRuntime
    runtime = runtime_class(args.checkpoint, device)
    probe_checkpoint = load_trusted_checkpoint(args.probes, map_location=device)
    state_probe = ArenaStateProbe().to(device)
    state_probe.load_state_dict(probe_checkpoint["state_probe"])
    state_probe.eval()
    results = {}

    for name, event_bit, policy in (
        ("coin", EVENT_COIN, POLICY_COIN),
        ("collision", EVENT_COLLISION, POLICY_COLLISION),
    ):
        completed = 0
        passed = 0
        episode = 0
        pending: list[list[int]] = []
        while completed < args.target_events:
            seed = args.seed + episode * 7919 + policy * 101
            episode += 1
            env = ToyArenaV2(seed)
            context_frames, context_actions = initialize_context(env, runtime.context)
            runtime.initialize(context_frames, context_actions)
            rng = np.random.default_rng(seed ^ 0xE7E7)
            for step in range(args.episode_steps):
                action = choose_policy_action(env, rng, policy, step)
                _, event = env.step(action)
                prediction = runtime.step(action)
                with torch.inference_mode():
                    state = decode_state_probe(state_probe(frames_tensor([prediction], device)))
                snapshot = env.snapshot()
                if event_bit == EVENT_COIN:
                    error = float(
                        torch.linalg.vector_norm(
                            state["coin_pos"] - torch.tensor(snapshot["coin_pos"], device=device)[None], dim=1
                        ).item()
                    )
                    success = int(state["coin_score"].item() >= 0.35 and error <= 12.0)
                else:
                    error = float(
                        torch.linalg.vector_norm(
                            state["player_pos"] - torch.tensor(snapshot["player_pos"], device=device)[None], dim=1
                        ).item()
                    )
                    success = int(state["player_score"].item() >= 0.35 and error <= 12.0)

                for item in pending:
                    item[0] -= 1
                    item[1] += success
                finished = [item for item in pending if item[0] <= 0]
                pending = [item for item in pending if item[0] > 0]
                for item in finished:
                    completed += 1
                    passed += int(item[1] >= 4)
                    if completed >= args.target_events:
                        break
                if completed >= args.target_events:
                    break
                if event & event_bit:
                    pending.append([4, success])
            if episode > 10_000:
                raise RuntimeError(f"could not collect {args.target_events} {name} events")
        results[name] = {"events": completed, "passed": passed, "consistency": passed / completed, "episodes": episode}

    metrics = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        **results,
        "event_consistency": float(np.mean([results["coin"]["consistency"], results["collision"]["consistency"]])),
    }
    metrics["gate_pass"] = bool(
        results["coin"]["events"] >= 200
        and results["collision"]["events"] >= 200
        and metrics["event_consistency"] >= 0.95
    )
    out = Path(args.out or Path(args.checkpoint).parent / f"{args.model_type}_event_metrics.json")
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

