from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_v2 import decode_visible_game_state, load_toy_arena_v2
from src.eval_v2 import decode_state_probe
from src.model_v2 import InverseDynamicsProbe, arena_state_probe_from_checkpoint
from src.runtime_v2 import LatentWorldRuntime, RGBWorldRuntime, SemanticWorldRuntime, frames_tensor
from src.toy_arena_v2 import (
    ACTION_COUNT,
    EVENT_COIN,
    EVENT_COLLISION,
    EVENT_LOSE,
    EVENT_WIN,
    GAME_RUNNING,
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
    NOOP,
    POLICY_COIN,
    POLICY_COLLISION,
    POLICY_RANDOM,
    POLICY_SCRIPTED,
    ToyArenaV2,
    choose_policy_action,
)
from src.training_v2 import load_trusted_checkpoint


MODE_NAMES = ("random", "coin", "collision", "scripted", "noop", "counterfactual")


def eval_action(env: ToyArenaV2, rng: np.random.Generator, mode: int, step: int) -> int:
    if env.state.game_status != GAME_RUNNING:
        return NOOP
    if mode < 4:
        return choose_policy_action(env, rng, mode, step)
    if mode == 4:
        return NOOP
    pattern = (MOVE_UP, MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT, 5, NOOP)
    return pattern[(step // 8) % len(pattern)]


def initialize_context(env: ToyArenaV2, context: int) -> tuple[list[np.ndarray], list[int]]:
    frames = [env.render()]
    actions = []
    warmup = (MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT, MOVE_UP, NOOP)
    for step in range(context - 1):
        action = warmup[(step // 4) % len(warmup)]
        actions.append(action)
        frame, _ = env.step(action)
        frames.append(frame)
    return frames, actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure closed-loop Toy Arena V2 acceptance evaluator.")
    parser.add_argument("--data", default="data/toy_arena_v2_60k")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--model-type", choices=("latent", "rgb", "semantic"), default="latent")
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--rollouts", type=int, default=24)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--video-rollouts", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--project-latent", action="store_true", help="Diagnostic codec-manifold projection.")
    parser.add_argument("--quantize-step", type=float, default=0.0, help="Diagnostic normalized latent scalar snap.")
    parser.add_argument("--collision-threshold", type=float, default=None)
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    arrays = load_toy_arena_v2(args.data)
    probe_checkpoint = load_trusted_checkpoint(args.probes, map_location=device)
    state_probe = arena_state_probe_from_checkpoint(probe_checkpoint).to(device)
    state_probe.load_state_dict(probe_checkpoint["state_probe"])
    state_probe.eval()
    inverse_probe = InverseDynamicsProbe(int(arrays.metadata["action_count"])).to(device)
    inverse_probe.load_state_dict(probe_checkpoint["inverse_probe"])
    inverse_probe.eval()
    runtime_class = {
        "latent": LatentWorldRuntime,
        "rgb": RGBWorldRuntime,
        "semantic": SemanticWorldRuntime,
    }[args.model_type]
    runtime = (
        runtime_class(
            args.checkpoint,
            device,
            project_latent=args.project_latent,
            quantize_step=args.quantize_step,
        )
        if args.model_type == "latent"
        else (
            runtime_class(args.checkpoint, device, collision_threshold=args.collision_threshold)
            if args.model_type == "semantic"
            else runtime_class(args.checkpoint, device)
        )
    )
    fps = int(arrays.metadata["fps"])
    steps = args.seconds * fps
    test_episode_ids = np.flatnonzero(np.asarray(arrays.episode_splits) == 2)
    if len(test_episode_ids) == 0:
        raise ValueError("dataset contains no test episodes")
    seeds = np.asarray(arrays.episode_seeds[test_episode_ids], dtype=np.uint32)
    out_dir = Path(args.out_dir or Path(args.checkpoint).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollout_records = []

    for rollout_index in range(args.rollouts):
        seed = int(seeds[rollout_index % len(seeds)] + np.uint32((rollout_index // len(seeds)) * 100_003))
        mode = rollout_index % len(MODE_NAMES)
        env = ToyArenaV2(seed=seed)
        context_frames, context_actions = initialize_context(env, runtime.context)
        runtime.initialize(context_frames, context_actions)
        rng = np.random.default_rng(seed ^ 0x5EED1234)
        previous_previous = context_frames[-2]
        previous_prediction = context_frames[-1]
        presence = {"player": 0.0, "coin": 0.0, "enemy": 0.0}
        self_presence = {"player": 0.0, "coin": 0.0, "enemy": 0.0}
        errors = {"player": 0.0, "coin": 0.0, "enemy": 0.0}
        action_correct = 0.0
        event_correct = 0.0
        event_count = 0
        health_correct = 0.0
        progress_correct = 0.0
        terminal_correct = 0.0
        collapsed = False
        video = []
        started = time.perf_counter()
        for step in range(steps):
            action = eval_action(env, rng, mode, step)
            target_frame, event = env.step(action)
            prediction = runtime.step(action)
            frame_tensor = frames_tensor([prediction], device)
            previous_tensor = frames_tensor([previous_prediction], device)
            previous_previous_tensor = frames_tensor([previous_previous], device)
            with torch.inference_mode():
                state = decode_state_probe(state_probe(frame_tensor))
                inverse = inverse_probe(previous_previous_tensor, previous_tensor, frame_tensor).argmax(dim=1)
            snapshot = env.snapshot()
            visible_game = decode_visible_game_state(prediction)[0]
            predicted_health = int(round(float(visible_game[0]) * 3.0))
            predicted_progress = int(round(float(visible_game[1]) * 3.0))
            predicted_status = 1 if visible_game[2] > 0.5 else (-1 if visible_game[3] > 0.5 else 0)
            health_correct += float(predicted_health == int(snapshot["health"]))
            progress_correct += float(predicted_progress == min(int(snapshot["score"]), 3))
            terminal_correct += float(predicted_status == int(snapshot["game_status"]))
            target_player = torch.tensor(snapshot["player_pos"], device=device)[None]
            target_coin = torch.tensor(snapshot["coin_pos"], device=device)[None]
            target_enemy = torch.tensor(snapshot["enemy_pos"], device=device)[None]
            player_error = float(torch.linalg.vector_norm(state["player_pos"] - target_player, dim=1).item())
            coin_error = float(torch.linalg.vector_norm(state["coin_pos"] - target_coin, dim=1).item())
            enemy_distance = torch.cdist(state["enemy_pos"], target_enemy).amin(dim=2).mean()
            errors["player"] += player_error
            errors["coin"] += coin_error
            errors["enemy"] += float(enemy_distance.item())
            presence["player"] += float(state["player_score"].item() >= 0.35 and player_error <= 12.0)
            presence["coin"] += float(state["coin_score"].item() >= 0.35 and coin_error <= 12.0)
            presence["enemy"] += float(
                (state["enemy_score"] >= 0.35).float().mean().item() * (float(enemy_distance.item()) <= 12.0)
            )
            self_presence["player"] += float(state["player_score"].item() >= 0.35)
            self_presence["coin"] += float(state["coin_score"].item() >= 0.35)
            self_presence["enemy"] += float((state["enemy_score"] >= 0.35).float().mean().item())
            action_correct += float((inverse == action).float().item())
            if event & (EVENT_COIN | EVENT_COLLISION | EVENT_WIN | EVENT_LOSE):
                event_count += 1
                if event & EVENT_COIN:
                    event_correct += float(predicted_progress == min(int(snapshot["score"]), 3))
                elif event & EVENT_COLLISION:
                    event_correct += float(predicted_health == int(snapshot["health"]))
                elif event & EVENT_WIN:
                    event_correct += float(predicted_status == 1)
                else:
                    event_correct += float(predicted_status == -1)
            if not np.isfinite(prediction).all() or float(prediction.std()) < 3.0:
                collapsed = True
            if rollout_index < args.video_rollouts:
                video.append(np.concatenate([prediction, target_frame], axis=1))
            previous_previous = previous_prediction
            previous_prediction = prediction
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        record = {
            "seed": seed,
            "mode": MODE_NAMES[mode],
            "seconds": args.seconds,
            "collapsed": collapsed,
            "fps": steps / elapsed,
            "player_presence": presence["player"] / steps,
            "coin_presence": presence["coin"] / steps,
            "enemy_presence": presence["enemy"] / steps,
            "self_player_presence": self_presence["player"] / steps,
            "self_coin_presence": self_presence["coin"] / steps,
            "self_enemy_presence": self_presence["enemy"] / steps,
            "player_error": errors["player"] / steps,
            "coin_error": errors["coin"] / steps,
            "enemy_error": errors["enemy"] / steps,
            "action_following": action_correct / steps,
            "event_consistency": event_correct / max(event_count, 1),
            "event_correct_count": event_correct,
            "health_accuracy": health_correct / steps,
            "progress_accuracy": progress_correct / steps,
            "terminal_accuracy": terminal_correct / steps,
            "event_count": event_count,
        }
        rollout_records.append(record)
        if video:
            imageio.mimsave(out_dir / f"{args.model_type}_{args.seconds}s_rollout_{rollout_index}.mp4", video, fps=fps)
        print(json.dumps(record))

    aggregate = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "seconds": args.seconds,
        "rollouts": args.rollouts,
        "project_latent": args.project_latent,
        "quantize_step": args.quantize_step,
        "collision_threshold": args.collision_threshold,
        "no_collapse_rate": float(np.mean([not record["collapsed"] for record in rollout_records])),
        **{
            key: float(np.mean([record[key] for record in rollout_records]))
            for key in (
                "fps",
                "player_presence",
                "coin_presence",
                "enemy_presence",
                "self_player_presence",
                "self_coin_presence",
                "self_enemy_presence",
                "player_error",
                "coin_error",
                "enemy_error",
                "action_following",
                "health_accuracy",
                "progress_accuracy",
                "terminal_accuracy",
            )
        },
        "event_count": int(sum(record["event_count"] for record in rollout_records)),
    }
    aggregate["event_correct_count"] = float(sum(record["event_correct_count"] for record in rollout_records))
    aggregate["event_consistency"] = aggregate["event_correct_count"] / max(aggregate["event_count"], 1)
    aggregate["object_presence"] = float(
        np.mean([aggregate["player_presence"], aggregate["coin_presence"], aggregate["enemy_presence"]])
    )
    aggregate["self_object_presence"] = float(
        np.mean(
            [
                aggregate["self_player_presence"],
                aggregate["self_coin_presence"],
                aggregate["self_enemy_presence"],
            ]
        )
    )
    aggregate["self_stability_gate"] = bool(
        aggregate["no_collapse_rate"] == 1.0
        and aggregate["self_object_presence"] >= 0.97
        and aggregate["fps"] >= 12.0
    )
    aggregate["gate_10s"] = bool(
        args.seconds == 10
        and aggregate["no_collapse_rate"] == 1.0
        and aggregate["player_presence"] >= 0.99
        and aggregate["coin_presence"] >= 0.97
        and aggregate["enemy_presence"] >= 0.97
        and aggregate["action_following"] >= 0.90
        and aggregate["event_consistency"] >= 0.95
        and aggregate["health_accuracy"] >= 0.95
        and aggregate["progress_accuracy"] >= 0.95
        and aggregate["terminal_accuracy"] >= 0.95
    )
    aggregate["gate_60s"] = bool(
        args.seconds == 60
        and aggregate["no_collapse_rate"] >= 22 / 24
        and aggregate["object_presence"] >= 0.95
        and aggregate["fps"] >= 12.0
    )
    (out_dir / f"{args.model_type}_{args.seconds}s_metrics.json").write_text(
        json.dumps({"aggregate": aggregate, "rollouts": rollout_records}, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
