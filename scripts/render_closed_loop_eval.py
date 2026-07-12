from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_toy_arena_npz import choose_action
from scripts.play_world_model import NOOP, ToyArenaRuntime, frames_to_tensor, scheduled_stabilize, stabilize_prediction
from src.data import load_coinrun_npz
from src.model import ActionConditionedNextFrame


def load_model(path: str, device: str) -> tuple[ActionConditionedNextFrame, int]:
    checkpoint = torch.load(path, map_location=device)
    context = int(checkpoint["context"])
    model = ActionConditionedNextFrame(
        action_count=int(checkpoint["action_count"]),
        context=context,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, context


def scripted_actions(steps: int, action_count: int, mode: str, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    if mode == "event":
        raise ValueError("event action mode is stateful and handled inside run_eval.")
    if mode == "random":
        return [int(rng.integers(0, action_count)) for _ in range(steps)]
    if mode == "mixed":
        base = scripted_actions(steps, action_count, "scripted", seed)
        return [int(rng.integers(0, action_count)) if rng.random() < 0.35 else action for action in base]

    pattern = [
        4, 4, 4, 4, 4,  # right
        2, 2, 2, 2,  # down
        3, 3, 3, 3, 3,  # left
        1, 1, 1, 1,  # up
        5,  # dash
        0, 0,
    ]
    actions = [pattern[i % len(pattern)] for i in range(steps)]
    return [a if a < action_count else NOOP for a in actions]


def predict_frame(
    model: ActionConditionedNextFrame,
    history: deque[np.ndarray],
    action: int,
    device: str,
    stabilize: float,
    sharpen: float,
    palette_snap: float,
    foreground_persist: float,
    foreground_threshold: float,
) -> np.ndarray:
    pred = model(frames_to_tensor(history, device), torch.tensor([action], dtype=torch.long, device=device))
    raw = np.clip(pred[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    return stabilize_prediction(
        raw,
        history[-1],
        blend=stabilize,
        sharpen=sharpen,
        palette_snap_amount=palette_snap,
        foreground_persist=foreground_persist,
        foreground_threshold=foreground_threshold,
    )


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


def mse(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float32) / 255.0
    bf = b.astype(np.float32) / 255.0
    return float(np.mean((af - bf) ** 2))


def foreground_mse(a: np.ndarray, b: np.ndarray, threshold: float = 0.18) -> float:
    af = a.astype(np.float32) / 255.0
    bf = b.astype(np.float32) / 255.0
    max_channel = bf.max(axis=2, keepdims=True)
    min_channel = bf.min(axis=2, keepdims=True)
    saturation = max_channel - min_channel
    brightness = bf.mean(axis=2, keepdims=True)
    mask = ((saturation > threshold) | (brightness > 0.48)).astype(np.float32)
    return float(np.sum((af - bf) ** 2 * mask) / max(float(mask.sum() * 3), 1.0))


def run_eval(
    base_model: ActionConditionedNextFrame,
    stable_model: ActionConditionedNextFrame,
    context: int,
    size: int,
    action_count: int,
    steps: int,
    seed: int,
    action_mode: str,
    device: str,
    stabilize: float,
    stabilize_end: float,
    stabilize_ramp_steps: int,
    history_sharpen: float,
    palette_snap: float,
    foreground_persist: float,
    foreground_threshold: float,
    render_frames: bool,
) -> tuple[float, float, float, float, list[np.ndarray]]:
    runtime = ToyArenaRuntime(size=size, seed=seed)
    engine_history = runtime.warmup(context)
    base_history = deque(engine_history, maxlen=context)
    stable_history = deque(engine_history, maxlen=context)

    frames = []
    base_losses = []
    stable_losses = []
    base_fg_losses = []
    stable_fg_losses = []
    separator = np.full((size, 8, 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    action_sequence = [] if action_mode == "event" else scripted_actions(steps, action_count, action_mode, seed)

    with torch.no_grad():
        for step in range(steps):
            if action_mode == "event":
                action = choose_action(runtime.player, runtime.gem, runtime.enemies, rng, "event")
                action = action if action < action_count else NOOP
            else:
                action = action_sequence[step]
            engine = runtime.step(action)
            blend = scheduled_stabilize(stabilize, stabilize_end, stabilize_ramp_steps, step)
            base = predict_frame(
                base_model,
                base_history,
                action,
                device,
                blend,
                history_sharpen,
                palette_snap,
                foreground_persist,
                foreground_threshold,
            )
            stable = predict_frame(
                stable_model,
                stable_history,
                action,
                device,
                blend,
                history_sharpen,
                palette_snap,
                foreground_persist,
                foreground_threshold,
            )

            base_losses.append(mse(base, engine))
            stable_losses.append(mse(stable, engine))
            base_fg_losses.append(foreground_mse(base, engine))
            stable_fg_losses.append(foreground_mse(stable, engine))

            base_history.append(base)
            stable_history.append(stable)

            if render_frames:
                panels = [
                    label_panel(engine, "engine reference"),
                    label_panel(base, "base closed-loop"),
                    label_panel(stable, "rollout-stable closed-loop"),
                ]
                frames.append(np.concatenate([panels[0], separator, panels[1], separator, panels[2]], axis=1))

    return (
        float(np.mean(base_losses)),
        float(np.mean(stable_losses)),
        float(np.mean(base_fg_losses)),
        float(np.mean(stable_fg_losses)),
        frames,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a side-by-side closed-loop comparison.")
    parser.add_argument("--data", default="data/toy_arena_mixed_128_4k.npz")
    parser.add_argument("--base", default="runs/improved/best.pt")
    parser.add_argument("--stable", default="runs/closed_loop_stable_v2/best.pt")
    parser.add_argument("--out", default="runs/closed_loop_eval/base_vs_stable.mp4")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds for aggregate eval. Overrides --seed.")
    parser.add_argument("--action-mode", choices=["scripted", "mixed", "random", "event"], default="scripted")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--metrics-out", default=None, help="Optional JSON path for aggregate metrics.")
    parser.add_argument("--stabilize", type=float, default=0.20)
    parser.add_argument("--stabilize-end", type=float, default=0.35)
    parser.add_argument("--stabilize-ramp-steps", type=int, default=120)
    parser.add_argument("--history-sharpen", type=float, default=0.0)
    parser.add_argument("--palette-snap", type=float, default=0.0)
    parser.add_argument("--foreground-persist", type=float, default=0.12)
    parser.add_argument("--foreground-threshold", type=float, default=0.18)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arrays = load_coinrun_npz(args.data)
    size = int(arrays.frames.shape[1])
    action_count = int(arrays.action_count or arrays.actions.max() + 1)

    base_model, base_context = load_model(args.base, device)
    stable_model, stable_context = load_model(args.stable, device)
    if base_context != stable_context:
        raise ValueError("Base and stable checkpoints must use the same context.")
    context = base_context

    seeds = [int(item.strip()) for item in args.seeds.split(",")] if args.seeds else [args.seed]
    all_base = []
    all_stable = []
    all_base_fg = []
    all_stable_fg = []
    per_seed = []
    frames = []
    for idx, seed in enumerate(seeds):
        base_loss, stable_loss, base_fg_loss, stable_fg_loss, seed_frames = run_eval(
            base_model=base_model,
            stable_model=stable_model,
            context=context,
            size=size,
            action_count=action_count,
            steps=args.steps,
            seed=seed,
            action_mode=args.action_mode,
            device=device,
            stabilize=args.stabilize,
            stabilize_end=args.stabilize if args.stabilize_end is None else args.stabilize_end,
            stabilize_ramp_steps=args.stabilize_ramp_steps,
            history_sharpen=args.history_sharpen,
            palette_snap=args.palette_snap,
            foreground_persist=args.foreground_persist,
            foreground_threshold=args.foreground_threshold,
            render_frames=not args.no_video and idx == 0,
        )
        all_base.append(base_loss)
        all_stable.append(stable_loss)
        all_base_fg.append(base_fg_loss)
        all_stable_fg.append(stable_fg_loss)
        per_seed.append(
            {
                "seed": seed,
                "base_mse": base_loss,
                "stable_mse": stable_loss,
                "base_foreground_mse": base_fg_loss,
                "stable_foreground_mse": stable_fg_loss,
            }
        )
        frames.extend(seed_frames)
        print(
            f"seed={seed} base_closed_loop_mse={base_loss:.6f} stable_closed_loop_mse={stable_loss:.6f} "
            f"base_foreground_mse={base_fg_loss:.6f} stable_foreground_mse={stable_fg_loss:.6f}"
        )

    if frames:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, frames, fps=args.fps)
        print(f"wrote {out}")
    base_mean = float(np.mean(all_base))
    stable_mean = float(np.mean(all_stable))
    base_fg_mean = float(np.mean(all_base_fg))
    stable_fg_mean = float(np.mean(all_stable_fg))
    print(f"base_closed_loop_mse={base_mean:.6f}")
    print(f"stable_closed_loop_mse={stable_mean:.6f}")
    print(f"base_foreground_mse={base_fg_mean:.6f}")
    print(f"stable_foreground_mse={stable_fg_mean:.6f}")
    if args.metrics_out:
        metrics = {
            "base": args.base,
            "stable": args.stable,
            "data": args.data,
            "steps": args.steps,
            "action_mode": args.action_mode,
            "stabilize": args.stabilize,
            "stabilize_end": args.stabilize if args.stabilize_end is None else args.stabilize_end,
            "stabilize_ramp_steps": args.stabilize_ramp_steps,
            "history_sharpen": args.history_sharpen,
            "palette_snap": args.palette_snap,
            "foreground_persist": args.foreground_persist,
            "foreground_threshold": args.foreground_threshold,
            "base_closed_loop_mse": base_mean,
            "stable_closed_loop_mse": stable_mean,
            "base_foreground_mse": base_fg_mean,
            "stable_foreground_mse": stable_fg_mean,
            "relative_improvement": (base_mean - stable_mean) / base_mean if base_mean else 0.0,
            "foreground_relative_improvement": (base_fg_mean - stable_fg_mean) / base_fg_mean if base_fg_mean else 0.0,
            "per_seed": per_seed,
        }
        metrics_out = Path(args.metrics_out)
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"wrote {metrics_out}")


if __name__ == "__main__":
    main()
