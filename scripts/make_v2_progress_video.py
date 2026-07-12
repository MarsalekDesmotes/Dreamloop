from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_v2_world_model import eval_action, initialize_context
from src.model_v2 import V2RepresentationCodec
from src.runtime_v2 import LatentWorldRuntime, RGBWorldRuntime, SemanticWorldRuntime, frames_tensor
from src.toy_arena_v2 import ToyArenaV2
from src.training_v2 import load_trusted_checkpoint


def label(frame: np.ndarray, text: str) -> np.ndarray:
    canvas = np.full((frame.shape[0] + 16, frame.shape[1], 3), (10, 13, 18), dtype=np.uint8)
    canvas[16:] = frame
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.text((4, 3), text, fill=(240, 244, 250), font=ImageFont.load_default())
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the final four-panel Toy Arena V2 progress video.")
    parser.add_argument("--data", default="data/toy_arena_v2_60k")
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--latent", default=None)
    parser.add_argument("--semantic", default=None)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--out", default="runs/v2_final_progress_4panel.mp4")
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=4041)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if bool(args.latent) == bool(args.semantic):
        raise ValueError("provide exactly one of --latent or --semantic")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    rgb = RGBWorldRuntime(args.rgb, device)
    world = SemanticWorldRuntime(args.semantic, device) if args.semantic else LatentWorldRuntime(args.latent, device)
    checkpoint = load_trusted_checkpoint(args.codec, map_location=device)
    codec = V2RepresentationCodec(int(checkpoint["latent_channels"]), int(checkpoint.get("semantic_dim", 0))).to(device)
    codec.load_state_dict(checkpoint["model"])
    codec.eval()
    seed = args.seed

    def reset_world(current_seed: int):
        env = ToyArenaV2(current_seed)
        context = max(world.context, rgb.context)
        context_frames, context_actions = initialize_context(env, context)
        world.initialize(context_frames[-world.context :], context_actions[-(world.context - 1) :])
        rgb.initialize(context_frames[-rgb.context :], context_actions[-(rgb.context - 1) :])
        return env

    env = reset_world(seed)
    rng = np.random.default_rng(args.seed ^ 0xBEEF)
    output = []
    terminal_frames = 0
    with torch.inference_mode():
        for step in range(args.seconds * args.fps):
            action = eval_action(env, rng, 1, step)
            real, _ = env.step(action)
            rgb_frame = rgb.step(action)
            world_frame = world.step(action)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                reconstructed = codec(frames_tensor([real], device))[0][0]
            codec_frame = np.clip(reconstructed.float().permute(1, 2, 0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            top = np.concatenate([label(real, "ENGINE GAME"), label(codec_frame, "CODEC RECONSTRUCTION")], axis=1)
            final_label = "FINAL GUARDED HYBRID" if args.semantic else "LATENT WORLD MODEL"
            bottom = np.concatenate([label(rgb_frame, "EARLY DIRECT RGB"), label(world_frame, final_label)], axis=1)
            output.append(np.concatenate([top, bottom], axis=0))
            terminal_frames = terminal_frames + 1 if env.state.game_status != 0 else 0
            if terminal_frames >= 18:
                seed += 1
                env = reset_world(seed)
                terminal_frames = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, output, fps=args.fps)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
