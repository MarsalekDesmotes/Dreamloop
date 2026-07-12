from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_v2_semantic_renderer import decode_normalized
from src.data_v2 import ToyArenaV2SemanticFrameDataset, load_toy_arena_v2, load_v2_semantic_states
from src.eval_v2 import aggregate_metric_records, foreground_mse_tensor, probe_batch_metrics
from src.model_v2 import (
    ArenaStateProbe,
    SemanticLatentRenderer,
    SemanticRGBRenderer,
    SemanticSpriteRenderer,
    V2RepresentationCodec,
    arena_state_probe_from_checkpoint,
)
from src.training_v2 import load_trusted_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate semantic neural renderer reconstruction.")
    parser.add_argument("--data", default="data/toy_arena_v2_60k")
    parser.add_argument("--semantic-cache", default="data/toy_arena_v2_60k/semantic_cache")
    parser.add_argument("--latent-cache", default="data/toy_arena_v2_60k/latent_cache")
    parser.add_argument("--renderer", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=2048)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arrays = load_toy_arena_v2(args.data)
    states, _ = load_v2_semantic_states(args.semantic_cache, arrays)
    renderer_checkpoint = load_trusted_checkpoint(args.renderer, map_location=device)
    renderer_type = renderer_checkpoint.get("model_type")
    direct_rgb = renderer_type in ("v2_semantic_rgb_renderer", "v2_semantic_conv_renderer", "v2_semantic_sprite_renderer")
    dataset = ToyArenaV2SemanticFrameDataset(
        arrays, states, None if direct_rgb else args.latent_cache, "test", args.max_samples, 1337
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    if renderer_type == "v2_semantic_sprite_renderer":
        renderer = SemanticSpriteRenderer(
            int(renderer_checkpoint["state_dim"]),
            int(renderer_checkpoint["output_hw"]),
            int(renderer_checkpoint["sprite_size"]),
        ).to(device)
    elif direct_rgb:
        renderer = SemanticRGBRenderer(
            int(renderer_checkpoint["state_dim"]),
            int(renderer_checkpoint["feature_hw"]),
            int(renderer_checkpoint["output_hw"]),
        ).to(device)
    else:
        renderer = SemanticLatentRenderer(
            int(renderer_checkpoint["state_dim"]), int(renderer_checkpoint["latent_channels"])
        ).to(device)
    renderer.load_state_dict(renderer_checkpoint["model"])
    renderer.eval()
    codec = None
    mean = None
    std = None
    if not direct_rgb:
        codec_checkpoint = load_trusted_checkpoint(renderer_checkpoint["codec"], map_location=device)
        codec = V2RepresentationCodec(
            int(codec_checkpoint["latent_channels"]), int(codec_checkpoint.get("semantic_dim", 0))
        ).to(device)
        codec.load_state_dict(codec_checkpoint["model"])
        codec.eval()
        latent_metadata = json.loads((Path(args.latent_cache) / "metadata.json").read_text(encoding="utf-8"))
        mean = torch.tensor(latent_metadata["mean"], device=device)[None, :, None, None]
        std = torch.tensor(latent_metadata["std"], device=device)[None, :, None, None]
    probe_checkpoint = load_trusted_checkpoint(args.probes, map_location=device)
    probe = arena_state_probe_from_checkpoint(probe_checkpoint).to(device)
    probe.load_state_dict(probe_checkpoint["state_probe"])
    probe.eval()

    records = []
    previews = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="semantic renderer eval"):
            state = batch["state"].to(device)
            target = batch["target_frame"].to(device)
            prediction = renderer(state) if direct_rgb else decode_normalized(codec, renderer(state), mean, std)
            indices = batch["index"].numpy()
            metrics = probe_batch_metrics(
                probe(prediction),
                torch.from_numpy(np.asarray(arrays.player_pos[indices], dtype=np.float32)).to(device),
                torch.from_numpy(np.asarray(arrays.coin_pos[indices], dtype=np.float32)).to(device),
                torch.from_numpy(np.asarray(arrays.enemy_pos[indices], dtype=np.float32)).to(device),
            )
            records.append(
                {
                    "l1": float(F.l1_loss(prediction, target).item()),
                    "foreground_mse": float(foreground_mse_tensor(prediction, target).item()),
                    **{key: float(value.item()) for key, value in metrics.items()},
                }
            )
            if len(previews) < 8:
                real = np.clip(target[0].permute(1, 2, 0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
                generated = np.clip(prediction[0].permute(1, 2, 0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
                previews.append(np.concatenate([real, generated], axis=1))

    result = aggregate_metric_records(records)
    median = float(np.median([result["player_error"], result["coin_error"], result["enemy_error"]]))
    result["median_centroid_error"] = median
    result["gate_pass"] = bool(
        result["player_recall"] >= 0.99
        and result["coin_recall"] >= 0.99
        and result["enemy_recall"] >= 0.98
        and median <= 1.5
        and result["foreground_mse"] <= 0.04
    )
    out_dir = Path(args.out_dir or Path(args.renderer).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "semantic_renderer_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if previews:
        imageio.imwrite(out_dir / "semantic_renderer_contact_sheet.png", np.concatenate(previews, axis=0))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
