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

from src.data_v2 import ToyArenaV2FramePairDataset, load_toy_arena_v2
from src.eval_v2 import aggregate_metric_records, foreground_mse_tensor, probe_batch_metrics
from src.model_v2 import ArenaStateProbe, V2RepresentationCodec
from src.training_v2 import load_trusted_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Toy Arena V2 codec reconstruction gates.")
    parser.add_argument("--data", default="data/toy_arena_v2_60k")
    parser.add_argument("--codec", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arrays = load_toy_arena_v2(args.data)
    dataset = ToyArenaV2FramePairDataset(arrays, "test", args.max_samples, 1337)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    codec_checkpoint = load_trusted_checkpoint(args.codec, map_location=device)
    codec = V2RepresentationCodec(
        int(codec_checkpoint["latent_channels"]), int(codec_checkpoint.get("semantic_dim", 0))
    ).to(device)
    codec.load_state_dict(codec_checkpoint["model"])
    codec.eval()
    probe_checkpoint = load_trusted_checkpoint(args.probes, map_location=device)
    probe = ArenaStateProbe().to(device)
    probe.load_state_dict(probe_checkpoint["state_probe"])
    probe.eval()

    records = []
    preview_frames = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="codec eval"):
            frames = batch["frame"].to(device)
            next_frames = batch["next_frame"].to(device)
            reconstruction = codec(frames)[0]
            next_reconstruction = codec(next_frames)[0]
            logits = probe(reconstruction)
            state = probe_batch_metrics(
                logits,
                batch["player_pos"].to(device),
                batch["coin_pos"].to(device),
                batch["enemy_pos"].to(device),
            )
            record = {
                "l1": float(F.l1_loss(reconstruction, frames).item()),
                "foreground_mse": float(foreground_mse_tensor(reconstruction, frames).item()),
                "temporal_l1": float(
                    F.l1_loss(next_reconstruction - reconstruction, next_frames - frames).item()
                ),
                **{key: float(value.item()) for key, value in state.items()},
            }
            records.append(record)
            if len(preview_frames) < 8:
                real = np.clip(frames[0].permute(1, 2, 0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
                recon = np.clip(reconstruction[0].permute(1, 2, 0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
                preview_frames.append(np.concatenate([real, recon], axis=1))

    metrics = aggregate_metric_records(records)
    recalls = np.mean([metrics["player_recall"], metrics["coin_recall"], metrics["enemy_recall"]])
    median_centroid = float(np.median([metrics["player_error"], metrics["coin_error"], metrics["enemy_error"]]))
    metrics["median_centroid_error"] = median_centroid
    metrics["semantic_composite"] = float(
        0.40 * recalls
        + 0.30 * np.exp(-median_centroid / 1.5)
        + 0.20 * np.exp(-metrics["l1"] / 0.04)
        + 0.10 * np.exp(-metrics["temporal_l1"] / 0.02)
    )
    metrics["gate_pass"] = bool(
        metrics["player_recall"] >= 0.99
        and metrics["coin_recall"] >= 0.99
        and metrics["enemy_recall"] >= 0.98
        and median_centroid <= 1.5
        and metrics["foreground_mse"] <= 0.04
    )
    out_dir = Path(args.out_dir or Path(args.codec).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "codec_eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if preview_frames:
        imageio.imwrite(out_dir / "codec_eval_contact_sheet.png", np.concatenate(preview_frames, axis=0))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
