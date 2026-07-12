from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_v2 import load_toy_arena_v2
from src.model_v2 import V2RepresentationCodec
from src.training_v2 import checkpoint_sha256, load_trusted_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized FP16 latent cache for Toy Arena V2.")
    parser.add_argument("--data", default="data/toy_arena_v2_60k")
    parser.add_argument("--codec", required=True)
    parser.add_argument("--out", default="data/toy_arena_v2_60k/latent_cache")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    arrays = load_toy_arena_v2(args.data)
    checkpoint = load_trusted_checkpoint(args.codec, map_location=device)
    codec = V2RepresentationCodec(
        latent_channels=int(checkpoint["latent_channels"]), semantic_dim=int(checkpoint.get("semantic_dim", 0))
    ).to(device)
    codec.load_state_dict(checkpoint["model"])
    codec.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    latent_path = out / "latents.npy"
    if latent_path.exists() and not args.overwrite:
        raise FileExistsError(f"{latent_path} exists; pass --overwrite")

    count = len(arrays.frames)
    channels = int(checkpoint["latent_channels"])
    cache = open_memmap(latent_path, mode="w+", dtype=np.float16, shape=(count, channels, 16, 16))
    channel_sum = np.zeros(channels, dtype=np.float64)
    channel_square_sum = np.zeros(channels, dtype=np.float64)
    values_per_channel = count * 16 * 16
    with torch.no_grad():
        for start in tqdm(range(0, count, args.batch_size), desc="encode latent cache"):
            end = min(start + args.batch_size, count)
            values = np.asarray(arrays.frames[start:end], dtype=np.float32) / 255.0
            frames = torch.from_numpy(np.transpose(values, (0, 3, 1, 2)).copy()).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                latents = codec.encode(frames).float()
            cpu = latents.cpu().numpy()
            cache[start:end] = cpu.astype(np.float16)
            channel_sum += cpu.sum(axis=(0, 2, 3), dtype=np.float64)
            channel_square_sum += np.square(cpu, dtype=np.float64).sum(axis=(0, 2, 3))
    cache.flush()
    mean = channel_sum / values_per_channel
    variance = channel_square_sum / values_per_channel - mean**2
    std = np.sqrt(np.maximum(variance, 1e-6))
    for start in tqdm(range(0, count, args.batch_size), desc="normalize latent cache"):
        end = min(start + args.batch_size, count)
        values = np.asarray(cache[start:end], dtype=np.float32)
        values = (values - mean[None, :, None, None]) / std[None, :, None, None]
        cache[start:end] = values.astype(np.float16)
    cache.flush()
    metadata = {
        "version": 1,
        "data": args.data,
        "dataset_manifest": arrays.metadata["manifest_hash"],
        "codec": args.codec,
        "codec_sha256": checkpoint_sha256(args.codec),
        "shape": list(cache.shape),
        "dtype": "float16",
        "normalized": True,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {latent_path}")


if __name__ == "__main__":
    main()
