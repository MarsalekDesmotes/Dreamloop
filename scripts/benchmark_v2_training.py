from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_v2_dynamics import dynamics_loss, latent_stats, rollout_segment
from src.data_v2 import ToyArenaV2LatentSequenceDataset, load_toy_arena_v2
from src.model_v2 import InverseDynamicsProbe, StreamingLatentDynamics, V2RepresentationCodec
from src.training_v2 import load_trusted_checkpoint, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the exact V2 dynamics training workload.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")
    set_seed(1337)
    device = "cuda"
    arrays = load_toy_arena_v2(args.data)
    dataset = ToyArenaV2LatentSequenceDataset(
        arrays,
        args.cache,
        "train",
        args.context,
        args.horizon,
        max_samples=args.steps * args.batch_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    codec_checkpoint = load_trusted_checkpoint(args.codec, map_location=device)
    codec = V2RepresentationCodec(
        int(codec_checkpoint["latent_channels"]), int(codec_checkpoint.get("semantic_dim", 0))
    ).to(device)
    codec.load_state_dict(codec_checkpoint["model"])
    codec.eval()
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    probe_checkpoint = load_trusted_checkpoint(args.probes, map_location=device)
    inverse = InverseDynamicsProbe(int(arrays.metadata["action_count"])).to(device)
    inverse.load_state_dict(probe_checkpoint["inverse_probe"])
    inverse.eval()
    for parameter in inverse.parameters():
        parameter.requires_grad_(False)
    model = StreamingLatentDynamics(int(arrays.metadata["action_count"]), int(codec_checkpoint["latent_channels"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    mean, std = latent_stats(dataset.cache_metadata, device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    completed = 0
    for batch in loader:
        context = batch["context_latents"].to(device, non_blocking=True)
        context_actions = batch["context_actions"].to(device, non_blocking=True)
        targets = batch["target_latents"].to(device, non_blocking=True)
        actions = batch["future_actions"].to(device, non_blocking=True)
        target_frames = batch["target_frames"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            current, hidden = model.prefill(context, context_actions)
            predictions, _, _ = rollout_segment(model, current, hidden, actions, targets, 0.75)
            loss, _ = dynamics_loss(predictions, targets, current, actions, target_frames, codec, inverse, mean, std, 2)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        completed += 1
        if completed >= args.steps:
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "steps": completed,
        "batch_size": args.batch_size,
        "seconds_per_batch": elapsed / completed,
        "samples_per_second": completed * args.batch_size / elapsed,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

