from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_v2 import (
    SEMANTIC_STATE_DIM,
    StratifiedEventSampler,
    ToyArenaV2SemanticFrameDataset,
    load_toy_arena_v2,
    load_v2_semantic_states,
)
from src.losses_v2 import edge_l1, object_balanced_l1, soft_object_segmentation_loss
from src.model_v2 import SemanticLatentRenderer, V2RepresentationCodec
from src.training_v2 import (
    append_jsonl,
    capture_rng_state,
    cosine_with_warmup,
    load_trusted_checkpoint,
    restore_rng_state,
    set_seed,
)


def decode_normalized(
    codec: V2RepresentationCodec,
    normalized: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return codec.decode(normalized * std + mean)


def renderer_loss(
    predicted_latent: torch.Tensor,
    target_latent: torch.Tensor,
    target_frame: torch.Tensor,
    codec: V2RepresentationCodec,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    latent = F.smooth_l1_loss(predicted_latent, target_latent)
    predicted_frame = decode_normalized(codec, predicted_latent, mean, std)
    pixel = F.l1_loss(predicted_frame, target_frame)
    objects = object_balanced_l1(predicted_frame, target_frame)
    edges = edge_l1(predicted_frame, target_frame)
    segmentation = soft_object_segmentation_loss(predicted_frame, target_frame)
    total = latent + pixel + 2.0 * objects + 0.10 * edges + 0.10 * segmentation
    return total, {
        "latent": latent,
        "pixel": pixel,
        "objects": objects,
        "edge": edges,
        "segmentation": segmentation,
    }


def initialize_base_latent(
    renderer: SemanticLatentRenderer,
    dataset: ToyArenaV2SemanticFrameDataset,
    sample_count: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    count = min(sample_count, len(dataset))
    selected = rng.choice(dataset.indices, size=count, replace=False)
    total = np.zeros(dataset.latents.shape[1:], dtype=np.float64)
    for start in range(0, count, 256):
        total += np.asarray(dataset.latents[selected[start : start + 256]], dtype=np.float32).sum(axis=0)
    with torch.no_grad():
        renderer.base_latent.copy_(torch.from_numpy((total / count).astype(np.float32))[None])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train semantic-state to codec-latent renderer.")
    parser.add_argument("--data", default="data/toy_arena_v2_60k")
    parser.add_argument("--semantic-cache", default="data/toy_arena_v2_60k/semantic_cache")
    parser.add_argument("--latent-cache", default="data/toy_arena_v2_60k/latent_cache")
    parser.add_argument("--codec", required=True)
    parser.add_argument("--out-dir", default="runs/v2_semantic_renderer_60k")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=12000)
    parser.add_argument("--val-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    arrays = load_toy_arena_v2(args.data)
    states, semantic_metadata = load_v2_semantic_states(args.semantic_cache, arrays)
    train_set = ToyArenaV2SemanticFrameDataset(
        arrays, states, args.latent_cache, "train", args.max_samples, args.seed
    )
    val_max = max(32, args.max_samples // 4) if args.max_samples is not None else args.val_samples
    val_set = ToyArenaV2SemanticFrameDataset(
        arrays, states, args.latent_cache, "val", val_max, args.seed + 1
    )
    train_samples = min(len(train_set), args.samples_per_epoch)
    sampler = StratifiedEventSampler(
        train_set,
        num_samples=train_samples,
        seed=args.seed,
        allow_missing=args.max_samples is not None,
    )
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=use_amp)
    train_loader = DataLoader(train_set, sampler=sampler, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    codec_checkpoint = load_trusted_checkpoint(args.codec, map_location=device)
    latent_channels = int(codec_checkpoint["latent_channels"])
    codec = V2RepresentationCodec(latent_channels, int(codec_checkpoint.get("semantic_dim", 0))).to(device)
    codec.load_state_dict(codec_checkpoint["model"])
    codec.eval()
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    cache_metadata = json.loads((Path(args.latent_cache) / "metadata.json").read_text(encoding="utf-8"))
    mean = torch.tensor(cache_metadata["mean"], device=device)[None, :, None, None]
    std = torch.tensor(cache_metadata["std"], device=device)[None, :, None, None]
    renderer = SemanticLatentRenderer(SEMANTIC_STATE_DIM, latent_channels).to(device)
    if not args.resume:
        initialize_base_latent(renderer, train_set, 2048, args.seed)
    optimizer = torch.optim.AdamW(renderer.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    updates_per_epoch = math.ceil(math.ceil(train_samples / args.batch_size) / args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_with_warmup(step, total_updates, max(10, int(total_updates * 0.05))),
    )
    start_epoch = 1
    best = float("inf")
    if args.resume:
        checkpoint = load_trusted_checkpoint(args.resume, map_location=device)
        if checkpoint["dataset_manifest"] != arrays.metadata["manifest_hash"]:
            raise ValueError("resume checkpoint dataset mismatch")
        renderer.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("best", float("inf")))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs + 1):
        sampler.set_epoch(epoch)
        renderer.train()
        optimizer.zero_grad(set_to_none=True)
        train_totals: dict[str, float] = {}
        for batch_index, batch in enumerate(tqdm(train_loader, desc=f"renderer train {epoch}"), start=1):
            state = batch["state"].to(device, non_blocking=True)
            target_latent = batch["target_latent"].to(device, non_blocking=True)
            target_frame = batch["target_frame"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                prediction = renderer(state)
                loss, components = renderer_loss(prediction, target_latent, target_frame, codec, mean, std)
            scaler.scale(loss / args.grad_accum).backward()
            if batch_index % args.grad_accum == 0 or batch_index == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(renderer.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            train_totals["loss"] = train_totals.get("loss", 0.0) + float(loss.item()) * len(state)
            for key, value in components.items():
                train_totals[key] = train_totals.get(key, 0.0) + float(value.item()) * len(state)

        renderer.eval()
        val_totals: dict[str, float] = {}
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"renderer val {epoch}"):
                state = batch["state"].to(device, non_blocking=True)
                target_latent = batch["target_latent"].to(device, non_blocking=True)
                target_frame = batch["target_frame"].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    prediction = renderer(state)
                    loss, components = renderer_loss(prediction, target_latent, target_frame, codec, mean, std)
                val_totals["loss"] = val_totals.get("loss", 0.0) + float(loss.item()) * len(state)
                for key, value in components.items():
                    val_totals[key] = val_totals.get(key, 0.0) + float(value.item()) * len(state)

        record = {
            "epoch": epoch,
            "train_samples": train_samples,
            "val_samples": len(val_set),
            "lr": scheduler.get_last_lr()[0],
            **{f"train_{key}": value / train_samples for key, value in train_totals.items()},
            **{f"val_{key}": value / len(val_set) for key, value in val_totals.items()},
        }
        append_jsonl(out_dir / "metrics.jsonl", record)
        print(json.dumps(record, indent=2))
        is_best = record["val_loss"] < best
        if is_best:
            best = record["val_loss"]
        checkpoint = {
            "model_type": "v2_semantic_latent_renderer",
            "model": renderer.state_dict(),
            "state_dim": SEMANTIC_STATE_DIM,
            "latent_channels": latent_channels,
            "epoch": epoch,
            "best": best,
            "metrics": record,
            "data": args.data,
            "semantic_cache": args.semantic_cache,
            "semantic_cache_probe": semantic_metadata["probes"],
            "latent_cache": args.latent_cache,
            "codec": args.codec,
            "dataset_manifest": arrays.metadata["manifest_hash"],
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_state": capture_rng_state(),
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if is_best:
            torch.save(checkpoint, out_dir / "best.pt")


if __name__ == "__main__":
    main()
