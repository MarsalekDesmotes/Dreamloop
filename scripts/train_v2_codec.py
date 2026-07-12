from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_v2 import ToyArenaV2FramePairDataset, load_toy_arena_v2
from src.losses_v2 import codec_reconstruction_loss
from src.model_v2 import V2RepresentationCodec
from src.training_v2 import (
    append_jsonl,
    capture_rng_state,
    cosine_with_warmup,
    load_trusted_checkpoint,
    restore_rng_state,
    set_seed,
)


def load_dino(device: str):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def dino_features(model, frames: torch.Tensor) -> torch.Tensor:
    resized = F.interpolate(frames, size=(224, 224), mode="bilinear", align_corners=False)
    mean = frames.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = frames.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    features = model.forward_features((resized - mean) / std)["x_norm_patchtokens"]
    return features.reshape(len(frames), 16, 16, -1).permute(0, 3, 1, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Toy Arena V2 representation codec.")
    parser.add_argument("--data", default="data/toy_arena_v2_60k")
    parser.add_argument("--out-dir", default="runs/v2_codec_cnn_60k")
    parser.add_argument("--semantic", choices=("none", "dinov2"), default="none")
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_toy_arena_v2(args.data)
    train_set = ToyArenaV2FramePairDataset(arrays, "train", args.max_samples, args.seed)
    val_limit = None if args.max_samples is None else max(32, args.max_samples // 4)
    val_set = ToyArenaV2FramePairDataset(arrays, "val", val_limit, args.seed + 1)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=use_amp)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    semantic_dim = 384 if args.semantic == "dinov2" else 0
    codec = V2RepresentationCodec(args.latent_channels, semantic_dim).to(device)
    dino = load_dino(device) if args.semantic == "dinov2" else None
    optimizer = torch.optim.AdamW(codec.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = optimizer_steps_per_epoch * args.epochs
    warmup_steps = max(10, int(total_steps * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_with_warmup(step, total_steps, warmup_steps)
    )
    start_epoch = 1
    best_val = float("inf")
    if args.resume:
        checkpoint = load_trusted_checkpoint(args.resume, map_location=device)
        if checkpoint["dataset_manifest"] != arrays.metadata["manifest_hash"]:
            raise ValueError("resume checkpoint dataset manifest mismatch")
        codec.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", checkpoint["val_loss"]))

    metrics_path = out_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        codec.train()
        optimizer.zero_grad(set_to_none=True)
        train_total = 0.0
        for batch_index, batch in enumerate(tqdm(train_loader, desc=f"codec train {epoch}"), start=1):
            frame = batch["frame"].to(device, non_blocking=True)
            next_frame = batch["next_frame"].to(device, non_blocking=True)
            combined = torch.cat([frame, next_frame], dim=0)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                reconstruction, latents = codec(combined)
                current_recon, next_recon = reconstruction.chunk(2)
                loss, _ = codec_reconstruction_loss(
                    next_recon,
                    next_frame,
                    previous_prediction=current_recon,
                    previous_target=frame,
                )
                current_loss, _ = codec_reconstruction_loss(current_recon, frame, temporal_weight=0.0)
                loss = 0.5 * (loss + current_loss)
                if dino is not None:
                    with torch.no_grad():
                        semantic_target = dino_features(dino, combined)
                    semantic_pred = codec.project_semantic(latents)
                    semantic_loss = 1.0 - F.cosine_similarity(semantic_pred, semantic_target, dim=1).mean()
                    loss = loss + 0.10 * semantic_loss
            scaler.scale(loss / args.grad_accum).backward()
            if batch_index % args.grad_accum == 0 or batch_index == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            train_total += float(loss.item()) * len(frame)

        codec.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"codec val {epoch}"):
                frame = batch["frame"].to(device, non_blocking=True)
                next_frame = batch["next_frame"].to(device, non_blocking=True)
                combined = torch.cat([frame, next_frame], dim=0)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    reconstruction, latents = codec(combined)
                    current_recon, next_recon = reconstruction.chunk(2)
                    loss, _ = codec_reconstruction_loss(
                        next_recon, next_frame, previous_prediction=current_recon, previous_target=frame
                    )
                    current_loss, _ = codec_reconstruction_loss(current_recon, frame, temporal_weight=0.0)
                    loss = 0.5 * (loss + current_loss)
                    if dino is not None:
                        semantic_target = dino_features(dino, combined)
                        loss = loss + 0.10 * (
                            1.0 - F.cosine_similarity(codec.project_semantic(latents), semantic_target, dim=1).mean()
                        )
                val_total += float(loss.item()) * len(frame)

        train_loss = train_total / len(train_set)
        val_loss = val_total / len(val_set)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": scheduler.get_last_lr()[0]}
        append_jsonl(metrics_path, record)
        print(json.dumps(record, indent=2))
        checkpoint = {
            "model_type": "v2_representation_codec",
            "model": codec.state_dict(),
            "latent_channels": args.latent_channels,
            "semantic": args.semantic,
            "semantic_dim": semantic_dim,
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val": min(best_val, val_loss),
            "data": args.data,
            "dataset_manifest": arrays.metadata["manifest_hash"],
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_state": capture_rng_state(),
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, out_dir / "best.pt")


if __name__ == "__main__":
    main()
