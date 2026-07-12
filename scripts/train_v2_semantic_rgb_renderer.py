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
from src.model_v2 import SemanticRGBRenderer, SemanticSpriteRenderer
from src.training_v2 import append_jsonl, capture_rng_state, cosine_with_warmup, set_seed


def renderer_loss(prediction: torch.Tensor, target: torch.Tensor):
    pixel = F.l1_loss(prediction, target)
    objects = object_balanced_l1(prediction, target)
    edges = edge_l1(prediction, target)
    segmentation = soft_object_segmentation_loss(prediction, target)
    total = pixel + 4.0 * objects + 0.35 * edges + 0.25 * segmentation
    return total, {"pixel": pixel, "objects": objects, "edge": edges, "segmentation": segmentation}


def initialize_base(renderer: SemanticRGBRenderer, dataset: ToyArenaV2SemanticFrameDataset, seed: int) -> None:
    rng = np.random.default_rng(seed)
    candidates = dataset.indices
    if dataset.states.shape[1] >= 23:
        candidates = candidates[dataset.states[candidates, 21:23].max(axis=1) < 0.5]
    selected = rng.choice(candidates, size=min(257, len(candidates)), replace=False)
    frames = np.asarray(dataset.arrays.frames[selected], dtype=np.float32) / 255.0
    median = np.median(frames, axis=0).transpose(2, 0, 1)
    probability = torch.from_numpy(median.copy()).clamp(1e-4, 1.0 - 1e-4)
    with torch.no_grad():
        renderer.base_logits.copy_(torch.logit(probability)[None])


def initialize_sprite_atlas(
    renderer: SemanticSpriteRenderer,
    dataset: ToyArenaV2SemanticFrameDataset,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    normal = np.flatnonzero(dataset.event_flags == 0)
    if dataset.states.shape[1] >= 23:
        normal = normal[dataset.states[dataset.indices[normal], 21:23].max(axis=1) < 0.5]
    selected_items = rng.choice(normal, size=min(512, len(normal)), replace=False)
    selected = dataset.indices[selected_items]
    frames = torch.from_numpy(
        np.transpose(np.asarray(dataset.arrays.frames[selected], dtype=np.float32) / 255.0, (0, 3, 1, 2)).copy()
    )
    states = torch.from_numpy(dataset.states[selected].copy())
    axis = torch.linspace(-1.0, 1.0, renderer.sprite_size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    local = torch.stack([xx, yy], dim=-1)[None]

    def aligned_median(source: torch.Tensor, centers: torch.Tensor, half_extent: float) -> torch.Tensor:
        patches = []
        for start in range(0, len(source), 64):
            frame_batch = source[start : start + 64]
            center_batch = centers[start : start + 64]
            grid = center_batch[:, None, None] + local * half_extent
            patches.append(F.grid_sample(frame_batch, grid, mode="bilinear", align_corners=True))
        return torch.cat(patches).median(dim=0).values.clamp(1e-4, 1.0 - 1e-4)

    player = aligned_median(frames, states[:, 0:2], 11.0 / 64.0)
    coin = aligned_median(frames, states[:, 4:6], 12.0 / 64.0)
    enemy_centers = states[:, 6:12].reshape(-1, 3, 2)
    enemy = aligned_median(
        frames.repeat_interleave(3, dim=0), enemy_centers.reshape(-1, 2), 11.0 / 64.0
    )
    collision_items = np.flatnonzero((dataset.event_flags & 2) > 0)
    if dataset.states.shape[1] >= 23:
        collision_items = collision_items[
            dataset.states[dataset.indices[collision_items], 21:23].max(axis=1) < 0.5
        ]
    collision_items = rng.choice(collision_items, size=min(256, len(collision_items)), replace=False)
    collision_indices = dataset.indices[collision_items]
    collision_frames = torch.from_numpy(
        np.transpose(
            np.asarray(dataset.arrays.frames[collision_indices], dtype=np.float32) / 255.0,
            (0, 3, 1, 2),
        ).copy()
    )
    collision_states = torch.from_numpy(dataset.states[collision_indices].copy())
    flash = aligned_median(collision_frames, collision_states[:, 0:2], 15.0 / 64.0)
    with torch.no_grad():
        renderer.player_sprite[:, 0:3].copy_(torch.logit(player)[None])
        renderer.coin_sprite[:, 0:3].copy_(torch.logit(coin)[None])
        renderer.enemy_sprite[:, 0:3].copy_(torch.logit(enemy)[None])
        renderer.flash_sprite[:, 0:3].copy_(torch.logit(flash)[None])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train direct semantic-state to RGB renderer.")
    parser.add_argument("--data", default="data/toy_arena_v21_60k")
    parser.add_argument("--semantic-cache", default="data/toy_arena_v21_60k/semantic_cache")
    parser.add_argument("--out-dir", default="runs/v21_semantic_rgb_renderer")
    parser.add_argument("--architecture", choices=("conv", "sprite"), default="conv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=12000)
    parser.add_argument("--val-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    arrays = load_toy_arena_v2(args.data)
    states, semantic_metadata = load_v2_semantic_states(args.semantic_cache, arrays)
    state_dim = int(states.shape[1])
    train_set = ToyArenaV2SemanticFrameDataset(arrays, states, None, "train", args.max_samples, args.seed)
    val_limit = max(32, args.max_samples // 4) if args.max_samples is not None else args.val_samples
    val_set = ToyArenaV2SemanticFrameDataset(arrays, states, None, "val", val_limit, args.seed + 1)
    train_samples = min(len(train_set), args.samples_per_epoch)
    sampler = StratifiedEventSampler(
        train_set, num_samples=train_samples, seed=args.seed, allow_missing=args.max_samples is not None
    )
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=use_amp)
    train_loader = DataLoader(train_set, sampler=sampler, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    renderer = (
        SemanticRGBRenderer(state_dim)
        if args.architecture == "conv"
        else SemanticSpriteRenderer(state_dim)
    ).to(device)
    initialize_base(renderer, train_set, args.seed)
    if args.architecture == "sprite":
        initialize_sprite_atlas(renderer, train_set, args.seed)
        renderer.base_logits.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in renderer.parameters() if parameter.requires_grad],
            lr=args.lr,
            weight_decay=0.0,
        )
    else:
        optimizer = torch.optim.AdamW(renderer.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    updates = math.ceil(math.ceil(train_samples / args.batch_size) / args.grad_accum) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_with_warmup(step, updates, max(10, int(updates * 0.05)))
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        renderer.train()
        optimizer.zero_grad(set_to_none=True)
        train_totals: dict[str, float] = {}
        for batch_index, batch in enumerate(tqdm(train_loader, desc=f"rgb renderer train {epoch}"), 1):
            state = batch["state"].to(device, non_blocking=True)
            target = batch["target_frame"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                prediction = renderer(state)
                loss, components = renderer_loss(prediction, target)
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
        with torch.inference_mode():
            for batch in tqdm(val_loader, desc=f"rgb renderer val {epoch}"):
                state = batch["state"].to(device)
                target = batch["target_frame"].to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    prediction = renderer(state)
                    loss, components = renderer_loss(prediction, target)
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
        best = min(best, record["val_loss"])
        checkpoint = {
            "model_type": f"v2_semantic_{args.architecture}_renderer",
            "model": renderer.state_dict(),
            "state_dim": state_dim,
            "feature_hw": getattr(renderer, "feature_hw", None),
            "output_hw": renderer.output_hw,
            "sprite_size": getattr(renderer, "sprite_size", None),
            "epoch": epoch,
            "best": best,
            "metrics": record,
            "data": args.data,
            "semantic_cache": args.semantic_cache,
            "semantic_cache_probe": semantic_metadata["probes"],
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
