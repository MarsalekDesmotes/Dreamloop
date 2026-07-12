from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_v2 import StratifiedEventSampler, ToyArenaV2SequenceDataset, load_toy_arena_v2
from src.losses_v2 import edge_l1, object_balanced_l1
from src.model_v2 import V2RGBNextFrame
from src.training_v2 import (
    append_jsonl,
    capture_rng_state,
    cosine_with_warmup,
    load_trusted_checkpoint,
    restore_rng_state,
    set_seed,
)


@dataclass(frozen=True)
class Stage:
    name: str
    horizon: int
    epochs: int
    predicted_probability: float


def rollout_rgb(
    model: V2RGBNextFrame,
    history: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    predicted_probability: float,
) -> torch.Tensor:
    predictions = []
    for step in range(actions.shape[1]):
        flat = history.flatten(1, 2)
        prediction = model(flat, actions[:, step])
        predictions.append(prediction)
        if predicted_probability <= 0.0:
            next_frame = targets[:, step]
        elif predicted_probability >= 1.0:
            next_frame = prediction
        else:
            mask = torch.rand(len(prediction), 1, 1, 1, device=prediction.device) < predicted_probability
            next_frame = torch.where(mask, prediction, targets[:, step])
        history = torch.cat([history[:, 1:], next_frame[:, None]], dim=1)
    return torch.stack(predictions, dim=1)


def rgb_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = prediction.flatten(0, 1)
    truth = target.flatten(0, 1)
    return F.l1_loss(pred, truth) + object_balanced_l1(pred, truth) + 0.10 * edge_l1(pred, truth)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a direct-RGB Toy Arena V2 baseline.")
    parser.add_argument("--data", default="data/toy_arena_v2_60k")
    parser.add_argument("--out-dir", default="runs/v2_rgb_baseline_60k")
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--teacher-epochs", type=int, default=5)
    parser.add_argument("--rollout-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--samples-per-epoch", type=int, default=12000)
    parser.add_argument("--val-samples", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    stages = [Stage("teacher", 1, args.teacher_epochs, 0.0), Stage("rollout", 8, args.rollout_epochs, 1.0)]
    stages = [stage for stage in stages if stage.epochs > 0]
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    arrays = load_toy_arena_v2(args.data)
    model = V2RGBNextFrame(int(arrays.metadata["action_count"]), args.context).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    estimated_batches = math.ceil(min(args.samples_per_epoch, args.max_samples or args.samples_per_epoch) / args.batch_size)
    total_steps = math.ceil(estimated_batches / args.grad_accum) * sum(stage.epochs for stage in stages)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_with_warmup(step, total_steps, max(10, int(total_steps * 0.05)))
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_rollout = float("inf")
    start_stage = 0
    start_epoch = 1
    if args.resume:
        checkpoint = load_trusted_checkpoint(args.resume, map_location=device)
        if checkpoint["dataset_manifest"] != arrays.metadata["manifest_hash"]:
            raise ValueError("resume checkpoint dataset mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng_state"])
        start_stage = int(checkpoint.get("stage_index", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_rollout = float(checkpoint.get("best_rollout", float("inf")))
        if start_epoch > stages[start_stage].epochs:
            start_stage += 1
            start_epoch = 1

    for stage_index in range(start_stage, len(stages)):
        stage = stages[stage_index]
        train_set = ToyArenaV2SequenceDataset(
            arrays, "train", args.context, stage.horizon, args.max_samples, args.seed
        )
        val_set = ToyArenaV2SequenceDataset(
            arrays,
            "val",
            args.context,
            stage.horizon,
            (
                max(32, args.max_samples // 4)
                if args.max_samples is not None
                else (None if args.val_samples <= 0 else args.val_samples)
            ),
            args.seed + 1,
        )
        num_samples = min(len(train_set), args.samples_per_epoch)
        sampler = StratifiedEventSampler(
            train_set,
            num_samples=num_samples,
            seed=args.seed + stage_index * 100,
            allow_missing=args.max_samples is not None,
        )
        train_loader_args = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=use_amp,
            persistent_workers=args.num_workers > 0,
        )
        train_loader = DataLoader(train_set, sampler=sampler, **train_loader_args)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_amp)

        first_epoch = start_epoch if stage_index == start_stage else 1
        for epoch in range(first_epoch, stage.epochs + 1):
            sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_total = 0.0
            for batch_index, batch in enumerate(tqdm(train_loader, desc=f"rgb {stage.name} train {epoch}"), start=1):
                history = batch["context_frames"].to(device, non_blocking=True)
                targets = batch["target_frames"].to(device, non_blocking=True)
                actions = batch["future_actions"].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    predictions = rollout_rgb(model, history, actions, targets, stage.predicted_probability)
                    loss = rgb_loss(predictions, targets)
                scaler.scale(loss / args.grad_accum).backward()
                if batch_index % args.grad_accum == 0 or batch_index == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                train_total += float(loss.item()) * len(history)

            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"rgb {stage.name} val {epoch}"):
                    history = batch["context_frames"].to(device, non_blocking=True)
                    targets = batch["target_frames"].to(device, non_blocking=True)
                    actions = batch["future_actions"].to(device, non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        predictions = rollout_rgb(model, history, actions, targets, 1.0)
                        loss = rgb_loss(predictions, targets)
                    val_total += float(loss.item()) * len(history)
            record = {
                "stage": stage.name,
                "epoch": epoch,
                "train_loss": train_total / num_samples,
                "val_loss": val_total / len(val_set),
                "lr": scheduler.get_last_lr()[0],
            }
            append_jsonl(out_dir / "metrics.jsonl", record)
            print(json.dumps(record, indent=2))
            checkpoint = {
                "model_type": "v2_rgb_next_frame",
                "model": model.state_dict(),
                "context": args.context,
                "action_count": int(arrays.metadata["action_count"]),
                "stage": stage.name,
                "stage_index": stage_index,
                "epoch": epoch,
                "best_rollout": best_rollout,
                "metrics": record,
                "data": args.data,
                "dataset_manifest": arrays.metadata["manifest_hash"],
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng_state": capture_rng_state(),
            }
            if stage.name == "rollout" and record["val_loss"] < best_rollout:
                best_rollout = record["val_loss"]
                checkpoint["best_rollout"] = best_rollout
                torch.save(checkpoint, out_dir / "best.pt")
            elif len(stages) == 1 and record["val_loss"] < best_rollout:
                best_rollout = record["val_loss"]
                checkpoint["best_rollout"] = best_rollout
                torch.save(checkpoint, out_dir / "best.pt")
            torch.save(checkpoint, out_dir / "last.pt")


if __name__ == "__main__":
    main()
