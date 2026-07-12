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

from src.data_v2 import (
    SEMANTIC_STATE_DIM,
    StratifiedEventSampler,
    ToyArenaV2SemanticSequenceDataset,
    load_toy_arena_v2,
    load_v2_semantic_states,
)
from src.model_v2 import NeuralSemanticStateDynamics, StructuredSemanticStateDynamics
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
    truncate_every: int


STATE_WEIGHTS = torch.tensor(
    [4, 4, 1, 1, 4, 4, 2, 2, 2, 2, 2, 2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 3, 3, 4, 4],
    dtype=torch.float32,
)


def rollout_segment(
    model,
    current: torch.Tensor,
    hidden: tuple[torch.Tensor, torch.Tensor],
    actions: torch.Tensor,
    targets: torch.Tensor,
    predicted_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
    predictions = []
    coin_gate_logits = []
    collision_gate_logits = []
    for step in range(actions.shape[1]):
        prediction, hidden = model.step(current, actions[:, step], hidden)
        predictions.append(prediction)
        if hasattr(model, "last_auxiliary"):
            coin_gate_logits.append(model.last_auxiliary["coin_gate_logits"])
            collision_gate_logits.append(model.last_auxiliary["collision_gate_logits"])
        if predicted_probability <= 0.0:
            current = targets[:, step]
        elif predicted_probability >= 1.0:
            current = prediction
        else:
            use_prediction = torch.rand(len(prediction), 1, device=prediction.device) < predicted_probability
            current = torch.where(use_prediction, prediction, targets[:, step])
    auxiliary = {}
    if coin_gate_logits:
        auxiliary = {
            "coin_gate_logits": torch.stack(coin_gate_logits, dim=1),
            "collision_gate_logits": torch.stack(collision_gate_logits, dim=1),
        }
    return torch.stack(predictions, dim=1), current, hidden, auxiliary


def semantic_state_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    start_state: torch.Tensor,
    target_start_state: torch.Tensor,
    auxiliary: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = STATE_WEIGHTS[: predictions.shape[-1]].to(device=predictions.device, dtype=predictions.dtype)
    per_value = F.smooth_l1_loss(predictions, targets, reduction="none")
    state = (per_value * weights).sum(dim=-1).mean() / weights.sum()
    previous = torch.cat([start_state[:, None], predictions[:, :-1]], dim=1)
    target_previous = torch.cat([target_start_state[:, None], targets[:, :-1]], dim=1)

    # Position deltas are small in normalized coordinates. Compare scaled local
    # motion against ground truth so a stationary prediction cannot minimize the
    # dynamics objective merely by also predicting zero velocity.
    player_motion = (predictions[:, :, 0:2] - previous[:, :, 0:2]) * 8.0
    target_player_motion = (targets[:, :, 0:2] - target_previous[:, :, 0:2]) * 8.0
    enemy_motion = (predictions[:, :, 6:12] - previous[:, :, 6:12]) * 16.0
    target_enemy_motion = (targets[:, :, 6:12] - target_previous[:, :, 6:12]) * 16.0
    kinematic = F.smooth_l1_loss(player_motion, target_player_motion)
    kinematic = kinematic + F.smooth_l1_loss(predictions[:, :, 2:4], targets[:, :, 2:4])
    kinematic = kinematic + F.smooth_l1_loss(enemy_motion, target_enemy_motion)
    kinematic = kinematic + F.smooth_l1_loss(predictions[:, :, 12:18], targets[:, :, 12:18])

    coin_motion = predictions[:, :, 4:6] - previous[:, :, 4:6]
    target_coin_motion = targets[:, :, 4:6] - target_previous[:, :, 4:6]
    transition = F.smooth_l1_loss(coin_motion * 4.0, target_coin_motion * 4.0)
    flash_probability = predictions[:, :, 18].clamp(1e-4, 1.0 - 1e-4)
    flash = -(
        targets[:, :, 18] * torch.log(flash_probability)
        + (1.0 - targets[:, :, 18]) * torch.log1p(-flash_probability)
    ).mean()
    previous_target = torch.cat([target_start_state[:, None, 4:6], targets[:, :-1, 4:6]], dim=1)
    coin_changed = (torch.linalg.vector_norm(targets[:, :, 4:6] - previous_target, dim=-1) > 0.10).to(predictions.dtype)
    positives = coin_changed.sum().clamp_min(1.0)
    negatives = coin_changed.numel() - positives
    coin_gate = predictions.new_zeros(())
    coin_gate_accuracy = predictions.new_ones(())
    previous_flash = torch.cat([target_start_state[:, None, 18], targets[:, :-1, 18]], dim=1)
    collision_started = ((targets[:, :, 18] > 0.5) & (previous_flash <= 0.5)).to(predictions.dtype)
    collision_positives = collision_started.sum().clamp_min(1.0)
    collision_negatives = collision_started.numel() - collision_positives
    collision_gate = predictions.new_zeros(())
    collision_gate_accuracy = predictions.new_ones(())
    if auxiliary:
        coin_logits = auxiliary["coin_gate_logits"].squeeze(-1)
        coin_gate = F.binary_cross_entropy_with_logits(
            coin_logits, coin_changed, pos_weight=(negatives / positives).clamp(1.0, 40.0)
        )
        coin_gate_accuracy = ((coin_logits >= 0) == coin_changed.bool()).float().mean()
        collision_logits = auxiliary["collision_gate_logits"].squeeze(-1)
        collision_gate = F.binary_cross_entropy_with_logits(
            collision_logits,
            collision_started,
            pos_weight=(collision_negatives / collision_positives).clamp(1.0, 60.0),
        )
        collision_gate_accuracy = ((collision_logits >= 0) == collision_started.bool()).float().mean()
    gameplay = predictions.new_zeros(())
    terminal_accuracy = predictions.new_ones(())
    if predictions.shape[-1] >= 23:
        gameplay = F.smooth_l1_loss(predictions[:, :, 19:21], targets[:, :, 19:21])
        terminal_probability = predictions[:, :, 21:23].clamp(1e-4, 1.0 - 1e-4)
        terminal_target = targets[:, :, 21:23]
        terminal = -(
            terminal_target * torch.log(terminal_probability)
            + (1.0 - terminal_target) * torch.log1p(-terminal_probability)
        ).mean()
        gameplay = gameplay + terminal
        terminal_accuracy = ((terminal_probability >= 0.5) == (targets[:, :, 21:23] >= 0.5)).float().mean()
    total = (
        state
        + 0.50 * kinematic
        + 0.10 * transition
        + 0.05 * flash
        + 0.15 * coin_gate
        + 0.15 * collision_gate
        + 0.20 * gameplay
    )
    return total, {
        "state": state,
        "kinematic": kinematic,
        "transition": transition,
        "flash": flash,
        "coin_gate": coin_gate,
        "coin_gate_accuracy": coin_gate_accuracy,
        "collision_gate": collision_gate,
        "collision_gate_accuracy": collision_gate_accuracy,
        "gameplay": gameplay,
        "terminal_accuracy": terminal_accuracy,
        "player_error_px": torch.linalg.vector_norm(predictions[:, :, 0:2] - targets[:, :, 0:2], dim=-1).mean() * 64,
        "coin_error_px": torch.linalg.vector_norm(predictions[:, :, 4:6] - targets[:, :, 4:6], dim=-1).mean() * 64,
        "enemy_error_px": (
            torch.linalg.vector_norm(
                predictions[:, :, 6:12].reshape(*predictions.shape[:2], 3, 2)
                - targets[:, :, 6:12].reshape(*targets.shape[:2], 3, 2),
                dim=-1,
            ).mean()
            * 64
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train streaming semantic-state Toy Arena V2 dynamics.")
    parser.add_argument("--data", default="data/toy_arena_v2_60k")
    parser.add_argument("--semantic-cache", default="data/toy_arena_v2_60k/semantic_cache")
    parser.add_argument("--renderer", required=True)
    parser.add_argument("--probes", default=None)
    parser.add_argument("--architecture", choices=("structured", "neural"), default="structured")
    parser.add_argument("--out-dir", default="runs/v2_semantic_dynamics_60k")
    parser.add_argument("--context", type=int, default=24)
    parser.add_argument("--teacher-epochs", type=int, default=5)
    parser.add_argument("--scheduled-epochs", type=int, default=10)
    parser.add_argument("--closed-epochs", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=20000)
    parser.add_argument("--val-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-noise", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    stages = [
        Stage("teacher", 8, args.teacher_epochs, 0.0, 8),
        Stage("scheduled", 32, args.scheduled_epochs, 0.75, 16),
        Stage("closed", 64, args.closed_epochs, 1.0, 16),
    ]
    stages = [stage for stage in stages if stage.epochs > 0]
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    arrays = load_toy_arena_v2(args.data)
    states, semantic_metadata = load_v2_semantic_states(args.semantic_cache, arrays)
    state_dim = int(states.shape[1])
    val_max = max(32, args.max_samples // 4) if args.max_samples is not None else args.val_samples
    datasets = {
        (stage.name, split): ToyArenaV2SemanticSequenceDataset(
            arrays,
            states,
            split,
            args.context,
            stage.horizon,
            args.max_samples if split == "train" else val_max,
            args.seed + (0 if split == "train" else 1),
        )
        for stage in stages
        for split in ("train", "val")
    }
    sample_counts = {
        stage.name: min(len(datasets[(stage.name, "train")]), args.samples_per_epoch) for stage in stages
    }
    updates = {
        stage.name: math.ceil(math.ceil(sample_counts[stage.name] / args.batch_size) / args.grad_accum)
        for stage in stages
    }
    model = (
        NeuralSemanticStateDynamics(int(arrays.metadata["action_count"]), state_dim)
        if args.architecture == "neural"
        else StructuredSemanticStateDynamics(int(arrays.metadata["action_count"]), state_dim)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_stage = 0
    start_epoch = 1
    stage_best: dict[str, float] = {}
    resume_scheduler_state = None
    if args.resume:
        checkpoint = load_trusted_checkpoint(args.resume, map_location=device)
        if checkpoint["dataset_manifest"] != arrays.metadata["manifest_hash"]:
            raise ValueError("resume checkpoint dataset mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        resume_scheduler_state = checkpoint["scheduler"]
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng_state"])
        start_stage = int(checkpoint["stage_index"])
        start_epoch = int(checkpoint["stage_epoch"]) + 1
        stage_best = dict(checkpoint.get("stage_best", {}))
        if start_epoch > stages[start_stage].epochs:
            start_stage += 1
            start_epoch = 1
            resume_scheduler_state = None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stage_index in range(start_stage, len(stages)):
        stage = stages[stage_index]
        train_set = datasets[(stage.name, "train")]
        val_set = datasets[(stage.name, "val")]
        sample_count = sample_counts[stage.name]
        sampler = StratifiedEventSampler(
            train_set,
            num_samples=sample_count,
            seed=args.seed + stage_index * 100,
            allow_missing=args.max_samples is not None,
        )
        loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=use_amp)
        train_loader = DataLoader(train_set, sampler=sampler, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, **loader_args)
        for group in optimizer.param_groups:
            group["lr"] = args.lr
            group["initial_lr"] = args.lr
        stage_updates = updates[stage.name] * stage.epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: cosine_with_warmup(step, stage_updates, max(5, int(stage_updates * 0.05))),
        )
        if stage_index == start_stage and resume_scheduler_state is not None:
            scheduler.load_state_dict(resume_scheduler_state)
        first_epoch = start_epoch if stage_index == start_stage else 1
        best = float(stage_best.get(stage.name, float("inf")))
        for epoch in range(first_epoch, stage.epochs + 1):
            sampler.set_epoch(epoch)
            if stage.name == "scheduled":
                probability = stage.predicted_probability * (epoch - 1) / max(stage.epochs - 1, 1)
            else:
                probability = stage.predicted_probability
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_totals: dict[str, float] = {}
            for batch_index, batch in enumerate(tqdm(train_loader, desc=f"semantic {stage.name} train {epoch}"), 1):
                context = batch["context_states"].to(device, non_blocking=True)
                context_actions = batch["context_actions"].to(device, non_blocking=True)
                targets = batch["target_states"].to(device, non_blocking=True)
                actions = batch["future_actions"].to(device, non_blocking=True)
                noisy_context = context + torch.randn_like(context) * (torch.rand((), device=device) * args.max_noise)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    current, hidden = model.prefill(noisy_context, context_actions)
                segment_count = math.ceil(stage.horizon / stage.truncate_every)
                batch_metrics: dict[str, float] = {}
                for start in range(0, stage.horizon, stage.truncate_every):
                    end = min(start + stage.truncate_every, stage.horizon)
                    segment_start = current
                    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        predictions, current, hidden, auxiliary = rollout_segment(
                            model, current, hidden, actions[:, start:end], targets[:, start:end], probability
                        )
                        target_start = context[:, -1] if start == 0 else targets[:, start - 1]
                        loss, components = semantic_state_loss(
                            predictions, targets[:, start:end], segment_start, target_start, auxiliary
                        )
                    scaler.scale(loss / (args.grad_accum * segment_count)).backward()
                    batch_metrics["loss"] = batch_metrics.get("loss", 0.0) + float(loss.item()) / segment_count
                    for key, value in components.items():
                        batch_metrics[key] = batch_metrics.get(key, 0.0) + float(value.item()) / segment_count
                    if end < stage.horizon:
                        current = current.detach()
                        hidden = model.detach_hidden(hidden)
                if batch_index % args.grad_accum == 0 or batch_index == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                for key, value in batch_metrics.items():
                    train_totals[key] = train_totals.get(key, 0.0) + value * len(context)

            model.eval()
            val_totals: dict[str, float] = {}
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"semantic {stage.name} val {epoch}"):
                    context = batch["context_states"].to(device)
                    context_actions = batch["context_actions"].to(device)
                    targets = batch["target_states"].to(device)
                    actions = batch["future_actions"].to(device)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        current, hidden = model.prefill(context, context_actions)
                        predictions, _, _, auxiliary = rollout_segment(model, current, hidden, actions, targets, 1.0)
                        loss, components = semantic_state_loss(
                            predictions, targets, current, context[:, -1], auxiliary
                        )
                    val_totals["loss"] = val_totals.get("loss", 0.0) + float(loss.item()) * len(context)
                    for key, value in components.items():
                        val_totals[key] = val_totals.get(key, 0.0) + float(value.item()) * len(context)

            record = {
                "stage": stage.name,
                "stage_index": stage_index,
                "epoch": epoch,
                "predicted_probability": probability,
                "train_samples": sample_count,
                "val_samples": len(val_set),
                "lr": scheduler.get_last_lr()[0],
                **{f"train_{key}": value / sample_count for key, value in train_totals.items()},
                **{f"val_{key}": value / len(val_set) for key, value in val_totals.items()},
            }
            append_jsonl(out_dir / "metrics.jsonl", record)
            print(json.dumps(record, indent=2))
            is_best = record["val_loss"] < best
            if is_best:
                best = record["val_loss"]
                stage_best[stage.name] = best
            checkpoint = {
                "model_type": f"v12_{args.architecture}_semantic_state_dynamics",
                "model": model.state_dict(),
                "state_dim": state_dim,
                "kinematic_base": args.architecture == "structured",
                "structured_v3": args.architecture == "structured",
                "neural_v4": args.architecture == "neural",
                "action_count": int(arrays.metadata["action_count"]),
                "context": args.context,
                "stage": stage.name,
                "stage_index": stage_index,
                "stage_epoch": epoch,
                "stage_best": stage_best,
                "metrics": record,
                "data": args.data,
                "semantic_cache": args.semantic_cache,
                "semantic_cache_probe": args.probes or semantic_metadata["probes"],
                "renderer": args.renderer,
                "dataset_manifest": arrays.metadata["manifest_hash"],
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng_state": capture_rng_state(),
            }
            torch.save(checkpoint, out_dir / "last.pt")
            if is_best:
                torch.save(checkpoint, out_dir / f"best_{stage.name}.pt")
                if stage.name == "closed" or stage_index == len(stages) - 1:
                    torch.save(checkpoint, out_dir / "best.pt")


if __name__ == "__main__":
    main()
