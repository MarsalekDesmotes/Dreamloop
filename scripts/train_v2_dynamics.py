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

from src.data_v2 import StratifiedEventSampler, ToyArenaV2LatentSequenceDataset, load_toy_arena_v2
from src.losses_v2 import edge_l1, object_balanced_l1, soft_object_segmentation_loss
from src.model_v2 import ArenaStateProbe, InverseDynamicsProbe, StreamingLatentDynamics, V2RepresentationCodec
from src.training_v2 import (
    append_jsonl,
    capture_rng_state,
    checkpoint_sha256,
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
    max_predicted_probability: float
    truncate_every: int


def latent_stats(cache_metadata: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(cache_metadata["mean"], device=device, dtype=torch.float32)[None, :, None, None]
    std = torch.tensor(cache_metadata["std"], device=device, dtype=torch.float32)[None, :, None, None]
    return mean, std


def decode_normalized(
    codec: V2RepresentationCodec,
    normalized: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    shape = normalized.shape
    flat = normalized.reshape(-1, shape[-3], shape[-2], shape[-1])
    decoded = codec.decode(flat * std + mean)
    return decoded.reshape(*shape[:-3], *decoded.shape[-3:])


def rollout_segment(
    model: StreamingLatentDynamics,
    current: torch.Tensor,
    hidden: tuple[torch.Tensor, torch.Tensor],
    actions: torch.Tensor,
    targets: torch.Tensor,
    predicted_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    predictions = []
    for step in range(actions.shape[1]):
        prediction, hidden = model.step(current, actions[:, step], hidden)
        predictions.append(prediction)
        if predicted_probability <= 0.0:
            current = targets[:, step]
        elif predicted_probability >= 1.0:
            current = prediction
        else:
            use_prediction = torch.rand(len(prediction), 1, 1, 1, device=prediction.device) < predicted_probability
            current = torch.where(use_prediction, prediction, targets[:, step])
    return torch.stack(predictions, dim=1), current, hidden


def dynamics_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    start_latent: torch.Tensor,
    actions: torch.Tensor,
    target_frames: torch.Tensor,
    codec: V2RepresentationCodec,
    inverse_probe: InverseDynamicsProbe,
    mean: torch.Tensor,
    std: torch.Tensor,
    decoded_stride: int,
    decoded_weight: float = 0.50,
    velocity_weight: float = 0.25,
    action_weight: float = 0.10,
    object_boost: float = 1.0,
    edge_weight: float = 0.0,
    pixel_velocity_weight: float = 0.0,
    state_probe: ArenaStateProbe | None = None,
    state_probe_weight: float = 0.0,
    state_probe_map_weight: float = 0.0,
    segmentation_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    latent = F.smooth_l1_loss(predictions, targets)
    pred_velocity = torch.diff(torch.cat([start_latent[:, None], predictions], dim=1), dim=1)
    target_velocity = torch.diff(torch.cat([start_latent[:, None], targets], dim=1), dim=1)
    velocity = F.smooth_l1_loss(pred_velocity, target_velocity)

    decoded = decode_normalized(codec, predictions, mean, std)
    selected = torch.arange(0, predictions.shape[1], decoded_stride, device=predictions.device)
    pred_selected = decoded.index_select(1, selected).flatten(0, 1)
    target_selected = target_frames.index_select(1, selected).flatten(0, 1)
    decoded_pixel = F.l1_loss(pred_selected, target_selected)
    decoded_objects = object_balanced_l1(pred_selected, target_selected)
    decoded_l1 = decoded_pixel + object_boost * decoded_objects
    edges = edge_l1(pred_selected, target_selected)
    pixel_velocity = F.l1_loss(
        decoded[:, 1:] - decoded[:, :-1],
        target_frames[:, 1:] - target_frames[:, :-1],
    )

    start_frame = decode_normalized(codec, start_latent, mean, std)
    previous = torch.cat([start_frame[:, None], decoded[:, :-2]], dim=1).flatten(0, 1)
    before = decoded[:, :-1].flatten(0, 1)
    after = decoded[:, 1:].flatten(0, 1)
    inverse_logits = inverse_probe(previous, before, after)
    action_labels = actions[:, 1:].flatten()
    action_loss = F.cross_entropy(inverse_logits, action_labels)
    probe_kl = predictions.new_zeros(())
    probe_map = predictions.new_zeros(())
    segmentation = predictions.new_zeros(())
    if state_probe is not None and (state_probe_weight > 0.0 or state_probe_map_weight > 0.0):
        predicted_probe_logits = state_probe(pred_selected)
        with torch.no_grad():
            target_probe_logits = state_probe(target_selected)
        probe_kl = F.kl_div(
            F.log_softmax(predicted_probe_logits.flatten(2), dim=-1),
            F.softmax(target_probe_logits.flatten(2), dim=-1),
            reduction="batchmean",
        ) / predicted_probe_logits.shape[1]
        probe_map = F.mse_loss(torch.sigmoid(predicted_probe_logits), torch.sigmoid(target_probe_logits))
    if segmentation_weight > 0.0:
        segmentation = soft_object_segmentation_loss(pred_selected, target_selected)
    total = (
        latent
        + decoded_weight * decoded_l1
        + velocity_weight * velocity
        + action_weight * action_loss
        + edge_weight * edges
        + pixel_velocity_weight * pixel_velocity
        + state_probe_weight * probe_kl
        + state_probe_map_weight * probe_map
        + segmentation_weight * segmentation
    )
    return total, {
        "latent": latent,
        "decoded": decoded_l1,
        "decoded_pixel": decoded_pixel,
        "decoded_objects": decoded_objects,
        "edge": edges,
        "pixel_velocity": pixel_velocity,
        "state_probe_kl": probe_kl,
        "state_probe_map": probe_map,
        "segmentation": segmentation,
        "velocity": velocity,
        "action": action_loss,
        "action_accuracy": (inverse_logits.argmax(dim=1) == action_labels).float().mean(),
    }


def stage_probability(stage: Stage, epoch: int) -> float:
    if stage.max_predicted_probability in (0.0, 1.0) or stage.epochs <= 1:
        return stage.max_predicted_probability
    return stage.max_predicted_probability * (epoch - 1) / (stage.epochs - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train streaming Toy Arena V2 latent dynamics.")
    parser.add_argument("--data", default="data/toy_arena_v2_60k")
    parser.add_argument("--cache", default="data/toy_arena_v2_60k/latent_cache")
    parser.add_argument("--codec", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--out-dir", default="runs/v2_dynamics_60k")
    parser.add_argument("--context", type=int, default=24)
    parser.add_argument("--teacher-epochs", type=int, default=5)
    parser.add_argument("--scheduled-epochs", type=int, default=10)
    parser.add_argument("--closed-epochs", type=int, default=5)
    parser.add_argument("--closed-horizon", type=int, default=32)
    parser.add_argument("--truncate-every", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decoded-stride", type=int, default=2)
    parser.add_argument("--decoded-weight", type=float, default=0.50)
    parser.add_argument("--velocity-weight", type=float, default=0.25)
    parser.add_argument("--action-weight", type=float, default=0.10)
    parser.add_argument("--object-boost", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=0.0)
    parser.add_argument("--pixel-velocity-weight", type=float, default=0.0)
    parser.add_argument("--state-probe-weight", type=float, default=0.0)
    parser.add_argument("--state-probe-map-weight", type=float, default=0.0)
    parser.add_argument("--segmentation-weight", type=float, default=0.0)
    parser.add_argument("--max-noise", type=float, default=0.10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--episode-id",
        type=int,
        default=None,
        help="Restrict train and validation windows to one training episode for the overfit gate.",
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=12000,
        help="Stratified training windows drawn per epoch; use 0 for every valid window.",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=2048,
        help="Deterministic validation windows per stage; use 0 for the full validation split.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-checkpoint", default=None, help="Initialize model weights without restoring optimizer state.")
    args = parser.parse_args()
    if args.resume and args.init_checkpoint:
        raise ValueError("resume and init-checkpoint are mutually exclusive")
    if args.closed_horizon < 2 or args.truncate_every < 2:
        raise ValueError("closed-horizon and truncate-every must be at least 2")

    stages = [
        Stage("teacher", 8, args.teacher_epochs, 0.0, 8),
        Stage("scheduled", 16, args.scheduled_epochs, 0.75, 16),
        Stage("closed", args.closed_horizon, args.closed_epochs, 1.0, args.truncate_every),
    ]
    stages = [stage for stage in stages if stage.epochs > 0]
    if not stages:
        raise ValueError("at least one stage must contain epochs")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    arrays = load_toy_arena_v2(args.data)
    if args.episode_id is not None:
        if args.episode_id < 0 or args.episode_id >= len(arrays.episode_splits):
            raise ValueError("episode-id is outside the dataset")
        if int(arrays.episode_splits[args.episode_id]) != 0:
            raise ValueError("overfit episode must belong to the training split")
    codec_checkpoint = load_trusted_checkpoint(args.codec, map_location=device)
    codec = V2RepresentationCodec(
        int(codec_checkpoint["latent_channels"]), int(codec_checkpoint.get("semantic_dim", 0))
    ).to(device)
    codec.load_state_dict(codec_checkpoint["model"])
    codec.eval()
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    probe_checkpoint = load_trusted_checkpoint(args.probes, map_location=device)
    inverse_probe = InverseDynamicsProbe(int(arrays.metadata["action_count"])).to(device)
    inverse_probe.load_state_dict(probe_checkpoint["inverse_probe"])
    inverse_probe.eval()
    for parameter in inverse_probe.parameters():
        parameter.requires_grad_(False)
    state_probe = ArenaStateProbe().to(device)
    state_probe.load_state_dict(probe_checkpoint["state_probe"])
    state_probe.eval()
    for parameter in state_probe.parameters():
        parameter.requires_grad_(False)

    cache_metadata = json.loads((Path(args.cache) / "metadata.json").read_text(encoding="utf-8"))
    if cache_metadata["codec_sha256"] != checkpoint_sha256(args.codec):
        raise ValueError("latent cache was not built from the selected codec")
    mean, std = latent_stats(cache_metadata, device)
    latent_channels = int(codec_checkpoint["latent_channels"])
    model = StreamingLatentDynamics(
        action_count=int(arrays.metadata["action_count"]), latent_channels=latent_channels
    ).to(device)
    if args.init_checkpoint:
        initial = load_trusted_checkpoint(args.init_checkpoint, map_location=device)
        if initial.get("dataset_manifest") != arrays.metadata["manifest_hash"]:
            raise ValueError("initial checkpoint dataset mismatch")
        model.load_state_dict(initial["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if args.max_samples is not None:
        val_max_samples = max(32, args.max_samples // 4)
    else:
        val_max_samples = None if args.val_samples <= 0 else args.val_samples
    datasets = {
        (stage.name, split): ToyArenaV2LatentSequenceDataset(
            arrays,
            args.cache,
            split="train" if args.episode_id is not None else split,
            context=args.context,
            horizon=stage.horizon,
            max_samples=args.max_samples if split == "train" else val_max_samples,
            seed=args.seed + (0 if split == "train" else 1),
            episode_ids=None if args.episode_id is None else [args.episode_id],
        )
        for stage in stages
        for split in ("train", "val")
    }
    train_sample_counts = {
        stage.name: (
            len(datasets[(stage.name, "train")])
            if args.samples_per_epoch <= 0
            else min(len(datasets[(stage.name, "train")]), args.samples_per_epoch)
        )
        for stage in stages
    }
    train_steps = {
        stage.name: math.ceil(math.ceil(train_sample_counts[stage.name] / args.batch_size) / args.grad_accum)
        for stage in stages
    }
    total_steps = sum(train_steps[stage.name] * stage.epochs for stage in stages)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_with_warmup(step, total_steps, max(10, int(total_steps * 0.05)))
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    start_stage = 0
    start_epoch = 1
    stage_best: dict[str, float] = {}
    if args.resume:
        checkpoint = load_trusted_checkpoint(args.resume, map_location=device)
        if checkpoint["dataset_manifest"] != arrays.metadata["manifest_hash"]:
            raise ValueError("resume checkpoint dataset mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng_state"])
        start_stage = int(checkpoint["stage_index"])
        start_epoch = int(checkpoint["stage_epoch"]) + 1
        stage_best = dict(checkpoint.get("stage_best", {}))
        if start_epoch > stages[start_stage].epochs:
            start_stage += 1
            start_epoch = 1

    for stage_index in range(start_stage, len(stages)):
        stage = stages[stage_index]
        train_set = datasets[(stage.name, "train")]
        val_set = datasets[(stage.name, "val")]
        train_sample_count = train_sample_counts[stage.name]
        sampler = StratifiedEventSampler(
            train_set,
            num_samples=train_sample_count,
            seed=args.seed + stage_index * 100,
            allow_missing=args.episode_id is not None,
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
        best = float(stage_best.get(stage.name, float("inf")))

        for stage_epoch in range(first_epoch, stage.epochs + 1):
            sampler.set_epoch(stage_epoch)
            predicted_probability = stage_probability(stage, stage_epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_totals: dict[str, float] = {}
            for batch_index, batch in enumerate(tqdm(train_loader, desc=f"{stage.name} train {stage_epoch}"), start=1):
                context = batch["context_latents"].to(device, non_blocking=True)
                context_actions = batch["context_actions"].to(device, non_blocking=True)
                targets = batch["target_latents"].to(device, non_blocking=True)
                actions = batch["future_actions"].to(device, non_blocking=True)
                target_frames = batch["target_frames"].to(device, non_blocking=True)
                noise_scale = torch.rand((), device=device) * args.max_noise
                noisy_context = context + torch.randn_like(context) * noise_scale
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    current, hidden = model.prefill(noisy_context, context_actions)
                segment_count = math.ceil(stage.horizon / stage.truncate_every)
                batch_metrics: dict[str, float] = {}
                for segment_index, start in enumerate(range(0, stage.horizon, stage.truncate_every)):
                    end = min(start + stage.truncate_every, stage.horizon)
                    segment_start = current
                    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        predictions, current, hidden = rollout_segment(
                            model,
                            current,
                            hidden,
                            actions[:, start:end],
                            targets[:, start:end],
                            predicted_probability,
                        )
                        loss, components = dynamics_loss(
                            predictions,
                            targets[:, start:end],
                            segment_start,
                            actions[:, start:end],
                            target_frames[:, start:end],
                            codec,
                            inverse_probe,
                            mean,
                            std,
                            args.decoded_stride,
                            args.decoded_weight,
                            args.velocity_weight,
                            args.action_weight,
                            args.object_boost,
                            args.edge_weight,
                            args.pixel_velocity_weight,
                            state_probe,
                            args.state_probe_weight,
                            args.state_probe_map_weight,
                            args.segmentation_weight,
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
                for batch in tqdm(val_loader, desc=f"{stage.name} val {stage_epoch}"):
                    context = batch["context_latents"].to(device, non_blocking=True)
                    context_actions = batch["context_actions"].to(device, non_blocking=True)
                    targets = batch["target_latents"].to(device, non_blocking=True)
                    actions = batch["future_actions"].to(device, non_blocking=True)
                    target_frames = batch["target_frames"].to(device, non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        current, hidden = model.prefill(context, context_actions)
                        predictions, _, _ = rollout_segment(model, current, hidden, actions, targets, 1.0)
                        loss, components = dynamics_loss(
                            predictions,
                            targets,
                            current,
                            actions,
                            target_frames,
                            codec,
                            inverse_probe,
                            mean,
                            std,
                            args.decoded_stride,
                            args.decoded_weight,
                            args.velocity_weight,
                            args.action_weight,
                            args.object_boost,
                            args.edge_weight,
                            args.pixel_velocity_weight,
                            state_probe,
                            args.state_probe_weight,
                            args.state_probe_map_weight,
                            args.segmentation_weight,
                        )
                    val_totals["loss"] = val_totals.get("loss", 0.0) + float(loss.item()) * len(context)
                    for key, value in components.items():
                        val_totals[key] = val_totals.get(key, 0.0) + float(value.item()) * len(context)

            record = {
                "stage": stage.name,
                "stage_index": stage_index,
                "epoch": stage_epoch,
                "predicted_probability": predicted_probability,
                "lr": scheduler.get_last_lr()[0],
                "train_samples": train_sample_count,
                "val_samples": len(val_set),
                **{f"train_{key}": value / train_sample_count for key, value in train_totals.items()},
                **{f"val_{key}": value / len(val_set) for key, value in val_totals.items()},
            }
            append_jsonl(metrics_path, record)
            print(json.dumps(record, indent=2))
            val_loss = record["val_loss"]
            stage_best[stage.name] = min(best, val_loss)
            checkpoint = {
                "model_type": "v2_streaming_latent_dynamics",
                "model": model.state_dict(),
                "latent_channels": latent_channels,
                "action_count": int(arrays.metadata["action_count"]),
                "context": args.context,
                "stage": stage.name,
                "stage_index": stage_index,
                "stage_epoch": stage_epoch,
                "horizon": stage.horizon,
                "truncate_every": stage.truncate_every,
                "stage_best": stage_best,
                "metrics": record,
                "data": args.data,
                "cache": args.cache,
                "codec": args.codec,
                "probes": args.probes,
                "episode_id": args.episode_id,
                "init_checkpoint": args.init_checkpoint,
                "loss_weights": {
                    "latent": 1.0,
                    "decoded": args.decoded_weight,
                    "velocity": args.velocity_weight,
                    "action": args.action_weight,
                    "object_boost": args.object_boost,
                    "edge": args.edge_weight,
                    "pixel_velocity": args.pixel_velocity_weight,
                    "state_probe": args.state_probe_weight,
                    "state_probe_map": args.state_probe_map_weight,
                    "segmentation": args.segmentation_weight,
                },
                "dataset_manifest": arrays.metadata["manifest_hash"],
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng_state": capture_rng_state(),
            }
            torch.save(checkpoint, out_dir / "last.pt")
            if val_loss < best:
                best = val_loss
                stage_best[stage.name] = val_loss
                torch.save(checkpoint, out_dir / f"best_{stage.name}.pt")
                if stage.name == "closed" or stage_index == len(stages) - 1:
                    torch.save(checkpoint, out_dir / "best.pt")


if __name__ == "__main__":
    main()
