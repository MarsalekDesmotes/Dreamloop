from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_v2 import ToyArenaV2FramePairDataset, load_toy_arena_v2
from src.eval_v2 import probe_batch_metrics
from src.losses_v2 import gaussian_heatmaps
from src.model_v2 import ArenaStateProbe, InverseDynamicsProbe, V2RepresentationCodec
from src.training_v2 import append_jsonl, capture_rng_state, load_trusted_checkpoint, set_seed


def heatmap_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive_weight = logits.new_tensor(18.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight)
    return bce + 0.25 * F.mse_loss(torch.sigmoid(logits), target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen state and inverse-dynamics probes for Toy Arena V2.")
    parser.add_argument("--data", default="data/toy_arena_v2_60k")
    parser.add_argument("--out-dir", default="runs/v2_probes_60k")
    parser.add_argument("--codec", default=None, help="Optional codec checkpoint used for 50% reconstruction augmentation.")
    parser.add_argument("--state-checkpoint", default=None, help="Optional probe checkpoint used to initialize state weights.")
    parser.add_argument("--freeze-state", action="store_true")
    parser.add_argument("--inverse-checkpoint", default=None, help="Optional probe checkpoint used to initialize inverse weights.")
    parser.add_argument("--freeze-inverse", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--state-resolution", type=int, choices=(32, 64), default=32)
    parser.add_argument("--terminal-fraction", type=float, default=0.0)
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
    if not 0.0 <= args.terminal_fraction < 1.0:
        raise ValueError("terminal fraction must be in [0, 1)")
    sampler = None
    if args.terminal_fraction > 0.0:
        terminal = np.asarray(arrays.game_status[train_set.indices]) != 0
        terminal_count = int(terminal.sum())
        running_count = len(terminal) - terminal_count
        if terminal_count == 0 or running_count == 0:
            raise ValueError("terminal oversampling requires both running and terminal samples")
        weights = np.where(
            terminal,
            args.terminal_fraction / terminal_count,
            (1.0 - args.terminal_fraction) / running_count,
        )
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights.astype(np.float64)),
            num_samples=len(train_set),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    train_loader = DataLoader(train_set, shuffle=sampler is None, sampler=sampler, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    state_probe = ArenaStateProbe(args.state_resolution).to(device)
    inverse_probe = InverseDynamicsProbe(action_count=int(arrays.metadata["action_count"])).to(device)
    if args.state_checkpoint:
        state_checkpoint = load_trusted_checkpoint(args.state_checkpoint, map_location=device)
        state_probe.load_state_dict(state_checkpoint["state_probe"])
    if args.freeze_state:
        if not args.state_checkpoint:
            raise ValueError("--freeze-state requires --state-checkpoint")
        state_probe.eval()
        for parameter in state_probe.parameters():
            parameter.requires_grad_(False)
    if args.inverse_checkpoint:
        inverse_checkpoint = load_trusted_checkpoint(args.inverse_checkpoint, map_location=device)
        inverse_probe.load_state_dict(inverse_checkpoint["inverse_probe"])
    if args.freeze_inverse:
        if not args.inverse_checkpoint:
            raise ValueError("--freeze-inverse requires --inverse-checkpoint")
        inverse_probe.eval()
        for parameter in inverse_probe.parameters():
            parameter.requires_grad_(False)
    codec = None
    if args.codec:
        checkpoint = load_trusted_checkpoint(args.codec, map_location=device)
        codec = V2RepresentationCodec(
            latent_channels=int(checkpoint["latent_channels"]), semantic_dim=int(checkpoint.get("semantic_dim", 0))
        ).to(device)
        codec.load_state_dict(checkpoint["model"])
        codec.eval()
        for parameter in codec.parameters():
            parameter.requires_grad_(False)

    trainable_parameters = [parameter for parameter in state_probe.parameters() if parameter.requires_grad]
    trainable_parameters += [parameter for parameter in inverse_probe.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("at least one probe must remain trainable")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score = float("inf")
    metrics_path = out_dir / "metrics.jsonl"

    for epoch in range(1, args.epochs + 1):
        state_probe.train(not args.freeze_state)
        inverse_probe.train(not args.freeze_inverse)
        train_total = 0.0
        for batch in tqdm(train_loader, desc=f"probe train {epoch}"):
            previous_frame = batch["previous_frame"].to(device, non_blocking=True)
            frame = batch["frame"].to(device, non_blocking=True)
            next_frame = batch["next_frame"].to(device, non_blocking=True)
            player = batch["player_pos"].to(device, non_blocking=True)
            coin = batch["coin_pos"].to(device, non_blocking=True)
            enemy = batch["enemy_pos"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)
            state_frame = frame
            if codec is not None and torch.rand(()) < 0.5:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    state_frame = codec(frame)[0]

            target_heatmaps = gaussian_heatmaps(
                player, coin, enemy, size=args.state_resolution, sigma=1.25 * args.state_resolution / 32
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                state_logits = state_probe(state_frame)
                inverse_logits = inverse_probe(previous_frame, frame, next_frame)
                state_loss = heatmap_loss(state_logits, target_heatmaps)
                inverse_loss = F.cross_entropy(inverse_logits, action)
                if args.freeze_state:
                    loss = inverse_loss
                elif args.freeze_inverse:
                    loss = state_loss
                else:
                    loss = state_loss + inverse_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.item()) * len(frame)

        state_probe.eval()
        inverse_probe.eval()
        val_total = 0.0
        accuracy_total = 0.0
        probe_totals: dict[str, float] = {}
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"probe val {epoch}"):
                previous_frame = batch["previous_frame"].to(device, non_blocking=True)
                frame = batch["frame"].to(device, non_blocking=True)
                next_frame = batch["next_frame"].to(device, non_blocking=True)
                player = batch["player_pos"].to(device, non_blocking=True)
                coin = batch["coin_pos"].to(device, non_blocking=True)
                enemy = batch["enemy_pos"].to(device, non_blocking=True)
                action = batch["action"].to(device, non_blocking=True)
                target_heatmaps = gaussian_heatmaps(
                    player, coin, enemy, size=args.state_resolution, sigma=1.25 * args.state_resolution / 32
                )
                state_frame = codec(frame)[0] if codec is not None else frame
                state_logits = state_probe(state_frame)
                inverse_logits = inverse_probe(previous_frame, frame, next_frame)
                loss = heatmap_loss(state_logits, target_heatmaps) + F.cross_entropy(inverse_logits, action)
                metrics = probe_batch_metrics(state_logits, player, coin, enemy)
                val_total += float(loss.item()) * len(frame)
                accuracy_total += float((inverse_logits.argmax(dim=1) == action).float().sum().item())
                for key, value in metrics.items():
                    probe_totals[key] = probe_totals.get(key, 0.0) + float(value.item()) * len(frame)

        record = {
            "epoch": epoch,
            "train_loss": train_total / len(train_set),
            "val_loss": val_total / len(val_set),
            "inverse_accuracy": accuracy_total / len(val_set),
            **{key: value / len(val_set) for key, value in probe_totals.items()},
        }
        if args.freeze_inverse:
            selection_score = (
                record["player_error"]
                + record["coin_error"]
                + record["enemy_error"]
                + 10.0 * (1.0 - record["player_recall"])
                + 10.0 * (1.0 - record["coin_recall"])
                + 10.0 * (1.0 - record["enemy_recall"])
            )
        elif args.freeze_state:
            selection_score = 1.0 - record["inverse_accuracy"]
        else:
            selection_score = record["val_loss"]
        record["selection_score"] = selection_score
        append_jsonl(metrics_path, record)
        print(json.dumps(record, indent=2))
        checkpoint = {
            "model_type": "v2_probes",
            "state_probe": state_probe.state_dict(),
            "inverse_probe": inverse_probe.state_dict(),
            "action_count": int(arrays.metadata["action_count"]),
            "state_resolution": args.state_resolution,
            "epoch": epoch,
            "metrics": record,
            "data": args.data,
            "dataset_manifest": arrays.metadata["manifest_hash"],
            "codec": args.codec,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_state": capture_rng_state(),
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if selection_score < best_score:
            best_score = selection_score
            torch.save(checkpoint, out_dir / "best.pt")


if __name__ == "__main__":
    main()
