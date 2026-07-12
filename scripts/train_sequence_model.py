from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CoinRunArrays, load_coinrun_npz
from src.model import ActionConditionedSequencePredictor


class SequenceDataset(Dataset):
    def __init__(
        self,
        arrays: CoinRunArrays,
        context: int,
        horizon: int,
        rollout_chunks: int = 1,
        max_samples: int | None = None,
        sample_seed: int = 123,
        index_start: int | None = None,
        index_end: int | None = None,
    ):
        self.arrays = arrays
        self.context = context
        self.horizon = horizon
        self.rollout_chunks = rollout_chunks
        self.total_horizon = horizon * rollout_chunks
        self.index_start = index_start
        self.index_end = index_end
        self.indices = self._valid_indices()
        if max_samples is not None:
            rng = np.random.default_rng(sample_seed)
            self.indices = rng.permutation(self.indices)[:max_samples]

    def _valid_indices(self) -> np.ndarray:
        valid: list[int] = []
        dones = self.arrays.dones
        start = max(self.context, self.index_start or self.context)
        end = min(self.index_end or len(self.arrays.frames), len(self.arrays.frames) - self.total_horizon)
        for i in range(start, end):
            if not dones[i - self.context : i + self.total_horizon].any():
                valid.append(i)
        return np.asarray(valid, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        target_idx = int(self.indices[item])
        context_frames = self.arrays.frames[target_idx - self.context : target_idx].astype(np.float32) / 255.0
        targets = self.arrays.frames[target_idx : target_idx + self.total_horizon].astype(np.float32) / 255.0
        actions = self.arrays.actions[target_idx - 1 : target_idx - 1 + self.total_horizon].astype(np.int64)

        context_frames = np.transpose(context_frames, (0, 3, 1, 2)).reshape(
            self.context * 3, context_frames.shape[1], context_frames.shape[2]
        )
        targets = np.transpose(targets, (0, 3, 1, 2))
        return {
            "frames": torch.from_numpy(context_frames),
            "actions": torch.from_numpy(actions),
            "targets": torch.from_numpy(targets),
        }


def foreground_mask(target: torch.Tensor, threshold: float) -> torch.Tensor:
    max_channel = target.max(dim=2, keepdim=True).values
    min_channel = target.min(dim=2, keepdim=True).values
    saturation = max_channel - min_channel
    brightness = target.mean(dim=2, keepdim=True)
    return ((saturation > threshold) | (brightness > 0.48)).float()


def player_blue_mask(target: torch.Tensor) -> torch.Tensor:
    red = target[:, :, 0:1]
    green = target[:, :, 1:2]
    blue = target[:, :, 2:3]
    return ((blue > 0.45) & (green > 0.25) & (red < 0.55) & (blue - green > 0.15) & (blue - red > 0.20)).float()


def sequence_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    foreground_weight: float,
    foreground_threshold: float,
    player_weight: float,
    temporal_weight: float,
) -> torch.Tensor:
    error = (pred - target).square()
    if foreground_weight > 0 or player_weight > 0:
        weights = torch.ones_like(error[:, :, 0:1])
        if foreground_weight > 0:
            weights = weights + foreground_weight * foreground_mask(target, foreground_threshold)
        if player_weight > 0:
            weights = weights + player_weight * player_blue_mask(target)
        error = error * weights
    loss = error.mean()
    if temporal_weight > 0 and pred.shape[1] > 1:
        pred_delta = pred[:, 1:] - pred[:, :-1]
        target_delta = target[:, 1:] - target[:, :-1]
        loss = loss + temporal_weight * (pred_delta - target_delta).square().mean()
    return loss


def rollout_sequence_loss(
    model: ActionConditionedSequencePredictor,
    frames: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    horizon: int,
    rollout_chunks: int,
    scheduled_sampling: float,
    detach_rollout: bool,
    foreground_weight: float,
    foreground_threshold: float,
    player_weight: float,
    temporal_weight: float,
) -> torch.Tensor:
    history = frames
    losses = []
    batch, _, height, width = frames.shape
    context = history.shape[1] // 3
    history_frames = history.reshape(batch, context, 3, height, width)
    for chunk in range(rollout_chunks):
        start = chunk * horizon
        end = start + horizon
        pred = model(history, actions[:, start:end])
        target = targets[:, start:end]
        losses.append(
            sequence_loss(
                pred,
                target,
                foreground_weight=foreground_weight,
                foreground_threshold=foreground_threshold,
                player_weight=player_weight,
                temporal_weight=temporal_weight,
            )
        )
        use_pred = random.random() < scheduled_sampling
        next_frames = pred if use_pred else target
        if detach_rollout:
            next_frames = next_frames.detach()
        history_frames = torch.cat([history_frames[:, horizon:], next_frames], dim=1)
        history = history_frames.reshape(batch, context * 3, height, width)
    weights = torch.linspace(1.0, 1.35, steps=rollout_chunks, device=frames.device)
    return torch.stack([loss * weights[i] for i, loss in enumerate(losses)]).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a direct multi-frame action-conditioned video predictor.")
    parser.add_argument("--data", default="data/toy_arena_mixed_event_128_16k.npz")
    parser.add_argument("--checkpoint", default=None, help="Optional sequence checkpoint to fine-tune.")
    parser.add_argument("--out-dir", default="runs/sequence_model")
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument("--index-start", type=int, default=None)
    parser.add_argument("--index-end", type=int, default=None)
    parser.add_argument("--rollout-chunks", type=int, default=1)
    parser.add_argument("--scheduled-sampling", type=float, default=0.0)
    parser.add_argument("--full-bptt", action="store_true")
    parser.add_argument("--foreground-weight", type=float, default=0.25)
    parser.add_argument("--foreground-threshold", type=float, default=0.18)
    parser.add_argument("--player-weight", type=float, default=0.0)
    parser.add_argument("--temporal-weight", type=float, default=0.25)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_coinrun_npz(args.data)
    action_count = int(arrays.action_count or max(arrays.actions.max() + 1, 1))
    dataset = SequenceDataset(
        arrays,
        context=args.context,
        horizon=args.horizon,
        rollout_chunks=args.rollout_chunks,
        max_samples=args.max_samples,
        sample_seed=args.sample_seed,
        index_start=args.index_start,
        index_end=args.index_end,
    )
    if len(dataset) < 2:
        raise ValueError("Not enough valid sequences.")

    val_size = max(1, int(0.05 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device == "cuda")

    model = ActionConditionedSequencePredictor(action_count=action_count, context=args.context, horizon=args.horizon).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if int(checkpoint["context"]) != args.context or int(checkpoint["horizon"]) != args.horizon:
            raise ValueError("Checkpoint context/horizon must match --context/--horizon.")
        model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch} sequence train"):
            frames = batch["frames"].to(device)
            actions = batch["actions"].to(device)
            targets = batch["targets"].to(device)
            loss = rollout_sequence_loss(
                model,
                frames,
                actions,
                targets,
                horizon=args.horizon,
                rollout_chunks=args.rollout_chunks,
                scheduled_sampling=args.scheduled_sampling,
                detach_rollout=not args.full_bptt,
                foreground_weight=args.foreground_weight,
                foreground_threshold=args.foreground_threshold,
                player_weight=args.player_weight,
                temporal_weight=args.temporal_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * frames.shape[0]

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"epoch {epoch} sequence val"):
                frames = batch["frames"].to(device)
                actions = batch["actions"].to(device)
                targets = batch["targets"].to(device)
                loss = rollout_sequence_loss(
                    model,
                    frames,
                    actions,
                    targets,
                    horizon=args.horizon,
                    rollout_chunks=args.rollout_chunks,
                    scheduled_sampling=1.0,
                    detach_rollout=True,
                    foreground_weight=args.foreground_weight,
                    foreground_threshold=args.foreground_threshold,
                    player_weight=args.player_weight,
                    temporal_weight=args.temporal_weight,
                )
                val_loss += loss.item() * frames.shape[0]

        train_loss /= train_size
        val_loss /= val_size
        print(f"epoch={epoch} sequence_train_loss={train_loss:.6f} sequence_val_loss={val_loss:.6f}")

        checkpoint = {
            "model_type": "action_conditioned_sequence",
            "model": model.state_dict(),
            "context": args.context,
            "horizon": args.horizon,
            "action_count": action_count,
            "epoch": epoch,
            "val_loss": val_loss,
            "source_checkpoint": args.checkpoint,
            "rollout_chunks": args.rollout_chunks,
            "scheduled_sampling": args.scheduled_sampling,
            "foreground_weight": args.foreground_weight,
            "foreground_threshold": args.foreground_threshold,
            "player_weight": args.player_weight,
            "temporal_weight": args.temporal_weight,
            "data": args.data,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, out_dir / "best.pt")


if __name__ == "__main__":
    main()
