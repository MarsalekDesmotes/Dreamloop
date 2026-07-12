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
from src.model import ActionConditionedNextFrame


class RolloutDataset(Dataset):
    def __init__(
        self,
        arrays: CoinRunArrays,
        context: int,
        horizon: int,
        max_samples: int | None = None,
        sample_seed: int = 123,
    ):
        self.arrays = arrays
        self.context = context
        self.horizon = horizon
        self.indices = self._valid_indices()
        if max_samples is not None:
            rng = np.random.default_rng(sample_seed)
            self.indices = rng.permutation(self.indices)
            self.indices = self.indices[:max_samples]

    def _valid_indices(self) -> np.ndarray:
        valid: list[int] = []
        dones = self.arrays.dones
        end = len(self.arrays.frames) - self.horizon
        for i in range(self.context, end):
            if not dones[i - self.context : i + self.horizon].any():
                valid.append(i)
        return np.asarray(valid, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        target_idx = int(self.indices[item])
        start = target_idx - self.context
        context_frames = self.arrays.frames[start:target_idx].astype(np.float32) / 255.0
        targets = self.arrays.frames[target_idx : target_idx + self.horizon].astype(np.float32) / 255.0
        actions = self.arrays.actions[target_idx - 1 : target_idx - 1 + self.horizon].astype(np.int64)

        context_frames = np.transpose(context_frames, (0, 3, 1, 2))
        targets = np.transpose(targets, (0, 3, 1, 2))
        return {
            "frames": torch.from_numpy(context_frames),
            "actions": torch.from_numpy(actions),
            "targets": torch.from_numpy(targets),
        }


def flatten_context(frames: torch.Tensor) -> torch.Tensor:
    batch, context, channels, height, width = frames.shape
    return frames.reshape(batch, context * channels, height, width)


def foreground_mask(target: torch.Tensor, threshold: float) -> torch.Tensor:
    max_channel = target.max(dim=1, keepdim=True).values
    min_channel = target.min(dim=1, keepdim=True).values
    saturation = max_channel - min_channel
    brightness = target.mean(dim=1, keepdim=True)
    return ((saturation > threshold) | (brightness > 0.48)).float()


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, foreground_weight: float, threshold: float) -> torch.Tensor:
    error = (pred - target).square()
    if foreground_weight <= 0:
        return error.mean()
    mask = foreground_mask(target, threshold)
    weights = 1.0 + foreground_weight * mask
    return (error * weights).mean()


def rollout_loss(
    model: ActionConditionedNextFrame,
    frames: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    scheduled_sampling: float,
    detach_rollout: bool,
    foreground_weight: float,
    foreground_threshold: float,
) -> torch.Tensor:
    history = frames
    losses = []
    horizon = actions.shape[1]

    for step in range(horizon):
        pred = model(flatten_context(history), actions[:, step])
        target = targets[:, step]
        losses.append(weighted_mse(pred, target, foreground_weight, foreground_threshold))

        use_pred = random.random() < scheduled_sampling
        next_frame = pred if use_pred else target
        if detach_rollout:
            next_frame = next_frame.detach()
        history = torch.cat([history[:, 1:], next_frame[:, None]], dim=1)

    weights = torch.linspace(1.0, 1.5, steps=horizon, device=targets.device)
    return torch.stack([loss * weights[i] for i, loss in enumerate(losses)]).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a next-frame model for more stable closed-loop rollouts.")
    parser.add_argument("--data", default="data/toy_arena_balanced_128_10k.npz")
    parser.add_argument("--checkpoint", default="runs/improved/best.pt")
    parser.add_argument("--out-dir", default="runs/closed_loop_stable")
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument("--scheduled-sampling", type=float, default=0.65)
    parser.add_argument("--full-bptt", action="store_true")
    parser.add_argument("--foreground-weight", type=float, default=0.0)
    parser.add_argument("--foreground-threshold", type=float, default=0.18)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arrays = load_coinrun_npz(args.data)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    context = int(args.context or checkpoint["context"])
    action_count = int(checkpoint["action_count"])

    dataset = RolloutDataset(
        arrays,
        context=context,
        horizon=args.horizon,
        max_samples=args.max_samples,
        sample_seed=args.sample_seed,
    )
    if len(dataset) < 2:
        raise ValueError("Not enough valid rollout sequences.")
    val_size = max(1, int(0.05 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ActionConditionedNextFrame(action_count=action_count, context=context).to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch} rollout train"):
            frames = batch["frames"].to(device)
            actions = batch["actions"].to(device)
            targets = batch["targets"].to(device)
            loss = rollout_loss(
                model,
                frames,
                actions,
                targets,
                scheduled_sampling=args.scheduled_sampling,
                detach_rollout=not args.full_bptt,
                foreground_weight=args.foreground_weight,
                foreground_threshold=args.foreground_threshold,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * frames.shape[0]

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"epoch {epoch} rollout val"):
                frames = batch["frames"].to(device)
                actions = batch["actions"].to(device)
                targets = batch["targets"].to(device)
                loss = rollout_loss(
                    model,
                    frames,
                    actions,
                    targets,
                    scheduled_sampling=1.0,
                    detach_rollout=True,
                    foreground_weight=args.foreground_weight,
                    foreground_threshold=args.foreground_threshold,
                )
                val_loss += loss.item() * frames.shape[0]

        train_loss /= train_size
        val_loss /= val_size
        print(f"epoch={epoch} rollout_train_mse={train_loss:.6f} rollout_val_mse={val_loss:.6f}")

        save = {
            "model": model.state_dict(),
            "context": context,
            "action_count": action_count,
            "epoch": epoch,
            "val_loss": val_loss,
            "source_checkpoint": str(args.checkpoint),
            "rollout_horizon": args.horizon,
            "scheduled_sampling": args.scheduled_sampling,
            "foreground_weight": args.foreground_weight,
            "foreground_threshold": args.foreground_threshold,
        }
        torch.save(save, out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(save, out_dir / "best.pt")


if __name__ == "__main__":
    main()
