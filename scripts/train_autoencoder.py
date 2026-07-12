from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CoinRunArrays, load_coinrun_npz
from src.model import ToyArenaAutoencoder


class FrameDataset(Dataset):
    def __init__(self, arrays: CoinRunArrays, max_samples: int | None = None, sample_seed: int = 123):
        self.arrays = arrays
        self.indices = np.arange(len(arrays.frames), dtype=np.int64)
        if max_samples is not None:
            rng = np.random.default_rng(sample_seed)
            self.indices = rng.permutation(self.indices)[:max_samples]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> torch.Tensor:
        frame = self.arrays.frames[int(self.indices[item])].astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        return torch.from_numpy(frame)


def foreground_mask(target: torch.Tensor, threshold: float) -> torch.Tensor:
    max_channel = target.max(dim=1, keepdim=True).values
    min_channel = target.min(dim=1, keepdim=True).values
    saturation = max_channel - min_channel
    brightness = target.mean(dim=1, keepdim=True)
    return ((saturation > threshold) | (brightness > 0.48)).float()


def player_blue_mask(target: torch.Tensor) -> torch.Tensor:
    red = target[:, 0:1]
    green = target[:, 1:2]
    blue = target[:, 2:3]
    return ((blue > 0.45) & (green > 0.25) & (red < 0.55) & (blue - green > 0.15) & (blue - red > 0.20)).float()


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    foreground_weight: float,
    foreground_threshold: float,
    player_weight: float,
    edge_weight: float,
) -> torch.Tensor:
    weights = torch.ones_like(target[:, 0:1])
    if foreground_weight > 0:
        weights = weights + foreground_weight * foreground_mask(target, foreground_threshold)
    if player_weight > 0:
        weights = weights + player_weight * player_blue_mask(target)
    loss = (pred - target).abs().mul(weights).mean()
    if edge_weight > 0:
        loss = loss + edge_weight * edge_loss(pred, target)
    return loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a toy-arena frame autoencoder.")
    parser.add_argument("--data", default="data/toy_arena_mixed_event_128_16k.npz")
    parser.add_argument("--out-dir", default="runs/autoencoder_l64_16x16_e20_gpu")
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument("--foreground-weight", type=float, default=0.35)
    parser.add_argument("--foreground-threshold", type=float, default=0.18)
    parser.add_argument("--player-weight", type=float, default=1.25)
    parser.add_argument("--edge-weight", type=float, default=0.10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_coinrun_npz(args.data)
    dataset = FrameDataset(arrays, max_samples=args.max_samples, sample_seed=args.sample_seed)
    if len(dataset) < 2:
        raise ValueError("Not enough frames.")
    val_size = max(1, int(0.05 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device == "cuda")

    model = ToyArenaAutoencoder(latent_channels=args.latent_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for frames in tqdm(train_loader, desc=f"epoch {epoch} autoencoder train"):
            frames = frames.to(device)
            pred, _ = model(frames)
            loss = reconstruction_loss(
                pred,
                frames,
                foreground_weight=args.foreground_weight,
                foreground_threshold=args.foreground_threshold,
                player_weight=args.player_weight,
                edge_weight=args.edge_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * frames.shape[0]

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames in tqdm(val_loader, desc=f"epoch {epoch} autoencoder val"):
                frames = frames.to(device)
                pred, _ = model(frames)
                loss = reconstruction_loss(
                    pred,
                    frames,
                    foreground_weight=args.foreground_weight,
                    foreground_threshold=args.foreground_threshold,
                    player_weight=args.player_weight,
                    edge_weight=args.edge_weight,
                )
                val_loss += loss.item() * frames.shape[0]

        train_loss /= train_size
        val_loss /= val_size
        print(f"epoch={epoch} autoencoder_train_loss={train_loss:.6f} autoencoder_val_loss={val_loss:.6f}")

        checkpoint = {
            "model_type": "toy_arena_autoencoder",
            "model": model.state_dict(),
            "latent_channels": args.latent_channels,
            "epoch": epoch,
            "val_loss": val_loss,
            "data": args.data,
            "foreground_weight": args.foreground_weight,
            "foreground_threshold": args.foreground_threshold,
            "player_weight": args.player_weight,
            "edge_weight": args.edge_weight,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, out_dir / "best.pt")


if __name__ == "__main__":
    main()
