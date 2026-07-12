from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_autoencoder import reconstruction_loss
from src.data import CoinRunArrays, load_coinrun_npz
from src.model import ActionConditionedLatentDynamics, ToyArenaAutoencoder


class LatentSequenceDataset(Dataset):
    def __init__(
        self,
        arrays: CoinRunArrays,
        context: int,
        horizon: int,
        rollout_chunks: int,
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
        start = max(self.context, self.index_start or self.context)
        end = min(self.index_end or len(self.arrays.frames), len(self.arrays.frames) - self.total_horizon)
        for i in range(start, end):
            if not self.arrays.dones[i - self.context : i + self.total_horizon].any():
                valid.append(i)
        return np.asarray(valid, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        target_idx = int(self.indices[item])
        frames = self.arrays.frames[target_idx - self.context : target_idx].astype(np.float32) / 255.0
        targets = self.arrays.frames[target_idx : target_idx + self.total_horizon].astype(np.float32) / 255.0
        actions = self.arrays.actions[target_idx - 1 : target_idx - 1 + self.total_horizon].astype(np.int64)
        frames = np.transpose(frames, (0, 3, 1, 2))
        targets = np.transpose(targets, (0, 3, 1, 2))
        return {
            "frames": torch.from_numpy(frames),
            "actions": torch.from_numpy(actions),
            "targets": torch.from_numpy(targets),
        }


def encode_sequence(autoencoder: ToyArenaAutoencoder, frames: torch.Tensor) -> torch.Tensor:
    batch, steps, channels, height, width = frames.shape
    flat = frames.reshape(batch * steps, channels, height, width)
    latents = autoencoder.encode(flat)
    return latents.reshape(batch, steps, latents.shape[1], latents.shape[2], latents.shape[3])


def decode_sequence(autoencoder: ToyArenaAutoencoder, latents: torch.Tensor) -> torch.Tensor:
    batch, steps, channels, height, width = latents.shape
    flat = latents.reshape(batch * steps, channels, height, width)
    frames = autoencoder.decode(flat)
    return frames.reshape(batch, steps, frames.shape[1], frames.shape[2], frames.shape[3])


def rollout_latent_loss(
    model: ActionConditionedLatentDynamics,
    autoencoder: ToyArenaAutoencoder,
    context_latents: torch.Tensor,
    actions: torch.Tensor,
    target_latents: torch.Tensor,
    target_frames: torch.Tensor,
    horizon: int,
    rollout_chunks: int,
    scheduled_sampling: float,
    detach_rollout: bool,
    latent_weight: float,
    decoded_weight: float,
    foreground_weight: float,
    foreground_threshold: float,
    player_weight: float,
) -> torch.Tensor:
    history = context_latents
    losses = []
    for chunk in range(rollout_chunks):
        start = chunk * horizon
        end = start + horizon
        pred_latents = model(history, actions[:, start:end])
        chunk_target_latents = target_latents[:, start:end]
        latent_loss = F.mse_loss(pred_latents, chunk_target_latents)
        decoded = decode_sequence(autoencoder, pred_latents)
        decoded_loss = reconstruction_loss(
            decoded.reshape(-1, decoded.shape[2], decoded.shape[3], decoded.shape[4]),
            target_frames[:, start:end].reshape(-1, target_frames.shape[2], target_frames.shape[3], target_frames.shape[4]),
            foreground_weight=foreground_weight,
            foreground_threshold=foreground_threshold,
            player_weight=player_weight,
            edge_weight=0.0,
        )
        losses.append(latent_weight * latent_loss + decoded_weight * decoded_loss)

        use_pred = random.random() < scheduled_sampling
        next_latents = pred_latents if use_pred else chunk_target_latents
        if detach_rollout:
            next_latents = next_latents.detach()
        history = torch.cat([history[:, horizon:], next_latents], dim=1)
    weights = torch.linspace(1.0, 1.35, steps=rollout_chunks, device=context_latents.device)
    return torch.stack([loss * weights[i] for i, loss in enumerate(losses)]).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train action-conditioned latent dynamics for toy arena.")
    parser.add_argument("--data", default="data/toy_arena_mixed_event_128_16k.npz")
    parser.add_argument("--autoencoder", default="runs/autoencoder_l64_16x16_e20_gpu/best.pt")
    parser.add_argument("--checkpoint", default=None, help="Optional latent dynamics checkpoint to fine-tune.")
    parser.add_argument("--out-dir", default="runs/latent_dynamics_c8_h8_e20_gpu")
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--rollout-chunks", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument("--index-start", type=int, default=None)
    parser.add_argument("--index-end", type=int, default=None)
    parser.add_argument("--scheduled-sampling", type=float, default=1.0)
    parser.add_argument("--full-bptt", action="store_true")
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--decoded-weight", type=float, default=0.35)
    parser.add_argument("--foreground-weight", type=float, default=0.35)
    parser.add_argument("--foreground-threshold", type=float, default=0.18)
    parser.add_argument("--player-weight", type=float, default=1.25)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ae_checkpoint = torch.load(args.autoencoder, map_location=device)
    latent_channels = int(ae_checkpoint["latent_channels"])
    autoencoder = ToyArenaAutoencoder(latent_channels=latent_channels).to(device)
    autoencoder.load_state_dict(ae_checkpoint["model"])
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad_(False)

    arrays = load_coinrun_npz(args.data)
    action_count = int(arrays.action_count or max(arrays.actions.max() + 1, 1))
    dataset = LatentSequenceDataset(
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

    model = ActionConditionedLatentDynamics(
        action_count=action_count,
        latent_channels=latent_channels,
        context=args.context,
        horizon=args.horizon,
    ).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if int(checkpoint["context"]) != args.context or int(checkpoint["horizon"]) != args.horizon:
            raise ValueError("Checkpoint context/horizon must match --context/--horizon.")
        if int(checkpoint["latent_channels"]) != latent_channels:
            raise ValueError("Checkpoint latent_channels must match the autoencoder latent_channels.")
        model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch} latent train"):
            frames = batch["frames"].to(device)
            targets = batch["targets"].to(device)
            actions = batch["actions"].to(device)
            with torch.no_grad():
                context_latents = encode_sequence(autoencoder, frames)
                target_latents = encode_sequence(autoencoder, targets)
            loss = rollout_latent_loss(
                model,
                autoencoder,
                context_latents,
                actions,
                target_latents,
                targets,
                horizon=args.horizon,
                rollout_chunks=args.rollout_chunks,
                scheduled_sampling=args.scheduled_sampling,
                detach_rollout=not args.full_bptt,
                latent_weight=args.latent_weight,
                decoded_weight=args.decoded_weight,
                foreground_weight=args.foreground_weight,
                foreground_threshold=args.foreground_threshold,
                player_weight=args.player_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * frames.shape[0]

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"epoch {epoch} latent val"):
                frames = batch["frames"].to(device)
                targets = batch["targets"].to(device)
                actions = batch["actions"].to(device)
                context_latents = encode_sequence(autoencoder, frames)
                target_latents = encode_sequence(autoencoder, targets)
                loss = rollout_latent_loss(
                    model,
                    autoencoder,
                    context_latents,
                    actions,
                    target_latents,
                    targets,
                    horizon=args.horizon,
                    rollout_chunks=args.rollout_chunks,
                    scheduled_sampling=1.0,
                    detach_rollout=True,
                    latent_weight=args.latent_weight,
                    decoded_weight=args.decoded_weight,
                    foreground_weight=args.foreground_weight,
                    foreground_threshold=args.foreground_threshold,
                    player_weight=args.player_weight,
                )
                val_loss += loss.item() * frames.shape[0]

        train_loss /= train_size
        val_loss /= val_size
        print(f"epoch={epoch} latent_train_loss={train_loss:.6f} latent_val_loss={val_loss:.6f}")
        checkpoint = {
            "model_type": "action_conditioned_latent_dynamics",
            "model": model.state_dict(),
            "context": args.context,
            "horizon": args.horizon,
            "rollout_chunks": args.rollout_chunks,
            "latent_channels": latent_channels,
            "action_count": action_count,
            "epoch": epoch,
            "val_loss": val_loss,
            "autoencoder_checkpoint": args.autoencoder,
            "source_checkpoint": args.checkpoint,
            "data": args.data,
            "scheduled_sampling": args.scheduled_sampling,
            "latent_weight": args.latent_weight,
            "decoded_weight": args.decoded_weight,
            "foreground_weight": args.foreground_weight,
            "foreground_threshold": args.foreground_threshold,
            "player_weight": args.player_weight,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, out_dir / "best.pt")


if __name__ == "__main__":
    main()
