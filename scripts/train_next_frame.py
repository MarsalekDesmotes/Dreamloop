from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CoinRunNextFrameDataset, load_coinrun_npz
from src.model import ActionConditionedNextFrame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/coinrun_20k.npz")
    parser.add_argument("--out-dir", default="runs/coinrun_next_frame")
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_coinrun_npz(args.data)
    action_count = int(arrays.action_count or max(arrays.actions.max() + 1, 1))
    dataset = CoinRunNextFrameDataset(arrays, context=args.context, max_samples=args.max_samples)
    val_size = max(1, int(0.05 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device == "cuda")

    model = ActionConditionedNextFrame(action_count=action_count, context=args.context).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch} train"):
            frames = batch["frames"].to(device)
            action = batch["action"].to(device)
            target = batch["target"].to(device)

            pred = model(frames, action)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * frames.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"epoch {epoch} val"):
                frames = batch["frames"].to(device)
                action = batch["action"].to(device)
                target = batch["target"].to(device)
                pred = model(frames, action)
                val_loss += F.mse_loss(pred, target).item() * frames.size(0)

        train_loss /= train_size
        val_loss /= val_size
        print(f"epoch={epoch} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

        checkpoint = {
            "model": model.state_dict(),
            "context": args.context,
            "action_count": action_count,
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, out_dir / "best.pt")


if __name__ == "__main__":
    main()
