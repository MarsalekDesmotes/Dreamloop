from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CoinRunArrays:
    frames: np.ndarray
    actions: np.ndarray
    dones: np.ndarray
    action_count: int | None = None


def load_coinrun_npz(path: str) -> CoinRunArrays:
    data = np.load(path)
    frames = data["frames"]
    actions = data["actions"]
    dones = data["dones"]

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames shaped [T,H,W,3], got {frames.shape}")
    if len(actions) != len(frames) or len(dones) != len(frames):
        raise ValueError("frames, actions, and dones must have the same first dimension")

    action_count = int(data["action_count"]) if "action_count" in data else None
    return CoinRunArrays(frames=frames, actions=actions, dones=dones, action_count=action_count)


class CoinRunNextFrameDataset(Dataset):
    def __init__(self, arrays: CoinRunArrays, context: int = 4, max_samples: int | None = None):
        self.arrays = arrays
        self.context = context
        self.indices = self._valid_indices()
        if max_samples is not None:
            self.indices = self.indices[:max_samples]

    def _valid_indices(self) -> np.ndarray:
        # Target index i predicts frame i from frames [i-context, i) and action i-1.
        valid = []
        dones = self.arrays.dones
        for i in range(self.context, len(self.arrays.frames)):
            if not dones[i - self.context : i].any():
                valid.append(i)
        return np.asarray(valid, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        target_idx = int(self.indices[item])
        start = target_idx - self.context

        frames = self.arrays.frames[start:target_idx].astype(np.float32) / 255.0
        target = self.arrays.frames[target_idx].astype(np.float32) / 255.0
        action = int(self.arrays.actions[target_idx - 1])

        # [C,H,W] where C is context*RGB.
        frames = np.transpose(frames, (0, 3, 1, 2)).reshape(self.context * 3, frames.shape[1], frames.shape[2])
        target = np.transpose(target, (2, 0, 1))

        return {
            "frames": torch.from_numpy(frames),
            "action": torch.tensor(action, dtype=torch.long),
            "target": torch.from_numpy(target),
        }
