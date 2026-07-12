from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.toy_arena_v2 import ENEMY_SPEED, EVENT_COIN, EVENT_COLLISION, PLAYER_SPEED


SPLIT_NAMES = ("train", "val", "test")
SEMANTIC_STATE_DIM_V2 = 19
SEMANTIC_STATE_DIM = 23


@dataclass(frozen=True)
class ToyArenaV2Arrays:
    root: Path
    metadata: dict
    frames: np.ndarray
    actions: np.ndarray
    dones: np.ndarray
    events: np.ndarray
    episode_ids: np.ndarray
    episode_seeds: np.ndarray
    episode_splits: np.ndarray
    player_pos: np.ndarray
    player_vel: np.ndarray
    coin_pos: np.ndarray
    coin_pad: np.ndarray
    enemy_pos: np.ndarray
    enemy_vel: np.ndarray
    score: np.ndarray
    collision_cooldown: np.ndarray
    health: np.ndarray | None
    portal_unlocked: np.ndarray | None
    game_status: np.ndarray | None


def dataset_manifest_hash(metadata: dict) -> str:
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_toy_arena_v2(path: str | Path, mmap_mode: str = "r") -> ToyArenaV2Arrays:
    root = Path(path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    expected_hash = metadata.get("manifest_hash")
    unhashed = {key: value for key, value in metadata.items() if key != "manifest_hash"}
    if expected_hash and dataset_manifest_hash(unhashed) != expected_hash:
        raise ValueError("dataset metadata manifest hash does not match")

    def load(name: str) -> np.ndarray:
        return np.load(root / f"{name}.npy", mmap_mode=mmap_mode)

    def load_optional(name: str) -> np.ndarray | None:
        path = root / f"{name}.npy"
        return np.load(path, mmap_mode=mmap_mode) if path.exists() else None

    arrays = ToyArenaV2Arrays(
        root=root,
        metadata=metadata,
        frames=load("frames"),
        actions=load("actions"),
        dones=load("dones"),
        events=load("events"),
        episode_ids=load("episode_ids"),
        episode_seeds=load("episode_seeds"),
        episode_splits=load("episode_splits"),
        player_pos=load("player_pos"),
        player_vel=load("player_vel"),
        coin_pos=load("coin_pos"),
        coin_pad=load("coin_pad"),
        enemy_pos=load("enemy_pos"),
        enemy_vel=load("enemy_vel"),
        score=load("score"),
        collision_cooldown=load("collision_cooldown"),
        health=load_optional("health"),
        portal_unlocked=load_optional("portal_unlocked"),
        game_status=load_optional("game_status"),
    )
    frame_count = int(metadata["frame_count"])
    for name in ("frames", "actions", "dones", "events", "episode_ids"):
        if len(getattr(arrays, name)) != frame_count:
            raise ValueError(f"{name} length does not match metadata frame_count")
    return arrays


def load_v2_semantic_states(
    path: str | Path,
    arrays: ToyArenaV2Arrays,
    mmap_mode: str = "r",
) -> tuple[np.ndarray, dict]:
    root = Path(path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata["dataset_manifest"] != arrays.metadata["manifest_hash"]:
        raise ValueError("semantic cache dataset manifest mismatch")
    state_cache_path = root / "states.npy"
    state_cache_metadata_path = root / "states_metadata.json"
    if state_cache_path.exists() and state_cache_metadata_path.exists():
        state_cache_metadata = json.loads(state_cache_metadata_path.read_text(encoding="utf-8"))
        if state_cache_metadata.get("dataset_manifest") != arrays.metadata["manifest_hash"]:
            raise ValueError("semantic state cache dataset manifest mismatch")
        states = np.load(state_cache_path, mmap_mode=mmap_mode)
        expected_shape = (len(arrays.frames), int(state_cache_metadata["state_dim"]))
        if states.shape != expected_shape:
            raise ValueError("semantic state cache shape mismatch")
        return states, metadata
    player = np.load(root / "player_pos.npy", mmap_mode=mmap_mode)
    coin = np.load(root / "coin_pos.npy", mmap_mode=mmap_mode)
    enemies = np.load(root / "enemy_pos.npy", mmap_mode=mmap_mode)
    flash = np.load(root / "flash.npy", mmap_mode=mmap_mode)
    game_state_path = root / "game_state.npy"
    game_state = np.load(game_state_path, mmap_mode=mmap_mode) if game_state_path.exists() else None
    if not all(len(values) == len(arrays.frames) for values in (player, coin, enemies, flash)):
        raise ValueError("semantic cache frame count mismatch")

    state_dim = int(metadata.get("state_dim", SEMANTIC_STATE_DIM_V2))
    if state_dim not in (SEMANTIC_STATE_DIM_V2, SEMANTIC_STATE_DIM):
        raise ValueError(f"unsupported semantic state dimension {state_dim}")
    if state_dim == SEMANTIC_STATE_DIM and game_state is None:
        raise ValueError("23-value semantic cache requires game_state.npy")
    states = np.zeros((len(arrays.frames), state_dim), dtype=np.float32)
    states[:, 0:2] = np.asarray(player, dtype=np.float32) / 64.0 - 1.0
    states[:, 4:6] = np.asarray(coin, dtype=np.float32) / 64.0 - 1.0
    states[:, 6:12] = np.asarray(enemies, dtype=np.float32).reshape(len(states), 6) / 64.0 - 1.0
    states[:, 18:19] = np.asarray(flash, dtype=np.float32)
    if game_state is not None:
        states[:, 19:23] = np.asarray(game_state, dtype=np.float32)
    segment_start = 0
    segment_ends = np.flatnonzero(np.asarray(arrays.dones, dtype=np.bool_)) + 1
    for end in segment_ends:
        start = segment_start
        segment_start = int(end)
        if end - start < 2:
            continue
        player_pixels = np.asarray(player[start:end], dtype=np.float32)
        coin_pixels = np.asarray(coin[start:end], dtype=np.float32)
        enemy_pixels = np.asarray(enemies[start:end], dtype=np.float32)
        _, player_velocity = estimate_action_conditioned_player_motion(
            player_pixels, coin_pixels, np.asarray(arrays.actions[start : end - 1], dtype=np.int64)
        )
        states[start:end, 2:4] = player_velocity / 8.0
        states[start:end, 12:18] = (
            estimate_discrete_enemy_velocity(enemy_pixels) / 4.0
        ).reshape(-1, 6)
    if segment_start < len(states):
        start, end = segment_start, len(states)
        player_pixels = np.asarray(player[start:end], dtype=np.float32)
        coin_pixels = np.asarray(coin[start:end], dtype=np.float32)
        enemy_pixels = np.asarray(enemies[start:end], dtype=np.float32)
        _, player_velocity = estimate_action_conditioned_player_motion(
            player_pixels, coin_pixels, np.asarray(arrays.actions[start : end - 1], dtype=np.int64)
        )
        states[start:end, 2:4] = player_velocity / 8.0
        states[start:end, 12:18] = (
            estimate_discrete_enemy_velocity(enemy_pixels) / 4.0
        ).reshape(-1, 6)
    if state_dim >= SEMANTIC_STATE_DIM:
        terminal = states[:, 21:23].max(axis=1) > 0.5
        states[terminal, 2:4] = 0.0
        states[terminal, 12:18] = 0.0
    np.clip(states[:, 2:4], -1.5, 1.5, out=states[:, 2:4])
    np.clip(states[:, 12:18], -1.5, 1.5, out=states[:, 12:18])
    return states, metadata


def decode_visible_game_state(frames: np.ndarray) -> np.ndarray:
    """Read health, progress and terminal state from pixels visible to the model."""
    values = np.asarray(frames, dtype=np.float32) / 255.0
    if values.ndim == 3:
        values = values[None]
    health_pixels = values[:, 7, (8, 17, 26)]
    health_active = (health_pixels[..., 0] > 0.28) & (
        health_pixels[..., 0] > health_pixels[..., 1] * 1.8
    )
    progress_pixels = values[:, 6, (103, 110, 117)]
    progress_active = (progress_pixels[..., 1] > 0.28) & (
        progress_pixels[..., 1] > progress_pixels[..., 0] * 1.8
    )
    banner = values[:, 48:80, 14:114]
    win_color = np.asarray((19, 57, 52), dtype=np.float32) / 255.0
    lose_color = np.asarray((67, 26, 35), dtype=np.float32) / 255.0
    won = (np.max(np.abs(banner - win_color), axis=-1) < 0.01).sum(axis=(1, 2)) > 100
    lost = (np.max(np.abs(banner - lose_color), axis=-1) < 0.01).sum(axis=(1, 2)) > 100
    return np.stack(
        [
            health_active.mean(axis=1),
            progress_active.mean(axis=1),
            won.astype(np.float32),
            lost.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def decode_visible_collision_flash(frames: np.ndarray) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float32) / 255.0
    if values.ndim == 3:
        values = values[None]
    normal = np.asarray((255, 246, 155), dtype=np.float32) / 255.0
    terminal = np.asarray((172, 168, 109), dtype=np.float32) / 255.0
    normal_pixels = np.max(np.abs(values - normal), axis=-1) < 0.015
    terminal_pixels = np.max(np.abs(values - terminal), axis=-1) < 0.015
    return ((normal_pixels | terminal_pixels).sum(axis=(1, 2)) >= 8).astype(np.float32)


def estimate_semantic_velocity(positions: np.ndarray, scale: float, window: int = 3) -> np.ndarray:
    """Estimate velocity from RGB-derived positions without amplifying one-frame probe jitter."""
    positions = np.asarray(positions, dtype=np.float32)
    velocity = np.zeros_like(positions)
    if len(positions) < 2:
        return velocity
    deltas = np.diff(positions, axis=0)
    for index in range(1, len(positions)):
        first = max(0, index - window)
        velocity[index] = np.median(deltas[first:index], axis=0) * scale
    velocity[0] = velocity[1]
    return velocity


def estimate_chasing_enemy_velocity(
    player_positions: np.ndarray,
    enemy_positions: np.ndarray,
    alpha: float = 0.6,
    beta: float = 0.2,
) -> np.ndarray:
    """Causal alpha-beta filter using the visible shared chase motion as its prediction model."""
    player_positions = np.asarray(player_positions, dtype=np.float32)
    enemy_positions = np.asarray(enemy_positions, dtype=np.float32)
    velocity = np.zeros_like(enemy_positions)
    if len(enemy_positions) < 2:
        return velocity
    filtered_position = enemy_positions[0].copy()
    filtered_velocity = enemy_positions[1] - enemy_positions[0]
    velocity[0] = filtered_velocity
    for index in range(1, len(enemy_positions)):
        predicted_velocity = filtered_velocity
        predicted_position = filtered_position + predicted_velocity
        residual = enemy_positions[index] - predicted_position
        filtered_position = predicted_position + alpha * residual
        filtered_velocity = predicted_velocity + beta * residual
        velocity[index] = filtered_velocity
    return velocity


def _fit_discrete_enemy_velocity(enemy_positions: np.ndarray, speed: float = ENEMY_SPEED) -> np.ndarray:
    positions_observed = np.asarray(enemy_positions, dtype=np.float32)
    if len(positions_observed) == 0:
        return np.zeros_like(positions_observed)
    directions = np.asarray(
        ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)),
        dtype=np.float32,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    candidates = directions * speed
    result = np.zeros_like(positions_observed)
    for enemy_index in range(3):
        positions = np.repeat(positions_observed[0, enemy_index][None], len(candidates), axis=0)
        velocities = candidates.copy()
        errors = np.zeros(len(candidates), dtype=np.float32)
        velocity_history = [velocities.copy()]
        for step in range(1, len(positions_observed)):
            next_positions = positions + velocities
            hit = (next_positions < 14.0) | (next_positions > 114.0)
            velocities = np.where(hit, -velocities, velocities)
            positions = np.clip(next_positions, 14.0, 114.0)
            squared_error = np.square(positions - positions_observed[step, enemy_index]).sum(axis=1)
            errors += np.minimum(squared_error, 16.0)
            velocity_history.append(velocities.copy())
        selected = int(errors.argmin())
        result[:, enemy_index] = np.stack(velocity_history, axis=0)[:, selected]
    return result


def estimate_discrete_enemy_velocity(
    enemy_positions: np.ndarray, speed: float = ENEMY_SPEED, window: int = 24
) -> np.ndarray:
    positions = np.asarray(enemy_positions, dtype=np.float32)
    result = np.zeros_like(positions)
    for index in range(len(positions)):
        start = max(0, index - window + 1)
        result[index] = _fit_discrete_enemy_velocity(positions[start : index + 1], speed)[-1]
    return result


def fit_discrete_enemy_trajectory(
    enemy_positions: np.ndarray,
    speed: float = ENEMY_SPEED,
    offset_range: float = 1.5,
    offset_step: float = 0.125,
) -> tuple[np.ndarray, np.ndarray]:
    """Denoise a short RGB-derived track by fitting all legal bounce trajectories."""
    observed = np.asarray(enemy_positions, dtype=np.float32)
    if observed.ndim != 3 or observed.shape[1:] != (3, 2):
        raise ValueError("enemy_positions must have shape [T,3,2]")
    if len(observed) == 0:
        return observed.copy(), observed.copy()
    directions = np.asarray(
        ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)),
        dtype=np.float32,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    base_velocities = directions * speed
    offsets = np.arange(-offset_range, offset_range + offset_step * 0.5, offset_step, dtype=np.float32)
    offset_y, offset_x = np.meshgrid(offsets, offsets, indexing="ij")
    offset_grid = np.stack([offset_x.ravel(), offset_y.ravel()], axis=1)
    candidate_velocities = np.repeat(base_velocities, len(offset_grid), axis=0)
    candidate_offsets = np.tile(offset_grid, (len(base_velocities), 1))
    fitted_positions = np.zeros_like(observed)
    fitted_velocities = np.zeros_like(observed)
    for enemy_index in range(3):
        positions = observed[0, enemy_index][None] + candidate_offsets
        velocities = candidate_velocities.copy()
        errors = np.square(positions - observed[0, enemy_index]).sum(axis=1)
        position_history = [positions.copy()]
        velocity_history = [velocities.copy()]
        for step in range(1, len(observed)):
            next_positions = positions + velocities
            hit = (next_positions < 14.0) | (next_positions > 114.0)
            velocities = np.where(hit, -velocities, velocities)
            positions = np.clip(next_positions, 14.0, 114.0)
            squared_error = np.square(positions - observed[step, enemy_index]).sum(axis=1)
            errors += np.minimum(squared_error, 16.0)
            position_history.append(positions.copy())
            velocity_history.append(velocities.copy())
        selected = int(errors.argmin())
        fitted_positions[:, enemy_index] = np.stack(position_history, axis=0)[:, selected]
        fitted_velocities[:, enemy_index] = np.stack(velocity_history, axis=0)[:, selected]
    return fitted_positions, fitted_velocities


def fit_direct_player_trajectory(
    player_positions: np.ndarray,
    coin_positions: np.ndarray,
    enemy_positions: np.ndarray,
    actions: np.ndarray,
    offset_range: float = 1.5,
    offset_step: float = 0.125,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit direct-action player motion, including walls and visible contact repulsion."""
    observed = np.asarray(player_positions, dtype=np.float32)
    coins = np.asarray(coin_positions, dtype=np.float32)
    enemies = np.asarray(enemy_positions, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    if len(actions) != max(0, len(observed) - 1):
        raise ValueError("actions must contain one transition per player position pair")
    offsets = np.arange(-offset_range, offset_range + offset_step * 0.5, offset_step, dtype=np.float32)
    offset_y, offset_x = np.meshgrid(offsets, offsets, indexing="ij")
    candidates = observed[0][None] + np.stack([offset_x.ravel(), offset_y.ravel()], axis=1)
    errors = np.square(candidates - observed[0]).sum(axis=1)
    position_history = [candidates.copy()]
    velocity_history = [np.zeros_like(candidates)]
    for step, action in enumerate(actions, start=1):
        velocity = np.zeros_like(candidates)
        if 1 <= action <= 4:
            direction = np.asarray(
                [float(action == 4) - float(action == 3), float(action == 2) - float(action == 1)],
                dtype=np.float32,
            )
            velocity[:] = direction * PLAYER_SPEED
        elif action == 5:
            direction = coins[step - 1][None] - candidates
            direction /= np.linalg.norm(direction, axis=1, keepdims=True).clip(1e-5)
            velocity = direction * 5.6
        next_positions = candidates + velocity
        hit = (next_positions < 14.0) | (next_positions > 114.0)
        velocity = np.where(hit, velocity * -0.35, velocity)
        candidates = np.clip(next_positions, 14.0, 114.0)
        deltas = candidates[:, None] - enemies[step][None]
        distances = np.linalg.norm(deltas, axis=2)
        away = deltas / distances[:, :, None].clip(1e-5)
        strength = np.clip((24.0 - distances) / 24.0, 0.0, 1.0)
        contact_force = (away * strength[:, :, None]).sum(axis=1)
        velocity += contact_force * 1.2
        candidates = np.clip(candidates + contact_force * 1.5, 14.0, 114.0)
        squared_error = np.square(candidates - observed[step]).sum(axis=1)
        errors += np.minimum(squared_error, 16.0)
        position_history.append(candidates.copy())
        velocity_history.append(velocity.copy())
    selected = int(errors.argmin())
    return (
        np.stack(position_history, axis=0)[:, selected],
        np.stack(velocity_history, axis=0)[:, selected],
    )


def estimate_action_conditioned_player_motion(
    player_positions: np.ndarray,
    coin_positions: np.ndarray,
    actions: np.ndarray,
    alpha: float = 0.7,
    beta: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover direct-action player motion from visible actions and geometry."""
    del alpha, beta
    player_positions = np.asarray(player_positions, dtype=np.float32)
    coin_positions = np.asarray(coin_positions, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    if len(actions) != max(0, len(player_positions) - 1):
        raise ValueError("actions must contain one transition per player position pair")
    filtered_positions = player_positions.copy()
    velocities = np.zeros_like(player_positions)
    if len(player_positions) == 0:
        return filtered_positions, velocities
    for index, action in enumerate(actions, start=1):
        predicted_velocity = np.zeros(2, dtype=np.float32)
        if 1 <= action <= 4:
            direction = np.asarray(
                [float(action == 4) - float(action == 3), float(action == 2) - float(action == 1)],
                dtype=np.float32,
            )
            predicted_velocity = direction * 2.4
        elif action == 5:
            direction = coin_positions[index - 1] - player_positions[index - 1]
            direction = direction / max(float(np.linalg.norm(direction)), 1e-5)
            predicted_velocity = direction * 5.6
        predicted_position = player_positions[index - 1] + predicted_velocity
        hit = (predicted_position < 14.0) | (predicted_position > 114.0)
        predicted_velocity = np.where(hit, predicted_velocity * -0.35, predicted_velocity)
        velocities[index] = predicted_velocity
    if len(velocities) > 1:
        velocities[0] = velocities[1]
    return filtered_positions, velocities


class ToyArenaV2SequenceDataset(Dataset):
    """Sequences where action t transitions frame t into frame t+1."""

    def __init__(
        self,
        arrays: ToyArenaV2Arrays,
        split: str = "train",
        context: int = 24,
        horizon: int = 8,
        max_samples: int | None = None,
        seed: int = 1337,
        return_frames: bool = True,
        episode_ids: list[int] | tuple[int, ...] | None = None,
    ):
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}")
        if context < 2 or horizon < 1:
            raise ValueError("context must be >=2 and horizon must be >=1")
        self.arrays = arrays
        self.split = split
        self.split_id = SPLIT_NAMES.index(split)
        self.context = context
        self.horizon = horizon
        self.return_frames = return_frames
        self.episode_ids = None if episode_ids is None else frozenset(int(value) for value in episode_ids)
        self.indices = self._valid_indices()
        if max_samples is not None and max_samples < len(self.indices):
            rng = np.random.default_rng(seed)
            self.indices = rng.choice(self.indices, size=max_samples, replace=False)
        self.event_flags = self._event_flags()
        self.event_classes = self._event_classes()

    def _valid_indices(self) -> np.ndarray:
        episode_length = int(self.arrays.metadata["episode_length"])
        valid: list[int] = []
        for episode_id, split_id in enumerate(self.arrays.episode_splits):
            if int(split_id) != self.split_id:
                continue
            if self.episode_ids is not None and episode_id not in self.episode_ids:
                continue
            start = episode_id * episode_length
            end = start + episode_length
            first_anchor = start + self.context - 1
            last_anchor = end - self.horizon - 1
            if first_anchor <= last_anchor:
                for anchor in range(first_anchor, last_anchor + 1):
                    context_start = anchor - self.context + 1
                    transition_end = anchor + self.horizon
                    if not np.any(self.arrays.dones[context_start:transition_end]):
                        valid.append(anchor)
        return np.asarray(valid, dtype=np.int64)

    def _event_flags(self) -> np.ndarray:
        flags = np.zeros(len(self.indices), dtype=np.uint8)
        for item, anchor in enumerate(self.indices):
            future_events = self.arrays.events[anchor : anchor + self.horizon]
            if np.any(future_events & EVENT_COIN):
                flags[item] |= EVENT_COIN
            if np.any(future_events & EVENT_COLLISION):
                flags[item] |= EVENT_COLLISION
        return flags

    def _event_classes(self) -> np.ndarray:
        classes = np.zeros(len(self.event_flags), dtype=np.uint8)
        classes[(self.event_flags & EVENT_COIN) > 0] = 1
        classes[(self.event_flags & EVENT_COLLISION) > 0] = 2
        return classes

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def _frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
        values = np.asarray(frames, dtype=np.float32) / 255.0
        return torch.from_numpy(np.transpose(values, (0, 3, 1, 2)).copy())

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        anchor = int(self.indices[item])
        context_start = anchor - self.context + 1
        target_end = anchor + self.horizon + 1
        result = {
            "context_actions": torch.from_numpy(np.asarray(self.arrays.actions[context_start:anchor], dtype=np.int64).copy()),
            "future_actions": torch.from_numpy(np.asarray(self.arrays.actions[anchor : anchor + self.horizon], dtype=np.int64).copy()),
            "target_player_pos": torch.from_numpy(np.asarray(self.arrays.player_pos[anchor + 1 : target_end], dtype=np.float32).copy()),
            "target_coin_pos": torch.from_numpy(np.asarray(self.arrays.coin_pos[anchor + 1 : target_end], dtype=np.float32).copy()),
            "target_enemy_pos": torch.from_numpy(np.asarray(self.arrays.enemy_pos[anchor + 1 : target_end], dtype=np.float32).copy()),
            "events": torch.from_numpy(np.asarray(self.arrays.events[anchor : anchor + self.horizon], dtype=np.uint8).copy()),
            "anchor": torch.tensor(anchor, dtype=torch.long),
        }
        if self.return_frames:
            result["context_frames"] = self._frames_to_tensor(self.arrays.frames[context_start : anchor + 1])
            result["target_frames"] = self._frames_to_tensor(self.arrays.frames[anchor + 1 : target_end])
        return result


class ToyArenaV2FramePairDataset(Dataset):
    def __init__(
        self,
        arrays: ToyArenaV2Arrays,
        split: str = "train",
        max_samples: int | None = None,
        seed: int = 1337,
    ):
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}")
        split_id = SPLIT_NAMES.index(split)
        episode_length = int(arrays.metadata["episode_length"])
        indices: list[int] = []
        for episode_id, value in enumerate(arrays.episode_splits):
            if int(value) == split_id:
                start = episode_id * episode_length
                for index in range(start + 1, start + episode_length - 1):
                    if not bool(arrays.dones[index - 1]) and not bool(arrays.dones[index]):
                        indices.append(index)
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        if max_samples is not None and max_samples < len(self.indices):
            rng = np.random.default_rng(seed)
            self.indices = rng.choice(self.indices, size=max_samples, replace=False)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])

        def frame_tensor(frame: np.ndarray) -> torch.Tensor:
            value = np.asarray(frame, dtype=np.float32) / 255.0
            return torch.from_numpy(np.transpose(value, (2, 0, 1)).copy())

        return {
            "previous_frame": frame_tensor(self.arrays.frames[index - 1]),
            "frame": frame_tensor(self.arrays.frames[index]),
            "next_frame": frame_tensor(self.arrays.frames[index + 1]),
            "action": torch.tensor(int(self.arrays.actions[index]), dtype=torch.long),
            "player_pos": torch.from_numpy(np.asarray(self.arrays.player_pos[index], dtype=np.float32).copy()),
            "coin_pos": torch.from_numpy(np.asarray(self.arrays.coin_pos[index], dtype=np.float32).copy()),
            "enemy_pos": torch.from_numpy(np.asarray(self.arrays.enemy_pos[index], dtype=np.float32).copy()),
            "next_player_pos": torch.from_numpy(np.asarray(self.arrays.player_pos[index + 1], dtype=np.float32).copy()),
            "next_coin_pos": torch.from_numpy(np.asarray(self.arrays.coin_pos[index + 1], dtype=np.float32).copy()),
            "next_enemy_pos": torch.from_numpy(np.asarray(self.arrays.enemy_pos[index + 1], dtype=np.float32).copy()),
            "index": torch.tensor(index, dtype=torch.long),
        }


class ToyArenaV2LatentSequenceDataset(Dataset):
    def __init__(
        self,
        arrays: ToyArenaV2Arrays,
        latent_cache: str | Path,
        split: str = "train",
        context: int = 24,
        horizon: int = 8,
        max_samples: int | None = None,
        seed: int = 1337,
        episode_ids: list[int] | tuple[int, ...] | None = None,
    ):
        self.base = ToyArenaV2SequenceDataset(
            arrays,
            split=split,
            context=context,
            horizon=horizon,
            max_samples=max_samples,
            seed=seed,
            return_frames=False,
            episode_ids=episode_ids,
        )
        cache_root = Path(latent_cache)
        self.cache_metadata = json.loads((cache_root / "metadata.json").read_text(encoding="utf-8"))
        if self.cache_metadata["dataset_manifest"] != arrays.metadata["manifest_hash"]:
            raise ValueError("latent cache dataset manifest mismatch")
        if not self.cache_metadata.get("normalized", False):
            raise ValueError("latent cache must contain normalized latents")
        self.latents = np.load(cache_root / "latents.npy", mmap_mode="r")
        if len(self.latents) != len(arrays.frames):
            raise ValueError("latent cache frame count mismatch")
        self.arrays = arrays
        self.context = context
        self.horizon = horizon
        self.indices = self.base.indices
        self.event_flags = self.base.event_flags
        self.event_classes = self.base.event_classes

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        result = self.base[item]
        anchor = int(result["anchor"])
        context_start = anchor - self.context + 1
        target_end = anchor + self.horizon + 1
        result["context_latents"] = torch.from_numpy(
            np.asarray(self.latents[context_start : anchor + 1], dtype=np.float32).copy()
        )
        result["target_latents"] = torch.from_numpy(
            np.asarray(self.latents[anchor + 1 : target_end], dtype=np.float32).copy()
        )
        result["target_frames"] = ToyArenaV2SequenceDataset._frames_to_tensor(
            self.arrays.frames[anchor + 1 : target_end]
        )
        return result


class ToyArenaV2SemanticSequenceDataset(Dataset):
    def __init__(
        self,
        arrays: ToyArenaV2Arrays,
        states: np.ndarray,
        split: str = "train",
        context: int = 24,
        horizon: int = 64,
        max_samples: int | None = None,
        seed: int = 1337,
    ):
        self.base = ToyArenaV2SequenceDataset(
            arrays,
            split=split,
            context=context,
            horizon=horizon,
            max_samples=max_samples,
            seed=seed,
            return_frames=False,
        )
        if states.ndim != 2 or states.shape[0] != len(arrays.frames) or states.shape[1] not in (
            SEMANTIC_STATE_DIM_V2,
            SEMANTIC_STATE_DIM,
        ):
            raise ValueError("semantic states have an invalid shape")
        self.states = states
        self.context = context
        self.horizon = horizon
        self.indices = self.base.indices
        self.event_flags = self.base.event_flags
        self.event_classes = self.base.event_classes

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        result = self.base[item]
        anchor = int(result["anchor"])
        context_start = anchor - self.context + 1
        target_end = anchor + self.horizon + 1
        result["context_states"] = torch.from_numpy(self.states[context_start : anchor + 1].copy())
        result["target_states"] = torch.from_numpy(self.states[anchor + 1 : target_end].copy())
        return result


class ToyArenaV2SemanticFrameDataset(Dataset):
    def __init__(
        self,
        arrays: ToyArenaV2Arrays,
        states: np.ndarray,
        latent_cache: str | Path | None,
        split: str = "train",
        max_samples: int | None = None,
        seed: int = 1337,
    ):
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}")
        split_id = SPLIT_NAMES.index(split)
        episode_length = int(arrays.metadata["episode_length"])
        indices: list[int] = []
        for episode_id, value in enumerate(arrays.episode_splits):
            if int(value) == split_id:
                start = episode_id * episode_length
                indices.extend(range(start, start + episode_length))
        self.indices = np.asarray(indices, dtype=np.int64)
        if max_samples is not None and max_samples < len(self.indices):
            rng = np.random.default_rng(seed)
            self.indices = rng.choice(self.indices, size=max_samples, replace=False)
        self.latents = None
        if latent_cache is not None:
            cache_root = Path(latent_cache)
            cache_metadata = json.loads((cache_root / "metadata.json").read_text(encoding="utf-8"))
            if cache_metadata["dataset_manifest"] != arrays.metadata["manifest_hash"]:
                raise ValueError("latent cache dataset manifest mismatch")
            self.latents = np.load(cache_root / "latents.npy", mmap_mode="r")
        self.arrays = arrays
        self.states = states
        source_events = []
        for index in self.indices:
            episode_start = (int(index) // episode_length) * episode_length
            source_events.append(0 if int(index) == episode_start else int(arrays.events[int(index) - 1]))
        self.event_flags = np.asarray(source_events, dtype=np.uint8) & (EVENT_COIN | EVENT_COLLISION)
        self.event_classes = np.zeros(len(self.event_flags), dtype=np.uint8)
        self.event_classes[(self.event_flags & EVENT_COIN) > 0] = 1
        self.event_classes[(self.event_flags & EVENT_COLLISION) > 0] = 2

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        frame = np.asarray(self.arrays.frames[index], dtype=np.float32) / 255.0
        result = {
            "state": torch.from_numpy(self.states[index].copy()),
            "target_frame": torch.from_numpy(np.transpose(frame, (2, 0, 1)).copy()),
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self.latents is not None:
            result["target_latent"] = torch.from_numpy(
                np.asarray(self.latents[index], dtype=np.float32).copy()
            )
        return result


class StratifiedEventSampler(Sampler[int]):
    def __init__(
        self,
        dataset: ToyArenaV2SequenceDataset,
        normal_fraction: float = 0.40,
        coin_fraction: float = 0.30,
        collision_fraction: float = 0.30,
        num_samples: int | None = None,
        seed: int = 1337,
        allow_missing: bool = False,
    ):
        fractions = np.asarray((normal_fraction, coin_fraction, collision_fraction), dtype=np.float64)
        if not np.isclose(fractions.sum(), 1.0):
            raise ValueError("sampling fractions must sum to 1")
        if hasattr(dataset, "event_flags"):
            flags = np.asarray(dataset.event_flags)
            self.pools = [
                np.flatnonzero(flags == 0),
                np.flatnonzero((flags & EVENT_COIN) > 0),
                np.flatnonzero((flags & EVENT_COLLISION) > 0),
            ]
        else:
            self.pools = [np.flatnonzero(dataset.event_classes == event_class) for event_class in range(3)]
        available = np.asarray([len(pool) > 0 for pool in self.pools])
        if not available.all() and not allow_missing:
            raise ValueError("all normal/coin/collision pools must contain samples")
        fractions[~available] = 0.0
        self.fractions = fractions / fractions.sum()
        self.num_samples = int(num_samples or len(dataset))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        counts = np.floor(self.fractions * self.num_samples).astype(int)
        remainder_target = int(np.flatnonzero(self.fractions > 0)[0])
        counts[remainder_target] += self.num_samples - int(counts.sum())
        selections = [
            rng.choice(pool, size=count, replace=count > len(pool))
            for pool, count in zip(self.pools, counts)
            if count > 0
        ]
        combined = np.concatenate(selections)
        rng.shuffle(combined)
        return iter(combined.tolist())
