from __future__ import annotations

import itertools

import numpy as np
import torch

from src.losses_v2 import object_masks_from_rgb


def _refine_peak(heatmap: torch.Tensor, x: torch.Tensor, y: torch.Tensor, radius: int = 2) -> torch.Tensor:
    offsets = torch.arange(-radius, radius + 1, device=heatmap.device)
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    sample_x = (x[:, None] + offset_x.flatten()[None]).clamp(0, heatmap.shape[2] - 1)
    sample_y = (y[:, None] + offset_y.flatten()[None]).clamp(0, heatmap.shape[1] - 1)
    indices = sample_y * heatmap.shape[2] + sample_x
    values = heatmap.flatten(1).gather(1, indices).clamp_min(1e-6).pow(4)
    denominator = values.sum(dim=1).clamp_min(1e-6)
    return torch.stack(
        [
            (sample_x.to(values.dtype) * values).sum(dim=1) / denominator,
            (sample_y.to(values.dtype) * values).sum(dim=1) / denominator,
        ],
        dim=1,
    )


def _top_enemy_peaks(heatmap: torch.Tensor, count: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    maps = heatmap.clone()
    positions = []
    scores = []
    for _ in range(count):
        flat_index = maps.flatten(1).argmax(dim=1)
        y = flat_index // maps.shape[2]
        x = flat_index % maps.shape[2]
        score = maps[torch.arange(len(maps), device=maps.device), y, x]
        refine_radius = max(2, round(heatmap.shape[-1] / 16))
        positions.append(_refine_peak(heatmap, x, y, refine_radius))
        scores.append(score)
        suppression_radius = max(3, round(heatmap.shape[-1] * 3 / 32))
        yy = torch.arange(maps.shape[1], device=maps.device)[None, :, None]
        xx = torch.arange(maps.shape[2], device=maps.device)[None, None, :]
        suppress = ((xx - x[:, None, None]).abs() <= suppression_radius) & (
            (yy - y[:, None, None]).abs() <= suppression_radius
        )
        maps.masked_fill_(suppress, -1.0)
    return torch.stack(positions, dim=1), torch.stack(scores, dim=1)


def decode_state_probe(logits: torch.Tensor, frame_size: int = 128) -> dict[str, torch.Tensor]:
    heatmaps = torch.sigmoid(logits)
    scale = frame_size / logits.shape[-1]
    decoded: dict[str, torch.Tensor] = {}
    for name, channel in (("player", 0), ("coin", 1)):
        values = heatmaps[:, channel]
        flat_index = values.flatten(1).argmax(dim=1)
        y = flat_index // values.shape[2]
        x = flat_index % values.shape[2]
        refine_radius = max(2, round(values.shape[-1] / 16))
        decoded[f"{name}_pos"] = _refine_peak(values, x, y, refine_radius) * scale
        decoded[f"{name}_score"] = values[torch.arange(len(values), device=values.device), y, x]
    enemy_pos, enemy_score = _top_enemy_peaks(heatmaps[:, 2])
    decoded["enemy_pos"] = enemy_pos * scale
    decoded["enemy_score"] = enemy_score
    return decoded


def _enemy_assignment_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    errors = []
    for permutation in itertools.permutations(range(3)):
        candidate = prediction[:, list(permutation)]
        errors.append(torch.linalg.vector_norm(candidate - target, dim=-1).mean(dim=1))
    return torch.stack(errors, dim=1).amin(dim=1)


def probe_batch_metrics(
    logits: torch.Tensor,
    player_pos: torch.Tensor,
    coin_pos: torch.Tensor,
    enemy_pos: torch.Tensor,
    presence_threshold: float = 0.35,
    match_radius: float = 8.0,
) -> dict[str, torch.Tensor]:
    decoded = decode_state_probe(logits)
    player_error = torch.linalg.vector_norm(decoded["player_pos"] - player_pos, dim=-1)
    coin_error = torch.linalg.vector_norm(decoded["coin_pos"] - coin_pos, dim=-1)
    enemy_error = _enemy_assignment_error(decoded["enemy_pos"], enemy_pos)
    return {
        "player_error": player_error.mean(),
        "coin_error": coin_error.mean(),
        "enemy_error": enemy_error.mean(),
        "player_recall": ((decoded["player_score"] >= presence_threshold) & (player_error <= match_radius)).float().mean(),
        "coin_recall": ((decoded["coin_score"] >= presence_threshold) & (coin_error <= match_radius)).float().mean(),
        "enemy_recall": (
            (decoded["enemy_score"] >= presence_threshold).float().mean(dim=1)
            * (enemy_error <= match_radius).float()
        ).mean(),
    }


def foreground_mse_tensor(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = object_masks_from_rgb(target).amax(dim=1, keepdim=True)
    error = (prediction - target).square().mean(dim=1, keepdim=True)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def aggregate_metric_records(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    return {key: float(np.mean([record[key] for record in records])) for key in records[0]}
