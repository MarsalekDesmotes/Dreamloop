from __future__ import annotations

import torch
import torch.nn.functional as F


def object_masks_from_rgb(frames: torch.Tensor) -> torch.Tensor:
    red, green, blue = frames[:, 0:1], frames[:, 1:2], frames[:, 2:3]
    player = (blue > 0.58) & (green > 0.30) & (red < 0.55) & (blue > red + 0.20)
    coin = (green > 0.68) & (red > 0.28) & (blue < 0.82) & ~player
    enemy = (red > 0.68) & (green < 0.78) & (blue < 0.78) & (red > green + 0.12)
    return torch.cat([player, coin, enemy], dim=1).float()


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    error = (prediction - target).abs().mean(dim=1, keepdim=True)
    numerator = (error * mask).flatten(1).sum(dim=1)
    denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return (numerator / denominator).mean()


def object_balanced_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    masks = object_masks_from_rgb(target)
    return torch.stack([masked_l1(prediction, target, masks[:, index : index + 1]) for index in range(3)]).mean()


def soft_object_segmentation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize missing object colors and false-positive color trails over the full frame."""
    red, green, blue = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]

    def conjunction(*conditions: torch.Tensor) -> torch.Tensor:
        return 20.0 * torch.stack(conditions, dim=0).amin(dim=0)

    logits = torch.cat(
        [
            conjunction(blue - 0.58, green - 0.30, 0.55 - red, blue - red - 0.20),
            conjunction(green - 0.68, red - 0.28, 0.82 - blue),
            conjunction(red - 0.68, 0.78 - green, 0.78 - blue, red - green - 0.12),
        ],
        dim=1,
    )
    masks = object_masks_from_rgb(target)
    positive_count = masks.flatten(2).sum(dim=2).clamp_min(1.0)
    negative_count = (1.0 - masks).flatten(2).sum(dim=2).clamp_min(1.0)
    positive = -(masks * F.logsigmoid(logits)).flatten(2).sum(dim=2) / positive_count
    negative = -((1.0 - masks) * F.logsigmoid(-logits)).flatten(2).sum(dim=2) / negative_count

    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * masks).flatten(2).sum(dim=2)
    denominator = probabilities.flatten(2).sum(dim=2) + masks.flatten(2).sum(dim=2)
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return (0.5 * (positive + negative) + dice).mean()


def edge_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_y = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


def codec_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    previous_prediction: torch.Tensor | None = None,
    previous_target: torch.Tensor | None = None,
    l1_weight: float = 1.0,
    object_weight: float = 1.0,
    edge_weight: float = 0.10,
    temporal_weight: float = 0.20,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    l1 = F.l1_loss(prediction, target)
    objects = object_balanced_l1(prediction, target)
    edges = edge_l1(prediction, target)
    temporal = prediction.new_zeros(())
    if previous_prediction is not None and previous_target is not None:
        temporal = F.l1_loss(prediction - previous_prediction, target - previous_target)
    total = l1_weight * l1 + object_weight * objects + edge_weight * edges + temporal_weight * temporal
    return total, {"l1": l1, "objects": objects, "edge": edges, "temporal": temporal}


def gaussian_heatmaps(
    player_pos: torch.Tensor,
    coin_pos: torch.Tensor,
    enemy_pos: torch.Tensor,
    size: int = 32,
    frame_size: int = 128,
    sigma: float = 1.25,
) -> torch.Tensor:
    axis = torch.arange(size, device=player_pos.device, dtype=player_pos.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)
    scale = size / frame_size

    def single(positions: torch.Tensor) -> torch.Tensor:
        delta = grid[None] - positions[:, None, None] * scale
        return torch.exp(-delta.square().sum(dim=-1) / (2.0 * sigma * sigma))

    player = single(player_pos)
    coin = single(coin_pos)
    batch = enemy_pos.shape[0]
    enemy_flat = enemy_pos.reshape(batch * enemy_pos.shape[1], 2)
    enemies = single(enemy_flat).reshape(batch, enemy_pos.shape[1], size, size).amax(dim=1)
    return torch.stack([player, coin, enemies], dim=1)
