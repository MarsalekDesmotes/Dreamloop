from __future__ import annotations

import torch
from torch import nn


class ActionConditionedNextFrame(nn.Module):
    def __init__(self, action_count: int, context: int = 4, action_dim: int = 8):
        super().__init__()
        in_channels = context * 3
        self.action_channels = action_dim
        self.action_embed = nn.Embedding(action_count, action_dim)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels + action_dim, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 48, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(48, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 48, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(48, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, frames: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = frames.shape
        action_map = self.action_embed(action).view(batch, self.action_channels, 1, 1)
        action_map = action_map.expand(batch, self.action_channels, height, width)
        return self.net(torch.cat([frames, action_map], dim=1))


class ActionConditionedSequencePredictor(nn.Module):
    def __init__(self, action_count: int, context: int = 8, horizon: int = 8, action_dim: int = 8):
        super().__init__()
        self.context = context
        self.horizon = horizon
        self.action_channels = action_dim
        in_channels = context * 3 + horizon * action_dim
        self.action_embed = nn.Embedding(action_count, action_dim)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 80, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(80, 96, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(96, 80, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(80, 48, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(48, horizon * 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = frames.shape
        action_tokens = self.action_embed(actions).reshape(batch, self.horizon * self.action_channels, 1, 1)
        action_map = action_tokens.expand(batch, self.horizon * self.action_channels, height, width)
        out = self.net(torch.cat([frames, action_map], dim=1))
        return out.reshape(batch, self.horizon, 3, height, width)


class ToyArenaAutoencoder(nn.Module):
    def __init__(self, latent_channels: int = 64):
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 48, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(48, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, latent_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(latent_channels, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 48, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(48, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encoder(frames)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(frames)
        return self.decode(latents), latents


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, hidden_channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None) -> torch.Tensor:
        if hidden is None:
            hidden = x.new_zeros(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3])
        combined = torch.cat([x, hidden], dim=1)
        reset, update = self.gates(combined).chunk(2, dim=1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * hidden], dim=1)))
        return (1.0 - update) * hidden + update * candidate


class ActionConditionedLatentDynamics(nn.Module):
    def __init__(
        self,
        action_count: int,
        latent_channels: int = 64,
        context: int = 8,
        horizon: int = 8,
        action_dim: int = 16,
        hidden_channels: int = 96,
    ):
        super().__init__()
        self.context = context
        self.horizon = horizon
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.action_dim = action_dim
        self.action_embed = nn.Embedding(action_count, action_dim)
        self.in_proj = nn.Conv2d(latent_channels + action_dim, hidden_channels, 3, padding=1)
        self.gru = ConvGRUCell(hidden_channels, hidden_channels)
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, latent_channels, 3, padding=1),
        )

    def _action_map(self, actions: torch.Tensor, height: int, width: int) -> torch.Tensor:
        action_tokens = self.action_embed(actions).view(actions.shape[0], self.action_dim, 1, 1)
        return action_tokens.expand(actions.shape[0], self.action_dim, height, width)

    def forward(self, context_latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch, context, channels, height, width = context_latents.shape
        if context != self.context or channels != self.latent_channels:
            raise ValueError("context_latents shape does not match model context/latent_channels.")
        hidden: torch.Tensor | None = None
        for step in range(context):
            noop_actions = torch.zeros(batch, dtype=torch.long, device=context_latents.device)
            x = torch.cat([context_latents[:, step], self._action_map(noop_actions, height, width)], dim=1)
            hidden = self.gru(self.in_proj(x), hidden)

        current = context_latents[:, -1]
        preds = []
        for step in range(actions.shape[1]):
            x = torch.cat([current, self._action_map(actions[:, step], height, width)], dim=1)
            hidden = self.gru(self.in_proj(x), hidden)
            delta = self.out_proj(hidden)
            current = current + delta
            preds.append(current)
        return torch.stack(preds, dim=1)
