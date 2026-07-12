from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class V2RepresentationCodec(nn.Module):
    def __init__(self, latent_channels: int = 64, semantic_dim: int = 0):
        super().__init__()
        self.latent_channels = latent_channels
        self.semantic_dim = semantic_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            ResidualBlock(32),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            ResidualBlock(64),
            nn.Conv2d(64, 96, 4, stride=2, padding=1),
            ResidualBlock(96),
            nn.Conv2d(96, latent_channels, 4, stride=2, padding=1),
            ResidualBlock(latent_channels),
            nn.GroupNorm(8, latent_channels),
        )
        self.decoder = nn.Sequential(
            ResidualBlock(latent_channels),
            nn.ConvTranspose2d(latent_channels, 96, 4, stride=2, padding=1),
            ResidualBlock(96),
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            ResidualBlock(32),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )
        self.semantic_projection = nn.Conv2d(latent_channels, semantic_dim, 1) if semantic_dim > 0 else None

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encoder(frames)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def project_semantic(self, latents: torch.Tensor) -> torch.Tensor:
        if self.semantic_projection is None:
            raise RuntimeError("codec was created without a semantic projection")
        return self.semantic_projection(latents)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(frames)
        return self.decode(latents), latents


class ConvGRUCellV2(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, hidden_channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=1)

    def forward(self, value: torch.Tensor, hidden: torch.Tensor | None) -> torch.Tensor:
        if hidden is None:
            hidden = value.new_zeros(value.shape[0], self.hidden_channels, value.shape[2], value.shape[3])
        reset, update = torch.sigmoid(self.gates(torch.cat([value, hidden], dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([value, reset * hidden], dim=1)))
        return (1.0 - update) * hidden + update * candidate


class StreamingLatentDynamics(nn.Module):
    def __init__(
        self,
        action_count: int = 6,
        latent_channels: int = 64,
        action_dim: int = 32,
        hidden_channels: int = 128,
        latent_hw: int = 16,
    ):
        super().__init__()
        self.action_count = action_count
        self.latent_channels = latent_channels
        self.action_dim = action_dim
        self.hidden_channels = hidden_channels
        self.latent_hw = latent_hw
        self.action_embedding = nn.Embedding(action_count, action_dim)
        self.input_projection = nn.Sequential(
            nn.Conv2d(latent_channels + action_dim + 2, hidden_channels, 3, padding=1),
            nn.SiLU(),
            ResidualBlock(hidden_channels),
        )
        self.gru1 = ConvGRUCellV2(hidden_channels, hidden_channels)
        self.gru2 = ConvGRUCellV2(hidden_channels, hidden_channels)
        self.output_projection = nn.Sequential(
            ResidualBlock(hidden_channels),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, latent_channels, 3, padding=1),
        )
        nn.init.zeros_(self.output_projection[-1].weight)
        nn.init.zeros_(self.output_projection[-1].bias)

        axis = torch.linspace(-1.0, 1.0, latent_hw)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("coordinates", torch.stack([xx, yy], dim=0)[None], persistent=False)

    def _condition(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = latent.shape
        if (height, width) != (self.latent_hw, self.latent_hw):
            raise ValueError(f"expected {self.latent_hw}x{self.latent_hw} latents")
        action_map = self.action_embedding(action).view(batch, self.action_dim, 1, 1).expand(-1, -1, height, width)
        coords = self.coordinates.to(dtype=latent.dtype).expand(batch, -1, -1, -1)
        return self.input_projection(torch.cat([latent, action_map, coords], dim=1))

    def step(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        hidden: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hidden1, hidden2 = hidden if hidden is not None else (None, None)
        conditioned = self._condition(latent, action)
        hidden1 = self.gru1(conditioned, hidden1)
        hidden2 = self.gru2(hidden1, hidden2)
        return latent + self.output_projection(hidden2), (hidden1, hidden2)

    def prefill(
        self,
        context_latents: torch.Tensor,
        context_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context_latents.ndim != 5:
            raise ValueError("context_latents must have shape [B,T,C,H,W]")
        if context_actions.shape[1] != context_latents.shape[1] - 1:
            raise ValueError("context_actions must contain one transition per context frame pair")
        hidden = None
        prediction = context_latents[:, 0]
        for step in range(context_actions.shape[1]):
            prediction, hidden = self.step(context_latents[:, step], context_actions[:, step], hidden)
        if hidden is None:
            raise ValueError("prefill requires at least two context frames")
        return context_latents[:, -1], hidden

    @staticmethod
    def detach_hidden(hidden: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden[0].detach(), hidden[1].detach()


class SemanticStateDynamics(nn.Module):
    def __init__(
        self,
        action_count: int = 6,
        state_dim: int = 19,
        action_dim: int = 16,
        hidden_dim: int = 256,
        kinematic_base: bool = True,
    ):
        super().__init__()
        self.action_count = action_count
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.kinematic_base = kinematic_base
        self.action_embedding = nn.Embedding(action_count, action_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.gru1 = nn.GRUCell(128, hidden_dim)
        self.gru2 = nn.GRUCell(hidden_dim, hidden_dim)
        self.output = nn.Sequential(nn.Linear(hidden_dim, 256), nn.SiLU(), nn.Linear(256, state_dim))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hidden1, hidden2 = hidden if hidden is not None else (None, None)
        value = self.input_projection(torch.cat([state, self.action_embedding(action)], dim=1))
        if hidden1 is None:
            hidden1 = value.new_zeros(len(value), self.hidden_dim)
        if hidden2 is None:
            hidden2 = value.new_zeros(len(value), self.hidden_dim)
        hidden1 = self.gru1(value, hidden1)
        hidden2 = self.gru2(hidden1, hidden2)
        base = state
        if self.kinematic_base:
            base = state.clone()
            base[:, 0:2] = state[:, 0:2] + state[:, 2:4] / 8.0
            base[:, 6:12] = state[:, 6:12] + state[:, 12:18] / 16.0
            base[:, 18:19] = state[:, 18:19] * 0.5
        return base + self.output(hidden2), (hidden1, hidden2)

    def prefill(
        self,
        context_states: torch.Tensor,
        context_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if context_states.ndim != 3 or context_states.shape[-1] != self.state_dim:
            raise ValueError("context_states must have shape [B,T,state_dim]")
        if context_actions.shape[1] != context_states.shape[1] - 1:
            raise ValueError("context_actions must contain one transition per state pair")
        hidden = None
        for step in range(context_actions.shape[1]):
            _, hidden = self.step(context_states[:, step], context_actions[:, step], hidden)
        if hidden is None:
            raise ValueError("prefill requires at least two context states")
        return context_states[:, -1], hidden

    @staticmethod
    def detach_hidden(hidden: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden[0].detach(), hidden[1].detach()


class NeuralSemanticStateDynamics(nn.Module):
    """Fully learned streaming transition over the observable 23-value game state."""

    def __init__(
        self,
        action_count: int = 6,
        state_dim: int = 23,
        action_dim: int = 24,
        hidden_dim: int = 384,
    ):
        super().__init__()
        if state_dim != 23:
            raise ValueError("neural semantic dynamics requires the 23-value gameplay state")
        self.action_count = action_count
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_embedding = nn.Embedding(action_count, action_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
        )
        self.gru1 = nn.GRUCell(256, hidden_dim)
        self.gru2 = nn.GRUCell(hidden_dim, hidden_dim)
        self.output = nn.Sequential(nn.Linear(hidden_dim, 384), nn.SiLU(), nn.Linear(384, state_dim))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hidden1, hidden2 = hidden if hidden is not None else (None, None)
        value = self.input_projection(torch.cat([state, self.action_embedding(action)], dim=1))
        if hidden1 is None:
            hidden1 = value.new_zeros(len(value), self.hidden_dim)
        if hidden2 is None:
            hidden2 = value.new_zeros(len(value), self.hidden_dim)
        hidden1 = self.gru1(value, hidden1)
        hidden2 = self.gru2(hidden1, hidden2)
        raw = state + self.output(hidden2)
        prediction = torch.cat(
            [
                raw[:, 0:2].clamp(-0.95, 0.95),
                raw[:, 2:4].clamp(-1.5, 1.5),
                raw[:, 4:12].clamp(-0.95, 0.95),
                raw[:, 12:18].clamp(-1.5, 1.5),
                raw[:, 18:23].clamp(0.0, 1.0),
            ],
            dim=1,
        )
        terminal = state[:, 21:23].max(dim=1, keepdim=True).values > 0.5
        prediction = torch.where(terminal, state, prediction)
        return prediction, (hidden1, hidden2)

    def prefill(
        self, context_states: torch.Tensor, context_actions: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if context_states.ndim != 3 or context_states.shape[-1] != self.state_dim:
            raise ValueError("context_states must have shape [B,T,state_dim]")
        if context_actions.shape[1] != context_states.shape[1] - 1:
            raise ValueError("context_actions must contain one transition per state pair")
        hidden = None
        for step in range(context_actions.shape[1]):
            _, hidden = self.step(context_states[:, step], context_actions[:, step], hidden)
        if hidden is None:
            raise ValueError("prefill requires at least two context states")
        return context_states[:, -1], hidden

    @staticmethod
    def detach_hidden(hidden: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(value.detach() for value in hidden)


class StructuredSemanticStateDynamics(nn.Module):
    """Streaming semantic dynamics with stable geometry and learned residual/event heads."""

    def __init__(
        self,
        action_count: int = 6,
        state_dim: int = 19,
        action_dim: int = 16,
        hidden_dim: int = 256,
        collision_threshold: float = 24.0,
    ):
        super().__init__()
        if state_dim not in (19, 23):
            raise ValueError("structured semantic dynamics requires the 19-value or 23-value state")
        self.action_count = action_count
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.collision_threshold = float(collision_threshold)
        self.action_embedding = nn.Embedding(action_count, action_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU()
        )
        self.gru1 = nn.GRUCell(128, hidden_dim)
        self.gru2 = nn.GRUCell(hidden_dim, hidden_dim)
        self.residual_head = nn.Sequential(nn.Linear(hidden_dim, 256), nn.SiLU(), nn.Linear(256, 17))
        self.coin_gate = nn.Sequential(nn.Linear(hidden_dim + 4, 128), nn.SiLU(), nn.Linear(128, 1))
        self.collision_gate = nn.Sequential(nn.Linear(hidden_dim + 4, 128), nn.SiLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.coin_gate[-1].weight)
        nn.init.constant_(self.coin_gate[-1].bias, -4.0)
        nn.init.zeros_(self.collision_gate[-1].weight)
        nn.init.constant_(self.collision_gate[-1].bias, -4.0)
        pads = torch.tensor(
            [(24, 24), (64, 20), (104, 24), (108, 64), (104, 104), (64, 108), (24, 104), (20, 64)],
            dtype=torch.float32,
        )
        self.register_buffer("coin_pads", pads / 64.0 - 1.0)
        self.last_auxiliary: dict[str, torch.Tensor] = {}

    @staticmethod
    def _unit(value: torch.Tensor) -> torch.Tensor:
        return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-5)

    def _physics_base(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        direction = state.new_zeros(len(state), 2)
        direction[:, 1] = (action == 2).to(state.dtype) - (action == 1).to(state.dtype)
        direction[:, 0] = (action == 4).to(state.dtype) - (action == 3).to(state.dtype)
        player_velocity = state.new_zeros(len(state), 2)
        moving = ((action >= 1) & (action <= 4)).to(state.dtype)[:, None]
        player_velocity = direction * (2.4 / 8.0) * moving
        dash = action == 5
        coin_direction = self._unit(state[:, 4:6] - state[:, 0:2])
        player_velocity = torch.where(dash[:, None], coin_direction * (5.6 / 8.0), player_velocity)
        player_position = state[:, 0:2] + player_velocity / 8.0
        boundary = 1.0 - 14.0 / 64.0
        hit = (player_position < -boundary) | (player_position > boundary)
        player_velocity = torch.where(hit, player_velocity * -0.35, player_velocity)
        player_position = player_position.clamp(-boundary, boundary)

        enemy_position = state[:, 6:12].reshape(-1, 3, 2)
        enemy_velocity = state[:, 12:18].reshape(-1, 3, 2)
        next_enemy = enemy_position + enemy_velocity / 16.0
        enemy_hit = (next_enemy < -boundary) | (next_enemy > boundary)
        enemy_velocity = torch.where(enemy_hit, -enemy_velocity, enemy_velocity)
        next_enemy = next_enemy.clamp(-boundary, boundary)

        nearest_pad = torch.cdist(state[:, 4:6].float(), self.coin_pads.float()).argmin(dim=1)
        current_coin = self.coin_pads[nearest_pad].to(state.dtype)
        base = torch.cat(
            [
                player_position,
                player_velocity,
                current_coin,
                next_enemy.flatten(1),
                enemy_velocity.flatten(1),
                state[:, 18:19] * 0.85,
                state[:, 19:] if self.state_dim > 19 else state[:, 19:19],
            ],
            dim=1,
        )
        return base, nearest_pad

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden: tuple[torch.Tensor | None, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        terminal = (
            state[:, 21:23].max(dim=1, keepdim=True).values > 0.5
            if self.state_dim > 19
            else torch.zeros(len(state), 1, dtype=torch.bool, device=state.device)
        )
        hidden1 = hidden[0] if hidden is not None else None
        hidden2 = hidden[1] if hidden is not None else None
        cooldown = hidden[2] if hidden is not None and len(hidden) > 2 else None
        previous_flash = hidden[3] if hidden is not None and len(hidden) > 3 else None
        flash_timer = hidden[4] if hidden is not None and len(hidden) > 4 else None
        value = self.input_projection(torch.cat([state, self.action_embedding(action)], dim=1))
        if hidden1 is None:
            hidden1 = value.new_zeros(len(value), self.hidden_dim)
        if hidden2 is None:
            hidden2 = value.new_zeros(len(value), self.hidden_dim)
        if cooldown is None:
            cooldown = value.new_zeros(len(value), 1)
        if previous_flash is None:
            previous_flash = value.new_zeros(len(value), 1)
        if flash_timer is None:
            flash_timer = value.new_zeros(len(value), 1)
        hidden1 = self.gru1(value, hidden1)
        hidden2 = self.gru2(hidden1, hidden2)
        observed_flash = (state[:, 18:19] > 0.5).to(state.dtype)
        flash_started = (observed_flash > 0.5) & (previous_flash <= 0.5)
        cooldown = torch.where(flash_started, torch.full_like(cooldown, 36.0), cooldown)
        flash_timer = torch.where(flash_started, torch.full_like(flash_timer, 8.0), flash_timer)
        cooldown = (cooldown - 1.0).clamp_min(0.0)
        flash_timer = (flash_timer - 1.0).clamp_min(0.0)
        base, current_pad = self._physics_base(state, action)
        residual = self.residual_head(hidden2)
        state_enemies = state[:, 6:12].reshape(-1, 3, 2)
        enemy_clearance = torch.linalg.vector_norm(
            state_enemies - state[:, None, 0:2], dim=2
        ).amin(dim=1, keepdim=True)
        coin_clearance = torch.linalg.vector_norm(state[:, 4:6] - state[:, 0:2], dim=1, keepdim=True)
        wall_clearance = (1.0 - 14.0 / 64.0) - state[:, 0:2].abs().amax(dim=1, keepdim=True)
        safe_residual = (
            (enemy_clearance > 40.0 / 64.0)
            & (coin_clearance > 24.0 / 64.0)
            & (wall_clearance > 6.0 / 64.0)
        ).to(state.dtype)
        residual = residual * safe_residual
        # The deterministic base already captures the dominant motion. Keep the
        # learned residual in a micro-correction range so a tiny one-step bias
        # cannot accumulate into multi-pixel drift over a 10-60 second rollout.
        player_position = base[:, 0:2] + torch.tanh(residual[:, 0:2]) * 0.0001
        player_velocity = base[:, 2:4] + torch.tanh(residual[:, 2:4]) * 0.002
        enemy_position = base[:, 6:12] + torch.tanh(residual[:, 4:10]) * 0.0001
        enemy_velocity = base[:, 12:18] + torch.tanh(residual[:, 10:16]) * 0.002
        flash = (flash_timer > 0.0).to(state.dtype)
        enemies = enemy_position.reshape(-1, 3, 2)
        enemy_velocities = enemy_velocity.reshape(-1, 3, 2)
        contact_delta = player_position[:, None] - enemies
        contact_distance = torch.linalg.vector_norm(contact_delta, dim=2, keepdim=True)
        contact_away = contact_delta / contact_distance.clamp_min(1e-5)
        contact_strength = ((24.0 / 64.0 - contact_distance) / (24.0 / 64.0)).clamp(0.0, 1.0)
        contact_force = (contact_away * contact_strength).sum(dim=1)
        player_velocity = player_velocity + contact_force * (1.2 / 8.0)
        player_position = (player_position + contact_force * (1.5 / 64.0)).clamp(
            -(1.0 - 14.0 / 64.0), 1.0 - 14.0 / 64.0
        )
        enemy_distances = torch.linalg.vector_norm(enemies - player_position[:, None], dim=2)
        nearest_enemy = enemy_distances.argmin(dim=1)
        nearest_distance = enemy_distances.gather(1, nearest_enemy[:, None])
        collision_logits = self.collision_gate(
            torch.cat(
                [hidden2, nearest_distance, state[:, 18:19], action[:, None].to(state.dtype) / 5.0, torch.ones_like(nearest_distance)],
                dim=1,
            )
        )
        collision_probability = torch.sigmoid(collision_logits)
        nearest_velocity = enemy_velocities.gather(
            1, nearest_enemy[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        to_enemy = enemies.gather(1, nearest_enemy[:, None, None].expand(-1, 1, 2)).squeeze(1) - player_position
        closing = ((player_velocity * 8.0 - nearest_velocity * 4.0) * to_enemy).sum(dim=1, keepdim=True) > 0.0
        damage_contact = (nearest_distance < (self.collision_threshold / 64.0)) & (
            closing | (nearest_distance < (15.0 / 64.0))
        )
        collision = (damage_contact & (cooldown <= 0.0)).to(state.dtype)
        cooldown = torch.where(collision > 0.5, torch.full_like(cooldown, 36.0), cooldown)
        flash_timer = torch.where(collision > 0.5, torch.full_like(flash_timer, 8.0), flash_timer)
        flash = (flash_timer > 0.0).to(state.dtype)
        base = torch.cat(
            [
                player_position,
                player_velocity,
                base[:, 4:6],
                enemy_position,
                enemy_velocities.flatten(1),
                flash,
                state[:, 19:] if self.state_dim > 19 else state[:, 19:19],
            ],
            dim=1,
        )

        distance = torch.linalg.vector_norm(base[:, 0:2] - base[:, 4:6], dim=1, keepdim=True)
        gate_logits = self.coin_gate(torch.cat([hidden2, distance, state[:, 18:19], action[:, None].to(state.dtype) / 5.0, torch.ones_like(distance)], dim=1))
        probability = torch.sigmoid(gate_logits)
        gate = (distance < (12.0 / 64.0)).to(state.dtype)
        next_pad = (current_pad + 3) % len(self.coin_pads)
        next_coin = self.coin_pads[next_pad].to(state.dtype)
        coin = base[:, 4:6] * (1.0 - gate) + next_coin * gate
        base = torch.cat([base[:, 0:4], coin, base[:, 6:]], dim=1)
        if self.state_dim > 19:
            health = (state[:, 19:20] - collision / 3.0).clamp(0.0, 1.0)
            progress = (state[:, 20:21] + gate / 3.0).clamp(0.0, 1.0)
            lost = (health <= 1e-4).to(state.dtype)
            portal_distance = torch.linalg.vector_norm(player_position, dim=1, keepdim=True)
            won = ((progress >= 1.0 - 1e-4) & (portal_distance < 13.0 / 64.0) & (lost < 0.5)).to(state.dtype)
            gameplay = torch.cat([health, progress, won, lost], dim=1)
            base = torch.cat([base[:, :19], gameplay], dim=1)
        base = torch.where(terminal, state, base)
        self.last_auxiliary = {
            "coin_gate_logits": gate_logits,
            "coin_pad": next_pad,
            "coin_probability": probability,
            "collision_gate_logits": collision_logits,
            "collision_probability": collision_probability,
        }
        return base, (hidden1, hidden2, cooldown, flash, flash_timer)

    def prefill(self, context_states: torch.Tensor, context_actions: torch.Tensor):
        if context_states.ndim != 3 or context_states.shape[-1] != self.state_dim:
            raise ValueError("context_states must have shape [B,T,state_dim]")
        hidden = None
        for step in range(context_actions.shape[1]):
            _, hidden = self.step(context_states[:, step], context_actions[:, step], hidden)
        if hidden is None:
            raise ValueError("prefill requires at least two context states")
        return context_states[:, -1], hidden

    @staticmethod
    def detach_hidden(hidden: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(value.detach() for value in hidden)


class SemanticLatentRenderer(nn.Module):
    def __init__(self, state_dim: int = 19, latent_channels: int = 64, latent_hw: int = 16):
        super().__init__()
        self.state_dim = state_dim
        self.latent_channels = latent_channels
        self.latent_hw = latent_hw
        self.base_latent = nn.Parameter(torch.zeros(1, latent_channels, latent_hw, latent_hw))
        self.net = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, latent_channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        axis = torch.linspace(-1.0, 1.0, latent_hw)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("coordinates", torch.stack([xx, yy], dim=0)[None], persistent=False)

    def spatial_features(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError("state must have shape [B,state_dim]")
        positions = torch.cat([state[:, 0:2], state[:, 4:6], state[:, 6:12]], dim=1).reshape(-1, 5, 2)
        coords = self.coordinates.to(dtype=state.dtype).expand(len(state), -1, -1, -1)
        delta = coords[:, None] - positions[:, :, :, None, None]
        heatmaps = torch.exp(-delta.square().sum(dim=2) / (2.0 * 0.085**2))
        player_velocity = heatmaps[:, 0:1] * state[:, 2:4, None, None]
        enemy_velocity = state[:, 12:18].reshape(-1, 3, 2)
        enemy_velocity_maps = (heatmaps[:, 2:5, None] * enemy_velocity[:, :, :, None, None]).flatten(1, 2)
        flash = state[:, 18:19, None, None].expand(-1, -1, self.latent_hw, self.latent_hw)
        return torch.cat([heatmaps, player_velocity, enemy_velocity_maps, coords, flash], dim=1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.base_latent + self.net(self.spatial_features(state))


class SemanticRGBRenderer(nn.Module):
    """Direct semantic-state RGB decoder that preserves sharp static and object detail."""

    def __init__(self, state_dim: int = 19, feature_hw: int = 64, output_hw: int = 128):
        super().__init__()
        if output_hw != feature_hw * 2:
            raise ValueError("output_hw must be twice feature_hw")
        self.state_dim = state_dim
        self.feature_hw = feature_hw
        self.output_hw = output_hw
        self.base_logits = nn.Parameter(torch.zeros(1, 3, output_hw, output_hw))
        self.net = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 48, 3, padding=1),
            ResidualBlock(48),
            ResidualBlock(48),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, 3, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        axis = torch.linspace(-1.0, 1.0, feature_hw)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("coordinates", torch.stack([xx, yy], dim=0)[None], persistent=False)

    def spatial_features(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError("state must have shape [B,state_dim]")
        positions = torch.cat([state[:, 0:2], state[:, 4:6], state[:, 6:12]], dim=1).reshape(-1, 5, 2)
        coords = self.coordinates.to(dtype=state.dtype).expand(len(state), -1, -1, -1)
        delta = coords[:, None] - positions[:, :, :, None, None]
        heatmaps = torch.exp(-delta.square().sum(dim=2) / (2.0 * 0.065**2))
        player_velocity = heatmaps[:, 0:1] * state[:, 2:4, None, None]
        enemy_velocity = state[:, 12:18].reshape(-1, 3, 2)
        enemy_velocity_maps = (heatmaps[:, 2:5, None] * enemy_velocity[:, :, :, None, None]).flatten(1, 2)
        flash = state[:, 18:19, None, None].expand(-1, -1, self.feature_hw, self.feature_hw)
        return torch.cat([heatmaps, player_velocity, enemy_velocity_maps, coords, flash], dim=1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.base_logits + self.net(self.spatial_features(state)))


class SemanticSpriteRenderer(nn.Module):
    """Differentiable learned RGBA atlas renderer for deterministic semantic frames."""

    def __init__(self, state_dim: int = 19, output_hw: int = 128, sprite_size: int = 33):
        super().__init__()
        self.state_dim = state_dim
        self.output_hw = output_hw
        self.sprite_size = sprite_size
        self.base_logits = nn.Parameter(torch.zeros(1, 3, output_hw, output_hw))
        self.coin_sprite = nn.Parameter(self._initial_sprite(sprite_size, 0.72))
        self.enemy_sprite = nn.Parameter(self._initial_sprite(sprite_size, 0.62))
        self.trail_sprite = nn.Parameter(self._initial_sprite(sprite_size, 0.48))
        self.flash_sprite = nn.Parameter(self._initial_ring(sprite_size, 0.62, 0.88))
        self.player_sprite = nn.Parameter(self._initial_sprite(sprite_size, 0.66))
        axis = torch.linspace(-1.0, 1.0, output_hw)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("coordinates", torch.stack([xx, yy], dim=-1)[None], persistent=False)
        if state_dim >= 23:
            self._register_gameplay_masks()

    def _register_gameplay_masks(self) -> None:
        size = self.output_hw
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")

        def disk(cx: int, cy: int, radius: float) -> torch.Tensor:
            return (((xx - cx).float().square() + (yy - cy).float().square()) <= radius**2).float()[None, None]

        def rect(x: int, y: int, width: int, height: int) -> torch.Tensor:
            mask = torch.zeros(1, 1, size, size)
            mask[:, :, y : y + height, x : x + width] = 1.0
            return mask

        def outline_rect(x: int, y: int, width: int, height: int, thickness: int = 2) -> torch.Tensor:
            return (
                rect(x, y, width, thickness)
                + rect(x, y + height - thickness, width, thickness)
                + rect(x, y, thickness, height)
                + rect(x + width - thickness, y, thickness, height)
            ).clamp_max(1.0)

        def ring(cx: int, cy: int, inner: float, outer: float) -> torch.Tensor:
            distance = (xx - cx).float().square() + (yy - cy).float().square()
            return ((distance >= inner**2) & (distance <= outer**2)).float()[None, None]

        def text_mask(text: str, x: int, y: int, scale: int = 3) -> torch.Tensor:
            glyphs = {
                "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
                "I": ("111", "010", "010", "010", "010", "010", "111"),
                "N": ("1001", "1101", "1101", "1011", "1011", "1001", "1001"),
                "L": ("100", "100", "100", "100", "100", "100", "111"),
                "O": ("0110", "1001", "1001", "1001", "1001", "1001", "0110"),
                "S": ("0111", "1000", "1000", "0110", "0001", "0001", "1110"),
                "E": ("111", "100", "100", "110", "100", "100", "111"),
            }
            mask = torch.zeros(1, 1, size, size)
            cursor = x
            for character in text:
                glyph = glyphs[character]
                for row, bits in enumerate(glyph):
                    for column, bit in enumerate(bits):
                        if bit == "1":
                            mask[:, :, y + row * scale : y + (row + 1) * scale, cursor + column * scale : cursor + (column + 1) * scale] = 1.0
                cursor += (len(glyph[0]) + 1) * scale
            return mask

        for index, center_x in enumerate((8, 17, 26)):
            self.register_buffer(f"health_mask_{index}", disk(center_x, 7, 3.0), persistent=False)
        for index, start_x in enumerate((101, 108, 115)):
            self.register_buffer(f"progress_mask_{index}", rect(start_x, 4, 5, 6), persistent=False)
        self.register_buffer("portal_outer_mask", ring(64, 64, 10.0, 12.0), persistent=False)
        self.register_buffer("portal_inner_mask", ring(64, 64, 4.0, 6.0), persistent=False)
        self.register_buffer("win_rect_mask", outline_rect(20, 48, 88, 32), persistent=False)
        self.register_buffer("lose_rect_mask", outline_rect(14, 48, 100, 32), persistent=False)
        self.register_buffer("win_text_mask", text_mask("WIN", 31, 53), persistent=False)
        self.register_buffer("lose_text_mask", text_mask("LOSE", 22, 53), persistent=False)

    @staticmethod
    def _paint(canvas: torch.Tensor, mask: torch.Tensor, color: tuple[int, int, int], opacity: torch.Tensor | None = None) -> torch.Tensor:
        alpha = mask.to(device=canvas.device, dtype=canvas.dtype)
        if opacity is not None:
            alpha = alpha * opacity[:, :, None, None]
        value = canvas.new_tensor(color)[None, :, None, None] / 255.0
        return canvas * (1.0 - alpha) + value * alpha

    @staticmethod
    def _initial_sprite(size: int, radius: float) -> torch.Tensor:
        axis = torch.linspace(-1.0, 1.0, size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        inside = (xx.square() + yy.square()) <= radius**2
        alpha = torch.where(inside, torch.full_like(xx, 2.0), torch.full_like(xx, -7.0))
        return torch.cat([torch.zeros(3, size, size), alpha[None]], dim=0)[None]

    @staticmethod
    def _initial_ring(size: int, inner: float, outer: float) -> torch.Tensor:
        axis = torch.linspace(-1.0, 1.0, size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        inside = (radius >= inner) & (radius <= outer)
        alpha = torch.where(inside, torch.full_like(xx, 2.0), torch.full_like(xx, -7.0))
        return torch.cat([torch.zeros(3, size, size), alpha[None]], dim=0)[None]

    def _stamp(
        self,
        canvas: torch.Tensor,
        sprite: torch.Tensor,
        center: torch.Tensor,
        half_extent: float,
        opacity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grid = (self.coordinates.to(dtype=canvas.dtype) - center[:, None, None]) / half_extent
        layer = F.grid_sample(
            sprite.to(dtype=canvas.dtype).expand(len(canvas), -1, -1, -1),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        rgb = torch.sigmoid(layer[:, 0:3])
        valid = (grid.abs() <= 1.0).all(dim=-1)[:, None].to(canvas.dtype)
        alpha = torch.sigmoid(layer[:, 3:4]) * valid
        if opacity is not None:
            alpha = alpha * opacity[:, :, None, None]
        return canvas * (1.0 - alpha) + rgb * alpha

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError("state must have shape [B,state_dim]")
        canvas = torch.sigmoid(self.base_logits).to(dtype=state.dtype).expand(len(state), -1, -1, -1)
        if self.state_dim >= 23:
            progress = state[:, 20:21].clamp(0.0, 1.0) * 3.0
            unlocked = (progress >= 2.5).to(state.dtype)
            canvas = self._paint(canvas, self.portal_outer_mask, (88, 248, 184), unlocked)
            canvas = self._paint(canvas, self.portal_inner_mask, (93, 185, 255), unlocked)
        canvas = self._stamp(canvas, self.coin_sprite, state[:, 4:6], 12.0 / 64.0)
        enemies = state[:, 6:12].reshape(-1, 3, 2)
        for index in range(3):
            canvas = self._stamp(canvas, self.enemy_sprite, enemies[:, index], 11.0 / 64.0)
        for trail in (0.75, 0.50, 0.25):
            center = state[:, 0:2] - state[:, 2:4] * (trail / 8.0)
            canvas = self._stamp(canvas, self.trail_sprite, center, 8.0 / 64.0)
        canvas = self._stamp(
            canvas, self.flash_sprite, state[:, 0:2], 15.0 / 64.0, state[:, 18:19].clamp(0.0, 1.0)
        )
        canvas = self._stamp(canvas, self.player_sprite, state[:, 0:2], 11.0 / 64.0)
        if self.state_dim >= 23:
            health = state[:, 19:20].clamp(0.0, 1.0) * 3.0
            progress = state[:, 20:21].clamp(0.0, 1.0) * 3.0
            for index in range(3):
                mask = getattr(self, f"health_mask_{index}")
                canvas = self._paint(canvas, mask, (62, 48, 59))
                canvas = self._paint(canvas, mask, (247, 79, 91), (health > index + 0.5).to(state.dtype))
                mask = getattr(self, f"progress_mask_{index}")
                canvas = self._paint(canvas, mask, (48, 63, 73))
                canvas = self._paint(canvas, mask, (88, 248, 184), (progress > index + 0.5).to(state.dtype))
            won = (state[:, 21:22] > 0.5).to(state.dtype)
            lost = (state[:, 22:23] > 0.5).to(state.dtype)
            terminal = torch.maximum(won, lost)
            overlay = canvas.new_tensor((8, 12, 18))[None, :, None, None] / 255.0
            alpha = terminal[:, :, None, None] * (1.0 / 3.0)
            canvas = canvas * (1.0 - alpha) + overlay * alpha
            canvas = self._paint(canvas, self.win_rect_mask, (19, 57, 52), won)
            canvas = self._paint(canvas, self.win_text_mask, (112, 255, 196), won)
            canvas = self._paint(canvas, self.lose_rect_mask, (67, 26, 35), lost)
            canvas = self._paint(canvas, self.lose_text_mask, (255, 126, 136), lost)
        return canvas


class ArenaStateProbe(nn.Module):
    """Predicts player, coin and combined-enemy heatmaps at 32x32."""

    def __init__(self, output_resolution: int = 32):
        super().__init__()
        if output_resolution not in (32, 64):
            raise ValueError("state probe output_resolution must be 32 or 64")
        self.output_resolution = output_resolution
        second = (
            nn.Conv2d(32, 64, 4, stride=2, padding=1)
            if output_resolution == 32
            else nn.Conv2d(32, 64, 3, stride=1, padding=1)
        )
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            second,
            nn.SiLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 3, 1),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.net(frames)


def arena_state_probe_from_checkpoint(checkpoint: dict) -> ArenaStateProbe:
    return ArenaStateProbe(int(checkpoint.get("state_resolution", 32)))


class InverseDynamicsProbe(nn.Module):
    def __init__(self, action_count: int = 6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(15, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 96, 4, stride=2, padding=1),
            nn.SiLU(),
            ResidualBlock(96),
        )
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.maximum_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(192, 128), nn.SiLU(), nn.Linear(128, action_count))

    def forward(self, previous: torch.Tensor, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        features = self.encoder(
            torch.cat([previous, before, after, before - previous, after - before], dim=1)
        )
        features = torch.cat([self.average_pool(features), self.maximum_pool(features)], dim=1).flatten(1)
        return self.classifier(features)


class V2RGBNextFrame(nn.Module):
    def __init__(self, action_count: int = 6, context: int = 8, action_dim: int = 16):
        super().__init__()
        self.context = context
        self.action_dim = action_dim
        self.action_embedding = nn.Embedding(action_count, action_dim)
        self.net = nn.Sequential(
            nn.Conv2d(context * 3 + action_dim, 64, 3, padding=1),
            nn.SiLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 96, 4, stride=2, padding=1),
            ResidualBlock(96),
            nn.Conv2d(96, 128, 4, stride=2, padding=1),
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 96, 4, stride=2, padding=1),
            ResidualBlock(96),
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1),
            ResidualBlock(64),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, frames: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = frames.shape
        action_map = self.action_embedding(action).view(batch, self.action_dim, 1, 1).expand(-1, -1, height, width)
        return self.net(torch.cat([frames, action_map], dim=1))
