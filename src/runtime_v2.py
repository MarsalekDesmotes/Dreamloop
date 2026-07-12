from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import torch

from src.eval_v2 import decode_state_probe
from src.model_v2 import (
    ArenaStateProbe,
    NeuralSemanticStateDynamics,
    SemanticLatentRenderer,
    SemanticRGBRenderer,
    SemanticSpriteRenderer,
    SemanticStateDynamics,
    StructuredSemanticStateDynamics,
    StreamingLatentDynamics,
    V2RGBNextFrame,
    V2RepresentationCodec,
    arena_state_probe_from_checkpoint,
)
from src.data_v2 import (
    decode_visible_collision_flash,
    decode_visible_game_state,
    estimate_action_conditioned_player_motion,
    estimate_semantic_velocity,
    fit_direct_player_trajectory,
    fit_discrete_enemy_trajectory,
)
from src.training_v2 import load_trusted_checkpoint


def frames_tensor(frames: list[np.ndarray] | np.ndarray, device: str) -> torch.Tensor:
    values = np.asarray(frames, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(values, (0, 3, 1, 2)).copy()).to(device)


class LatentWorldRuntime:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        project_latent: bool = False,
        quantize_step: float = 0.0,
    ):
        self.device = device
        self.project_latent = project_latent
        self.quantize_step = float(quantize_step)
        checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=device)
        self.checkpoint = checkpoint
        codec_checkpoint = load_trusted_checkpoint(checkpoint["codec"], map_location=device)
        self.codec = V2RepresentationCodec(
            int(codec_checkpoint["latent_channels"]), int(codec_checkpoint.get("semantic_dim", 0))
        ).to(device)
        self.codec.load_state_dict(codec_checkpoint["model"])
        self.codec.eval()
        self.model = StreamingLatentDynamics(
            action_count=int(checkpoint["action_count"]), latent_channels=int(checkpoint["latent_channels"])
        ).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        cache_metadata = json.loads((Path(checkpoint["cache"]) / "metadata.json").read_text(encoding="utf-8"))
        self.mean = torch.tensor(cache_metadata["mean"], device=device)[None, :, None, None]
        self.std = torch.tensor(cache_metadata["std"], device=device)[None, :, None, None]
        self.context = int(checkpoint["context"])
        self.current: torch.Tensor | None = None
        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    @torch.inference_mode()
    def initialize(self, frames: list[np.ndarray], actions: list[int]) -> None:
        if len(frames) != self.context or len(actions) != self.context - 1:
            raise ValueError("initial frames/actions do not match checkpoint context")
        frame_values = frames_tensor(frames, self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            raw = self.codec.encode(frame_values)
            normalized = (raw - self.mean) / self.std
            self.current, self.hidden = self.model.prefill(normalized[None], torch.tensor(actions, device=self.device)[None])

    @torch.inference_mode()
    def step(self, action: int) -> np.ndarray:
        if self.current is None or self.hidden is None:
            raise RuntimeError("runtime must be initialized")
        action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            self.current, self.hidden = self.model.step(self.current, action_tensor, self.hidden)
            if self.quantize_step > 0.0:
                self.current = torch.round(self.current / self.quantize_step) * self.quantize_step
            frame = self.codec.decode(self.current * self.std + self.mean)[0]
            if self.project_latent:
                projected = self.codec.encode(frame[None])
                self.current = (projected - self.mean) / self.std
        return np.clip(frame.float().permute(1, 2, 0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)


class RGBWorldRuntime:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=device)
        self.context = int(checkpoint["context"])
        self.model = V2RGBNextFrame(int(checkpoint["action_count"]), self.context).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.history: torch.Tensor | None = None

    @torch.inference_mode()
    def initialize(self, frames: list[np.ndarray], actions: list[int]) -> None:
        del actions
        if len(frames) != self.context:
            raise ValueError("initial frames do not match checkpoint context")
        self.history = frames_tensor(frames, self.device)[None]

    @torch.inference_mode()
    def step(self, action: int) -> np.ndarray:
        if self.history is None:
            raise RuntimeError("runtime must be initialized")
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            prediction = self.model(
                self.history.flatten(1, 2), torch.tensor([action], dtype=torch.long, device=self.device)
            )
        self.history = torch.cat([self.history[:, 1:], prediction[:, None]], dim=1)
        return np.clip(prediction[0].float().permute(1, 2, 0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)


class SemanticWorldRuntime:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        collision_threshold: float | None = None,
    ):
        self.device = device
        checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=device)
        self.context = int(checkpoint["context"])
        self.state_dim = int(checkpoint["state_dim"])
        if checkpoint.get("neural_v4", False):
            self.model = NeuralSemanticStateDynamics(
                int(checkpoint["action_count"]), int(checkpoint["state_dim"])
            ).to(device)
        elif checkpoint.get("structured_v3", False):
            self.model = StructuredSemanticStateDynamics(
                int(checkpoint["action_count"]),
                int(checkpoint["state_dim"]),
                collision_threshold=(
                    float(collision_threshold)
                    if collision_threshold is not None
                    else float(checkpoint.get("collision_threshold", 24.0))
                ),
            ).to(device)
        else:
            self.model = SemanticStateDynamics(
                int(checkpoint["action_count"]),
                int(checkpoint["state_dim"]),
                kinematic_base=bool(checkpoint.get("kinematic_base", False)),
            ).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        renderer_checkpoint = load_trusted_checkpoint(checkpoint["renderer"], map_location=device)
        renderer_type = renderer_checkpoint.get("model_type")
        self.direct_rgb_renderer = renderer_type in (
            "v2_semantic_rgb_renderer",
            "v2_semantic_conv_renderer",
            "v2_semantic_sprite_renderer",
        )
        if renderer_type == "v2_semantic_sprite_renderer":
            self.renderer = SemanticSpriteRenderer(
                int(renderer_checkpoint["state_dim"]),
                int(renderer_checkpoint["output_hw"]),
                int(renderer_checkpoint["sprite_size"]),
            ).to(device)
        elif self.direct_rgb_renderer:
            self.renderer = SemanticRGBRenderer(
                int(renderer_checkpoint["state_dim"]),
                int(renderer_checkpoint["feature_hw"]),
                int(renderer_checkpoint["output_hw"]),
            ).to(device)
        else:
            self.renderer = SemanticLatentRenderer(
                int(renderer_checkpoint["state_dim"]), int(renderer_checkpoint["latent_channels"])
            ).to(device)
        self.renderer.load_state_dict(renderer_checkpoint["model"])
        self.renderer.eval()
        self.codec = None
        self.mean = None
        self.std = None
        if not self.direct_rgb_renderer:
            codec_checkpoint = load_trusted_checkpoint(renderer_checkpoint["codec"], map_location=device)
            self.codec = V2RepresentationCodec(
                int(codec_checkpoint["latent_channels"]), int(codec_checkpoint.get("semantic_dim", 0))
            ).to(device)
            self.codec.load_state_dict(codec_checkpoint["model"])
            self.codec.eval()
        probe_checkpoint = load_trusted_checkpoint(checkpoint["semantic_cache_probe"], map_location=device)
        self.probe = arena_state_probe_from_checkpoint(probe_checkpoint).to(device)
        self.probe.load_state_dict(probe_checkpoint["state_probe"])
        self.probe.eval()
        if not self.direct_rgb_renderer:
            latent_metadata = json.loads(
                (Path(renderer_checkpoint["latent_cache"]) / "metadata.json").read_text(encoding="utf-8")
            )
            self.mean = torch.tensor(latent_metadata["mean"], device=device)[None, :, None, None]
            self.std = torch.tensor(latent_metadata["std"], device=device)[None, :, None, None]
        self.current: torch.Tensor | None = None
        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    @torch.inference_mode()
    def _states_from_frames(self, frames: list[np.ndarray], actions: list[int] | None = None) -> torch.Tensor:
        values = frames_tensor(frames, self.device)
        decoded = decode_state_probe(self.probe(values))
        player = decoded["player_pos"].float().cpu().numpy()
        coin = decoded["coin_pos"].float().cpu().numpy()
        enemies = decoded["enemy_pos"].float().cpu().numpy()
        permutations = np.asarray(list(itertools.permutations(range(3))), dtype=np.int64)
        first_order = np.lexsort((enemies[0, :, 1], enemies[0, :, 0]))
        enemies[0] = enemies[0, first_order]
        for index in range(1, len(enemies)):
            candidates = enemies[index][permutations]
            costs = np.linalg.norm(candidates - enemies[index - 1][None], axis=2).sum(axis=1)
            enemies[index] = candidates[int(costs.argmin())]
        enemies, enemy_velocity = fit_discrete_enemy_trajectory(enemies)
        fitted_player_velocity = None
        if actions is not None:
            player, fitted_player_velocity = fit_direct_player_trajectory(
                player, coin, enemies, np.asarray(actions, dtype=np.int64)
            )
        states = np.zeros((len(frames), self.state_dim), dtype=np.float32)
        states[:, 0:2] = player / 64.0 - 1.0
        states[:, 4:6] = coin / 64.0 - 1.0
        states[:, 6:12] = enemies.reshape(len(frames), 6) / 64.0 - 1.0
        if actions is None:
            states[:, 2:4] = estimate_semantic_velocity(states[:, 0:2], scale=8.0)
        elif fitted_player_velocity is None:
            _, player_velocity = estimate_action_conditioned_player_motion(player, coin, np.asarray(actions))
            states[:, 2:4] = player_velocity / 8.0
        else:
            states[:, 2:4] = fitted_player_velocity / 8.0
        states[:, 12:18] = (enemy_velocity / 4.0).reshape(-1, 6)
        states[:, 18] = decode_visible_collision_flash(np.asarray(frames))
        if self.state_dim >= 23:
            states[:, 19:23] = decode_visible_game_state(np.asarray(frames))
            terminal = states[:, 21:23].max(axis=1) > 0.5
            states[terminal, 2:4] = 0.0
            states[terminal, 12:18] = 0.0
        return torch.from_numpy(states).to(self.device)

    @torch.inference_mode()
    def initialize(self, frames: list[np.ndarray], actions: list[int]) -> None:
        if len(frames) != self.context or len(actions) != self.context - 1:
            raise ValueError("initial frames/actions do not match checkpoint context")
        states = self._states_from_frames(frames, actions)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            self.current, self.hidden = self.model.prefill(
                states[None], torch.tensor(actions, dtype=torch.long, device=self.device)[None]
            )

    @torch.inference_mode()
    def step(self, action: int) -> np.ndarray:
        if self.current is None or self.hidden is None:
            raise RuntimeError("runtime must be initialized")
        action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            self.current, self.hidden = self.model.step(self.current, action_tensor, self.hidden)
            self.current[:, [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]].clamp_(-0.95, 0.95)
            self.current[:, 2:4].clamp_(-1.5, 1.5)
            self.current[:, 12:18].clamp_(-1.5, 1.5)
            self.current[:, 18:19].clamp_(0.0, 1.0)
            if self.state_dim >= 23:
                self.current[:, 19:23].clamp_(0.0, 1.0)
            if self.direct_rgb_renderer:
                frame = self.renderer(self.current)[0]
            else:
                normalized = self.renderer(self.current)
                frame = self.codec.decode(normalized * self.std + self.mean)[0]
        return np.clip(frame.float().permute(1, 2, 0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
