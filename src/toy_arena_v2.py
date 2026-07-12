from __future__ import annotations

from dataclasses import dataclass

import numpy as np


NOOP, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, DASH = range(6)
ACTION_COUNT = 6

EVENT_COIN = 1 << 0
EVENT_COLLISION = 1 << 1
EVENT_DASH = 1 << 2
EVENT_EPISODE_END = 1 << 3
EVENT_WIN = 1 << 4
EVENT_LOSE = 1 << 5

GAME_RUNNING = 0
GAME_WON = 1
GAME_LOST = -1
MAX_HEALTH = 3
GOAL_COINS = 3
PLAYER_SPEED = 2.4
ENEMY_SPEED = 1.25
COLLISION_RADIUS = 24.0
COLLISION_COOLDOWN = 36

POLICY_RANDOM, POLICY_COIN, POLICY_COLLISION, POLICY_SCRIPTED = range(4)
POLICY_NAMES = ("random", "coin", "collision", "scripted")


@dataclass
class ArenaState:
    player: np.ndarray
    player_velocity: np.ndarray
    coin_pad: int
    enemies: np.ndarray
    enemy_velocity: np.ndarray
    score: int = 0
    collision_cooldown: int = 0
    collision_flash: int = 0
    step: int = 0
    health: int = MAX_HEALTH
    game_status: int = GAME_RUNNING


def _clamp_position(position: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(position, lo, hi).astype(np.float32)


def _unit(vector: np.ndarray, fallback: tuple[float, float] = (1.0, 0.0)) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return (vector / norm).astype(np.float32)


class ToyArenaV2:
    """Deterministic arena with an explicit frame/action transition contract."""

    def __init__(self, seed: int = 0, size: int = 128):
        if size < 64 or size % 8 != 0:
            raise ValueError("size must be at least 64 and divisible by 8")
        self.size = size
        self.margin = max(14, size // 9)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        scale = size / 128.0
        self.coin_pads = np.asarray(
            [
                (24, 24),
                (64, 20),
                (104, 24),
                (108, 64),
                (104, 104),
                (64, 108),
                (24, 104),
                (20, 64),
            ],
            dtype=np.float32,
        ) * scale
        self.portal_position = np.asarray((64.0, 64.0), dtype=np.float32) * scale
        self._background = self._make_background()
        self.state = self._new_state()

    def _new_state(self) -> ArenaState:
        lo = float(self.margin + 8)
        hi = float(self.size - self.margin - 8)
        player = self._rng.uniform(lo, hi, size=2).astype(np.float32)
        enemies = self._rng.uniform(lo, hi, size=(3, 2)).astype(np.float32)
        directions = np.asarray(
            ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)),
            dtype=np.float32,
        )
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        enemy_velocity = directions[self._rng.integers(0, len(directions), size=3)] * ENEMY_SPEED
        return ArenaState(
            player=player,
            player_velocity=np.zeros(2, dtype=np.float32),
            coin_pad=int(self._rng.integers(0, len(self.coin_pads))),
            enemies=enemies,
            enemy_velocity=enemy_velocity,
        )

    @property
    def coin_position(self) -> np.ndarray:
        return self.coin_pads[self.state.coin_pad]

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.state = self._new_state()
        return self.render()

    def _move_player(self, action: int) -> int:
        direction = np.zeros(2, dtype=np.float32)
        if action == MOVE_UP:
            direction[1] = -1.0
        elif action == MOVE_DOWN:
            direction[1] = 1.0
        elif action == MOVE_LEFT:
            direction[0] = -1.0
        elif action == MOVE_RIGHT:
            direction[0] = 1.0

        event = 0
        velocity = np.zeros(2, dtype=np.float32)
        if action == DASH:
            dash_direction = _unit(self.coin_position - self.state.player)
            velocity = dash_direction * 5.6
            event |= EVENT_DASH
        elif action != NOOP:
            velocity = direction * PLAYER_SPEED

        next_position = self.state.player + velocity
        lo = float(self.margin)
        hi = float(self.size - self.margin)
        for axis in range(2):
            if next_position[axis] < lo or next_position[axis] > hi:
                velocity[axis] *= -0.35
        self.state.player = _clamp_position(next_position, lo, hi)
        self.state.player_velocity = velocity.astype(np.float32)
        return event

    def _move_enemies(self) -> None:
        lo = float(self.margin)
        hi = float(self.size - self.margin)
        for index in range(len(self.state.enemies)):
            velocity = self.state.enemy_velocity[index]
            next_position = self.state.enemies[index] + velocity
            for axis in range(2):
                if next_position[axis] < lo or next_position[axis] > hi:
                    velocity[axis] *= -1.0
                    next_position[axis] = np.clip(next_position[axis], lo, hi)
            self.state.enemies[index] = next_position
            self.state.enemy_velocity[index] = velocity

    def _resolve_events(self) -> int:
        event = 0
        deltas = self.state.player[None] - self.state.enemies
        distances = np.linalg.norm(deltas, axis=1)
        away = deltas / np.clip(distances[:, None], 1e-5, None)
        contact_strength = np.clip((24.0 - distances) / 24.0, 0.0, 1.0)
        contact_force = (away * contact_strength[:, None]).sum(axis=0)
        self.state.player_velocity += contact_force * 1.2
        self.state.player = _clamp_position(
            self.state.player + contact_force * 1.5,
            float(self.margin),
            float(self.size - self.margin),
        )
        if float(np.linalg.norm(self.state.player - self.coin_position)) < 12.0:
            self.state.coin_pad = (self.state.coin_pad + 3) % len(self.coin_pads)
            self.state.score += 1
            event |= EVENT_COIN

        if self.state.collision_cooldown == 0:
            distances = np.linalg.norm(self.state.enemies - self.state.player[None], axis=1)
            enemy_index = int(np.argmin(distances))
            to_enemy = self.state.enemies[enemy_index] - self.state.player
            relative_velocity = self.state.player_velocity - self.state.enemy_velocity[enemy_index]
            closing = float(np.dot(relative_velocity, to_enemy)) > 0.0
            if float(distances[enemy_index]) < COLLISION_RADIUS and (
                closing or float(distances[enemy_index]) < 15.0
            ):
                self.state.health = max(0, self.state.health - 1)
                self.state.collision_cooldown = COLLISION_COOLDOWN
                self.state.collision_flash = 8
                event |= EVENT_COLLISION
        if self.state.health <= 0:
            self.state.game_status = GAME_LOST
            event |= EVENT_LOSE
        elif self.state.score >= GOAL_COINS and float(
            np.linalg.norm(self.state.player - self.portal_position)
        ) < 13.0:
            self.state.game_status = GAME_WON
            event |= EVENT_WIN
        return event

    def step(self, action: int) -> tuple[np.ndarray, int]:
        if not 0 <= int(action) < ACTION_COUNT:
            raise ValueError(f"invalid action {action}")
        if self.state.game_status != GAME_RUNNING:
            return self.render(), 0
        if self.state.collision_cooldown > 0:
            self.state.collision_cooldown -= 1
        if self.state.collision_flash > 0:
            self.state.collision_flash -= 1
        event = self._move_player(int(action))
        self._move_enemies()
        event |= self._resolve_events()
        self.state.step += 1
        return self.render(), event

    def snapshot(self) -> dict[str, np.ndarray | int]:
        return {
            "player_pos": self.state.player.copy(),
            "player_vel": self.state.player_velocity.copy(),
            "coin_pos": self.coin_position.copy(),
            "coin_pad": self.state.coin_pad,
            "enemy_pos": self.state.enemies.copy(),
            "enemy_vel": self.state.enemy_velocity.copy(),
            "score": self.state.score,
            "collision_cooldown": self.state.collision_cooldown,
            "collision_flash": self.state.collision_flash,
            "health": self.state.health,
            "goal_coins": GOAL_COINS,
            "portal_unlocked": int(self.state.score >= GOAL_COINS),
            "game_status": self.state.game_status,
        }

    def _make_background(self) -> np.ndarray:
        frame = np.full((self.size, self.size, 3), (16, 22, 31), dtype=np.uint8)
        tile = max(8, self.size // 12)
        for y in range(0, self.size, tile):
            for x in range(0, self.size, tile):
                if (x // tile + y // tile) % 2 == 0:
                    frame[y : y + tile, x : x + tile] = (21, 29, 40)
        margin = max(6, self.size // 18)
        frame[:margin] = (67, 77, 96)
        frame[-margin:] = (67, 77, 96)
        frame[:, :margin] = (67, 77, 96)
        frame[:, -margin:] = (67, 77, 96)
        return frame

    @staticmethod
    def _disk(frame: np.ndarray, center: np.ndarray, radius: float, color: tuple[int, int, int]) -> None:
        height, width = frame.shape[:2]
        cx, cy = float(center[0]), float(center[1])
        x0, x1 = max(0, int(cx - radius - 1)), min(width, int(cx + radius + 2))
        y0, y1 = max(0, int(cy - radius - 1)), min(height, int(cy + radius + 2))
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        patch = frame[y0:y1, x0:x1]
        patch[mask] = np.asarray(color, dtype=np.uint8)

    @staticmethod
    def _ring(
        frame: np.ndarray,
        center: np.ndarray,
        inner: float,
        outer: float,
        color: tuple[int, int, int],
    ) -> None:
        height, width = frame.shape[:2]
        cx, cy = float(center[0]), float(center[1])
        x0, x1 = max(0, int(cx - outer - 1)), min(width, int(cx + outer + 2))
        y0, y1 = max(0, int(cy - outer - 1)), min(height, int(cy + outer + 2))
        yy, xx = np.ogrid[y0:y1, x0:x1]
        distance = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = (inner * inner <= distance) & (distance <= outer * outer)
        patch = frame[y0:y1, x0:x1]
        patch[mask] = np.asarray(color, dtype=np.uint8)

    @staticmethod
    def _rect(frame: np.ndarray, x: int, y: int, width: int, height: int, color: tuple[int, int, int]) -> None:
        frame[y : y + height, x : x + width] = np.asarray(color, dtype=np.uint8)

    @classmethod
    def _outline_rect(
        cls, frame: np.ndarray, x: int, y: int, width: int, height: int, color: tuple[int, int, int], thickness: int = 2
    ) -> None:
        cls._rect(frame, x, y, width, thickness, color)
        cls._rect(frame, x, y + height - thickness, width, thickness, color)
        cls._rect(frame, x, y, thickness, height, color)
        cls._rect(frame, x + width - thickness, y, thickness, height, color)

    def _text(self, frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], scale: int = 3) -> None:
        glyphs = {
            "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
            "I": ("111", "010", "010", "010", "010", "010", "111"),
            "N": ("1001", "1101", "1101", "1011", "1011", "1001", "1001"),
            "L": ("100", "100", "100", "100", "100", "100", "111"),
            "O": ("0110", "1001", "1001", "1001", "1001", "1001", "0110"),
            "S": ("0111", "1000", "1000", "0110", "0001", "0001", "1110"),
            "E": ("111", "100", "100", "110", "100", "100", "111"),
        }
        cursor = x
        for character in text:
            glyph = glyphs[character]
            for row, bits in enumerate(glyph):
                for column, bit in enumerate(bits):
                    if bit == "1":
                        self._rect(frame, cursor + column * scale, y + row * scale, scale, scale, color)
            cursor += (len(glyph[0]) + 1) * scale

    def render(self) -> np.ndarray:
        frame = self._background.copy()
        for pad in self.coin_pads:
            self._ring(frame, pad, 8.5, 9.5, (49, 67, 78))

        if self.state.score >= GOAL_COINS:
            self._ring(frame, self.portal_position, 10.0, 12.0, (88, 248, 184))
            self._ring(frame, self.portal_position, 4.0, 6.0, (93, 185, 255))

        pulse = 1.0
        self._ring(frame, self.coin_position, 5.0 * pulse, 8.0 * pulse, (255, 224, 88))
        self._disk(frame, self.coin_position, 4.0 * pulse, (88, 248, 184))

        for enemy in self.state.enemies:
            self._disk(frame, enemy, 7.0, (247, 79, 91))
            self._disk(frame, enemy + np.asarray((-2.0, -2.0), dtype=np.float32), 2.0, (255, 188, 188))

        velocity = self.state.player_velocity
        for trail in (0.75, 0.50, 0.25):
            self._disk(frame, self.state.player - velocity * trail, 5.0, (32, 77, 139))
        if self.state.collision_flash > 0:
            self._ring(frame, self.state.player, 10.0, 13.0, (255, 246, 155))
        self._disk(frame, self.state.player, 8.0, (69, 142, 250))
        self._disk(frame, self.state.player + np.asarray((-3.0, -4.0), dtype=np.float32), 3.0, (224, 241, 255))
        self._disk(frame, self.state.player + np.asarray((4.0, -1.0), dtype=np.float32), 2.0, (18, 33, 66))
        for index in range(MAX_HEALTH):
            color = (247, 79, 91) if index < self.state.health else (62, 48, 59)
            self._disk(frame, np.asarray((8 + index * 9, 7), dtype=np.float32), 3.0, color)
        for index in range(GOAL_COINS):
            color = (88, 248, 184) if index < self.state.score else (48, 63, 73)
            self._rect(frame, 101 + index * 7, 4, 5, 6, color)
        if self.state.game_status != GAME_RUNNING:
            overlay = np.full_like(frame, (8, 12, 18))
            frame = ((frame.astype(np.uint16) * 2 + overlay.astype(np.uint16)) // 3).astype(np.uint8)
            if self.state.game_status == GAME_WON:
                self._outline_rect(frame, 20, 48, 88, 32, (19, 57, 52))
                self._text(frame, "WIN", 31, 53, (112, 255, 196), scale=3)
            else:
                self._outline_rect(frame, 14, 48, 100, 32, (67, 26, 35))
                self._text(frame, "LOSE", 22, 53, (255, 126, 136), scale=3)
        return frame


def choose_policy_action(env: ToyArenaV2, rng: np.random.Generator, policy: int, step: int) -> int:
    if policy == POLICY_RANDOM:
        return int(rng.integers(0, ACTION_COUNT))

    if policy == POLICY_SCRIPTED:
        pattern = (MOVE_RIGHT, MOVE_RIGHT, MOVE_DOWN, DASH, MOVE_LEFT, MOVE_UP, NOOP, MOVE_UP)
        return pattern[(step // 5) % len(pattern)]

    if policy == POLICY_COLLISION:
        target = env.state.enemies[int(np.argmin(np.linalg.norm(env.state.enemies - env.state.player, axis=1)))]
    elif policy == POLICY_COIN:
        target = env.portal_position if env.state.score >= GOAL_COINS else env.coin_position
    else:
        raise ValueError(f"unknown policy {policy}")

    if rng.random() < 0.15:
        return int(rng.integers(0, ACTION_COUNT))
    direction = target - env.state.player
    if float(np.linalg.norm(direction)) > 18.0 and rng.random() < 0.12:
        return DASH
    if abs(float(direction[0])) > abs(float(direction[1])):
        return MOVE_RIGHT if direction[0] > 0 else MOVE_LEFT
    return MOVE_DOWN if direction[1] > 0 else MOVE_UP
