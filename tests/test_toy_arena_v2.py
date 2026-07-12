from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.generate_toy_arena_v2 import main as generate_main
from src.data_v2 import (
    StratifiedEventSampler,
    ToyArenaV2SequenceDataset,
    decode_visible_collision_flash,
    decode_visible_game_state,
    load_toy_arena_v2,
)
from src.model_v2 import NeuralSemanticStateDynamics, StructuredSemanticStateDynamics
from src.eval_v2 import decode_state_probe
from src.toy_arena_v2 import (
    EVENT_COIN,
    EVENT_COLLISION,
    EVENT_LOSE,
    EVENT_WIN,
    GAME_LOST,
    GAME_WON,
    GOAL_COINS,
    MOVE_RIGHT,
    ToyArenaV2,
)


def test_simulator_is_deterministic() -> None:
    first = ToyArenaV2(seed=42)
    second = ToyArenaV2(seed=42)
    actions = [MOVE_RIGHT, MOVE_RIGHT, 5, 0, 1, 3]
    assert np.array_equal(first.render(), second.render())
    for action in actions:
        first_frame, first_event = first.step(action)
        second_frame, second_event = second.step(action)
        assert first_event == second_event
        assert np.array_equal(first_frame, second_frame)


def test_action_transitions_current_frame_to_next_frame() -> None:
    env = ToyArenaV2(seed=7)
    before = env.snapshot()["player_pos"].copy()
    frame_before = env.render()
    frame_after, _ = env.step(MOVE_RIGHT)
    after = env.snapshot()["player_pos"]
    assert after[0] > before[0]
    assert not np.array_equal(frame_before, frame_after)


def test_player_motion_is_directly_observable_from_each_action() -> None:
    env = ToyArenaV2(seed=71)
    env.state.enemies[:] = 100.0
    env.step(MOVE_RIGHT)
    moved = env.snapshot()["player_pos"].copy()
    env.step(0)
    assert np.array_equal(env.snapshot()["player_pos"], moved)


def test_enemy_velocities_use_the_discrete_direction_codebook() -> None:
    env = ToyArenaV2(seed=72)
    velocity = env.snapshot()["enemy_vel"]
    assert np.allclose(np.linalg.norm(velocity, axis=1), 1.25)
    directions = velocity / np.linalg.norm(velocity, axis=1, keepdims=True)
    assert np.all(np.isclose(np.abs(directions), 0.0) | np.isclose(np.abs(directions), 1.0) | np.isclose(np.abs(directions), 2**-0.5))


def test_coin_respawn_is_visible_and_deterministic() -> None:
    env = ToyArenaV2(seed=3)
    env.state.player = env.coin_position.copy()
    old_pad = env.state.coin_pad
    _, event = env.step(0)
    assert event & EVENT_COIN
    assert env.state.coin_pad == (old_pad + 3) % len(env.coin_pads)


def test_three_enemy_contacts_produce_a_frozen_lose_state() -> None:
    env = ToyArenaV2(seed=11)
    event = 0
    for _ in range(3):
        env.state.enemies[0] = env.state.player.copy()
        env.state.collision_cooldown = 0
        _, event = env.step(0)
    assert event & EVENT_COLLISION
    assert event & EVENT_LOSE
    assert env.state.game_status == GAME_LOST
    frozen = env.snapshot()["player_pos"].copy()
    env.step(MOVE_RIGHT)
    assert np.array_equal(env.snapshot()["player_pos"], frozen)


def test_collecting_goal_and_reaching_portal_wins() -> None:
    env = ToyArenaV2(seed=12)
    env.state.score = GOAL_COINS
    env.state.player = env.portal_position.copy()
    env.state.enemies[:] = 100.0
    _, event = env.step(0)
    assert event & EVENT_WIN
    assert env.state.game_status == GAME_WON


def test_visible_game_state_is_decoded_from_running_and_terminal_frames() -> None:
    env = ToyArenaV2(seed=13)
    running = decode_visible_game_state(env.render())[0]
    assert np.allclose(running, (1.0, 0.0, 0.0, 0.0))

    env.state.health = 2
    env.state.score = 2
    partial = decode_visible_game_state(env.render())[0]
    assert np.allclose(partial, (2.0 / 3.0, 2.0 / 3.0, 0.0, 0.0))

    env.state.game_status = GAME_WON
    won = decode_visible_game_state(env.render())[0]
    assert np.allclose(won, (2.0 / 3.0, 2.0 / 3.0, 1.0, 0.0))

    env.state.game_status = GAME_LOST
    lost = decode_visible_game_state(env.render())[0]
    assert np.allclose(lost, (2.0 / 3.0, 2.0 / 3.0, 0.0, 1.0))

    env.state.collision_flash = 8
    assert decode_visible_collision_flash(env.render())[0] == 1.0


def test_structured_gameplay_dynamics_predicts_loss_and_freezes_terminal() -> None:
    model = StructuredSemanticStateDynamics(state_dim=23)
    state = torch.zeros(1, 23)
    state[:, 4:6] = torch.tensor([[-0.625, -0.625]])
    state[:, 6:12] = torch.tensor([[0.0, 0.0, 0.75, 0.75, -0.75, 0.75]])
    state[:, 19] = 1.0 / 3.0
    prediction, hidden = model.step(state, torch.tensor([0]))
    assert torch.allclose(prediction[:, 19], torch.zeros(1))
    assert torch.allclose(prediction[:, 22], torch.ones(1))
    frozen, _ = model.step(prediction, torch.tensor([MOVE_RIGHT]), hidden)
    assert torch.equal(frozen, prediction)


def test_structured_collision_flash_does_not_restart_cooldown() -> None:
    model = StructuredSemanticStateDynamics(state_dim=23)
    state = torch.zeros(1, 23)
    state[:, 4:6] = torch.tensor([[-0.625, -0.625]])
    state[:, 6:12] = torch.tensor([[0.0, 0.0, 0.75, 0.75, -0.75, 0.75]])
    state[:, 19] = 1.0
    flashes = []
    hidden = None
    for _ in range(10):
        state, hidden = model.step(state, torch.tensor([0]), hidden)
        flashes.append(float(state[0, 18].detach().item()))
    assert flashes[:8] == [1.0] * 8
    assert flashes[8:] == [0.0, 0.0]
    assert float(hidden[2][0].detach().item()) == 27.0


def test_structured_gameplay_dynamics_predicts_coin_progress_and_portal_win() -> None:
    model = StructuredSemanticStateDynamics(state_dim=23)
    state = torch.zeros(1, 23)
    state[:, 0:2] = torch.tensor([[-0.625, -0.625]])
    state[:, 4:6] = state[:, 0:2]
    state[:, 6:12] = torch.tensor([[0.75, 0.75, 0.75, -0.75, -0.75, 0.75]])
    state[:, 19] = 1.0
    state[:, 20] = 2.0 / 3.0
    prediction, hidden = model.step(state, torch.tensor([0]))
    assert torch.allclose(prediction[:, 20], torch.ones(1))
    assert torch.allclose(prediction[:, 21:23], torch.zeros(1, 2))

    prediction[:, 0:2] = 0.0
    won, _ = model.step(prediction, torch.tensor([0]), hidden)
    assert torch.allclose(won[:, 21], torch.ones(1))
    assert torch.allclose(won[:, 22], torch.zeros(1))


def test_state_probe_decoder_vectorizes_batches() -> None:
    decoded = decode_state_probe(torch.randn(7, 3, 64, 64))
    assert decoded["player_pos"].shape == (7, 2)
    assert decoded["coin_pos"].shape == (7, 2)
    assert decoded["enemy_pos"].shape == (7, 3, 2)


def test_neural_semantic_dynamics_streams_and_freezes_terminal_state() -> None:
    model = NeuralSemanticStateDynamics(state_dim=23)
    state = torch.zeros(2, 23)
    prediction, hidden = model.step(state, torch.tensor([0, MOVE_RIGHT]))
    assert prediction.shape == state.shape
    assert len(hidden) == 2
    terminal = prediction.clone()
    terminal[:, 21] = 1.0
    frozen, _ = model.step(terminal, torch.tensor([MOVE_RIGHT, MOVE_RIGHT]), hidden)
    assert torch.equal(frozen, terminal)


def test_dataset_splits_do_not_share_episodes(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "v2"
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_toy_arena_v2.py",
            "--out",
            str(out),
            "--episodes",
            "10",
            "--episode-length",
            "80",
        ],
    )
    generate_main()
    arrays = load_toy_arena_v2(out)
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["transition_contract"].startswith("frame[t]")
    assert arrays.health is not None
    assert arrays.portal_unlocked is not None
    assert arrays.game_status is not None
    assert arrays.health.dtype == np.uint8
    assert arrays.game_status.dtype == np.int8
    assert np.all(np.asarray(arrays.actions)[np.asarray(arrays.game_status) != 0] == 0)

    datasets = {
        split: ToyArenaV2SequenceDataset(arrays, split=split, context=8, horizon=4)
        for split in ("train", "val", "test")
    }
    episodes = {
        split: set(np.asarray(arrays.episode_ids[dataset.indices], dtype=np.int64).tolist())
        for split, dataset in datasets.items()
    }
    assert episodes["train"].isdisjoint(episodes["val"])
    assert episodes["train"].isdisjoint(episodes["test"])
    assert episodes["val"].isdisjoint(episodes["test"])
    for dataset in datasets.values():
        for anchor in dataset.indices:
            transition_start = int(anchor) - dataset.context + 1
            transition_end = int(anchor) + dataset.horizon
            assert not np.any(arrays.dones[transition_start:transition_end])
            start_episode = int(arrays.episode_ids[int(anchor) - dataset.context + 1])
            end_episode = int(arrays.episode_ids[int(anchor) + dataset.horizon])
            assert start_episode == end_episode


def test_sequence_can_be_restricted_to_one_episode(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "v2_filtered"
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_toy_arena_v2.py",
            "--out",
            str(out),
            "--episodes",
            "10",
            "--episode-length",
            "80",
        ],
    )
    generate_main()
    arrays = load_toy_arena_v2(out)
    episode_id = int(np.flatnonzero(np.asarray(arrays.episode_splits) == 0)[0])
    dataset = ToyArenaV2SequenceDataset(
        arrays, split="train", context=4, horizon=3, episode_ids=[episode_id]
    )
    assert len(dataset) > 0
    assert set(np.asarray(arrays.episode_ids)[dataset.indices].tolist()) == {episode_id}


def test_sampler_can_renormalize_missing_overfit_classes() -> None:
    class CollisionOnlyDataset:
        event_classes = np.full(5, 2, dtype=np.uint8)

        def __len__(self) -> int:
            return len(self.event_classes)

    sampler = StratifiedEventSampler(CollisionOnlyDataset(), num_samples=7, allow_missing=True)
    sampled = list(sampler)
    assert len(sampled) == 7
    assert set(sampled).issubset(set(range(5)))


def test_sampler_keeps_overlapping_coin_and_collision_windows() -> None:
    class MultiEventDataset:
        event_flags = np.asarray([0, EVENT_COIN, EVENT_COLLISION, EVENT_COIN | EVENT_COLLISION], dtype=np.uint8)
        event_classes = np.asarray([0, 1, 2, 2], dtype=np.uint8)

        def __len__(self) -> int:
            return len(self.event_flags)

    sampler = StratifiedEventSampler(MultiEventDataset(), num_samples=12)
    assert sampler.pools[1].tolist() == [1, 3]
    assert sampler.pools[2].tolist() == [2, 3]
