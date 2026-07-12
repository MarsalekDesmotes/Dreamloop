from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_v2 import (
    SEMANTIC_STATE_DIM,
    decode_visible_collision_flash,
    decode_visible_game_state,
    load_toy_arena_v2,
)
from src.eval_v2 import decode_state_probe
from src.model_v2 import arena_state_probe_from_checkpoint
from src.training_v2 import checkpoint_sha256, load_trusted_checkpoint
from src.toy_arena_v2 import COLLISION_COOLDOWN


ENEMY_PERMUTATIONS = np.asarray(list(itertools.permutations(range(3))), dtype=np.int64)


def match_enemy_tracks(positions: np.ndarray, scores: np.ndarray, dones: np.ndarray) -> None:
    start = 0
    segments = (np.flatnonzero(np.asarray(dones, dtype=np.bool_)) + 1).tolist()
    if not segments or segments[-1] != len(positions):
        segments.append(len(positions))
    for end in tqdm(segments, desc="match enemy tracks"):
        if end <= start:
            continue
        first_order = np.lexsort((positions[start, :, 1], positions[start, :, 0]))
        positions[start] = positions[start, first_order]
        scores[start, 2:] = scores[start, 2:][first_order]
        previous = positions[start].copy()
        for index in range(start + 1, end):
            candidates = positions[index][ENEMY_PERMUTATIONS]
            costs = np.linalg.norm(candidates - previous[None], axis=2).sum(axis=1)
            selected_index = int(costs.argmin())
            selected = candidates[selected_index]
            positions[index] = selected
            scores[index, 2:] = scores[index, 2:][ENEMY_PERMUTATIONS[selected_index]]
            previous = selected
        start = end


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RGB-derived semantic state cache for Toy Arena V2.")
    parser.add_argument("--data", default="data/toy_arena_v2_60k")
    parser.add_argument("--probes", required=True)
    parser.add_argument("--out", default="data/toy_arena_v2_60k/semantic_cache")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    arrays = load_toy_arena_v2(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    expected_files = [
        out / name
        for name in ("player_pos.npy", "coin_pos.npy", "enemy_pos.npy", "scores.npy", "flash.npy", "game_state.npy")
    ]
    if any(path.exists() for path in expected_files) and not args.overwrite:
        raise FileExistsError("semantic cache exists; pass --overwrite to rebuild")

    checkpoint = load_trusted_checkpoint(args.probes, map_location=device)
    probe = arena_state_probe_from_checkpoint(checkpoint).to(device)
    probe.load_state_dict(checkpoint["state_probe"])
    probe.eval()
    frame_count = len(arrays.frames)
    player_pos = open_memmap(out / "player_pos.npy", mode="w+", dtype=np.float32, shape=(frame_count, 2))
    coin_pos = open_memmap(out / "coin_pos.npy", mode="w+", dtype=np.float32, shape=(frame_count, 2))
    enemy_pos = open_memmap(out / "enemy_pos.npy", mode="w+", dtype=np.float32, shape=(frame_count, 3, 2))
    scores = open_memmap(out / "scores.npy", mode="w+", dtype=np.float32, shape=(frame_count, 5))
    flash = open_memmap(out / "flash.npy", mode="w+", dtype=np.float32, shape=(frame_count, 1))
    game_state = open_memmap(out / "game_state.npy", mode="w+", dtype=np.float32, shape=(frame_count, 4))

    with torch.inference_mode():
        for start in tqdm(range(0, frame_count, args.batch_size), desc="decode semantic cache"):
            end = min(start + args.batch_size, frame_count)
            values = np.asarray(arrays.frames[start:end], dtype=np.float32) / 255.0
            frames = torch.from_numpy(np.transpose(values, (0, 3, 1, 2)).copy()).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                decoded = decode_state_probe(probe(frames))
            player_pos[start:end] = decoded["player_pos"].float().cpu().numpy()
            coin_pos[start:end] = decoded["coin_pos"].float().cpu().numpy()
            enemy_pos[start:end] = decoded["enemy_pos"].float().cpu().numpy()
            scores[start:end, 0] = decoded["player_score"].float().cpu().numpy()
            scores[start:end, 1] = decoded["coin_score"].float().cpu().numpy()
            scores[start:end, 2:] = decoded["enemy_score"].float().cpu().numpy()
            flash[start:end, 0] = decode_visible_collision_flash(np.asarray(arrays.frames[start:end]))
            game_state[start:end] = decode_visible_game_state(np.asarray(arrays.frames[start:end]))

    match_enemy_tracks(enemy_pos, scores, arrays.dones)
    player_pos.flush()
    coin_pos.flush()
    enemy_pos.flush()
    scores.flush()
    flash.flush()
    game_state.flush()
    player_error = np.linalg.norm(np.asarray(player_pos) - np.asarray(arrays.player_pos), axis=1)
    coin_error = np.linalg.norm(np.asarray(coin_pos) - np.asarray(arrays.coin_pos), axis=1)
    enemy_candidates = np.asarray(arrays.enemy_pos)[:, ENEMY_PERMUTATIONS]
    enemy_error = np.linalg.norm(np.asarray(enemy_pos)[:, None] - enemy_candidates, axis=3).mean(axis=2).min(axis=1)
    metrics = {
        "player_error": float(player_error.mean()),
        "coin_error": float(coin_error.mean()),
        "enemy_error": float(enemy_error.mean()),
        "player_p95": float(np.percentile(player_error, 95)),
        "coin_p95": float(np.percentile(coin_error, 95)),
        "enemy_p95": float(np.percentile(enemy_error, 95)),
        "flash_accuracy": float(
            (
                (np.asarray(flash[:, 0]) > 0.5)
                == (np.asarray(arrays.collision_cooldown) > COLLISION_COOLDOWN - 8)
            ).mean()
        ),
        "health_accuracy": float(
            (np.rint(np.asarray(game_state[:, 0]) * 3.0) == np.asarray(arrays.health)).mean()
        ) if arrays.health is not None else None,
        "progress_accuracy": float(
            (np.rint(np.asarray(game_state[:, 1]) * 3.0) == np.minimum(np.asarray(arrays.score), 3)).mean()
        ),
        "terminal_accuracy": float(
            (
                np.where(np.asarray(game_state[:, 2]) > 0.5, 1, np.where(np.asarray(game_state[:, 3]) > 0.5, -1, 0))
                == np.asarray(arrays.game_status)
            ).mean()
        ) if arrays.game_status is not None else None,
    }
    metadata = {
        "version": 2,
        "state_dim": SEMANTIC_STATE_DIM,
        "data": args.data,
        "dataset_manifest": arrays.metadata["manifest_hash"],
        "probes": args.probes,
        "probe_sha256": checkpoint_sha256(args.probes),
        "frame_count": frame_count,
        "source": "frozen RGB state probe; privileged state used only for reported cache metrics",
        "metrics": metrics,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
