from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_v2 import load_toy_arena_v2, load_v2_semantic_states


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize RGB-derived semantic states for fast training startup.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--semantic-cache", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.semantic_cache)
    state_path = root / "states.npy"
    metadata_path = root / "states_metadata.json"
    if (state_path.exists() or metadata_path.exists()) and not args.overwrite:
        raise FileExistsError("semantic states cache exists; pass --overwrite to rebuild")
    state_path.unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)

    arrays = load_toy_arena_v2(args.data)
    states, _ = load_v2_semantic_states(root, arrays, mmap_mode="r")
    np.save(state_path, np.asarray(states, dtype=np.float32))
    metadata = {
        "version": 1,
        "dataset_manifest": arrays.metadata["manifest_hash"],
        "frame_count": len(states),
        "state_dim": int(states.shape[1]),
        "source": "RGB-derived semantic positions, visible UI, and causal trajectory estimates",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
