from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the Toy Arena V2 codec from fixed QA metrics.")
    parser.add_argument("--cnn-metrics", required=True)
    parser.add_argument("--cnn-checkpoint", required=True)
    parser.add_argument("--dino-metrics", required=True)
    parser.add_argument("--dino-checkpoint", required=True)
    parser.add_argument("--out", default="runs/v2_codec_selection.json")
    args = parser.parse_args()

    candidates = {
        "cnn": {
            "metrics_path": args.cnn_metrics,
            "checkpoint": args.cnn_checkpoint,
            "metrics": json.loads(Path(args.cnn_metrics).read_text(encoding="utf-8")),
        },
        "dinov2": {
            "metrics_path": args.dino_metrics,
            "checkpoint": args.dino_checkpoint,
            "metrics": json.loads(Path(args.dino_metrics).read_text(encoding="utf-8")),
        },
    }
    passing = {name: value for name, value in candidates.items() if value["metrics"].get("gate_pass", False)}
    if not passing:
        raise RuntimeError("neither codec passed the reconstruction gate")
    selected_name, selected = max(
        passing.items(), key=lambda item: float(item[1]["metrics"]["semantic_composite"])
    )
    result = {
        "selected": selected_name,
        "checkpoint": selected["checkpoint"],
        "selection_rule": "highest semantic_composite among gate-passing codecs",
        "candidates": candidates,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

