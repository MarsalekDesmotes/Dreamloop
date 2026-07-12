from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


FILES = {
    "active_rgb": Path("runs/sequence_model_c8_h8_player_wide_e8_gpu/best.pt"),
    "latent_full": Path("runs/latent_dynamics_c8_h8_e20_gpu/best.pt"),
    "latent_event_focus": Path("runs/latent_dynamics_event_focus_fg_e4_gpu/best.pt"),
    "autoencoder": Path("runs/autoencoder_l64_16x16_e20_gpu/best.pt"),
}

METRICS = {
    "active_rgb": {
        "mixed_closed_loop": {"foreground_mse": 0.096712, "player_mse": 0.048376},
        "event_closed_loop": {"foreground_mse": 0.124817, "player_mse": 0.099516},
    },
    "latent_full": {
        "mixed_closed_loop": {"foreground_mse": 0.10108182369731367, "player_mse": 0.017210113999681198},
        "event_closed_loop": {"foreground_mse": 0.16949263075366616, "player_mse": 0.026008750748587772},
    },
    "latent_event_focus": {
        "mixed_closed_loop": {"foreground_mse": 0.10497486777603626, "player_mse": 0.017793992687074933},
        "event_closed_loop": {"foreground_mse": 0.1580496276728809, "player_mse": 0.024758404462772887},
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [str(path) for path in FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing V1 artifacts: {missing}")
    manifest = {
        "version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "reason": "Toy Arena V1 is frozen before the deterministic V2 transition contract.",
        "active_demo": "active_rgb",
        "files": {
            name: {"path": str(path).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in FILES.items()
        },
        "metrics": METRICS,
    }
    out = Path("runs/v1_baseline_manifest.json")
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

