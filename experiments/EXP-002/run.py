"""CLI entry point for EXP-002: training shortcut strength sweep."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sweep import run_training_strength_sweep
from src.utils import configure_logging, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "EXP-002" / "config.yaml",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the first configured seed and isolate outputs under smoke/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    arguments = parse_args()
    configuration = copy.deepcopy(load_config(arguments.config))
    output_root = PROJECT_ROOT / "results" / "EXP-002"
    if arguments.smoke:
        configuration["seeds"] = configuration["seeds"][:1]
        output_root = output_root / "smoke"
    started = time.perf_counter()
    output = run_training_strength_sweep(configuration, output_root)
    duration = time.perf_counter() - started
    metadata = {
        "experiment_id": "EXP-002",
        "mode": "smoke" if arguments.smoke else "complete",
        "runtime_seconds": round(duration, 2),
        "seeds": configuration["seeds"],
        "rho_train_values": configuration["data"]["rho_train_values"],
    }
    (output["summaries"] / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"EXP-002 runtime_seconds={duration:.2f}")
