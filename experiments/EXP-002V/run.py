"""Run EXP-002V preregistered verification without touching prior experiments."""

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

from src.utils import configure_logging, load_config
from src.verification_experiment import run_verification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "EXP-002V" / "config.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    arguments = parse_args()
    config = copy.deepcopy(load_config(arguments.config))
    output_root = PROJECT_ROOT / "results" / "EXP-002V"
    if arguments.smoke:
        config["replicates"]["default"] = 1
        config["replicates"]["anchor"] = 1
        config["statistics"]["bootstrap_resamples"] = 500
        output_root = output_root / "smoke"
    started = time.perf_counter()
    outputs = run_verification(config, output_root, PROJECT_ROOT)
    duration = time.perf_counter() - started
    metadata = {
        "experiment_id": "EXP-002V",
        "mode": "smoke" if arguments.smoke else "complete",
        "runtime_seconds": round(duration, 2),
        "bootstrap_resamples": config["statistics"]["bootstrap_resamples"],
        "replicates_default": config["replicates"]["default"],
        "replicates_anchor": config["replicates"]["anchor"],
    }
    (outputs["summaries"] / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"EXP-002V runtime_seconds={duration:.2f}")
