"""Command-line entry point for preregistered Phase 2 analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase2_analysis import analyze_phase2
from src.utils import PROJECT_ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "experiments/PHASE-2/config.yaml")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    root = args.root or PROJECT_ROOT / config["outputs"]["formal_root"]
    print(analyze_phase2(root, config))


if __name__ == "__main__":
    main()
