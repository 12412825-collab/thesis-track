"""Command-line entry point for Phase 2 results and audit reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase2_reporting import write_reports
from src.utils import PROJECT_ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "experiments/PHASE-2/config.yaml")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "results/phase2")
    args = parser.parse_args()
    config = load_config(args.config)
    print(write_reports(args.root, config, PROJECT_ROOT))


if __name__ == "__main__":
    main()
