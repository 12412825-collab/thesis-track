"""Command-line entry point for the frozen Phase 2 experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase2_experiment import run_phase2
from src.utils import PROJECT_ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "experiments/PHASE-2/config.yaml")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-fits", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_root is None:
        output = PROJECT_ROOT / (config["outputs"]["smoke_root"] if args.smoke else config["outputs"]["formal_root"])
    else:
        output = args.output_root
    summary = run_phase2(config, output, resume=not args.no_resume, smoke=args.smoke, max_fits=args.max_fits)
    print(summary)


if __name__ == "__main__":
    main()
