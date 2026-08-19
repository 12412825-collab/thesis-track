"""Command-line entry point for the frozen learner-only Bridge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.bridge_experiment import run_bridge, run_bridge_smoke
from src.utils import PROJECT_ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-config", type=Path, default=PROJECT_ROOT / "experiments/PHASE-2/config.yaml")
    parser.add_argument("--phase1-config", type=Path, default=PROJECT_ROOT / "experiments/PHASE-1/config.yaml")
    parser.add_argument("--phase2-root", type=Path, default=PROJECT_ROOT / "results/phase2")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results/bridge_p1mlp_l")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    phase2_config = load_config(args.phase2_config)
    if args.smoke:
        summary = run_bridge_smoke(phase2_config, args.phase1_config, PROJECT_ROOT / "results/bridge_p1mlp_l_smoke")
    else:
        summary = run_bridge(phase2_config, args.phase1_config, args.phase2_root, args.output_root, resume=not args.no_resume)
    print(summary)


if __name__ == "__main__":
    main()
