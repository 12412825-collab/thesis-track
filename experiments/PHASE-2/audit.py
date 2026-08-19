"""Command-line entry point for the formal Phase 2 adversarial audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase2_audit import run_audit
from src.utils import PROJECT_ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "experiments/PHASE-2/config.yaml")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "results/phase2")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = (PROJECT_ROOT / "Methodology_PreRegistration_Phase2_Dynamics_vs_Capacity.md", PROJECT_ROOT / "Methodology_PreRegistration_Phase2_Dynamics_vs_Capacity_AMENDMENT.md")
    print(run_audit(args.root, config, paths)["summary"])


if __name__ == "__main__":
    main()
