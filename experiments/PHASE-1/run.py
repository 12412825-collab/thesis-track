"""Run Phase 1 design calibration, smoke validation, or the complete grid."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1_experiment import run_capacity_calibration, run_phase1_grid
from src.phase1_wrong_line import run_wrong_line_guard
from src.phase1_validation import write_smoke_report
from src.utils import configure_logging, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "PHASE-1" / "config.yaml",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--calibrate-capacity", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    config = copy.deepcopy(load_config(args.config))
    root = PROJECT_ROOT / "results" / "phase1"
    if args.calibrate_capacity:
        run_capacity_calibration(config, root / "design_calibration", PROJECT_ROOT)
    elif args.smoke:
        smoke_root = root / "smoke"
        run_phase1_grid(config, smoke_root, PROJECT_ROOT, smoke=True)
        wrong_line, failures = run_wrong_line_guard(
            config, smoke_root / "summaries" / "wrong_line_guard.csv", smoke=True
        )
        write_smoke_report(
            smoke_root,
            PROJECT_ROOT / "Phase1_Smoke_Test_Report.md",
            wrong_line,
            failures,
        )
        if failures:
            raise RuntimeError("Smoke wrong-line stop gate failed: " + "; ".join(failures))
    else:
        run_wrong_line_guard(config, root / "summaries" / "wrong_line_guard.csv", smoke=False)
        run_phase1_grid(config, root, PROJECT_ROOT, smoke=False)
