"""Run all frozen Phase 1 analyses and publication figures from master results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1_analysis import (
    adjusted_family_rho,
    bootstrap_regression_increments,
    capacity_matching,
    decide_g4a,
    fit_level_table,
    ranking_reversal,
    regression_models,
    regression_sensitivities,
    robustness_envelope,
    variance_decomposition,
)
from src.phase1_plotting import make_all_figures
from src.utils import load_config, save_json


if __name__ == "__main__":
    config = load_config(PROJECT_ROOT / "experiments" / "PHASE-1" / "config.yaml")
    root = PROJECT_ROOT / "results" / "phase1"
    summaries = root / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(root / "master_results.csv")
    wrong_line = pd.read_csv(summaries / "wrong_line_guard.csv")
    fits = fit_level_table(master)
    resamples = int(config["statistics"]["bootstrap_resamples"])
    analysis_seed = int(config["experiment"]["master_seed"])

    variance = variance_decomposition(fits, resamples, analysis_seed + 1)
    regression, folds, _ = regression_models(fits)
    bootstrap = bootstrap_regression_increments(fits, resamples, analysis_seed + 2)
    sensitivities = regression_sensitivities(fits)
    matching = capacity_matching(
        fits,
        float(config["statistics"]["matching_caliper_sd"]),
        resamples,
        analysis_seed + 3,
    )
    envelope = robustness_envelope(fits)
    reversal, pairwise = ranking_reversal(fits)
    adjusted = adjusted_family_rho(fits)
    thresholds = {
        key: float(config["statistics"][key])
        for key in [
            "practical_family_incremental_r2",
            "practical_cv_rmse_improvement",
            "practical_accuracy_difference",
        ]
    }
    decision = decide_g4a(
        regression, folds, bootstrap, sensitivities, matching, wrong_line, thresholds
    )

    outputs = {
        "fit_level_results.csv": fits,
        "variance_decomposition.csv": variance,
        "regression_models.csv": regression,
        "regression_cv_folds.csv": folds,
        "regression_bootstrap.csv": bootstrap,
        "regression_sensitivities.csv": sensitivities,
        "capacity_matching.csv": matching,
        "robustness_envelope.csv": envelope,
        "ranking_reversal.csv": reversal,
        "pairwise_family_ordering.csv": pairwise,
        "adjusted_family_rho.csv": adjusted,
    }
    for filename, table in outputs.items():
        table.to_csv(summaries / filename, index=False)
    save_json(decision, summaries / "g4a_decision.json")
    make_all_figures(
        variance, adjusted, matching, envelope, reversal, wrong_line, root / "figures"
    )
    print(json.dumps(decision, indent=2))

