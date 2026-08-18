"""Run all frozen Phase 1 analyses and publication figures from master results."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
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


def common_ood_sensitivity(master: pd.DataFrame, fits: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    """Post hoc Reviewer-2 check using only OOD rhos shared by every regime."""
    successful = master.loc[master["status"].eq("success")]
    sets = [
        set(group.loc[group["environment_type"].eq("ood"), "rho_environment"].unique())
        for _, group in successful.groupby("rho_train")
    ]
    common = sorted(set.intersection(*sets))
    ood = successful.loc[
        successful["environment_type"].eq("ood")
        & successful["rho_environment"].isin(common)
    ].groupby("fit_id")["accuracy"].mean()
    iid = successful.loc[successful["environment_type"].eq("iid_test")].set_index("fit_id")["accuracy"]
    common_fits = fits.copy().set_index("fit_id")
    common_fits["iid_accuracy"] = iid
    common_fits["average_ood_accuracy"] = ood
    common_fits["ood_gap"] = iid - ood
    common_fits = common_fits.reset_index()
    metrics, _, _ = regression_models(common_fits)
    indexed = metrics.set_index("model")
    variance = variance_decomposition(common_fits, 500, 20260829).set_index("component")
    output = pd.DataFrame(
        [
            {
                "scope": "common_ood_grid_post_hoc",
                "common_rho_values": json.dumps(common),
                "family_incremental_r2": indexed.loc["M2", "r2"] - indexed.loc["M1", "r2"],
                "family_incremental_cv_r2": indexed.loc["M2", "cv_r2"] - indexed.loc["M1", "cv_r2"],
                "family_cv_rmse_reduction": 1.0 - indexed.loc["M2", "cv_rmse"] / indexed.loc["M1", "cv_rmse"],
                "interaction_incremental_cv_r2": indexed.loc["M3", "cv_r2"] - indexed.loc["M2", "cv_r2"],
                "variance_share_family": variance.loc["family", "variance_share"],
                "variance_share_family_x_rho": variance.loc["family_x_rho", "variance_share"],
                "variance_share_rho": variance.loc["rho_train", "variance_share"],
            }
        ]
    )
    return output, common


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    common_sensitivity, common_rhos = common_ood_sensitivity(master, fits)
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
        "common_ood_grid_sensitivity_post_hoc.csv": common_sensitivity,
    }
    for filename, table in outputs.items():
        table.to_csv(summaries / filename, index=False)
    save_json(decision, summaries / "g4a_decision.json")
    make_all_figures(
        variance, adjusted, matching, envelope, reversal, wrong_line, root / "figures"
    )
    analysis_metadata = {
        "analysis_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "master_results_sha256": sha256(root / "master_results.csv"),
        "config_sha256": sha256(PROJECT_ROOT / "experiments" / "PHASE-1" / "config.yaml"),
        "analysis_script_sha256": sha256(Path(__file__)),
        "bootstrap_resamples": resamples,
        "post_hoc_common_ood_rhos": common_rhos,
        "g4a_decision": decision,
    }
    save_json(analysis_metadata, summaries / "analysis_metadata.json")
    manifest_rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or ".partial." in path.name:
            continue
        manifest_rows.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    save_json(
        {
            "experiment_id": config["experiment"]["id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": manifest_rows,
        },
        root / "manifest.json",
    )
    print(json.dumps(decision, indent=2))
