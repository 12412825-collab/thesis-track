"""Tests for EXP-002 controls, aggregation, and ranking analysis."""

import numpy as np
import pandas as pd

from src.data_generation import STABLE_FEATURES, generate_dataset
from src.evaluation import aggregate_model_summary
from src.sweep import build_ranking_reversal_summary, rho_test_grid


def test_rho_sweep_changes_only_spurious_feature_for_same_seed() -> None:
    weak = generate_dataset(2_000, rho=0.5, seed=321)
    strong = generate_dataset(2_000, rho=0.9, seed=321)
    pd.testing.assert_frame_equal(weak.X[STABLE_FEATURES], strong.X[STABLE_FEATURES])
    np.testing.assert_array_equal(weak.y, strong.y)
    assert not np.array_equal(weak.X["x_spurious"], strong.X["x_spurious"])


def test_relative_test_grid_has_exactly_one_iid_environment() -> None:
    common = [0.3, 0.0, -0.3, -0.6, -0.9]
    for rho_train in [0.5, 0.7, 0.9]:
        grid = rho_test_grid(rho_train, common)
        assert grid[0] == rho_train
        assert sum(np.isclose(grid, rho_train)) == 1
        assert len(grid) == 6


def test_ranking_reversal_summary_uses_declared_metrics() -> None:
    summary = pd.DataFrame(
        {
            "rho_train": [0.5, 0.5, 0.7, 0.7],
            "model": ["a", "b", "a", "b"],
            "feature_condition": ["all_features"] * 4,
            "validation_accuracy_mean": [0.9, 0.8, 0.8, 0.9],
            "average_ood_accuracy_mean": [0.7, 0.8, 0.7, 0.6],
            "worst_environment_accuracy_mean": [0.5, 0.6, 0.5, 0.4],
        }
    )
    rankings = build_ranking_reversal_summary(summary)
    first = rankings.loc[np.isclose(rankings["rho_train"], 0.5)].iloc[0]
    second = rankings.loc[np.isclose(rankings["rho_train"], 0.7)].iloc[0]
    assert first["IID_best_model"] == "a"
    assert first["OOD_best_model"] == "b"
    assert bool(first["ranking_reversal"])
    assert second["IID_best_model"] == "b"
    assert second["OOD_best_model"] == "a"
    assert bool(second["ranking_reversal"])


def test_seed_aggregation_computes_mean_and_standard_error() -> None:
    metric_names = [
        "validation_accuracy", "iid_accuracy", "average_ood_accuracy",
        "worst_environment_accuracy", "ood_gap", "severe_shift_accuracy",
        "degradation_slope", "spurious_reliance_score",
    ]
    rows = []
    for seed, value in [(0, 0.6), (1, 0.8)]:
        row = {
            "seed": seed,
            "model": "example",
            "feature_condition": "all_features",
        }
        row.update({metric: value for metric in metric_names})
        rows.append(row)
    summary = aggregate_model_summary(pd.DataFrame(rows)).iloc[0]
    assert np.isclose(summary["iid_accuracy_mean"], 0.7)
    assert np.isclose(summary["iid_accuracy_se"], 0.1)
    assert summary["n_seeds"] == 2
