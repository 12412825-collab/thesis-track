"""Focused tests for the preregistered EXP-002V implementation."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_generation import STABLE_FEATURES
from src.verification_data import (
    VERIFICATION_ALL_FEATURES,
    build_regime_bundles,
    generate_verification_dataset,
)
from src.verification_diagnostics import counterfactual_d3, loco_metrics
from src.verification_experiment import classifier_metrics
from src.verification_statistics import build_ranking_outputs


def _config() -> dict:
    with open("experiments/EXP-002V/config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = copy.deepcopy(config)
    config["data"].update(n_train=500, n_val=400, n_test=450)
    config["data"]["rho_train_values"] = [0.0, 0.9]
    config["data"]["rho_ood_values"] = [-0.9]
    config["replicates"].update(default=1, anchor=1)
    return config


def test_noise_feature_is_reproducible_and_empirically_independent() -> None:
    kwargs = dict(
        n_samples=20_000,
        rho=0.8,
        base_seed=31,
        noise_seed=32,
        stable_coefficients=[1.2, -1.0, 0.8],
        label_noise=0.5,
        split="test",
    )
    first = generate_verification_dataset(**kwargs)
    second = generate_verification_dataset(**kwargs)
    pd.testing.assert_frame_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    assert abs(np.corrcoef(first.X["z"], first.y)[0, 1]) < 0.03
    for feature in [*STABLE_FEATURES, "x_spurious"]:
        assert abs(np.corrcoef(first.X["z"], first.X[feature])[0, 1]) < 0.03


def test_rho_intervention_reuses_all_latents_except_shortcut_transform() -> None:
    bundles = build_regime_bundles(_config())
    neutral, strong = bundles
    for split in ["train", "validation", "iid_test"]:
        neutral_data = getattr(neutral, split)
        strong_data = getattr(strong, split)
        pd.testing.assert_frame_equal(
            neutral_data.X[[*STABLE_FEATURES, "z"]],
            strong_data.X[[*STABLE_FEATURES, "z"]],
        )
        np.testing.assert_array_equal(neutral_data.y, strong_data.y)
        assert not np.array_equal(
            neutral_data.X["x_spurious"].to_numpy(),
            strong_data.X["x_spurious"].to_numpy(),
        )


def test_d3_excess_detects_shortcut_above_noise_floor() -> None:
    train = generate_verification_dataset(
        n_samples=5000,
        rho=0.9,
        base_seed=41,
        noise_seed=42,
        stable_coefficients=[1.2, -1.0, 0.8],
        label_noise=0.5,
        split="train",
    )
    validation = generate_verification_dataset(
        n_samples=3000,
        rho=0.9,
        base_seed=43,
        noise_seed=44,
        stable_coefficients=[1.2, -1.0, 0.8],
        label_noise=0.5,
        split="validation",
    )
    model = Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
    ).fit(train.X[VERIFICATION_ALL_FEATURES], train.y)
    result = counterfactual_d3(
        model,
        train.X[VERIFICATION_ALL_FEATURES],
        validation.X[VERIFICATION_ALL_FEATURES],
    )
    assert result["d3_flip_rate_spurious"] > result["d3_flip_rate_noise"]
    assert result["d3_excess"] > 0.05
    assert result["d3_probability_sensitivity_excess"] > 0.0


def test_loco_preserves_negative_values_and_uses_error_normalization() -> None:
    result = loco_metrics(accuracy_all=0.70, accuracy_stable=0.80)
    assert np.isclose(result["raw_loco"], -0.10)
    assert np.isclose(result["loco_normalized"], -0.50)


def test_classifier_metrics_exports_every_preregistered_metric() -> None:
    train = generate_verification_dataset(
        n_samples=1000,
        rho=0.5,
        base_seed=51,
        noise_seed=52,
        stable_coefficients=[1.2, -1.0, 0.8],
        label_noise=0.5,
        split="train",
    )
    model = LogisticRegression(max_iter=1000).fit(
        train.X[VERIFICATION_ALL_FEATURES], train.y
    )
    metrics = classifier_metrics(model, train.X[VERIFICATION_ALL_FEATURES], train.y)
    assert set(metrics) == {
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "brier_score",
        "log_loss",
    }
    assert all(np.isfinite(value) for value in metrics.values())


def test_ranking_uses_independent_iid_test_not_validation() -> None:
    rows = []
    for model, validation, iid, ood in [
        ("logistic_regression", 0.99, 0.80, 0.70),
        ("rbf_svm", 0.70, 0.90, 0.60),
        ("random_forest", 0.60, 0.70, 0.50),
        ("gradient_boosting", 0.50, 0.60, 0.40),
        ("mlp", 0.40, 0.50, 0.30),
    ]:
        rows.append(
            {
                "rho_train": 0.0,
                "replicate": 0,
                "model": model,
                "feature_condition": "all_features",
                "validation_accuracy": validation,
                "iid_accuracy": iid,
                "average_ood_accuracy": ood,
            }
        )
    ranking, _, _ = build_ranking_outputs(pd.DataFrame(rows))
    iid = ranking.loc[ranking["metric"].eq("iid_test")]
    winner = iid.loc[iid["is_best"], "model"].item()
    assert winner == "rbf_svm"
