"""Preregistered shortcut-reliance diagnostics for EXP-002V."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_generation import STABLE_FEATURES


def fit_conditional_ols(X_train: pd.DataFrame, target: str) -> Pipeline:
    """Fit target ~ standardized stable features using training data only."""
    model = Pipeline([("scale", StandardScaler()), ("ols", LinearRegression())])
    model.fit(X_train[STABLE_FEATURES], X_train[target])
    return model


def _positive_probability(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError("EXP-002V requires predict_proba for sensitivity diagnostics")
    return np.asarray(model.predict_proba(X))[:, 1]


def counterfactual_d3(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
) -> dict[str, float]:
    """Compute stable-conditional D3 for x_spurious and calibrated noise z."""
    original_prediction = np.asarray(model.predict(X_validation))
    original_probability = _positive_probability(model, X_validation)
    outputs: dict[str, float] = {}
    for target, prefix in [("x_spurious", "spurious"), ("z", "noise")]:
        auxiliary = fit_conditional_ols(X_train, target)
        expected = auxiliary.predict(X_validation[STABLE_FEATURES])
        counterfactual = X_validation.copy()
        counterfactual[target] = expected
        changed_prediction = np.asarray(model.predict(counterfactual))
        changed_probability = _positive_probability(model, counterfactual)
        outputs[f"d3_flip_rate_{prefix}"] = float(
            np.mean(original_prediction != changed_prediction)
        )
        outputs[f"d3_probability_sensitivity_{prefix}"] = float(
            np.mean(np.abs(original_probability - changed_probability))
        )
        outputs[f"auxiliary_train_r2_{prefix}"] = float(
            auxiliary.score(X_train[STABLE_FEATURES], X_train[target])
        )
    outputs["d3_excess"] = (
        outputs["d3_flip_rate_spurious"] - outputs["d3_flip_rate_noise"]
    )
    outputs["d3_probability_sensitivity_excess"] = (
        outputs["d3_probability_sensitivity_spurious"]
        - outputs["d3_probability_sensitivity_noise"]
    )
    return outputs


def conditional_permutation_importance(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """Permute validation residuals around a training-only conditional mean."""
    auxiliary = fit_conditional_ols(X_train, "x_spurious")
    expected = auxiliary.predict(X_validation[STABLE_FEATURES])
    residual = X_validation["x_spurious"].to_numpy() - expected
    permuted = X_validation.copy()
    permuted["x_spurious"] = expected + np.random.default_rng(seed).permutation(residual)
    original_accuracy = float(accuracy_score(y_validation, model.predict(X_validation)))
    conditional_accuracy = float(accuracy_score(y_validation, model.predict(permuted)))
    return {
        "d2_original_accuracy": original_accuracy,
        "d2_conditional_permuted_accuracy": conditional_accuracy,
        "d2_conditional_permutation_drop": original_accuracy - conditional_accuracy,
    }


def consistent_contradictory_gap(
    model: BaseEstimator, X_validation: pd.DataFrame, y_validation: np.ndarray
) -> dict[str, float | int]:
    """Measure IID accuracy for shortcut-consistent versus contradictory groups."""
    shortcut_label = (X_validation["x_spurious"].to_numpy() >= 0.0).astype(int)
    consistent = shortcut_label == y_validation
    predictions = np.asarray(model.predict(X_validation))
    n_consistent = int(consistent.sum())
    n_contradictory = int((~consistent).sum())
    accuracy_consistent = float(np.mean(predictions[consistent] == y_validation[consistent]))
    accuracy_contradictory = float(
        np.mean(predictions[~consistent] == y_validation[~consistent])
    )
    return {
        "d4_n_consistent": n_consistent,
        "d4_n_contradictory": n_contradictory,
        "d4_accuracy_consistent": accuracy_consistent,
        "d4_accuracy_contradictory": accuracy_contradictory,
        "d4_accuracy_gap": accuracy_consistent - accuracy_contradictory,
    }


def model_specific_usage(
    model_name: str, model: BaseEstimator, feature_names: list[str]
) -> dict[str, Any]:
    """Return a model-specific D6 audit without pretending measures are comparable."""
    index = feature_names.index("x_spurious")
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    result: dict[str, Any] = {
        "d6_measure": "not_applicable",
        "d6_value": np.nan,
        "d6_split_count": np.nan,
        "d6_note": "No clean model-specific audit implemented.",
    }
    if model_name == "logistic_regression":
        weights = np.abs(estimator.coef_[0])
        result.update(
            d6_measure="normalized_absolute_standardized_coefficient",
            d6_value=float(weights[index] / weights.sum()),
            d6_note="Within-model normalized coefficient; not cross-family comparable.",
        )
    elif model_name == "rbf_svm":
        result["d6_note"] = "RBF kernel has no meaningful input-space feature weight."
    elif model_name == "random_forest":
        split_count = sum(
            int(np.sum(tree.tree_.feature == index)) for tree in estimator.estimators_
        )
        result.update(
            d6_measure="impurity_importance_fraction",
            d6_value=float(estimator.feature_importances_[index]),
            d6_split_count=split_count,
            d6_note="Native impurity importance and split count; audit only.",
        )
    elif model_name == "gradient_boosting":
        split_count = 0
        gain = 0.0
        total_gain = 0.0
        for iteration in getattr(estimator, "_predictors", []):
            for predictor in iteration:
                nodes = predictor.nodes
                internal = nodes["is_leaf"] == 0
                node_gain = np.maximum(nodes["gain"][internal], 0.0)
                total_gain += float(node_gain.sum())
                selected = internal & (nodes["feature_idx"] == index)
                split_count += int(selected.sum())
                gain += float(np.maximum(nodes["gain"][selected], 0.0).sum())
        result.update(
            d6_measure="private_histgb_gain_fraction",
            d6_value=float(gain / total_gain) if total_gain > 0 else np.nan,
            d6_split_count=split_count,
            d6_note="Audit based on sklearn private tree nodes; version-sensitive.",
        )
    elif model_name == "mlp":
        first_layer = np.abs(estimator.coefs_[0])
        per_input = first_layer.sum(axis=1)
        result.update(
            d6_measure="normalized_first_layer_absolute_weight",
            d6_value=float(per_input[index] / per_input.sum()),
            d6_note="First-layer weight audit; not a causal or cross-family importance.",
        )
    return result


def loco_metrics(accuracy_all: float, accuracy_stable: float) -> dict[str, float]:
    """Compute raw and preregistered normalized LOCO without clipping negatives."""
    raw = accuracy_all - accuracy_stable
    stable_error = 1.0 - accuracy_stable
    normalized = raw / stable_error if stable_error > 0 else np.nan
    return {"raw_loco": raw, "loco_normalized": normalized}
