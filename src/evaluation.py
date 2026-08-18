"""Leakage-resistant fitting, evaluation, and robustness aggregation."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from src.data_generation import GeneratedDataset
from src.models import build_model


def select_and_fit_model(
    model_name: str,
    model_config: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    seed: int,
) -> tuple[BaseEstimator, dict[str, float]]:
    """Fit fixed V1 settings and score only on IID validation data.

    Deliberately no test/OOD argument is accepted. Future tuning must remain
    inside this boundary and may use only ``X_train`` and ``X_val``.
    """
    model = build_model(model_name, model_config, seed)
    model.fit(X_train, y_train)
    validation_metrics = evaluate_predictions(model, X_val, y_val)
    return model, validation_metrics


def evaluate_predictions(
    model: BaseEstimator, X: pd.DataFrame, y: np.ndarray
) -> dict[str, float]:
    """Calculate the common classification metrics for one environment."""
    predictions = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "log_loss": float("nan"),
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        metrics["log_loss"] = float(log_loss(y, probabilities, labels=[0, 1]))
    return metrics


def evaluate_environments(
    model: BaseEstimator,
    environments: Mapping[float, GeneratedDataset],
    features: list[str],
    seed: int,
    model_name: str,
    feature_condition: str,
) -> list[dict[str, float | int | str]]:
    """Evaluate an already-fitted model; test environments cannot affect fitting."""
    rows: list[dict[str, float | int | str]] = []
    for rho, dataset in environments.items():
        metrics = evaluate_predictions(model, dataset.X[features], dataset.y)
        rows.append(
            {
                "seed": seed,
                "model": model_name,
                "feature_condition": feature_condition,
                "rho_test": float(rho),
                **metrics,
            }
        )
    return rows


def per_seed_robustness(
    results: pd.DataFrame,
    diagnostics: pd.DataFrame,
    rho_train: float,
) -> pd.DataFrame:
    """Compute descriptive robustness metrics for each fitted model."""
    records: list[dict[str, float | int | str]] = []
    keys = ["seed", "model", "feature_condition"]
    diag_index = diagnostics.set_index(keys)
    for key, group in results.groupby(keys, sort=False):
        seed, model, condition = key
        iid = float(group.loc[np.isclose(group["rho_test"], rho_train), "accuracy"].iloc[0])
        ood = group.loc[~np.isclose(group["rho_test"], rho_train)]
        average_ood = float(ood["accuracy"].mean())
        severe = group.loc[group["rho_test"].isin([-0.6, -0.9]), "accuracy"]
        shift_distance = np.abs(group["rho_test"].to_numpy(dtype=float) - rho_train)
        slope = float(np.polyfit(shift_distance, group["accuracy"].to_numpy(), 1)[0])
        diagnostic = diag_index.loc[key]
        records.append(
            {
                "seed": int(seed),
                "model": model,
                "feature_condition": condition,
                "validation_accuracy": float(diagnostic["validation_accuracy"]),
                "iid_accuracy": iid,
                "average_ood_accuracy": average_ood,
                "worst_environment_accuracy": float(group["accuracy"].min()),
                "ood_gap": iid - average_ood,
                "severe_shift_accuracy": float(severe.mean()),
                "degradation_slope": slope,
                "spurious_reliance_score": float(diagnostic["spurious_reliance_score"]),
            }
        )
    return pd.DataFrame.from_records(records)


def aggregate_model_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed model statistics with standard errors."""
    metrics = [
        "validation_accuracy",
        "iid_accuracy",
        "average_ood_accuracy",
        "worst_environment_accuracy",
        "ood_gap",
        "severe_shift_accuracy",
        "degradation_slope",
        "spurious_reliance_score",
    ]
    grouped = per_seed.groupby(["model", "feature_condition"], sort=False)[metrics]
    mean = grouped.mean().add_suffix("_mean")
    sem = grouped.sem().fillna(0.0).add_suffix("_se")
    count = grouped.size().rename("n_seeds")
    return pd.concat([mean, sem, count], axis=1).reset_index()

