"""Model-agnostic shortcut reliance and data sanity diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


def permutation_reliance_score(
    model: BaseEstimator,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    seed: int,
) -> tuple[float, float, float]:
    """Measure validation accuracy lost by permuting the spurious feature."""
    original_accuracy = float(accuracy_score(y_val, model.predict(X_val)))
    if "x_spurious" not in X_val.columns:
        return original_accuracy, original_accuracy, 0.0
    permuted = X_val.copy()
    rng = np.random.default_rng(seed)
    permuted["x_spurious"] = rng.permutation(permuted["x_spurious"].to_numpy())
    permuted_accuracy = float(accuracy_score(y_val, model.predict(permuted)))
    return original_accuracy, permuted_accuracy, original_accuracy - permuted_accuracy


def validate_environment_metadata(metadata: pd.DataFrame) -> list[str]:
    """Return human-readable warnings for failed generation sanity checks."""
    warnings: list[str] = []
    if ((metadata["positive_rate"] < 0.45) | (metadata["positive_rate"] > 0.55)).any():
        warnings.append("At least one environment has class balance outside [0.45, 0.55].")
    correlation_error = (
        metadata["empirical_spurious_correlation"] - metadata["rho_requested"]
    ).abs()
    if (correlation_error > 0.08).any():
        warnings.append("At least one empirical spurious correlation differs from rho by > 0.08.")
    for feature in ["x1", "x2", "x3"]:
        if (metadata[f"{feature}_mean"].abs() > 0.12).any():
            warnings.append(f"At least one {feature} mean is unexpectedly far from zero.")
        if ((metadata[f"{feature}_std"] - 1.0).abs() > 0.12).any():
            warnings.append(f"At least one {feature} std is unexpectedly far from one.")
    return warnings

