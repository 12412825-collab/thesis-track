"""Synthetic binary classification data with a controlled spurious feature."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


STABLE_FEATURES = ["x1", "x2", "x3"]
ALL_FEATURES = [*STABLE_FEATURES, "x_spurious"]


@dataclass(frozen=True)
class GeneratedDataset:
    """A generated feature matrix, labels, and generation diagnostics."""

    X: pd.DataFrame
    y: np.ndarray
    metadata: dict[str, float | int | str]


def generate_dataset(
    n_samples: int,
    rho: float,
    seed: int,
    stable_coefficients: list[float] | tuple[float, ...] = (1.2, -1.0, 0.8),
    label_noise: float = 0.5,
    split: str = "unspecified",
) -> GeneratedDataset:
    """Generate one environment while changing only the spurious mechanism.

    Stable features and label noise are sampled from identical distributions in
    every environment. ``rho`` controls the correlation between ``x_spurious``
    and the signed target and must lie in [-1, 1].
    """
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [-1, 1], received {rho}")
    coefficients = np.asarray(stable_coefficients, dtype=float)
    if coefficients.shape != (3,):
        raise ValueError("stable_coefficients must contain exactly three values")
    if n_samples <= 0 or label_noise < 0:
        raise ValueError("n_samples must be positive and label_noise non-negative")

    rng = np.random.default_rng(seed)
    stable = rng.normal(size=(n_samples, 3))
    score = stable @ coefficients + rng.normal(scale=label_noise, size=n_samples)
    y = (score > 0.0).astype(np.int64)
    y_signed = 2 * y - 1
    epsilon = rng.normal(size=n_samples)
    x_spurious = rho * y_signed + np.sqrt(max(0.0, 1.0 - rho**2)) * epsilon

    X = pd.DataFrame(stable, columns=STABLE_FEATURES)
    X["x_spurious"] = x_spurious
    empirical_correlation = float(np.corrcoef(x_spurious, y_signed)[0, 1])
    metadata: dict[str, float | int | str] = {
        "split": split,
        "seed": seed,
        "n_samples": n_samples,
        "rho_requested": float(rho),
        "empirical_spurious_correlation": empirical_correlation,
        "positive_rate": float(y.mean()),
    }
    for feature in STABLE_FEATURES:
        metadata[f"{feature}_mean"] = float(X[feature].mean())
        metadata[f"{feature}_std"] = float(X[feature].std(ddof=0))
    return GeneratedDataset(X=X, y=y, metadata=metadata)

