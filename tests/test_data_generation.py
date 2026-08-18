"""Tests for deterministic and isolated environment generation."""

import numpy as np
import pandas as pd

from src.data_generation import STABLE_FEATURES, generate_dataset


def test_same_seed_produces_identical_dataset() -> None:
    first = generate_dataset(1000, rho=0.6, seed=42)
    second = generate_dataset(1000, rho=0.6, seed=42)
    pd.testing.assert_frame_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    assert first.metadata == second.metadata


def test_rho_changes_spurious_correlation_in_expected_direction() -> None:
    positive = generate_dataset(20_000, rho=0.8, seed=10)
    neutral = generate_dataset(20_000, rho=0.0, seed=11)
    negative = generate_dataset(20_000, rho=-0.8, seed=12)
    correlations = [
        positive.metadata["empirical_spurious_correlation"],
        neutral.metadata["empirical_spurious_correlation"],
        negative.metadata["empirical_spurious_correlation"],
    ]
    assert correlations[0] > 0.7
    assert abs(correlations[1]) < 0.04
    assert correlations[2] < -0.7


def test_stable_feature_distributions_are_unchanged_by_rho() -> None:
    low = generate_dataset(50_000, rho=-0.9, seed=101)
    high = generate_dataset(50_000, rho=0.9, seed=102)
    for feature in STABLE_FEATURES:
        assert abs(low.X[feature].mean() - high.X[feature].mean()) < 0.03
        assert abs(low.X[feature].std() - high.X[feature].std()) < 0.03


def test_requested_rho_is_empirically_approximated() -> None:
    for index, rho in enumerate([-0.9, -0.3, 0.0, 0.6, 0.9]):
        dataset = generate_dataset(30_000, rho=rho, seed=200 + index)
        assert abs(dataset.metadata["empirical_spurious_correlation"] - rho) < 0.04

