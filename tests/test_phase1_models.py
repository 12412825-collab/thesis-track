"""Phase 1 registry, capacity, and leakage-boundary contracts."""

from __future__ import annotations

import inspect

import numpy as np

from src.phase1_experiment import fit_phase1_model
from src.phase1_models import (
    add_empirical_capacity_index,
    build_phase1_model,
    extract_capacity_metrics,
    load_registry,
)
from src.utils import load_config
from src.verification_data import VERIFICATION_ALL_FEATURES, generate_verification_dataset


def _config() -> dict:
    return load_config("experiments/PHASE-1/config.yaml")


def test_registry_has_three_unique_configs_per_frozen_family() -> None:
    entries = load_registry(_config())
    assert len(entries) == 15
    assert len({entry.config_id for entry in entries}) == 15
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.family] = counts.get(entry.family, 0) + 1
    assert set(counts.values()) == {3}


def test_fit_boundary_cannot_receive_validation_or_ood_data() -> None:
    parameters = set(inspect.signature(fit_phase1_model).parameters)
    assert parameters == {"entry", "X_train", "y_train", "seed"}


def test_every_family_exports_a_finite_primary_capacity_proxy() -> None:
    config = _config()
    dataset = generate_verification_dataset(
        n_samples=300,
        rho=0.5,
        base_seed=101,
        noise_seed=102,
        stable_coefficients=config["data"]["stable_coefficients"],
        label_noise=config["data"]["label_noise"],
        split="train",
    )
    seen: set[str] = set()
    for index, entry in enumerate(load_registry(config)):
        if entry.family in seen:
            continue
        seen.add(entry.family)
        model = build_phase1_model(entry, seed=200 + index).fit(
            dataset.X[VERIFICATION_ALL_FEATURES], dataset.y
        )
        capacity = extract_capacity_metrics(
            entry, model, dataset.X[VERIFICATION_ALL_FEATURES], kernel_rank_max_samples=128
        )
        assert capacity["capacity_proxy_name"]
        assert np.isfinite(capacity["learned_capacity_primary"])

