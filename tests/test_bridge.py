"""Gates for the frozen learner-only Bridge implementation."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

from src.bridge_experiment import (
    BRIDGE_INIT_COMPONENT,
    DATASET_SEEDS,
    FEATURE_ORDER,
    RHO_VALUES,
    build_phase2_bundle,
    fit_bridge_mlp_l,
    load_mlp_l_entry,
    rho_code,
    run_bridge_smoke,
    validate_frozen_task_config,
)
from src.utils import child_seed, load_config


def _phase2_config() -> dict:
    return load_config("experiments/PHASE-2/config.yaml")


def _small_config() -> dict:
    config = copy.deepcopy(_phase2_config())
    config["data"].update(n_train=96, n_validation=64, n_iid_test=64, n_ood_test=64, n_probe=80)
    return config


def test_bridge_matrix_and_frozen_task_gate() -> None:
    config = _phase2_config()
    validate_frozen_task_config(config)
    assert RHO_VALUES == (0.9, 0.0)
    assert DATASET_SEEDS == tuple(range(20))
    assert len(RHO_VALUES) * len(DATASET_SEEDS) == 40
    assert FEATURE_ORDER == ("x1", "x2", "x3", "x_spurious", "z")


def test_mlp_l_extraction_and_seed() -> None:
    entry = load_mlp_l_entry(Path("experiments/PHASE-1/config.yaml"))
    assert entry.config_id == "MLP-L"
    assert entry.family == "mlp"
    assert entry.parameters == {"hidden_layer_sizes": [16], "alpha": 0.0001, "learning_rate_init": 0.001, "max_iter": 250, "early_stopping": True}
    assert child_seed(20260818, 3, rho_code(0.9), BRIDGE_INIT_COMPONENT) == child_seed(20260818, 3, 1900, 9100)


def test_task_pairing_reuses_phase2_stable_components() -> None:
    config = _small_config()
    neutral = build_phase2_bundle(config, 0.0, 0)
    strong = build_phase2_bundle(config, 0.9, 0)
    np.testing.assert_array_equal(neutral.train.y, strong.train.y)
    np.testing.assert_allclose(neutral.train.X[["x1", "x2", "x3", "z"]], strong.train.X[["x1", "x2", "x3", "z"]])
    np.testing.assert_allclose(neutral.probe_X, strong.probe_X)
    assert np.all(neutral.probe_X["x_spurious"].to_numpy() == 0.0)
    assert np.all(neutral.probe_X["z"].to_numpy() == 0.0)


def test_scaler_is_fit_on_training_data_only() -> None:
    config = _small_config()
    entry = load_mlp_l_entry(Path("experiments/PHASE-1/config.yaml"))
    bundle = build_phase2_bundle(config, 0.9, 0)
    baseline = fit_bridge_mlp_l(config, bundle, entry)
    # The fit function receives only bundle.train.X/y; test identity by changing only evaluation data.
    altered_probe = bundle.probe_X.copy()
    altered_probe.loc[:, list(FEATURE_ORDER)] = altered_probe.loc[:, list(FEATURE_ORDER)] + 10000.0
    altered = type(bundle)(bundle.rho_train, bundle.dataset_seed, bundle.train, bundle.validation, bundle.iid_test, bundle.ood, altered_probe, bundle.probe_y)
    changed_eval = fit_bridge_mlp_l(config, altered, entry)
    np.testing.assert_allclose(baseline["scaler_train_mean"], changed_eval["scaler_train_mean"])
    np.testing.assert_allclose(baseline["scaler_train_scale"], changed_eval["scaler_train_scale"])
    assert baseline["n_iter"] == changed_eval["n_iter"]


def test_bridge_smoke_is_separate_from_formal_statistics(tmp_path: Path) -> None:
    summary = run_bridge_smoke(_phase2_config(), Path("experiments/PHASE-1/config.yaml"), tmp_path)
    assert summary["status"] == "SMOKE_PASS"
    assert summary["formal_statistics"] is False
    assert summary["fits"] == 2
