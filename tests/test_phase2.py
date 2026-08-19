"""Protocol, pairing, and resume-safety tests for frozen Phase 2."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from src.phase2_experiment import (
    CHECKPOINTS,
    SCHEDULES,
    WIDTHS,
    build_phase2_bundle,
    cosine_learning_rate,
    expected_fit_count,
    fit_one,
    run_phase2,
)


def _config() -> dict:
    with open("experiments/PHASE-2/config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = copy.deepcopy(config)
    config["data"]["dataset_seeds"] = [0]
    config["data"]["rho_train_values"] = [0.7, 0.0]
    config["data"].update(n_train=96, n_validation=64, n_iid_test=64, n_ood_test=64, n_probe=80)
    config["optimization"]["epochs"] = 20
    config["measurements"]["checkpoints"] = [20]
    return config


def test_frozen_factor_levels_and_formal_fit_count() -> None:
    with open("experiments/PHASE-2/config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert tuple(config["model"]["widths"]) == WIDTHS
    assert tuple(config["optimization"]["schedules"]) == SCHEDULES
    assert expected_fit_count(config) == 360
    assert config["optimization"]["epochs"] == 200
    assert config["optimization"]["batch_size"] == 64
    assert config["optimization"]["optimizer"] == "sgd"
    assert config["optimization"]["early_stopping"] is False


def test_cosine_schedule_has_frozen_endpoints() -> None:
    assert np.isclose(cosine_learning_rate(1, 200, 0.01, 0.001), 0.01)
    assert np.isclose(cosine_learning_rate(200, 200, 0.01, 0.001), 0.001)
    assert all(cosine_learning_rate(i, 200, 0.01, 0.001) > cosine_learning_rate(i + 1, 200, 0.01, 0.001) for i in range(1, 200))


def test_bundle_pairs_data_and_probe_is_neutral() -> None:
    config = _config()
    neutral = build_phase2_bundle(config, 0.0, 0)
    strong = build_phase2_bundle(config, 0.9, 0)
    np.testing.assert_array_equal(neutral.train.y, strong.train.y)
    np.testing.assert_allclose(neutral.train.X[["x1", "x2", "x3", "z"]], strong.train.X[["x1", "x2", "x3", "z"]])
    assert np.all(neutral.probe_X["x_spurious"].to_numpy() == 0.0)
    assert np.all(neutral.probe_X["z"].to_numpy() == 0.0)


def test_schedule_pair_uses_same_initialization_and_shuffle_seed() -> None:
    config = _config()
    bundle = build_phase2_bundle(config, 0.7, 0)
    constant = fit_one(config, bundle, 8, "constant", dataset_seed=0)
    cosine = fit_one(config, bundle, 8, "cosine", dataset_seed=0)
    assert constant["init_seed"] == cosine["init_seed"]
    assert constant["shuffle_seed"] == cosine["shuffle_seed"]
    assert constant["n_probe"] == 80
    assert len(constant["trajectory"]) == 1


def test_smoke_run_is_resumable(tmp_path: Path) -> None:
    config = _config()
    first = run_phase2(config, tmp_path, smoke=True, resume=True, max_fits=2)
    assert first["successful_fits"] == 2
    second = run_phase2(config, tmp_path, smoke=True, resume=True, max_fits=2)
    assert second["successful_fits"] == 2
    assert len(list((tmp_path / "fits").glob("*.json"))) == 2
    records = [json.loads(path.read_text(encoding="utf-8")) for path in (tmp_path / "fits").glob("*.json")]
    assert all(record["status"] == "SUCCESS" for record in records)
