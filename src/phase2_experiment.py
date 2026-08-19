"""Frozen Phase 2 training-regime experiment.

This module intentionally implements only the supervisor-approved 3x2 MLP
design.  It has no model-selection logic and no hyperparameter search.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data_generation import GeneratedDataset
from src.utils import child_seed, save_json
from src.verification_data import generate_verification_dataset


PHASE2_PREREG_SHA = "654e72b39310847b0d369796c31825e379aaf909"
CHECKPOINTS = (20, 40, 60, 80, 100, 120, 140, 160, 180, 200)
WIDTHS = (8, 64, 512)
SCHEDULES = ("constant", "cosine")


@dataclass(frozen=True)
class Phase2Bundle:
    """All data used by one rho and dataset seed, shared by all model cells."""

    rho_train: float
    dataset_seed: int
    train: GeneratedDataset
    validation: GeneratedDataset
    iid_test: GeneratedDataset
    ood: dict[float, GeneratedDataset]
    probe_X: pd.DataFrame
    probe_y: np.ndarray


class OneHiddenLayerMLP:
    """Small NumPy MLP with exactly one ReLU hidden layer and SGD momentum."""

    def __init__(self, n_features: int, width: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, math.sqrt(2.0 / n_features), (n_features, width))
        self.b1 = np.zeros(width, dtype=np.float64)
        self.W2 = rng.normal(0.0, math.sqrt(2.0 / width), (width, 1))
        self.b2 = np.zeros(1, dtype=np.float64)
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        result = np.empty_like(logits)
        positive = logits >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        result[~positive] = exp_logits / (1.0 + exp_logits)
        return result

    def logits(self, X: np.ndarray) -> np.ndarray:
        hidden = np.maximum(X @ self.W1 + self.b1, 0.0)
        return (hidden @ self.W2 + self.b2).ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.logits(X) >= 0.0).astype(np.int64)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        logits = self.logits(X)
        values = np.maximum(logits, 0.0) - logits * y + np.log1p(np.exp(-np.abs(logits)))
        return float(np.mean(values))

    def train_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        hidden_linear = X @ self.W1 + self.b1
        hidden = np.maximum(hidden_linear, 0.0)
        logits = (hidden @ self.W2 + self.b2).ravel()
        residual = self._sigmoid(logits) - y
        scale = 1.0 / len(y)
        dW2 = (hidden.T @ residual[:, None]) * scale
        db2 = residual.mean(keepdims=True)
        dhidden = residual[:, None] @ self.W2.T
        dhidden[hidden_linear <= 0.0] = 0.0
        dW1 = (X.T @ dhidden) * scale
        db1 = dhidden.sum(axis=0) * scale

        self.vW2 = momentum * self.vW2 - learning_rate * (dW2 + weight_decay * self.W2)
        self.vb2 = momentum * self.vb2 - learning_rate * db2
        self.vW1 = momentum * self.vW1 - learning_rate * (dW1 + weight_decay * self.W1)
        self.vb1 = momentum * self.vb1 - learning_rate * db1
        self.W2 += self.vW2
        self.b2 += self.vb2
        self.W1 += self.vW1
        self.b1 += self.vb1


def _rho_code(rho: float) -> int:
    return int(round((rho + 1.0) * 1000))


def _as_array(dataset: GeneratedDataset) -> np.ndarray:
    return dataset.X[["x1", "x2", "x3", "x_spurious", "z"]].to_numpy(dtype=np.float64)


def build_phase2_bundle(config: dict[str, Any], rho_train: float, dataset_seed: int) -> Phase2Bundle:
    """Generate paired data by reusing the frozen project generator."""
    experiment = config["experiment"]
    data = config["data"]
    master = int(experiment["master_seed"])
    coefficients = data["stable_coefficients"]
    label_noise = float(data["label_noise"])
    train = generate_verification_dataset(
        n_samples=int(data["n_train"]),
        rho=rho_train,
        base_seed=child_seed(master, dataset_seed, 1),
        noise_seed=child_seed(master, dataset_seed, 1, 999),
        stable_coefficients=coefficients,
        label_noise=label_noise,
        split="train",
    )
    validation = generate_verification_dataset(
        n_samples=int(data["n_validation"]),
        rho=rho_train,
        base_seed=child_seed(master, dataset_seed, 2),
        noise_seed=child_seed(master, dataset_seed, 2, 999),
        stable_coefficients=coefficients,
        label_noise=label_noise,
        split="validation",
    )
    iid_test = generate_verification_dataset(
        n_samples=int(data["n_iid_test"]),
        rho=rho_train,
        base_seed=child_seed(master, dataset_seed, 100),
        noise_seed=child_seed(master, dataset_seed, 100, 999),
        stable_coefficients=coefficients,
        label_noise=label_noise,
        split="iid_test",
    )
    ood: dict[float, GeneratedDataset] = {}
    for rho_ood in data["rho_ood_values"]:
        rho_ood = float(rho_ood)
        if np.isclose(rho_ood, rho_train):
            continue
        ood[rho_ood] = generate_verification_dataset(
            n_samples=int(data["n_ood_test"]),
            rho=rho_ood,
            base_seed=child_seed(master, dataset_seed, 100, _rho_code(rho_ood)),
            noise_seed=child_seed(master, dataset_seed, 100, _rho_code(rho_ood), 999),
            stable_coefficients=coefficients,
            label_noise=label_noise,
            split=f"ood_{rho_ood}",
        )

    probe = generate_verification_dataset(
        n_samples=int(data["n_probe"]),
        rho=0.0,
        base_seed=child_seed(master, dataset_seed, 9000),
        noise_seed=child_seed(master, dataset_seed, 9000, 999),
        stable_coefficients=coefficients,
        label_noise=0.0,
        split="stable_boundary_probe",
    )
    probe_X = probe.X.copy()
    probe_X.loc[:, "x_spurious"] = 0.0
    probe_X.loc[:, "z"] = 0.0
    return Phase2Bundle(
        rho_train=rho_train,
        dataset_seed=dataset_seed,
        train=train,
        validation=validation,
        iid_test=iid_test,
        ood=ood,
        probe_X=probe_X,
        probe_y=probe.y.copy(),
    )


def cosine_learning_rate(epoch: int, epochs: int, initial: float, terminal: float) -> float:
    """Return the frozen cosine schedule, with terminal LR at epoch ``epochs``."""
    if not 1 <= epoch <= epochs:
        raise ValueError("epoch must be within the training budget")
    fraction = (epoch - 1) / max(1, epochs - 1)
    return terminal + 0.5 * (initial - terminal) * (1.0 + math.cos(math.pi * fraction))


def _checkpoint_metrics(
    model: OneHiddenLayerMLP,
    bundle: Phase2Bundle,
    train_X: np.ndarray,
    train_y: np.ndarray,
    iid_X: np.ndarray,
    iid_y: np.ndarray,
) -> dict[str, float | int]:
    probe_prediction = model.predict(_as_array_from_frame(bundle.probe_X))
    iid_prediction = model.predict(iid_X)
    original = iid_prediction
    flipped = model.predict(iid_X.copy() * 1.0)  # copied below to preserve column order
    flipped_X = iid_X.copy()
    flipped_X[:, 3] *= -1.0
    flipped = model.predict(flipped_X)
    return {
        "checkpoint": 0,
        "fidelity": float(np.mean(probe_prediction == bundle.probe_y)),
        "d3": float(np.mean(original != flipped)),
        "train_loss": model.loss(train_X, train_y),
        "iid_accuracy": float(np.mean(iid_prediction == iid_y)),
    }


def _as_array_from_frame(frame: pd.DataFrame) -> np.ndarray:
    return frame[["x1", "x2", "x3", "x_spurious", "z"]].to_numpy(dtype=np.float64)


def fit_one(
    config: dict[str, Any],
    bundle: Phase2Bundle,
    width: int,
    schedule: str,
    *,
    dataset_seed: int,
) -> dict[str, Any]:
    """Fit one preregistered cell and return terminal plus checkpoint records."""
    if width not in WIDTHS or schedule not in SCHEDULES:
        raise ValueError("Phase 2 only permits widths {8,64,512} and two schedules")
    optimization = config["optimization"]
    measurements = config["measurements"]
    experiment = config["experiment"]
    rho_code = _rho_code(bundle.rho_train)
    master = int(experiment["master_seed"])
    init_seed = child_seed(master, dataset_seed, rho_code, width, 7100)
    shuffle_seed = child_seed(master, dataset_seed, rho_code, 7200)
    model = OneHiddenLayerMLP(n_features=5, width=width, seed=init_seed)
    train_X = _as_array(bundle.train)
    train_y = bundle.train.y.astype(np.float64)
    iid_X = _as_array(bundle.iid_test)
    iid_y = bundle.iid_test.y
    epochs = int(optimization["epochs"])
    batch_size = int(optimization["batch_size"])
    momentum = float(optimization["momentum"])
    weight_decay = float(optimization["weight_decay"])
    initial_lr = float(optimization["initial_lr"])
    terminal_lr = float(optimization["cosine_terminal_lr"])
    checkpoints = tuple(int(x) for x in measurements["checkpoints"])
    if checkpoints[-1] != epochs:
        raise ValueError("the final preregistered checkpoint must equal the epoch budget")

    order_rng = np.random.default_rng(shuffle_seed)
    trajectory: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        learning_rate = initial_lr if schedule == "constant" else cosine_learning_rate(
            epoch, epochs, initial_lr, terminal_lr
        )
        order = order_rng.permutation(len(train_y))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            model.train_batch(
                train_X[batch],
                train_y[batch],
                learning_rate=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        if epoch in checkpoints:
            probe_prediction = model.predict(_as_array_from_frame(bundle.probe_X))
            iid_prediction = model.predict(iid_X)
            flipped_X = iid_X.copy()
            flipped_X[:, 3] *= -1.0
            trajectory.append(
                {
                    "checkpoint": epoch,
                    "fidelity": float(np.mean(probe_prediction == bundle.probe_y)),
                    "d3": float(np.mean(iid_prediction != model.predict(flipped_X))),
                    "train_loss": model.loss(train_X, train_y),
                    "iid_accuracy": float(np.mean(iid_prediction == iid_y)),
                    "learning_rate": float(learning_rate),
                }
            )

    terminal = trajectory[-1]
    ood_rows: list[dict[str, float | int]] = []
    for rho_ood, dataset in sorted(bundle.ood.items()):
        prediction = model.predict(_as_array(dataset))
        accuracy = float(np.mean(prediction == dataset.y))
        ood_rows.append(
            {
                "rho_ood": float(rho_ood),
                "accuracy": accuracy,
                "ood_gap": float(terminal["iid_accuracy"] - accuracy),
            }
        )
    d3_values = [float(row["d3"]) for row in trajectory]
    sat = math.inf
    for previous, current in zip(trajectory, trajectory[1:]):
        if previous["d3"] >= float(measurements["d3_threshold"]) and current["d3"] >= float(measurements["d3_threshold"]):
            sat = int(current["checkpoint"])
            break
    fidelity_values = [float(row["fidelity"]) for row in trajectory]
    f20 = fidelity_values[0]
    fmin = min(fidelity_values)
    recovery = float(terminal["fidelity"] - fmin)
    drop = float(f20 - fmin)
    if drop >= float(measurements["escape_drop_threshold"]) and recovery >= float(measurements["escape_recovery_threshold"]):
        trajectory_class = "Escape-shaped"
    elif drop >= float(measurements["escape_drop_threshold"]):
        trajectory_class = "Plateau"
    elif drop < float(measurements["escape_drop_threshold"]):
        trajectory_class = "No-acquisition / no-distortion"
    else:
        trajectory_class = "Other"
    fit_id = f"rho{bundle.rho_train:g}_seed{dataset_seed:02d}_width{width}_{schedule}".replace(".", "p")
    return {
        "fit_id": fit_id,
        "status": "SUCCESS",
        "prereg_frozen_sha": PHASE2_PREREG_SHA,
        "rho_train": float(bundle.rho_train),
        "dataset_seed": int(dataset_seed),
        "width": int(width),
        "schedule": schedule,
        "init_seed": int(init_seed),
        "shuffle_seed": int(shuffle_seed),
        "n_train": int(len(train_y)),
        "n_validation": int(len(bundle.validation.y)),
        "n_iid_test": int(len(iid_y)),
        "n_probe": int(len(bundle.probe_y)),
        "epochs": epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "initial_lr": initial_lr,
        "cosine_terminal_lr": terminal_lr,
        "fit_seconds": float(time.perf_counter() - started),
        "terminal_fidelity": float(terminal["fidelity"]),
        "terminal_d3": float(terminal["d3"]),
        "terminal_train_loss": float(terminal["train_loss"]),
        "terminal_iid_accuracy": float(terminal["iid_accuracy"]),
        "final_ood_gap": float(np.mean([row["ood_gap"] for row in ood_rows])) if ood_rows else float("nan"),
        "sat": sat,
        "fidelity_min": float(fmin),
        "fidelity_recovery": recovery,
        "fidelity_drop_from_20": drop,
        "trajectory_class": trajectory_class,
        "trajectory": trajectory,
        "ood": ood_rows,
    }


def iter_formal_cells(config: dict[str, Any]) -> Iterable[tuple[float, int, int, str]]:
    """Yield the exact 360-cell factorial in a stable order."""
    data = config["data"]
    for rho in data["rho_train_values"]:
        for dataset_seed in data["dataset_seeds"]:
            for width in config["model"]["widths"]:
                for schedule in config["optimization"]["schedules"]:
                    yield float(rho), int(dataset_seed), int(width), str(schedule)


def expected_fit_count(config: dict[str, Any]) -> int:
    return len(list(iter_formal_cells(config)))


def run_phase2(
    config: dict[str, Any],
    output_root: Path,
    *,
    resume: bool = True,
    smoke: bool = False,
    max_fits: int | None = None,
) -> dict[str, Any]:
    """Run formal cells or a clearly separate smoke subset with per-fit resume."""
    run_config = json.loads(json.dumps(config))
    if smoke:
        run_config["data"]["dataset_seeds"] = [0]
        run_config["data"]["rho_train_values"] = [0.7, 0.0]
        run_config["data"]["n_train"] = 320
        run_config["data"]["n_validation"] = 160
        run_config["data"]["n_iid_test"] = 200
        run_config["data"]["n_ood_test"] = 200
        run_config["data"]["n_probe"] = 300
        run_config["optimization"]["epochs"] = 20
        run_config["measurements"]["checkpoints"] = [20]
        run_config["statistics"]["bootstrap_resamples"] = 200
    output_root.mkdir(parents=True, exist_ok=True)
    fit_root = output_root / "fits"
    fit_root.mkdir(parents=True, exist_ok=True)
    cells = list(iter_formal_cells(run_config))
    if max_fits is not None:
        cells = cells[:max_fits]
    completed = 0
    failures = 0
    runtimes: list[float] = []
    bundle_cache: dict[tuple[float, int], Phase2Bundle] = {}
    for index, (rho, dataset_seed, width, schedule) in enumerate(cells, start=1):
        fit_id = f"rho{rho:g}_seed{dataset_seed:02d}_width{width}_{schedule}".replace(".", "p")
        path = fit_root / f"{fit_id}.json"
        print(f"Phase2 fit {index}/{len(cells)}: {fit_id}", flush=True)
        if resume and path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("status") == "SUCCESS":
                    completed += 1
                    runtimes.append(float(existing.get("fit_seconds", 0.0)))
                    print(f"  resumed SUCCESS ({existing.get('fit_seconds', 0.0):.2f}s)", flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        key = (rho, dataset_seed)
        if key not in bundle_cache:
            bundle_cache[key] = build_phase2_bundle(run_config, rho, dataset_seed)
        try:
            result = fit_one(run_config, bundle_cache[key], width, schedule, dataset_seed=dataset_seed)
            path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
            completed += 1
            runtimes.append(float(result["fit_seconds"]))
            print(f"  completed SUCCESS ({result['fit_seconds']:.2f}s)", flush=True)
        except Exception as exc:  # preserve a machine-readable failure for the audit
            failures += 1
            failure = {
                "fit_id": fit_id,
                "status": "FAILED",
                "rho_train": rho,
                "dataset_seed": dataset_seed,
                "width": width,
                "schedule": schedule,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            path.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
            print(f"  completed FAILED ({type(exc).__name__}: {exc})", flush=True)
    summary = {
        "prereg_frozen_sha": PHASE2_PREREG_SHA,
        "smoke": smoke,
        "expected_fits": len(cells),
        "successful_fits": completed,
        "failed_fits": failures,
        "fit_seconds_total": float(sum(runtimes)),
        "fit_seconds_mean": float(np.mean(runtimes)) if runtimes else 0.0,
        "fit_seconds_max": float(np.max(runtimes)) if runtimes else 0.0,
        "formal_expected_fits": expected_fit_count(config),
    }
    save_json(summary, output_root / "run_summary.json")
    return summary
