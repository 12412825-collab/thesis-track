"""Noise-free stable-boundary fidelity guard for Phase 1."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.phase1_experiment import fit_phase1_model
from src.phase1_models import load_registry
from src.utils import child_seed
from src.verification_data import generate_verification_dataset


def _normalized(coefficients: list[float], target_norm: float) -> list[float]:
    values = np.asarray(coefficients, dtype=float)
    return (values / np.linalg.norm(values) * target_norm).tolist()


def run_wrong_line_guard(
    config: dict[str, Any],
    output_path: Path,
    *,
    smoke: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Scan geometry × rho × capacity using only a noise-free oracle probe."""
    settings = config["wrong_line_guard"]
    entries = load_registry(config)
    master = int(config["experiment"]["master_seed"])
    features = list(config["data"]["features"])
    seeds = [0] if smoke else [int(x) for x in settings["seeds"]]
    rho_values = [0.0, 0.9] if smoke else [float(x) for x in settings["rho_train_values"]]
    n_train = min(700, int(settings["n_train"])) if smoke else int(settings["n_train"])
    n_probe = min(1200, int(settings["n_probe"])) if smoke else int(settings["n_probe"])
    target_norm = float(np.linalg.norm(config["data"]["stable_coefficients"]))
    rows: list[dict[str, Any]] = []
    for geometry_index, (geometry, raw_coefficients) in enumerate(settings["geometries"].items()):
        coefficients = _normalized(raw_coefficients, target_norm)
        for seed in seeds:
            probe = generate_verification_dataset(
                n_samples=n_probe,
                rho=0.0,
                base_seed=child_seed(master, seed, geometry_index, 8100),
                noise_seed=child_seed(master, seed, geometry_index, 8100, 999),
                stable_coefficients=coefficients,
                label_noise=0.0,
                split="wrong_line_probe",
            )
            probe.X.loc[:, "x_spurious"] = 0.0
            probe.X.loc[:, "z"] = 0.0
            for rho_train in rho_values:
                train = generate_verification_dataset(
                    n_samples=n_train,
                    rho=rho_train,
                    base_seed=child_seed(master, seed, geometry_index, 8000),
                    noise_seed=child_seed(master, seed, geometry_index, 8000, 999),
                    stable_coefficients=coefficients,
                    label_noise=0.0,
                    split="wrong_line_train",
                )
                for entry_index, entry in enumerate(entries):
                    started = time.perf_counter()
                    model, captured = fit_phase1_model(
                        entry,
                        train.X[features],
                        train.y,
                        child_seed(master, seed, geometry_index, entry_index, 8200),
                    )
                    prediction = np.asarray(model.predict(probe.X[features]))
                    rows.append(
                        {
                            "geometry": geometry,
                            "stable_coefficients": str(coefficients),
                            "rho_train": rho_train,
                            "seed": seed,
                            "config_id": entry.config_id,
                            "config_hash": entry.config_hash,
                            "family": entry.family,
                            "capacity_tier": entry.capacity_tier,
                            "capacity_tier_score": entry.capacity_tier_score,
                            "boundary_fidelity": float(np.mean(prediction == probe.y)),
                            "fit_seconds": time.perf_counter() - started,
                            "warning_count": len(captured),
                            "warning_messages": " | ".join(captured),
                        }
                    )
    table = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    neutral_high = table.loc[
        np.isclose(table["rho_train"], 0.0) & table["capacity_tier"].eq("high")
    ]
    means = neutral_high.groupby("family")["boundary_fidelity"].mean()
    threshold = float(settings["neutral_rho_min_boundary_fidelity"])
    failures = [
        f"wrong-line high-tier neutral fidelity for {family} is {value:.4f} < {threshold:.4f}"
        for family, value in means.items()
        if value < threshold
    ]
    if set(means.index) != set(table["family"].unique()):
        failures.append("wrong-line guard is missing a high-capacity family")
    return table, failures

