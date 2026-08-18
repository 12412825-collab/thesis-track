"""Execution engine and result contract for Phase 1."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.phase1_models import (
    ModelConfiguration,
    add_empirical_capacity_index,
    build_phase1_model,
    canonical_json,
    extract_capacity_metrics,
    load_registry,
    stable_hash,
)
from src.phase1_validation import expected_master_rows, validate_result_contract
from src.utils import child_seed, save_json, set_global_seed
from src.verification_data import VerificationDataset, generate_verification_dataset


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Phase1Bundle:
    """Paired datasets for one rho/seed cell."""

    rho_train: float
    seed: int
    train: VerificationDataset
    validation: VerificationDataset
    iid_test: VerificationDataset
    ood: dict[float, VerificationDataset]


def _rho_code(rho: float) -> int:
    return int(round((rho + 1.0) * 1000))


def _dataset(
    config: dict[str, Any],
    *,
    n_samples: int,
    rho: float,
    base_seed: int,
    noise_seed: int,
    split: str,
    stable_coefficients: list[float] | None = None,
    label_noise: float | None = None,
) -> VerificationDataset:
    data = config["data"]
    return generate_verification_dataset(
        n_samples=n_samples,
        rho=rho,
        base_seed=base_seed,
        noise_seed=noise_seed,
        stable_coefficients=stable_coefficients or data["stable_coefficients"],
        label_noise=float(data["label_noise"] if label_noise is None else label_noise),
        split=split,
    )


def build_phase1_bundle(
    config: dict[str, Any],
    rho_train: float,
    seed: int,
    *,
    n_train: int | None = None,
    n_val: int | None = None,
    n_test: int | None = None,
    rho_ood_values: list[float] | None = None,
) -> Phase1Bundle:
    """Build datasets paired across rho regimes and shared by all configurations."""
    data = config["data"]
    master = int(config["experiment"]["master_seed"])
    n_train = int(n_train or data["n_train"])
    n_val = int(n_val or data["n_val"])
    n_test = int(n_test or data["n_test"])
    common = data["rho_ood_values"] if rho_ood_values is None else rho_ood_values
    train = _dataset(
        config,
        n_samples=n_train,
        rho=rho_train,
        base_seed=child_seed(master, seed, 1),
        noise_seed=child_seed(master, seed, 1, 999),
        split="train",
    )
    validation = _dataset(
        config,
        n_samples=n_val,
        rho=rho_train,
        base_seed=child_seed(master, seed, 2),
        noise_seed=child_seed(master, seed, 2, 999),
        split="iid_validation",
    )
    iid_test = _dataset(
        config,
        n_samples=n_test,
        rho=rho_train,
        base_seed=child_seed(master, seed, 100),
        noise_seed=child_seed(master, seed, 100, 999),
        split="iid_test",
    )
    ood: dict[float, VerificationDataset] = {}
    for value in common:
        rho_ood = float(value)
        if np.isclose(rho_ood, rho_train):
            continue
        ood[rho_ood] = _dataset(
            config,
            n_samples=n_test,
            rho=rho_ood,
            base_seed=child_seed(master, seed, 100, _rho_code(rho_ood)),
            noise_seed=child_seed(master, seed, 100, _rho_code(rho_ood), 999),
            split=f"ood_{rho_ood}",
        )
    return Phase1Bundle(rho_train, seed, train, validation, iid_test, ood)


def fit_phase1_model(
    entry: ModelConfiguration,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> tuple[Any, list[str]]:
    """Fit one frozen configuration; deliberately accepts no validation/OOD input."""
    model = build_phase1_model(entry, seed)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)
    messages = [str(item.message) for item in captured]
    return model, messages


def classifier_metrics(model: Any, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    """Return the common Phase 1 outcome metrics."""
    prediction = np.asarray(model.predict(X))
    probability = np.asarray(model.predict_proba(X))[:, 1]
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }


def _generator_rows(bundle: Phase1Bundle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    datasets = [("train", "train", bundle.train), ("iid_validation", "iid", bundle.validation), ("iid_test", "iid", bundle.iid_test)]
    datasets.extend((f"ood_{rho}", "ood", data) for rho, data in bundle.ood.items())
    for split, environment_type, dataset in datasets:
        rows.append(
            {
                **dataset.metadata,
                "rho_train": bundle.rho_train,
                "experiment_seed": bundle.seed,
                "split": split,
                "environment_type": environment_type,
            }
        )
    return rows


def _git_value(project_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_run_metadata(
    config: dict[str, Any], project_root: Path, mode: str, started_at: str
) -> dict[str, Any]:
    """Capture enough provenance to distinguish software and configuration runs."""
    config_path = project_root / "experiments" / "PHASE-1" / "config.yaml"
    source_paths = sorted((project_root / "src").glob("phase1_*.py"))
    source_manifest = {str(path.relative_to(project_root)): _file_sha256(path) for path in source_paths}
    config_hash = stable_hash(config)
    return {
        "experiment_id": config["experiment"]["id"],
        "mode": mode,
        "run_id": f"{config['experiment']['id']}-{mode}-{config_hash[:12]}",
        "started_at_utc": started_at,
        "git_commit": _git_value(project_root, "rev-parse", "HEAD"),
        "git_branch": _git_value(project_root, "branch", "--show-current"),
        "git_status_porcelain": _git_value(project_root, "status", "--porcelain"),
        "config_sha256": _file_sha256(config_path),
        "config_canonical_hash": config_hash,
        "source_manifest": source_manifest,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "pyyaml": yaml.__version__,
        },
    }


def _output_paths(output_root: Path) -> dict[str, Path]:
    paths = {
        "root": output_root,
        "summaries": output_root / "summaries",
        "figures": output_root / "figures",
        "logs": output_root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _fit_id(experiment_id: str, rho_train: float, seed: int, entry: ModelConfiguration) -> str:
    return stable_hash(
        {
            "experiment_id": experiment_id,
            "rho_train": rho_train,
            "seed": seed,
            "config_hash": entry.config_hash,
        }
    )


def _base_fit_record(
    metadata: dict[str, Any], entry: ModelConfiguration, rho_train: float, seed: int
) -> dict[str, Any]:
    fit_id = _fit_id(metadata["experiment_id"], rho_train, seed, entry)
    return {
        "experiment_id": metadata["experiment_id"],
        "run_id": metadata["run_id"],
        "fit_id": fit_id,
        "config_id": entry.config_id,
        "config_hash": entry.config_hash,
        "family": entry.family,
        "capacity_tier": entry.capacity_tier,
        "capacity_tier_score": entry.capacity_tier_score,
        "regularization_tier": entry.regularization_tier,
        "regularization_flexibility_score": entry.regularization_flexibility_score,
        "hyperparameters_json": canonical_json(entry.parameters),
        "rho_train": rho_train,
        "seed": seed,
    }


def run_capacity_calibration(
    config: dict[str, Any], output_root: Path, project_root: Path
) -> pd.DataFrame:
    """Fit training-only design cells and export no performance outcomes."""
    paths = _output_paths(output_root)
    entries = load_registry(config)
    settings = config["capacity"]
    rows: list[dict[str, Any]] = []
    for rho_value in settings["calibration_rho_values"]:
        rho_train = float(rho_value)
        for seed_value in settings["calibration_seeds"]:
            seed = int(seed_value)
            bundle = build_phase1_bundle(
                config,
                rho_train,
                seed,
                n_train=int(settings["calibration_n_train"]),
                n_val=100,
                n_test=100,
                rho_ood_values=[],
            )
            features = config["data"]["features"]
            for index, entry in enumerate(entries):
                LOGGER.info("Capacity calibration rho=%s seed=%s config=%s", rho_train, seed, entry.config_id)
                started = time.perf_counter()
                model, captured = fit_phase1_model(
                    entry,
                    bundle.train.X[features],
                    bundle.train.y,
                    child_seed(config["experiment"]["master_seed"], seed, index, 7000),
                )
                capacity = extract_capacity_metrics(
                    entry,
                    model,
                    bundle.train.X[features],
                    kernel_rank_max_samples=int(settings["kernel_rank_max_samples"]),
                )
                rows.append(
                    {
                        "rho_train": rho_train,
                        "seed": seed,
                        "config_id": entry.config_id,
                        "config_hash": entry.config_hash,
                        "family": entry.family,
                        "capacity_tier": entry.capacity_tier,
                        "capacity_tier_score": entry.capacity_tier_score,
                        "regularization_tier": entry.regularization_tier,
                        "regularization_flexibility_score": entry.regularization_flexibility_score,
                        "fit_seconds": time.perf_counter() - started,
                        "warning_count": len(captured),
                        **capacity,
                    }
                )
    table = add_empirical_capacity_index(pd.DataFrame(rows))
    table.to_csv(paths["root"] / "capacity_calibration.csv", index=False)
    summary = (
        table.groupby(["family", "config_id", "capacity_tier_score"], sort=False)
        .agg(
            learned_capacity_mean=("learned_capacity_primary", "mean"),
            learned_capacity_min=("learned_capacity_primary", "min"),
            learned_capacity_max=("learned_capacity_primary", "max"),
            fit_seconds_mean=("fit_seconds", "mean"),
            warning_count=("warning_count", "sum"),
        )
        .reset_index()
    )
    summary["tier_order_matches_proxy"] = summary.groupby("family")[
        "learned_capacity_mean"
    ].transform(lambda x: bool(np.all(np.diff(x.to_numpy()) > 0)))
    summary.to_csv(paths["root"] / "capacity_calibration_summary.csv", index=False)
    metadata = build_run_metadata(
        config, project_root, "capacity-calibration", datetime.now(timezone.utc).isoformat()
    )
    metadata["contains_outcome_performance"] = False
    save_json(metadata, paths["root"] / "run_metadata.json")
    return table


def _validate_smoke_generator(rows: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    settings = config["smoke"]
    failures: list[str] = []
    rho_error = (rows["empirical_spurious_correlation"] - rows["rho_requested"]).abs().max()
    if rho_error > float(settings["max_abs_rho_error"]):
        failures.append(f"maximum realized rho error {rho_error:.4f} exceeds gate")
    if rows["positive_rate"].min() < float(settings["min_positive_rate"]):
        failures.append("positive rate below smoke gate")
    if rows["positive_rate"].max() > float(settings["max_positive_rate"]):
        failures.append("positive rate above smoke gate")
    if rows["z_target_correlation"].abs().max() > float(settings["max_abs_noise_target_correlation"]):
        failures.append("noise-target correlation exceeds smoke gate")
    return failures


def run_phase1_grid(
    config: dict[str, Any],
    output_root: Path,
    project_root: Path,
    *,
    smoke: bool = False,
) -> dict[str, Path]:
    """Run the frozen family/config/rho/seed grid with resumable checkpoints."""
    paths = _output_paths(output_root)
    mode = "smoke" if smoke else "complete"
    started_at = datetime.now(timezone.utc).isoformat()
    metadata = build_run_metadata(config, project_root, mode, started_at)
    save_json(metadata, paths["summaries"] / "run_metadata.started.json")
    entries = load_registry(config)
    data = config["data"]
    if smoke:
        settings = config["smoke"]
        seeds = [int(x) for x in settings["seeds"]]
        rho_values = [float(x) for x in settings["rho_train_values"]]
        rho_ood = [float(x) for x in settings["rho_ood_values"]]
        n_train, n_val, n_test = (int(settings[x]) for x in ["n_train", "n_val", "n_test"])
    else:
        seeds = [int(x) for x in config["seeds"]]
        rho_values = [float(x) for x in data["rho_train_values"]]
        rho_ood = [float(x) for x in data["rho_ood_values"]]
        n_train, n_val, n_test = (int(data[x]) for x in ["n_train", "n_val", "n_test"])

    fit_checkpoint = paths["root"] / "fit_records.partial.csv"
    result_checkpoint = paths["root"] / "master_results.partial.csv"
    generator_checkpoint = paths["summaries"] / "generator_validation.partial.csv"
    fit_rows = pd.read_csv(fit_checkpoint).to_dict("records") if fit_checkpoint.exists() else []
    result_rows = pd.read_csv(result_checkpoint).to_dict("records") if result_checkpoint.exists() else []
    generator_rows = pd.read_csv(generator_checkpoint).to_dict("records") if generator_checkpoint.exists() else []
    completed = {str(row["fit_id"]) for row in fit_rows if row.get("status") == "success"}
    features = list(data["features"])
    run_started = time.perf_counter()

    for rho_train in rho_values:
        for seed in seeds:
            set_global_seed(child_seed(int(config["experiment"]["master_seed"]), seed))
            bundle = build_phase1_bundle(
                config,
                rho_train,
                seed,
                n_train=n_train,
                n_val=n_val,
                n_test=n_test,
                rho_ood_values=rho_ood,
            )
            if not any(
                np.isclose(float(row["rho_train"]), rho_train)
                and int(row["experiment_seed"]) == seed
                for row in generator_rows
            ):
                generator_rows.extend(_generator_rows(bundle))
            for index, entry in enumerate(entries):
                base = _base_fit_record(metadata, entry, rho_train, seed)
                if base["fit_id"] in completed:
                    continue
                LOGGER.info("Phase1 %s rho=%s seed=%s config=%s", mode, rho_train, seed, entry.config_id)
                fit_started = time.perf_counter()
                try:
                    model, captured = fit_phase1_model(
                        entry,
                        bundle.train.X[features],
                        bundle.train.y,
                        child_seed(int(config["experiment"]["master_seed"]), seed, index, 9000),
                    )
                    fit_seconds = time.perf_counter() - fit_started
                    capacity = extract_capacity_metrics(
                        entry,
                        model,
                        bundle.train.X[features],
                        kernel_rank_max_samples=int(config["capacity"]["kernel_rank_max_samples"]),
                    )
                    validation_metrics = classifier_metrics(
                        model, bundle.validation.X[features], bundle.validation.y
                    )
                    fit_record = {
                        **base,
                        "status": "success",
                        "failure_type": "",
                        "failure_message": "",
                        "fit_seconds": fit_seconds,
                        "warning_count": len(captured),
                        "warning_messages": " | ".join(captured),
                        **{f"validation_{k}": v for k, v in validation_metrics.items()},
                        **capacity,
                    }
                    environments = [("iid_test", rho_train, bundle.iid_test)]
                    environments.extend(("ood", rho, dataset) for rho, dataset in bundle.ood.items())
                    evaluated: list[dict[str, Any]] = []
                    for environment_type, rho_environment, dataset in environments:
                        eval_started = time.perf_counter()
                        metrics = classifier_metrics(model, dataset.X[features], dataset.y)
                        evaluated.append(
                            {
                                **base,
                                "row_id": stable_hash(
                                    {
                                        "fit_id": base["fit_id"],
                                        "environment_type": environment_type,
                                        "rho_environment": rho_environment,
                                    }
                                ),
                                "status": "success",
                                "failure_type": "",
                                "failure_message": "",
                                "environment_type": environment_type,
                                "rho_environment": rho_environment,
                                "evaluation_seconds": time.perf_counter() - eval_started,
                                **metrics,
                            }
                        )
                    iid_accuracy = next(x["accuracy"] for x in evaluated if x["environment_type"] == "iid_test")
                    ood_accuracies = [x["accuracy"] for x in evaluated if x["environment_type"] == "ood"]
                    for row in evaluated:
                        row["iid_accuracy"] = iid_accuracy
                        row["average_ood_accuracy"] = float(np.mean(ood_accuracies))
                        row["worst_ood_accuracy"] = float(np.min(ood_accuracies))
                        row["ood_gap"] = iid_accuracy - float(np.mean(ood_accuracies))
                        row["environment_delta"] = iid_accuracy - row["accuracy"]
                    result_rows.extend(evaluated)
                except Exception as error:  # failure is data, not silent omission
                    fit_record = {
                        **base,
                        "status": "failed",
                        "failure_type": type(error).__name__,
                        "failure_message": str(error),
                        "failure_traceback": traceback.format_exc(),
                        "fit_seconds": time.perf_counter() - fit_started,
                    }
                    result_rows.append(
                        {
                            **base,
                            "row_id": stable_hash({"fit_id": base["fit_id"], "failure": True}),
                            "status": "failed",
                            "failure_type": type(error).__name__,
                            "failure_message": str(error),
                            "environment_type": "fit_failure",
                            "rho_environment": np.nan,
                        }
                    )
                    LOGGER.exception("Fit failed: %s", base["fit_id"])
                fit_rows.append(fit_record)
                pd.DataFrame(fit_rows).to_csv(fit_checkpoint, index=False)
                pd.DataFrame(result_rows).to_csv(result_checkpoint, index=False)
                pd.DataFrame(generator_rows).to_csv(generator_checkpoint, index=False)

    fits = add_empirical_capacity_index(pd.DataFrame(fit_rows))
    results = pd.DataFrame(result_rows)
    capacity_columns = [
        "fit_id", "capacity_proxy_name", "learned_capacity_primary",
        "empirical_capacity_index", "structural_parameter_count",
        "effective_degrees_of_freedom", "kernel_effective_rank",
        "support_vector_fraction", "tree_total_leaves", "tree_mean_depth",
        "boosting_total_leaves", "boosting_iterations", "mlp_parameter_count",
        "optimization_iterations", "converged", "fit_seconds", "warning_count",
    ]
    available = [column for column in capacity_columns if column in fits]
    results = results.drop(columns=[column for column in available if column != "fit_id" and column in results], errors="ignore")
    results = results.merge(fits[available], on="fit_id", how="left", validate="many_to_one")
    fits.to_csv(paths["summaries"] / "fit_records.csv", index=False)
    results.to_csv(paths["root"] / "master_results.csv", index=False)
    generator = pd.DataFrame(generator_rows)
    generator.to_csv(paths["summaries"] / "generator_validation.csv", index=False)

    failure_fraction = float((fits["status"] != "success").mean())
    smoke_failures = _validate_smoke_generator(generator, config) if smoke else []
    contract = validate_result_contract(
        results,
        fits,
        expected_fits=len(entries) * len(seeds) * len(rho_values),
        expected_rows=expected_master_rows(len(entries), seeds, rho_values, rho_ood),
    )
    smoke_failures.extend(contract["failures"] if smoke else [])
    if smoke and failure_fraction > float(config["smoke"]["max_fit_failure_fraction"]):
        smoke_failures.append(f"fit failure fraction {failure_fraction:.4f} exceeds gate")
    metadata.update(
        {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": time.perf_counter() - run_started,
            "n_fit_records": len(fits),
            "n_master_rows": len(results),
            "fit_failure_fraction": failure_fraction,
            "result_contract": contract,
            "smoke_gate_failures": smoke_failures,
            "master_results_sha256": _file_sha256(paths["root"] / "master_results.csv"),
        }
    )
    save_json(metadata, paths["summaries"] / "run_metadata.json")
    if not smoke_failures:
        fit_checkpoint.unlink(missing_ok=True)
        result_checkpoint.unlink(missing_ok=True)
        generator_checkpoint.unlink(missing_ok=True)
    if smoke_failures:
        raise RuntimeError("Smoke stop gate failed: " + "; ".join(smoke_failures))
    return {key: value for key, value in paths.items() if key != "root"} | {"root": paths["root"]}
