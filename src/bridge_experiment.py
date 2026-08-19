"""Frozen learner-only Bridge: Phase-1 MLP-L on the Phase-2 task."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn

from src.phase1_experiment import fit_phase1_model
from src.phase1_models import ModelConfiguration, load_registry
from src.phase2_experiment import Phase2Bundle, build_phase2_bundle
from src.utils import child_seed, load_config, save_json


BRIDGE_PREREG_SHA = "664d02a9f1c928713169866b2f5b3225b3554435"
PHASE2_PREREG_SHA = "654e72b39310847b0d369796c31825e379aaf909"
EXPECTED_SKLEARN = "1.9.0"
FEATURE_ORDER = ("x1", "x2", "x3", "x_spurious", "z")
RHO_VALUES = (0.9, 0.0)
DATASET_SEEDS = tuple(range(20))
REFERENCE_WIDTH = 64
REFERENCE_SCHEDULE = "constant"
BRIDGE_INIT_COMPONENT = 9100


def rho_code(rho: float) -> int:
    """Use the same rho code as the frozen Phase-2 construction."""
    return int(round((rho + 1.0) * 1000))


def _fit_id(rho: float, dataset_seed: int) -> str:
    return f"rho{rho:g}_seed{dataset_seed:02d}_mlp_l".replace(".", "p")


def _reference_fit_id(rho: float, dataset_seed: int) -> str:
    return f"rho{rho:g}_seed{dataset_seed:02d}_width64_constant".replace(".", "p")


def _feature_frame(dataset: Any) -> pd.DataFrame:
    """Return the frozen five-column feature order without changing the task."""
    return dataset.X.loc[:, list(FEATURE_ORDER)].copy()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return _sha256_bytes(data), len(data)


def _json_hash(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def dataset_content_hash(dataset: Any) -> str:
    """Hash dataframe schema, values, labels, and generator metadata."""
    values = np.ascontiguousarray(dataset.X.loc[:, list(FEATURE_ORDER)].to_numpy(dtype=np.float64))
    labels = np.ascontiguousarray(np.asarray(dataset.y, dtype=np.int64))
    payload = {
        "columns": list(FEATURE_ORDER),
        "x_sha256": _sha256_bytes(values.tobytes()),
        "y_sha256": _sha256_bytes(labels.tobytes()),
        "metadata": dataset.metadata,
    }
    return _json_hash(payload)


def task_hash_rows(bundle: Phase2Bundle) -> list[dict[str, Any]]:
    rows = [
        ("train", bundle.train),
        ("validation", bundle.validation),
        ("iid_test", bundle.iid_test),
    ]
    rows.extend((f"ood_{rho:g}", bundle.ood[rho]) for rho in sorted(bundle.ood))
    probe_payload = {
        "columns": list(FEATURE_ORDER),
        "x_sha256": _sha256_bytes(
            np.ascontiguousarray(bundle.probe_X.loc[:, list(FEATURE_ORDER)].to_numpy(dtype=np.float64)).tobytes()
        ),
        "y_sha256": _sha256_bytes(np.ascontiguousarray(np.asarray(bundle.probe_y, dtype=np.int64)).tobytes()),
    }
    output = [
        {
            "rho_train": float(bundle.rho_train),
            "dataset_seed": int(bundle.dataset_seed),
            "component": split,
            "content_sha256": dataset_content_hash(dataset),
        }
        for split, dataset in rows
    ]
    output.append(
        {
            "rho_train": float(bundle.rho_train),
            "dataset_seed": int(bundle.dataset_seed),
            "component": "probe",
            "content_sha256": _json_hash(probe_payload),
        }
    )
    return output


def load_mlp_l_entry(phase1_config_path: Path) -> ModelConfiguration:
    config = load_config(phase1_config_path)
    entries = load_registry(config)
    matches = [entry for entry in entries if entry.config_id == "MLP-L"]
    if len(matches) != 1 or matches[0].family != "mlp":
        raise RuntimeError("Bridge requires exactly one Phase-1 MLP-L registry entry")
    entry = matches[0]
    expected_keys = {"hidden_layer_sizes", "alpha", "learning_rate_init", "max_iter", "early_stopping"}
    if set(entry.parameters) != expected_keys:
        raise RuntimeError(f"MLP-L parameter extraction changed: {sorted(entry.parameters)}")
    if entry.parameters != {
        "hidden_layer_sizes": [16],
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
        "max_iter": 250,
        "early_stopping": True,
    }:
        raise RuntimeError(f"Unexpected frozen MLP-L parameters: {entry.parameters}")
    return entry


def validate_frozen_task_config(phase2_config: dict[str, Any]) -> None:
    experiment = phase2_config["experiment"]
    data = phase2_config["data"]
    if int(experiment["master_seed"]) != 20260818:
        raise RuntimeError("Phase-2 master seed does not match frozen Bridge task")
    if list(map(float, data["stable_coefficients"])) != [1.2, -1.0, 0.8]:
        raise RuntimeError("Phase-2 stable coefficients do not match frozen Bridge task")
    if float(data["label_noise"]) != 0.5:
        raise RuntimeError("Phase-2 label noise does not match frozen Bridge task")
    expected_sizes = {"n_train": 5000, "n_validation": 2000, "n_iid_test": 3000, "n_ood_test": 3000, "n_probe": 4000}
    if {key: int(data[key]) for key in expected_sizes} != expected_sizes:
        raise RuntimeError("Phase-2 sample sizes do not match frozen Bridge task")
    if list(map(float, data["rho_ood_values"])) != [0.3, 0.0, -0.3, -0.6, -0.9]:
        raise RuntimeError("Phase-2 OOD grid does not match frozen Bridge task")
    if str(experiment["prereg_frozen_sha"]) != PHASE2_PREREG_SHA:
        raise RuntimeError("Phase-2 preregistration SHA mismatch")


def _metric_row(model: Any, bundle: Phase2Bundle) -> dict[str, Any]:
    iid_X = _feature_frame(bundle.iid_test)
    iid_y = np.asarray(bundle.iid_test.y, dtype=np.int64)
    iid_prediction = np.asarray(model.predict(iid_X), dtype=np.int64)
    flipped_X = iid_X.copy()
    flipped_X.loc[:, "x_spurious"] *= -1.0
    flipped_prediction = np.asarray(model.predict(flipped_X), dtype=np.int64)
    probe_prediction = np.asarray(model.predict(bundle.probe_X.loc[:, list(FEATURE_ORDER)]), dtype=np.int64)
    iid_accuracy = float(np.mean(iid_prediction == iid_y))
    ood_rows = []
    for rho_ood, dataset in sorted(bundle.ood.items()):
        prediction = np.asarray(model.predict(_feature_frame(dataset)), dtype=np.int64)
        accuracy = float(np.mean(prediction == np.asarray(dataset.y, dtype=np.int64)))
        ood_rows.append({"rho_ood": float(rho_ood), "accuracy": accuracy, "ood_gap": iid_accuracy - accuracy})
    return {
        "terminal_fidelity": float(np.mean(probe_prediction == np.asarray(bundle.probe_y, dtype=np.int64))),
        "terminal_d3": float(np.mean(iid_prediction != flipped_prediction)),
        "terminal_iid_accuracy": iid_accuracy,
        "final_ood_gap": float(np.mean([row["ood_gap"] for row in ood_rows])),
        "ood": ood_rows,
    }


def fit_bridge_mlp_l(
    phase2_config: dict[str, Any],
    bundle: Phase2Bundle,
    entry: ModelConfiguration,
) -> dict[str, Any]:
    """Fit MLP-L on training data only and evaluate all other bundle components."""
    master = int(phase2_config["experiment"]["master_seed"])
    init_seed = child_seed(master, bundle.dataset_seed, rho_code(bundle.rho_train), BRIDGE_INIT_COMPONENT)
    X_train = _feature_frame(bundle.train)
    y_train = np.asarray(bundle.train.y, dtype=np.int64)
    started = time.perf_counter()
    model, warning_messages = fit_phase1_model(entry, X_train, y_train, seed=init_seed)
    metrics = _metric_row(model, bundle)
    estimator = model.named_steps["model"]
    scaler = model.named_steps["scale"]
    validation_scores = getattr(estimator, "validation_scores_", None)
    result = {
        "fit_id": _fit_id(bundle.rho_train, bundle.dataset_seed),
        "status": "SUCCESS",
        "bridge_prereg_frozen_sha": BRIDGE_PREREG_SHA,
        "phase2_prereg_frozen_sha": PHASE2_PREREG_SHA,
        "rho_train": float(bundle.rho_train),
        "dataset_seed": int(bundle.dataset_seed),
        "learner": "phase1_mlp_l",
        "feature_order": list(FEATURE_ORDER),
        "bridge_init_seed": int(init_seed),
        "n_train": int(len(y_train)),
        "n_iid_test": int(len(bundle.iid_test.y)),
        "n_probe": int(len(bundle.probe_y)),
        "early_stopping": bool(estimator.early_stopping),
        "max_iter": int(estimator.max_iter),
        "n_iter": int(estimator.n_iter_),
        "converged_before_max_iter": bool(int(estimator.n_iter_) < int(estimator.max_iter)),
        "loss_curve_length": int(len(getattr(estimator, "loss_curve_", []))),
        "best_validation_score": None if getattr(estimator, "best_validation_score_", None) is None else float(estimator.best_validation_score_),
        "validation_score_length": None if validation_scores is None else int(len(validation_scores)),
        "warning_messages": list(warning_messages),
        "scaler_train_mean": [float(value) for value in scaler.mean_],
        "scaler_train_scale": [float(value) for value in scaler.scale_],
        "fit_seconds": float(time.perf_counter() - started),
        **metrics,
    }
    return result


def _reference_manifest_rows(phase2_root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = phase2_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("prereg_frozen_sha") != PHASE2_PREREG_SHA:
        raise RuntimeError("Phase-2 manifest preregistration SHA mismatch")
    return manifest["files"]


def load_reference_records(phase2_root: Path, rhos: Iterable[float], seeds: Iterable[int]) -> tuple[dict[tuple[float, int], dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows = _reference_manifest_rows(phase2_root)
    records: dict[tuple[float, int], dict[str, Any]] = {}
    hash_rows: list[dict[str, Any]] = []
    for rho in rhos:
        for seed in seeds:
            fit_id = _reference_fit_id(rho, seed)
            relative = f"fits/{fit_id}.json"
            path = phase2_root / relative
            if relative not in manifest_rows or not path.exists():
                raise RuntimeError(f"Missing or unmanifested Phase-2 reference: {relative}")
            observed_sha, observed_bytes = sha256_file(path)
            expected = manifest_rows[relative]
            match = observed_sha == expected["sha256"] and observed_bytes == int(expected["bytes"])
            hash_rows.append({"path": relative, "expected_sha256": expected["sha256"], "observed_sha256": observed_sha, "expected_bytes": int(expected["bytes"]), "observed_bytes": observed_bytes, "match": match})
            if not match:
                raise RuntimeError(f"Phase-2 reference hash mismatch: {relative}")
            record = json.loads(path.read_text(encoding="utf-8"))
            expected_rho = float(rho)
            if record.get("status") != "SUCCESS" or float(record["rho_train"]) != expected_rho or int(record["dataset_seed"]) != int(seed) or int(record["width"]) != REFERENCE_WIDTH or record["schedule"] != REFERENCE_SCHEDULE:
                raise RuntimeError(f"Phase-2 reference metadata mismatch: {relative}")
            if record.get("prereg_frozen_sha") != PHASE2_PREREG_SHA:
                raise RuntimeError(f"Phase-2 reference prereg SHA mismatch: {relative}")
            records[(expected_rho, int(seed))] = record
    return records, hash_rows


def _bootstrap_summary(values: np.ndarray, *, seed: int, resamples: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize_paired(paired: pd.DataFrame, *, master_seed: int, resamples: int) -> pd.DataFrame:
    metrics = [
        ("delta_fidelity_ref_minus_mlpl", "Fidelity"),
        ("delta_d3_ref_minus_mlpl", "D3"),
        ("delta_ood_gap_ref_minus_mlpl", "OOD gap"),
    ]
    rows: list[dict[str, Any]] = []
    for rho in RHO_VALUES:
        subset = paired.loc[np.isclose(paired["rho_train"], rho)].sort_values("dataset_seed")
        for index, (column, label) in enumerate(metrics):
            values = subset[column].to_numpy(dtype=float)
            ci_low, ci_high = _bootstrap_summary(values, seed=child_seed(master_seed, rho_code(rho), 9800 + index), resamples=resamples)
            rows.append({
                "rho_train": float(rho),
                "metric": label,
                "delta_column": column,
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "sd": float(np.std(values, ddof=1)),
                "bootstrap_resamples": int(resamples),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "positive_count": int(np.sum(values > 0.0)),
                "positive_over_0p02_count": int(np.sum(values > 0.02)),
                "positive_over_0p05_count": int(np.sum(values > 0.05)),
            })
    return pd.DataFrame(rows)


def _write_csv(rows: list[dict[str, Any]] | pd.DataFrame, path: Path) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def run_bridge(
    phase2_config: dict[str, Any],
    phase1_config_path: Path,
    phase2_root: Path,
    output_root: Path,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Run exactly the 40 new MLP-L fits and analyze only the frozen matrix."""
    if sklearn.__version__ != EXPECTED_SKLEARN:
        raise RuntimeError(f"Bridge requires scikit-learn=={EXPECTED_SKLEARN}; found {sklearn.__version__}")
    validate_frozen_task_config(phase2_config)
    entry = load_mlp_l_entry(phase1_config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    fit_root = output_root / "fits"
    fit_root.mkdir(parents=True, exist_ok=True)
    reference_records, reference_hash_rows = load_reference_records(phase2_root, RHO_VALUES, DATASET_SEEDS)
    _write_csv(reference_hash_rows, output_root / "reference_hashes.csv")

    fit_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    runtimes: list[float] = []
    failures: list[dict[str, Any]] = []
    for rho in RHO_VALUES:
        for dataset_seed in DATASET_SEEDS:
            bundle = build_phase2_bundle(phase2_config, rho, dataset_seed)
            task_rows.extend(task_hash_rows(bundle))
            fit_id = _fit_id(rho, dataset_seed)
            path = fit_root / f"{fit_id}.json"
            try:
                if resume and path.exists():
                    result = json.loads(path.read_text(encoding="utf-8"))
                    if result.get("status") != "SUCCESS":
                        raise RuntimeError(f"Existing Bridge record is not successful: {fit_id}")
                else:
                    result = fit_bridge_mlp_l(phase2_config, bundle, entry)
                    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
                fit_rows.append(result)
                runtimes.append(float(result["fit_seconds"]))
                reference = reference_records[(float(rho), int(dataset_seed))]
                paired_rows.append({
                    "rho_train": float(rho),
                    "dataset_seed": int(dataset_seed),
                    "reference_fit_id": reference["fit_id"],
                    "bridge_fit_id": result["fit_id"],
                    "reference_fidelity": float(reference["terminal_fidelity"]),
                    "mlpl_fidelity": float(result["terminal_fidelity"]),
                    "delta_fidelity_ref_minus_mlpl": float(reference["terminal_fidelity"] - result["terminal_fidelity"]),
                    "delta_fidelity_mlpl_minus_ref": float(result["terminal_fidelity"] - reference["terminal_fidelity"]),
                    "reference_d3": float(reference["terminal_d3"]),
                    "mlpl_d3": float(result["terminal_d3"]),
                    "delta_d3_ref_minus_mlpl": float(reference["terminal_d3"] - result["terminal_d3"]),
                    "reference_ood_gap": float(reference["final_ood_gap"]),
                    "mlpl_ood_gap": float(result["final_ood_gap"]),
                    "delta_ood_gap_ref_minus_mlpl": float(reference["final_ood_gap"] - result["final_ood_gap"]),
                    "reference_iid_accuracy": float(reference["terminal_iid_accuracy"]),
                    "mlpl_iid_accuracy": float(result["terminal_iid_accuracy"]),
                })
            except Exception as exc:
                failure = {"fit_id": fit_id, "rho_train": float(rho), "dataset_seed": int(dataset_seed), "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}
                failures.append(failure)
                path.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
                break
        if failures:
            break

    if failures:
        summary = {"bridge_prereg_frozen_sha": BRIDGE_PREREG_SHA, "status": "FAILED", "expected_new_fits": 40, "successful_fits": len(fit_rows), "failed_fits": len(failures), "failures": failures}
        save_json(summary, output_root / "run_summary.json")
        raise RuntimeError(f"Bridge stopped after fit failure: {failures[0]}")
    if len(fit_rows) != 40 or len(paired_rows) != 40:
        raise RuntimeError(f"Bridge expected exactly 40 fits, observed {len(fit_rows)}")

    paired = pd.DataFrame(paired_rows).sort_values(["rho_train", "dataset_seed"])
    summaries = summarize_paired(paired, master_seed=int(phase2_config["experiment"]["master_seed"]), resamples=5000)
    _write_csv(fit_rows, output_root / "raw_fit_results.csv")
    paired.to_csv(output_root / "paired_results.csv", index=False)
    summaries.to_csv(output_root / "paired_statistics.csv", index=False)
    _write_csv(task_rows, output_root / "task_hashes.csv")
    summary = {
        "bridge_prereg_frozen_sha": BRIDGE_PREREG_SHA,
        "phase2_prereg_frozen_sha": PHASE2_PREREG_SHA,
        "status": "SUCCESS",
        "expected_new_fits": 40,
        "successful_fits": 40,
        "failed_fits": 0,
        "rho_train_values": list(RHO_VALUES),
        "dataset_seeds": list(DATASET_SEEDS),
        "reference_arm": {"width": REFERENCE_WIDTH, "schedule": REFERENCE_SCHEDULE, "reused": True, "hash_checks": len(reference_hash_rows)},
        "sklearn_version": sklearn.__version__,
        "feature_order": list(FEATURE_ORDER),
        "fit_seconds_total": float(sum(runtimes)),
        "fit_seconds_mean": float(np.mean(runtimes)),
        "fit_seconds_max": float(np.max(runtimes)),
        "bootstrap_resamples": 5000,
        "primary_delta_definition": "reference Fidelity minus Phase-1 MLP-L Fidelity; positive means lower MLP-L Fidelity",
    }
    save_json(summary, output_root / "run_summary.json")
    return summary


def run_bridge_smoke(phase2_config: dict[str, Any], phase1_config_path: Path, output_root: Path) -> dict[str, Any]:
    """Run a clearly separate tiny preprocessing/learner smoke, never formal statistics."""
    if sklearn.__version__ != EXPECTED_SKLEARN:
        raise RuntimeError(f"Bridge requires scikit-learn=={EXPECTED_SKLEARN}; found {sklearn.__version__}")
    entry = load_mlp_l_entry(phase1_config_path)
    config = copy.deepcopy(phase2_config)
    config["data"].update(n_train=320, n_validation=160, n_iid_test=200, n_ood_test=200, n_probe=300)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for rho in RHO_VALUES:
        bundle = build_phase2_bundle(config, rho, 0)
        result = fit_bridge_mlp_l(config, bundle, entry)
        if not np.isfinite([result["terminal_fidelity"], result["terminal_d3"], result["final_ood_gap"]]).all():
            raise RuntimeError(f"Non-finite Bridge smoke metric at rho={rho}")
        rows.append({"rho_train": rho, "fit_id": result["fit_id"], "terminal_fidelity": result["terminal_fidelity"], "terminal_d3": result["terminal_d3"], "final_ood_gap": result["final_ood_gap"], "n_iter": result["n_iter"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "smoke_results.csv", index=False)
    summary = {"bridge_prereg_frozen_sha": BRIDGE_PREREG_SHA, "status": "SMOKE_PASS", "formal_statistics": False, "fits": 2, "sklearn_version": sklearn.__version__}
    save_json(summary, output_root / "run_summary.json")
    return summary
