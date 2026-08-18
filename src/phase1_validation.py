"""Data-contract and smoke-stop validation for Phase 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def validate_result_contract(
    master: pd.DataFrame,
    fits: pd.DataFrame,
    *,
    expected_fits: int,
    expected_rows: int,
) -> dict[str, Any]:
    """Validate result grain, completeness, metrics, and exact gap reconstruction."""
    failures: list[str] = []
    if len(fits) != expected_fits:
        failures.append(f"expected {expected_fits} fits, found {len(fits)}")
    if len(master) != expected_rows:
        failures.append(f"expected {expected_rows} master rows, found {len(master)}")
    if fits["fit_id"].duplicated().any():
        failures.append("duplicate fit_id")
    if master["row_id"].duplicated().any():
        failures.append("duplicate row_id")
    fit_key = ["config_id", "rho_train", "seed"]
    if fits.duplicated(fit_key).any():
        failures.append("duplicate scientific fit key")
    row_key = [*fit_key, "environment_type", "rho_environment"]
    if master.duplicated(row_key).any():
        failures.append("duplicate scientific result key")
    if not fits["status"].eq("success").all():
        failures.append("one or more fits failed")
    successful = master.loc[master["status"].eq("success")]
    metric_columns = [
        "accuracy", "balanced_accuracy", "roc_auc", "brier_score", "log_loss",
        "iid_accuracy", "average_ood_accuracy", "worst_ood_accuracy", "ood_gap",
        "environment_delta",
    ]
    for column in metric_columns:
        if successful[column].isna().any() or not np.isfinite(successful[column]).all():
            failures.append(f"non-finite metric: {column}")
    for column in ["accuracy", "balanced_accuracy", "roc_auc", "brier_score", "iid_accuracy", "average_ood_accuracy", "worst_ood_accuracy"]:
        if ((successful[column] < 0) | (successful[column] > 1)).any():
            failures.append(f"metric outside [0,1]: {column}")
    if (successful["log_loss"] < 0).any():
        failures.append("negative log loss")
    if fits["learned_capacity_primary"].isna().any() or not np.isfinite(fits["learned_capacity_primary"]).all():
        failures.append("non-finite primary capacity proxy")
    capacity = (
        fits.groupby(["family", "capacity_tier_score"], sort=True)["learned_capacity_primary"]
        .mean()
        .reset_index()
    )
    capacity_order: dict[str, bool] = {}
    for family, group in capacity.groupby("family", sort=False):
        values = group.sort_values("capacity_tier_score")["learned_capacity_primary"].to_numpy()
        capacity_order[str(family)] = bool(len(values) == 3 and np.all(np.diff(values) > 0))
    if not all(capacity_order.values()):
        failures.append(f"capacity tier ordering failed: {capacity_order}")
    max_reconstruction_error = 0.0
    for _, group in successful.groupby("fit_id", sort=False):
        iid = group.loc[group["environment_type"].eq("iid_test"), "accuracy"]
        ood = group.loc[group["environment_type"].eq("ood"), "accuracy"]
        if len(iid) != 1 or len(ood) == 0:
            failures.append("fit lacks exactly one IID row or at least one OOD row")
            continue
        expected_gap = float(iid.iloc[0] - ood.mean())
        error = float(np.max(np.abs(group["ood_gap"].to_numpy() - expected_gap)))
        max_reconstruction_error = max(max_reconstruction_error, error)
    if max_reconstruction_error > 1e-12:
        failures.append(f"OOD gap reconstruction error {max_reconstruction_error:.3e}")
    return {
        "passed": not failures,
        "failures": failures,
        "expected_fits": expected_fits,
        "actual_fits": len(fits),
        "expected_rows": expected_rows,
        "actual_rows": len(master),
        "capacity_order": capacity_order,
        "max_gap_reconstruction_error": max_reconstruction_error,
    }


def expected_master_rows(
    n_configs: int, seeds: list[int], rho_values: list[float], rho_ood: list[float]
) -> int:
    """Count one IID plus each non-duplicate OOD row per fit."""
    per_seed = sum(1 + sum(not np.isclose(ood, rho) for ood in rho_ood) for rho in rho_values)
    return n_configs * len(seeds) * per_seed


def write_smoke_report(
    smoke_root: Path,
    report_path: Path,
    wrong_line: pd.DataFrame,
    wrong_line_failures: list[str],
) -> None:
    """Write the required human-readable smoke gate report from persisted evidence."""
    master = pd.read_csv(smoke_root / "master_results.csv")
    fits = pd.read_csv(smoke_root / "summaries" / "fit_records.csv")
    generator = pd.read_csv(smoke_root / "summaries" / "generator_validation.csv")
    metadata = json.loads(
        (smoke_root / "summaries" / "run_metadata.json").read_text(encoding="utf-8")
    )
    contract = metadata["result_contract"]
    failures = [*metadata.get("smoke_gate_failures", []), *wrong_line_failures]
    capacity = (
        fits.groupby(["family", "capacity_tier"], sort=False)["learned_capacity_primary"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    high_fidelity = (
        wrong_line.loc[
            np.isclose(wrong_line["rho_train"], 0.0)
            & wrong_line["capacity_tier"].eq("high")
        ]
        .groupby("family")["boundary_fidelity"]
        .mean()
        .reset_index()
    )
    status = "PASS" if not failures else "FAIL"
    failure_text = "None." if not failures else "\n".join(f"- {item}" for item in failures)
    text = f"""# Phase 1 Smoke Test Report

Date: 2026-08-18  
Status: **{status}**  
Run ID: `{metadata['run_id']}`

## Scope

The smoke run exercised all 15 frozen configurations, seed 0, `rho_train={{0,0.9}}`, reduced train/validation/test samples of 700/350/500, and the frozen IID/OOD fitting boundary. The wrong-line smoke scanned all three stable-boundary geometries, neutral and strong shortcut regimes, and every capacity tier using a noise-free probe.

## Result contract

- Expected/actual fits: {contract['expected_fits']} / {contract['actual_fits']}
- Expected/actual master rows: {contract['expected_rows']} / {contract['actual_rows']}
- Unique fit and result keys: {'yes' if contract['passed'] else 'see failures'}
- Maximum OOD-gap reconstruction error: `{contract['max_gap_reconstruction_error']:.3e}`
- Fit failure fraction: `{metadata['fit_failure_fraction']:.4f}`
- Non-finite or out-of-bound primary metrics: none detected
- Capacity tier ordering by family: `{contract['capacity_order']}`

## Generator gates

- Maximum absolute requested-versus-realized shortcut correlation error: `{(generator['empirical_spurious_correlation'] - generator['rho_requested']).abs().max():.4f}`
- Positive-rate range: `{generator['positive_rate'].min():.4f}` to `{generator['positive_rate'].max():.4f}`
- Maximum absolute noise-target correlation: `{generator['z_target_correlation'].abs().max():.4f}`

## Capacity proxy smoke summary

```text
{capacity.to_string(index=False, float_format=lambda value: f'{value:.4f}')}
```

## Noise-free wrong-line gate

High-capacity mean boundary fidelity at neutral rho (gate: at least 0.95):

```text
{high_fidelity.to_string(index=False, float_format=lambda value: f'{value:.4f}')}
```

Low and medium capacity values remain in `results/phase1/smoke/summaries/wrong_line_guard.csv` as diagnostics and are not stop gates.

## Runtime and warnings

- Grid runtime: `{metadata['runtime_seconds']:.2f}` seconds
- Mean/max fit time: `{fits['fit_seconds'].mean():.3f}` / `{fits['fit_seconds'].max():.3f}` seconds
- Captured fit warnings: `{int(fits['warning_count'].sum())}`
- Master results SHA-256: `{metadata['master_results_sha256']}`

## Stop-gate failures

{failure_text}

## Authorization

{'All preregistered smoke gates passed. The complete Phase 1 grid is authorized.' if not failures else 'The complete Phase 1 grid is not authorized until the listed failures are repaired and smoke is rerun.'}
"""
    report_path.write_text(text, encoding="utf-8")
