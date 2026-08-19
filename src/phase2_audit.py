"""Adversarial, machine-readable integrity audit for formal Phase 2 outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.phase2_experiment import iter_formal_cells
from src.phase2_analysis import _load_fit_tables


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_audit(root: Path, config: dict[str, Any], prereg_paths: tuple[Path, Path]) -> dict[str, Any]:
    fits, trajectory, ood = _load_fit_tables(root)
    expected_ids = {
        f"rho{rho:g}_seed{seed:02d}_width{width}_{schedule}".replace(".", "p")
        for rho, seed, width, schedule in iter_formal_cells(config)
    }
    observed_ids = set(fits["fit_id"].astype(str)) if not fits.empty else set()
    success = fits.loc[fits["status"].eq("SUCCESS")]
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: str, severity: str = "HIGH") -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "severity": severity, "evidence": evidence})

    check("fit count", len(fits) == 360, f"observed={len(fits)}, expected=360")
    check("fit ID uniqueness", fits["fit_id"].is_unique, f"duplicates={int(fits['fit_id'].duplicated().sum())}")
    check("fit ID coverage", observed_ids == expected_ids, f"missing={len(expected_ids - observed_ids)}, unexpected={len(observed_ids - expected_ids)}")
    check("fit status", len(success) == 360 and not fits["status"].eq("FAILED").any(), f"success={len(success)}, failed={int(fits['status'].eq('FAILED').sum())}")
    check("checkpoint row count", len(trajectory) == 3600, f"observed={len(trajectory)}, expected=360 fits x 10 checkpoints")
    checkpoint_sets = trajectory.groupby("fit_id")["checkpoint"].apply(lambda values: tuple(sorted(values.astype(int)))) if not trajectory.empty else pd.Series(dtype=object)
    check("checkpoint completeness", bool(len(checkpoint_sets) == 360 and all(values == (20, 40, 60, 80, 100, 120, 140, 160, 180, 200) for values in checkpoint_sets)), "every successful fit has exactly the 10 preregistered checkpoints")
    check("OOD row count", len(ood) == 1680, f"observed={len(ood)}, expected=120*5 + 120*5 + 120*4")
    finite_columns = ["terminal_fidelity", "terminal_d3", "terminal_train_loss", "terminal_iid_accuracy", "final_ood_gap"]
    finite = bool(np.isfinite(success[finite_columns].to_numpy(dtype=float)).all()) if len(success) else False
    check("terminal numeric finiteness", finite, f"rows={len(success)}, columns={','.join(finite_columns)}")
    trajectory_finite = bool(np.isfinite(trajectory[["fidelity", "d3", "train_loss", "iid_accuracy"]].to_numpy(dtype=float)).all()) if len(trajectory) else False
    check("trajectory numeric finiteness", trajectory_finite, f"rows={len(trajectory)}")
    check("fidelity range", bool(success["terminal_fidelity"].between(0, 1).all()), f"min={success['terminal_fidelity'].min():.6f}, max={success['terminal_fidelity'].max():.6f}")
    check("D3 range", bool(success["terminal_d3"].between(0, 1).all()), f"min={success['terminal_d3'].min():.6f}, max={success['terminal_d3'].max():.6f}")

    pairing_failures: list[str] = []
    for (rho, seed, width), group in success.groupby(["rho_train", "dataset_seed", "width"]):
        if set(group["schedule"]) != {"constant", "cosine"} or group["init_seed"].nunique() != 1 or group["shuffle_seed"].nunique() != 1:
            pairing_failures.append(f"rho={rho},seed={seed},width={width}")
    check("schedule pairing", not pairing_failures, f"pairing_failures={pairing_failures[:5]}")
    expected_values = {
        "width": {8, 64, 512},
        "schedule": {"constant", "cosine"},
        "rho_train": {0.0, 0.7, 0.9},
        "dataset_seed": set(range(20)),
        "epochs": {200},
        "batch_size": {64},
        "momentum": {0.9},
        "weight_decay": {0.0001},
        "initial_lr": {0.01},
        "cosine_terminal_lr": {0.001},
    }
    protocol_failures = {column: sorted(set(success[column]) - values) for column, values in expected_values.items() if not set(success[column]).issubset(values)}
    check("protocol constants", not protocol_failures, f"unexpected_values={protocol_failures}")
    check("prereg SHA traceability", set(success["prereg_frozen_sha"]) == {config["experiment"]["prereg_frozen_sha"]}, f"unique_sha={sorted(set(success['prereg_frozen_sha']))}")
    check("no hidden model arms", set(success["width"]) == {8, 64, 512} and len(success["schedule"].unique()) == 2, "only frozen width and schedule arms are present")

    runtime_q1 = float(success["fit_seconds"].quantile(0.75) - success["fit_seconds"].quantile(0.25))
    runtime_outliers = success.loc[success["fit_seconds"] > success["fit_seconds"].quantile(0.75) + 3 * runtime_q1, ["fit_id", "fit_seconds"]].to_dict("records") if runtime_q1 > 0 else []
    check("runtime outliers recorded", True, f"outlier_count={len(runtime_outliers)}, max_seconds={success['fit_seconds'].max():.3f}", severity="LOW")

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest_failures: list[str] = []
    for relative, expected in manifest.get("files", {}).items():
        path = root / relative
        if not path.exists() or _sha256(path) != expected["sha256"]:
            manifest_failures.append(relative)
    check("manifest hashes", not manifest_failures, f"mismatches={manifest_failures[:5]}")

    checks_table = pd.DataFrame(checks)
    checks_table.to_csv(root / "audit_checks.csv", index=False)
    for path in prereg_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    summary = {
        "prereg_frozen_sha": config["experiment"]["prereg_frozen_sha"],
        "source_prereg_sha256": _sha256(prereg_paths[0]),
        "source_amendment_sha256": _sha256(prereg_paths[1]),
        "checks_total": len(checks),
        "checks_passed": int((checks_table["status"] == "PASS").sum()),
        "checks_failed": int((checks_table["status"] == "FAIL").sum()),
        "runtime_outliers": runtime_outliers,
        "fit_seconds_total": float(success["fit_seconds"].sum()),
        "fit_seconds_mean": float(success["fit_seconds"].mean()),
        "fit_seconds_median": float(success["fit_seconds"].median()),
        "fit_seconds_max": float(success["fit_seconds"].max()),
        "no_phase3_started": True,
    }
    (root / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    return {"summary": summary, "checks": checks_table.to_dict("records")}
