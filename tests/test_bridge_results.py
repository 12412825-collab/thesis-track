"""Result-contract and frozen decision checks for the completed Bridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("results/bridge_p1mlp_l")


def test_bridge_formal_result_contract_and_reference_hashes() -> None:
    summary = json.loads((ROOT / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "SUCCESS"
    assert summary["expected_new_fits"] == 40
    assert summary["successful_fits"] == 40
    assert summary["failed_fits"] == 0
    assert summary["sklearn_version"] == "1.9.0"

    fit_files = sorted((ROOT / "fits").glob("*.json"))
    assert len(fit_files) == 40
    raw = pd.read_csv(ROOT / "raw_fit_results.csv")
    paired = pd.read_csv(ROOT / "paired_results.csv")
    assert len(raw) == len(paired) == 40
    assert raw["status"].eq("SUCCESS").all()
    assert raw[["terminal_fidelity", "terminal_d3", "final_ood_gap"]].apply(np.isfinite).all().all()
    assert paired[["rho_train", "dataset_seed"]].drop_duplicates().shape[0] == 40

    reference_hashes = pd.read_csv(ROOT / "reference_hashes.csv")
    assert len(reference_hashes) == 40
    assert reference_hashes["match"].astype(str).str.lower().eq("true").all()

    task_hashes = pd.read_csv(ROOT / "task_hashes.csv")
    assert len(task_hashes) == 340
    assert task_hashes[["rho_train", "dataset_seed", "component"]].drop_duplicates().shape[0] == 340


def test_bridge_paired_metrics_match_raw_records() -> None:
    raw = pd.read_csv(ROOT / "raw_fit_results.csv").set_index(["rho_train", "dataset_seed"])
    paired = pd.read_csv(ROOT / "paired_results.csv")
    for row in paired.itertuples(index=False):
        fit = raw.loc[(row.rho_train, row.dataset_seed)]
        assert np.isclose(fit["terminal_fidelity"], row.mlpl_fidelity)
        assert np.isclose(fit["terminal_d3"], row.mlpl_d3)
        assert np.isclose(fit["final_ood_gap"], row.mlpl_ood_gap)


def test_frozen_decision_is_case_b_and_not_case_a_or_c() -> None:
    stats = pd.read_csv(ROOT / "paired_statistics.csv")
    raw = pd.read_csv(ROOT / "raw_fit_results.csv")
    r9 = stats[(np.isclose(stats.rho_train, 0.9)) & stats.metric.eq("Fidelity")].iloc[0]
    r0 = stats[(np.isclose(stats.rho_train, 0.0)) & stats.metric.eq("Fidelity")].iloc[0]
    r9_raw = raw[np.isclose(raw.rho_train, 0.9)]

    assert np.isclose(r9["mean"], 0.00655)
    assert r9["bootstrap_ci95_low"] < 0 < r9["bootstrap_ci95_high"]
    assert r9["positive_over_0p05_count"] == 1
    assert abs(r9["mean"]) < 0.02
    assert r9_raw["terminal_d3"].mean() >= 0.5
    assert r9_raw["final_ood_gap"].mean() >= 0.30
    assert not (r9["mean"] >= 0.05 and r9["bootstrap_ci95_low"] > 0 and r9["positive_over_0p02_count"] >= 15)
    assert not (r0["mean"] >= 0.05 and r0["bootstrap_ci95_low"] > 0)
