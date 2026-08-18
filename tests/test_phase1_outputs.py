"""Contract checks for the completed Phase 1 evidence package."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "phase1"
SUMMARIES = RESULTS / "summaries"


def test_complete_master_and_fit_ledgers_have_frozen_grain() -> None:
    master = pd.read_csv(RESULTS / "master_results.csv")
    fits = pd.read_csv(SUMMARIES / "fit_records.csv")
    assert master.shape == (3450, 46)
    assert fits.shape == (600, 39)
    assert not master["row_id"].duplicated().any()
    assert not fits["fit_id"].duplicated().any()
    assert master["status"].eq("success").all()
    assert fits["status"].eq("success").all()


def test_preregistered_analysis_and_decision_artifacts_exist() -> None:
    expected = {
        "variance_decomposition.csv": 7,
        "regression_models.csv": 4,
        "regression_bootstrap.csv": 2,
        "regression_sensitivities.csv": 8,
        "capacity_matching.csv": 120,
        "robustness_envelope.csv": 20,
        "ranking_reversal.csv": 4,
        "wrong_line_guard.csv": 405,
    }
    for filename, rows in expected.items():
        table = pd.read_csv(SUMMARIES / filename)
        assert len(table) == rows
        assert not table.duplicated().any()
    decision = json.loads((SUMMARIES / "g4a_decision.json").read_text(encoding="utf-8"))
    assert decision["branch"] == "G4a-C"
    assert decision["decision"] == "REVISE HYPOTHESIS"


def test_all_five_publication_figures_have_png_and_pdf() -> None:
    for index in range(1, 6):
        assert len(list((RESULTS / "figures").glob(f"{index:02d}_*.png"))) == 1
        assert len(list((RESULTS / "figures").glob(f"{index:02d}_*.pdf"))) == 1


def test_required_reports_and_phase2_gate() -> None:
    reports = [
        "Research_Status_Audit_Phase1.md",
        "Methodology_PreRegistration_Phase1_Family_Capacity.md",
        "Phase1_Smoke_Test_Report.md",
        "Phase1_Family_Capacity_Results.md",
        "Post_Experiment_Audit_Phase1.md",
        "G4a_Decision_Memo.md",
    ]
    for filename in reports:
        assert (ROOT / filename).exists()
    assert (ROOT / "Phase1_Family_Capacity_Results.md").read_text(
        encoding="utf-8"
    ).strip().endswith("REVISE HYPOTHESIS")
    assert not (ROOT / "Phase2_Design_Learned_Bias_Isolation.md").exists()

