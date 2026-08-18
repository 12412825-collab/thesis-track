# Phase 1 — Family vs Capacity Under Distribution Shift

Phase 1 tests whether model family predicts OOD degradation after approximate controls for capacity, regularization, IID performance, shortcut strength, and paired seed. It uses the five existing families and a frozen 15-configuration registry.

## Reproduce

```powershell
.\.venv\Scripts\python.exe experiments\PHASE-1\run.py --calibrate-capacity
.\.venv\Scripts\python.exe experiments\PHASE-1\run.py --smoke
.\.venv\Scripts\python.exe experiments\PHASE-1\run.py
.\.venv\Scripts\python.exe experiments\PHASE-1\analyze.py
.\.venv\Scripts\python.exe -m pytest -q
```

If a complete run is interrupted after valid checkpoints:

```powershell
.\.venv\Scripts\python.exe experiments\PHASE-1\run.py --resume
```

## Evidence map

- Audit: `Research_Status_Audit_Phase1.md`
- Frozen methods: `Methodology_PreRegistration_Phase1_Family_Capacity.md`
- Smoke authorization: `Phase1_Smoke_Test_Report.md`
- Main results: `Phase1_Family_Capacity_Results.md`
- Reviewer-2 audit: `Post_Experiment_Audit_Phase1.md`
- Decision: `G4a_Decision_Memo.md`
- Master data: `results/phase1/master_results.csv`
- Tables, provenance, and decision JSON: `results/phase1/summaries/`
- Figures: `results/phase1/figures/`
- Immutability manifest: `results/phase1/manifest.json`

## Outcome

G4a-C: the family residual is smaller than the preregistered practical threshold and is dominated by family × shortcut-strength interaction. Phase 2 was neither designed nor run because G4a-B was not satisfied.

REVISE HYPOTHESIS
