# Phase 2 — Training Regime and Boundary Distortion

## Frozen protocol

- Preregistration commit: `654e72b39310847b0d369796c31825e379aaf909`
- Preregistration SHA-256: `4778332AEB630E682E67E61AF1AE2DA51B83F14DE7A7ECE9C025607ADE26082A`
- Amendment SHA-256: `C94DD303D28C7AE32C247D2316338997C3F5ABB15760A450C22C23B44275BBF4`
- Formal matrix: 20 seeds × `{0.7, 0.9}` headline rho × 3 widths × 2 schedules, plus 20 seeds × rho=0 sanity × 3 widths × 2 schedules = 360 fits
- Model: one-hidden-layer ReLU MLP only; width `{8, 64, 512}`
- Optimizer: SGD, momentum `0.9`, initial LR `0.01`, constant vs cosine to `0.001`
- Weight decay: `1e-4`; batch size `64`; 200 epochs; no early stopping, dropout, augmentation, Adam, or extra arms
- Primary endpoint: `Fidelity(200)` on the fixed 4,000-point neutral stable-oracle probe
- Secondary/process measures: OOD gap, fidelity trajectory, D3 trajectory, SAT, train loss, IID accuracy

## Execution

```powershell
.venv\Scripts\python.exe experiments\PHASE-2\run.py --output-root results\phase2
.venv\Scripts\python.exe experiments\PHASE-2\analyze.py --root results\phase2
.venv\Scripts\python.exe experiments\PHASE-2\audit.py --root results\phase2
.venv\Scripts\python.exe experiments\PHASE-2\report.py --root results\phase2
```

The runner writes one JSON record per fit under `results/phase2/fits/` and resumes completed successful fits. Formal output is separate from the engineering smoke directory, which is not part of the formal evidence.

## Formal outcome

- 360/360 fits successful; 0 failures
- IID-match, rho=0 sanity, width ladder, pairing, completeness, and manifest checks: PASS
- Q1 schedule residual: below the 0.02 negligible gate in every high-rho width cell
- Q2 width-regime range: below the 0.02 negligible gate at both high-rho values
- Q3 interaction range: below 0.02; no direction change across widths
- 360/360 trajectories: `No-acquisition / no-distortion`
- Frozen decision: `Both weak / STOP`
- Reviewer-2 verdict: `NOT ESTABLISHED`
- Phase 3: not started

Primary report: [`Phase2_Training_Regime_Results.md`](../../Phase2_Training_Regime_Results.md)  
Audit report: [`Post_Experiment_Audit_Phase2.md`](../../Post_Experiment_Audit_Phase2.md)

## Figure contract

The shipped figure set is deliberately minimal and uses static Matplotlib output with an explicit blue/gold palette and line-style width encoding:

| Figure | Question | Form | Evidence |
| --- | --- | --- | --- |
| 01 | How does stable-boundary fidelity evolve? | Two-panel multi-series line with SEM ribbon | Fidelity checkpoints by high-rho, width, schedule |
| 02 | What is the paired schedule residual? | Point/error-bar comparison | Cosine minus constant at fixed width and rho |
| 03 | What is the terminal width pattern? | Two-panel point/error-bar comparison | Fidelity(200) by width and schedule |
| 04 | Does shortcut sensitivity change over training? | Two-panel multi-series line with SEM ribbon | D3 checkpoints; descriptive only |

No figure is a leaderboard, and D3 is not used as a substitute for Fidelity(200).

## Evidence files

`results/phase2/manifest.json` is the checksum manifest for raw fit records, checkpoint trajectories, OOD results, summaries, guards, audit outputs, and figures. `results/phase2/audit_checks.csv` contains the machine-readable 17-check audit.
