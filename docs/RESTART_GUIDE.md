# Restart Guide

This guide is intended to let a researcher understand and resume the frozen project in roughly 10–20 minutes after cloning it.

## 1. Environment setup

Python 3.10+ is the documented baseline. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```

The repository currently uses NumPy, pandas, SciPy, scikit-learn, Matplotlib, PyYAML, and pytest. Preserve the versions recorded in each experiment's metadata when reproducing a frozen result.

## 2. Important configuration and scripts

- Baseline config: `configs/default.yaml`
- EXP-001: `experiments/EXP-001/config.yaml` and its `README.md`
- EXP-002 / EXP-002V: their local configs, READMEs, and `run.py`
- Phase 1: `experiments/PHASE-1/config.yaml`, `run.py`, `analyze.py`
- Phase 2: `experiments/PHASE-2/config.yaml`, `run.py`, `analyze.py`, `audit.py`, `report.py`
- Bridge: `experiments/BRIDGE-P1MLP-L-P2TASK/`

Do not overwrite archived result namespaces. Use an isolated output root for any reproduction.

## 3. Experiment registry and results

| Study | Question | Primary evidence |
|---|---|---|
| EXP-001 | Baseline shortcut shift and exploratory family behavior | `results/EXP-001/` |
| EXP-002 | Does the pattern vary with training shortcut strength? | `results/EXP-002/` |
| EXP-002V | Paired verification and reliance diagnostics | `results/EXP-002V/` |
| PHASE1-FAMILY-CAPACITY | Family vs approximate capacity controls | `results/phase1/` |
| PHASE-2 | Training regime / boundary-distortion intervention | `results/phase2/` |
| BRIDGE-P1MLP-L-P2TASK | Approved learner-only bridge | `results/bridge_p1mlp_l/` |

Read the result report and the corresponding audit together. CSV files are the machine-readable evidence; manifests and hashes protect frozen outputs; figures are descriptive support, not independent claims.

## 4. Tests and leakage boundary

Run `pytest -q` before any restart. The most important boundary is that OOD/test outputs cannot enter semantic selection, fitting, tuning, hypothesis scoring, or retry decisions. The existing tests include `tests/test_no_leakage.py` and experiment-specific output/audit checks.

## 5. Frozen Git state

The stable archive is identified by the local freeze branch and annotated tag created with this handoff:

```text
branch: distshift/frozen-phase1
tag:    2026-08-distshift-freeze
```

If the tag is already present, verify it with `git show --stat 2026-08-distshift-freeze`. If the branch/tag cannot be found on a clone, recover the exact commit from the handoff report before reproducing anything.

## 6. First recommended experiment after restart

Do not expand the benchmark horizontally. First write and freeze a small mechanism-intervention preregistration answering what M is, how it is manipulated, how search budget is controlled, which measurements are train-only, and which competing explanations are ruled out. Start only after the intervention passes an independent smoke test and leakage audit.

