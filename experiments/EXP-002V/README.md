# EXP-002V — Preregistered Verification

EXP-002V verifies the EXP-002 degradation and shortcut-reliance observations under a stricter paired-replicate and uncertainty protocol. It does not introduce a new shift type or model family.

The primary reliance diagnostic is stable-conditional counterfactual prediction flip rate calibrated against an independent Gaussian noise feature (`D3_excess`). LOCO is secondary. OOD data remain evaluation-only.

Run a one-replicate smoke test:

```powershell
.\.venv\Scripts\python.exe experiments\EXP-002V\run.py --smoke
```

Run the preregistered 10-replicate experiment with 20 replicates at `rho_train=0.9`:

```powershell
.\.venv\Scripts\python.exe experiments\EXP-002V\run.py
```

Smoke outputs are isolated under `results/EXP-002V/smoke/`; complete outputs are written under `results/EXP-002V/`.
