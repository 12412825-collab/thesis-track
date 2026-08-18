# EXP-002 — Training Shortcut Strength Sweep

This controlled experiment asks whether the EXP-001 IID/OOD model-ranking reversal persists as the training shortcut strength changes.

Only `rho_train` changes: `0.5`, `0.7`, and `0.9`. For each regime, `rho_val = rho_train`. The test grid is `[rho_train, 0.3, 0.0, -0.3, -0.6, -0.9]`, following the standardized relative-grid option in the study specification. All other data, model, preprocessing, and seed settings are frozen to EXP-001.

OOD environments are evaluation-only. They are never available to fitting, calibration, early stopping, or model selection.

Run a one-seed smoke test:

```powershell
.\.venv\Scripts\python.exe experiments\EXP-002\run.py --smoke
```

Run the complete five-seed experiment:

```powershell
.\.venv\Scripts\python.exe experiments\EXP-002\run.py
```

Smoke outputs are isolated under `results/EXP-002/smoke/`. Complete outputs are written under `results/EXP-002/`.

