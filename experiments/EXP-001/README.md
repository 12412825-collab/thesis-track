# EXP-001 — Baseline Spurious Correlation Shift

## Research question

Does changing the correlation of a spurious feature produce different OOD degradation patterns across model families?

## Frozen configuration

The complete machine-readable snapshot is [`config.yaml`](config.yaml). The experiment uses:

- training and IID-validation shortcut correlation: `rho_train = rho_val = 0.9`;
- test correlations: `[0.9, 0.6, 0.3, 0.0, -0.3, -0.6, -0.9]`;
- sample sizes: 5,000 training, 2,000 validation, and 3,000 per test environment;
- stable target rule: `1.2*x1 - 1.0*x2 + 0.8*x3 + Normal(0, 0.5) > 0`;
- seeds: `[0, 1, 2, 3, 4]`;
- model families: regularized logistic regression, calibrated RBF-SVM, random forest, histogram gradient boosting, and a two-layer MLP;
- fixed model hyperparameters exactly as listed in the configuration snapshot;
- feature conditions: all features and stable features only;
- per-environment metrics: accuracy, balanced accuracy, and log loss;
- derived metrics: IID accuracy, average OOD accuracy, worst-environment accuracy, OOD gap, severe-shift accuracy, degradation slope, and permutation-based spurious reliance.

OOD environments are evaluation-only and cannot be passed to the fitting/model-selection interface.

## Reproduction

From the repository root:

```powershell
.\.venv\Scripts\python.exe experiments\run_spurious_shift.py --config experiments\EXP-001\config.yaml
```

The frozen successful outputs are under `results/EXP-001/`. The command above retains the original V1 behavior and writes a fresh reproduction to the legacy `results/raw`, `results/summaries`, and `results/figures` paths; it does not overwrite the frozen archive.

## Observed preliminary result

In this single configuration, the IID-validation best model was Logistic Regression, while MLP had the highest average OOD and worst-environment accuracy. Removing `x_spurious` made accuracy approximately stable across environments.

## Important warning

This is one synthetic configuration only and is not evidence of a universal model-family property. These measurements are preliminary observations, not causal or theoretical conclusions.

