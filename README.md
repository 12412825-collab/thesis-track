# Distribution Shift Model Selection Research Sandbox V1

A small, reproducible research sandbox for studying model selection when training data contains a strong but unstable shortcut.

## Research question

IID validation can reward a model for using a feature whose relationship with the target is specific to the training environment. This experiment asks how five common model families behave when that relationship changes at evaluation time, and whether IID validation selects the same model as robustness-oriented criteria.

The project is exploratory. It generates evidence and diagnostics; it does not make automatic causal or theoretical claims.

## Experimental design

The binary target is generated from three stable Gaussian features:

```text
score = 1.2*x1 - 1.0*x2 + 0.8*x3 + Normal(0, 0.5)
y = 1(score > 0)
```

The configurable coefficients and label noise live in `configs/default.yaml`. A fourth feature is generated as:

```text
x_spurious = rho * (2*y - 1) + sqrt(1-rho^2) * epsilon
```

Training and IID validation use `rho=0.9`. Evaluation sweeps from `rho=0.9` to `rho=-0.9`. Stable feature and label mechanisms do not intentionally change. Empirical shortcut correlations, class balance, and stable-feature moments are saved for every generated environment.

Models: regularized logistic regression, RBF SVM, random forest, histogram gradient boosting, and a small MLP. Each model runs under two conditions: all four features, and the three stable features only. The stable-only condition is an oracle diagnostic, not a claim about information available in real applications.

V1 uses fixed, sensible hyperparameters. OOD environments are evaluation-only: the fitting interface accepts training and IID-validation data but has no OOD/test argument.

## Setup and running

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
python experiments/run_spurious_shift.py
```

Run tests with:

```bash
pytest -q
```

To use another configuration:

```bash
python experiments/run_spurious_shift.py --config configs/default.yaml
```

## Outputs

The run writes tidy per-environment metrics to `results/raw/results.csv`; aggregate and per-seed summaries, environment metadata, reliance diagnostics, and an automated text report to `results/summaries/`; and PNG/PDF figures to `results/figures/`.

Reported metrics include accuracy, balanced accuracy, log loss, IID accuracy, average OOD accuracy, worst-environment accuracy, severe-shift accuracy, OOD gap, and a descriptive degradation slope. Shortcut reliance is measured as the IID-validation accuracy drop after permuting `x_spurious`; it is model-agnostic, but remains an exploratory diagnostic.

## Interpretation warning

This is a synthetic controlled experiment. It does not establish a universal ranking of models. Results can depend on fixed hyperparameters, sample size, random seeds, and the data-generating process. The stable-only feature set uses privileged synthetic knowledge. The aim is to understand behavior, surface mechanisms worth testing, and generate research hypotheses—not to benchmark-chase or recommend a production model.

Future versions can add other controlled shifts through new data generators without changing the fitting/evaluation boundary. V1 intentionally excludes AutoML, domain adaptation, invariant learning, large benchmarks, and deep neural networks.

## Experiment registry

- [`EXP-001`](experiments/EXP-001/README.md): frozen baseline spurious-correlation shift (`rho_train=0.9`).
- [`EXP-002`](experiments/EXP-002/README.md): controlled training-shortcut-strength sweep (`rho_train=0.5, 0.7, 0.9`).
