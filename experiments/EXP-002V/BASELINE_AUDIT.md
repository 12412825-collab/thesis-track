# EXP-002V Baseline Audit

This audit was completed before EXP-002V was run. EXP-001 and EXP-002 are treated as immutable evidence inputs.

## Audited artifacts

- `experiments/EXP-002/config.yaml`
- `experiments/EXP-002/README.md`
- `experiments/EXP-002/REPORT.md`
- `src/data_generation.py`, `src/models.py`, `src/diagnostics.py`, `src/evaluation.py`, and `src/sweep.py`
- `results/EXP-002/raw/raw_results.csv` (900 rows; no duplicate rows at model × rho_train × rho_test × seed × feature-condition grain)
- every CSV/JSON file under `results/EXP-002/summaries/`
- all five EXP-002 PNG figures and their PDF counterparts

No EXP-002 result is reinterpreted in this audit.

## Exact baseline specification

1. **Shortcut construction.** The target is encoded as `y_signed = 2*y - 1`, then `x_spurious = rho*y_signed + sqrt(1-rho^2)*epsilon`, with independent standard Gaussian `epsilon`. The shortcut is therefore label-conditional. Because `Y` is generated from the stable features, `x_spurious` is also marginally associated with the stable features through `Y`; it is not constructed as an independent covariate.
2. **Definition of rho_train.** `rho_train` is the coefficient in the label-conditional shortcut equation for the training environment. With balanced labels and independent unit-variance noise, it approximates `corr(x_spurious, y_signed)`; realized correlations were saved separately.
3. **Test/OOD environments.** EXP-002 used `[rho_train, 0.3, 0.0, -0.3, -0.6, -0.9]`, deduplicated in order. The `rho_test=rho_train` environment was treated as IID; every other value was evaluation-only OOD.
4. **Sample sizes.** 5,000 training rows, 2,000 IID-validation rows, and 3,000 rows per test environment.
5. **Label noise.** Independent Gaussian score noise with standard deviation `0.5`.
6. **Stable decision mechanism.** `Y = 1[1.2*x1 - 1.0*x2 + 0.8*x3 + Normal(0, 0.5) > 0]`, with each stable feature independently distributed as `N(0,1)`.
7. **Model hyperparameters.** Logistic Regression: `C=1`, `max_iter=1000`. Calibrated RBF-SVM: `C=1`, `gamma=scale`, cache 1000 MB, sigmoid calibration with 3 folds. Random Forest: 250 trees, `min_samples_leaf=2`, `n_jobs=-1`. Histogram Gradient Boosting: learning rate 0.08, 200 iterations, 31 maximum leaf nodes, L2 regularization 0.1. MLP: hidden layers `(64,64)`, alpha 0.001, learning rate 0.001, 250 maximum iterations, training-only early stopping.
8. **Preprocessing.** Logistic Regression, RBF-SVM, and MLP used `StandardScaler` pipelines. Random Forest and Histogram Gradient Boosting used unscaled features.
9. **Seeds.** `[0,1,2,3,4]`. Model families shared each generated train/validation/test realization within a seed. Stable samples and labels were also paired across rho_train regimes by deterministic child seeds.
10. **Existing reliance metric.** IID-validation accuracy minus accuracy after an unrestricted random permutation of `x_spurious`. It used IID data only, but did not preserve the conditional relationship between stable features and the shortcut. It is retained only as historical EXP-002 evidence and is not the EXP-002V headline metric.

## Preregistration requirements already satisfied

- The scientific question and model-family set are fixed.
- Training strength values 0.5, 0.7, and 0.9 already exist.
- `rho_val = rho_train`; OOD environments are excluded from fitting and selection by API design.
- All model families are paired on common generated samples within each regime/seed.
- Fixed hyperparameters and preprocessing avoid OOD-guided tuning.
- All-features and stable-only fits already exist.
- Accuracy, balanced accuracy, log loss, IID accuracy, average OOD accuracy, worst-environment accuracy, OOD gap, and severe-shift accuracy already exist.
- Requested versus realized shortcut correlation, class balance, and stable-feature means/standard deviations were saved.
- Data generation is deterministic for fixed seeds, and tests enforce the no-OOD fitting boundary.

## Missing preregistration requirements

- The `rho_train=0.0` no-shortcut control.
- 10 replicates for all regimes and 20 at the `rho_train=0.9` anchor.
- Independent Gaussian noise feature `z` and its explicit noise-floor diagnostic.
- Primary stable-conditional D3 counterfactual flip rate and `D3_excess`.
- Raw and normalized D1 LOCO at every model × rho_train × replicate.
- Conditional permutation, consistent/contradictory groups, and model-specific usage audits.
- ROC-AUC and Brier score in per-environment performance records.
- Stable-feature variance differences, KS checks, stable-rule oracle invariance, shortcut-only AUROC, and mutual information.
- Bootstrap confidence intervals for H1 levels and consecutive differences.
- Replicate-level ranks, pairwise ordering probabilities, and ranking-reversal probability.
- Pearson/Spearman confidence intervals, IID-adjusted regression, partial correlation, R-squared comparison, leave-one-family-out sensitivity, and D3-versus-LOCO agreement.
- The preregistered EXP-002V tables, seven figures, and decision-classification report.

## Audit conclusion

EXP-002 provides a usable, leakage-protected baseline, but it does not satisfy the stricter uncertainty and mechanism-measurement protocol. EXP-002V must be run as a separate experiment and must not overwrite any audited artifact.
