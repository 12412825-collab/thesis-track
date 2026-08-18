# EXP-002V — Preregistered Verification

## Purpose

OBSERVATION: EXP-002V evaluates whether the EXP-002 degradation pattern and IID-only shortcut-reliance association survive a paired replicate design, a no-shortcut control, calibrated D3 measurement, and bootstrap uncertainty.

INTERPRETATION: This is verification of the existing question, not discovery of a new shift or a universal model-family comparison.

## Relationship to EXP-002

OBSERVATION: EXP-001 and EXP-002 files were not modified. EXP-002V preserves their stable mechanism, sample sizes, model families, hyperparameters, preprocessing, label noise, and OOD geometry, while adding `rho_train=0`, an independent noise feature, 10 replicates per regime, and 20 at the 0.9 anchor.

UNRESOLVED CONFOUNDER: Capacity, regularization, representation, and optimization remain uncontrolled across model families.

## Generator Validation

OBSERVATION: The shortcut is label-conditional. Maximum absolute target-versus-realized rho error was `0.0773`. The stable-rule oracle and stable-feature distribution checks passed the preregistered stop gate.

STATISTICAL EVIDENCE: Full per-environment means, variances, KS statistics, class balance, shortcut AUROC, mutual information, and noise correlations are in `results/EXP-002V/summaries/generator_validation.csv`.

## H1 — Degradation Structure

OBSERVATION: Mean OOD gaps and 10,000-resample 95% bootstrap intervals were:

```text
              model  rho_to  estimate  ci_lower  ci_upper  n_replicates
  gradient_boosting  0.0000    0.0029   -0.0012    0.0071            10
  gradient_boosting  0.5000    0.0759    0.0712    0.0809            10
  gradient_boosting  0.7000    0.1942    0.1871    0.2020            10
  gradient_boosting  0.9000    0.5435    0.5409    0.5461            20
logistic_regression  0.0000    0.0030   -0.0014    0.0070            10
logistic_regression  0.5000    0.0701    0.0656    0.0746            10
logistic_regression  0.7000    0.1660    0.1612    0.1716            10
logistic_regression  0.9000    0.4471    0.4428    0.4517            20
                mlp  0.0000    0.0029   -0.0022    0.0081            10
                mlp  0.5000    0.0878    0.0739    0.1037            10
                mlp  0.7000    0.1953    0.1810    0.2098            10
                mlp  0.9000    0.3984    0.3847    0.4113            20
      random_forest  0.0000    0.0028   -0.0014    0.0071            10
      random_forest  0.5000    0.0843    0.0801    0.0887            10
      random_forest  0.7000    0.2269    0.2202    0.2330            10
      random_forest  0.9000    0.5828    0.5811    0.5846            20
            rbf_svm  0.0000    0.0036   -0.0008    0.0083            10
            rbf_svm  0.5000    0.0832    0.0784    0.0884            10
            rbf_svm  0.7000    0.1820    0.1767    0.1880            10
            rbf_svm  0.9000    0.4454    0.4413    0.4499            20
```

STATISTICAL EVIDENCE: Monotonic non-decrease by family: {'gradient_boosting': True, 'logistic_regression': True, 'mlp': True, 'random_forest': True, 'rbf_svm': True}.

STATISTICAL EVIDENCE: Paired consecutive OOD-gap changes (higher rho minus lower rho) were:

```text
              model  rho_from  rho_to  estimate  ci_lower  ci_upper  n_replicates
logistic_regression    0.0000  0.5000    0.0671    0.0632    0.0711            10
logistic_regression    0.5000  0.7000    0.0960    0.0931    0.0987            10
logistic_regression    0.7000  0.9000    0.2785    0.2720    0.2854            10
            rbf_svm    0.0000  0.5000    0.0796    0.0755    0.0839            10
            rbf_svm    0.5000  0.7000    0.0988    0.0959    0.1013            10
            rbf_svm    0.7000  0.9000    0.2628    0.2555    0.2703            10
      random_forest    0.0000  0.5000    0.0815    0.0780    0.0848            10
      random_forest    0.5000  0.7000    0.1426    0.1374    0.1478            10
      random_forest    0.7000  0.9000    0.3544    0.3481    0.3609            10
  gradient_boosting    0.0000  0.5000    0.0729    0.0685    0.0778            10
  gradient_boosting    0.5000  0.7000    0.1183    0.1142    0.1228            10
  gradient_boosting    0.7000  0.9000    0.3480    0.3419    0.3542            10
                mlp    0.0000  0.5000    0.0849    0.0728    0.0989            10
                mlp    0.5000  0.7000    0.1075    0.0871    0.1272            10
                mlp    0.7000  0.9000    0.2094    0.1851    0.2344            10
```

OBSERVATION: At the `rho_train=0` control, the mean all-family OOD gap was `0.0030`.

INTERPRETATION: The curves describe degradation under the fixed synthetic mechanism; they do not establish an inherent family property.

## Ranking Reversal Validation

OBSERVATION: Tie-aware probability that the IID-best and OOD-best sets were disjoint:

```text
 rho_train  probability_iid_best_differs_from_ood_best  n_replicates
     0.000                                       0.400            10
     0.500                                       0.300            10
     0.700                                       0.600            10
     0.900                                       0.650            20
```

STATISTICAL EVIDENCE: At `rho_train=0.9`, `P(LR > MLP IID)=0.600` and `P(MLP > LR OOD)=0.900` across 20 replicates; overall best-set reversal probability was `0.650`.

## D3 Shortcut Reliance

OBSERVATION: The no-shortcut control mean D3 excess was `-0.0006`. D3 replaced `x_spurious` by a training-only OLS estimate conditional on standardized stable features and subtracted the identical flip-rate diagnostic for independent noise `z`.

STATISTICAL EVIDENCE: D3 noise-floor values are exported separately; probability-sensitivity variants are retained as secondary audit fields.

## D1 LOCO Cross-Check

OBSERVATION: Raw and normalized LOCO retain negative values and are reported separately from D3. Aggregate D3-versus-LOCO rank correlation was `0.944`; exact rank-order agreement was `False`.

STATISTICAL EVIDENCE: Within-rho family-ranking agreement was:

```text
        scope  spearman_correlation  exact_rank_order_match
rho_train_0.0                 0.700                   False
rho_train_0.5                -0.600                   False
rho_train_0.7                -0.600                   False
rho_train_0.9                 0.800                   False
```

INTERPRETATION: The two diagnostics are not averaged. Their non-identical rankings remain a measurement confound even when the aggregate configuration correlation is high.

## H2 — Reliance vs OOD Degradation

OBSERVATION: D3_excess Pearson association was `0.863` (95% CI `0.818` to `0.903`); Spearman was `0.937` (95% CI `0.892` to `0.956`). LOCO Pearson was `0.967`.

STATISTICAL EVIDENCE: In `OOD_gap ~ IID_test_accuracy + D3_excess`, the D3 coefficient was `0.437` (95% CI `0.078` to `1.050`), partial correlation was `0.110` (95% CI `0.021` to `0.232`), baseline R² was `0.905`, full R² was `0.906`, and incremental R² was `0.001` (95% CI `0.000` to `0.005`).

INTERPRETATION: EXPLORATORY ONLY — INSUFFICIENT CONFIGURATION DIVERSITY. This regression is predictive/descriptive and not causal evidence for H3.

## Leave-One-Family-Out Analysis

OBSERVATION: Did D3 coefficients remain positive after every family exclusion? `False`. Did every cluster-bootstrap lower confidence bound remain above zero? `False`. Exclusions with a coefficient interval crossing zero: `random_forest, mlp`.

STATISTICAL EVIDENCE: Full coefficient, correlation, partial-correlation, and incremental-R² sensitivity results are in `leave_one_family_out.csv`.

```text
    excluded_family  pearson  d3_coefficient  d3_coefficient_ci_lower  d3_coefficient_ci_upper  incremental_r2
logistic_regression    0.865           0.550                    0.164                    1.201           0.002
            rbf_svm    0.861           0.555                    0.178                    1.191           0.002
      random_forest    0.872          -0.025                   -0.433                    0.518           0.000
  gradient_boosting    0.853           0.525                    0.201                    1.118           0.002
                mlp    0.866           0.184                   -0.202                    0.813           0.000
```

## Stable-Only Control

OBSERVATION: The maximum absolute model-by-rho mean stable-only OOD gap was `0.0036`. The figure and environment-level table show whether residual movement is sampling noise rather than a changed stable mechanism.

## What Survived EXP-002

OBSERVATION: OOD degradation increased monotonically in the mean for every family, and all 15 paired consecutive-difference intervals were above zero: `True`. At `rho_train=0.9`, MLP beat LR in average OOD accuracy in `90%` of replicates. The overall D3 coefficient remained positive after IID-test accuracy adjustment, with its full-sample bootstrap interval above zero.

## What Failed to Replicate

OBSERVATION: The specific ranking reversal was not stable enough for an unqualified replication claim: LR beat MLP IID in only `60%` of anchor replicates, and the tie-aware probability that the overall IID-best and OOD-best sets were disjoint was `65%`. D3 and LOCO did not produce identical family rankings in any rho regime. Leave-one-family-out coefficient intervals crossed zero for `random_forest, mlp`, so the adjusted mechanism signal is family-dependent.

## What We Can Claim

STATISTICAL EVIDENCE: For this fixed synthetic generator and fixed model configurations, the exported bootstrap intervals, replicate ordering probabilities, and IID-only reliance associations are reproducible measurements.

## What We Cannot Claim

UNRESOLVED CONFOUNDER: EXP-002V cannot establish that a family has a better inductive bias, that D3 reliance causes degradation, or that adding reliance will select robust models across broader capacity and regularization configurations.

## Decision Recommendation

INTERPRETATION: **C. Reliance explains degradation beyond IID accuracy, but capacity/tuning is still uncontrolled.** The overall adjusted association clears zero, but its incremental R² is small and leave-one-family-out uncertainty is not uniformly positive. Do not start another experiment without PI approval.
