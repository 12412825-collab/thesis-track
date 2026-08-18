# EXP-002 — Training Shortcut Strength Sweep

## Question

Does the preliminary IID/OOD model-ranking reversal from EXP-001 persist when training-time shortcut strength changes?

## Experimental Intervention

`rho_train` is varied over `0.5`, `0.7`, and `0.9`; `rho_val` always equals `rho_train`. Each regime is evaluated on `[rho_train, 0.3, 0.0, -0.3, -0.6, -0.9]`. All sample sizes, stable mechanism, feature distributions, label noise, model families, architectures, hyperparameters, preprocessing, metrics, and five seeds are held fixed from EXP-001. OOD data are evaluation-only.

## Controls

OBSERVATION: The stable-feature-only maximum absolute aggregate OOD gap was `0.0039`.

OBSERVATION: Sanity-check warnings: None triggered.

## Results

OBSERVATION: Aggregate all-feature measurements (mean across seeds):

```text
 rho_train               model  validation_accuracy_mean  average_ood_accuracy_mean  worst_environment_accuracy_mean  ood_gap_mean  spurious_reliance_score_mean
    0.5000 logistic_regression                    0.9316                     0.8637                           0.8139        0.0657                        0.0467
    0.5000             rbf_svm                    0.9292                     0.8515                           0.7953        0.0763                        0.0554
    0.5000       random_forest                    0.9243                     0.8446                           0.7811        0.0760                        0.0530
    0.5000   gradient_boosting                    0.9213                     0.8496                           0.7987        0.0717                        0.0521
    0.5000                 mlp                    0.9316                     0.8508                           0.7926        0.0772                        0.0551
    0.7000 logistic_regression                    0.9550                     0.7894                           0.6802        0.1631                        0.1226
    0.7000             rbf_svm                    0.9530                     0.7722                           0.6518        0.1801                        0.1300
    0.7000       random_forest                    0.9488                     0.7248                           0.5616        0.2196                        0.1605
    0.7000   gradient_boosting                    0.9453                     0.7531                           0.6227        0.1898                        0.1448
    0.7000                 mlp                    0.9542                     0.7612                           0.6304        0.1898                        0.1368
    0.9000 logistic_regression                    0.9941                     0.5523                           0.2593        0.4399                        0.3736
    0.9000             rbf_svm                    0.9939                     0.5605                           0.2745        0.4320                        0.3634
    0.9000       random_forest                    0.9900                     0.4129                           0.0608        0.5767                        0.4678
    0.9000   gradient_boosting                    0.9909                     0.4522                           0.1019        0.5394                        0.4512
    0.9000                 mlp                    0.9918                     0.5994                           0.3433        0.3910                        0.3238
```

## Ranking Reversal

OBSERVATION:

```text
 rho_train      IID_best_model      OOD_best_model worst_case_best_model  ranking_reversal
       0.5 logistic_regression logistic_regression   logistic_regression             False
       0.7 logistic_regression logistic_regression   logistic_regression             False
       0.9 logistic_regression                 mlp                   mlp              True
```

INTERPRETATION: A reversal flag only states that the IID-validation winner and average-OOD winner differ within this controlled configuration. It does not identify a causal model-family property.

## Spurious Reliance

OBSERVATION: Descriptive Pearson correlations between aggregate permutation reliance and OOD gap were:

```text
           scope  rho_train  n_model_points  pearson_correlation
   all_rho_train        NaN              15               0.9987
within_rho_train     0.5000               5               0.9653
within_rho_train     0.7000               5               0.9730
within_rho_train     0.9000               5               0.9950
```

HYPOTHESIS: Shortcut reliance may help explain some between-model OOD degradation differences. This requires testing across additional independently varied regimes.

## Unexpected Results

OBSERVATION: No configured sanity or stable-only control warning was triggered.

## What We Can Claim

CLAIM: For these exact synthetic regimes, seeds, fixed models, and fixed hyperparameters, the recorded ranking-reversal table and degradation measurements are reproducible descriptive results.

## What We Cannot Claim

CLAIM: These experiments do not establish that any model family is inherently robust or fragile, that reliance causes degradation, or that the ranking pattern generalizes to real datasets or other shift mechanisms.

## Suggested Next Experiment

HYPOTHESIS: Repeating EXP-002 with 10 seeds, without changing another factor, would provide a more precise estimate of ranking stability before introducing EXP-003.
