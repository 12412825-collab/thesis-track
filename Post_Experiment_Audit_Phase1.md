# Post-Experiment Audit — Phase 1

Audit date: 2026-08-18  
Primary run commit: `61abc6ef373e672757b2c2189ec06a3a6d37a60e`  
Decision: `G4a-C / REVISE HYPOTHESIS`

## Audit opinion

The Phase 1 evidence package is internally consistent, reproducible at its declared grain, and adequate for the narrow G4a-C conclusion. It is **not** adequate for causal model-family claims or for launching Phase 2. The most important limitation is not a failed run: it is the weak cross-family overlap of the approximate capacity/regularization controls and the fact that family × rho interaction exceeds the stable family component.

## Protocol adherence

| Requirement | Status | Evidence |
|---|---|---|
| Audit before experiment | Passed | `Research_Status_Audit_Phase1.md` |
| Frozen preregistration before outcome analysis | Passed | Configuration SHA-256 embedded in preregistration |
| Five existing families only | Passed | LR, RBF-SVM, RF, HGB, MLP |
| Exactly 15 configurations | Passed | Three unique hashed configurations per family |
| Training-only capacity calibration | Passed | Design artifact contains no performance outcome |
| OOD excluded from fitting | Passed | Fit API signature test and implementation boundary |
| Smoke before formal run | Passed | 30 fits, 75 rows, all stop gates green |
| Full family × config × rho × seed grid | Passed | 600 unique successful fits |
| Long master result contract | Passed | 3,450 unique rows, 46 columns |
| Failure/timing/capacity metadata | Passed | Fit ledger and master table |
| Variance, regression, matching, envelope, reversal, guard | Passed | Frozen summary tables |
| Five publication figures in PNG/PDF | Passed | Ten figure files, visually inspected |
| G4a rule applied without discretionary override | Passed | `g4a_decision.json` |
| Phase 2 not run | Passed | G4a-B was not met |

## Execution incident and recovery audit

The first formal grid process completed 136 full fit/result units and wrote a 137th fit record before a transient OneDrive `OSError 22` interrupted the environment-result checkpoint write. This was an infrastructure write failure, not a model failure.

Recovery actions:

1. Audited both partial tables: 136 fit IDs had complete five-row rho-0 environment records; exactly one fit ID had no result rows.
2. Changed checkpoint persistence to same-directory temporary write plus atomic replacement with bounded retries.
3. Changed resume logic so a fit is complete only if both its fit record and expected environment rows are present.
4. Removed the half-committed fit record in memory and deterministically reran that one scientific cell.
5. Added `--resume` so the completed 405-cell wrong-line guard was not rerun.
6. Ran all 24 tests before resuming and recorded the new source commit/hash.

No hyperparameter, seed, data, metric, capacity proxy, threshold, or analysis rule changed. The final result table has no duplicate cell from recovery. Run metadata records `resume_count=1`. Reported process runtime covers the resumed process; approximate full execution also included the initial five-minute partial grid and four-minute wrong-line scan, which is a metadata limitation rather than an outcome limitation.

## Data-quality reconciliation

- `fit_records.csv`: 600 rows; one per `config_id × rho_train × seed`; no duplicate key.
- `master_results.csv`: 3,450 rows; one per fit × IID/OOD environment; no duplicate `row_id` or scientific key.
- Expected environment counts: five rows per rho-0 fit and six per positive-rho fit.
- All primary metrics finite; bounded scores remained in `[0,1]`; log loss was non-negative.
- OOD gaps reconstructed from independent IID and raw OOD rows to maximum error `3.19e-16`.
- All 600 fit statuses were success.
- The one captured warning concerned Windows physical-core discovery, not convergence.
- Source tables used for every plot agree with the plotted values.
- `results/phase1/manifest.json` records path, byte size, and SHA-256 for every frozen result artifact except itself.

## Reviewer-2 stress tests

### 1. Capacity is not truly cross-family equivalent

The common empirical capacity index is a within-family/rho percentile. It enables ordering, not metric equivalence. Effective logistic degrees of freedom, RBF kernel rank, tree leaves, and MLP parameters describe different hypothesis spaces. The phrase “capacity controlled” must therefore remain qualified as approximate measured control.

### 2. Matching overlap is poor

Only 2 of 120 pairwise strata met 50% coverage. This means the matched analysis cannot falsify all capacity-based explanations. The preregistered fallback was used and stayed below the practical threshold, but fallback regression inherits modeling assumptions.

### 3. Three configurations per family are sparse

The 15-point registry samples broad low/medium/high paths. It cannot independently identify every architecture, regularization, optimization, and calibration dimension. LR capacity is particularly inseparable from regularization because architecture is fixed.

### 4. IID accuracy may mediate rather than confound

Controlling IID accuracy can remove part of the mechanism of shortcut exploitation. The prespecified no-IID sensitivity increased family CV increment from 0.0156 to 0.0229, still below 0.05 and still below the interaction increment. The G4a-C conclusion survives.

### 5. Unequal OOD environment counts by regime

The primary average uses four OOD environments at rho 0 and five at positive rho. A post hoc common-four-environment sensitivity produced family CV increment 0.0151 and interaction 0.0273, nearly identical to primary values. This concern does not explain G4a-C.

### 6. Calibration and optimization differ by family

RBF-SVM includes training-only three-fold sigmoid calibration; MLP uses internal early stopping; tree methods and LR do not use identical optimization protocols. These differences are part of “family” and cannot be separated by this design.

### 7. Ten seed clusters limit uncertainty resolution

Five thousand bootstrap resamples do not create more than ten independent seed clusters. The narrow intervals reflect a strongly balanced synthetic design, not broad population coverage.

### 8. Single synthetic shift mechanism

The shortcut is label-conditional, and requested rho is an operational coefficient rather than an exact finite-sample correlation. No evidence here transfers automatically to other shift types or real domains.

### 9. Wrong-line behavior is itself regime-dependent

Neutral high-capacity fidelity passed, but strong shortcut produced geometry/capacity-specific boundary distortion. This does not invalidate the main run; it suggests that “learned bias” should be operationalized as a conditional interaction rather than an intrinsic family trait.

## Claim audit

Safe claim: in this frozen synthetic design, rho dominates variance and family effects are small but detectably regime-dependent after measured controls.

Unsafe claim: Random Forest is inherently non-robust, MLP is inherently robust, or family causes the observed gap. The rho-0 near-equality, changing rankings, configuration spreads, and limited matching overlap directly contradict such generalization.

## Reproducibility handoff

1. Install `requirements.txt` in Python 3.10+.
2. Run `python experiments/PHASE-1/run.py --calibrate-capacity` for the outcome-free design check.
3. Run `python experiments/PHASE-1/run.py --smoke` and require a PASS report.
4. Run `python experiments/PHASE-1/run.py`; if interrupted after valid checkpoints, use `--resume`.
5. Run `python experiments/PHASE-1/analyze.py`.
6. Run `pytest -q` and reconcile hashes with `results/phase1/manifest.json`.

## Audit conclusion

The evidence is suitable for the Phase 1 decision and unsuitable for Phase 2 authorization. The next research move should revise the hypothesis and improve overlap/identification, not isolate a presumed stable family bias.

