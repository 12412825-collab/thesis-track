# Research Status Audit — Phase 1 Family vs Capacity

Audit date: 2026-08-18  
Audit basis: commit `efb1477923ba5e483f58caed5682a0dbf1cd664b` on `codex/exp-002-verification`  
Phase 1 working branch: `codex/phase1-family-capacity`

## Audit conclusion

**Status: READY WITH MATERIAL DESIGN CONSTRAINTS.** The repository contains a reproducible, leakage-protected synthetic spurious-correlation benchmark and a completed verification experiment. It does **not** yet contain a capacity-controlled model-family experiment, an existing 15-configuration registry, or the external review and roadmap artifacts named in the mission. Phase 1 may proceed only after a new configuration registry, approximate cross-family capacity protocol, analysis thresholds, stop gates, and immutable output contract are frozen in a dedicated preregistration.

EXP-001, EXP-002, and EXP-002V and all of their archived results are immutable evidence inputs. Phase 1 will add new files under its own experiment and result namespaces and will not reinterpret or overwrite prior outputs.

## Inputs found and not found

| Input | Status | Audit finding |
|---|---|---|
| Root README and experiment registry | Found | Documents EXP-001 and EXP-002; EXP-002V is present in the repository but not yet listed in the root registry. |
| EXP-001 frozen configuration/results | Found | Five seeds, one `rho_train=0.9` regime, five fixed families, all-features and stable-only controls. |
| EXP-002 frozen configuration/results | Found | Five seeds, `rho_train={0.5,0.7,0.9}`, same five fixed configurations. |
| EXP-002V audit, configuration, report, code, and results | Found | Ten paired replicates per regime and twenty at `rho_train=0.9`; adds `rho_train=0`, independent noise feature, D3/LOCO diagnostics, uncertainty, and generator stop gates. |
| Independent Review | Not found | No repository or Git-history match. It is not reconstructed or attributed. |
| Methodology Pre-Registration outside existing experiment docs | Not found | No separate upstream document is available. Phase 1 will create its own frozen preregistration. |
| Literature / Research Map | Not found | No repository or Git-history match. No literature coverage or Phase 3 map is inferred. |
| G0–G6 definition document | Not found | No repository or Git-history match. Existing gate semantics are not invented or changed. Only G4a, explicitly defined in the Phase 1 mission, will be operationalized. |
| Existing 15-configuration grid | Not found | Current code contains one fixed configuration per family. A new 15-configuration registry must be designed and frozen before any formal Phase 1 result is observed. |

The same missing-artifact search was also performed across the supplied Codex attachment texts available on this machine. Only the current Phase 1 mission and the prior EXP-002V request matched; no additional source document was recovered.

## Existing evidence ledger

### EXP-001

- Generator: three independent Gaussian stable features; target score `1.2*x1 - 1.0*x2 + 0.8*x3 + Normal(0,0.5)`; label-conditional shortcut `rho*(2y-1) + sqrt(1-rho^2)*epsilon`.
- Design: five seeds; five fixed model families; all-features and oracle stable-only conditions.
- Archived raw grain: `seed × model × feature_condition × rho_test`, 350 rows, no duplicate grain keys or missing values.
- Reproduction note reports exact accuracy agreement and only final-decimal floating-point differences in four log-loss values.
- Limitation: a single hyperparameter/configuration per family makes family, capacity, regularization, representation, and optimization inseparable.

### EXP-002

- Intervention: only training shortcut strength changes over `rho_train={0.5,0.7,0.9}`; stable latent samples and labels are paired across regimes by seed.
- Archived raw grain: `rho_train × seed × model × feature_condition × rho_test`, 900 rows, no duplicate grain keys or missing values.
- The fixed-configuration ranking reversal appeared only at `rho_train=0.9`; this remained exploratory.
- Existing permutation reliance and OOD gap were highly correlated in aggregate, but neither configuration diversity nor uncertainty was sufficient for a family-level explanation.

### EXP-002V

- Design: `rho_train={0,0.5,0.7,0.9}`; 10 paired replicates per regime and 20 at the 0.9 anchor; independent IID test split; five OOD environments excluding any duplicate IID point.
- Archived raw environment grain: `rho_train × replicate × model × feature_condition × environment_type × rho_environment`, 3,400 rows, no duplicate grain keys. The 1,000 missing `rho_ood` values occur only for IID validation/test rows and are structurally expected.
- The 500-row replicate performance table was independently reconstructed from raw environment records. Maximum absolute reconstruction discrepancies were 0 for IID and worst OOD accuracy, `2.22e-16` for average OOD accuracy, and `3.19e-16` for OOD gap.
- Verification findings: degradation increased monotonically with training shortcut strength for all five fixed families; the unqualified IID/OOD ranking-reversal claim did not stably replicate; adjusted D3 signal had very small incremental R² and failed leave-one-family-out stability for Random Forest and MLP.
- EXP-002V decision was branch C: mechanism signal remained family-dependent while capacity/tuning was uncontrolled. This motivates Phase 1 but is not evidence that family has an intrinsic causal effect.

## Repository and implementation audit

### Strengths available for reuse

- Deterministic child-seed generation and paired latent designs.
- Explicit split construction with independent train, IID validation, IID test, and OOD data.
- Fitting APIs that cannot receive OOD/test environments; tests enforce this boundary.
- Five existing model-family builders with family-appropriate preprocessing.
- Generator validation, stable-only controls, independent noise-floor feature, uncertainty utilities, and PNG/PDF plotting conventions.
- Complete baseline test suite: **19 passed** on the audited environment.

### Gaps Phase 1 must close

- No model/config registry with stable IDs, canonical serialized hyperparameters, or hashes.
- No raw or empirical cross-family capacity variables.
- No capacity calibration or matching procedure.
- No variance decomposition separating family, configuration nested within family, shift regime, interactions, seed, and residual variation.
- No configuration-level failure/status ledger or per-fit duration.
- Existing run metadata omits Git commit, dirty-tree state, config/source hashes, Python/package versions, platform/hardware, and output manifest hashes.
- Existing outputs are not guarded by an immutability manifest.
- No predefined practical-effect thresholds for G4a.

## Data-quality assessment

| Archive | Raw rows | Declared-grain duplicates | Unexpected missingness | Metric reconstruction |
|---|---:|---:|---:|---|
| EXP-001 | 350 | 0 | 0 | Consistent with frozen summaries and reproduction note |
| EXP-002 | 900 | 0 | 0 | Summary/report structure consistent with raw design |
| EXP-002V | 3,400 | 0 | 0 | Replicate IID/OOD metrics reconstructed to floating-point precision |

Metric ranges were finite and valid for the stored classifiers: accuracy, balanced accuracy, ROC-AUC, and Brier score remained in `[0,1]`; log loss was non-negative. Full-row duplicates were absent from every CSV under the three experiment archives. Nulls in EXP-002V audit/statistics tables are attributable to inapplicable fields or transition rows, not missing primary outcomes.

Archived raw SHA-256 checksums at audit time:

- EXP-001: `dda29cbbf1e1292622b5bbd2449bbd0f025b684ffc9e74197492adcd0530c12e`
- EXP-002: `fb3a15082e7067e739e33d926af8107eda79b3d15fb60bce7a0afb8c85de30ef`
- EXP-002V: `1411a79110340d6e483f953441362b983e66799aa4abfc0e13ae84469c715b5a`

## Leakage and comparison-boundary audit

The existing fitting interface accepts training and IID validation only, while OOD evaluation receives an already fitted model. EXP-002V ranking uses the independent IID test split rather than validation. Phase 1 must preserve these boundaries:

1. OOD outcomes may not select hyperparameters, capacity tiers, thresholds, calibration mappings, early stopping, or failed-run retries.
2. Empirical capacity calibration must use training data and model internals only; it may not use IID or OOD outcome performance.
3. IID validation may be used only where a family requires training-time calibration or early stopping already declared in configuration.
4. Primary ranking and degradation use the independent IID test and frozen OOD environments.

## Identification risks

1. **Family is not randomized.** Algorithms differ in representation, optimization, regularization semantics, probability calibration, and raw capacity measures. “Family residual” is a predictive residual under measured controls, not a causal family effect.
2. **Cross-family capacity is only approximately observable.** Parameter count, support-vector count, tree leaves, boosting leaves/stages, and MLP weights are not interchangeable. The preregistration must expose raw variables and treat any common scale as empirical and approximate.
3. **A 15-configuration total grid is sparse.** With three configurations per family, capacity and regularization cannot be independently and densely identified within every family. Configuration is therefore modeled as nested within family; capacity/regularization regression and matching are corroborating analyses, not a complete causal adjustment.
4. **Shortcut parameter is operational.** Requested `rho` is the coefficient in a label-conditional construction and only approximates realized correlation in finite samples.
5. **Synthetic scope is narrow.** Phase 1 studies one spurious-correlation shift generator. Results do not generalize automatically to covariate, label, concept, temporal, or real-world domain shifts.
6. **Calibration asymmetry.** The RBF-SVM currently uses cross-validated sigmoid calibration, unlike most other families. Fit-time accounting and configuration descriptions must expose this difference.

## Phase 1 readiness requirements

Before a formal run, the Phase 1 preregistration must freeze:

- exactly 15 total configurations, with three ordered presets per family and no OOD-guided adjustment;
- raw structural-capacity, learned-capacity, regularization, optimization, and convergence variables for every family;
- a training-only empirical capacity index and its diagnostics, explicitly labelled approximate;
- `rho_train`, OOD grid, sample sizes, seeds, stable rule, label noise, split pairing, and failure policy;
- master-result grain, deterministic IDs/hashes, timing, environment, and manifest schema;
- smoke-test stop gates, including generator, leakage, result-contract, failure, and wrong-line guards;
- orthogonal balanced-design variance decomposition, regression-control, matched/stratified analysis, robustness-envelope, and ranking-reversal methods;
- G4a practical and statistical thresholds, sensitivity analyses, and the rule mapping to `PROCEED`, `DO NOT PROCEED`, or `REVISE HYPOTHESIS`.

## Go-forward decision

Proceed to preregistration and smoke implementation. Do not run the complete Phase 1 grid until the preregistration is committed in the working tree and all smoke stop gates pass. Do not run Phase 2 in this task; at most, write a design document if G4a justifies it.
