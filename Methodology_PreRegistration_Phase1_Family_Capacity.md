# Methodology Pre-Registration — Phase 1 Family vs Capacity

Freeze date: 2026-08-18  
Experiment ID: `PHASE1-FAMILY-CAPACITY`  
Configuration: `experiments/PHASE-1/config.yaml`  
Frozen configuration SHA-256: `7f012f1f170e3bffd953334eeca4e1422f1dc54aaed46e3a79646da9645c852a`  
Status: **FROZEN BEFORE SMOKE AND FORMAL OUTCOME ANALYSIS**

## Research question

After controlling as far as this finite synthetic design permits for measured capacity, regularization, declared hyperparameters, IID performance, shift strength, and paired seed, how much variance in OOD degradation remains predictively attributable to model family?

The estimand is a **descriptive family residual**, not a causal effect of algorithmic family. Family is not randomized and its representation, optimization, calibration, and regularization semantics cannot be made identical.

## Prior evidence and separation from Phase 1

EXP-001, EXP-002, and EXP-002V are fixed exploratory/verification evidence. Their outcomes motivated this question but are not pooled into Phase 1 estimates. The Phase 1 configuration registry did not previously exist. It was designed and checked using training-only structural diagnostics before this freeze; no IID validation, IID test, or OOD performance was calculated during design calibration.

No Independent Review, Literature/Research Map, G0–G6 definition document, or external 15-configuration grid was available. Their contents are not reconstructed. This preregistration operationalizes only G4a as requested in the Phase 1 mission.

## Frozen generator and split design

- Stable features: independent `N(0,1)` variables `x1`, `x2`, and `x3`.
- Stable score: `1.2*x1 - 1.0*x2 + 0.8*x3 + Normal(0,0.5)`.
- Label: one when the score is positive, zero otherwise.
- Shortcut: `x_spurious = rho*(2y-1) + sqrt(1-rho^2)*epsilon`.
- Independent noise-floor feature: `z ~ N(0,1)` from a separate deterministic RNG stream.
- Training regimes: `rho_train={0.0,0.5,0.7,0.9}`; IID validation and IID test use the same requested rho as training.
- OOD grid: `{0.3,0.0,-0.3,-0.6,-0.9}`, excluding a value when it equals the current IID rho.
- Samples per cell: 5,000 train, 2,000 IID validation, 3,000 independent IID test, and 3,000 per OOD environment.
- Seeds: 0–9. The same latent stable samples, labels, shortcut noise, and `z` are paired across rho regimes where split identity permits; all configurations share each generated cell.
- Features used by every primary model: `x1,x2,x3,x_spurious,z`.

OOD data are evaluation-only. The fit function accepts only `entry, X_train, y_train, seed`; it cannot receive validation, test, rho-environment, or OOD arguments. RBF calibration uses three training-only folds. MLP early stopping uses its internal training split. IID validation is recorded as a fit audit and is not used to choose a configuration. Primary IID outcomes come from the independent IID test split.

## Frozen 15-configuration registry

There are three declared design-capacity tiers per family, 15 configurations total. The registry is intentionally small enough to run fully but too sparse for complete causal separation of every hyperparameter. Capacity tier and regularization flexibility are both exported; sensitivity analyses must state their remaining collinearity.

| ID | Family | Capacity tier | Regularization flexibility | Frozen defining settings |
|---|---|---:|---:|---|
| LR-L | Logistic Regression | 0.0 | 0.0 | `C=0.03`, `max_iter=1500` |
| LR-M | Logistic Regression | 0.5 | 0.5 | `C=1`, `max_iter=1500` |
| LR-H | Logistic Regression | 1.0 | 1.0 | `C=30`, `max_iter=1500` |
| SVM-L | RBF SVM | 0.0 | 0.5 | `C=1`, `gamma=0.05`, 3-fold sigmoid calibration |
| SVM-M | RBF SVM | 0.5 | 1.0 | `C=10`, `gamma=scale`, 3-fold sigmoid calibration |
| SVM-H | RBF SVM | 1.0 | 0.0 | `C=0.3`, `gamma=1`, 3-fold sigmoid calibration |
| RF-L | Random Forest | 0.0 | 1.0 | 200 trees, depth 4, leaf minimum 1 |
| RF-M | Random Forest | 0.5 | 0.0 | 250 trees, depth 10, leaf minimum 10 |
| RF-H | Random Forest | 1.0 | 0.5 | 300 trees, unrestricted depth, leaf minimum 2 |
| HGB-L | HistGradientBoosting | 0.0 | 0.0 | 100 stages, 7 leaves, L2 1.0, LR 0.08 |
| HGB-M | HistGradientBoosting | 0.5 | 1.0 | 200 stages, 31 leaves, L2 0.01, LR 0.08 |
| HGB-H | HistGradientBoosting | 1.0 | 0.5 | 300 stages, 63 leaves, L2 0.1, LR 0.05 |
| MLP-L | MLP | 0.0 | 1.0 | `(16)`, alpha 0.0001, max 250 iterations |
| MLP-M | MLP | 0.5 | 0.5 | `(64,64)`, alpha 0.001, max 300 iterations |
| MLP-H | MLP | 1.0 | 0.0 | `(128,128,64)`, alpha 0.01, max 350 iterations |

The flexibility score is family-relative and oriented so larger means weaker nominal regularization. It is not assumed to be metrically equivalent across algorithms. LR has no structural architecture tier in this feature space; its capacity path is induced by regularization and is therefore the clearest remaining capacity/regularization confound.

## Capacity operationalization

Every fitted model exports raw variables before any common index is formed:

| Family | Primary learned capacity proxy | Additional raw variables |
|---|---|---|
| Logistic Regression | Approximate ridge-logistic effective degrees of freedom, `tr[(X'WX+lambda I)^-1 X'WX]` | coefficient/intercept count, iterations, convergence, `C` |
| RBF SVM | Effective rank of the fitted RBF kernel on a deterministic first-512 training subsample | fitted gamma, support-vector fraction, support/dual storage count, `C` |
| Random Forest | Total learned leaves | mean tree depth, total nodes, tree count, leaf minimum |
| HistGradientBoosting | Total learned leaves | learned stages, total nodes, configured leaves, L2, learning rate |
| MLP | Learned parameter count | architecture, iterations, convergence, alpha |

The common `empirical_capacity_index` is the percentile rank of the primary proxy **within family and rho_train**, pooled over configurations and seeds. It uses no performance outcome. This index is an approximate matching coordinate, not a universal unit of functional capacity.

### Design calibration result available at freeze

Training-only calibration used 1,500 samples, `rho={0,0.9}`, and seeds `{0,1}`. All five families had strictly increasing mean primary proxies from declared low to medium to high. Approximate means were:

- LR effective df: 3.70, 4.62, 4.92.
- SVM kernel effective rank: 2.29, 10.17, 143.19.
- RF total leaves: 2,811, 8,020, 20,489.
- HGB total leaves: 680, 4,277, 8,278.
- MLP parameters: 113, 4,609, 25,601.

The calibration artifact contains no accuracy, loss, ranking, or OOD outcome. After this freeze, capacity settings may change only to repair an implementation error that invalidates the run; such a repair requires a new schema version and a documented preregistration amendment before rerunning.

## Primary outcomes and result grain

Primary fit-level outcome:

`OOD_gap = IID_test_accuracy - mean(accuracy over the frozen OOD environments)`.

Secondary outcomes: average OOD accuracy, worst OOD accuracy, environment-specific degradation, balanced accuracy, ROC-AUC, Brier score, and log loss.

`results/phase1/master_results.csv` is long at:

`experiment_id × run_id × fit_id × config_id × rho_train × seed × environment_type × rho_environment`.

Each successful fit contributes one independent IID-test row and four or five OOD rows. Fit-level IID accuracy, average/worst OOD accuracy, OOD gap, capacity variables, configuration hash, hyperparameter JSON, timing, warnings, and status are repeated for traceability. Failed fits contribute an explicit failure row and are never silently dropped. Deterministic `fit_id` and `row_id` hashes are based on scientific cell identity; `run_id` includes experiment, mode, and canonical configuration hash.

## Reproducibility metadata

The run records UTC start/finish, Git commit and branch, dirty-tree porcelain, file and canonical configuration hashes, hashes of all `src/phase1_*.py` files, Python/platform/processor/logical CPU count, package versions, fit/evaluation durations, warnings, failures, row counts, and final master-result SHA-256. Partial checkpoints make the formal run resumable without changing cell IDs.

## Smoke test and stop gates

Smoke uses all 15 configurations, seed 0, `rho_train={0,0.9}`, OOD `{0,-0.9}`, and reduced samples 700/350/500. It must pass before the formal run.

Stop conditions:

1. Registry is not exactly five families × three unique configurations.
2. Any fit fails, any primary metric is non-finite/out of bounds, any ID/grain key duplicates, or expected rows/cells are missing.
3. Maximum realized shortcut-correlation error exceeds 0.12; class balance leaves `[0.40,0.60]`; or absolute `corr(z,y)` exceeds 0.12.
4. OOD/test inputs become reachable from the fitting API.
5. Capacity proxies are non-finite or their smoke tier means are not strictly ordered within a family.
6. IID-derived and environment-derived OOD gaps fail exact reconstruction within `1e-12`.
7. The smoke wrong-line probe at neutral rho violates the preregistered fidelity gate described below.

A stopped smoke run writes its failures and artifacts but does not authorize the formal grid.

## Primary analysis 1 — balanced variance decomposition

Analysis uses one fit-level OOD gap per `family × configuration × rho_train × seed` cell (600 planned observations). Because configurations are nested in family, a crossed “family × same configuration” interpretation is forbidden.

For the balanced design, sum-to-zero components are computed from marginal means in this order-independent orthogonal form:

- family;
- rho_train;
- family × rho_train;
- configuration nested within family;
- nested configuration × rho_train;
- paired seed;
- residual, including unmodeled seed interactions.

Each component sum of squares and its fraction of total corrected sum of squares is reported. A seed-cluster bootstrap with 5,000 resamples supplies 95% intervals. The decomposition is descriptive; it is not a random-effects population estimate over a sampled universe of algorithms.

## Primary analysis 2 — regression control

OLS models use one fit-level OOD gap. Categorical variables use treatment-independent one-hot coding; reported R² comparisons are invariant to reference choice.

- M0: `gap ~ rho_train + paired seed`.
- M1 capacity baseline: M0 + IID test accuracy + empirical capacity index + nominal regularization flexibility.
- M2 family residual: M1 + family.
- M3 regime dependence: M2 + family × rho_train.
- Sensitivity S1: repeat M1–M3 without IID accuracy because it may mediate shortcut use.
- Sensitivity S2: replace empirical index with declared capacity tier.
- Sensitivity S3: omit nominal regularization flexibility because it is only family-relative.

Primary family residual is `R²(M2)-R²(M1)`. Primary regime dependence is `R²(M3)-R²(M2)`. Report in-sample partial R², seed-cluster bootstrap intervals, and leave-one-seed-out cross-validated R²/RMSE. Family predictive value is also the relative reduction in cross-validated RMSE from M1 to M2. Leave-one-rho-out and leave-one-family-out estimates are sensitivity diagnostics, not replacements for the omnibus primary comparison.

## Primary analysis 3 — capacity-matched/stratified comparison

Exact strata are `rho_train × declared capacity tier`. Within every family pair and stratum, nearest-neighbor matching with replacement uses standardized empirical capacity index, IID test accuracy, and nominal regularization flexibility. A match must be within 0.25 pooled SD on empirical capacity and IID accuracy and within 0.5 absolute units on flexibility. Report coverage, standardized differences, paired gap differences, and seed-cluster bootstrap intervals.

If fewer than 50% of eligible observations receive matches for a family pair, that pair is labelled underpowered rather than relaxed post hoc. The prespecified fallback is a stratified regression containing rho, capacity-tier strata, IID accuracy, flexibility, seed, and family; both the failed coverage and fallback result must be reported.

The matched/stratified analysis cannot certify causal exchangeability. Agreement means only that the family residual is not obviously removed by these measured proxies.

## Secondary analyses

### Robustness envelope

For each family and rho, report the minimum, median, maximum, and 10th/90th percentile of configuration-seed OOD gap and OOD accuracy. The envelope shows overlap and whether a claimed family advantage is smaller than within-family configuration spread.

### IID/OOD ranking reversal

Within each rho/seed, identify tie-aware best sets using an absolute accuracy tolerance of 0.001. Report the probability that IID-best and average-OOD-best configuration sets are disjoint, both at configuration and family level, plus all pairwise ordering probabilities. No winner is selected by OOD for training.

### Noise-free tabular wrong-line guard

This is a mandatory mechanism audit, not a new shift family. Train with zero label noise over:

- `rho_train={0,0.5,0.9}`;
- stable geometries baseline `(1.2,-1,0.8)`, axis-aligned `(1,0,0)`, and equal-oblique `(1,1,1)`;
- all 15 configurations and seeds `{0,1,2}`.

Geometry vectors are normalized to equal L2 norm. For each fitted model, evaluate a 4,000-point noise-free probe with `x_spurious=0` and `z=0`; boundary fidelity is agreement with the exact stable oracle. At neutral rho, every **high-capacity family mean across geometries and seeds** must be at least 0.95. Low/medium tiers remain fully reported but are not a stop gate because approximation error is an intended capacity manipulation. A high-tier violation means the main family residual may reflect basic stable-boundary fitting failure (“wrong line”) and overrides G4a to `REVISE HYPOTHESIS`. Results at positive rho are diagnostic and show whether shortcut strength rotates or distorts the learned stable boundary differently by geometry/capacity.

## Missing data, warnings, and failure policy

- No imputation of primary outcomes.
- Any failure remains in the fit/status ledger with type, message, traceback, and time.
- Formal analysis requires at least 95% successful fits overall and at least 8 of 10 seeds in every family × config × rho cell. Otherwise no G4a “PROCEED” outcome is permitted.
- Convergence warnings are reported. A model reaching its iteration limit is not silently relabelled successful; sensitivity tables exclude non-converged fits while the primary intention-to-run table retains them if predictions are finite.
- Implementation errors require code repair, a new source hash, invalidation of the affected run, and full deterministic rerun of affected cells. Statistical “bad results” never justify retries or grid changes.

## Figure contract

Five publication figures, each in PNG and PDF:

1. **Variance decomposition:** component shares with bootstrap intervals; direct labels and ordered components.
2. **Matched capacity:** family residual gap estimates by rho/capacity tier, with coverage encoded separately from color.
3. **Robustness envelope:** configuration-seed OOD-gap bands by family and rho; family shown by color and line style.
4. **IID/OOD reversal:** paired IID-versus-OOD ranking/probability view, with tie rule in caption.
5. **Wrong-line guard:** geometry × rho small multiples showing boundary fidelity by capacity tier and family, with the 0.95 neutral-rho gate.

All axes show metric units, uncertainty definitions, sample counts, and accessible non-color encodings. Every PNG is visually inspected at full size; source tables are reconciled to plotted values.

## G4a decision rule

Let `F_cv` be leave-one-seed-out cross-validated incremental R² from M1 to M2; `F_rmse` the relative CV RMSE reduction; `F_boot` the seed-bootstrap interval for in-sample incremental R²; and `I_cv` the incremental CV R² from family × rho interaction (M2 to M3).

Practical thresholds frozen before outcomes:

- meaningful family variance: 0.05 incremental R²;
- meaningful predictive improvement: 5% relative CV RMSE reduction;
- meaningful accuracy/gap contrast: 0.02 absolute;
- stable matched evidence: stratified/matched family partial R² at least 0.05 or at least one family contrast of 0.02 with bootstrap interval excluding zero, with adequate coverage.

### G4a-A — family mostly acts as a proxy

Classify A when `F_cv < 0.05`, `F_rmse < 5%`, the bootstrap lower bound does not establish a positive family increment, and matched/stratified analysis finds neither partial R² ≥0.05 nor an adequately covered ≥0.02 family contrast with interval excluding zero. Final decision: **DO NOT PROCEED** to Phase 2.

### G4a-B — stable meaningful family residual

Classify B only when all hold:

1. `F_cv ≥ 0.05` and `F_rmse ≥ 5%`;
2. the bootstrap lower bound for the in-sample family increment is above zero;
3. every leave-one-seed estimate is positive and every leave-one-rho `F_cv` is at least 0.03;
4. matched/stratified evidence reaches the practical criterion with adequate coverage;
5. `I_cv < F_cv`, so regime interaction does not dominate the stable family component;
6. smoke, data-quality, completeness, and wrong-line gates all pass.

Final decision: **PROCEED** and write a Phase 2 mechanism-isolation design, but do not run Phase 2.

### G4a-C — regime-dependent or method-dependent residual

Classify C when A and B are both false and at least one of the following holds: `I_cv ≥ F_cv`; leave-one-rho estimates fail the 0.03 stability threshold; family rankings materially change by rho; a ≥0.02 contrast appears only in some regimes; or regression and matched/stratified analyses disagree. Final decision: **REVISE HYPOTHESIS** toward explicit family × shift-regime or configuration interactions. Phase 2 is not authorized under C.

Any primary stop-gate or completeness failure also yields **REVISE HYPOTHESIS** rather than a substantive A/B claim.

## Claim boundary

Permitted claims concern this exact synthetic generator, five families, frozen 15 configurations, ten seeds, and stated capacity proxies. The study cannot establish intrinsic robustness, a universal family ranking, causal learned bias, real-world validity, or equivalence of cross-family capacity. The mandatory final line must be exactly one of:

- `PROCEED`
- `DO NOT PROCEED`
- `REVISE HYPOTHESIS`
