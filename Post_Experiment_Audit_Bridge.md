# Post-Experiment Audit — Phase-1 MLP-L on Frozen Phase-2 Task

Audit status: **PASS**
Bridge preregistration SHA: `664d02a9f1c928713169866b2f5b3225b3554435`
Bridge preregistration tag: `bridge-phase1learner-phase2task-prereg-frozen`
Phase-2 task/reference preregistration SHA: `654e72b39310847b0d369796c31825e379aaf909`
Formal output: `results/bridge_p1mlp_l/`
Phase 3: **NOT RUN**

## 1. Scope and data sources

This audit covers only the approved Bridge. It does not reopen Phase 2, alter any preregistration, or add a benchmark factor.

- New fit records: 40 JSON files under `results/bridge_p1mlp_l/fits/`.
- Frozen reference records: 40 Phase-2 `width64_constant` JSON files under `results/phase2/fits/`.
- Paired table: `results/bridge_p1mlp_l/paired_results.csv`.
- Task content hashes: `results/bridge_p1mlp_l/task_hashes.csv`.
- Reference hash checks: `results/bridge_p1mlp_l/reference_hashes.csv`.
- Runtime: scikit-learn `1.9.0`, exactly as required by the frozen Bridge protocol.

## 2. Protocol and integrity checks

| Check | Result | Evidence |
|---|---|---|
| New fit matrix | PASS | exactly 20 seeds × rho `{0.9, 0.0}` = 40 |
| New fit status | PASS | 40 successful, 0 failed |
| Extra fits | PASS | exactly 40 Bridge fit JSON files; no rho=0.7, width, schedule, or optimizer sweep |
| Reference reuse | PASS | width 64, constant; 40/40 Phase-2 files reused without refitting |
| Reference hashes | PASS | 40/40 SHA-256 and byte-size checks match `results/phase2/manifest.json` |
| Task construction | PASS | implementation imports and calls `build_phase2_bundle`; no duplicate generator |
| Seed pairing | PASS | dataset seeds 0–19 for both rho values; stable features/labels/probe pair exactly across rho sanity builds |
| Task hash coverage | PASS | 340 environment/probe hashes, matching 20 seeds × (9 components at rho=0.9 + 8 at rho=0) |
| Feature order | PASS | `x1, x2, x3, x_spurious, z` in fit and evaluation |
| sklearn version | PASS | observed `1.9.0` |
| MLP-L extraction | PASS | exact Phase-1 registry row and `build_phase1_model` call path |
| Scaler leakage | PASS | `StandardScaler` is fitted inside the pipeline on training features only; leakage test passes |
| Metric finiteness | PASS | all Fidelity, D3, IID accuracy, and OOD-gap values finite |
| Early stopping | PASS | all 40 fits used `early_stopping=True` and stopped before `max_iter=250`; no convergence warnings |

The Phase-2 reference JSON files predate the Bridge task-hash artifact and do not contain task-content hashes themselves. Their identity is therefore established by the frozen Phase-2 manifest hash, exact reference metadata, frozen generator/config call path, and paired `(rho_train, dataset_seed)` keys; no silent refit was used.

## 3. Learner reconstruction

`Phase1_MLPL_Protocol_Extraction.md` records the source-level extraction. The Bridge used the repository's `MLP-L` registry entry and `fit_phase1_model` with training features and labels only. The learner bundle was not split: no Adam-vs-SGD, scaler, early-stopping, width, alpha, or optimizer ablation was run.

## 4. Metric implementation

- Fidelity: agreement with the Phase-2 neutral probe (`x_spurious=0`, `z=0`).
- D3: Phase-2 IID-test prediction-change rate after flipping the ordered `x_spurious` column.
- OOD gap: IID accuracy minus mean accuracy over the frozen OOD environments for the same rho.
- Primary delta convention: `Delta_F = Fidelity(Phase-2 reference) - Fidelity(Phase-1 MLP-L)`, so a positive value means lower MLP-L Fidelity. This is the sign convention used in the preregistration's primary and decision sections.

## 5. Outlier and warning review

At rho=0.9, Fidelity deltas ranged from `-0.046` (seed 1) to `0.051` (seed 7), while the leave-one-seed-out mean ranged only from `0.00421` to `0.00932`. The D3 and OOD effects also remained positive after omitting any one seed. The primary result is therefore not driven by one seed. No fit emitted a convergence warning; all 40 MLP-L fits stopped early.

## 6. Decision audit

| Case | Rule check | Result |
|---|---|---|
| A — learner-protocol-dependent Fidelity collapse | rho=0.9 mean Delta_F ≥ 0.05, CI excludes 0, and ≥15/20 seeds ≥0.02 | **Not met** |
| B — Fidelity is not the appropriate core Behavior | rho=0.9 `abs(mean Delta_F)<0.02`, MLP-L mean D3 ≥0.5, MLP-L mean OOD gap ≥0.30 | **Met** |
| C — unresolved / not shortcut-specific | rho=0 mean Delta_F ≥0.05 and CI excludes 0 | **Not met** |
| D — protocol blocker | task/bundle/hash/version/leakage failure | **Not met** |

Frozen Bridge decision: **Case B — FIDELITY IS NOT THE APPROPRIATE CORE BEHAVIOR FOR THIS BASELINE TASK**.

## 7. Reviewer-2 verdict

**SUPPORTED**, narrowly for the preregistered Case-B routing. The evidence supports stopping the Fidelity-collapse mechanism line for this baseline task. It does not identify which internal component of the Phase-1 MLP-L bundle produced any observed difference.

## 8. Claims not permitted

The Bridge does not support claims that Adam caused collapse, that early stopping, scaling, width 16, or any sklearn default is the operative cause, that Phase 2 Q1/Q2/Q3 were invalid, that 0.72225 should have reappeared, or that the result generalizes across model families, optimizers, datasets, geometries, or shifts.
