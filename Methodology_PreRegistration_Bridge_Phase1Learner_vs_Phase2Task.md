# Methodology Pre-Registration — Bridge: Phase-1 MLP-L Learner Bundle on the Frozen Phase-2 Task

**Project:** https://github.com/12412825-collab/thesis-track  
**Experiment ID:** `BRIDGE-P1MLP-L-ON-P2TASK`  
**Status:** SUPERVISOR-APPROVED FOR FREEZE; NOT YET RUN  
**Role:** phenomenon-existence / identifiability bridge, not a mechanism decomposition, not Phase 3  
**One-sentence design:** We replace only the learner bundle while freezing the task.

---

## 1. Executive Rationale

Phase 2 showed that, under the frozen high-shortcut generator, a NumPy one-hidden-layer SGD MLP can have:

- high shortcut-sign sensitivity and large OOD gap, and
- high neutral-slice `Fidelity(200)` at the same time.

Q1/Q2/Q3 were therefore a genuine null on **Fidelity**, not a demonstration that shortcut learning vanished.

Phase 1’s striking low-fidelity cell (0.72225) was measured on a **different training object** (noise-free labels, axis-aligned geometry, `n_train=2500`). It is not a replication target for this Bridge.

The remaining scientific uncertainty is narrower:

> On the **same Phase-2 task realization**, does the **Phase-1 MLP-L complete pipeline** recreate a material drop in neutral-slice Fidelity relative to the frozen Phase-2 reference learner?

This is a **learner-bundle intervention**. It does not isolate Adam, early stopping, scaling, or width.

If Fidelity drops: the collapse is protocol-conditional; splitting the bundle is allowed later.  
If Fidelity stays high while D3/OOD stay high: Fidelity collapse is the wrong core Behavior for this baseline task; stop that line.

---

## 2. Phase-1 vs Phase-2 Forensic Delta Table

Sources: `src/phase1_experiment.py`, `src/phase1_models.py`, `src/phase1_wrong_line.py`, `experiments/PHASE-1/config.yaml`, `src/phase2_experiment.py`, `experiments/PHASE-2/config.yaml`, `src/verification_data.py`, `src/data_generation.py`, `results/phase1/summaries/run_metadata.json`, `results/phase1/summaries/wrong_line_guard.csv`, `results/phase2/raw_fit_results.csv`.

### 2.1 Task side

| Item | Phase 1 main OOD | Phase 1 wrong-line diagnostic | Phase 2 formal | Bridge freeze |
|---|---|---|---|---|
| Generator | `generate_verification_dataset` | same | same | **Phase 2** `build_phase2_bundle` |
| Master seed | `20260818` | same | same | same |
| Dataset-seed mapping | `child_seed(master, seed, 1/2/100/…)` | `…, geometry_index, 8000/8100` | `child_seed(master, dataset_seed, 1/2/100/9000)` | **Phase 2 mapping** |
| Overlapping-seed train draws | seeds 0–9 match Phase 2 | **no** | seeds 0–19 | Phase 2 seeds 0–19 |
| Stable rule | `score = 1.2 x1 − x2 + 0.8 x3 + N(0,σ); y=1{score>0}` | σ = **0** | σ = **0.5** | **0.5** |
| Stable coefficients | `[1.2, -1.0, 0.8]` | three geometries, L2-normalized | `[1.2, -1.0, 0.8]` | **baseline only** |
| Shortcut | `ρ(2y−1)+√(1−ρ²)ε` | same formula | same | same |
| Rho semantics | requested coefficient, not finite-sample corr | same | same | same |
| `rho_train` | `{0, 0.5, 0.7, 0.9}` | `{0, 0.5, 0.9}` | `{0.7, 0.9, 0}` | **`{0.9, 0}` only** |
| Label noise | 0.5 | **0.0** | 0.5 | **0.5** |
| Geometry | baseline | baseline / axis-aligned / equal-oblique | baseline | **baseline** |
| `z` | independent N(0,1) | 0 on probe | same as main; 0 on probe | Phase 2 |
| Feature order | `[x1,x2,x3,x_spurious,z]` | same | same array order | same |
| `n_train / val / iid / ood / probe` | 5000 / 2000 / 3000 / 3000 / — | **2500 / — / — / — / 4000** | 5000 / 2000 / 3000 / 3000 / 4000 | **Phase 2 sizes** |
| OOD grid | `{0.3,0,-0.3,-0.6,-0.9}` minus IID duplicate | none | same as Phase 1 main | **Phase 2** |
| Class balance | generator-checked ~0.5 | ~0.5, noise-free | same generator | Phase 2 |
| Task-level scaling | none (raw features) | none | none | **none** |
| Learner-internal scaling | MLP pipeline `StandardScaler` | same | **none** | **bundle-specific** (see §7–8) |

**Identifiability:** a learner-only contrast is possible because `build_phase2_bundle` can be shared. The 0.72225 cell is **not** this task.

### 2.2 Learner side

#### Phase-1 MLP-L (registry `MLP-L`)

From `experiments/PHASE-1/config.yaml` and `build_phase1_model` in `src/phase1_models.py`. Frozen Phase-1 environment: `scikit-learn==1.9.0` (`results/phase1/summaries/run_metadata.json`).

| Item | Explicit in repo? | Frozen Bridge value |
|---|---|---|
| Implementation | yes | `sklearn.pipeline.Pipeline` |
| Scaler | yes | `StandardScaler()` with **no extra kwargs**; fit on **training features only** inside `Pipeline.fit` |
| Estimator | yes | `sklearn.neural_network.MLPClassifier` |
| `hidden_layer_sizes` | yes | `(16,)` |
| `alpha` | yes | `0.0001` |
| `learning_rate_init` | yes | `0.001` |
| `max_iter` | yes | `250` |
| `early_stopping` | yes | `True` |
| `random_state` | yes | Bridge init seed (see §9); passed as `random_state` |
| `solver` | **not passed** | sklearn default under frozen `scikit-learn 1.9.0` |
| `activation` | **not passed** | sklearn default under frozen `scikit-learn 1.9.0` |
| `learning_rate` (schedule name) | **not passed** | sklearn default under frozen `scikit-learn 1.9.0` |
| `batch_size` | **not passed** | sklearn default under frozen `scikit-learn 1.9.0` |
| `validation_fraction` | **not passed** | sklearn default under frozen `scikit-learn 1.9.0` |
| `n_iter_no_change` | **not passed** | sklearn default under frozen `scikit-learn 1.9.0` |
| `tol` | **not passed** | sklearn default under frozen `scikit-learn 1.9.0` |
| `beta_1` | **not passed** | sklearn default under frozen `scikit-learn 1.9.0` |
| `beta_2` | **not passed** | sklearn default under frozen `scikit-learn 1.9.0` |
| `epsilon` | **not passed** | sklearn default under frozen `scikit-learn 1.9.0` |
| `shuffle`, `momentum`, `nesterovs_momentum`, `power_t`, `warm_start`, etc. | **not passed** | sklearn default under frozen `scikit-learn 1.9.0` |
| Capacity tier (registry label) | yes | `low` (score 0.0) |
| Regularization tier (registry label) | yes | `weak` / flexibility 1.0 |

Do not invent numeric defaults in the paper or in Codex comments. Re-instantiate the constructor **exactly** as `src/phase1_models.py` does, under a recorded `scikit-learn` version that must match Phase 1 (`1.9.0`) or abort.

#### Phase-2 reference learner (canonical arm)

From `experiments/PHASE-2/config.yaml` and `OneHiddenLayerMLP` in `src/phase2_experiment.py`.

| Item | Frozen value |
|---|---|
| Implementation | project NumPy `OneHiddenLayerMLP` |
| Scaler | none; raw 5-D features |
| Hidden layers | one |
| Width | **64** (single reference arm; not a sweep) |
| Activation | ReLU (`np.maximum(·, 0)`) |
| Optimizer | SGD + momentum `0.9` |
| Weight decay | `1e-4` |
| Initial LR | `0.01` |
| Schedule | **constant** (no cosine) |
| Epochs | 200 |
| Batch size | 64 |
| Early stopping | false |
| Init | He: `N(0, √(2/fan_in))`, paired `init_seed` |
| Loss | binary logistic (stable BCE) |
| Predict | `logit ≥ 0` |

This arm already exists as frozen Phase-2 cells `rho{0\|0p9}_seed{00–19}_width64_constant`.

---

## 3. Exact Source of the Phase-1 0.7223 Fidelity Cell

The published “0.7223” is **0.72225** in `results/phase1/summaries/wrong_line_guard.csv`.

| Field | Value |
|---|---|
| File | `results/phase1/summaries/wrong_line_guard.csv` row 177 (0-based file order as exported) |
| `boundary_fidelity` | **0.72225** |
| `config_id` | `MLP-L` |
| `family` | `mlp` |
| `capacity_tier` | `low` |
| `capacity_tier_score` | 0.0 |
| Registry regularization | weak / flexibility 1.0 (`alpha=1e-4`) |
| Learner | Phase-1 MLP-L pipeline (scaler + `MLPClassifier`) |
| `rho_train` | **0.9** |
| `seed` | **0** |
| `geometry` | **`axis_aligned`** |
| `stable_coefficients` | `[1.7549928774784245, 0.0, 0.0]` (L2-normalized from `[1,0,0]`) |
| Label noise **at training** | **0.0** (`wrong_line_guard.label_noise`) |
| `n_train` | **2500** |
| Probe | 4000 noise-free points, `rho=0`, `x_spurious=0`, `z=0`, RNG `child_seed(master, seed, geometry_index, 8100)` |
| Train RNG | `child_seed(master, seed, geometry_index, 8000)` — **not** the Phase-2 train RNG |

Same MLP-L, **baseline** geometry, `rho=0.9`, wrong-line protocol: fidelities **0.97225 / 0.95300 / 0.91175** (seeds 0/1/2). Those cells are already close to Phase-2 Fidelity.

**Comparable to Phase-2 noisy-label + baseline geometry? No.**

The 0.72225 cell differs in geometry, label noise, sample size, and data RNG. This Bridge **must not** treat 0.72225 as the quantity to recover. The Bridge asks whether MLP-L changes Fidelity on the **Phase-2 task**, not whether the axis-aligned noise-free cell replicates.

---

## 4. Scientific Question

On frozen Phase-2 task realizations (`label_noise=0.5`, baseline coefficients, Phase-2 sizes, Phase-2 probe, Phase-2 OOD, seeds 0–19):

> Does replacing the Phase-2 reference learner (`width=64`, constant SGD) with the Phase-1 MLP-L complete pipeline produce a stable, practically meaningful drop in terminal neutral-slice Fidelity?

Auxiliary (not primary): do D3 and OOD gap remain large under that same swap?

---

## 5. Competing Explanations A/B

### Explanation A — Learner-protocol-dependent Fidelity collapse

MLP-L on the frozen Phase-2 task yields substantially lower Fidelity than the Phase-2 reference, while rho=0 does not show a matching collapse.

**Allowed claim:** the existence of Fidelity collapse is conditional on the **learner/training-protocol bundle**.  
**Forbidden claim:** “Adam caused it.” The bundle is not split.

### Explanation B — Behavior mismatch

MLP-L Fidelity remains high on the same task, while D3 and OOD gap remain large (shortcut use / OOD failure still present).

**Allowed claim:** neutral-slice Fidelity collapse is not the appropriate core Behavior for this baseline shortcut task.  
**Action:** stop the Fidelity-collapse mechanism line. Subsequent work, if any, must target shortcut acquisition / OOD failure, under a new preregistration.

---

## 6. Frozen Task Components

Reuse `build_phase2_bundle` in `src/phase2_experiment.py` with `experiments/PHASE-2/config.yaml` data block, except `rho_train_values` restricted to `{0.9, 0.0}`.

Frozen:

- generator, master seed `20260818`, dataset seeds `0–19`
- stable coefficients `[1.2, -1.0, 0.8]`, `label_noise=0.5`
- shortcut formula, rho semantics, feature order, `z`
- `n_train=5000`, `n_validation=2000`, `n_iid_test=3000`, `n_ood_test=3000`, `n_probe=4000`
- OOD `{0.3, 0.0, -0.3, -0.6, -0.9}` excluding the IID duplicate
- probe: noise-free, `rho=0`, `x_spurious=0`, `z=0`, RNG `child_seed(master, dataset_seed, 9000)`

**Not used:** axis-aligned / equal-oblique geometries; wrong-line `n_train=2500`; `label_noise=0`; `rho=0.7`.

**Why rho=0.7 is omitted:** this is a phenomenon-existence test (does MLP-L drop Fidelity at the high-shortcut regime that produced Phase-2 D3/OOD?), plus a rho=0 specificity check. A third rho only redraws a degradation curve and does not separate A vs B.

---

## 7. Exact Repo-Derived Phase-1 MLP-L Learner Bundle

The Bridge learner is **one object**:

```text
Pipeline([
  ("scale", StandardScaler()),
  ("model", MLPClassifier(
       hidden_layer_sizes=(16,),
       alpha=0.0001,
       learning_rate_init=0.001,
       max_iter=250,
       early_stopping=True,
       random_state=<bridge_init_seed>,
  )),
])
```

Call site must remain `src.phase1_models.build_phase1_model` on the frozen `MLP-L` registry row, then `fit` on `bundle.train` features `["x1","x2","x3","x_spurious","z"]` and labels only. **No OOD, probe, or test data in `fit`.**

`scikit-learn` version: **1.9.0** (Phase-1 recorded). If the runtime version differs, stop and do not analyze.

Init seed (new, frozen, no leakage):

`bridge_init_seed = child_seed(master_seed, dataset_seed, rho_code, 9100)`

with `rho_code = int(round((rho+1)*1000))` as in Phase 2. This seed is **not** the Phase-2 `init_seed` (that includes width and `7100`).

Terminal state for MLP-L: sklearn’s fitted state after early stopping or `max_iter`, whichever the estimator actually uses. That is the protocol terminal. Do not force 200 epochs.

---

## 8. Phase-2 Reference Learner

Single frozen comparator:

- `width=64`, `schedule=constant`, 200 epochs, SGD+momentum, no scaler, no ES.

Reuse already-archived Phase-2 terminals for

`rho0p9_seed{00-19}_width64_constant` and `rho0_seed{00-19}_width64_constant`

from `results/phase2/fits/` **without refitting**, provided file SHA-256 still matches `results/phase2/manifest.json`. If a hash mismatches, abort (do not silently refit).

Do not average over widths or schedules. Those other arms are out of scope.

---

## 9. Experimental Matrix

| Factor | Levels |
|---|---|
| `rho_train` | `{0.9, 0.0}` |
| `dataset_seed` | `{0,…,19}` |
| Learner | `{phase2_ref = width64-constant, p1_mlp_l}` |

New fits: **40** (MLP-L × 20 seeds × 2 rho).  
Reference fits: **40** reused Phase-2 files.  
No other cells.

Pairing key: `(rho_train, dataset_seed)` — identical `Phase2Bundle`.

---

## 10. Fidelity / D3 / OOD Definitions

All three are computed for **both** learners on the **same** bundle, using **Phase-2 metric code paths** (not EXP-002V conditional D3, not wrong-line-only geometry).

### Fidelity (terminal)

Agreement of `predict` with `bundle.probe_y` on `probe_X` (`x_spurious=0`, `z=0`, 4000 noise-free points).  
Phase-2 ref: `Fidelity(200)`.  
MLP-L: fidelity at sklearn terminal.  
Denote both `F`.

### D3 (terminal)

Phase-2 definition: IID-test sign-flip of `x_spurious` (column 3 in the ordered array); D3 = fraction of prediction changes.  
Threshold 0.5 is **descriptive only** here (SAT is not a Bridge decision variable).

### OOD gap (terminal)

`IID_test_accuracy − mean(accuracy on frozen OOD environments for that rho)`, same environments as Phase 2.

Report all three. Do not promote D3 or OOD to primary after seeing results.

---

## 11. Primary Discriminand

At `rho_train=0.9`, for each seed `s`:

`ΔF_s = F_s(phase2_ref) − F_s(p1_mlp_l)`

**Primary:** mean paired `ΔF`, seed-cluster bootstrap 95% CI (5000 resamples), and the count of seeds with `ΔF_s > 0`.

Secondary (same pairing, same rho): `ΔD3_s`, `ΔOOD_s`.

Sanity: the same quantities at `rho_train=0`.

---

## 12. Paired Statistical Protocol

- One row per `rho × seed × learner`.
- Estimator: paired mean difference across 20 seeds.
- Uncertainty: seed bootstrap, 5000 resamples, percentile 95% CI.
- Effect size: paired mean `ΔF` and the fraction of seeds with `ΔF > 0.02` and `> 0.05`.
- No mixed models, no extra covariates, no OOD-based model selection.

Practical gates (same numbers as Phase 2, still conventions not laws):

- negligible: `0.02`
- practical: `0.05`

---

## 13. Decision Rules

Apply **after** guards pass (finite metrics, 40/40 new fits, leakage API holds, sklearn version match, Phase-2 reference hashes match).

### Case A — LEARNER-PROTOCOL-DEPENDENT FIDELITY COLLAPSE

At `rho=0.9`:

1. mean `ΔF ≥ 0.05`;
2. bootstrap CI for mean `ΔF` excludes 0;
3. at least 15/20 seeds have `ΔF ≥ 0.02`.

And Case C does **not** hold.

**Implication:** Fidelity collapse can be manufactured by the Phase-1 learner bundle on this task. **Do not split the bundle in this experiment.** A later, separately preregistered one-axis split is allowed.

### Case B — FIDELITY IS NOT THE APPROPRIATE CORE BEHAVIOR FOR THIS BASELINE TASK

At `rho=0.9`:

1. mean `|ΔF| < 0.02` **or** the CI includes 0 **and** `|mean ΔF| < 0.05`;
2. MLP-L mean terminal D3 `≥ 0.5`;
3. MLP-L mean OOD gap `≥ 0.30` (still a large shortcut-shift failure, comparable to Phase 2’s ~0.44).

**Implication:** stop the Fidelity-collapse mechanism line. Do not start Phase 3. Any next study must preregister shortcut-use / OOD as Behavior, not Fidelity collapse.

### Case C — UNRESOLVED (not shortcut-specific)

At `rho=0`, mean `ΔF ≥ 0.05` and CI excludes 0 (MLP-L also collapses without the shortcut).

**Implication:** do not interpret as shortcut-induced Fidelity collapse. Do not split the bundle. Supervisor review; no automatic follow-up grid.

### Case D — BRIDGE NOT IDENTIFIABLE

Any of: bundle reconstruction fails; feature-order mismatch; probe/OOD leakage into `fit`; sklearn version mismatch; Phase-2 reference hash failure; inability to share `Phase2Bundle` across learners.

**Implication:** do not analyze as A/B. Repair identity, do not expand factors.

### Intermediate `ΔF` (0.02 ≤ mean ΔF < 0.05 at rho=0.9, Case C false)

**UNRESOLVED — small protocol effect, not practical collapse.** Do not split the bundle. Do not declare Case B if D3/OOD still large; report the interval and stop automatic follow-ups.

---

## 14. Falsification / Retreat Rules

- Stop if any new fit fails.
- Stop if MLP-L `fit` can receive probe/OOD/test (API test required).
- Do not add width, LR, scaler-off, ES-off, Adam-vs-SGD, geometry, rho=0.7, or a second family to “rescue” a null.
- Do not change primary to OOD/D3 after seeing Fidelity.
- Do not cite 0.72225 as a failed replication of this Bridge.
- Case A forbids the sentence “Adam caused Fidelity collapse.”

---

## 15. What This Bridge Can Claim

- Whether the Phase-1 MLP-L **bundle**, on the frozen Phase-2 **task**, changes terminal Fidelity relative to the Phase-2 width-64 constant learner.
- Whether D3 and OOD gap remain coupled or decoupled from that Fidelity change.
- A binary-ish scientific routing: continue Fidelity-collapse work (later split) vs stop that line.

---

## 16. What It Cannot Claim

- That Adam, early stopping, `StandardScaler`, width 16, or sklearn internals is the operative cause (bundle not split).
- That Phase 2 Q1/Q2/Q3 were invalid.
- That 0.72225 should have reappeared.
- Family-level robustness, optimizer rankings, or AGI-scale generality.
- That shortcut learning is absent if Fidelity is high.

---

## 17. Anti-Benchmark Checklist

| Item | This design |
|---|---|
| One generator | yes, Phase 2 |
| One family | MLP only |
| One geometry | baseline |
| One new learner | MLP-L bundle |
| One Phase-2 reference arm | width 64, constant |
| No new dataset/shift | yes |
| No optimizer factorial | yes |
| No scaler/ES/width/alpha/LR sweep | yes |
| Primary chosen before results | `ΔF` at rho=0.9 |
| Sentence test | **We replace only the learner bundle while freezing the task.** |

---

## 18. Exact Minimal Codex Specification

1. Do not start Phase 3. Do not write unused model registries.
2. Pin `scikit-learn==1.9.0` for this run; abort otherwise.
3. Import `build_phase2_bundle` from `src/phase2_experiment.py`. Generate one bundle per `(rho, seed)` in `{0.9, 0.0} × {0…19}`.
4. For each bundle, fit **only** `MLP-L` via `src.phase1_models.build_phase1_model` + `fit_phase1_model` / equivalent `Pipeline.fit` on train features+labels. Init seed `child_seed(20260818, seed, rho_code, 9100)`.
5. Evaluate MLP-L with the **same** Fidelity, sign-flip D3, and OOD-gap functions used for Phase 2 terminals (probe/IID/OOD from that bundle).
6. Load Phase-2 reference terminals from existing `width64_constant` fit JSON; verify manifest hashes; do not refit.
7. Write paired tables: `ΔF`, `ΔD3`, `ΔOOD` by rho and seed; bootstrap CIs; apply §13 with no discretion.
8. Tests: 40 new successes; no leakage; sklearn version; hash check; exact MLP-L constructor kwargs listed in §7.
9. Forbidden: any second architecture, cosine arm, width 8/512, rho=0.7, geometry loop, scaler ablation, ES ablation, Adam-vs-SGD, new plots that rank learners as a leaderboard.
10. Output namespace: `results/bridge_p1mlp_l/` plus a short results memo after analysis — analysis is a later task; this freeze is design-only.

**Phase 3 / model selection: not started.**

---

## Freeze declaration

This preregistration was written from repository code and archived results, before any Bridge outcome. After it is committed and hashed, Codex may implement **only** the specification in §18.

Suggested commit message:

`method: freeze bridge preregistration for Phase-1 MLP-L on Phase-2 task`
