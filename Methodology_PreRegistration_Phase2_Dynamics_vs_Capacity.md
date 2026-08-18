# Methodology Pre-Registration: Phase 2 — Training Regime and Boundary Distortion

**Project:** What Predicts Robustness Under Distribution Shift: Model Family, Capacity, or Learned Bias?  
**Repository:** https://github.com/12412825-collab/thesis-track  
**Status:** SUPERVISOR-APPROVED FOR FREEZE; NOT YET RUN  
**Scientific role:** mechanism-identification experiment, not a benchmark  
**Scope:** MLP only; existing spurious-correlation generator only; no new dataset/model family/shift type

---

## 0. Frozen Phase-1 basis

Phase 1 established the following working facts:

1. `rho_train` is the dominant observed task variable for OOD degradation.
2. A stable, additive model-family main effect was weakened and did not meet the preregistered practical threshold.
3. `family × rho` is statistically detectable but is a redirection signal, not a mechanism diagnosis.
4. Capacity/regularization were not cleanly identified in Phase 1 because cross-family overlap/matching was inadequate.
5. High-`rho` settings exhibit configuration-level boundary distortion / wrong-line behavior.
6. The main missing link is therefore **Mechanism → Behavior**, not another model-ranking table.

This Phase 2 does **not** attempt to recover a family-level ranking claim.

---

## 1. Scientific question

### Core question

Under a fixed high-shortcut task regime and a fixed model family (MLP), how do:

- **width / capacity-regime**, and
- **learning-rate time-profile**

jointly shape configuration-level boundary distortion?

The design no longer treats “dynamics” and “capacity” as two theoretically pure, independent causes. Instead it estimates three quantities:

### Q1 — Schedule residual within fixed width

At fixed width, does changing the LR time-profile change terminal boundary fidelity?

### Q2 — Width / capacity-regime residual

Within the fixed training protocol, does changing width systematically change terminal boundary fidelity?

### Q3 — Coupling / interaction

Does the schedule effect change in magnitude or direction across widths?

If Q3 dominates, the scientific conclusion is **not** that A or B “wins”. The candidate mechanism becomes a **coupled training regime**:

`Training Regime = (Width / Capacity-Regime, LR Time-Profile)`

---

## 2. T–M–B mapping

### T — Task structure

Training-side spurious correlation:

`rho_train ∈ {0.7, 0.9}`

with the existing generator, stable rule, shortcut construction, and OOD geometry held fixed.

`rho_train = 0` is used only as a sanity control and is not part of the headline claim.

### M — Candidate mechanism

A controllable **training regime** described by:

1. width / capacity-regime;
2. LR time-profile under a fixed optimizer class.

The two components are allowed to interact.

### B — Behavior

Configuration-level boundary distortion, measured primarily by terminal stable-boundary fidelity.

---

## 3. What this experiment is NOT

This experiment is not asking:

- Which model is most robust?
- Is cosine better than constant LR?
- Is one MLP configuration the “winner”?
- Which optimizer gives the highest OOD accuracy?
- Which model family should be ranked first?

No leaderboard-style conclusion is permitted.

---

## 4. Why MLP only

MLP is intentionally the only model family in Phase 2.

Reasons:

1. The candidate mechanism concerns gradient-based feature acquisition over training time.
2. MLP provides a continuous train-time trajectory in which shortcut acquisition and later boundary recovery can be measured.
3. Adding RF/SVM/LR/HGB would reintroduce family comparison without a common feature-learning trajectory.
4. The experiment does not make a family-general claim.

**Scope limitation:** Phase 2 does not explain RF-vs-MLP differences and does not restore any family-level robustness claim.

---

## 5. Protocol difference from Phase 1

Phase 1 MLP configurations used a different training protocol, including combinations involving Adam, early stopping, and architecture/regularization changes.

Phase 2 uses:

- SGD + momentum;
- fixed 200-epoch budget;
- no early stopping;
- single hidden layer;
- fixed weight decay.

Therefore Phase 2 is **not** a direct decomposition of Phase-1 MLP-L/M/H and the two phases must not be linked as an unverified causal chain.

---

## 6. Minimal experimental design

### Model

- MLP
- one hidden layer
- ReLU activation

### Width / capacity-regime

`width ∈ {8, 64, 512}`

Interpretation: low / medium / high **width / capacity-regime**.

This is not called “pure capacity”. Width may simultaneously affect:

- hypothesis-class size;
- optimization geometry;
- feature-learning speed;
- lazy vs rich behavior;
- implicit regularization.

The experiment uses width as a controlled regime intervention, not as a universal metric of effective capacity.

### Regularization

- weight decay = `1e-4`
- fixed across all cells
- no dropout
- no regularization sweep

### Optimizer

- SGD
- momentum = `0.9`
- fixed optimizer class

### LR time-profile

Initial LR for both arms: `0.01`

Two arms only:

1. **constant:** LR remains `0.01`
2. **cosine:** cosine decay from `0.01` toward `0.001` by the end of training

This is described as an **optimizer-class-fixed path / annealing intervention**.

It is explicitly acknowledged that LR schedule may affect:

- optimization noise scale;
- reachable solution;
- implicit regularization;
- late-stage refinement.

Therefore it is not a “pure dynamics separator”.

### Training budget

- 200 epochs
- fixed batch size = 64
- early stopping OFF
- no data augmentation

---

## 7. Data and task regime

Use the already established synthetic spurious-correlation platform.

### Split sizes

- train: 5,000
- IID validation: 2,000 (not used for model selection or early stopping)
- IID test: 3,000
- each OOD environment: 3,000

### Shortcut construction

Use the frozen project generator, including the established symmetric shortcut construction.

No generator redesign is allowed in Phase 2.

### Main task values

`rho_train ∈ {0.7, 0.9}`

### Sanity control

`rho_train = 0`

### OOD grid

Use the existing project OOD grid:

`{0.3, 0.0, -0.3, -0.6, -0.9}`

OOD metrics are secondary only.

### Seeds

20 paired dataset seeds:

`0, 1, ..., 19`

For each dataset seed:

- the same generated data must be reused across all width × schedule cells;
- schedule contrasts within the same width must use matched initialization logic;
- no condition may silently use a different dataset realization.

---

## 8. Experiment matrix

### Headline factorial

`20 seeds × 2 rho × 3 widths × 2 schedules = 240 fits`

### rho=0 sanity

`20 seeds × 1 rho × 3 widths × 2 schedules = 120 fits`

### Total preregistered model fits

`360 fits`

No Adam sensitivity analysis, second model family, additional width, extra LR, extra scheduler, or regularization sweep is part of the frozen Phase-2 experiment.

---

## 9. Boundary Fidelity(t)

### Neutral probe

Reuse the project’s stable-boundary probe logic.

- fixed probe set;
- 4,000 points;
- shortcut neutralized with `x_spurious = 0`;
- any other nuisance variable required by the existing wrong-line probe is fixed exactly as in the established generator/probe implementation;
- labels come from the exact stable oracle.

### Definition

At checkpoint `t`:

`Fidelity(t) = fraction of neutral-probe predictions agreeing with the stable oracle`

Range: `[0, 1]`.

This is intended to measure whether the learned classifier retains the stable decision rule once the shortcut is silenced.

### Checkpoints

Record at:

`{20, 40, 60, 80, 100, 120, 140, 160, 180, 200}`

The same probe must be used at every checkpoint.

---

## 10. D3(t): shortcut-sensitivity trajectory

D3(t) is retained as an auxiliary process measure, not a causal identification claim.

At checkpoint `t`, on the fixed IID test set:

- counterfactually flip the shortcut sign according to the established symmetric construction;
- compute the fraction of predictions that change.

Interpretation:

- high D3(t): high prediction sensitivity to the shortcut;
- low D3(t): low shortcut-sign sensitivity.

D3(t) is a **proxy for instantaneous shortcut sensitivity**. It does not mean the shortcut has been causally removed.

---

## 11. Shortcut Acquisition Time

Define the shortcut acquisition time (SAT) using the predeclared D3 threshold:

`SAT = earliest checkpoint t where D3(t) >= 0.5 for two consecutive checkpoints`

If no such checkpoint exists:

`SAT = infinity`

SAT is descriptive / process-level and does not drive the primary scientific decision.

---

## 12. Fidelity-first trajectory classification

Trajectory categories are descriptive mechanism support. They do not override the primary endpoint decision rules.

Let:

`F_min = min_t Fidelity(t)`

`Recovery = Fidelity(200) - F_min`

### Escape-shaped trajectory

An Escape-shaped trajectory requires all of the following:

1. clear early/mid-training degradation:
   `F_min <= Fidelity(20) - 0.05`
2. later recovery:
   `Recovery >= 0.05`
3. terminal fidelity is therefore at least 0.05 above the trajectory minimum.

D3 decline may be reported as corroborating evidence, but:

**D3 decline alone can never classify a trajectory as Escape.**

The word “escape” must therefore refer to a fidelity pattern of:

`stable-boundary loss → later stable-boundary recovery`

and not merely to reduced shortcut sensitivity.

### Plateau

A configuration shows meaningful fidelity degradation but no ≥0.05 terminal recovery from its minimum.

### No-acquisition / no-distortion

No clear entry into a shortcut-dominated / fidelity-degraded regime is observed.

### Other

Any trajectory not fitting the above frozen categories.

---

## 13. Endpoints

### Primary endpoint — exactly one

`Fidelity(200)`

Rationale:

- directly measures terminal boundary distortion;
- is closer to the mechanism question than OOD accuracy;
- avoids mechanically favoring one LR schedule through a trajectory-integral construction.

### Secondary endpoints

1. final OOD degradation gap, using the frozen OOD evaluation protocol;
2. fidelity recovery / trajectory shape summaries.

### Descriptive only

- D3(t)
- SAT
- Escape / Plateau / No-acquisition / Other
- train loss
- IID accuracy

No extra endpoint may be promoted after observing results.

---

## 14. Statistical protocol

### Pairing

All main contrasts are seed-paired.

### Q1 estimand — schedule residual

For each width `w`, rho, and seed:

`d_w,rho(seed) = Fidelity_cosine(200) - Fidelity_constant(200)`

Report:

- paired mean difference;
- paired bootstrap 95% CI;
- seed-level sign distribution.

### Q2 estimand — width / capacity-regime residual

Compare width levels under the fixed schedule conditions, using the same dataset seeds.

Report effect sizes and paired CIs. Do not interpret width as a pure capacity causal variable.

### Q3 estimand — interaction / coupling

Quantify how the schedule contrast changes across widths.

A simple preregistered interaction summary is:

`I(rho) = max_w d_w,rho - min_w d_w,rho`

The scientific purpose is to detect whether schedule effects are conditional on width.

### Bootstrap

- paired / seed-cluster bootstrap
- 5,000 resamples
- 95% CI

No unnecessary mixed-effects hierarchy or large model-selection analysis is introduced.

---

## 15. Practical thresholds

Retain the preregistered decision thresholds:

- `0.02`: negligible / guard-level scale
- `0.05`: smallest effect of practical interest

These are **decision gates**, not natural laws.

The scientific report must emphasize:

- effect sizes;
- confidence intervals;
- raw trajectory structure.

Threshold crossing alone is not the scientific result.

---

## 16. Guards

### IID-match guard

A schedule effect of practical interest must not simply reflect a large difference in IID performance.

If the absolute mean IID-accuracy difference between schedule arms in a relevant cell is ≥ `0.02`, mark that cell:

`CONFOUNDED / UNRESOLVED`

Do not use it for a clean path-dependence claim.

### rho=0 sanity

A large schedule effect at `rho=0` indicates a general optimization effect rather than a shortcut-specific effect.

If the sanity control contradicts shortcut specificity, narrow the conclusion to the observed training protocol and report it as unresolved with respect to shortcut-specific mechanism.

### Width ladder sanity

Use the rho=0 cells, train loss, interpolation status, and fidelity to verify that the chosen widths actually create meaningfully distinct width/capacity regimes.

If the width ladder is not meaningfully separated, the width-based mechanism interpretation is unresolved.

---

## 17. Decision rules

The following rules are frozen before Phase-2 results are observed.

### Case 1 — Both weak

If, at `rho=0.9`:

- schedule residual magnitude < `0.02`, and
- width-regime effect magnitude < `0.02`

while the width ladder itself is valid:

**Conclusion:** current candidate mechanism is weak.

**Action:** STOP this mechanism direction. Do not expand the benchmark to rescue it.

---

### Case 2 — Schedule / path residual supported

If:

- fixed-width schedule residual reaches ≥ `0.05`;
- paired CI excludes 0;
- the two high-rho regimes show the same direction and comparable scale;
- IID-match guard passes;
- rho=0 sanity does not show the same large general effect;

then:

**Conclusion:** terminal boundary distortion exhibits training-path dependence under the tested protocol.

Allowed claim:

> The terminal stable-boundary fidelity is path-dependent under fixed width and optimizer class.

Not automatically allowed:

> The model escaped a spurious attractor.

The latter requires the Fidelity-first Escape trajectory evidence from Section 12.

---

### Case 3 — Width / capacity-regime dependence supported

If:

- schedule residual is < `0.02` at both high-rho values; and
- width-regime effect is ≥ `0.05` and seed-stable;

then:

**Conclusion:** boundary distortion is strongly associated with the tested width/capacity regime.

Do not claim that “pure capacity causally determines robustness”.

---

### Case 4 — Coupled Training Regime

If:

- width × schedule interaction reaches ≥ `0.05`, or
- the schedule effect changes materially in magnitude/direction across width levels;

then:

**Conclusion:**

> Width/capacity-regime conditions the effect of LR time-profile; the candidate mechanism is a coupled training regime.

Do not announce A or B as the winner.

The mechanism variable is provisionally written as:

`M = Training Regime(width / capacity-regime, LR time-profile)`

---

### Case 5 — Confounded / unresolved

If:

- IID guard fails;
- width ladder is invalid;
- terminal comparison is dominated by clear non-convergence;
- or the result falls outside the frozen interpretable cases;

report:

`UNRESOLVED`

Do not add experiments post hoc to force a conclusion.

---

## 18. Falsification and retreat rules

1. If width-regime dependence dominates and schedule residual is negligible:
   - downgrade the dynamics hypothesis;
   - return to a capacity/regularization mechanism track.

2. If schedule residual dominates:
   - claim path dependence only;
   - require Fidelity-first trajectory evidence before using “escape”.

3. If interaction dominates:
   - adopt the coupled-training-regime interpretation;
   - do not force independent capacity-vs-dynamics causality.

4. If both effects are weak:
   - stop;
   - do not add datasets, models, shifts, optimizers, or hyperparameter grids to rescue the mechanism.

5. Any threshold/factor/level change after freeze requires a documented design amendment before any affected result is inspected.

---

## 19. What this experiment CAN claim

Conditional on results, Phase 2 may claim, within the frozen setting:

- whether terminal boundary distortion shows schedule/path residual;
- whether it varies with width/capacity regime;
- whether those two interventions are coupled;
- whether Fidelity trajectories show genuine loss-and-recovery patterns;
- whether a measurable training regime provides evidence for the missing Mechanism → Behavior link.

---

## 20. What this experiment CANNOT claim

Phase 2 cannot claim:

- family-general robustness;
- which model family is best;
- pure theoretical effective capacity;
- causal identification of regularization;
- general optimizer superiority;
- generality across datasets or shift families;
- a completed model-selection theory;
- an “escape mechanism” without Fidelity-first trajectory evidence.

---

## 21. Anti-Benchmark Checklist

All must be YES before execution.

1. One core scientific question? **YES**
2. One primary model family? **YES — MLP**
3. One shift family? **YES — existing spurious-correlation shift**
4. No new dataset? **YES**
5. No new model family? **YES**
6. No hyperparameter sweep beyond the frozen 3×2 mechanism design? **YES**
7. Every factor directly serves mechanism identification? **YES**
8. One primary endpoint? **YES — Fidelity(200)**
9. Explicit falsification / retreat rule? **YES**
10. If all accuracy rankings are identical, can the experiment still teach us mechanism? **YES**
11. Can the experiment be summarized as:

> We intervene on width/capacity-regime and LR time-profile to test whether they jointly define a training regime that shapes boundary distortion.

**YES**

If any item becomes NO, do not run until supervisor review.

---

## 22. Exact minimal execution specification for Codex

Codex may implement only the following:

### Data
- existing frozen synthetic generator
- train 5,000 / IID val 2,000 / IID test 3,000 / each OOD 3,000
- `rho_train = {0.7, 0.9}` main
- `rho_train = 0` sanity
- existing OOD grid `{0.3, 0.0, -0.3, -0.6, -0.9}`
- 20 dataset seeds, paired across conditions

### Model
- MLP
- one hidden layer
- ReLU
- widths `{8,64,512}`
- weight decay `1e-4`
- batch 64
- 200 epochs
- no dropout
- no augmentation
- no early stopping

### Optimization
- SGD
- momentum `0.9`
- initial LR `0.01`
- schedule arm 1: constant `0.01`
- schedule arm 2: cosine decay toward `0.001`

### Measurements
At checkpoints `{20,40,60,80,100,120,140,160,180,200}`:

- Fidelity(t)
- D3(t)
- train loss
- IID accuracy

At terminal checkpoint:

- Fidelity(200) primary
- OOD degradation secondary
- width-regime diagnostics
- trajectory class

### Explicit prohibitions
Do NOT add:

- Adam
- another optimizer
- another LR
- another scheduler
- another width
- another model family
- another dataset
- another shift
- another primary endpoint
- early stopping
- regularization sweep
- performance winner ranking

Do not alter thresholds after viewing results.

---

## 23. Freeze status

This document reflects the supervisor-approved minor amendments.

It is ready to be committed and tagged as the frozen Phase-2 preregistration.

**Scientific status:** `READY FOR CODEX IMPLEMENTATION AFTER GITHUB FREEZE`

