# Design Amendment Record — Phase 2 Dynamics vs Capacity

**Project:** https://github.com/12412825-collab/thesis-track  
**Target preregistration:** `Methodology_PreRegistration_Phase2_Dynamics_vs_Capacity.md`  
**Amendment timing:** before any Phase-2 formal experiment is run  
**Reason:** supervisor identifiability review  
**Result-driven?** No. No Phase-2 outcome was available when these amendments were made.

---

## 1. Provenance note

The original Phase-2 preregistration was produced outside the repository and was not present in any accessible Git branch/history at freeze time.

The final preregistration therefore reconstructs the approved design from:

1. the previously produced Phase-2 preregistration content;
2. the frozen Phase-1 scientific state;
3. the supervisor's `APPROVE WITH MINOR AMENDMENTS` review;
4. the project’s anti-benchmark constraints.

This amendment record exists to make that reconstruction explicit and auditable.

---

## 2. What did NOT change

The supervisor did **not** request a new experiment.

The following experimental factors remain fixed:

- MLP only;
- one hidden layer;
- ReLU;
- widths `{8,64,512}`;
- fixed weight decay `1e-4`;
- SGD + momentum `0.9`;
- initial LR `0.01`;
- constant vs cosine LR time-profile;
- 200 epochs;
- no early stopping;
- main `rho_train ∈ {0.7,0.9}`;
- `rho_train=0` sanity;
- 20 paired dataset seeds;
- primary endpoint `Fidelity(200)`;
- practical decision thresholds `0.02` and `0.05`.

No new dataset, model family, shift type, LR, width, scheduler, optimizer grid, or regularization sweep was added.

---

## 3. Amendment A — scientific question / estimands

### Previous framing

The design was initially phrased as a competition between:

- A: Optimization / Feature-Learning Dynamics
- B: Capacity / Regularization

as though the two interventions could be treated as theoretically independent mechanisms.

### Supervisor concern

Width changes more than abstract capacity. It can also change:

- optimization geometry;
- feature-learning speed;
- rich vs lazy regime;
- implicit regularization.

Likewise LR schedule changes more than a “pure path”; it can affect:

- optimization noise;
- reachable solution;
- implicit regularization;
- late-stage refinement.

Therefore a strict “A or B is the true independent cause” interpretation is overidentified.

### Final amendment

The experiment now estimates three quantities:

1. **Q1:** schedule residual within fixed width;
2. **Q2:** width / capacity-regime residual;
3. **Q3:** width × schedule coupling.

If Q3 dominates, the mechanism candidate is:

`Training Regime = (Width / Capacity-Regime, LR Time-Profile)`

and the analysis must not declare an A/B winner.

---

## 4. Amendment B — terminology: capacity → width / capacity-regime

### Previous wording

Width was sometimes described as a “capacity manipulation” or “pure capacity” axis.

### Final wording

Use:

**width / capacity-regime**

The experiment does not claim width is a universal or theoretically pure effective-capacity metric.

This is a claim-scope correction, not a factor change.

---

## 5. Amendment C — LR schedule interpretation

### Previous wording

The schedule contrast was described too strongly as if it changed only the optimization path and not the endpoint implicit bias.

### Final wording

The schedule contrast is:

**an optimizer-class-fixed path / annealing intervention**

It may affect:

- optimization noise scale;
- reachable solution;
- implicit regularization;
- late-stage refinement.

The intervention is retained because it is a smaller and more interpretable first perturbation than switching optimizer classes.

No new dynamics factor is added.

---

## 6. Amendment D — formal Coupled Training Regime branch

A new interpretation branch is made explicit.

If the schedule effect changes materially across widths, the result is:

> Width/capacity-regime conditions the effect of LR time-profile.

The experiment then identifies a **coupled training regime**, not two independent mechanisms.

This branch prevents post-hoc forcing of results into “Dynamics wins” or “Capacity wins”.

---

## 7. Amendment E — Fidelity-first Escape definition

### Previous issue

Shortcut-sensitivity decline in D3 could be overinterpreted as “escape”.

A cosine schedule may reduce shortcut sensitivity without restoring the stable decision boundary.

### Final rule

Let:

`F_min = min_t Fidelity(t)`

`Recovery = Fidelity(200) - F_min`

An **Escape-shaped** trajectory requires:

1. `F_min <= Fidelity(20) - 0.05`
2. `Recovery >= 0.05`
3. terminal fidelity is ≥0.05 above the trajectory minimum

D3 decline is corroborating evidence only.

**D3 decline alone is neither necessary nor sufficient to label Escape.**

This change affects trajectory classification / interpretation, not the primary endpoint.

---

## 8. Amendment F — primary endpoint

No change.

The sole primary endpoint remains:

`Fidelity(200)`

OOD degradation remains secondary.

The `0.02` / `0.05` thresholds remain preregistered decision gates and are not treated as natural scientific constants.

---

## 9. Amendment G — Phase-1 vs Phase-2 protocol separation

A required scope note is added.

Phase 1 MLP and Phase 2 MLP use materially different training protocols.

Therefore Phase 2 is not a direct causal decomposition of Phase-1 MLP-L/M/H, and Phase-1 rankings must not be used as if they were generated by the same training object.

---

## 10. Anti-benchmark consequence

The amendment intentionally does **not** add:

- Adam sensitivity analysis;
- a second family;
- new rho values;
- new width levels;
- new LR levels;
- new schedulers;
- a regularization grid;
- another primary endpoint.

The experiment remains a small mechanism-identification study.

---

## 11. Final decision logic after amendment

### Case A — Schedule/path residual
At fixed width, schedule changes terminal fidelity by a stable, practically meaningful amount while guards pass.

Allowed claim: **path dependence**.

Not automatically allowed: “spurious-attractor escape”.

### Case B — Width/capacity-regime residual
Schedule effect is negligible, while width-regime effect is stable and practically meaningful.

Allowed claim: **width/capacity-regime dependence**.

Not allowed: pure capacity causality.

### Case C — Coupled Training Regime
Schedule effect depends materially on width.

Allowed claim:

> width/capacity-regime conditions the effect of LR time-profile.

No A/B winner.

### Case D — Both weak
Both effects are small while the width ladder is valid.

Action: STOP current mechanism direction. Do not expand the benchmark to rescue it.

### Case E — Unresolved
IID guard, width ladder, convergence, or other preregistered validity checks fail.

Action: report unresolved; do not add post-hoc experiments.

---

## 12. Freeze declaration

These amendments were made before Phase-2 formal outcomes were generated and were motivated by identifiability and claim-scope concerns, not observed results.

After this document and the final preregistration are committed to GitHub and tagged, the design should be treated as frozen.

Suggested commit message:

`method: freeze amended Phase 2 mechanism preregistration`

Suggested tag:

`phase2-dynamics-capacity-prereg-frozen`

**Status after remote push/tag:** `READY FOR CODEX IMPLEMENTATION`
