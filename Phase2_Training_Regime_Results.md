# Phase 2 Training-Regime Results

Status: **FORMAL RUN COMPLETE — Both weak / STOP**  
Formal Phase 2 results were generated only after the supervisor-approved preregistration freeze.

## 1. Question

Under the existing high-shortcut synthetic task and MLP only, how do width / capacity-regime and LR time-profile jointly shape terminal stable-boundary fidelity?

The sole primary endpoint is `Fidelity(200)`. This is a mechanism-identification result, not a model leaderboard.

## 2. Frozen preregistration

- `PREREG_FROZEN_SHA=654e72b39310847b0d369796c31825e379aaf909`
- preregistration SHA-256: `4778332AEB630E682E67E61AF1AE2DA51B83F14DE7A7ECE9C025607ADE26082A`
- amendment SHA-256: `C94DD303D28C7AE32C247D2316338997C3F5ABB15760A450C22C23B44275BBF4`
- primary endpoint: `Fidelity(200)`
- Phase 3 / model selection: **not started**

## 3. Experimental integrity

| Quantity | Value |
| --- | --- |
| Expected formal fits | 360 |
| Successful fits | 360 |
| Failed fits | 0 |
| Checkpoint rows | 3,600 (360 × 10) |
| Dataset seeds | 0–19, paired |
| Widths | 8, 64, 512 |
| Schedules | constant, cosine |
| Runtime total | 3254.82 s |
| Runtime median | 3.17 s |
| Runtime maximum | 552.66 s |

The formal result manifest is `results/phase2/manifest.json`. Four runtime outliers are recorded in `results/phase2/audit_summary.json`; all four completed successfully and are not outcome exclusions.

## 4. Q1 — schedule residual

The estimand is paired `Fidelity(200)_cosine − Fidelity(200)_constant` within width and rho. The high-rho results were small in every width cell:

| rho_train | width | schedule_effect_cosine_minus_constant | ci95_low | ci95_high | iid_accuracy_difference |
|---|---|---|---|---|---|
| 0.70000 | 8.00000 | 0.00157 | 0.00051 | 0.00270 | 0.00002 |
| 0.70000 | 64.00000 | 0.00287 | 0.00194 | 0.00377 | 0.00050 |
| 0.70000 | 512.00000 | 0.00833 | 0.00626 | 0.01064 | 0.00093 |
| 0.90000 | 8.00000 | 0.00232 | 0.00067 | 0.00416 | 0.00008 |
| 0.90000 | 64.00000 | 0.00204 | 0.00096 | 0.00320 | 0.00002 |
| 0.90000 | 512.00000 | 0.00386 | 0.00236 | 0.00534 | -0.00003 |

No high-rho schedule cell reached the preregistered 0.05 practical threshold. The IID-match guard passed in every cell.

## 5. Q2 — width / capacity-regime residual

The width summary is the paired seed-cluster range across width levels under the fixed training protocol:

| rho_train | width_regime_range | ci95_low | ci95_high | width_at_max_mean_fidelity | width_at_min_mean_fidelity |
|---|---|---|---|---|---|
| 0.70000 | 0.01185 | 0.00934 | 0.01446 | 8.00000 | 512.00000 |
| 0.90000 | 0.00901 | 0.00472 | 0.01303 | 8.00000 | 512.00000 |

The observed width-regime ranges were below 0.02 at both headline rho values, well below the 0.05 practical threshold. This does not identify “pure capacity” causality.

## 6. Q3 — coupling

The preregistered interaction summary is `I(rho) = max_w d_w,rho − min_w d_w,rho`:

| rho_train | interaction_range_I | interaction_ci95_low | interaction_ci95_high | direction_change_across_width |
|---|---|---|---|---|
| 0.70000 | 0.00675 | 0.00447 | 0.00928 | False |
| 0.90000 | 0.00182 | 0.00052 | 0.00406 | False |

The interaction range stayed below 0.02 and no schedule-effect direction change occurred across widths. The coupled-training-regime case was therefore not supported.

## 7. Fidelity trajectories

All 360 formal trajectories were classified as **No-acquisition / no-distortion** under the frozen Fidelity-first rule:

`{'No-acquisition / no-distortion': 360}`

No trajectory met the required early fidelity drop and subsequent ≥0.05 recovery for an Escape-shaped label. D3 decline alone was not used as an escape label.

Interpretation note: `No-acquisition / no-distortion` is a preregistered Fidelity-first trajectory label and must not be interpreted as absence of shortcut acquisition when D3/SAT show otherwise. The raw SAT distribution is infinity for all 120 rho=0.0 fits, infinity for all 120 rho=0.7 fits, and SAT=40 for all 120 rho=0.9 fits; rho=0.9 mean D3 remains approximately 0.742–0.748 across checkpoints. The label therefore means that the preregistered Fidelity collapse/recovery trajectory was not observed, not that the model failed to acquire shortcut sensitivity.

## 8. D3 trajectories

D3 was retained as a descriptive shortcut-sensitivity process measure. It was not promoted to a primary endpoint or used alone to claim escape. The four minimal figures are in `results/phase2/figures/`.

## 9. Decision-rule outcome

**Both weak / STOP**

Frozen-rule rationale: Both high-rho schedule and width-regime effects were below the preregistered 0.02 gate while guards passed.

The rho=0 sanity schedule effect remained below 0.05, the IID-match guard passed, and the width ladder diagnostic passed. The correct action under the preregistration is to stop this mechanism direction rather than expand the benchmark to rescue it.

## 10. Scientific interpretation

Within this exact synthetic generator, high-shortcut task regime, MLP family, width ladder, SGD protocol, and 200-epoch budget, the data do not establish a practically meaningful terminal boundary-fidelity residual for either the schedule or width intervention. The tested training regime did not produce the pre-specified stable-boundary loss-and-recovery pattern.

This is a conditional null result, not a claim that no training dynamics can ever matter.

## 11. What was falsified or weakened

- The preregistered **strong** claim that constant versus cosine LR would create a ≥0.05 terminal Fidelity residual under the tested setting was not supported.
- The preregistered **strong** claim that the tested width regimes would create a ≥0.05 terminal Fidelity residual was not supported.
- A family-general ranking claim was not tested and is not inferred.

## 12. What remains possible

The null result leaves open whether a different, separately approved task regime would actually induce measurable boundary distortion. It also leaves open mechanisms outside this frozen MLP-only protocol, but those are outside the present evidence.

## 13. What we cannot claim

We cannot claim general optimizer superiority, pure effective-capacity causality, family-general robustness, cross-dataset generality, or an escape mechanism. We also cannot use the high training accuracy / terminal fidelity values as proof that the mechanism is absent in every regime.

## 14. Task → Mechanism → Behavior

- **Task:** existing spurious-correlation generator with `rho_train ∈ {0.7, 0.9}` and rho=0 sanity.
- **Mechanism:** fixed MLP training regime `(width / capacity-regime, LR time-profile)`.
- **Behavior:** terminal neutral-probe boundary fidelity, plus D3 and trajectory diagnostics.
- **Observed link:** the tested interventions produced only small residuals and no preregistered Fidelity collapse/recovery pattern.

## Mechanism map

### CONFIRMED

- The 360-cell preregistered factorial was executed with no failures.
- Paired seed, initialization, and shuffle records passed integrity checks.
- The exact primary endpoint and frozen guards were computed.

### WEAKENED

- Schedule/path dependence as a practically meaningful explanation under this protocol.
- Width/capacity-regime dependence as a practically meaningful explanation under this protocol.

### REJECTED

- No broad theory is rejected. The tested ≥0.05 mechanism cases were not established; no family-ranking or pure-capacity claim was evaluated.

### STILL POSSIBLE

- A coupled training regime outside the observed effect range, or a mechanism that requires a task regime that actually induces boundary distortion.

### UNKNOWN

- Generalization beyond this generator, MLP family, rho values, widths, optimizer class, and epoch budget.

## Reviewer-2 verdict

**NOT ESTABLISHED**

The experiment is internally valid and informative, but the proposed Mechanism → Behavior link was not established at the preregistered practical scale.

## Strongest scientific result

Across 360 formal fits, both candidate residuals stayed below the 0.02 negligible gate and all trajectories were No-acquisition / no-distortion.

## Strongest negative result

The high-shortcut task did not induce the preregistered boundary-distortion-and-recovery pattern in any formal trajectory.

## One next-step recommendation

Do not start another experiment yet; first obtain supervisor review of this valid null result and decide whether a new task-regime preregistration is scientifically justified.
