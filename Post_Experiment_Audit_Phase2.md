# Post-Experiment Audit — Phase 2

Audit status: **PASS**  
Frozen preregistration SHA: `654e72b39310847b0d369796c31825e379aaf909`  
Formal results: `results/phase2/`  
Phase 3: **NOT RUN**

## A. Data integrity

| Check | Evidence |
| --- | --- |
| Expected vs observed fits | 360 expected; 360 observed |
| Successful / failed | 360 / 0 |
| Unique fit IDs | 360 unique; no duplicates |
| Checkpoint rows | 3,600; every fit has checkpoints 20–200 at 20-epoch intervals |
| OOD rows | 1,680; matches 5+5+4 OOD environments per rho group |
| Missing / non-finite terminal metrics | None |
| Result manifest | All listed hashes rechecked |
| Source prereg SHA-256 | `4778332aeb630e682e67e61af1ae2da51b83f14de7a7ece9c025607ade26082a` |
| Source amendment SHA-256 | `c94dd303d28c7ae32c247d2316338997c3f5abb15760a450c22c23b44275bbf4` |

Runtime outliers were retained, not excluded: `rho0_seed03_width512_cosine, rho0_seed10_width512_cosine, rho0p7_seed08_width512_constant, rho0p9_seed04_width512_cosine`. Their fit records have status SUCCESS and are included in all summaries.

## B. Protocol compliance

| check | status | evidence |
|---|---|---|
| fit count | PASS | observed=360, expected=360 |
| fit ID uniqueness | PASS | duplicates=0 |
| fit ID coverage | PASS | missing=0, unexpected=0 |
| fit status | PASS | success=360, failed=0 |
| checkpoint row count | PASS | observed=3600, expected=360 fits x 10 checkpoints |
| checkpoint completeness | PASS | every successful fit has exactly the 10 preregistered checkpoints |
| OOD row count | PASS | observed=1680, expected=120*5 + 120*5 + 120*4 |
| terminal numeric finiteness | PASS | rows=360, columns=terminal_fidelity,terminal_d3,terminal_train_loss,terminal_iid_accuracy,final_ood_gap |
| trajectory numeric finiteness | PASS | rows=3600 |
| fidelity range | PASS | min=0.925750, max=0.996250 |
| D3 range | PASS | min=0.005000, max=0.809667 |
| schedule pairing | PASS | pairing_failures=[] |
| protocol constants | PASS | unexpected_values={} |
| prereg SHA traceability | PASS | unique_sha=['654e72b39310847b0d369796c31825e379aaf909'] |
| no hidden model arms | PASS | only frozen width and schedule arms are present |
| runtime outliers recorded | PASS | outlier_count=4, max_seconds=552.657 |
| manifest hashes | PASS | mismatches=[] |

All 17 machine checks passed. The configured factors were exactly MLP one-hidden-layer ReLU, widths 8/64/512, SGD momentum 0.9, initial LR 0.01, constant/cosine, terminal cosine LR 0.001, weight decay 1e-4, batch 64, 200 epochs, no early stopping, and no added factors.

## C. Q1 schedule residual

The paired contrast was cosine minus constant at fixed width, rho, and dataset seed. No high-rho cell reached the 0.05 practical threshold. The highest high-rho mean was 0.00833.

## D. Q2 width-regime residual

The high-rho width-regime ranges were 0.00901 to 0.01185, below both the 0.02 negligible gate and the 0.05 practical threshold.

## E. Q3 interaction / coupling

The high-rho interaction ranges were 0.00182 to 0.00675. No direction change occurred across width levels, so the Coupled Training Regime case was not triggered.

## F. Guards

| guard | status |
|---|---|
| IID-match | PASS |
| rho=0 sanity | PASS |
| width ladder | PASS |
| width ladder diagnostics | PASS |

- IID-match: PASS in all 9 rho × width cells.
- rho=0 sanity: PASS; maximum schedule residual magnitude was 0.00585.
- width ladder: PASS as a diagnostic separation; raw train-loss and fidelity ranges are retained.

## G. Trajectory evidence

All 360 trajectories were No-acquisition / no-distortion. SAT was infinity for all fits. No trajectory satisfied both the ≥0.05 early fidelity drop and ≥0.05 terminal recovery requirements for Escape-shaped classification.

## H. Strongest alternative explanation

The strongest attack is that the selected task/training regime may be too benign to produce the boundary distortion that the mechanism analysis was designed to explain. The observed null therefore supports “not established under this protocol,” not “training mechanisms never matter.” This is precisely why the frozen retreat rule is STOP rather than post-hoc expansion.

## I. Reviewer-2 verdict

**NOT ESTABLISHED**

The data and code pass the integrity audit, but neither schedule/path residual nor width/capacity-regime residual reached the preregistered practical scale, and the Fidelity-first trajectory evidence did not show acquisition followed by recovery.

## Audit artifacts

- `results/phase2/audit_summary.json`
- `results/phase2/audit_checks.csv`
- `results/phase2/manifest.json`
- `results/phase2/figures/`
