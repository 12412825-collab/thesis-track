# G4a Decision Memo

Date: 2026-08-18  
Gate: family residual after measured capacity/regularization/IID/rho/seed controls

## Decision

**G4a-C — regime-dependent residual.** Do not authorize Phase 2 learned-bias isolation under the current hypothesis.

## Threshold ledger

| Frozen criterion | Observed | Pass? |
|---|---:|---:|
| Family incremental CV R² ≥ 0.05 | 0.0156 | No |
| CV RMSE reduction ≥ 5% | 15.3% | Yes |
| Bootstrap lower bound for family increment > 0 | 0.0103 | Yes |
| Every held-out seed improvement positive | 10/10 | Yes |
| Every leave-one-rho family CV increment ≥ 0.03 | Minimum 0.0086 | No |
| Matched/stratified practical signal | None | No |
| Interaction incremental CV R² < family increment | 0.0279 > 0.0156 | No |
| Wrong-line and completeness gates | Passed | Yes |

G4a-A fails because the family term has a positive bootstrap lower bound and meaningful RMSE improvement. G4a-B fails four required conditions: effect-size threshold, leave-rho stability, matched/stratified corroboration, and interaction dominance. The only allowed remaining branch is G4a-C.

## Interpretation

The data support a narrower claim than “family does not matter.” Family helps predict degradation after measured controls, but the stable increment is small and the relationship changes more by shortcut regime than the family main term explains. At rho 0 the five family gaps are nearly equal; at rho 0.7 LR is lowest; at rho 0.9 MLP is lowest while RF and HGB degrade much more. This is a family × regime pattern.

The post hoc common-OOD-grid sensitivity retains the same ordering of evidence, so the branch is not caused by unequal environment counts.

## Consequence

- Do not create or run `Phase2_Design_Learned_Bias_Isolation.md` under the frozen B-only authorization rule.
- Revise the hypothesis toward boundary distortion conditional on shortcut strength, stable geometry, and effective capacity.
- Before another experiment, construct stronger cross-family overlap—more than three configurations per family and separate structural capacity from regularization where possible.
- Preserve the present result as a negative/redirecting gate, not as a failed experiment.

REVISE HYPOTHESIS

