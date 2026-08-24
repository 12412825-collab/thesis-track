# Distribution Shift Research Status

**Status: PAUSED / FROZEN**  
**Not abandoned.**

This repository is the frozen research record for the Distribution Shift / Learned Bias line. It is intentionally preserved as an auditable archive. Historical experiment code, results, negative findings, audits, and preregistrations must not be deleted or silently reinterpreted.

## Research question

> What predicts robustness under distribution shift: model family, capacity, or learned bias?

The project studies a controlled synthetic distribution-shift setting in which stable features define the label while a shortcut / spurious feature changes across environments. The repository's fitting boundary keeps OOD environments out of fitting and model selection.

## Completed work

- A controlled distribution-shift generator creates stable features, labels, and a configurable spurious correlation. Generation diagnostics and hashes are recorded.
- The shortcut / spurious-correlation intervention is registered in `EXP-001`, `EXP-002`, and the stricter paired-replicate verification `EXP-002V`.
- Phase 1 (`PHASE1-FAMILY-CAPACITY`) completed the preregistered family/capacity study and archived its raw results, summaries, figures, manifest, and audit.
- Phase 2 (`PHASE-2`) completed the preregistered one-hidden-layer MLP training-regime study and archived its formal outputs and audit.
- The Bridge (`BRIDGE-P1MLP-L-P2TASK`) tested the approved learner-only comparison without reopening the frozen Phase 2 protocol.

## Most credible current conclusion

The Phase-1 evidence does not support treating model-family identity as a strong, stable explanatory variable for robustness. Training shortcut strength (`rho_train`) is the dominant controlled factor in the completed study, while the remaining family signal is smaller and regime/method dependent. The interaction between family and shortcut strength is more informative than a single stable family ordering. The completed Bridge also does not identify a learned-model component that explains the observed behavior.

The current scientific bottleneck is therefore mechanism identification: identify an intervention on the learned-model mechanism rather than expanding a horizontal comparison of model families.

## Why paused

> The project reached a mechanism-identification bottleneck. Further horizontal benchmark expansion would not reduce the central scientific uncertainty, so the project is frozen until a better intervention on the learned-model mechanism is designed.

This is a **strategic pause, not negative-result abandonment**. A future restart should preserve the existing evidence and begin with a preregistered mechanism intervention, not a larger family leaderboard.

## Evidence map

- Phase 1 result: `Phase1_Family_Capacity_Results.md`
- Phase 1 audit and constraints: `Research_Status_Audit_Phase1.md`, `Post_Experiment_Audit_Phase1.md`
- Phase 1 decision: `G4a_Decision_Memo.md`
- Phase 2 result: `Phase2_Training_Regime_Results.md`
- Phase 2 audit: `Post_Experiment_Audit_Phase2.md`
- Bridge result and audit: `Bridge_Phase1Learner_vs_Phase2Task_Results.md`, `Post_Experiment_Audit_Bridge.md`
- Frozen experiment registry: `experiments/EXP-001/`, `experiments/EXP-002/`, `experiments/EXP-002V/`, `experiments/PHASE-1/`, `experiments/PHASE-2/`, `experiments/BRIDGE-P1MLP-L-P2TASK/`

