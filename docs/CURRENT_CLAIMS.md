# Current Claims

This ledger separates claims supported by repository evidence from proposals and weakened hypotheses. It does not promote exploratory observations into universal conclusions.

## SUPPORTED

1. The repository contains a controlled synthetic distribution-shift generator with a stable label mechanism and an intervened shortcut / spurious-correlation mechanism. Evidence: `README.md`, `src/data_generation.py`, and the frozen configs under `experiments/EXP-001/` and `experiments/EXP-002/`.
2. The fitting/evaluation boundary excludes OOD/test environments from fitting and model selection. Evidence: `src/evaluation.py`, `tests/test_no_leakage.py`, and the EXP-002V / Phase 1 audits.
3. The preregistered Phase 1 study completed its planned fit matrix and supported decision G4a-C / `REVISE HYPOTHESIS`. Evidence: `Phase1_Family_Capacity_Results.md`, `experiments/PHASE-1/README.md`, and `G4a_Decision_Memo.md`.
4. In the completed Phase 1 design, training shortcut strength accounted for the largest variance component reported for OOD-gap variation; model family was a smaller component and family × shortcut-strength interaction was also material. Evidence: `Phase1_Family_Capacity_Results.md` and `results/phase1/summaries/variance_decomposition.csv`.
5. The Phase 2 formal study completed its declared matrix and its preregistered Fidelity-first endpoint did not establish the proposed schedule/width mechanism. Evidence: `experiments/PHASE-2/README.md`, `Phase2_Training_Regime_Results.md`, and `Post_Experiment_Audit_Phase2.md`.
6. The Bridge reproduced the approved learner-only comparison with its declared hash, pairing, version, and leakage checks; it routed to Case B and did not identify the operative internal component. Evidence: `Bridge_Phase1Learner_vs_Phase2Task_Results.md` and `Post_Experiment_Audit_Bridge.md`.

## PARTIALLY SUPPORTED

1. Family identity may explain a small residual under some controlled regimes, but the remaining signal is below the preregistered practical threshold for the Phase 1 family claim, is not stable across rho, and is not a causal family effect. Evidence: Phase 1 result tables and `G4a_Decision_Memo.md`.
2. Shortcut strength is a credible operational driver of robustness variation in this synthetic generator. Its scope is limited to the implemented generator and completed protocols; it is not a universal theory of distribution shift. Evidence: EXP-002/002V and Phase 1 results.
3. The completed Phase 2 measurements show shortcut-sensitive behavior under the high-rho condition, but the study's Fidelity-first label does not establish the specific learned mechanism. Evidence: `Phase2_Training_Regime_Results.md` and `Post_Experiment_Audit_Phase2.md`.

## NOT ESTABLISHED

1. No learned-model mechanism has been causally isolated as the explanation for why models trained on the same shifted data generalize differently.
2. No universal ordering of model families under distribution shift has been established.
3. No claim generalizes from this synthetic generator to arbitrary covariate, label, concept, temporal, or real-world shifts.
4. The project has not established that any specific optimizer, scaler, width, regularization setting, or training schedule is the operative mechanism in the Bridge/Phase 2 behavior.

## REJECTED / WEAKENED HYPOTHESES

1. **Model-family identity as the main explanatory variable** was weakened by Phase 1: the family main effect was smaller than shortcut-strength effects and family × shortcut-strength interaction, and did not meet the practical stability gate.
2. **A stable family ranking as the general explanation for robustness** was not supported by the preregistered Phase 1 decision.
3. **Fidelity collapse as the established core behavior mechanism** was stopped for the frozen Bridge task; the audit explicitly says it does not identify an internal cause.

