# Phase 2 Factual Erratum — SAT and Trajectory Interpretation

Status: **DOCUMENTATION CORRECTION ONLY**  
Phase 2 formal results, preregistration, raw result files, thresholds, and frozen decision are unchanged. No new experiment or fit was run.

## 1. Original statement and correction

The Phase 2 audit previously stated:

> SAT was infinity for all fits.

That statement is inconsistent with the raw fit records. The same ambiguity also allowed the classifier label `No-acquisition / no-distortion` to be read as “the model did not learn the shortcut.” That interpretation is not valid under the frozen analysis.

The classifier label is retained exactly as preregistered. `No-acquisition / no-distortion` is a preregistered Fidelity-first trajectory label and must not be interpreted as absence of shortcut acquisition when D3/SAT show otherwise. It means that the preregistered Fidelity collapse/recovery trajectory was not observed.

## 2. Independent raw-result audit

The audit read the 360 JSON records under `results/phase2/fits/` and cross-checked their 3,600 checkpoint rows in `results/phase2/checkpoint_trajectories.csv` and the SAT values in `results/phase2/trajectory_classification.csv`. SAT was independently recomputed using the frozen rule: the first current checkpoint in two consecutive checkpoints with D3 ≥ 0.5.

### SAT distribution

| rho_train | formal records | SAT distribution |
|---:|---:|---|
| 0.0 | 120 | infinity: 120 |
| 0.7 | 120 | infinity: 120 |
| 0.9 | 120 | 40: 120 |

At rho=0.9, D3 is already above 0.5 at checkpoints 20 and 40 for all formal records, yielding SAT=40. Thus the correct statement is not “SAT was infinity for all fits.”

### D3 and Fidelity trajectories

The following are independent means across the 120 records in each rho group; values are read from the fit trajectories and cross-checked against the checkpoint CSV.

| checkpoint | D3 mean, rho=0.0 | D3 mean, rho=0.7 | D3 mean, rho=0.9 | Fidelity mean, rho=0.0 | Fidelity mean, rho=0.7 | Fidelity mean, rho=0.9 |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.016075 | 0.275136 | 0.747922 | 0.985440 | 0.984842 | 0.972835 |
| 40 | 0.016603 | 0.274683 | 0.745358 | 0.986427 | 0.984077 | 0.971879 |
| 60 | 0.016769 | 0.270125 | 0.743408 | 0.983935 | 0.984365 | 0.972290 |
| 80 | 0.018136 | 0.271469 | 0.743136 | 0.984223 | 0.983938 | 0.971363 |
| 100 | 0.019275 | 0.272803 | 0.742703 | 0.985229 | 0.983229 | 0.971102 |
| 120 | 0.020208 | 0.272558 | 0.741806 | 0.983731 | 0.983996 | 0.971102 |
| 140 | 0.021139 | 0.273133 | 0.741381 | 0.983738 | 0.982812 | 0.970846 |
| 160 | 0.021061 | 0.272725 | 0.741108 | 0.983360 | 0.982996 | 0.969760 |
| 180 | 0.021797 | 0.272297 | 0.741411 | 0.982667 | 0.982794 | 0.969927 |
| 200 | 0.022556 | 0.272972 | 0.741478 | 0.983531 | 0.982533 | 0.969583 |

The rho=0.9 D3 trajectory therefore provides clear descriptive evidence of shortcut sensitivity, with mean D3 approximately 0.742–0.748. The Fidelity-first classifier remains unchanged: the largest individual early Fidelity drop was 0.0365, below the preregistered 0.05 threshold, and no trajectory met the required drop-plus-recovery rule.

## 3. What is affected

- The sentence “SAT was infinity for all fits” is corrected.
- The interpretation of `No-acquisition / no-distortion` is corrected so that it cannot be read as “no shortcut acquisition.”
- The descriptive SAT/D3 narrative is corrected to acknowledge shortcut sensitivity at rho=0.9.

## 4. What is not affected

- No raw JSON, CSV, figure, manifest, or result value was edited.
- The preregistration, amendment, trajectory-classifier name, thresholds, primary endpoint, and trajectory rule are unchanged.
- Q1, Q2, and Q3 remain based on terminal `Fidelity(200)` contrasts, not SAT.
- The frozen decision **Both weak / STOP** does not require recomputation: the primary decision was based on the preregistered Fidelity(200) endpoint and Q1/Q2/Q3 practical-scale gates. SAT is a secondary/process measure and was not part of that decision rule.
- Phase 1, Bridge Experiment, and Phase 3 were not started or changed.

## 5. Resolution

The three Phase 2-facing documents were corrected to preserve the raw SAT/D3 facts and the Fidelity-first meaning of the existing classifier label. The Phase 2 scientific status remains **NOT ESTABLISHED** with frozen decision **Both weak / STOP**.
