# Phase-1 MLP-L Protocol Extraction for the Bridge

Status: **EXTRACTED AFTER BRIDGE PREREGISTRATION FREEZE; USED WITHOUT DESIGN EXPANSION**  
Bridge preregistration: `664d02a9f1c928713169866b2f5b3225b3554435`  
Required runtime: `scikit-learn==1.9.0`  
Observed runtime for this execution: `1.9.0`

This file records the learner bundle from the repository implementation. It does not reinterpret or split the frozen bundle.

## Source trace

| Item | Repository source |
|---|---|
| MLP-L registry row | `experiments/PHASE-1/config.yaml`, `config_id: MLP-L` |
| Constructor | `src/phase1_models.py`, `build_phase1_model`, `family == "mlp"` |
| Fit call | `src/phase1_experiment.py`, `fit_phase1_model`; `model.fit(X_train, y_train)` |
| Task construction | `src/phase2_experiment.py`, `build_phase2_bundle` |
| Phase-1 recorded environment | `results/phase1/summaries/run_metadata.json` |

## Constructor extraction

| Parameter | Explicit/default | Frozen Bridge value | Source / version note |
|---|---|---|---|
| implementation | explicit | `sklearn.pipeline.Pipeline` | `src/phase1_models.py` |
| scaler | explicit | `StandardScaler()` with no extra kwargs | `src/phase1_models.py`; fit inside pipeline on training features |
| estimator | explicit | `sklearn.neural_network.MLPClassifier` | `src/phase1_models.py` |
| hidden_layer_sizes | explicit | `(16,)` | MLP-L row in `experiments/PHASE-1/config.yaml` |
| alpha | explicit | `0.0001` | MLP-L row in `experiments/PHASE-1/config.yaml` |
| learning_rate_init | explicit | `0.001` | MLP-L row in `experiments/PHASE-1/config.yaml` |
| max_iter | explicit | `250` | MLP-L row in `experiments/PHASE-1/config.yaml` |
| early_stopping | explicit | `True` | MLP-L row in `experiments/PHASE-1/config.yaml` |
| random_state | explicit at call site | paired Bridge init seed | `build_phase1_model(entry, seed)` |
| solver | not passed | sklearn default | default behavior under frozen `scikit-learn==1.9.0` |
| activation | not passed | sklearn default | default behavior under frozen `scikit-learn==1.9.0` |
| learning_rate | not passed | sklearn default | default behavior under frozen `scikit-learn==1.9.0` |
| batch_size | not passed | sklearn default | default behavior under frozen `scikit-learn==1.9.0` |
| validation_fraction | not passed | sklearn default | default behavior under frozen `scikit-learn==1.9.0` |
| n_iter_no_change | not passed | sklearn default | default behavior under frozen `scikit-learn==1.9.0` |
| tol | not passed | sklearn default | default behavior under frozen `scikit-learn==1.9.0` |
| beta_1 | not passed | sklearn default | default behavior under frozen `scikit-learn==1.9.0` |
| beta_2 | not passed | sklearn default | default behavior under frozen `scikit-learn==1.9.0` |
| epsilon | not passed | sklearn default | default behavior under frozen `scikit-learn==1.9.0` |
| other optimizer/fit kwargs | not passed | sklearn defaults | no values invented for the Bridge |

The Bridge call uses exactly `build_phase1_model` for the frozen `MLP-L` registry entry. It does not pass any OOD, IID-test, validation, probe, or D3 data to `fit`.

## Task-side and preprocessing extraction

- The task is generated only through `build_phase2_bundle` using `experiments/PHASE-2/config.yaml`.
- The feature order is `x1, x2, x3, x_spurious, z`.
- The scaler is fitted by the pipeline only on `bundle.train.X[feature_order]` and then reused for IID, OOD, probe, and D3 counterfactual inputs.
- The probe, IID test, and OOD environments are never used to fit the scaler or estimator.
- The paired init seed is `child_seed(20260818, dataset_seed, rho_code, 9100)`, with `rho_code=int(round((rho+1)*1000))`.

