# Bridge Results — Phase-1 MLP-L Learner on the Frozen Phase-2 Task

Status: **FORMAL RUN COMPLETE — Case B / STOP Fidelity-collapse line**  
Bridge preregistration: `664d02a9f1c928713169866b2f5b3225b3554435`  
Phase-2 reference preregistration: `654e72b39310847b0d369796c31825e379aaf909`

## 1. Question and frozen task

The Bridge replaced only the learner bundle while freezing the Phase-2 task. The task was generated through `build_phase2_bundle` using the Phase-2 generator/configuration, with stable coefficients `[1.2, -1.0, 0.8]`, label noise `0.5`, baseline geometry, Phase-2 sample sizes, Phase-2 probe, and Phase-2 OOD environments.

The formal matrix was exactly 20 dataset seeds × rho `{0.9, 0.0}`. The only new learner fits were Phase-1 `MLP-L` (40 total). The Phase-2 comparator was reused from the frozen `width=64`, `constant` reference files; no reference fit was rerun.

## 2. Phase-1 MLP-L protocol

The learner was the complete repository-derived bundle:

```text
Pipeline([
  ("scale", StandardScaler()),
  ("model", MLPClassifier(
      hidden_layer_sizes=(16,), alpha=0.0001,
      learning_rate_init=0.001, max_iter=250,
      early_stopping=True, random_state=bridge_init_seed
  ))
])
```

Unpassed sklearn parameters were left at the frozen `scikit-learn==1.9.0` defaults. The scaler was fitted only on training features and reused for IID, OOD, probe, and D3 inputs. See `Phase1_MLPL_Protocol_Extraction.md` and `Post_Experiment_Audit_Bridge.md`.

## 3. Terminal Behavior by arm

Means over 20 paired seeds:

| rho_train | learner | Fidelity | D3 | OOD gap | IID accuracy |
|---:|---|---:|---:|---:|---:|
| 0.9 | Phase-2 width64-constant reference | 0.968137 | 0.743117 | 0.440900 | 0.993600 |
| 0.9 | Phase-1 MLP-L | 0.961588 | 0.560267 | 0.340623 | 0.991083 |
| 0.0 | Phase-2 width64-constant reference | 0.982150 | 0.023967 | 0.001017 | 0.910183 |
| 0.0 | Phase-1 MLP-L | 0.978375 | 0.019983 | 0.002267 | 0.908933 |

At rho=0.9 the MLP-L learner still has high shortcut sensitivity and a large OOD gap while retaining high neutral-slice Fidelity. This is the central Behavior mismatch in the Bridge.

## 4. Primary paired Fidelity result

The preregistered primary delta is `Delta_F = Fidelity(reference) - Fidelity(MLP-L)`, so positive values mean lower MLP-L Fidelity.

| rho_train | mean Delta_F | median | SD | bootstrap 95% CI | seeds Delta_F>0 | >0.02 | >0.05 |
|---:|---:|---:|---:|---|---:|---:|---:|
| 0.9 | 0.006550 | 0.003750 | 0.021170 | [-0.002463, 0.015388] | 12/20 | 4/20 | 1/20 |
| 0.0 sanity | 0.003775 | 0.001375 | 0.012538 | [-0.001200, 0.009563] | 11/20 | 2/20 | 0/20 |

The rho=0.9 effect is below the frozen negligible gate of 0.02, its CI includes zero, and only one seed exceeds 0.05. This is not a stable, practically meaningful Fidelity collapse.

The raw seed-level paired table is `results/bridge_p1mlp_l/paired_results.csv`.

## 5. Secondary paired D3 and OOD-gap results

These use the same convention, reference minus MLP-L:

| rho_train | metric | mean delta | median | SD | bootstrap 95% CI |
|---:|---|---:|---:|---:|---|
| 0.9 | D3 | 0.182850 | 0.189000 | 0.097989 | [0.139230, 0.223568] |
| 0.9 | OOD gap | 0.100277 | 0.105300 | 0.052426 | [0.077686, 0.122047] |
| 0.0 sanity | D3 | 0.003983 | 0.004333 | 0.010545 | [-0.001100, 0.007983] |
| 0.0 sanity | OOD gap | -0.001250 | -0.001500 | 0.004487 | [-0.003242, 0.000700] |

The rho=0.9 MLP-L arm therefore continues to acquire/use the shortcut and to fail under OOD shift, but its neutral-slice Fidelity remains high. D3/OOD failure and Fidelity collapse are not the same Behavior in this task.

## 6. Frozen decision

- **Case A — learner-protocol-dependent Fidelity collapse:** not supported (`mean Delta_F=0.00655`, CI includes zero, 4/20 seeds exceed 0.02).
- **Case B — Fidelity is not the appropriate core Behavior for this baseline task:** supported (`abs(mean Delta_F)<0.02`, MLP-L mean D3 `0.560267≥0.5`, MLP-L mean OOD gap `0.340623≥0.30`).
- **Case C — unresolved / not shortcut-specific:** not supported; rho=0 mean Delta_F is `0.003775` with CI including zero.
- **Case D — protocol blocker:** not present; task, learner extraction, scaler, hashes, pairing, version, and fit-success gates passed.

**Frozen decision: Case B.** Stop the Fidelity-collapse mechanism line for this baseline task. Do not start Phase 3 or split the learner bundle without a new preregistration.

## 7. Reviewer-2 verdict

**SUPPORTED**, narrowly for the preregistered Case-B routing.

## 8. What this result can and cannot claim

It can claim that, on the frozen Phase-2 task, the complete Phase-1 MLP-L bundle does not produce a practical neutral-slice Fidelity collapse relative to the Phase-2 width64-constant reference, while D3 and OOD failure remain substantial at rho=0.9.

It cannot claim that Adam, early stopping, StandardScaler, width 16, or any sklearn default is the operative cause; the bundle was not split. It cannot claim that 0.72225 should have reappeared, that Phase 2 was invalid, or that this routing generalizes to other families, optimizers, datasets, geometries, or shifts.

## 9. Strongest findings and next question

Strongest positive result: the learner-only Bridge makes the Behavior mismatch explicit—MLP-L mean D3 `0.560267` and OOD gap `0.340623` coexist with Fidelity `0.961588`, so shortcut use/OOD failure does not require the preregistered Fidelity-collapse pattern.

Strongest negative result: there is no practical learner-dependent Fidelity collapse at rho=0.9; mean Delta_F is `0.006550`, the CI spans zero, and the leave-one-seed-out mean remains between `0.00421` and `0.00932`.

Fidelity line: **STOP**.

Recommended next scientific question: **Can shortcut acquisition/use and OOD failure be predicted directly by a preregistered Behavior metric on the same frozen task, without using neutral-slice Fidelity as its proxy?**

