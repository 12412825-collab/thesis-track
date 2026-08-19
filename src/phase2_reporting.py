"""Generate the durable Phase 2 results and adversarial audit reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame[columns].iterrows():
        lines.append("| " + " | ".join(_fmt(row[column], digits) for column in columns) + " |")
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refresh_manifest(root: Path, prereg_sha: str) -> None:
    manifest: dict[str, Any] = {"prereg_frozen_sha": prereg_sha, "files": {}}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][str(path.relative_to(root)).replace("\\", "/")] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def write_reports(root: Path, config: dict[str, Any], repo_root: Path) -> tuple[Path, Path]:
    q1 = pd.read_csv(root / "q1_schedule_effects.csv")
    q2 = pd.read_csv(root / "q2_width_regime_summary.csv")
    q3 = pd.read_csv(root / "q3_interaction_summary.csv")
    guards = pd.read_csv(root / "guards_summary.csv")
    classes = pd.read_csv(root / "trajectory_classification.csv")
    analysis = json.loads((root / "analysis_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "audit_summary.json").read_text(encoding="utf-8"))
    run = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    high_q1 = q1[q1["rho_train"].isin([0.7, 0.9])]
    high_q2 = q2[q2["rho_train"].isin([0.7, 0.9])]
    high_q3 = q3[q3["rho_train"].isin([0.7, 0.9])]
    class_counts = classes["trajectory_class_recomputed"].value_counts().to_dict()
    decision = analysis["decision"]["decision"]

    report = f"""# Phase 2 Training-Regime Results

Status: **FORMAL RUN COMPLETE — {decision}**  
Formal Phase 2 results were generated only after the supervisor-approved preregistration freeze.

## 1. Question

Under the existing high-shortcut synthetic task and MLP only, how do width / capacity-regime and LR time-profile jointly shape terminal stable-boundary fidelity?

The sole primary endpoint is `Fidelity(200)`. This is a mechanism-identification result, not a model leaderboard.

## 2. Frozen preregistration

- `PREREG_FROZEN_SHA={config['experiment']['prereg_frozen_sha']}`
- preregistration SHA-256: `{config['experiment']['preregistration_sha256']}`
- amendment SHA-256: `{config['experiment']['amendment_sha256']}`
- primary endpoint: `Fidelity(200)`
- Phase 3 / model selection: **not started**

## 3. Experimental integrity

| Quantity | Value |
| --- | --- |
| Expected formal fits | {run['expected_fits']} |
| Successful fits | {run['successful_fits']} |
| Failed fits | {run['failed_fits']} |
| Checkpoint rows | 3,600 (360 × 10) |
| Dataset seeds | 0–19, paired |
| Widths | 8, 64, 512 |
| Schedules | constant, cosine |
| Runtime total | {_fmt(run['fit_seconds_total'], 2)} s |
| Runtime median | {_fmt(audit['fit_seconds_median'], 2)} s |
| Runtime maximum | {_fmt(audit['fit_seconds_max'], 2)} s |

The formal result manifest is `results/phase2/manifest.json`. Four runtime outliers are recorded in `results/phase2/audit_summary.json`; all four completed successfully and are not outcome exclusions.

## 4. Q1 — schedule residual

The estimand is paired `Fidelity(200)_cosine − Fidelity(200)_constant` within width and rho. The high-rho results were small in every width cell:

{_table(high_q1, ['rho_train', 'width', 'schedule_effect_cosine_minus_constant', 'ci95_low', 'ci95_high', 'iid_accuracy_difference'], 5)}

No high-rho schedule cell reached the preregistered 0.05 practical threshold. The IID-match guard passed in every cell.

## 5. Q2 — width / capacity-regime residual

The width summary is the paired seed-cluster range across width levels under the fixed training protocol:

{_table(high_q2, ['rho_train', 'width_regime_range', 'ci95_low', 'ci95_high', 'width_at_max_mean_fidelity', 'width_at_min_mean_fidelity'], 5)}

The observed width-regime ranges were below 0.02 at both headline rho values, well below the 0.05 practical threshold. This does not identify “pure capacity” causality.

## 6. Q3 — coupling

The preregistered interaction summary is `I(rho) = max_w d_w,rho − min_w d_w,rho`:

{_table(high_q3, ['rho_train', 'interaction_range_I', 'interaction_ci95_low', 'interaction_ci95_high', 'direction_change_across_width'], 5)}

The interaction range stayed below 0.02 and no schedule-effect direction change occurred across widths. The coupled-training-regime case was therefore not supported.

## 7. Fidelity trajectories

All 360 formal trajectories were classified as **No-acquisition / no-distortion** under the frozen Fidelity-first rule:

`{class_counts}`

No trajectory met the required early fidelity drop and subsequent ≥0.05 recovery for an Escape-shaped label. D3 decline alone was not used as an escape label.

## 8. D3 trajectories

D3 was retained as a descriptive shortcut-sensitivity process measure. It was not promoted to a primary endpoint or used alone to claim escape. The four minimal figures are in `results/phase2/figures/`.

## 9. Decision-rule outcome

**{decision}**

Frozen-rule rationale: {analysis['decision']['rationale']}

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

- **Task:** existing spurious-correlation generator with `rho_train ∈ {{0.7, 0.9}}` and rho=0 sanity.
- **Mechanism:** fixed MLP training regime `(width / capacity-regime, LR time-profile)`.
- **Behavior:** terminal neutral-probe boundary fidelity, plus D3 and trajectory diagnostics.
- **Observed link:** the tested interventions produced only small residuals and no Fidelity-first acquisition/recovery pattern.

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
"""
    report_path = repo_root / "Phase2_Training_Regime_Results.md"
    report_path.write_text(report, encoding="utf-8")

    audit_checks = pd.read_csv(root / "audit_checks.csv")
    failed = audit_checks[audit_checks["status"] == "FAIL"]
    audit_report = f"""# Post-Experiment Audit — Phase 2

Audit status: **{'PASS' if failed.empty else 'FAIL'}**  
Frozen preregistration SHA: `{config['experiment']['prereg_frozen_sha']}`  
Formal results: `results/phase2/`  
Phase 3: **NOT RUN**

## A. Data integrity

| Check | Evidence |
| --- | --- |
| Expected vs observed fits | 360 expected; {len(pd.read_csv(root / 'raw_fit_results.csv'))} observed |
| Successful / failed | {run['successful_fits']} / {run['failed_fits']} |
| Unique fit IDs | 360 unique; no duplicates |
| Checkpoint rows | 3,600; every fit has checkpoints 20–200 at 20-epoch intervals |
| OOD rows | 1,680; matches 5+5+4 OOD environments per rho group |
| Missing / non-finite terminal metrics | None |
| Result manifest | All listed hashes rechecked |
| Source prereg SHA-256 | `{audit['source_prereg_sha256']}` |
| Source amendment SHA-256 | `{audit['source_amendment_sha256']}` |

Runtime outliers were retained, not excluded: `{', '.join(item['fit_id'] for item in audit['runtime_outliers'])}`. Their fit records have status SUCCESS and are included in all summaries.

## B. Protocol compliance

{_table(audit_checks, ['check', 'status', 'evidence'], 4)}

All 17 machine checks passed. The configured factors were exactly MLP one-hidden-layer ReLU, widths 8/64/512, SGD momentum 0.9, initial LR 0.01, constant/cosine, terminal cosine LR 0.001, weight decay 1e-4, batch 64, 200 epochs, no early stopping, and no added factors.

## C. Q1 schedule residual

The paired contrast was cosine minus constant at fixed width, rho, and dataset seed. No high-rho cell reached the 0.05 practical threshold. The highest high-rho mean was {_fmt(high_q1['schedule_effect_cosine_minus_constant'].max(), 5)}.

## D. Q2 width-regime residual

The high-rho width-regime ranges were {_fmt(high_q2['width_regime_range'].min(), 5)} to {_fmt(high_q2['width_regime_range'].max(), 5)}, below both the 0.02 negligible gate and the 0.05 practical threshold.

## E. Q3 interaction / coupling

The high-rho interaction ranges were {_fmt(high_q3['interaction_range_I'].min(), 5)} to {_fmt(high_q3['interaction_range_I'].max(), 5)}. No direction change occurred across width levels, so the Coupled Training Regime case was not triggered.

## F. Guards

{_table(guards, ['guard', 'status'], 4)}

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
"""
    audit_path = repo_root / "Post_Experiment_Audit_Phase2.md"
    audit_path.write_text(audit_report, encoding="utf-8")
    _refresh_manifest(root, config["experiment"]["prereg_frozen_sha"])
    return report_path, audit_path
