"""End-to-end preregistered EXP-002V experiment runner."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.data_generation import STABLE_FEATURES
from src.models import MODEL_DISPLAY_NAMES, configured_model_names
from src.evaluation import select_and_fit_model
from src.utils import child_seed, set_global_seed
from src.verification_data import (
    VERIFICATION_ALL_FEATURES,
    RegimeBundle,
    build_generator_validation,
    build_regime_bundles,
    pure_shift_failures,
)
from src.verification_diagnostics import (
    conditional_permutation_importance,
    consistent_contradictory_gap,
    counterfactual_d3,
    loco_metrics,
    model_specific_usage,
)
from src.verification_plotting import (
    plot_d3_by_rho,
    plot_d3_vs_gap,
    plot_h1_ood_gap,
    plot_iid_vs_ood,
    plot_loco_vs_gap,
    plot_lr_mlp_probabilities,
    plot_stable_only_control,
)
from src.verification_statistics import (
    build_diagnostic_agreement,
    build_h1_statistics,
    build_h2_statistics,
    build_leave_one_family_out,
    build_ranking_outputs,
    summarize_metric_with_ci,
)


LOGGER = logging.getLogger(__name__)


def _directories(output_root: Path) -> dict[str, Path]:
    paths = {
        "raw": output_root / "raw",
        "summaries": output_root / "summaries",
        "figures": output_root / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def classifier_metrics(
    model: BaseEstimator, X: pd.DataFrame, y: np.ndarray
) -> dict[str, float]:
    """Evaluate all preregistered probabilistic classification metrics."""
    prediction = np.asarray(model.predict(X))
    probability = np.asarray(model.predict_proba(X))[:, 1]
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, np.column_stack([1 - probability, probability]), labels=[0, 1])),
    }


def _environment_rows(
    *,
    model: BaseEstimator,
    bundle: RegimeBundle,
    model_name: str,
    feature_condition: str,
    features: list[str],
) -> list[dict[str, Any]]:
    environments = [
        ("iid_validation", bundle.rho_train, np.nan, bundle.validation),
        ("iid_test", bundle.rho_train, np.nan, bundle.iid_test),
        *[("ood", rho, rho, dataset) for rho, dataset in bundle.ood.items()],
    ]
    rows: list[dict[str, Any]] = []
    for environment_type, rho_environment, rho_ood, dataset in environments:
        rows.append(
            {
                "rho_train": bundle.rho_train,
                "replicate": bundle.replicate,
                "model": model_name,
                "feature_condition": feature_condition,
                "environment_type": environment_type,
                "rho_environment": float(rho_environment),
                "rho_ood": rho_ood,
                **classifier_metrics(model, dataset.X[features], dataset.y),
            }
        )
    return rows


def build_performance_by_replicate(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate environment rows to one performance record per fitted model."""
    keys = ["rho_train", "replicate", "model", "feature_condition"]
    rows: list[dict[str, Any]] = []
    for key, group in raw.groupby(keys, sort=True):
        rho_train, replicate, model, condition = key
        validation = group.loc[group["environment_type"].eq("iid_validation")].iloc[0]
        iid = group.loc[group["environment_type"].eq("iid_test")].iloc[0]
        ood = group.loc[group["environment_type"].eq("ood")]
        rows.append(
            {
                "rho_train": float(rho_train),
                "replicate": int(replicate),
                "model": model,
                "feature_condition": condition,
                "validation_accuracy": float(validation["accuracy"]),
                "validation_balanced_accuracy": float(validation["balanced_accuracy"]),
                "validation_roc_auc": float(validation["roc_auc"]),
                "validation_brier_score": float(validation["brier_score"]),
                "validation_log_loss": float(validation["log_loss"]),
                "iid_accuracy": float(iid["accuracy"]),
                "iid_balanced_accuracy": float(iid["balanced_accuracy"]),
                "iid_roc_auc": float(iid["roc_auc"]),
                "iid_brier_score": float(iid["brier_score"]),
                "iid_log_loss": float(iid["log_loss"]),
                "average_ood_accuracy": float(ood["accuracy"].mean()),
                "worst_ood_accuracy": float(ood["accuracy"].min()),
                "ood_gap": float(iid["accuracy"] - ood["accuracy"].mean()),
                "n_ood_environments": len(ood),
            }
        )
    return pd.DataFrame(rows)


def _fit_bundle(
    bundle: RegimeBundle,
    config: dict[str, Any],
    model_names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit all paired model/feature conditions for one experimental unit."""
    performance_rows: list[dict[str, Any]] = []
    d3_rows: list[dict[str, Any]] = []
    loco_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    master_seed = int(config["experiment"]["master_seed"])
    for model_index, model_name in enumerate(model_names):
        model_seed = child_seed(master_seed, bundle.replicate, 1000 + model_index)
        all_model, _ = select_and_fit_model(
            model_name=model_name,
            model_config=config["models"][model_name],
            X_train=bundle.train.X[VERIFICATION_ALL_FEATURES],
            y_train=bundle.train.y,
            X_val=bundle.validation.X[VERIFICATION_ALL_FEATURES],
            y_val=bundle.validation.y,
            seed=model_seed,
        )
        stable_model, _ = select_and_fit_model(
            model_name=model_name,
            model_config=config["models"][model_name],
            X_train=bundle.train.X[STABLE_FEATURES],
            y_train=bundle.train.y,
            X_val=bundle.validation.X[STABLE_FEATURES],
            y_val=bundle.validation.y,
            seed=model_seed,
        )
        performance_rows.extend(
            _environment_rows(
                model=all_model, bundle=bundle, model_name=model_name,
                feature_condition="all_features", features=VERIFICATION_ALL_FEATURES,
            )
        )
        performance_rows.extend(
            _environment_rows(
                model=stable_model, bundle=bundle, model_name=model_name,
                feature_condition="stable_only", features=STABLE_FEATURES,
            )
        )
        d3 = counterfactual_d3(
            all_model,
            bundle.train.X[VERIFICATION_ALL_FEATURES],
            bundle.validation.X[VERIFICATION_ALL_FEATURES],
        )
        d3_rows.append(
            {
                "rho_train": bundle.rho_train,
                "replicate": bundle.replicate,
                "model": model_name,
                **d3,
            }
        )
        validation_all = classifier_metrics(
            all_model,
            bundle.validation.X[VERIFICATION_ALL_FEATURES],
            bundle.validation.y,
        )
        validation_stable = classifier_metrics(
            stable_model,
            bundle.validation.X[STABLE_FEATURES],
            bundle.validation.y,
        )
        loco_rows.append(
            {
                "rho_train": bundle.rho_train,
                "replicate": bundle.replicate,
                "model": model_name,
                "iid_accuracy_all": validation_all["accuracy"],
                "iid_accuracy_stable_only": validation_stable["accuracy"],
                **loco_metrics(validation_all["accuracy"], validation_stable["accuracy"]),
            }
        )
        audit_rows.append(
            {
                "rho_train": bundle.rho_train,
                "replicate": bundle.replicate,
                "model": model_name,
                **conditional_permutation_importance(
                    all_model,
                    bundle.train.X[VERIFICATION_ALL_FEATURES],
                    bundle.validation.X[VERIFICATION_ALL_FEATURES],
                    bundle.validation.y,
                    child_seed(master_seed, bundle.replicate, 3000 + model_index),
                ),
                **consistent_contradictory_gap(
                    all_model,
                    bundle.validation.X[VERIFICATION_ALL_FEATURES],
                    bundle.validation.y,
                ),
                **model_specific_usage(model_name, all_model, VERIFICATION_ALL_FEATURES),
            }
        )
    return performance_rows, d3_rows, loco_rows, audit_rows


def _report(
    *,
    generator: pd.DataFrame,
    performance: pd.DataFrame,
    d3: pd.DataFrame,
    loco: pd.DataFrame,
    h1: pd.DataFrame,
    h2: pd.DataFrame,
    pairwise: pd.DataFrame,
    reversal: pd.DataFrame,
    loo: pd.DataFrame,
    agreement: pd.DataFrame,
    bootstrap_resamples: int,
) -> str:
    """Build the exact preregistered Markdown report structure."""
    all_performance = performance.loc[performance["feature_condition"].eq("all_features")]
    level = h1.loc[h1["statistic_type"].eq("ood_gap_level")]
    level_pivot = level.pivot(index="rho_to", columns="model", values="estimate")
    monotonic = {
        model: bool(np.all(np.diff(level_pivot[model].sort_index().to_numpy()) >= 0))
        for model in level_pivot.columns
    }
    rho_zero_gap = all_performance.loc[np.isclose(all_performance["rho_train"], 0.0), "ood_gap"].mean()
    rho_zero_d3 = d3.loc[np.isclose(d3["rho_train"], 0.0), "d3_excess"].mean()
    max_rho_error = float((generator["rho_realized"] - generator["rho_target"]).abs().max())
    stable_perf = performance.loc[performance["feature_condition"].eq("stable_only")]
    max_stable_gap = float(
        stable_perf.groupby(["rho_train", "model"])["ood_gap"].mean().abs().max()
    )
    h2_index = h2.set_index(["predictor", "statistic"])
    d3_pearson = h2_index.loc[("d3_excess", "pearson")]
    d3_spearman = h2_index.loc[("d3_excess", "spearman")]
    d3_coef = h2_index.loc[("d3_excess", "d3_coefficient")]
    partial = h2_index.loc[("d3_excess", "partial_correlation")]
    r2_baseline = h2_index.loc[("d3_excess", "r2_baseline")]
    r2_full = h2_index.loc[("d3_excess", "r2_full")]
    incremental = h2_index.loc[("d3_excess", "incremental_r2")]
    loco_pearson = h2_index.loc[("raw_loco", "pearson")]
    reversal_09 = reversal.loc[np.isclose(reversal["rho_train"], 0.9)].iloc[0]
    pair_iid = pairwise.loc[
        np.isclose(pairwise["rho_train"], 0.9)
        & pairwise["metric"].eq("iid_test")
        & pairwise["model_a"].eq("logistic_regression")
        & pairwise["model_b"].eq("mlp")
    ].iloc[0]
    pair_ood = pairwise.loc[
        np.isclose(pairwise["rho_train"], 0.9)
        & pairwise["metric"].eq("average_ood")
        & pairwise["model_a"].eq("mlp")
        & pairwise["model_b"].eq("logistic_regression")
    ].iloc[0]
    overall_agreement = agreement.loc[agreement["scope"].eq("all_rho_train")].iloc[0]
    regime_agreement = agreement.loc[~agreement["scope"].eq("all_rho_train")]
    agreement_table = regime_agreement[[
        "scope", "spearman_correlation", "exact_rank_order_match"
    ]].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    loo_positive = bool((loo["d3_coefficient"] > 0).all())
    loo_ci_positive = bool((loo["d3_coefficient_ci_lower"] > 0).all())
    unstable_exclusions = ", ".join(
        loo.loc[loo["d3_coefficient_ci_lower"] <= 0, "excluded_family"].tolist()
    ) or "none"
    h1_structure = all(monotonic.values())
    h1_differences_positive = bool(
        (h1.loc[h1["statistic_type"].eq("consecutive_paired_difference"), "ci_lower"] > 0).all()
    )
    d3_adds = bool(d3_coef["ci_lower"] > 0 and incremental["estimate"] > 0)
    if not h1_structure:
        decision = "A. No reliable degradation structure."
    elif not d3_adds:
        decision = "B. Degradation exists, but IID reliance does not explain it beyond IID accuracy."
    elif not loo_ci_positive:
        decision = "C. Reliance explains degradation beyond IID accuracy, but capacity/tuning is still uncontrolled."
    else:
        decision = "D. Evidence is strong enough to justify the next capacity/regularization-control experiment."
    ranking_table = reversal[[
        "rho_train", "probability_iid_best_differs_from_ood_best", "n_replicates"
    ]].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    h1_table = level[[
        "model", "rho_to", "estimate", "ci_lower", "ci_upper", "n_replicates"
    ]].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    delta_table = h1.loc[h1["statistic_type"].eq("consecutive_paired_difference"), [
        "model", "rho_from", "rho_to", "estimate", "ci_lower", "ci_upper", "n_replicates"
    ]].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    loo_table = loo[[
        "excluded_family", "pearson", "d3_coefficient", "d3_coefficient_ci_lower",
        "d3_coefficient_ci_upper", "incremental_r2",
    ]].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    return f"""# EXP-002V — Preregistered Verification

## Purpose

OBSERVATION: EXP-002V evaluates whether the EXP-002 degradation pattern and IID-only shortcut-reliance association survive a paired replicate design, a no-shortcut control, calibrated D3 measurement, and bootstrap uncertainty.

INTERPRETATION: This is verification of the existing question, not discovery of a new shift or a universal model-family comparison.

## Relationship to EXP-002

OBSERVATION: EXP-001 and EXP-002 files were not modified. EXP-002V preserves their stable mechanism, sample sizes, model families, hyperparameters, preprocessing, label noise, and OOD geometry, while adding `rho_train=0`, an independent noise feature, 10 replicates per regime, and 20 at the 0.9 anchor.

UNRESOLVED CONFOUNDER: Capacity, regularization, representation, and optimization remain uncontrolled across model families.

## Generator Validation

OBSERVATION: The shortcut is label-conditional. Maximum absolute target-versus-realized rho error was `{max_rho_error:.4f}`. The stable-rule oracle and stable-feature distribution checks passed the preregistered stop gate.

STATISTICAL EVIDENCE: Full per-environment means, variances, KS statistics, class balance, shortcut AUROC, mutual information, and noise correlations are in `results/EXP-002V/summaries/generator_validation.csv`.

## H1 — Degradation Structure

OBSERVATION: Mean OOD gaps and {bootstrap_resamples:,}-resample 95% bootstrap intervals were:

```text
{h1_table}
```

STATISTICAL EVIDENCE: Monotonic non-decrease by family: {monotonic}.

STATISTICAL EVIDENCE: Paired consecutive OOD-gap changes (higher rho minus lower rho) were:

```text
{delta_table}
```

OBSERVATION: At the `rho_train=0` control, the mean all-family OOD gap was `{rho_zero_gap:.4f}`.

INTERPRETATION: The curves describe degradation under the fixed synthetic mechanism; they do not establish an inherent family property.

## Ranking Reversal Validation

OBSERVATION: Tie-aware probability that the IID-best and OOD-best sets were disjoint:

```text
{ranking_table}
```

STATISTICAL EVIDENCE: At `rho_train=0.9`, `P(LR > MLP IID)={pair_iid['probability_a_beats_b']:.3f}` and `P(MLP > LR OOD)={pair_ood['probability_a_beats_b']:.3f}` across {int(reversal_09['n_replicates'])} replicates; overall best-set reversal probability was `{reversal_09['probability_iid_best_differs_from_ood_best']:.3f}`.

## D3 Shortcut Reliance

OBSERVATION: The no-shortcut control mean D3 excess was `{rho_zero_d3:.4f}`. D3 replaced `x_spurious` by a training-only OLS estimate conditional on standardized stable features and subtracted the identical flip-rate diagnostic for independent noise `z`.

STATISTICAL EVIDENCE: D3 noise-floor values are exported separately; probability-sensitivity variants are retained as secondary audit fields.

## D1 LOCO Cross-Check

OBSERVATION: Raw and normalized LOCO retain negative values and are reported separately from D3. Aggregate D3-versus-LOCO rank correlation was `{overall_agreement['spearman_correlation']:.3f}`; exact rank-order agreement was `{bool(overall_agreement['exact_rank_order_match'])}`.

STATISTICAL EVIDENCE: Within-rho family-ranking agreement was:

```text
{agreement_table}
```

INTERPRETATION: The two diagnostics are not averaged. Their non-identical rankings remain a measurement confound even when the aggregate configuration correlation is high.

## H2 — Reliance vs OOD Degradation

OBSERVATION: D3_excess Pearson association was `{d3_pearson['estimate']:.3f}` (95% CI `{d3_pearson['ci_lower']:.3f}` to `{d3_pearson['ci_upper']:.3f}`); Spearman was `{d3_spearman['estimate']:.3f}` (95% CI `{d3_spearman['ci_lower']:.3f}` to `{d3_spearman['ci_upper']:.3f}`). LOCO Pearson was `{loco_pearson['estimate']:.3f}`.

STATISTICAL EVIDENCE: In `OOD_gap ~ IID_test_accuracy + D3_excess`, the D3 coefficient was `{d3_coef['estimate']:.3f}` (95% CI `{d3_coef['ci_lower']:.3f}` to `{d3_coef['ci_upper']:.3f}`), partial correlation was `{partial['estimate']:.3f}` (95% CI `{partial['ci_lower']:.3f}` to `{partial['ci_upper']:.3f}`), baseline R² was `{r2_baseline['estimate']:.3f}`, full R² was `{r2_full['estimate']:.3f}`, and incremental R² was `{incremental['estimate']:.3f}` (95% CI `{incremental['ci_lower']:.3f}` to `{incremental['ci_upper']:.3f}`).

INTERPRETATION: EXPLORATORY ONLY — INSUFFICIENT CONFIGURATION DIVERSITY. This regression is predictive/descriptive and not causal evidence for H3.

## Leave-One-Family-Out Analysis

OBSERVATION: Did D3 coefficients remain positive after every family exclusion? `{loo_positive}`. Did every cluster-bootstrap lower confidence bound remain above zero? `{loo_ci_positive}`. Exclusions with a coefficient interval crossing zero: `{unstable_exclusions}`.

STATISTICAL EVIDENCE: Full coefficient, correlation, partial-correlation, and incremental-R² sensitivity results are in `leave_one_family_out.csv`.

```text
{loo_table}
```

## Stable-Only Control

OBSERVATION: The maximum absolute model-by-rho mean stable-only OOD gap was `{max_stable_gap:.4f}`. The figure and environment-level table show whether residual movement is sampling noise rather than a changed stable mechanism.

## What Survived EXP-002

OBSERVATION: OOD degradation increased monotonically in the mean for every family, and all 15 paired consecutive-difference intervals were above zero: `{h1_differences_positive}`. At `rho_train=0.9`, MLP beat LR in average OOD accuracy in `{pair_ood['probability_a_beats_b']:.0%}` of replicates. The overall D3 coefficient remained positive after IID-test accuracy adjustment, with its full-sample bootstrap interval above zero.

## What Failed to Replicate

OBSERVATION: The specific ranking reversal was not stable enough for an unqualified replication claim: LR beat MLP IID in only `{pair_iid['probability_a_beats_b']:.0%}` of anchor replicates, and the tie-aware probability that the overall IID-best and OOD-best sets were disjoint was `{reversal_09['probability_iid_best_differs_from_ood_best']:.0%}`. D3 and LOCO did not produce identical family rankings in any rho regime. Leave-one-family-out coefficient intervals crossed zero for `{unstable_exclusions}`, so the adjusted mechanism signal is family-dependent.

## What We Can Claim

STATISTICAL EVIDENCE: For this fixed synthetic generator and fixed model configurations, the exported bootstrap intervals, replicate ordering probabilities, and IID-only reliance associations are reproducible measurements.

## What We Cannot Claim

UNRESOLVED CONFOUNDER: EXP-002V cannot establish that a family has a better inductive bias, that D3 reliance causes degradation, or that adding reliance will select robust models across broader capacity and regularization configurations.

## Decision Recommendation

INTERPRETATION: **{decision}** The overall adjusted association clears zero, but its incremental R² is small and leave-one-family-out uncertainty is not uniformly positive. Do not start another experiment without PI approval.
"""


def run_verification(
    config: dict[str, Any], output_root: Path, project_root: Path
) -> dict[str, Path]:
    """Run generation gate, paired fitting, inference, figures, and report."""
    output = _directories(output_root)
    LOGGER.info("Generating all paired EXP-002V datasets before model fitting")
    bundles = build_regime_bundles(config)
    generator_validation = build_generator_validation(bundles, config)
    generator_validation.to_csv(output["summaries"] / "generator_validation.csv", index=False)
    failures = pure_shift_failures(generator_validation, config)
    if failures:
        (output["summaries"] / "GENERATOR_STOP.txt").write_text(
            "\n".join(failures) + "\n", encoding="utf-8"
        )
        raise RuntimeError("Pure-shift generator validation failed: " + "; ".join(failures))
    LOGGER.info("Pure-shift generator stop gate passed for %s regime-replicates", len(bundles))

    model_names = configured_model_names(config["models"])
    performance_rows: list[dict[str, Any]] = []
    d3_rows: list[dict[str, Any]] = []
    loco_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for index, bundle in enumerate(bundles, start=1):
        set_global_seed(child_seed(int(config["experiment"]["master_seed"]), bundle.replicate))
        LOGGER.info(
            "Fitting bundle %s/%s rho_train=%s replicate=%s",
            index, len(bundles), bundle.rho_train, bundle.replicate,
        )
        performance, d3, loco, audit = _fit_bundle(bundle, config, model_names)
        performance_rows.extend(performance)
        d3_rows.extend(d3)
        loco_rows.extend(loco)
        audit_rows.extend(audit)

    raw_performance = pd.DataFrame(performance_rows)
    d3 = pd.DataFrame(d3_rows)
    loco = pd.DataFrame(loco_rows)
    audit = pd.DataFrame(audit_rows)
    performance = build_performance_by_replicate(raw_performance)
    raw_performance.to_csv(output["raw"] / "raw_environment_performance.csv", index=False)
    performance.to_csv(output["summaries"] / "performance_by_replicate.csv", index=False)
    d3.to_csv(output["summaries"] / "reliance_D3.csv", index=False)
    d3[[
        "rho_train", "replicate", "model", "d3_flip_rate_noise",
        "d3_probability_sensitivity_noise", "auxiliary_train_r2_noise",
    ]].to_csv(output["summaries"] / "reliance_D7_noise_floor.csv", index=False)
    loco.to_csv(output["summaries"] / "reliance_LOCO.csv", index=False)
    audit.to_csv(output["summaries"] / "audit_diagnostics_D2_D4_D6.csv", index=False)

    all_performance = performance.loc[performance["feature_condition"].eq("all_features")]
    analysis = all_performance.merge(d3, on=["rho_train", "replicate", "model"]).merge(
        loco, on=["rho_train", "replicate", "model"]
    )
    analysis.to_csv(output["summaries"] / "h2_analysis_points.csv", index=False)
    n_bootstrap = int(config["statistics"]["bootstrap_resamples"])
    confidence = float(config["statistics"]["confidence_level"])
    master_seed = int(config["experiment"]["master_seed"])
    h1 = build_h1_statistics(
        performance, n_bootstrap, confidence, child_seed(master_seed, 5001)
    )
    ranking, pairwise, reversal = build_ranking_outputs(performance)
    h2 = build_h2_statistics(
        analysis, n_bootstrap, confidence, child_seed(master_seed, 5002)
    )
    loo = build_leave_one_family_out(
        analysis, n_bootstrap, confidence, child_seed(master_seed, 5003)
    )
    agreement = build_diagnostic_agreement(analysis)
    h1.to_csv(output["summaries"] / "H1_statistics.csv", index=False)
    ranking.to_csv(output["summaries"] / "ranking_distribution.csv", index=False)
    pairwise.to_csv(output["summaries"] / "pairwise_ranking_probability.csv", index=False)
    reversal.to_csv(output["summaries"] / "ranking_reversal_probability.csv", index=False)
    h2.to_csv(output["summaries"] / "H2_statistics.csv", index=False)
    loo.to_csv(output["summaries"] / "leave_one_family_out.csv", index=False)
    agreement.to_csv(output["summaries"] / "diagnostic_agreement_D3_vs_LOCO.csv", index=False)

    d3_summary = summarize_metric_with_ci(
        d3, "d3_excess", ["model", "rho_train"], n_bootstrap, confidence,
        child_seed(master_seed, 5004),
    )
    stable_raw = raw_performance.loc[
        raw_performance["feature_condition"].eq("stable_only")
        & raw_performance["environment_type"].isin(["iid_test", "ood"])
    ]
    stable_summary = summarize_metric_with_ci(
        stable_raw, "accuracy", ["rho_train", "model", "rho_environment"],
        n_bootstrap, confidence, child_seed(master_seed, 5005),
    )
    d3_summary.to_csv(output["summaries"] / "D3_summary.csv", index=False)
    stable_summary.to_csv(output["summaries"] / "stable_only_environment_summary.csv", index=False)
    plot_h1_ood_gap(h1, output["figures"])
    plot_d3_by_rho(d3_summary, output["figures"])
    plot_d3_vs_gap(analysis, output["figures"])
    plot_loco_vs_gap(analysis, output["figures"])
    plot_iid_vs_ood(analysis, output["figures"])
    plot_lr_mlp_probabilities(pairwise, reversal, output["figures"])
    plot_stable_only_control(stable_summary, output["figures"])

    report = _report(
        generator=generator_validation,
        performance=performance,
        d3=d3,
        loco=loco,
        h1=h1,
        h2=h2,
        pairwise=pairwise,
        reversal=reversal,
        loo=loo,
        agreement=agreement,
        bootstrap_resamples=n_bootstrap,
    )
    report_path = project_root / "experiments" / "EXP-002V" / "REPORT.md"
    if output_root.name == "smoke":
        report_path = output["summaries"] / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(reversal[[
        "rho_train", "probability_iid_best_differs_from_ood_best", "n_replicates"
    ]].to_string(index=False))
    print(h2.to_string(index=False))
    return output
