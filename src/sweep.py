"""Execution, aggregation, figures, and reporting for EXP-002."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

if os.name == "nt":
    os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_generation import ALL_FEATURES, STABLE_FEATURES, generate_dataset
from src.diagnostics import permutation_reliance_score, validate_environment_metadata
from src.evaluation import (
    aggregate_model_summary,
    evaluate_environments,
    per_seed_robustness,
    select_and_fit_model,
)
from src.models import MODEL_DISPLAY_NAMES, configured_model_names
from src.utils import child_seed, set_global_seed


LOGGER = logging.getLogger(__name__)


def rho_test_grid(rho_train: float, common_values: list[float]) -> list[float]:
    """Return the IID point followed by unique standardized OOD points."""
    values: list[float] = []
    for value in [rho_train, *common_values]:
        value = float(value)
        if not any(np.isclose(value, existing) for existing in values):
            values.append(value)
    return values


def _dataset(data_config: dict[str, Any], n: int, rho: float, seed: int, split: str):
    return generate_dataset(
        n_samples=n,
        rho=rho,
        seed=seed,
        stable_coefficients=data_config["stable_coefficients"],
        label_noise=float(data_config["label_noise"]),
        split=split,
    )


def _test_seed(seed: int, rho: float) -> int:
    """Use the same underlying test sample whenever a rho repeats across regimes."""
    rho_code = int(round((rho + 1.0) * 1000))
    return child_seed(seed, 100, rho_code)


def _output_directories(output_root: Path) -> dict[str, Path]:
    paths = {
        "raw": output_root / "raw",
        "summaries": output_root / "summaries",
        "figures": output_root / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def run_training_strength_sweep(
    config: dict[str, Any], output_root: Path
) -> dict[str, Path]:
    """Run EXP-002 while preserving a strict train/validation/OOD boundary."""
    output = _output_directories(output_root)
    data_config = config["data"]
    model_names = configured_model_names(config["models"])
    result_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []

    for rho_train_value in data_config["rho_train_values"]:
        rho_train = float(rho_train_value)
        rho_val = rho_train
        test_values = rho_test_grid(rho_train, data_config["rho_test_common"])
        for seed_value in config["seeds"]:
            seed = int(seed_value)
            set_global_seed(seed)
            LOGGER.info("EXP-002 rho_train=%s seed=%s: generating data", rho_train, seed)
            # Identical seeds across rho_train regimes hold stable X, labels, and
            # nuisance noise fixed; only the rho transformation changes.
            train = _dataset(
                data_config, int(data_config["n_train"]), rho_train,
                child_seed(seed, 1), "train",
            )
            validation = _dataset(
                data_config, int(data_config["n_val"]), rho_val,
                child_seed(seed, 2), "validation",
            )
            tests = {
                rho: _dataset(
                    data_config, int(data_config["n_test"]), rho,
                    _test_seed(seed, rho), f"test_rho_{rho}",
                )
                for rho in test_values
            }
            for dataset in [train, validation, *tests.values()]:
                metadata_rows.append(
                    {"experiment_seed": seed, "rho_train": rho_train, **dataset.metadata}
                )

            for condition, features in {
                "all_features": ALL_FEATURES,
                "stable_only": STABLE_FEATURES,
            }.items():
                for model_index, model_name in enumerate(model_names):
                    LOGGER.info(
                        "Fitting rho_train=%s seed=%s condition=%s model=%s",
                        rho_train, seed, condition, model_name,
                    )
                    model, validation_metrics = select_and_fit_model(
                        model_name=model_name,
                        model_config=config["models"][model_name],
                        X_train=train.X[features],
                        y_train=train.y,
                        X_val=validation.X[features],
                        y_val=validation.y,
                        seed=child_seed(seed, 1000 + model_index),
                    )
                    _, permuted, reliance = permutation_reliance_score(
                        model,
                        validation.X[features],
                        validation.y,
                        child_seed(seed, 2000 + model_index),
                    )
                    diagnostic_rows.append(
                        {
                            "rho_train": rho_train,
                            "rho_val": rho_val,
                            "seed": seed,
                            "model": model_name,
                            "feature_condition": condition,
                            "validation_accuracy": validation_metrics["accuracy"],
                            "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
                            "validation_log_loss": validation_metrics["log_loss"],
                            "permuted_validation_accuracy": permuted,
                            "spurious_reliance_score": reliance,
                        }
                    )
                    evaluated = evaluate_environments(
                        model, tests, features, seed, model_name, condition
                    )
                    for row in evaluated:
                        row["rho_train"] = rho_train
                        row["rho_val"] = rho_val
                    result_rows.extend(evaluated)

    raw_results = pd.DataFrame(result_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    environment_summary = pd.DataFrame(metadata_rows)
    per_seed_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    for rho_train, rho_results in raw_results.groupby("rho_train", sort=True):
        rho_diagnostics = diagnostics.loc[np.isclose(diagnostics["rho_train"], rho_train)]
        per_seed = per_seed_robustness(rho_results, rho_diagnostics, float(rho_train))
        per_seed.insert(0, "rho_train", float(rho_train))
        per_seed_parts.append(per_seed)
        summary = aggregate_model_summary(per_seed)
        summary.insert(0, "rho_train", float(rho_train))
        summary_parts.append(summary)
    per_seed_robustness_table = pd.concat(per_seed_parts, ignore_index=True)
    model_summary = pd.concat(summary_parts, ignore_index=True)
    ranking_summary = build_ranking_reversal_summary(model_summary)
    reliance_correlation = build_reliance_correlations(model_summary)

    raw_results.to_csv(output["raw"] / "raw_results.csv", index=False)
    diagnostics.to_csv(output["summaries"] / "spurious_reliance.csv", index=False)
    environment_summary.to_csv(output["summaries"] / "environment_summary.csv", index=False)
    per_seed_robustness_table.to_csv(
        output["summaries"] / "per_seed_robustness.csv", index=False
    )
    model_summary.to_csv(output["summaries"] / "model_summary.csv", index=False)
    ranking_summary.to_csv(
        output["summaries"] / "ranking_reversal_summary.csv", index=False
    )
    reliance_correlation.to_csv(
        output["summaries"] / "reliance_correlation.csv", index=False
    )

    plot_shift_sweep(raw_results, output["figures"])
    plot_ranking_comparison(model_summary, output["figures"])
    plot_reliance_vs_gap(model_summary, output["figures"])

    sanity_warnings = validate_environment_metadata(environment_summary)
    stable_control = model_summary.loc[
        model_summary["feature_condition"].eq("stable_only")
    ]
    max_stable_gap = float(stable_control["ood_gap_mean"].abs().max())
    stable_threshold = float(
        config["controls"]["stable_only_max_abs_ood_gap_warning"]
    )
    if max_stable_gap > stable_threshold:
        sanity_warnings.append(
            f"Stable-only absolute OOD gap {max_stable_gap:.4f} exceeds "
            f"the configured {stable_threshold:.4f} warning threshold."
        )
    report = build_research_report(
        model_summary,
        ranking_summary,
        reliance_correlation,
        max_stable_gap,
        sanity_warnings,
    )
    report_path = output_root.parents[1] / "experiments" / "EXP-002" / "REPORT.md"
    # For nonstandard output roots (for example tests), keep the report with outputs.
    if output_root.name == "smoke" or not (report_path.parent / "config.yaml").exists():
        report_path = output["summaries"] / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(ranking_summary.to_string(index=False))
    print(reliance_correlation.to_string(index=False))
    print("Sanity-check warnings:", sanity_warnings or "none")
    return output


def build_ranking_reversal_summary(model_summary: pd.DataFrame) -> pd.DataFrame:
    """Select IID, average-OOD, and worst-case winners for every regime."""
    rows: list[dict[str, Any]] = []
    all_features = model_summary.loc[model_summary["feature_condition"].eq("all_features")]
    for rho_train, group in all_features.groupby("rho_train", sort=True):
        iid_best = group.loc[group["validation_accuracy_mean"].idxmax(), "model"]
        ood_best = group.loc[group["average_ood_accuracy_mean"].idxmax(), "model"]
        worst_best = group.loc[
            group["worst_environment_accuracy_mean"].idxmax(), "model"
        ]
        rows.append(
            {
                "rho_train": float(rho_train),
                "IID_best_model": iid_best,
                "OOD_best_model": ood_best,
                "worst_case_best_model": worst_best,
                "ranking_reversal": bool(iid_best != ood_best),
            }
        )
    return pd.DataFrame(rows)


def build_reliance_correlations(model_summary: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive Pearson correlations across aggregate model points."""
    data = model_summary.loc[model_summary["feature_condition"].eq("all_features")]
    rows = [
        {
            "scope": "all_rho_train",
            "rho_train": np.nan,
            "n_model_points": len(data),
            "pearson_correlation": float(
                data["spurious_reliance_score_mean"].corr(data["ood_gap_mean"])
            ),
        }
    ]
    for rho_train, group in data.groupby("rho_train", sort=True):
        rows.append(
            {
                "scope": "within_rho_train",
                "rho_train": float(rho_train),
                "n_model_points": len(group),
                "pearson_correlation": float(
                    group["spurious_reliance_score_mean"].corr(group["ood_gap_mean"])
                ),
            }
        )
    return pd.DataFrame(rows)


def _figure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
        }
    )


def _save_both(figure: plt.Figure, base: Path) -> None:
    figure.savefig(base.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _rho_filename(rho: float) -> str:
    return str(rho).replace("-", "m").replace(".", "p")


def plot_shift_sweep(results: pd.DataFrame, output_dir: Path) -> None:
    """Create all-feature and stable-only mean ± SE curves for each rho_train."""
    _figure_style()
    colors = dict(zip(MODEL_DISPLAY_NAMES, plt.get_cmap("tab10").colors))
    for rho_train, regime in results.groupby("rho_train", sort=True):
        figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
        for axis, condition, title in zip(
            axes,
            ["all_features", "stable_only"],
            ["All Features", "Stable Features Only (Control)"],
        ):
            subset = regime.loc[regime["feature_condition"].eq(condition)]
            for model, group in subset.groupby("model", sort=False):
                stats = (
                    group.groupby("rho_test")["accuracy"]
                    .agg(["mean", "sem"])
                    .reset_index()
                    .sort_values("rho_test")
                )
                x = stats["rho_test"].to_numpy()
                mean = stats["mean"].to_numpy()
                sem = stats["sem"].fillna(0.0).to_numpy()
                axis.plot(
                    x, mean, marker="o", linewidth=2,
                    color=colors[model], label=MODEL_DISPLAY_NAMES[model],
                )
                axis.fill_between(x, mean - sem, mean + sem, color=colors[model], alpha=0.15)
            axis.axvline(rho_train, color="black", linestyle="--", linewidth=1)
            axis.set_title(title)
            axis.set_xlabel("Test spurious correlation (rho_test)")
            axis.set_xticks(sorted(subset["rho_test"].unique()))
            axis.set_ylim(0.0, 1.02)
        axes[0].set_ylabel("Classification accuracy")
        axes[0].legend(ncol=2, frameon=True)
        figure.suptitle(f"EXP-002 Shift Degradation — rho_train = {rho_train:.1f}")
        figure.tight_layout()
        _save_both(
            figure,
            output_dir / f"shift_curves_rho_train_{_rho_filename(float(rho_train))}",
        )


def plot_ranking_comparison(model_summary: pd.DataFrame, output_dir: Path) -> None:
    """Compare IID-validation and average-OOD rankings with uncertainty."""
    _figure_style()
    data = model_summary.loc[model_summary["feature_condition"].eq("all_features")]
    regimes = sorted(data["rho_train"].unique())
    figure, axes = plt.subplots(1, len(regimes), figsize=(15.0, 4.8), sharey=True)
    for axis, rho_train in zip(axes, regimes):
        group = data.loc[np.isclose(data["rho_train"], rho_train)]
        positions = np.arange(len(group))
        width = 0.36
        axis.bar(
            positions - width / 2,
            group["validation_accuracy_mean"],
            width,
            yerr=group["validation_accuracy_se"],
            capsize=2,
            label="IID validation",
        )
        axis.bar(
            positions + width / 2,
            group["average_ood_accuracy_mean"],
            width,
            yerr=group["average_ood_accuracy_se"],
            capsize=2,
            label="Average OOD",
        )
        axis.set_xticks(
            positions,
            [MODEL_DISPLAY_NAMES[name] for name in group["model"]],
            rotation=35,
            ha="right",
        )
        axis.set_title(f"rho_train = {rho_train:.1f}")
        axis.set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(loc="lower right")
    figure.suptitle("IID Validation vs Average OOD Model Selection")
    figure.tight_layout()
    _save_both(figure, output_dir / "iid_vs_average_ood_ranking")


def plot_reliance_vs_gap(model_summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot aggregate reliance versus OOD gap by model and training regime."""
    _figure_style()
    data = model_summary.loc[model_summary["feature_condition"].eq("all_features")]
    colors = dict(zip(MODEL_DISPLAY_NAMES, plt.get_cmap("tab10").colors))
    markers = {0.5: "o", 0.7: "s", 0.9: "^"}
    figure, axis = plt.subplots(figsize=(8.2, 5.8))
    for _, row in data.iterrows():
        model = row["model"]
        rho_train = float(row["rho_train"])
        axis.errorbar(
            row["spurious_reliance_score_mean"],
            row["ood_gap_mean"],
            xerr=row["spurious_reliance_score_se"],
            yerr=row["ood_gap_se"],
            fmt=markers[rho_train],
            color=colors[model],
            capsize=2,
            markersize=7,
        )
    model_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors[name], label=display)
        for name, display in MODEL_DISPLAY_NAMES.items()
    ]
    rho_handles = [
        plt.Line2D([0], [0], marker=marker, linestyle="", color="black", label=f"rho_train={rho:.1f}")
        for rho, marker in markers.items()
    ]
    first_legend = axis.legend(handles=model_handles, title="Model", loc="upper left")
    axis.add_artist(first_legend)
    axis.legend(handles=rho_handles, title="Training regime", loc="lower right")
    axis.set(
        title="Spurious Reliance vs OOD Degradation",
        xlabel="Validation accuracy drop after permuting x_spurious",
        ylabel="IID accuracy - average OOD accuracy",
    )
    figure.tight_layout()
    _save_both(figure, output_dir / "spurious_reliance_vs_ood_gap")


def build_research_report(
    model_summary: pd.DataFrame,
    ranking_summary: pd.DataFrame,
    reliance_correlation: pd.DataFrame,
    max_stable_gap: float,
    warnings: list[str],
) -> str:
    """Create the required cautious, evidence-labelled EXP-002 report."""
    all_features = model_summary.loc[model_summary["feature_condition"].eq("all_features")]
    key_columns = [
        "rho_train", "model", "validation_accuracy_mean", "average_ood_accuracy_mean",
        "worst_environment_accuracy_mean", "ood_gap_mean", "spurious_reliance_score_mean",
    ]
    table = all_features[key_columns].to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    )
    ranking_table = ranking_summary.to_string(index=False)
    correlation_table = reliance_correlation.to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    )
    warning_text = "; ".join(warnings) if warnings else "None triggered."
    return f"""# EXP-002 — Training Shortcut Strength Sweep

## Question

Does the preliminary IID/OOD model-ranking reversal from EXP-001 persist when training-time shortcut strength changes?

## Experimental Intervention

`rho_train` is varied over `0.5`, `0.7`, and `0.9`; `rho_val` always equals `rho_train`. Each regime is evaluated on `[rho_train, 0.3, 0.0, -0.3, -0.6, -0.9]`. All sample sizes, stable mechanism, feature distributions, label noise, model families, architectures, hyperparameters, preprocessing, metrics, and five seeds are held fixed from EXP-001. OOD data are evaluation-only.

## Controls

OBSERVATION: The stable-feature-only maximum absolute aggregate OOD gap was `{max_stable_gap:.4f}`.

OBSERVATION: Sanity-check warnings: {warning_text}

## Results

OBSERVATION: Aggregate all-feature measurements (mean across seeds):

```text
{table}
```

## Ranking Reversal

OBSERVATION:

```text
{ranking_table}
```

INTERPRETATION: A reversal flag only states that the IID-validation winner and average-OOD winner differ within this controlled configuration. It does not identify a causal model-family property.

## Spurious Reliance

OBSERVATION: Descriptive Pearson correlations between aggregate permutation reliance and OOD gap were:

```text
{correlation_table}
```

HYPOTHESIS: Shortcut reliance may help explain some between-model OOD degradation differences. This requires testing across additional independently varied regimes.

## Unexpected Results

OBSERVATION: {warning_text if warnings else "No configured sanity or stable-only control warning was triggered."}

## What We Can Claim

CLAIM: For these exact synthetic regimes, seeds, fixed models, and fixed hyperparameters, the recorded ranking-reversal table and degradation measurements are reproducible descriptive results.

## What We Cannot Claim

CLAIM: These experiments do not establish that any model family is inherently robust or fragile, that reliance causes degradation, or that the ranking pattern generalizes to real datasets or other shift mechanisms.

## Suggested Next Experiment

HYPOTHESIS: Repeating EXP-002 with 10 seeds, without changing another factor, would provide a more precise estimate of ranking stability before introducing EXP-003.
"""
