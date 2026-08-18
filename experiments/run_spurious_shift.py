"""Run the complete controlled spurious-correlation shift experiment."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation import ALL_FEATURES, STABLE_FEATURES, generate_dataset
from src.diagnostics import permutation_reliance_score, validate_environment_metadata
from src.evaluation import (
    aggregate_model_summary,
    evaluate_environments,
    per_seed_robustness,
    select_and_fit_model,
)
from src.models import MODEL_DISPLAY_NAMES, configured_model_names
from src.plotting import plot_model_selection, plot_shift_curves, plot_spurious_reliance
from src.utils import child_seed, configure_logging, ensure_output_directories, load_config, set_global_seed


LOGGER = logging.getLogger(__name__)


def _dataset(data_config: dict[str, Any], n: int, rho: float, seed: int, split: str):
    return generate_dataset(
        n_samples=n,
        rho=rho,
        seed=seed,
        stable_coefficients=data_config["stable_coefficients"],
        label_noise=float(data_config["label_noise"]),
        split=split,
    )


def run_experiment(config: dict[str, Any], project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    """Execute all seeds, save tidy outputs, figures, and a text report."""
    output = ensure_output_directories(project_root)
    data_config = config["data"]
    model_names = configured_model_names(config["models"])
    result_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []

    for seed in config["seeds"]:
        seed = int(seed)
        set_global_seed(seed)
        LOGGER.info("Generating environments for seed %s", seed)
        train = _dataset(data_config, int(data_config["n_train"]), float(data_config["rho_train"]), child_seed(seed, 1), "train")
        val = _dataset(data_config, int(data_config["n_val"]), float(data_config["rho_val"]), child_seed(seed, 2), "validation")
        tests = {
            float(rho): _dataset(data_config, int(data_config["n_test"]), float(rho), child_seed(seed, 100 + index), f"test_rho_{rho}")
            for index, rho in enumerate(data_config["rho_test"])
        }
        for item in [train, val, *tests.values()]:
            metadata_rows.append({"experiment_seed": seed, **item.metadata})

        feature_conditions = {
            "all_features": ALL_FEATURES,
            "stable_only": STABLE_FEATURES,
        }
        for condition, features in feature_conditions.items():
            for model_name in model_names:
                LOGGER.info("Fitting seed=%s condition=%s model=%s", seed, condition, model_name)
                model, validation_metrics = select_and_fit_model(
                    model_name=model_name,
                    model_config=config["models"][model_name],
                    X_train=train.X[features],
                    y_train=train.y,
                    X_val=val.X[features],
                    y_val=val.y,
                    seed=child_seed(seed, 1000 + model_names.index(model_name)),
                )
                original, permuted, reliance = permutation_reliance_score(
                    model, val.X[features], val.y, child_seed(seed, 2000 + model_names.index(model_name))
                )
                diagnostic_rows.append(
                    {
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
                result_rows.extend(
                    evaluate_environments(model, tests, features, seed, model_name, condition)
                )

    results = pd.DataFrame(result_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    environment_summary = pd.DataFrame(metadata_rows)
    per_seed = per_seed_robustness(results, diagnostics, float(data_config["rho_train"]))
    model_summary = aggregate_model_summary(per_seed)

    results.to_csv(output["raw"] / "results.csv", index=False)
    environment_summary.to_csv(output["summaries"] / "environment_summary.csv", index=False)
    diagnostics.to_csv(output["summaries"] / "spurious_reliance.csv", index=False)
    per_seed.to_csv(output["summaries"] / "per_seed_robustness.csv", index=False)
    model_summary.to_csv(output["summaries"] / "model_summary.csv", index=False)

    plot_shift_curves(results, output["figures"])
    plot_spurious_reliance(model_summary, output["figures"])
    plot_model_selection(model_summary, output["figures"])
    warnings = validate_environment_metadata(environment_summary)
    report = build_report(model_summary, warnings)
    (output["summaries"] / "research_summary.txt").write_text(report, encoding="utf-8")
    print(report)
    return output


def build_report(model_summary: pd.DataFrame, warnings: list[str]) -> str:
    """Build an honest lightweight summary from measured aggregate results."""
    all_features = model_summary.loc[model_summary["feature_condition"] == "all_features"].copy()
    columns = [
        "model", "validation_accuracy_mean", "average_ood_accuracy_mean",
        "worst_environment_accuracy_mean", "ood_gap_mean", "spurious_reliance_score_mean",
    ]
    display = all_features[columns].copy()
    display["model"] = display["model"].map(MODEL_DISPLAY_NAMES)
    iid_best = all_features.loc[all_features["validation_accuracy_mean"].idxmax(), "model"]
    avg_best = all_features.loc[all_features["average_ood_accuracy_mean"].idxmax(), "model"]
    worst_best = all_features.loc[all_features["worst_environment_accuracy_mean"].idxmax(), "model"]
    agrees = iid_best == avg_best == worst_best
    lines = [
        "SPURIOUS SHIFT V1 - AUTOMATED RESEARCH SUMMARY",
        "",
        display.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        f"Best model according to IID validation: {MODEL_DISPLAY_NAMES[iid_best]}",
        f"Best model according to average OOD: {MODEL_DISPLAY_NAMES[avg_best]}",
        f"Best model according to worst-environment performance: {MODEL_DISPLAY_NAMES[worst_best]}",
        "Selection criteria agree." if agrees else "IID and OOD model-selection criteria do not all agree.",
        "",
        "Sanity-check warnings: " + ("; ".join(warnings) if warnings else "none"),
        "",
        "Questions for human interpretation:",
        "1. Which model achieves the highest IID accuracy?",
        "2. Which model is most robust to the spurious shift?",
        "3. Does model ranking change as rho changes?",
        "4. Which model has the largest OOD degradation?",
        "5. Which model relies most strongly on x_spurious?",
        "6. Is spurious reliance associated with OOD degradation?",
        "7. Does removing x_spurious remove most of the OOD gap?",
        "8. Are differences consistent across seeds?",
        "9. Which observations appear model-specific?",
        "10. Which observations may reflect a more general inductive-bias phenomenon?",
        "",
        "Interpretation warning: this synthetic experiment is descriptive and does not establish a universal model ranking or causal mechanism.",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    arguments = parse_args()
    run_experiment(load_config(arguments.config))

