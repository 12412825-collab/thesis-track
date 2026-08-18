"""Publication-oriented figures for shift robustness and shortcut reliance."""

from __future__ import annotations

import os
from pathlib import Path

# Some minimal Windows runners omit WINDIR even though Matplotlib needs it for
# system-font discovery. Normal user environments keep their existing value.
if os.name == "nt":
    os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models import MODEL_DISPLAY_NAMES


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
    })


def plot_shift_curves(results: pd.DataFrame, output_dir: Path) -> None:
    """Plot mean accuracy with seed-level standard-error bands."""
    _style()
    filenames = {
        "all_features": "all_features_shift_curve",
        "stable_only": "stable_only_shift_curve",
    }
    titles = {
        "all_features": "Distribution Shift with All Features",
        "stable_only": "Distribution Shift with Stable Features Only",
    }
    for condition, subset in results.groupby("feature_condition", sort=False):
        figure, axis = plt.subplots(figsize=(8.2, 5.2))
        for model, group in subset.groupby("model", sort=False):
            stats = group.groupby("rho_test")["accuracy"].agg(["mean", "sem"]).reset_index()
            stats = stats.sort_values("rho_test")
            x = stats["rho_test"].to_numpy()
            mean = stats["mean"].to_numpy()
            sem = stats["sem"].fillna(0.0).to_numpy()
            axis.plot(x, mean, marker="o", linewidth=2, label=MODEL_DISPLAY_NAMES[model])
            axis.fill_between(x, mean - sem, mean + sem, alpha=0.16)
        axis.axvline(0.9, color="black", linestyle="--", linewidth=1, alpha=0.65, label="Training rho")
        axis.set(title=titles[condition], xlabel="Spurious correlation (rho)", ylabel="Classification accuracy")
        axis.set_xticks(sorted(subset["rho_test"].unique()))
        axis.set_ylim(0.0, 1.02)
        axis.legend(ncol=2, frameon=True)
        figure.tight_layout()
        base = output_dir / filenames[condition]
        figure.savefig(base.with_suffix(".png"), bbox_inches="tight")
        figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)


def plot_spurious_reliance(model_summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot shortcut reliance against OOD degradation for all-feature models."""
    _style()
    data = model_summary.loc[model_summary["feature_condition"] == "all_features"]
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    x = data["spurious_reliance_score_mean"]
    y = data["ood_gap_mean"]
    axis.errorbar(
        x,
        y,
        xerr=data["spurious_reliance_score_se"],
        yerr=data["ood_gap_se"],
        fmt="o",
        capsize=3,
        markersize=7,
    )
    annotation_offsets = {
        "logistic_regression": (12, 12),
        "rbf_svm": (8, -2),
        "random_forest": (8, 6),
        "gradient_boosting": (8, 7),
        "mlp": (8, 6),
    }
    for _, row in data.iterrows():
        axis.annotate(
            MODEL_DISPLAY_NAMES[row["model"]],
            (row["spurious_reliance_score_mean"], row["ood_gap_mean"]),
            xytext=annotation_offsets[row["model"]],
            textcoords="offset points",
            fontsize=9,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(
        title="Spurious Reliance and OOD Degradation",
        xlabel="Validation accuracy drop after permuting x_spurious",
        ylabel="IID accuracy - average OOD accuracy",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "spurious_reliance_vs_ood_gap.png", bbox_inches="tight")
    plt.close(figure)


def plot_model_selection(model_summary: pd.DataFrame, output_dir: Path) -> None:
    """Compare IID validation, average OOD, and worst-environment accuracy."""
    _style()
    data = model_summary.loc[model_summary["feature_condition"] == "all_features"].copy()
    labels = [MODEL_DISPLAY_NAMES[name] for name in data["model"]]
    positions = np.arange(len(data))
    width = 0.25
    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    for offset, column, label in [
        (-width, "validation_accuracy_mean", "IID validation"),
        (0.0, "average_ood_accuracy_mean", "Average OOD"),
        (width, "worst_environment_accuracy_mean", "Worst environment"),
    ]:
        axis.bar(positions + offset, data[column], width, label=label)
    axis.set_xticks(positions, labels, rotation=18, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Accuracy")
    axis.set_title("Model Selection Criteria (All Features)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "model_selection_comparison.png", bbox_inches="tight")
    plt.close(figure)
