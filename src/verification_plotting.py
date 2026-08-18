"""Seven preregistered static figures for EXP-002V."""

from __future__ import annotations

import os
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models import MODEL_DISPLAY_NAMES
from src.verification_statistics import MODEL_ORDER


MODEL_COLORS = {
    "logistic_regression": "#2563A6",
    "rbf_svm": "#D97706",
    "random_forest": "#6B8E23",
    "gradient_boosting": "#C2413A",
    "mlp": "#9A5EB5",
}
RHO_MARKERS = {0.0: "o", 0.5: "s", 0.7: "D", 0.9: "^"}


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
        }
    )


def _save(figure: plt.Figure, output_dir: Path, name: str) -> None:
    figure.savefig(output_dir / f"{name}.png", bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def _model_legend(axis: plt.Axes, **kwargs: object) -> None:
    handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS[name], marker="o", label=MODEL_DISPLAY_NAMES[name])
        for name in MODEL_ORDER
    ]
    axis.legend(handles=handles, **kwargs)


def plot_h1_ood_gap(h1: pd.DataFrame, output_dir: Path) -> None:
    """Figure 1: rho_train versus OOD gap with 95% bootstrap CI."""
    _style()
    levels = h1.loc[h1["statistic_type"].eq("ood_gap_level")]
    figure, axis = plt.subplots(figsize=(8.4, 5.5))
    for model in MODEL_ORDER:
        group = levels.loc[levels["model"].eq(model)].sort_values("rho_to")
        axis.errorbar(
            group["rho_to"], group["estimate"],
            yerr=[group["estimate"] - group["ci_lower"], group["ci_upper"] - group["estimate"]],
            marker="o", linewidth=2, capsize=3, color=MODEL_COLORS[model],
            label=MODEL_DISPLAY_NAMES[model],
        )
    axis.axhline(0.0, color="#333333", linewidth=1)
    axis.set(
        title="Training Shortcut Strength vs OOD Degradation Gap",
        xlabel="Training shortcut correlation (rho_train)",
        ylabel="IID accuracy - average OOD accuracy",
        xticks=[0.0, 0.5, 0.7, 0.9],
    )
    axis.legend(ncol=2)
    figure.tight_layout()
    _save(figure, output_dir, "01_rho_train_vs_ood_gap")


def plot_d3_by_rho(d3_summary: pd.DataFrame, output_dir: Path) -> None:
    """Figure 2: rho_train versus primary D3_excess reliance."""
    _style()
    figure, axis = plt.subplots(figsize=(8.4, 5.5))
    for model in MODEL_ORDER:
        group = d3_summary.loc[d3_summary["model"].eq(model)].sort_values("rho_train")
        axis.errorbar(
            group["rho_train"], group["mean"],
            yerr=[group["mean"] - group["ci_lower"], group["ci_upper"] - group["mean"]],
            marker="o", linewidth=2, capsize=3, color=MODEL_COLORS[model],
            label=MODEL_DISPLAY_NAMES[model],
        )
    axis.axhline(0.0, color="#333333", linewidth=1)
    axis.set(
        title="Training Shortcut Strength vs D3 Excess Flip Rate",
        xlabel="Training shortcut correlation (rho_train)",
        ylabel="D3 flip rate(x_spurious) - D3 flip rate(z)",
        xticks=[0.0, 0.5, 0.7, 0.9],
    )
    axis.legend(ncol=2)
    figure.tight_layout()
    _save(figure, output_dir, "02_rho_train_vs_d3_excess")


def _scatter_by_model_and_rho(
    axis: plt.Axes, data: pd.DataFrame, x: str, y: str
) -> None:
    for model in MODEL_ORDER:
        for rho_train, group in data.loc[data["model"].eq(model)].groupby("rho_train"):
            axis.scatter(
                group[x], group[y], s=30, alpha=0.58,
                color=MODEL_COLORS[model], marker=RHO_MARKERS[float(rho_train)],
                edgecolor="white", linewidth=0.35,
            )


def _add_scatter_legends(axis: plt.Axes) -> None:
    model_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=MODEL_COLORS[name], label=MODEL_DISPLAY_NAMES[name])
        for name in MODEL_ORDER
    ]
    rho_handles = [
        plt.Line2D([0], [0], marker=marker, linestyle="", color="#222222", label=f"rho_train={rho:.1f}")
        for rho, marker in RHO_MARKERS.items()
    ]
    legend_style = {
        "frameon": True,
        "facecolor": "white",
        "framealpha": 0.94,
        "edgecolor": "#D0D0D0",
    }
    model_legend = axis.legend(
        handles=model_handles,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        **legend_style,
    )
    axis.add_artist(model_legend)
    axis.legend(
        handles=rho_handles,
        title="Training regime",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        **legend_style,
    )


def plot_d3_vs_gap(analysis: pd.DataFrame, output_dir: Path) -> None:
    """Figure 3: replicate-level D3_excess versus OOD gap."""
    _style()
    figure, axis = plt.subplots(figsize=(8.5, 6.0))
    _scatter_by_model_and_rho(axis, analysis, "d3_excess", "ood_gap")
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.axvline(0.0, color="#333333", linewidth=0.8)
    axis.set(
        title="D3 Excess Flip Rate vs OOD Degradation Gap",
        xlabel="D3 excess flip rate (IID validation only)",
        ylabel="IID accuracy - average OOD accuracy",
    )
    _add_scatter_legends(axis)
    figure.tight_layout()
    _save(figure, output_dir, "03_d3_excess_vs_ood_gap")


def plot_loco_vs_gap(analysis: pd.DataFrame, output_dir: Path) -> None:
    """Figure 4: replicate-level raw LOCO versus OOD gap."""
    _style()
    figure, axis = plt.subplots(figsize=(8.5, 6.0))
    _scatter_by_model_and_rho(axis, analysis, "raw_loco", "ood_gap")
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.axvline(0.0, color="#333333", linewidth=0.8)
    axis.set(
        title="LOCO Accuracy Difference vs OOD Degradation Gap",
        xlabel="IID validation accuracy(all) - accuracy(stable-only)",
        ylabel="IID accuracy - average OOD accuracy",
    )
    _add_scatter_legends(axis)
    figure.tight_layout()
    _save(figure, output_dir, "04_loco_vs_ood_gap")


def plot_iid_vs_ood(analysis: pd.DataFrame, output_dir: Path) -> None:
    """Figure 5: independent IID test versus average OOD accuracy."""
    _style()
    figure, axis = plt.subplots(figsize=(8.5, 6.0))
    _scatter_by_model_and_rho(axis, analysis, "iid_accuracy", "average_ood_accuracy")
    limits = [
        min(analysis["iid_accuracy"].min(), analysis["average_ood_accuracy"].min()) - 0.02,
        max(analysis["iid_accuracy"].max(), analysis["average_ood_accuracy"].max()) + 0.02,
    ]
    axis.plot(limits, limits, linestyle="--", color="#333333", linewidth=1, label="Equal accuracy")
    axis.set(
        title="IID Test Accuracy vs Average OOD Accuracy",
        xlabel="IID test accuracy",
        ylabel="Average OOD accuracy",
        xlim=limits,
        ylim=limits,
    )
    _add_scatter_legends(axis)
    figure.tight_layout()
    _save(figure, output_dir, "05_iid_accuracy_vs_ood_accuracy")


def plot_lr_mlp_probabilities(
    pairwise: pd.DataFrame, reversal: pd.DataFrame, output_dir: Path
) -> None:
    """Figure 6: LR/MLP pairwise and overall reversal probabilities."""
    _style()
    rhos = sorted(pairwise["rho_train"].unique())
    iid = pairwise.loc[
        pairwise["metric"].eq("iid_test")
        & pairwise["model_a"].eq("logistic_regression")
        & pairwise["model_b"].eq("mlp")
    ].set_index("rho_train")
    ood = pairwise.loc[
        pairwise["metric"].eq("average_ood")
        & pairwise["model_a"].eq("mlp")
        & pairwise["model_b"].eq("logistic_regression")
    ].set_index("rho_train")
    rev = reversal.set_index("rho_train")
    positions = np.arange(len(rhos))
    width = 0.25
    figure, axis = plt.subplots(figsize=(8.6, 5.4))
    axis.bar(
        positions - width,
        [iid.loc[rho, "probability_a_beats_b"] for rho in rhos],
        width, color="#2563A6", edgecolor="#1D3557", label="P(LR > MLP IID)",
    )
    axis.bar(
        positions,
        [ood.loc[rho, "probability_a_beats_b"] for rho in rhos],
        width, color="#D97706", edgecolor="#8A4B08", label="P(MLP > LR OOD)",
    )
    axis.bar(
        positions + width,
        [rev.loc[rho, "probability_iid_best_differs_from_ood_best"] for rho in rhos],
        width, color="#F2D4A7", edgecolor="#333333", hatch="//", label="P(any best-model reversal)",
    )
    axis.set(
        title="LR vs MLP Ordering and Best-Model Reversal Probability",
        xlabel="Training shortcut correlation (rho_train)",
        ylabel="Replicate probability",
        ylim=(0.0, 1.05),
    )
    axis.set_xticks(positions, [f"{rho:.1f}" for rho in rhos])
    axis.legend(loc="upper left")
    figure.tight_layout()
    _save(figure, output_dir, "06_lr_vs_mlp_ranking_probability")


def plot_stable_only_control(
    stable_summary: pd.DataFrame, output_dir: Path
) -> None:
    """Figure 7: stable-only performance across explicit test environments."""
    _style()
    rhos = sorted(stable_summary["rho_train"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), sharex=False, sharey=True)
    for axis, rho_train in zip(axes.flat, rhos):
        regime = stable_summary.loc[np.isclose(stable_summary["rho_train"], rho_train)]
        for model in MODEL_ORDER:
            group = regime.loc[regime["model"].eq(model)].sort_values("rho_environment")
            axis.errorbar(
                group["rho_environment"], group["mean"],
                yerr=[group["mean"] - group["ci_lower"], group["ci_upper"] - group["mean"]],
                marker="o", linewidth=1.5, capsize=2, color=MODEL_COLORS[model],
            )
        axis.axvline(rho_train, linestyle="--", linewidth=1, color="#333333")
        axis.set_title(f"rho_train={rho_train:.1f}")
        axis.set_xlabel("Environment shortcut correlation")
        axis.set_xticks(sorted(regime["rho_environment"].unique()))
        axis.set_ylim(0.84, 0.94)
    axes[0, 0].set_ylabel("Stable-only accuracy")
    axes[1, 0].set_ylabel("Stable-only accuracy")
    _model_legend(axes[0, 0], ncol=2, loc="lower left")
    figure.suptitle("Stable-Only Performance Across IID and OOD Environments")
    figure.tight_layout()
    _save(figure, output_dir, "07_stable_only_performance")
