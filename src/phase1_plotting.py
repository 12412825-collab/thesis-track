"""Publication figure generation for Phase 1."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("WINDIR", r"C:\Windows")
_MPL_CACHE = Path(tempfile.gettempdir()) / "phase1-matplotlib-cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.phase1_models import FAMILY_DISPLAY_NAMES


COLORS = {
    "logistic_regression": "#0072B2",
    "rbf_svm": "#E69F00",
    "random_forest": "#009E73",
    "gradient_boosting": "#D55E00",
    "mlp": "#CC79A7",
}
MARKERS = dict(zip(COLORS, ["o", "s", "^", "D", "P"]))
LINESTYLES = dict(zip(COLORS, ["-", "--", "-.", ":", (0, (5, 1))]))


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
        }
    )


def _save(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_variance_decomposition(table: pd.DataFrame, output: Path) -> None:
    _style()
    labels = {
        "family": "Family", "rho_train": "Shift strength", "family_x_rho": "Family × shift",
        "config_within_family": "Config within family",
        "config_within_family_x_rho": "Config × shift",
        "seed": "Paired seed", "residual": "Residual",
    }
    data = table.copy()
    positions = np.arange(len(data))
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    values = data["variance_share"].to_numpy()
    errors = np.vstack([values - data["ci_lower"].to_numpy(), data["ci_upper"].to_numpy() - values])
    axis.bar(positions, values, color="#4C78A8", edgecolor="black", yerr=errors, capsize=3)
    axis.set_xticks(positions, [labels[x] for x in data["component"]], rotation=28, ha="right")
    axis.set_ylabel("Share of corrected OOD-gap variance")
    axis.set_title("Phase 1 variance decomposition (seed-cluster 95% bootstrap CI)")
    axis.set_ylim(0, max(0.05, float(data["ci_upper"].max()) * 1.12))
    for position, value in zip(positions, values):
        axis.text(position, value, f"{value:.1%}", ha="center", va="bottom", fontsize=8)
    figure.tight_layout()
    _save(figure, output)


def plot_matched_capacity(
    adjusted: pd.DataFrame, matching: pd.DataFrame, output: Path
) -> None:
    _style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    for family in COLORS:
        group = adjusted.loc[adjusted["family"].eq(family)].sort_values("rho_train")
        left.errorbar(
            group["rho_train"], group["mean"], yerr=1.96 * group["sem"],
            color=COLORS[family], marker=MARKERS[family], linestyle=LINESTYLES[family],
            capsize=2, label=FAMILY_DISPLAY_NAMES[family],
        )
    left.axhline(0, color="black", linewidth=1)
    left.set(xlabel="Training shortcut strength (rho_train)", ylabel="M1-adjusted OOD-gap residual", title="Capacity/IID-adjusted family residuals")
    left.set_xticks(sorted(adjusted["rho_train"].unique()))
    left.legend(ncol=2)
    coverage = matching.groupby("capacity_tier")["coverage"].agg(["mean", "min"]).reindex(["low", "medium", "high"])
    x = np.arange(len(coverage))
    right.bar(x - 0.18, coverage["mean"], 0.36, label="Mean coverage", color="#59A14F")
    right.bar(x + 0.18, coverage["min"], 0.36, label="Minimum coverage", color="#F28E2B")
    right.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Adequacy gate")
    right.set_xticks(x, [value.title() for value in coverage.index])
    right.set_ylim(0, 1.05)
    right.set(xlabel="Declared capacity tier", ylabel="Pairwise matching coverage", title="Preregistered match coverage")
    right.legend()
    figure.suptitle("Matched-capacity comparison; error bars are 95% normal intervals")
    figure.tight_layout()
    _save(figure, output)


def plot_robustness_envelope(table: pd.DataFrame, output: Path) -> None:
    _style()
    figure, axis = plt.subplots(figsize=(9.0, 5.4))
    for family in COLORS:
        group = table.loc[table["family"].eq(family)].sort_values("rho_train")
        x = group["rho_train"].to_numpy(float)
        axis.fill_between(x, group["ood_gap_q10"], group["ood_gap_q90"], color=COLORS[family], alpha=0.14)
        axis.plot(
            x, group["ood_gap_median"], color=COLORS[family], marker=MARKERS[family],
            linestyle=LINESTYLES[family], linewidth=2, label=FAMILY_DISPLAY_NAMES[family],
        )
    axis.set_xticks(sorted(table["rho_train"].unique()))
    axis.set(xlabel="Training shortcut strength (rho_train)", ylabel="OOD gap (IID accuracy − mean OOD accuracy)", title="Family robustness envelopes across configurations and seeds")
    axis.legend(ncol=2)
    axis.text(0.01, 0.98, "Bands: 10th–90th percentile; lines: median", transform=axis.transAxes, va="top")
    figure.tight_layout()
    _save(figure, output)


def plot_ranking_reversal(table: pd.DataFrame, output: Path) -> None:
    _style()
    x = np.arange(len(table))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    axis.bar(x - width / 2, table["configuration_reversal_probability"], width, color="#4C78A8", label="Configuration-level")
    axis.bar(x + width / 2, table["family_reversal_probability"], width, color="#F58518", hatch="//", label="Family-level")
    axis.set_xticks(x, [f"{rho:.1f}" for rho in table["rho_train"]])
    axis.set_ylim(0, 1.05)
    axis.set(xlabel="Training shortcut strength (rho_train)", ylabel="Probability IID-best and OOD-best sets are disjoint", title="Tie-aware IID/OOD ranking reversal across 10 paired seeds")
    axis.legend()
    axis.text(0.01, 0.98, "Absolute tie tolerance: 0.001 accuracy", transform=axis.transAxes, va="top")
    figure.tight_layout()
    _save(figure, output)


def plot_wrong_line(table: pd.DataFrame, output: Path) -> None:
    _style()
    geometries = list(table["geometry"].drop_duplicates())
    rhos = sorted(table["rho_train"].unique())
    tiers = ["low", "medium", "high"]
    figure, axes = plt.subplots(len(geometries), len(rhos), figsize=(13.2, 10.0), sharex=True, sharey=True)
    for row, geometry in enumerate(geometries):
        for column, rho in enumerate(rhos):
            axis = axes[row, column]
            subset = table.loc[table["geometry"].eq(geometry) & np.isclose(table["rho_train"], rho)]
            for family in COLORS:
                group = subset.loc[subset["family"].eq(family)].groupby("capacity_tier")["boundary_fidelity"].mean().reindex(tiers)
                axis.plot(
                    np.arange(3), group, color=COLORS[family], marker=MARKERS[family],
                    linestyle=LINESTYLES[family], label=FAMILY_DISPLAY_NAMES[family],
                )
            axis.axhline(0.95, color="black", linestyle="--", linewidth=0.8)
            if row == 0:
                axis.set_title(f"rho_train={rho:.1f}")
            if column == 0:
                axis.set_ylabel(f"{geometry}\nBoundary fidelity")
            if row == len(geometries) - 1:
                axis.set_xticks(np.arange(3), [x.title() for x in tiers])
                axis.set_xlabel("Capacity tier")
            axis.set_ylim(0.55, 1.01)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
    figure.suptitle("Noise-free stable-boundary wrong-line guard (means across 3 seeds)")
    figure.tight_layout(rect=[0, 0.04, 1, 0.97])
    _save(figure, output)


def make_all_figures(
    variance: pd.DataFrame,
    adjusted: pd.DataFrame,
    matching: pd.DataFrame,
    envelope: pd.DataFrame,
    reversal: pd.DataFrame,
    wrong_line: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_variance_decomposition(variance, output_dir / "01_variance_decomposition")
    plot_matched_capacity(adjusted, matching, output_dir / "02_matched_capacity_comparison")
    plot_robustness_envelope(envelope, output_dir / "03_robustness_envelope")
    plot_ranking_reversal(reversal, output_dir / "04_iid_ood_ranking_reversal")
    plot_wrong_line(wrong_line, output_dir / "05_wrong_line_guard")
