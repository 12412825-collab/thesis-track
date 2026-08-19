"""Preregistered Phase 2 summaries, guards, decision logic, and figures."""

from __future__ import annotations

import hashlib
import json
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

from src.phase2_experiment import PHASE2_PREREG_SHA


PALETTE = {"constant": "#2F5D8C", "cosine": "#C58A2A", "width": "#486B55", "neutral": "#4A4A4A"}


def _bootstrap_mean(values: np.ndarray, n_resamples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(n_resamples, len(values)))]
    means = samples.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _bootstrap_range(matrix: np.ndarray, n_resamples: int, seed: int) -> tuple[float, float]:
    matrix = np.asarray(matrix, dtype=float)
    rng = np.random.default_rng(seed)
    samples = matrix[rng.integers(0, len(matrix), size=(n_resamples, len(matrix)))]
    ranges = samples.mean(axis=1).max(axis=1) - samples.mean(axis=1).min(axis=1)
    return float(np.percentile(ranges, 2.5)), float(np.percentile(ranges, 97.5))


def _load_fit_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []
    for path in sorted((root / "fits").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        base = {key: value for key, value in record.items() if key not in {"trajectory", "ood"}}
        base["source_file"] = path.name
        fit_rows.append(base)
        for row in record.get("trajectory", []):
            trajectory_rows.append({**{key: base[key] for key in ["fit_id", "rho_train", "dataset_seed", "width", "schedule"]}, **row})
        for row in record.get("ood", []):
            ood_rows.append({**{key: base[key] for key in ["fit_id", "rho_train", "dataset_seed", "width", "schedule"]}, **row})
    return pd.DataFrame(fit_rows), pd.DataFrame(trajectory_rows), pd.DataFrame(ood_rows)


def _q1(fits: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    bootstrap_n = int(config["statistics"]["bootstrap_resamples"])
    rows: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    for (rho, width), group in fits.groupby(["rho_train", "width"], sort=True):
        pivot = group.pivot(index="dataset_seed", columns="schedule", values="terminal_fidelity")
        pivot_iid = group.pivot(index="dataset_seed", columns="schedule", values="terminal_iid_accuracy")
        if not {"constant", "cosine"}.issubset(pivot.columns):
            continue
        difference = (pivot["cosine"] - pivot["constant"]).to_numpy(dtype=float)
        low, high = _bootstrap_mean(difference, bootstrap_n, int(rho * 1000) + int(width))
        iid_diff = (pivot_iid["cosine"] - pivot_iid["constant"]).to_numpy(dtype=float)
        rows.append(
            {
                "rho_train": rho,
                "width": width,
                "n_seeds": len(difference),
                "schedule_effect_cosine_minus_constant": float(np.mean(difference)),
                "ci95_low": low,
                "ci95_high": high,
                "positive_seeds": int(np.sum(difference > 0)),
                "negative_seeds": int(np.sum(difference < 0)),
                "zero_seeds": int(np.sum(difference == 0)),
                "iid_accuracy_difference": float(np.mean(iid_diff)),
                "iid_match_guard_pass": bool(abs(np.mean(iid_diff)) < 0.02),
            }
        )
        for seed, value in zip(pivot.index, difference):
            effects.append({"rho_train": rho, "width": width, "dataset_seed": seed, "schedule_effect": float(value)})
    return pd.DataFrame(rows), pd.DataFrame(effects)


def _q2(fits: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    bootstrap_n = int(config["statistics"]["bootstrap_resamples"])
    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for (rho, schedule), group in fits.groupby(["rho_train", "schedule"], sort=True):
        pivot = group.pivot(index="dataset_seed", columns="width", values="terminal_fidelity")
        for width in [64, 512]:
            difference = (pivot[width] - pivot[8]).to_numpy(dtype=float)
            low, high = _bootstrap_mean(difference, bootstrap_n, int(rho * 1000) + width + 100)
            pair_rows.append(
                {
                    "rho_train": rho,
                    "schedule": schedule,
                    "comparison": f"width_{width}_minus_width_8",
                    "n_seeds": len(difference),
                    "width_effect": float(np.mean(difference)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive_seeds": int(np.sum(difference > 0)),
                    "negative_seeds": int(np.sum(difference < 0)),
                }
            )
    for rho, group in fits.groupby("rho_train", sort=True):
        pivot = group.pivot_table(index="dataset_seed", columns="width", values="terminal_fidelity", aggfunc="mean")
        matrix = pivot[[8, 64, 512]].to_numpy(dtype=float)
        low, high = _bootstrap_range(matrix, bootstrap_n, int(rho * 1000) + 900)
        means = pivot.mean(axis=0)
        summary_rows.append(
            {
                "rho_train": rho,
                "n_seeds": len(pivot),
                "width_regime_range": float(means.max() - means.min()),
                "width_at_max_mean_fidelity": int(means.idxmax()),
                "width_at_min_mean_fidelity": int(means.idxmin()),
                "ci95_low": low,
                "ci95_high": high,
                "seed_cluster_ci_excludes_zero": bool(low > 0),
            }
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(summary_rows)


def _q3(fits: pd.DataFrame, q1_effects: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    bootstrap_n = int(config["statistics"]["bootstrap_resamples"])
    rows: list[dict[str, Any]] = []
    for rho, group in q1_effects.groupby("rho_train", sort=True):
        pivot = group.pivot(index="dataset_seed", columns="width", values="schedule_effect")
        matrix = pivot[[8, 64, 512]].to_numpy(dtype=float)
        low, high = _bootstrap_range(matrix, bootstrap_n, int(rho * 1000) + 1700)
        means = pivot.mean(axis=0)
        rows.append(
            {
                "rho_train": rho,
                "interaction_range_I": float(means.max() - means.min()),
                "interaction_ci95_low": low,
                "interaction_ci95_high": high,
                "width_effect_at_8": float(means[8]),
                "width_effect_at_64": float(means[64]),
                "width_effect_at_512": float(means[512]),
                "direction_change_across_width": bool(means.min() < 0 < means.max()),
            }
        )
    return pd.DataFrame(rows)


def _guards(fits: pd.DataFrame, q1: pd.DataFrame, q2: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    guard_rows: list[dict[str, Any]] = []
    for _, row in q1.iterrows():
        guard_rows.append(
            {
                "guard": "IID-match",
                "rho_train": row["rho_train"],
                "width": row["width"],
                "value": abs(float(row["iid_accuracy_difference"])),
                "threshold": 0.02,
                "status": "PASS" if row["iid_match_guard_pass"] else "FAIL",
                "note": "Absolute paired schedule difference in IID-test accuracy.",
            }
        )
    rho0 = q1.loc[np.isclose(q1["rho_train"], 0.0)]
    max_rho0 = float(rho0["schedule_effect_cosine_minus_constant"].abs().max()) if len(rho0) else float("nan")
    guard_rows.append({"guard": "rho=0 sanity", "rho_train": 0.0, "width": "all", "value": max_rho0, "threshold": 0.05, "status": "PASS" if max_rho0 < 0.05 else "FAIL", "note": "Large schedule effect at rho=0 is a general optimization effect."})

    rho0_fits = fits.loc[np.isclose(fits["rho_train"], 0.0)]
    ladder = rho0_fits.groupby("width").agg(train_loss_mean=("terminal_train_loss", "mean"), fidelity_mean=("terminal_fidelity", "mean")).reset_index()
    loss_range = float(ladder["train_loss_mean"].max() - ladder["train_loss_mean"].min()) if len(ladder) else float("nan")
    fidelity_range = float(ladder["fidelity_mean"].max() - ladder["fidelity_mean"].min()) if len(ladder) else float("nan")
    ladder_valid = bool(len(ladder) == 3 and (loss_range > 1e-4 or fidelity_range > 0.01))
    guard_rows.append({"guard": "width ladder", "rho_train": 0.0, "width": "8/64/512", "value": max(loss_range, fidelity_range), "threshold": "diagnostic separation", "status": "PASS" if ladder_valid else "FAIL", "note": "Widths are distinct model regimes; report retains raw loss/fidelity diagnostics."})
    guard_table = pd.DataFrame(guard_rows)
    guard_summary = pd.DataFrame([
        {"guard": "IID-match", "status": "PASS" if not (q1["status"] if False else pd.Series(dtype=bool)).any() else "FAIL"},
        {"guard": "rho=0 sanity", "status": "PASS" if max_rho0 < 0.05 else "FAIL"},
        {"guard": "width ladder", "status": "PASS" if ladder_valid else "FAIL"},
    ])
    guard_summary.loc[guard_summary["guard"] == "IID-match", "status"] = "PASS" if bool((q1["iid_match_guard_pass"]).all()) else "FAIL"
    ladder.to_csv(config["outputs"].get("_ladder_path", "NUL"), index=False) if False else None
    return guard_table, pd.concat([guard_summary, pd.DataFrame([{ "guard": "width ladder diagnostics", "status": "PASS" if ladder_valid else "FAIL", "train_loss_range": loss_range, "fidelity_range": fidelity_range }])], ignore_index=True)


def _trajectory_classification(trajectory: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fit_id, group in trajectory.groupby("fit_id", sort=True):
        group = group.sort_values("checkpoint")
        f20 = float(group.iloc[0]["fidelity"])
        fmin = float(group["fidelity"].min())
        terminal = float(group.iloc[-1]["fidelity"])
        drop = f20 - fmin
        recovery = terminal - fmin
        threshold = float(config["measurements"]["d3_threshold"])
        sat = np.inf
        for prev, curr in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()):
            if prev.d3 >= threshold and curr.d3 >= threshold:
                sat = int(curr.checkpoint)
                break
        if drop >= 0.05 and recovery >= 0.05:
            label = "Escape-shaped"
        elif drop >= 0.05:
            label = "Plateau"
        elif drop < 0.05:
            label = "No-acquisition / no-distortion"
        else:
            label = "Other"
        rows.append({"fit_id": fit_id, "fidelity_min": fmin, "fidelity_drop_from_20": drop, "fidelity_recovery": recovery, "sat": sat, "trajectory_class_recomputed": label})
    return pd.DataFrame(rows)


def decide(q1: pd.DataFrame, q2: pd.DataFrame, q3: pd.DataFrame, guards: pd.DataFrame, fits: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    practical = float(config["statistics"]["practical_threshold"])
    negligible = float(config["statistics"]["negligible_threshold"])
    high = q1.loc[q1["rho_train"].isin([0.7, 0.9])]
    iid_fail = bool((high["iid_match_guard_pass"] == False).any())
    rho0_row = guards.loc[(guards["guard"] == "rho=0 sanity") & (guards["status"] == "FAIL")]
    ladder_fail = bool((guards["guard"] == "width ladder").eq(True).any()) if False else False
    ladder_status = guards.loc[guards["guard"] == "width ladder", "status"]
    ladder_fail = bool(len(ladder_status) and ladder_status.iloc[0] == "FAIL")
    failed_fits = int((fits["status"] != "SUCCESS").sum())
    common_reasons: list[str] = []
    if iid_fail:
        common_reasons.append("IID-match guard failed")
    if len(rho0_row):
        common_reasons.append("rho=0 sanity showed a large general schedule effect")
    if ladder_fail:
        common_reasons.append("width ladder was not diagnostically separated")
    if failed_fits:
        common_reasons.append(f"{failed_fits} fit(s) failed")
    if common_reasons:
        return {"decision": "UNRESOLVED", "rationale": "; ".join(common_reasons), "guard_blocked": True}

    interaction = q3.loc[q3["rho_train"].isin([0.7, 0.9])]
    if len(interaction) and bool(((interaction["interaction_range_I"] >= practical) | interaction["direction_change_across_width"]).any()):
        return {"decision": "Coupled Training Regime", "rationale": "The preregistered schedule contrast varies materially across width, so width/capacity-regime conditions LR time-profile.", "guard_blocked": False}

    q1_high = high.copy()
    schedule_supported = False
    supported_widths: list[int] = []
    for width, group in q1_high.groupby("width"):
        by_rho = {float(row.rho_train): row for row in group.itertuples()}
        if all(rho in by_rho for rho in [0.7, 0.9]):
            a, b = by_rho[0.7], by_rho[0.9]
            if abs(a.schedule_effect_cosine_minus_constant) >= practical and abs(b.schedule_effect_cosine_minus_constant) >= practical and np.sign(a.schedule_effect_cosine_minus_constant) == np.sign(b.schedule_effect_cosine_minus_constant) and not (a.ci95_low <= 0 <= a.ci95_high) and not (b.ci95_low <= 0 <= b.ci95_high):
                schedule_supported = True
                supported_widths.append(int(width))
    if schedule_supported:
        return {"decision": "Path dependence supported", "rationale": f"At fixed width(s) {supported_widths}, the schedule residual was >= 0.05 at both high-rho regimes with paired CIs excluding zero and guards passing.", "guard_blocked": False}

    width_rows = q2.loc[q2["rho_train"].isin([0.7, 0.9])]
    width_supported = False
    if len(width_rows) == 2:
        width_supported = bool((width_rows["width_regime_range"] >= practical).all() and (width_rows["ci95_low"] > 0).all())
    if width_supported and bool((q1_high["schedule_effect_cosine_minus_constant"].abs() < negligible).all()):
        return {"decision": "Width/capacity-regime dependence supported", "rationale": "Schedule residuals were negligible at both high-rho regimes while the paired width-regime range was >= 0.05 with seed-cluster CIs excluding zero.", "guard_blocked": False}

    q1_max = float(q1_high["schedule_effect_cosine_minus_constant"].abs().max()) if len(q1_high) else float("nan")
    q2_max = float(width_rows["width_regime_range"].max()) if len(width_rows) else float("nan")
    if q1_max < negligible and q2_max < negligible:
        return {"decision": "Both weak / STOP", "rationale": "Both high-rho schedule and width-regime effects were below the preregistered 0.02 gate while guards passed.", "guard_blocked": False}
    return {"decision": "UNRESOLVED", "rationale": "The observed effects did not satisfy a frozen interpretable decision case.", "guard_blocked": False}


def _plot_trajectories(trajectory: pd.DataFrame, output: Path, metric: str, filename: str, ylabel: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, rho in zip(axes, [0.7, 0.9]):
        subset = trajectory[np.isclose(trajectory["rho_train"], rho)]
        for (width, schedule), group in subset.groupby(["width", "schedule"], sort=True):
            summary = group.groupby("checkpoint")[metric].agg(["mean", "sem"]).reset_index()
            color = PALETTE[schedule]
            linestyle = "-" if width == 8 else ("--" if width == 64 else ":")
            ax.plot(summary["checkpoint"], summary["mean"], color=color, linestyle=linestyle, marker="o", markersize=3, label=f"w={width}, {schedule}")
            ax.fill_between(summary["checkpoint"], summary["mean"] - summary["sem"].fillna(0), summary["mean"] + summary["sem"].fillna(0), color=color, alpha=0.08)
        ax.set_title(f"rho_train = {rho:g}")
        ax.set_xlabel("Epoch checkpoint")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"{ylabel} trajectories by width and LR time-profile", y=1.10, fontsize=13)
    fig.tight_layout()
    fig.savefig(output / filename, dpi=180, bbox_inches="tight")
    fig.savefig(output / filename.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_q1(q1: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for color, rho in zip(["#2F5D8C", "#C58A2A"], [0.7, 0.9]):
        group = q1[np.isclose(q1["rho_train"], rho)].sort_values("width")
        ax.errorbar(group["width"].astype(str), group["schedule_effect_cosine_minus_constant"], yerr=[group["schedule_effect_cosine_minus_constant"] - group["ci95_low"], group["ci95_high"] - group["schedule_effect_cosine_minus_constant"]], fmt="o-", color=color, capsize=3, label=f"rho_train={rho:g}")
    ax.axhline(0, color=PALETTE["neutral"], linewidth=0.9)
    ax.axhline(0.05, color="#999999", linestyle="--", linewidth=0.8)
    ax.axhline(-0.05, color="#999999", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Width / capacity-regime")
    ax.set_ylabel("Fidelity(200): cosine − constant")
    ax.set_title("Paired schedule residual by width")
    ax.set_xlabel("Width / capacity-regime (model width)")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "02_paired_schedule_effect_by_width.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "02_paired_schedule_effect_by_width.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_terminal(fits: pd.DataFrame, output: Path) -> None:
    summary = fits[fits["rho_train"].isin([0.7, 0.9])].groupby(["rho_train", "width", "schedule"])["terminal_fidelity"].agg(["mean", "sem"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, rho in zip(axes, [0.7, 0.9]):
        group = summary[np.isclose(summary["rho_train"], rho)]
        for schedule in ["constant", "cosine"]:
            points = group[group["schedule"] == schedule].sort_values("width")
            ax.errorbar(points["width"].astype(str), points["mean"], yerr=points["sem"].fillna(0), marker="o", linestyle="-", color=PALETTE[schedule], capsize=3, label=schedule)
        ax.set_title(f"rho_train = {rho:g}")
        ax.set_xlabel("Width / capacity-regime")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    axes[0].set_ylabel("Terminal boundary fidelity, Fidelity(200)")
    axes[1].legend(frameon=False)
    fig.suptitle("Terminal boundary fidelity by width and LR time-profile", y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "03_terminal_fidelity_by_width.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "03_terminal_fidelity_by_width.pdf", bbox_inches="tight")
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_phase2(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fits, trajectory, ood = _load_fit_tables(root)
    if fits.empty:
        raise RuntimeError(f"No fit records found under {root / 'fits'}")
    fits.to_csv(root / "raw_fit_results.csv", index=False)
    trajectory.to_csv(root / "checkpoint_trajectories.csv", index=False)
    ood.to_csv(root / "ood_results.csv", index=False)
    q1, q1_effects = _q1(fits[fits["status"] == "SUCCESS"], config)
    q2_pairs, q2 = _q2(fits[fits["status"] == "SUCCESS"], config)
    q3 = _q3(fits[fits["status"] == "SUCCESS"], q1_effects, config)
    guard_table, guard_summary = _guards(fits[fits["status"] == "SUCCESS"], q1, q2, config)
    classes = _trajectory_classification(trajectory, config)
    q1.to_csv(root / "q1_schedule_effects.csv", index=False)
    q1_effects.to_csv(root / "q1_seed_effects.csv", index=False)
    q2_pairs.to_csv(root / "q2_pairwise_width_effects.csv", index=False)
    q2.to_csv(root / "q2_width_regime_summary.csv", index=False)
    q3.to_csv(root / "q3_interaction_summary.csv", index=False)
    guard_table.to_csv(root / "guards_detail.csv", index=False)
    guard_summary.to_csv(root / "guards_summary.csv", index=False)
    classes.to_csv(root / "trajectory_classification.csv", index=False)
    _plot_trajectories(trajectory, figures, "fidelity", "01_fidelity_trajectories.png", "Boundary fidelity")
    _plot_q1(q1, figures)
    _plot_terminal(fits[fits["status"] == "SUCCESS"], figures)
    _plot_trajectories(trajectory, figures, "d3", "04_d3_trajectories.png", "D3 shortcut-sensitivity")
    decision = decide(q1, q2, q3, guard_table, fits, config)
    summary = {
        "prereg_frozen_sha": PHASE2_PREREG_SHA,
        "expected_fits": 360,
        "observed_fit_records": int(len(fits)),
        "successful_fits": int((fits["status"] == "SUCCESS").sum()),
        "failed_fits": int((fits["status"] != "SUCCESS").sum()),
        "decision": decision,
        "trajectory_class_counts": classes["trajectory_class_recomputed"].value_counts().to_dict() if not classes.empty else {},
        "primary_endpoint": "Fidelity(200)",
        "no_phase3_started": True,
    }
    (root / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    manifest: dict[str, Any] = {"prereg_frozen_sha": PHASE2_PREREG_SHA, "files": {}}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][str(path.relative_to(root)).replace("\\", "/")] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return summary
