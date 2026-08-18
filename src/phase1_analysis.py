"""Preregistered statistical analyses for Phase 1."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


COMPONENTS = [
    "family",
    "rho_train",
    "family_x_rho",
    "config_within_family",
    "config_within_family_x_rho",
    "seed",
    "residual",
]


def fit_level_table(master: pd.DataFrame) -> pd.DataFrame:
    """Collapse long environment rows to one successful row per fitted model."""
    successful = master.loc[master["status"].eq("success")].copy()
    columns = [
        "fit_id", "config_id", "family", "capacity_tier", "capacity_tier_score",
        "regularization_tier", "regularization_flexibility_score", "rho_train", "seed",
        "iid_accuracy", "average_ood_accuracy", "worst_ood_accuracy", "ood_gap",
        "learned_capacity_primary", "empirical_capacity_index", "converged",
    ]
    table = successful[columns].drop_duplicates("fit_id").reset_index(drop=True)
    if table.duplicated(["config_id", "rho_train", "seed"]).any():
        raise ValueError("Duplicate fit-level scientific key")
    return table


def _variance_components(data: pd.DataFrame, seed_column: str = "seed") -> dict[str, float]:
    y = data["ood_gap"].astype(float)
    grand = float(y.mean())
    family_mean = data.groupby("family")["ood_gap"].transform("mean")
    rho_mean = data.groupby("rho_train")["ood_gap"].transform("mean")
    family_rho_mean = data.groupby(["family", "rho_train"])["ood_gap"].transform("mean")
    config_mean = data.groupby(["family", "config_id"])["ood_gap"].transform("mean")
    config_rho_mean = data.groupby(["family", "config_id", "rho_train"])["ood_gap"].transform("mean")
    seed_mean = data.groupby(seed_column)["ood_gap"].transform("mean")
    effects = {
        "family": family_mean - grand,
        "rho_train": rho_mean - grand,
        "family_x_rho": family_rho_mean - family_mean - rho_mean + grand,
        "config_within_family": config_mean - family_mean,
        "config_within_family_x_rho": config_rho_mean - config_mean - family_rho_mean + family_mean,
        "seed": seed_mean - grand,
    }
    fitted = grand + sum(effects.values())
    effects["residual"] = y - fitted
    total = float(np.square(y - grand).sum())
    result = {name: float(np.square(values).sum()) for name, values in effects.items()}
    result["total"] = total
    return result


def variance_decomposition(
    data: pd.DataFrame, bootstrap_resamples: int, seed: int
) -> pd.DataFrame:
    """Compute balanced orthogonal shares and seed-cluster bootstrap intervals."""
    observed = _variance_components(data)
    rng = np.random.default_rng(seed)
    seeds = np.array(sorted(data["seed"].unique()))
    bootstrap = {name: [] for name in COMPONENTS}
    for _ in range(bootstrap_resamples):
        parts = []
        for pseudo_seed, selected in enumerate(rng.choice(seeds, len(seeds), replace=True)):
            part = data.loc[data["seed"].eq(selected)].copy()
            part["bootstrap_seed"] = pseudo_seed
            parts.append(part)
        sampled = pd.concat(parts, ignore_index=True)
        values = _variance_components(sampled, "bootstrap_seed")
        for name in COMPONENTS:
            bootstrap[name].append(values[name] / values["total"] if values["total"] else np.nan)
    rows = []
    for name in COMPONENTS:
        samples = np.asarray(bootstrap[name], dtype=float)
        rows.append(
            {
                "component": name,
                "sum_squares": observed[name],
                "variance_share": observed[name] / observed["total"],
                "ci_lower": float(np.nanquantile(samples, 0.025)),
                "ci_upper": float(np.nanquantile(samples, 0.975)),
                "total_sum_squares": observed["total"],
            }
        )
    return pd.DataFrame(rows)


FAMILIES = [
    "logistic_regression", "rbf_svm", "random_forest", "gradient_boosting", "mlp"
]
RHO_VALUES = [0.0, 0.5, 0.7, 0.9]


def _design_matrix(
    data: pd.DataFrame,
    model: str,
    *,
    include_seed: bool,
    seed_column: str = "seed",
    capacity_column: str = "empirical_capacity_index",
    include_iid: bool = True,
    include_regularization: bool = True,
) -> np.ndarray:
    columns: list[np.ndarray] = [np.ones(len(data))]
    rho_dummies: list[np.ndarray] = []
    for rho in RHO_VALUES[1:]:
        values = np.isclose(data["rho_train"].to_numpy(float), rho).astype(float)
        columns.append(values)
        rho_dummies.append(values)
    if include_seed:
        seed_values = sorted(data[seed_column].unique())
        columns.extend((data[seed_column].to_numpy() == value).astype(float) for value in seed_values[1:])
    if model in {"M1", "M2", "M3"}:
        if include_iid:
            columns.append(data["iid_accuracy"].to_numpy(float))
        columns.append(data[capacity_column].to_numpy(float))
        if include_regularization:
            columns.append(data["regularization_flexibility_score"].to_numpy(float))
    family_dummies: list[np.ndarray] = []
    if model in {"M2", "M3"}:
        for family in FAMILIES[1:]:
            values = data["family"].eq(family).to_numpy(float)
            columns.append(values)
            family_dummies.append(values)
    if model == "M3":
        columns.extend(family * rho for family in family_dummies for rho in rho_dummies)
    return np.column_stack(columns)


def _fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ coefficients
    sse = float(residual @ residual)
    total = float(np.square(y - y.mean()).sum())
    return coefficients, 1.0 - sse / total if total else np.nan, sse


def _cross_validated_predictions(
    data: pd.DataFrame,
    model: str,
    *,
    capacity_column: str = "empirical_capacity_index",
    include_iid: bool = True,
    include_regularization: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    predictions = np.empty(len(data), dtype=float)
    fold_rows = []
    for held_out in sorted(data["seed"].unique()):
        train_mask = ~data["seed"].eq(held_out).to_numpy()
        test_mask = ~train_mask
        train = data.loc[train_mask]
        test = data.loc[test_mask]
        X_train = _design_matrix(
            train, model, include_seed=False, capacity_column=capacity_column,
            include_iid=include_iid, include_regularization=include_regularization,
        )
        X_test = _design_matrix(
            test, model, include_seed=False, capacity_column=capacity_column,
            include_iid=include_iid, include_regularization=include_regularization,
        )
        coefficients = np.linalg.lstsq(X_train, train["ood_gap"].to_numpy(float), rcond=None)[0]
        fold_prediction = X_test @ coefficients
        predictions[test_mask] = fold_prediction
        error = test["ood_gap"].to_numpy(float) - fold_prediction
        fold_rows.append({"seed": held_out, "model": model, "sse": float(error @ error), "n": len(test)})
    return predictions, pd.DataFrame(fold_rows)


def regression_models(
    data: pd.DataFrame,
    *,
    capacity_column: str = "empirical_capacity_index",
    include_iid: bool = True,
    include_regularization: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Fit M0–M3 and leave-one-seed-out prediction diagnostics."""
    y = data["ood_gap"].to_numpy(float)
    total = float(np.square(y - y.mean()).sum())
    rows = []
    folds = []
    predictions: dict[str, np.ndarray] = {}
    for model in ["M0", "M1", "M2", "M3"]:
        X = _design_matrix(
            data, model, include_seed=True, capacity_column=capacity_column,
            include_iid=include_iid, include_regularization=include_regularization,
        )
        coefficients, r2, _ = _fit_ols(X, y)
        cv_prediction, fold = _cross_validated_predictions(
            data, model, capacity_column=capacity_column,
            include_iid=include_iid, include_regularization=include_regularization,
        )
        predictions[model] = cv_prediction
        cv_error = y - cv_prediction
        rows.append(
            {
                "model": model,
                "r2": r2,
                "cv_r2": 1.0 - float(cv_error @ cv_error) / total,
                "cv_rmse": float(np.sqrt(np.mean(np.square(cv_error)))),
                "n_parameters": X.shape[1],
            }
        )
        folds.append(fold)
    return pd.DataFrame(rows), pd.concat(folds, ignore_index=True), predictions


def bootstrap_regression_increments(
    data: pd.DataFrame, bootstrap_resamples: int, seed: int
) -> pd.DataFrame:
    """Seed-cluster bootstrap family and interaction in-sample R² increments."""
    rng = np.random.default_rng(seed)
    seeds = np.array(sorted(data["seed"].unique()))
    rows = []
    for _ in range(bootstrap_resamples):
        parts = []
        for pseudo_seed, selected in enumerate(rng.choice(seeds, len(seeds), replace=True)):
            part = data.loc[data["seed"].eq(selected)].copy()
            part["bootstrap_seed"] = pseudo_seed
            parts.append(part)
        sampled = pd.concat(parts, ignore_index=True)
        y = sampled["ood_gap"].to_numpy(float)
        r2 = {}
        for model in ["M1", "M2", "M3"]:
            X = _design_matrix(sampled, model, include_seed=True, seed_column="bootstrap_seed")
            r2[model] = _fit_ols(X, y)[1]
        rows.append({"family_incremental_r2": r2["M2"] - r2["M1"], "interaction_incremental_r2": r2["M3"] - r2["M2"]})
    output = []
    frame = pd.DataFrame(rows)
    for metric in frame:
        output.append(
            {
                "metric": metric,
                "bootstrap_mean": float(frame[metric].mean()),
                "ci_lower": float(frame[metric].quantile(0.025)),
                "ci_upper": float(frame[metric].quantile(0.975)),
                "n_resamples": bootstrap_resamples,
            }
        )
    return pd.DataFrame(output)


def regression_sensitivities(data: pd.DataFrame) -> pd.DataFrame:
    """Run frozen proxy and covariate sensitivity variants plus leave-rho-out checks."""
    variants = [
        ("primary", "empirical_capacity_index", True, True),
        ("without_iid", "empirical_capacity_index", False, True),
        ("declared_capacity", "capacity_tier_score", True, True),
        ("without_regularization", "empirical_capacity_index", True, False),
    ]
    rows = []
    for name, capacity, include_iid, include_regularization in variants:
        metrics, _, _ = regression_models(
            data, capacity_column=capacity, include_iid=include_iid,
            include_regularization=include_regularization,
        )
        indexed = metrics.set_index("model")
        rows.append(
            {
                "scope": name,
                "excluded_rho": np.nan,
                "family_incremental_r2": indexed.loc["M2", "r2"] - indexed.loc["M1", "r2"],
                "family_incremental_cv_r2": indexed.loc["M2", "cv_r2"] - indexed.loc["M1", "cv_r2"],
                "family_cv_rmse_reduction": 1.0 - indexed.loc["M2", "cv_rmse"] / indexed.loc["M1", "cv_rmse"],
                "interaction_incremental_cv_r2": indexed.loc["M3", "cv_r2"] - indexed.loc["M2", "cv_r2"],
            }
        )
    for rho in RHO_VALUES:
        subset = data.loc[~np.isclose(data["rho_train"], rho)].copy()
        metrics, _, _ = regression_models(subset)
        indexed = metrics.set_index("model")
        rows.append(
            {
                "scope": "leave_one_rho_out",
                "excluded_rho": rho,
                "family_incremental_r2": indexed.loc["M2", "r2"] - indexed.loc["M1", "r2"],
                "family_incremental_cv_r2": indexed.loc["M2", "cv_r2"] - indexed.loc["M1", "cv_r2"],
                "family_cv_rmse_reduction": 1.0 - indexed.loc["M2", "cv_rmse"] / indexed.loc["M1", "cv_rmse"],
                "interaction_incremental_cv_r2": indexed.loc["M3", "cv_r2"] - indexed.loc["M2", "cv_r2"],
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_mean_ci(values: np.ndarray, resamples: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    samples = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def capacity_matching(
    data: pd.DataFrame, caliper_sd: float, bootstrap_resamples: int, seed: int
) -> pd.DataFrame:
    """Pairwise nearest-neighbor matching within rho and declared capacity tier."""
    rng = np.random.default_rng(seed)
    rows = []
    for rho in RHO_VALUES:
        for tier in ["low", "medium", "high"]:
            stratum = data.loc[np.isclose(data["rho_train"], rho) & data["capacity_tier"].eq(tier)]
            for family_a, family_b in combinations(FAMILIES, 2):
                left = stratum.loc[stratum["family"].eq(family_a)]
                right = stratum.loc[stratum["family"].eq(family_b)]
                pooled = pd.concat([left, right])
                sd_capacity = float(pooled["empirical_capacity_index"].std(ddof=1)) or 1.0
                sd_iid = float(pooled["iid_accuracy"].std(ddof=1)) or 1.0
                differences = []
                distances = []
                for _, source in left.iterrows():
                    candidates = right.loc[
                        (right["empirical_capacity_index"].sub(source["empirical_capacity_index"]).abs() <= caliper_sd * sd_capacity)
                        & (right["iid_accuracy"].sub(source["iid_accuracy"]).abs() <= caliper_sd * sd_iid)
                        & (right["regularization_flexibility_score"].sub(source["regularization_flexibility_score"]).abs() <= 0.5)
                    ].copy()
                    if candidates.empty:
                        continue
                    candidates["distance"] = np.sqrt(
                        np.square((candidates["empirical_capacity_index"] - source["empirical_capacity_index"]) / sd_capacity)
                        + np.square((candidates["iid_accuracy"] - source["iid_accuracy"]) / sd_iid)
                        + np.square(candidates["regularization_flexibility_score"] - source["regularization_flexibility_score"])
                    )
                    match = candidates.sort_values(["distance", "seed"]).iloc[0]
                    differences.append(float(source["ood_gap"] - match["ood_gap"]))
                    distances.append(float(match["distance"]))
                values = np.asarray(differences)
                lower, upper = _bootstrap_mean_ci(values, bootstrap_resamples, rng)
                rows.append(
                    {
                        "rho_train": rho,
                        "capacity_tier": tier,
                        "family_a": family_a,
                        "family_b": family_b,
                        "eligible": len(left),
                        "matched": len(values),
                        "coverage": len(values) / len(left) if len(left) else 0.0,
                        "mean_gap_difference_a_minus_b": float(values.mean()) if len(values) else np.nan,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "mean_match_distance": float(np.mean(distances)) if distances else np.nan,
                        "adequate_coverage": bool(len(values) / len(left) >= 0.5) if len(left) else False,
                    }
                )
    return pd.DataFrame(rows)


def robustness_envelope(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, rho), group in data.groupby(["family", "rho_train"], sort=False):
        row: dict[str, Any] = {"family": family, "rho_train": rho, "n": len(group)}
        for metric in ["ood_gap", "average_ood_accuracy"]:
            values = group[metric]
            row.update(
                {
                    f"{metric}_min": float(values.min()),
                    f"{metric}_q10": float(values.quantile(0.1)),
                    f"{metric}_median": float(values.median()),
                    f"{metric}_mean": float(values.mean()),
                    f"{metric}_q90": float(values.quantile(0.9)),
                    f"{metric}_max": float(values.max()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def ranking_reversal(data: pd.DataFrame, tolerance: float = 0.001) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return tie-aware configuration/family reversal probabilities and pairwise orderings."""
    rows = []
    pairwise = []
    for rho in RHO_VALUES:
        config_flags, family_flags = [], []
        subset = data.loc[np.isclose(data["rho_train"], rho)]
        for seed, group in subset.groupby("seed"):
            iid_best = set(group.loc[group["iid_accuracy"] >= group["iid_accuracy"].max() - tolerance, "config_id"])
            ood_best = set(group.loc[group["average_ood_accuracy"] >= group["average_ood_accuracy"].max() - tolerance, "config_id"])
            config_flags.append(iid_best.isdisjoint(ood_best))
            family = group.groupby("family")[["iid_accuracy", "average_ood_accuracy"]].mean().reset_index()
            iid_family = set(family.loc[family["iid_accuracy"] >= family["iid_accuracy"].max() - tolerance, "family"])
            ood_family = set(family.loc[family["average_ood_accuracy"] >= family["average_ood_accuracy"].max() - tolerance, "family"])
            family_flags.append(iid_family.isdisjoint(ood_family))
        rows.append(
            {
                "rho_train": rho,
                "configuration_reversal_probability": float(np.mean(config_flags)),
                "family_reversal_probability": float(np.mean(family_flags)),
                "n_seeds": len(config_flags),
                "tie_tolerance": tolerance,
            }
        )
        family_seed = subset.groupby(["seed", "family"])[["iid_accuracy", "average_ood_accuracy"]].mean().reset_index()
        for family_a, family_b in combinations(FAMILIES, 2):
            a = family_seed.loc[family_seed["family"].eq(family_a)].set_index("seed")
            b = family_seed.loc[family_seed["family"].eq(family_b)].set_index("seed")
            pairwise.append(
                {
                    "rho_train": rho,
                    "family_a": family_a,
                    "family_b": family_b,
                    "probability_a_beats_b_iid": float((a["iid_accuracy"] > b["iid_accuracy"]).mean()),
                    "probability_a_beats_b_ood": float((a["average_ood_accuracy"] > b["average_ood_accuracy"]).mean()),
                    "n_seeds": len(a),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pairwise)


def adjusted_family_rho(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize M1 residuals for the matched-capacity figure."""
    X = _design_matrix(data, "M1", include_seed=True)
    coefficients = np.linalg.lstsq(X, data["ood_gap"].to_numpy(float), rcond=None)[0]
    work = data.copy()
    work["adjusted_gap_residual"] = data["ood_gap"].to_numpy(float) - X @ coefficients
    return (
        work.groupby(["family", "rho_train"])["adjusted_gap_residual"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )


def decide_g4a(
    regression: pd.DataFrame,
    folds: pd.DataFrame,
    bootstrap: pd.DataFrame,
    sensitivities: pd.DataFrame,
    matching: pd.DataFrame,
    wrong_line: pd.DataFrame,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Apply the frozen G4a A/B/C decision rule without discretionary overrides."""
    indexed = regression.set_index("model")
    f_cv = float(indexed.loc["M2", "cv_r2"] - indexed.loc["M1", "cv_r2"])
    f_rmse = float(1.0 - indexed.loc["M2", "cv_rmse"] / indexed.loc["M1", "cv_rmse"])
    i_cv = float(indexed.loc["M3", "cv_r2"] - indexed.loc["M2", "cv_r2"])
    boot_lower = float(bootstrap.set_index("metric").loc["family_incremental_r2", "ci_lower"])
    fold_pivot = folds.pivot(index="seed", columns="model", values="sse")
    fold_improvements = 1.0 - fold_pivot["M2"] / fold_pivot["M1"]
    leave_rho = sensitivities.loc[sensitivities["scope"].eq("leave_one_rho_out"), "family_incremental_cv_r2"]
    adequate = matching.loc[matching["adequate_coverage"]]
    contrast = adequate.loc[
        adequate["mean_gap_difference_a_minus_b"].abs() >= thresholds["practical_accuracy_difference"]
    ]
    contrast = contrast.loc[(contrast["ci_lower"] > 0) | (contrast["ci_upper"] < 0)]
    primary_sensitivity = sensitivities.loc[sensitivities["scope"].eq("declared_capacity")].iloc[0]
    matched_signal = bool(
        len(contrast) > 0
        or primary_sensitivity["family_incremental_r2"] >= thresholds["practical_family_incremental_r2"]
    )
    neutral_high = wrong_line.loc[
        np.isclose(wrong_line["rho_train"], 0.0) & wrong_line["capacity_tier"].eq("high")
    ].groupby("family")["boundary_fidelity"].mean()
    wrong_line_pass = bool((neutral_high >= 0.95).all() and len(neutral_high) == len(FAMILIES))
    a = bool(
        f_cv < thresholds["practical_family_incremental_r2"]
        and f_rmse < thresholds["practical_cv_rmse_improvement"]
        and boot_lower <= 0
        and not matched_signal
    )
    b = bool(
        f_cv >= thresholds["practical_family_incremental_r2"]
        and f_rmse >= thresholds["practical_cv_rmse_improvement"]
        and boot_lower > 0
        and (fold_improvements > 0).all()
        and (leave_rho >= 0.03).all()
        and matched_signal
        and i_cv < f_cv
        and wrong_line_pass
    )
    if b:
        branch, decision = "G4a-B", "PROCEED"
    elif a:
        branch, decision = "G4a-A", "DO NOT PROCEED"
    else:
        branch, decision = "G4a-C", "REVISE HYPOTHESIS"
    return {
        "branch": branch,
        "decision": decision,
        "family_incremental_cv_r2": f_cv,
        "family_cv_rmse_reduction": f_rmse,
        "interaction_incremental_cv_r2": i_cv,
        "family_increment_bootstrap_lower": boot_lower,
        "all_seed_fold_improvements_positive": bool((fold_improvements > 0).all()),
        "minimum_leave_one_rho_incremental_cv_r2": float(leave_rho.min()),
        "matched_or_stratified_practical_signal": matched_signal,
        "adequately_covered_practical_contrasts": int(len(contrast)),
        "wrong_line_gate_passed": wrong_line_pass,
        "criteria_A": a,
        "criteria_B": b,
    }
