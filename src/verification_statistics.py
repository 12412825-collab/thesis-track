"""Bootstrap, ranking, and regression analyses for EXP-002V."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata


MODEL_ORDER = [
    "logistic_regression",
    "rbf_svm",
    "random_forest",
    "gradient_boosting",
    "mlp",
]


def _percentile_ci(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.nanquantile(samples, alpha)),
        float(np.nanquantile(samples, 1.0 - alpha)),
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return mean and percentile bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    lower, upper = _percentile_ci(draws, confidence)
    return float(values.mean()), lower, upper


def build_h1_statistics(
    performance: pd.DataFrame,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    """Estimate OOD-gap levels and paired consecutive changes by model."""
    data = performance.loc[performance["feature_condition"].eq("all_features")]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for (model, rho_train), group in data.groupby(["model", "rho_train"], sort=True):
        estimate, lower, upper = bootstrap_mean_ci(
            group["ood_gap"].to_numpy(), n_bootstrap, confidence, rng
        )
        rows.append(
            {
                "model": model,
                "statistic_type": "ood_gap_level",
                "rho_from": np.nan,
                "rho_to": float(rho_train),
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "n_replicates": len(group),
                "paired": False,
            }
        )
    rho_sequence = [0.0, 0.5, 0.7, 0.9]
    for model in MODEL_ORDER:
        model_data = data.loc[data["model"].eq(model)]
        for rho_from, rho_to in zip(rho_sequence[:-1], rho_sequence[1:]):
            left = model_data.loc[np.isclose(model_data["rho_train"], rho_from), [
                "replicate", "ood_gap"
            ]].rename(columns={"ood_gap": "gap_from"})
            right = model_data.loc[np.isclose(model_data["rho_train"], rho_to), [
                "replicate", "ood_gap"
            ]].rename(columns={"ood_gap": "gap_to"})
            paired = left.merge(right, on="replicate", how="inner")
            differences = (paired["gap_to"] - paired["gap_from"]).to_numpy()
            estimate, lower, upper = bootstrap_mean_ci(
                differences, n_bootstrap, confidence, rng
            )
            rows.append(
                {
                    "model": model,
                    "statistic_type": "consecutive_paired_difference",
                    "rho_from": rho_from,
                    "rho_to": rho_to,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_replicates": len(paired),
                    "paired": True,
                }
            )
    return pd.DataFrame(rows)


def summarize_metric_with_ci(
    data: pd.DataFrame,
    metric: str,
    group_columns: list[str],
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    """Build mean and bootstrap interval rows for plotting."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    group_key: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in data.groupby(group_key, sort=True):
        keys = (keys,) if len(group_columns) == 1 else tuple(keys)
        estimate, lower, upper = bootstrap_mean_ci(
            group[metric].to_numpy(), n_bootstrap, confidence, rng
        )
        row = dict(zip(group_columns, keys))
        row.update(
            metric=metric,
            mean=estimate,
            ci_lower=lower,
            ci_upper=upper,
            n_replicates=len(group),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_ranking_outputs(
    performance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return replicate ranks, pairwise probabilities, and reversal probabilities."""
    data = performance.loc[performance["feature_condition"].eq("all_features")].copy()
    rank_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    reversal_rows: list[dict[str, Any]] = []
    metrics = {
        "iid_test": "iid_accuracy",
        "average_ood": "average_ood_accuracy",
    }
    for rho_train, regime in data.groupby("rho_train", sort=True):
        for metric_name, column in metrics.items():
            for replicate, group in regime.groupby("replicate", sort=True):
                scores = group.set_index("model")[column]
                ranks = scores.rank(method="average", ascending=False)
                maximum = scores.max()
                for model, score in scores.items():
                    rank_rows.append(
                        {
                            "rho_train": float(rho_train),
                            "replicate": int(replicate),
                            "metric": metric_name,
                            "model": model,
                            "score": float(score),
                            "rank": float(ranks[model]),
                            "is_best": bool(np.isclose(score, maximum)),
                        }
                    )
            pivot = regime.pivot(index="replicate", columns="model", values=column)
            for model_a in MODEL_ORDER:
                for model_b in MODEL_ORDER:
                    if model_a == model_b:
                        continue
                    difference = pivot[model_a] - pivot[model_b]
                    pairwise_rows.append(
                        {
                            "rho_train": float(rho_train),
                            "metric": metric_name,
                            "model_a": model_a,
                            "model_b": model_b,
                            "probability_a_beats_b": float(np.mean(difference > 0)),
                            "probability_tie": float(np.mean(np.isclose(difference, 0.0))),
                            "n_replicates": len(difference),
                        }
                    )
        reversal_flags: list[bool] = []
        iid_winners: list[str] = []
        ood_winners: list[str] = []
        for replicate, group in regime.groupby("replicate", sort=True):
            iid_max = group["iid_accuracy"].max()
            ood_max = group["average_ood_accuracy"].max()
            iid_best = set(group.loc[np.isclose(group["iid_accuracy"], iid_max), "model"])
            ood_best = set(group.loc[np.isclose(group["average_ood_accuracy"], ood_max), "model"])
            reversal_flags.append(iid_best.isdisjoint(ood_best))
            iid_winners.append("|".join(sorted(iid_best)))
            ood_winners.append("|".join(sorted(ood_best)))
        reversal_rows.append(
            {
                "rho_train": float(rho_train),
                "probability_iid_best_differs_from_ood_best": float(np.mean(reversal_flags)),
                "n_replicates": len(reversal_flags),
                "iid_best_sets": ";".join(iid_winners),
                "ood_best_sets": ";".join(ood_winners),
            }
        )
    return (
        pd.DataFrame(rank_rows),
        pd.DataFrame(pairwise_rows),
        pd.DataFrame(reversal_rows),
    )


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.isclose(np.std(x), 0) or np.isclose(np.std(y), 0):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _association_statistics(
    data: np.ndarray,
    predictor_index: int,
    include_regression: bool,
) -> dict[str, float]:
    """Columns are [iid_accuracy, predictor(s), ood_gap]."""
    iid = data[:, 0]
    predictor = data[:, predictor_index]
    gap = data[:, -1]
    stats = {
        "pearson": _safe_correlation(predictor, gap),
        "spearman": _safe_correlation(rankdata(predictor), rankdata(gap)),
    }
    if include_regression:
        baseline_X = np.column_stack([np.ones(len(data)), iid])
        full_X = np.column_stack([np.ones(len(data)), iid, predictor])
        baseline_beta = np.linalg.lstsq(baseline_X, gap, rcond=None)[0]
        full_beta = np.linalg.lstsq(full_X, gap, rcond=None)[0]
        baseline_residual = gap - baseline_X @ baseline_beta
        full_residual = gap - full_X @ full_beta
        total = float(np.sum((gap - gap.mean()) ** 2))
        r2_baseline = 1.0 - float(np.sum(baseline_residual**2)) / total
        r2_full = 1.0 - float(np.sum(full_residual**2)) / total
        predictor_residual = predictor - baseline_X @ np.linalg.lstsq(
            baseline_X, predictor, rcond=None
        )[0]
        stats.update(
            d3_coefficient=float(full_beta[2]),
            partial_correlation=_safe_correlation(predictor_residual, baseline_residual),
            r2_baseline=r2_baseline,
            r2_full=r2_full,
            incremental_r2=r2_full - r2_baseline,
        )
    return stats


def _cluster_bootstrap(
    frame: pd.DataFrame,
    columns: list[str],
    statistic: Callable[[np.ndarray], dict[str, float]],
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    grouped = [group[columns].to_numpy(dtype=float) for _, group in frame.groupby(
        ["rho_train", "replicate"], sort=True
    )]
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {}
    for _ in range(n_bootstrap):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        sample = np.concatenate([grouped[index] for index in selected], axis=0)
        values = statistic(sample)
        for key, value in values.items():
            collected.setdefault(key, []).append(value)
    return {key: np.asarray(values) for key, values in collected.items()}


def build_h2_statistics(
    analysis: pd.DataFrame,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    """Compute preregistered D3/D1 associations and IID-adjusted regression."""
    columns = ["iid_accuracy", "d3_excess", "raw_loco", "ood_gap"]
    matrix = analysis[columns].to_numpy(dtype=float)
    d3_point = _association_statistics(matrix, predictor_index=1, include_regression=True)
    d1_point = _association_statistics(matrix, predictor_index=2, include_regression=False)
    d3_boot = _cluster_bootstrap(
        analysis,
        columns,
        lambda sample: _association_statistics(sample, 1, True),
        n_bootstrap,
        seed,
    )
    d1_boot = _cluster_bootstrap(
        analysis,
        columns,
        lambda sample: _association_statistics(sample, 2, False),
        n_bootstrap,
        seed + 1,
    )
    rows: list[dict[str, Any]] = []
    for predictor, point, boot in [
        ("d3_excess", d3_point, d3_boot),
        ("raw_loco", d1_point, d1_boot),
    ]:
        for statistic_name, estimate in point.items():
            lower, upper = _percentile_ci(boot[statistic_name], confidence)
            rows.append(
                {
                    "predictor": predictor,
                    "statistic": statistic_name,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_points": len(analysis),
                    "n_clusters": analysis[["rho_train", "replicate"]].drop_duplicates().shape[0],
                    "bootstrap_resamples": n_bootstrap,
                    "label": (
                        "EXPLORATORY ONLY - INSUFFICIENT CONFIGURATION DIVERSITY"
                        if statistic_name in {"d3_coefficient", "partial_correlation", "r2_baseline", "r2_full", "incremental_r2"}
                        else "preregistered descriptive association"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_leave_one_family_out(
    analysis: pd.DataFrame,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    """Re-estimate D3 associations after excluding each family."""
    columns = ["iid_accuracy", "d3_excess", "ood_gap"]
    rows: list[dict[str, Any]] = []
    for index, excluded in enumerate(MODEL_ORDER):
        subset = analysis.loc[~analysis["model"].eq(excluded)]
        matrix = subset[columns].to_numpy(dtype=float)
        point = _association_statistics(matrix, predictor_index=1, include_regression=True)
        boot = _cluster_bootstrap(
            subset,
            columns,
            lambda sample: _association_statistics(sample, 1, True),
            n_bootstrap,
            seed + index,
        )
        row: dict[str, Any] = {
            "excluded_family": excluded,
            "n_points": len(subset),
            "n_clusters": subset[["rho_train", "replicate"]].drop_duplicates().shape[0],
        }
        for name in ["pearson", "spearman", "d3_coefficient", "partial_correlation", "incremental_r2"]:
            lower, upper = _percentile_ci(boot[name], confidence)
            row[name] = point[name]
            row[f"{name}_ci_lower"] = lower
            row[f"{name}_ci_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def build_diagnostic_agreement(analysis: pd.DataFrame) -> pd.DataFrame:
    """Compare aggregate D3 and LOCO rankings without averaging diagnostics."""
    aggregate = analysis.groupby(["rho_train", "model"], as_index=False)[
        ["d3_excess", "raw_loco"]
    ].mean()
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("all_rho_train", aggregate)]
    scopes.extend(
        (f"rho_train_{rho:.1f}", group)
        for rho, group in aggregate.groupby("rho_train", sort=True)
    )
    for scope, group in scopes:
        labels = (
            group["model"] + "@rho=" + group["rho_train"].map(lambda value: f"{value:.1f}")
            if scope == "all_rho_train"
            else group["model"]
        )
        labeled = group.assign(configuration_label=labels)
        d3_order = labeled.sort_values("d3_excess", ascending=False)[
            "configuration_label"
        ].tolist()
        loco_order = labeled.sort_values("raw_loco", ascending=False)[
            "configuration_label"
        ].tolist()
        rows.append(
            {
                "scope": scope,
                "n_configurations": len(group),
                "spearman_correlation": _safe_correlation(
                    rankdata(group["d3_excess"]), rankdata(group["raw_loco"])
                ),
                "exact_rank_order_match": d3_order == loco_order,
                "d3_rank_order": ">".join(d3_order),
                "loco_rank_order": ">".join(loco_order),
            }
        )
    return pd.DataFrame(rows)
