"""Model registry and training-only capacity diagnostics for Phase 1."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


FAMILY_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "rbf_svm": "RBF SVM",
    "random_forest": "Random Forest",
    "gradient_boosting": "Histogram Gradient Boosting",
    "mlp": "MLP",
}


@dataclass(frozen=True)
class ModelConfiguration:
    """One immutable family/configuration registry entry."""

    config_id: str
    family: str
    capacity_tier: str
    capacity_tier_score: float
    regularization_tier: str
    regularization_flexibility_score: float
    parameters: dict[str, Any]
    config_hash: str


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any) -> str:
    """Return the SHA-256 hash of a canonical JSON value."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_registry(config: dict[str, Any]) -> list[ModelConfiguration]:
    """Validate and materialize the frozen 15-entry registry."""
    rows = config["configurations"]
    if len(rows) != 15:
        raise ValueError(f"Phase 1 requires exactly 15 configurations, received {len(rows)}")
    entries: list[ModelConfiguration] = []
    seen: set[str] = set()
    family_counts: dict[str, int] = {}
    for row in rows:
        config_id = str(row["config_id"])
        family = str(row["family"])
        if config_id in seen:
            raise ValueError(f"Duplicate config_id: {config_id}")
        if family not in FAMILY_DISPLAY_NAMES:
            raise ValueError(f"Unknown family: {family}")
        seen.add(config_id)
        family_counts[family] = family_counts.get(family, 0) + 1
        payload = {
            "config_id": config_id,
            "family": family,
            "capacity_tier": str(row["capacity_tier"]),
            "capacity_tier_score": float(row["capacity_tier_score"]),
            "regularization_tier": str(row["regularization_tier"]),
            "regularization_flexibility_score": float(
                row["regularization_flexibility_score"]
            ),
            "parameters": row["parameters"],
        }
        entries.append(
            ModelConfiguration(
                **payload,
                config_hash=stable_hash(payload),
            )
        )
    if set(family_counts) != set(FAMILY_DISPLAY_NAMES):
        raise ValueError("Registry must contain all five frozen model families")
    if any(count != 3 for count in family_counts.values()):
        raise ValueError(f"Registry must contain three configurations per family: {family_counts}")
    return entries


def build_phase1_model(entry: ModelConfiguration, seed: int) -> BaseEstimator:
    """Build a deterministic estimator from one Phase 1 registry entry."""
    p = entry.parameters
    if entry.family == "logistic_regression":
        estimator = LogisticRegression(
            C=float(p["C"]),
            max_iter=int(p["max_iter"]),
            solver=str(p.get("solver", "lbfgs")),
            random_state=seed,
        )
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])
    if entry.family == "rbf_svm":
        base = SVC(
            C=float(p["C"]),
            gamma=p["gamma"],
            cache_size=float(p.get("cache_size", 1000)),
            random_state=seed,
        )
        calibrated = CalibratedClassifierCV(
            estimator=base,
            method="sigmoid",
            cv=int(p.get("calibration_cv", 3)),
            ensemble=False,
        )
        return Pipeline([("scale", StandardScaler()), ("model", calibrated)])
    if entry.family == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(p["n_estimators"]),
            max_depth=None if p.get("max_depth") is None else int(p["max_depth"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            max_features=p.get("max_features", "sqrt"),
            n_jobs=int(p.get("n_jobs", -1)),
            random_state=seed,
        )
    if entry.family == "gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=float(p["learning_rate"]),
            max_iter=int(p["max_iter"]),
            max_leaf_nodes=int(p["max_leaf_nodes"]),
            l2_regularization=float(p["l2_regularization"]),
            early_stopping=bool(p.get("early_stopping", False)),
            random_state=seed,
        )
    if entry.family == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=tuple(int(x) for x in p["hidden_layer_sizes"]),
            alpha=float(p["alpha"]),
            learning_rate_init=float(p["learning_rate_init"]),
            max_iter=int(p["max_iter"]),
            early_stopping=bool(p.get("early_stopping", True)),
            random_state=seed,
        )
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])
    raise KeyError(entry.family)


def _inner_estimator(model: BaseEstimator) -> BaseEstimator:
    return model.named_steps["model"] if hasattr(model, "named_steps") else model


def _logistic_effective_df(
    model: Pipeline, X_train: pd.DataFrame, c_value: float
) -> float:
    """Approximate ridge-logistic effective degrees of freedom on training data."""
    scaled = model.named_steps["scale"].transform(X_train)
    estimator = model.named_steps["model"]
    probabilities = estimator.predict_proba(scaled)[:, 1]
    weights = probabilities * (1.0 - probabilities)
    information = scaled.T @ (scaled * weights[:, None])
    penalty = 1.0 / c_value
    regularized = information + penalty * np.eye(information.shape[0])
    return float(np.trace(np.linalg.solve(regularized, information)))


def _rbf_capacity(
    model: Pipeline, X_train: pd.DataFrame, max_samples: int
) -> tuple[float, float, float]:
    """Return kernel effective rank, support-vector fraction, and fitted gamma."""
    scaled = model.named_steps["scale"].transform(X_train)
    calibrated = model.named_steps["model"]
    svc = calibrated.calibrated_classifiers_[0].estimator
    gamma = float(svc._gamma)
    sample = scaled[: min(max_samples, len(scaled))]
    kernel = rbf_kernel(sample, sample, gamma=gamma)
    effective_rank = float(np.trace(kernel) ** 2 / np.square(kernel).sum())
    support_fraction = float(svc.support_.size / len(X_train))
    return effective_rank, support_fraction, gamma


def extract_capacity_metrics(
    entry: ModelConfiguration,
    model: BaseEstimator,
    X_train: pd.DataFrame,
    *,
    kernel_rank_max_samples: int = 512,
) -> dict[str, float | str]:
    """Extract family-native capacity measures without consulting outcomes."""
    base: dict[str, float | str] = {
        "capacity_proxy_name": "",
        "learned_capacity_primary": np.nan,
        "structural_parameter_count": np.nan,
        "effective_degrees_of_freedom": np.nan,
        "kernel_effective_rank": np.nan,
        "support_vector_fraction": np.nan,
        "tree_total_leaves": np.nan,
        "tree_mean_depth": np.nan,
        "boosting_total_leaves": np.nan,
        "boosting_iterations": np.nan,
        "mlp_parameter_count": np.nan,
        "optimization_iterations": np.nan,
        "converged": 1.0,
    }
    estimator = _inner_estimator(model)
    if entry.family == "logistic_regression":
        effective_df = _logistic_effective_df(model, X_train, float(entry.parameters["C"]))
        base.update(
            capacity_proxy_name="effective_degrees_of_freedom",
            learned_capacity_primary=effective_df,
            structural_parameter_count=float(estimator.coef_.size + estimator.intercept_.size),
            effective_degrees_of_freedom=effective_df,
            optimization_iterations=float(np.max(estimator.n_iter_)),
            converged=float(np.max(estimator.n_iter_) < int(entry.parameters["max_iter"])),
        )
    elif entry.family == "rbf_svm":
        effective_rank, support_fraction, gamma = _rbf_capacity(
            model, X_train, kernel_rank_max_samples
        )
        svc = estimator.calibrated_classifiers_[0].estimator
        base.update(
            capacity_proxy_name="kernel_effective_rank",
            learned_capacity_primary=effective_rank,
            kernel_effective_rank=effective_rank,
            support_vector_fraction=support_fraction,
            fitted_kernel_gamma=gamma,
            structural_parameter_count=float(svc.support_vectors_.size + svc.dual_coef_.size),
        )
    elif entry.family == "random_forest":
        leaves = np.array([tree.tree_.n_leaves for tree in estimator.estimators_], dtype=float)
        depths = np.array([tree.tree_.max_depth for tree in estimator.estimators_], dtype=float)
        total_leaves = float(leaves.sum())
        base.update(
            capacity_proxy_name="tree_total_leaves",
            learned_capacity_primary=total_leaves,
            tree_total_leaves=total_leaves,
            tree_mean_depth=float(depths.mean()),
            structural_parameter_count=float(
                sum(tree.tree_.node_count for tree in estimator.estimators_)
            ),
        )
    elif entry.family == "gradient_boosting":
        predictors = [tree for iteration in estimator._predictors for tree in iteration]
        total_leaves = float(sum(np.sum(tree.nodes["is_leaf"] == 1) for tree in predictors))
        base.update(
            capacity_proxy_name="boosting_total_leaves",
            learned_capacity_primary=total_leaves,
            boosting_total_leaves=total_leaves,
            boosting_iterations=float(estimator.n_iter_),
            structural_parameter_count=float(sum(len(tree.nodes) for tree in predictors)),
            optimization_iterations=float(estimator.n_iter_),
            converged=float(estimator.n_iter_ < int(entry.parameters["max_iter"]))
            if bool(entry.parameters.get("early_stopping", False))
            else 1.0,
        )
    elif entry.family == "mlp":
        parameter_count = float(
            sum(weights.size for weights in estimator.coefs_)
            + sum(bias.size for bias in estimator.intercepts_)
        )
        base.update(
            capacity_proxy_name="mlp_parameter_count",
            learned_capacity_primary=parameter_count,
            structural_parameter_count=parameter_count,
            mlp_parameter_count=parameter_count,
            optimization_iterations=float(estimator.n_iter_),
            converged=float(estimator.n_iter_ < int(entry.parameters["max_iter"])),
        )
    return base


def add_empirical_capacity_index(fits: pd.DataFrame) -> pd.DataFrame:
    """Add an outcome-free within-family/rho percentile capacity index."""
    result = fits.copy()
    keys = ["family", "rho_train"] if "rho_train" in result else ["family"]
    result["empirical_capacity_index"] = result.groupby(keys, dropna=False)[
        "learned_capacity_primary"
    ].rank(method="average", pct=True)
    return result
