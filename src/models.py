"""Fixed, intentionally small model-family registry for V1."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "rbf_svm": "RBF SVM",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "mlp": "MLP",
}


def build_model(name: str, config: dict[str, Any], seed: int) -> BaseEstimator:
    """Build a deterministic sklearn estimator from fixed V1 settings."""
    if name == "logistic_regression":
        classifier = LogisticRegression(
            C=float(config.get("C", 1.0)),
            max_iter=int(config.get("max_iter", 1000)),
            random_state=seed,
        )
        return Pipeline([("scale", StandardScaler()), ("model", classifier)])
    if name == "rbf_svm":
        svm = SVC(
            C=float(config.get("C", 1.0)),
            gamma=config.get("gamma", "scale"),
            cache_size=float(config.get("cache_size", 1000)),
            random_state=seed,
        )
        classifier = CalibratedClassifierCV(
            estimator=svm,
            method="sigmoid",
            cv=int(config.get("calibration_cv", 3)),
            ensemble=False,
        )
        return Pipeline([("scale", StandardScaler()), ("model", classifier)])
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(config.get("n_estimators", 250)),
            min_samples_leaf=int(config.get("min_samples_leaf", 2)),
            n_jobs=int(config.get("n_jobs", -1)),
            random_state=seed,
        )
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=float(config.get("learning_rate", 0.08)),
            max_iter=int(config.get("max_iter", 200)),
            max_leaf_nodes=int(config.get("max_leaf_nodes", 31)),
            l2_regularization=float(config.get("l2_regularization", 0.1)),
            random_state=seed,
        )
    if name == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=tuple(config.get("hidden_layer_sizes", [64, 64])),
                        alpha=float(config.get("alpha", 0.001)),
                        learning_rate_init=float(config.get("learning_rate_init", 0.001)),
                        max_iter=int(config.get("max_iter", 250)),
                        early_stopping=bool(config.get("early_stopping", True)),
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise KeyError(f"Unknown model: {name}")


def configured_model_names(models_config: dict[str, Any]) -> list[str]:
    """Validate and preserve configured model order."""
    unknown = set(models_config) - set(MODEL_DISPLAY_NAMES)
    if unknown:
        raise KeyError(f"Unknown configured models: {sorted(unknown)}")
    return list(models_config)
