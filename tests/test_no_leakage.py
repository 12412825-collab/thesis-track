"""Tests that preserve the train/IID-validation versus OOD evaluation boundary."""

import inspect

from src.evaluation import evaluate_environments, select_and_fit_model


def test_selection_api_cannot_receive_test_environments() -> None:
    parameters = set(inspect.signature(select_and_fit_model).parameters)
    forbidden = {name for name in parameters if "test" in name or "ood" in name or "environment" in name}
    assert not forbidden
    assert {"X_train", "y_train", "X_val", "y_val"}.issubset(parameters)


def test_ood_evaluation_api_receives_an_already_fitted_model() -> None:
    parameters = list(inspect.signature(evaluate_environments).parameters)
    assert parameters[0] == "model"
    assert "model_config" not in parameters
    assert "X_train" not in parameters

