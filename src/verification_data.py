"""Paired data generation and pure-shift validation for EXP-002V."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data_generation import GeneratedDataset, STABLE_FEATURES, generate_dataset
from src.utils import child_seed


VERIFICATION_ALL_FEATURES = [*STABLE_FEATURES, "x_spurious", "z"]


@dataclass(frozen=True)
class VerificationDataset:
    """Generated EXP-002V data with an independent noise-floor feature."""

    X: pd.DataFrame
    y: np.ndarray
    metadata: dict[str, float | int | str]


@dataclass(frozen=True)
class RegimeBundle:
    """All paired samples for one rho_train and replicate."""

    rho_train: float
    replicate: int
    train: VerificationDataset
    validation: VerificationDataset
    iid_test: VerificationDataset
    ood: dict[float, VerificationDataset]


def generate_verification_dataset(
    *,
    n_samples: int,
    rho: float,
    base_seed: int,
    noise_seed: int,
    stable_coefficients: list[float],
    label_noise: float,
    split: str,
) -> VerificationDataset:
    """Generate baseline data and append z from an independent RNG stream."""
    base: GeneratedDataset = generate_dataset(
        n_samples=n_samples,
        rho=rho,
        seed=base_seed,
        stable_coefficients=stable_coefficients,
        label_noise=label_noise,
        split=split,
    )
    X = base.X.copy()
    X["z"] = np.random.default_rng(noise_seed).normal(size=n_samples)
    metadata = dict(base.metadata)
    metadata.update(
        {
            "base_seed": base_seed,
            "noise_seed": noise_seed,
            "z_mean": float(X["z"].mean()),
            "z_variance": float(X["z"].var(ddof=0)),
            "z_target_correlation": float(np.corrcoef(X["z"], 2 * base.y - 1)[0, 1]),
        }
    )
    return VerificationDataset(X=X, y=base.y, metadata=metadata)


def replicate_count(rho_train: float, config: dict[str, Any]) -> int:
    """Return the preregistered replicate count for a training regime."""
    settings = config["replicates"]
    if np.isclose(rho_train, float(settings["anchor_rho"])):
        return int(settings["anchor"])
    return int(settings["default"])


def _rho_code(rho: float) -> int:
    return int(round((rho + 1.0) * 1000))


def build_regime_bundles(config: dict[str, Any]) -> list[RegimeBundle]:
    """Generate paired latent samples across rho regimes and model families."""
    data = config["data"]
    master_seed = int(config["experiment"]["master_seed"])
    bundles: list[RegimeBundle] = []
    for rho_value in data["rho_train_values"]:
        rho_train = float(rho_value)
        for replicate in range(replicate_count(rho_train, config)):
            # For replicate 0..9 the same latent stable samples, labels, epsilon,
            # and z are reused across rho regimes. Only the rho transformation
            # changes, giving a paired intervention. Replicates 10..19 are the
            # preregistered rho=0.9 anchor extension.
            train = generate_verification_dataset(
                n_samples=int(data["n_train"]), rho=rho_train,
                base_seed=child_seed(master_seed, replicate, 1),
                noise_seed=child_seed(master_seed, replicate, 1, 999),
                stable_coefficients=data["stable_coefficients"],
                label_noise=float(data["label_noise"]), split="train",
            )
            validation = generate_verification_dataset(
                n_samples=int(data["n_val"]), rho=rho_train,
                base_seed=child_seed(master_seed, replicate, 2),
                noise_seed=child_seed(master_seed, replicate, 2, 999),
                stable_coefficients=data["stable_coefficients"],
                label_noise=float(data["label_noise"]), split="validation",
            )
            iid_test = generate_verification_dataset(
                n_samples=int(data["n_test"]), rho=rho_train,
                base_seed=child_seed(master_seed, replicate, 100),
                noise_seed=child_seed(master_seed, replicate, 100, 999),
                stable_coefficients=data["stable_coefficients"],
                label_noise=float(data["label_noise"]), split="iid_test",
            )
            ood: dict[float, VerificationDataset] = {}
            for rho_ood_value in data["rho_ood_values"]:
                rho_ood = float(rho_ood_value)
                if np.isclose(rho_ood, rho_train):
                    continue
                ood[rho_ood] = generate_verification_dataset(
                    n_samples=int(data["n_test"]), rho=rho_ood,
                    base_seed=child_seed(master_seed, replicate, 100, _rho_code(rho_ood)),
                    noise_seed=child_seed(master_seed, replicate, 100, _rho_code(rho_ood), 999),
                    stable_coefficients=data["stable_coefficients"],
                    label_noise=float(data["label_noise"]), split=f"ood_{rho_ood}",
                )
            bundles.append(
                RegimeBundle(
                    rho_train=rho_train, replicate=replicate, train=train,
                    validation=validation, iid_test=iid_test, ood=ood,
                )
            )
    return bundles


def _dataset_validation_row(
    bundle: RegimeBundle,
    dataset: VerificationDataset,
    split: str,
    environment_type: str,
    rho_target: float,
    coefficients: np.ndarray,
    mi_seed: int,
) -> dict[str, Any]:
    train = bundle.train
    y_signed = 2 * dataset.y - 1
    oracle_prediction = (dataset.X[STABLE_FEATURES].to_numpy() @ coefficients > 0).astype(int)
    row: dict[str, Any] = {
        "rho_train": bundle.rho_train,
        "replicate": bundle.replicate,
        "split": split,
        "environment_type": environment_type,
        "rho_ood": rho_target if environment_type == "ood" else np.nan,
        "rho_target": rho_target,
        "rho_realized": float(np.corrcoef(dataset.X["x_spurious"], y_signed)[0, 1]),
        "n_samples": len(dataset.y),
        "positive_rate": float(dataset.y.mean()),
        "stable_oracle_accuracy": float(accuracy_score(dataset.y, oracle_prediction)),
        "shortcut_only_auroc": float(roc_auc_score(dataset.y, dataset.X["x_spurious"])),
        "shortcut_target_mi": float(
            mutual_info_classif(
                dataset.X[["x_spurious"]].to_numpy(), dataset.y,
                discrete_features=False, random_state=mi_seed,
            )[0]
        ),
        "z_target_correlation": float(np.corrcoef(dataset.X["z"], y_signed)[0, 1]),
        "z_mean": float(dataset.X["z"].mean()),
        "z_variance": float(dataset.X["z"].var(ddof=0)),
    }
    for feature in STABLE_FEATURES:
        values = dataset.X[feature].to_numpy()
        train_values = train.X[feature].to_numpy()
        ks = ks_2samp(train_values, values) if split != "train" else None
        row[f"{feature}_mean"] = float(values.mean())
        row[f"{feature}_variance"] = float(values.var(ddof=0))
        row[f"{feature}_train_mean_difference"] = float(values.mean() - train_values.mean())
        row[f"{feature}_train_variance_difference"] = float(
            values.var(ddof=0) - train_values.var(ddof=0)
        )
        row[f"{feature}_ks_statistic"] = 0.0 if ks is None else float(ks.statistic)
        row[f"{feature}_ks_pvalue"] = 1.0 if ks is None else float(ks.pvalue)
    return row


def build_generator_validation(
    bundles: list[RegimeBundle], config: dict[str, Any]
) -> pd.DataFrame:
    """Build one auditable validation row per generated environment."""
    coefficients = np.asarray(config["data"]["stable_coefficients"], dtype=float)
    master_seed = int(config["experiment"]["master_seed"])
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        datasets = [
            (bundle.train, "train", "train", bundle.rho_train),
            (bundle.validation, "validation", "iid_validation", bundle.rho_train),
            (bundle.iid_test, "iid_test", "iid_test", bundle.rho_train),
            *[
                (dataset, f"ood_{rho}", "ood", rho)
                for rho, dataset in bundle.ood.items()
            ],
        ]
        for index, (dataset, split, environment_type, rho_target) in enumerate(datasets):
            rows.append(
                _dataset_validation_row(
                    bundle, dataset, split, environment_type, rho_target,
                    coefficients, child_seed(master_seed, bundle.replicate, index, 777),
                )
            )
    return pd.DataFrame(rows)


def pure_shift_failures(
    validation: pd.DataFrame, config: dict[str, Any]
) -> list[str]:
    """Return preregistered stop-gate failures before model evaluation."""
    thresholds = config["generator_validation"]
    failures: list[str] = []
    rho_error = (validation["rho_realized"] - validation["rho_target"]).abs().max()
    if rho_error > float(thresholds["max_abs_rho_error"]):
        failures.append(f"maximum rho error {rho_error:.4f} exceeds threshold")
    min_rate, max_rate = validation["positive_rate"].min(), validation["positive_rate"].max()
    if min_rate < float(thresholds["positive_rate_min"]) or max_rate > float(
        thresholds["positive_rate_max"]
    ):
        failures.append(f"positive-rate range [{min_rate:.4f}, {max_rate:.4f}] is outside bounds")
    for feature in STABLE_FEATURES:
        mean_difference = validation[f"{feature}_train_mean_difference"].abs().max()
        variance_difference = validation[f"{feature}_train_variance_difference"].abs().max()
        ks_statistic = validation[f"{feature}_ks_statistic"].max()
        if mean_difference > float(thresholds["max_abs_stable_mean_difference"]):
            failures.append(f"{feature} mean difference {mean_difference:.4f} exceeds threshold")
        if variance_difference > float(thresholds["max_abs_stable_variance_difference"]):
            failures.append(f"{feature} variance difference {variance_difference:.4f} exceeds threshold")
        if ks_statistic > float(thresholds["max_stable_ks_statistic"]):
            failures.append(f"{feature} KS statistic {ks_statistic:.4f} exceeds threshold")
    train_oracle = validation.loc[validation["environment_type"].eq("train"), [
        "rho_train", "replicate", "stable_oracle_accuracy"
    ]].rename(columns={"stable_oracle_accuracy": "train_oracle_accuracy"})
    compared = validation.merge(train_oracle, on=["rho_train", "replicate"])
    oracle_difference = (
        compared["stable_oracle_accuracy"] - compared["train_oracle_accuracy"]
    ).abs().max()
    if oracle_difference > float(
        thresholds["max_abs_stable_oracle_accuracy_difference"]
    ):
        failures.append(f"stable-oracle difference {oracle_difference:.4f} exceeds threshold")
    noise_correlation = validation["z_target_correlation"].abs().max()
    if noise_correlation > float(thresholds["max_abs_noise_target_correlation"]):
        failures.append(f"noise-target correlation {noise_correlation:.4f} exceeds threshold")
    return failures
