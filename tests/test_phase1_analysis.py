"""Method-level tests for the preregistered Phase 1 analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.phase1_analysis import _variance_components, ranking_reversal


def test_balanced_variance_components_reconstruct_total_sum_squares() -> None:
    rows = []
    for family_index, family in enumerate(["a", "b"]):
        for config in [f"{family}-0", f"{family}-1"]:
            for rho in [0.0, 0.9]:
                for seed in [0, 1, 2]:
                    rows.append({"family": family, "config_id": config, "rho_train": rho, "seed": seed, "ood_gap": family_index + rho + 0.1 * seed})
    components = _variance_components(pd.DataFrame(rows))
    assert np.isclose(sum(components[name] for name in components if name != "total"), components["total"])

