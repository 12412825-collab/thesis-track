"""Focused checks for Phase 1 output validation helpers."""

from __future__ import annotations

import pandas as pd

from src.phase1_validation import expected_master_rows


def test_expected_master_rows_deduplicates_iid_rho_from_ood_grid() -> None:
    assert expected_master_rows(15, [0], [0.0, 0.9], [0.0, -0.9]) == 75
