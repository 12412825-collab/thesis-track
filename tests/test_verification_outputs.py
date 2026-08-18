"""Contract checks for the completed EXP-002V evidence package."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "EXP-002V" / "summaries"


def test_required_tables_exist_with_preregistered_grain() -> None:
    expected_shapes = {
        "generator_validation.csv": (390, 33),
        "performance_by_replicate.csv": (500, 18),
        "reliance_D3.csv": (250, 11),
        "reliance_D7_noise_floor.csv": (250, 6),
        "reliance_LOCO.csv": (250, 7),
        "ranking_distribution.csv": (500, 7),
        "pairwise_ranking_probability.csv": (160, 7),
        "H1_statistics.csv": (35, 9),
        "H2_statistics.csv": (9, 9),
        "leave_one_family_out.csv": (5, 18),
    }
    for filename, shape in expected_shapes.items():
        table = pd.read_csv(SUMMARY / filename)
        assert table.shape == shape
        assert not table.duplicated().any()


def test_all_seven_figures_have_png_and_pdf_versions() -> None:
    figures = ROOT / "results" / "EXP-002V" / "figures"
    for index in range(1, 8):
        assert len(list(figures.glob(f"{index:02d}_*.png"))) == 1
        assert len(list(figures.glob(f"{index:02d}_*.pdf"))) == 1


def test_report_uses_exact_preregistered_headings() -> None:
    report = (ROOT / "experiments" / "EXP-002V" / "REPORT.md").read_text(
        encoding="utf-8"
    )
    headings = [line for line in report.splitlines() if line.startswith("## ")]
    assert headings == [
        "## Purpose",
        "## Relationship to EXP-002",
        "## Generator Validation",
        "## H1 — Degradation Structure",
        "## Ranking Reversal Validation",
        "## D3 Shortcut Reliance",
        "## D1 LOCO Cross-Check",
        "## H2 — Reliance vs OOD Degradation",
        "## Leave-One-Family-Out Analysis",
        "## Stable-Only Control",
        "## What Survived EXP-002",
        "## What Failed to Replicate",
        "## What We Can Claim",
        "## What We Cannot Claim",
        "## Decision Recommendation",
    ]
