"""Configuration, reproducibility, logging, and filesystem helpers."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy for deterministic supporting operations."""
    random.seed(seed)
    np.random.seed(seed)


def child_seed(seed: int, *parts: int) -> int:
    """Derive a stable independent 32-bit seed for an experiment component."""
    sequence = np.random.SeedSequence([seed, *parts])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def ensure_output_directories(root: Path = PROJECT_ROOT) -> dict[str, Path]:
    """Create and return the standard output directories."""
    paths = {
        "raw": root / "results" / "raw",
        "summaries": root / "results" / "summaries",
        "figures": root / "results" / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save JSON metadata using readable, deterministic formatting."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def configure_logging() -> None:
    """Configure concise console logging for experiment runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
