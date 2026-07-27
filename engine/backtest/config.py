"""Load pre-registered config (hyperparameter grids, population filters).

Non-negotiable rule 7: grids and filters are declared in config BEFORE
results are seen. This module only reads what's committed; nothing here
computes or infers a grid.
"""
from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
LGBM_GRID_PATH = CONFIG_DIR / "lgbm_grid.yaml"


def load_lgbm_grid(path: Path = LGBM_GRID_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
