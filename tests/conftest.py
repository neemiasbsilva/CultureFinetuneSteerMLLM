"""Shared fixtures and collection rules for the test suite."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root.

    Returns:
        Absolute path to the repository root.
    """
    return ROOT
