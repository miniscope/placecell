"""Shared test fixtures."""

from pathlib import Path

import pytest

ASSETS_DIR = Path(__file__).parent / "assets"


@pytest.fixture
def assets_dir() -> Path:
    """Path to test assets directory."""
    return ASSETS_DIR


@pytest.fixture
def neural_path(assets_dir: Path) -> Path:
    """Path to neural data directory."""
    return assets_dir / "neural_data"


@pytest.fixture
def behavior_path(assets_dir: Path) -> Path:
    """Path to behavior data directory."""
    return assets_dir / "behavior"


@pytest.fixture
def example_config_path(assets_dir: Path) -> Path:
    """Path to test config file."""
    return assets_dir / "test_config.yaml"


@pytest.fixture
def regression_2d_dir() -> Path:
    """Path to 2D arena regression test assets directory."""
    return ASSETS_DIR / "regression_2d"


@pytest.fixture
def regression_1d_dir() -> Path:
    """Path to 1D maze regression test assets directory."""
    return ASSETS_DIR / "regression_1d"
