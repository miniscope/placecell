"""Tests for config loading."""

from pathlib import Path

import pytest

from placecell.config import AnalysisConfig, DataConfig, OasisConfig


def test_config_loads(example_config_path: Path) -> None:
    """Config file should load and have required sections."""
    cfg = AnalysisConfig.from_yaml(example_config_path)

    assert cfg.neural is not None
    assert cfg.behavior is not None
    assert cfg.neural.oasis is not None
    assert cfg.behavior.speed_threshold == 50.0


def test_oasis_config_g_required() -> None:
    """OasisConfig should require g parameter."""
    with pytest.raises(Exception):  # pydantic ValidationError
        OasisConfig(id="test", baseline="p10")  # Missing g


def test_data_paths_config_loads(assets_dir: Path) -> None:
    """DataConfig should load from YAML."""
    # Create a temporary data paths config
    import tempfile

    config_content = """
id: test_data_paths
mio_model: placecell.config.DataConfig
mio_version: 0.8.1
behavior_fps: 20.0
neural_path: neural_data
neural_timestamp: neural_data/neural_timestamp.csv
behavior_position: behavior/behavior_position.csv
behavior_timestamp: behavior/behavior_timestamp.csv
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        temp_path = Path(f.name)

    try:
        cfg = DataConfig.from_yaml(temp_path)
        assert cfg.neural_path == "neural_data"
        assert cfg.neural_timestamp == "neural_data/neural_timestamp.csv"
        assert cfg.behavior_position == "behavior/behavior_position.csv"
        assert cfg.behavior_timestamp == "behavior/behavior_timestamp.csv"
    finally:
        temp_path.unlink()
