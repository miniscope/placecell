"""Tests for config loading."""

from pathlib import Path

import pytest

from placecell.config import AnalysisConfig, DataPathsConfig, OasisConfig


def test_config_loads(example_config_path: Path) -> None:
    """Config file should load and have required sections."""
    cfg = AnalysisConfig.from_yaml(example_config_path)

    assert cfg.neural is not None
    assert cfg.behavior is not None
    assert cfg.neural.oasis is not None
    assert cfg.behavior.bodypart == "LED_clean"


def test_oasis_config_g_required() -> None:
    """OasisConfig should require g parameter."""
    with pytest.raises(Exception):  # pydantic ValidationError
        OasisConfig(id="test", baseline="p10")  # Missing g


def test_data_paths_config_loads(assets_dir: Path) -> None:
    """DataPathsConfig should load from YAML."""
    # Create a temporary data paths config
    import tempfile

    config_content = """
id: test_data_paths
mio_model: placecell.config.DataPathsConfig
mio_version: 0.8.1
neural_path: neural_data
neural_timestamp: neural_data/neural_timestamp.csv
behavior_position: behavior/behavior_position.csv
behavior_timestamp: behavior/behavior_timestamp.csv
curation_csv: null
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        temp_path = Path(f.name)

    try:
        cfg = DataPathsConfig.from_yaml(temp_path)
        assert cfg.neural_path == "neural_data"
        assert cfg.neural_timestamp == "neural_data/neural_timestamp.csv"
        assert cfg.behavior_position == "behavior/behavior_position.csv"
        assert cfg.behavior_timestamp == "behavior/behavior_timestamp.csv"
        assert cfg.curation_csv is None
    finally:
        temp_path.unlink()


def test_data_paths_config_with_curation(assets_dir: Path) -> None:
    """DataPathsConfig should accept curation_csv path."""
    import tempfile

    config_content = """
id: test_data_paths
mio_model: placecell.config.DataPathsConfig
mio_version: 0.8.1
neural_path: neural_data
neural_timestamp: neural_data/neural_timestamp.csv
behavior_position: behavior/behavior_position.csv
behavior_timestamp: behavior/behavior_timestamp.csv
curation_csv: neural_data/curation_results.csv
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        temp_path = Path(f.name)

    try:
        cfg = DataPathsConfig.from_yaml(temp_path)
        assert cfg.curation_csv == "neural_data/curation_results.csv"
    finally:
        temp_path.unlink()
