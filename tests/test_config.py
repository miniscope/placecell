"""Tests for config loading."""

from pathlib import Path

import pytest

from placecell.config import (
    AnalysisConfig,
    ArenaBehaviorDataConfig,
    DataConfig,
    MazeBehaviorDataConfig,
    OasisConfig,
)


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


def _write_yaml(text: str) -> Path:
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(text)
        return Path(f.name)


def test_data_config_combined() -> None:
    """DataConfig with both neural and behavior blocks loads + dispatches arena."""
    path = _write_yaml(
        """
neural:
  path: neural_data
  timestamp: neural_data/neural_timestamp.csv
behavior:
  type: arena
  fps: 20.0
  position: behavior/behavior_position.csv
  timestamp: behavior/behavior_timestamp.csv
"""
    )
    try:
        cfg = DataConfig.from_yaml(path)
        assert isinstance(cfg.behavior, ArenaBehaviorDataConfig)
        assert cfg.neural is not None
        assert cfg.neural.path == "neural_data"
        assert cfg.behavior.fps == 20.0
        assert cfg.behavior.position == "behavior/behavior_position.csv"
    finally:
        path.unlink()


def test_data_config_neural_only() -> None:
    """Behavior block may be omitted for neural-only sessions."""
    path = _write_yaml(
        """
neural:
  path: neural_data
  timestamp: neural_data/neural_timestamp.csv
"""
    )
    try:
        cfg = DataConfig.from_yaml(path)
        assert cfg.neural is not None
        assert cfg.behavior is None
    finally:
        path.unlink()


def test_data_config_behavior_only_maze() -> None:
    """Neural block may be omitted for behavior-only sessions; maze dispatches."""
    path = _write_yaml(
        """
behavior:
  type: maze
  fps: 20.0
  position: behavior/behavior_position.csv
  timestamp: behavior/behavior_timestamp.csv
  arm_order: [Arm_1, Arm_2]
"""
    )
    try:
        cfg = DataConfig.from_yaml(path)
        assert cfg.neural is None
        assert isinstance(cfg.behavior, MazeBehaviorDataConfig)
        assert cfg.behavior.arm_order == ["Arm_1", "Arm_2"]
    finally:
        path.unlink()


def test_data_config_requires_at_least_one_block() -> None:
    """An empty data config (no neural and no behavior) is rejected."""
    path = _write_yaml("config_override: {}\n")
    try:
        with pytest.raises(Exception):  # pydantic ValidationError
            DataConfig.from_yaml(path)
    finally:
        path.unlink()
