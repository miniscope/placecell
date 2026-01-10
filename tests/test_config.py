"""Tests for config loading."""

from pathlib import Path

from pcell.config import AppConfig


def test_config_loads(example_config_path: Path) -> None:
    """Config file should load and have required sections."""
    cfg = AppConfig.from_yaml(example_config_path)

    assert cfg.neural is not None
    assert cfg.behavior is not None
    assert cfg.neural.oasis is not None
    assert cfg.behavior.bodypart == "LED"
