"""Regression tests for I/O module."""

from pathlib import Path

from placecell.loaders import load_behavior_data, load_visualization_data


def test_load_behavior_data_shape(assets_dir: Path) -> None:
    """load_behavior_data should return the full behavior-rate trajectory."""
    trajectory = load_behavior_data(
        behavior_position=assets_dir / "behavior" / "behavior_position.csv",
        behavior_timestamp=assets_dir / "behavior" / "behavior_timestamp.csv",
        bodypart="LED_clean",
    )

    # Reference shape from test assets.
    assert len(trajectory) == 1500
    assert {"frame_index", "x", "y", "unix_time"}.issubset(trajectory.columns)


def test_load_visualization_data_shape(assets_dir: Path) -> None:
    """load_visualization_data should return expected shapes from test data."""
    traces, max_proj, footprints = load_visualization_data(
        neural_path=assets_dir / "neural_data",
        trace_name="C",
    )

    # Reference shapes from test assets
    assert traces.sizes["unit_id"] == 10
    assert traces.sizes["frame"] == 1000
