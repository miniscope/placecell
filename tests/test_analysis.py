"""Regression tests for analysis pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from placecell.analysis import (
    build_event_place_dataframe,
    compute_behavior_speed,
    load_curated_unit_ids,
    load_traces,
)


def test_event_place_regression(assets_dir: Path) -> None:
    """event_index → event_place should match reference output."""
    result = build_event_place_dataframe(
        event_index_path=assets_dir / "reference_event_index.csv",
        neural_timestamp_path=assets_dir / "neural_data" / "neural_timestamp.csv",
        behavior_position_path=assets_dir / "behavior" / "behavior_position.csv",
        behavior_timestamp_path=assets_dir / "behavior" / "behavior_timestamp.csv",
        bodypart="LED_clean",
        behavior_fps=20.0,
        speed_threshold=0.0,
        speed_window_frames=5,
    )

    reference = pd.read_csv(assets_dir / "reference_event_place.csv")

    # Same shape and columns
    assert list(result.columns) == list(reference.columns)
    assert len(result) == len(reference)

    # Values match (with float tolerance)
    pd.testing.assert_frame_equal(result, reference, rtol=1e-5)


def test_load_traces_shape(neural_path: Path) -> None:
    """load_traces should return correct dimensions."""
    C = load_traces(neural_path, trace_name="C")

    assert C.dims == ("unit_id", "frame")
    assert C.sizes["unit_id"] == 10
    assert C.sizes["frame"] == 1000


def test_deconv_zarr_structure(assets_dir: Path) -> None:
    """Deconv zarr should have C_deconv and S arrays."""
    ds = xr.open_zarr(assets_dir / "reference_deconv.zarr", consolidated=False)

    assert "C_deconv" in ds
    assert "S" in ds
    assert ds.attrs["fps"] == 20.0


def test_compute_behavior_speed_shape(assets_dir: Path) -> None:
    """compute_behavior_speed should return DataFrame with speed column."""
    positions = pd.DataFrame({
        "frame_index": [0, 1, 2, 3, 4],
        "x": [0.0, 10.0, 20.0, 30.0, 40.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    timestamps = pd.DataFrame({
        "frame_index": [0, 1, 2, 3, 4],
        "unix_time": [0.0, 0.1, 0.2, 0.3, 0.4],
    })

    result = compute_behavior_speed(positions, timestamps, window_frames=2)

    assert "speed" in result.columns
    assert len(result) == 5
    # Speed should be ~100 pixels/s (10 pixels / 0.1 s)
    assert result["speed"].iloc[0] == pytest.approx(100.0, rel=0.01)


def test_load_curated_unit_ids(assets_dir: Path) -> None:
    """load_curated_unit_ids should filter and sort unit IDs."""
    # Create a temporary curation CSV
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("unit_id,keep\n")
        f.write("5,1\n")
        f.write("2,0\n")
        f.write("3,1\n")
        f.write("1,1\n")
        f.write("4,0\n")
        temp_path = Path(f.name)

    try:
        result = load_curated_unit_ids(temp_path)
        assert result == [1, 3, 5]  # Sorted, only keep=1
    finally:
        temp_path.unlink()
