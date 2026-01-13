"""Regression tests for analysis pipeline."""

from pathlib import Path

import pandas as pd
import xarray as xr

from placecell.analysis import build_spike_place_dataframe, load_traces


def test_spike_place_regression(assets_dir: Path) -> None:
    """spike_index → spike_place should match reference output."""
    result = build_spike_place_dataframe(
        spike_index_path=assets_dir / "reference_spike_index.csv",
        neural_timestamp_path=assets_dir / "neural_data" / "neural_timestamp.csv",
        behavior_position_path=assets_dir / "behavior" / "behavior_position.csv",
        behavior_timestamp_path=assets_dir / "behavior" / "behavior_timestamp.csv",
        bodypart="LED",
        behavior_fps=30.0,
        speed_threshold=0.0,
        speed_window_frames=5,
    )

    reference = pd.read_csv(assets_dir / "reference_spike_place.csv")

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
