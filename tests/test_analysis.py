"""Regression tests for analysis pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from placecell.analysis import (
    compute_occupancy_map,
    compute_rate_map,
    compute_spatial_information,
    compute_unit_analysis,
    gaussian_filter_normalized,
)
from placecell.behavior import (
    build_event_place_dataframe,
    compute_behavior_speed,
)
from placecell.neural import load_calcium_traces


def test_event_place_regression(assets_dir: Path) -> None:
    """event_index → event_place should match reference output."""
    from placecell.behavior import _load_behavior_xy

    event_index = pd.read_csv(assets_dir / "reference_event_index.csv")
    beh_pos = _load_behavior_xy(assets_dir / "behavior" / "behavior_position.csv", "LED_clean")
    beh_ts = pd.read_csv(assets_dir / "behavior" / "behavior_timestamp.csv")
    behavior_with_speed = compute_behavior_speed(beh_pos, beh_ts, window_frames=5)

    result = build_event_place_dataframe(
        event_index=event_index,
        neural_timestamp_path=assets_dir / "neural_data" / "neural_timestamp.csv",
        behavior_with_speed=behavior_with_speed,
        behavior_fps=20.0,
        speed_threshold=0.0,
    )

    reference = pd.read_csv(assets_dir / "reference_event_place.csv")

    assert list(result.columns) == list(reference.columns)
    assert len(result) == len(reference)

    pd.testing.assert_frame_equal(result, reference, rtol=1e-5)


def test_load_calcium_traces_shape(neural_path: Path) -> None:
    """load_calcium_traces should return correct dimensions."""
    C = load_calcium_traces(neural_path, trace_name="C")

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


def test_gaussian_filter_normalized(assets_dir: Path) -> None:
    """gaussian_filter_normalized should match reference output."""
    ref = np.load(assets_dir / "reference_spatial.npz")
    
    result = gaussian_filter_normalized(ref["gaussian_input"], sigma=1.0)
    
    np.testing.assert_allclose(result, ref["gaussian_result"], rtol=1e-10)


def test_compute_occupancy_map(assets_dir: Path) -> None:
    """compute_occupancy_map should match reference output."""
    ref = np.load(assets_dir / "reference_spatial.npz")
    
    # Load full trajectory (same as reference generation)
    behavior_pos = pd.read_csv(
        assets_dir / "behavior/behavior_position.csv", header=[0, 1, 2], index_col=0
    )
    scorer = behavior_pos.columns[0][0]
    trajectory = pd.DataFrame({
        "x": behavior_pos[(scorer, "LED_clean", "x")].values,
        "y": behavior_pos[(scorer, "LED_clean", "y")].values,
    })
    trajectory = trajectory.dropna()

    occupancy, valid_mask, x_edges, y_edges = compute_occupancy_map(
        trajectory, bins=20, behavior_fps=20.0, occupancy_sigma=1.0, min_occupancy=0.1
    )

    np.testing.assert_allclose(occupancy, ref["occupancy"], rtol=1e-10)
    np.testing.assert_array_equal(valid_mask, ref["valid_mask"])
    np.testing.assert_allclose(x_edges, ref["x_edges"], rtol=1e-10)
    np.testing.assert_allclose(y_edges, ref["y_edges"], rtol=1e-10)


def test_compute_rate_map(assets_dir: Path) -> None:
    """compute_rate_map should match reference output."""
    ref = np.load(assets_dir / "reference_spatial.npz")
    
    # Load events
    event_place = pd.read_csv(assets_dir / "reference_event_place.csv")
    unit_id = event_place["unit_id"].iloc[0]
    unit_events = event_place[event_place["unit_id"] == unit_id][["x", "y", "s"]].dropna()
    
    x_edges = ref["x_edges"]
    y_edges = ref["y_edges"]
    unit_events = unit_events[
        (unit_events["x"] >= x_edges[0]) & (unit_events["x"] <= x_edges[-1]) &
        (unit_events["y"] >= y_edges[0]) & (unit_events["y"] <= y_edges[-1])
    ]

    rate_map = compute_rate_map(
        unit_events, ref["occupancy"], ref["valid_mask"], x_edges, y_edges, activity_sigma=1.0
    )

    # Use nan-aware comparison
    np.testing.assert_allclose(rate_map, ref["rate_map"], rtol=1e-10, equal_nan=True)


def test_compute_spatial_information(assets_dir: Path) -> None:
    """compute_spatial_information should match reference output."""
    ref = np.load(assets_dir / "reference_spatial.npz")
    
    # Load full trajectory (same as reference generation)
    behavior_pos = pd.read_csv(
        assets_dir / "behavior/behavior_position.csv", header=[0, 1, 2], index_col=0
    )
    scorer = behavior_pos.columns[0][0]
    trajectory = pd.DataFrame({
        "x": behavior_pos[(scorer, "LED_clean", "x")].values,
        "y": behavior_pos[(scorer, "LED_clean", "y")].values,
    })
    trajectory = trajectory.dropna()
    trajectory["beh_frame_index"] = range(len(trajectory))
    
    # Load events
    event_place = pd.read_csv(assets_dir / "reference_event_place.csv")
    unit_id = event_place["unit_id"].iloc[0]
    unit_events = event_place[event_place["unit_id"] == unit_id][["x", "y", "s", "beh_frame_index"]].dropna()
    unit_events = unit_events[
        (unit_events["beh_frame_index"] >= 0) & 
        (unit_events["beh_frame_index"] < len(trajectory))
    ]

    si, p_val, shuffled = compute_spatial_information(
        unit_events, trajectory, ref["occupancy"], ref["valid_mask"],
        ref["x_edges"], ref["y_edges"], n_shuffles=100, random_seed=42,
        activity_sigma=1.0,
    )

    assert si == pytest.approx(float(ref["spatial_info"]), rel=1e-10)
    assert p_val == pytest.approx(float(ref["spatial_info_pval"]), rel=1e-10)


def test_compute_unit_analysis(assets_dir: Path) -> None:
    """compute_unit_analysis should match reference rate map and SI values."""
    ref = np.load(assets_dir / "reference_spatial.npz")
    
    # Load full trajectory (same as reference generation)
    behavior_pos = pd.read_csv(
        assets_dir / "behavior/behavior_position.csv", header=[0, 1, 2], index_col=0
    )
    scorer = behavior_pos.columns[0][0]
    trajectory = pd.DataFrame({
        "x": behavior_pos[(scorer, "LED_clean", "x")].values,
        "y": behavior_pos[(scorer, "LED_clean", "y")].values,
    })
    trajectory = trajectory.dropna()
    trajectory["beh_frame_index"] = range(len(trajectory))
    
    # Load events - same filtering as reference generation
    event_place = pd.read_csv(assets_dir / "reference_event_place.csv")
    unit_id = int(event_place["unit_id"].iloc[0])
    
    # Filter same way as reference: valid beh_frame_index range
    df_filtered = event_place[
        (event_place["beh_frame_index"] >= 0) & 
        (event_place["beh_frame_index"] < len(trajectory))
    ].copy()
    
    # Also filter by spatial bounds (same as reference rate_map generation)
    x_edges = ref["x_edges"]
    y_edges = ref["y_edges"]
    df_filtered = df_filtered[
        (df_filtered["x"] >= x_edges[0]) & (df_filtered["x"] <= x_edges[-1]) &
        (df_filtered["y"] >= y_edges[0]) & (df_filtered["y"] <= y_edges[-1])
    ]

    result = compute_unit_analysis(
        unit_id=unit_id,
        df_filtered=df_filtered,
        trajectory_df=trajectory,
        occupancy_time=ref["occupancy"],
        valid_mask=ref["valid_mask"],
        x_edges=x_edges,
        y_edges=y_edges,
        activity_sigma=1.0,
        event_threshold_sigma=2.0,
        n_shuffles=100,
        random_seed=42,
    )

    # Rate map should match reference (same as compute_rate_map test)
    np.testing.assert_allclose(result["rate_map"], ref["rate_map"], rtol=1e-10, equal_nan=True)

    # SI should match reference (same as compute_spatial_information test)
    assert result["si"] == pytest.approx(float(ref["spatial_info"]), rel=1e-10)
    assert result["p_val"] == pytest.approx(float(ref["spatial_info_pval"]), rel=1e-10)

    # Vis threshold should be deterministic
    assert result["vis_threshold"] == pytest.approx(float(ref["vis_threshold"]), rel=1e-10)
