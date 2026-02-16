"""I/O functions for loading behavior and neural data."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from placecell.behavior import _load_behavior_xy, compute_behavior_speed, filter_by_speed
from placecell.logging import init_logger
from placecell.neural import load_calcium_traces

logger = init_logger(__name__)


def compute_overlap_time_range(
    neural_timestamp: Path,
    behavior_timestamp: Path,
    use_neural_last_timestamp: bool = True,
) -> tuple[float, float]:
    """Compute the overlapping time range between neural and behavior recordings.

    Parameters
    ----------
    neural_timestamp:
        Path to neural timestamp CSV (columns: frame, timestamp_first, timestamp_last).
    behavior_timestamp:
        Path to behavior timestamp CSV (columns: frame_index, unix_time).
    use_neural_last_timestamp:
        Whether to use timestamp_last for neural frames.

    Returns
    -------
    tuple[float, float]
        (start_time, end_time) of the overlapping window in unix time.
    """
    neural_ts = pd.read_csv(neural_timestamp)
    beh_ts = pd.read_csv(behavior_timestamp)

    ts_col = "timestamp_last" if use_neural_last_timestamp else "timestamp_first"
    neural_start = neural_ts[ts_col].min()
    neural_end = neural_ts[ts_col].max()

    beh_start = beh_ts["unix_time"].min()
    beh_end = beh_ts["unix_time"].max()

    overlap_start = max(neural_start, beh_start)
    overlap_end = min(neural_end, beh_end)

    if overlap_start >= overlap_end:
        raise ValueError(
            f"No time overlap between neural [{neural_start:.1f}, {neural_end:.1f}] "
            f"and behavior [{beh_start:.1f}, {beh_end:.1f}]."
        )

    neural_dur = neural_end - neural_start
    beh_dur = beh_end - beh_start
    overlap_dur = overlap_end - overlap_start
    logger.info(
        f"Time overlap: {overlap_dur:.1f}s "
        f"(neural: {neural_dur:.1f}s, behavior: {beh_dur:.1f}s)"
    )

    return overlap_start, overlap_end


def load_behavior_data(
    behavior_position: Path,
    behavior_timestamp: Path,
    bodypart: str,
    speed_window_frames: int,
    speed_threshold: float,
    time_range: tuple[float, float] | None = None,
    x_col: str = "x",
    y_col: str = "y",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load behavior data and compute speed-filtered trajectory.

    Parameters
    ----------
    behavior_position:
        Path to behavior position CSV file.
    behavior_timestamp:
        Path to behavior timestamp CSV file.
    bodypart:
        Body part name to use for trajectory.
    speed_window_frames:
        Window size for speed computation.
    speed_threshold:
        Minimum speed threshold.
    time_range:
        Optional (start_time, end_time) to restrict behavior data.
        Use :func:`compute_overlap_time_range` to get this from neural/behavior timestamps.
    x_col:
        Coordinate column name for the x-axis in the behavior CSV.
    y_col:
        Coordinate column name for the y-axis in the behavior CSV.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (trajectory_with_speed, trajectory_filtered) - full trajectory with speed
        and speed-filtered trajectory.
    """
    if not behavior_position.exists():
        raise FileNotFoundError(
            f"Behavior position file not found: {behavior_position}. "
            "This is required for full trajectory plotting and occupancy calculation."
        )
    if not behavior_timestamp.exists():
        raise FileNotFoundError(
            f"Behavior timestamp file not found: {behavior_timestamp}. "
            "This is required for speed calculation."
        )

    full_trajectory = _load_behavior_xy(
        behavior_position, bodypart=bodypart, x_col=x_col, y_col=y_col
    )
    behavior_timestamps = pd.read_csv(behavior_timestamp)

    trajectory_with_speed = compute_behavior_speed(
        positions=full_trajectory,
        timestamps=behavior_timestamps,
        window_frames=speed_window_frames,
    )

    # Trim to time range if specified
    if time_range is not None:
        t_start, t_end = time_range
        mask = (trajectory_with_speed["unix_time"] >= t_start) & (
            trajectory_with_speed["unix_time"] <= t_end
        )
        n_before = len(trajectory_with_speed)
        trajectory_with_speed = trajectory_with_speed[mask].reset_index(drop=True)
        logger.info(
            f"Trimmed behavior to overlap window: {len(trajectory_with_speed)}/{n_before} frames"
        )

    trajectory_filtered = filter_by_speed(trajectory_with_speed, speed_threshold)

    return trajectory_with_speed, trajectory_filtered


def load_visualization_data(
    neural_path: Path | None,
    trace_name: str,
) -> tuple[Any, np.ndarray | None, Any]:
    """Load visualization data: traces, max projection, and footprints.

    Parameters
    ----------
    neural_path:
        Path to neural data directory.
    trace_name:
        Name of trace zarr to load.

    Returns
    -------
    tuple
        (traces, max_proj, footprints) - xarray DataArray or None for each.
    """
    traces = None
    max_proj = None
    footprints = None

    if neural_path is None:
        return traces, max_proj, footprints

    neural_path = Path(neural_path)

    # Load traces
    try:
        traces = load_calcium_traces(neural_path, trace_name=trace_name)
    except FileNotFoundError:
        logger.warning(f"{trace_name}.zarr not found at {neural_path}. Trace display disabled.")
    except (KeyError, ValueError) as e:
        logger.warning(f"Failed to load traces from {trace_name}.zarr: {e}")

    # Load max projection and footprints
    try:
        max_proj_path = neural_path / "max_proj.zarr"
        if max_proj_path.exists():
            max_proj_ds = xr.open_zarr(max_proj_path, consolidated=False)
            if "max_proj" in max_proj_ds:
                mp = max_proj_ds["max_proj"]
            else:
                mp = max_proj_ds[list(max_proj_ds.data_vars)[0]]
            if "quantile" in mp.dims:
                mp = mp.isel(quantile=0)
            max_proj = np.asarray(mp.values, dtype=float)
        else:
            logger.warning(
                f"max_proj.zarr not found at {neural_path}. Cell footprint overlay disabled."
            )

        a_path = neural_path / "A.zarr"
        if a_path.exists():
            A_ds = xr.open_zarr(a_path, consolidated=False)
            footprints = A_ds["A"] if "A" in A_ds else A_ds[list(A_ds.data_vars)[0]]

            # Validate unit_id coordinates
            if "unit_id" in footprints.coords:
                unit_ids = footprints.coords["unit_id"].values
                if len(set(unit_ids)) == 1:
                    logger.warning(
                        f"A.zarr has corrupted unit_id coordinates (all values are {unit_ids[0]}). "
                        "Cell footprint overlay disabled. Re-export A.zarr with valid coordinates."
                    )
                    footprints = None
        else:
            logger.warning(f"A.zarr not found at {neural_path}. Cell footprint overlay disabled.")
    except (FileNotFoundError, KeyError, ValueError, OSError) as e:
        logger.warning(f"Failed to load neural visualization data: {e}")

    return traces, max_proj, footprints
