"""I/O functions for loading behavior and neural data."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from placecell.behavior import _load_behavior_xy, compute_behavior_speed
from placecell.logging import init_logger
from placecell.neural import load_calcium_traces

logger = init_logger(__name__)


def load_behavior_data(
    behavior_position: Path,
    behavior_timestamp: Path,
    bodypart: str,
    speed_window_frames: int,
    speed_threshold: float,
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

    full_trajectory = _load_behavior_xy(behavior_position, bodypart=bodypart)
    behavior_timestamps = pd.read_csv(behavior_timestamp)

    trajectory_with_speed = compute_behavior_speed(
        positions=full_trajectory,
        timestamps=behavior_timestamps,
        window_frames=speed_window_frames,
    )

    trajectory_filtered = trajectory_with_speed[trajectory_with_speed["speed"] >= speed_threshold]
    trajectory_filtered = trajectory_filtered.sort_values("frame_index")
    trajectory_filtered = trajectory_filtered.rename(columns={"frame_index": "beh_frame_index"})

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
