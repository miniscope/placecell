"""Maze behavior processing for 1D tube analysis."""

import numpy as np
import pandas as pd

from placecell.logging import init_logger

logger = init_logger(__name__)


def serialize_tube_position(
    trajectory: pd.DataFrame,
    tube_order: list[str],
    zone_column: str = "zone",
    tube_position_column: str = "tube_position",
) -> pd.DataFrame:
    """Convert zone + tube_position to a concatenated 1D position.

    Filters to tube-only frames and maps each tube's [0,1] position
    to [i, i+1] on the concatenated axis based on tube_order.

    Parameters
    ----------
    trajectory:
        DataFrame with zone, tube_position, and standard columns.
    tube_order:
        Ordered list of tube zone names.
    zone_column:
        Column containing zone labels.
    tube_position_column:
        Column containing within-tube position (0-1).

    Returns
    -------
    pd.DataFrame
        Filtered to tube frames only, with new columns ``pos_1d``
        and ``tube_index``.
    """
    tube_set = set(tube_order)
    tp = pd.to_numeric(trajectory[tube_position_column], errors="coerce")
    mask = trajectory[zone_column].isin(tube_set) & tp.notna()
    df = trajectory[mask].copy()
    df[tube_position_column] = tp[mask]

    zone_to_offset = {zone: float(i) for i, zone in enumerate(tube_order)}
    df["tube_index"] = df[zone_column].map({z: i for i, z in enumerate(tube_order)})
    df["pos_1d"] = df[tube_position_column].astype(float) + df[zone_column].map(zone_to_offset)

    logger.info(
        "Serialized %d/%d frames to 1D (tubes: %s)",
        len(df),
        len(trajectory),
        tube_order,
    )
    return df


def compute_speed_1d(
    trajectory: pd.DataFrame,
    window_frames: int = 5,
) -> pd.DataFrame:
    """Compute 1D speed along the tube axis.

    Speed is computed as |delta(pos_1d)| / delta(time) over a window,
    but only within the same tube (cross-tube transitions get speed=0).

    Parameters
    ----------
    trajectory:
        DataFrame with pos_1d, tube_index, unix_time, frame_index columns.
    window_frames:
        Look-ahead window for speed computation.

    Returns
    -------
    pd.DataFrame with ``speed_1d`` column added.
    """
    df = trajectory.sort_values("frame_index").copy()
    pos = df["pos_1d"].values
    t = df["unix_time"].values
    tube_idx = df["tube_index"].values
    n = len(df)

    speed = np.zeros(n)
    for i in range(n):
        end = min(i + window_frames, n - 1)
        if end > i and tube_idx[i] == tube_idx[end]:
            dt = t[end] - t[i]
            if dt > 0:
                speed[i] = abs(pos[end] - pos[i]) / dt
    df["speed_1d"] = speed
    return df


def filter_tube_by_speed(
    trajectory: pd.DataFrame,
    speed_threshold: float,
) -> pd.DataFrame:
    """Filter 1D trajectory to frames above speed threshold.

    Parameters
    ----------
    trajectory:
        DataFrame with speed_1d and frame_index columns.
    speed_threshold:
        Minimum 1D speed to keep.

    Returns
    -------
    Filtered DataFrame with frame_index renamed to beh_frame_index.
    """
    filtered = trajectory[trajectory["speed_1d"] >= speed_threshold].copy()
    filtered = filtered.sort_values("frame_index")
    filtered = filtered.rename(columns={"frame_index": "beh_frame_index"})
    return filtered
