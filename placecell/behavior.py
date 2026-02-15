"""Behavior data loading and processing."""

from pathlib import Path

import numpy as np
import pandas as pd


def _load_behavior_xy(
    csv_path: Path,
    bodypart: str,
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """Load DeepLabCut-style behavior CSV and return x/y coordinates per frame.

    Parameters
    ----------
    csv_path:
        Path to DeepLabCut CSV file with multi-index header.
    bodypart:
        Body part name to extract (e.g. 'LED').
    x_col:
        Coordinate column name for the x-axis (default 'x').
    y_col:
        Coordinate column name for the y-axis (default 'y').
    """

    # Read CSV with multi-index header (scorer, bodypart, coord)
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    scorer = None
    for col in df.columns[1:]:
        if col[1] == bodypart and col[2] == x_col:
            scorer = col[0]
            break

    if scorer is None:
        available_bodyparts = {col[1] for col in df.columns[1:]}
        raise ValueError(
            f"Bodypart '{bodypart}' not found in CSV. "
            f"Available bodyparts: {sorted(available_bodyparts)}"
        )

    x = df[(scorer, bodypart, x_col)]
    y = df[(scorer, bodypart, y_col)]
    frame_index = df.iloc[:, 0]

    out = pd.DataFrame({"frame_index": frame_index, "x": x, "y": y})
    return out


def remove_position_jumps(
    positions: pd.DataFrame,
    threshold_px: float,
) -> tuple[pd.DataFrame, int]:
    """Replace implausible position jumps with linear interpolation.

    A jump is detected when the frame-to-frame displacement exceeds
    *threshold_px*.  Jumped frames have their x/y replaced by linearly
    interpolated values from the surrounding good frames.

    Parameters
    ----------
    positions:
        DataFrame with columns ``x``, ``y`` (and any others, preserved).
    threshold_px:
        Maximum plausible frame-to-frame displacement in pixels.

    Returns
    -------
    tuple of (DataFrame with jumps interpolated, number of frames fixed).
    """
    df = positions.copy()
    x = df["x"].values.astype(float)
    y = df["y"].values.astype(float)

    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx**2 + dy**2)

    bad = np.zeros(len(x), dtype=bool)
    # Mark frames that *arrive* via a jump.
    bad[1:] = dist > threshold_px
    # If a frame also *departs* via a jump (single-frame glitch), the next
    # good frame is already fine; we only NaN the jumped-to frame.

    n_bad = bad.sum()
    if n_bad > 0:
        x[bad] = np.nan
        y[bad] = np.nan
        # pandas interpolation handles NaN edges via ffill/bfill
        df["x"] = pd.Series(x).interpolate(limit_direction="both").values
        df["y"] = pd.Series(y).interpolate(limit_direction="both").values

    return df, int(n_bad)


def correct_perspective(
    positions: pd.DataFrame,
    arena_bounds: tuple[float, float, float, float],
    camera_height_mm: float,
    tracking_height_mm: float,
) -> pd.DataFrame:
    """Correct perspective distortion from overhead camera parallax.

    An LED at height *h* above the floor appears shifted radially outward
    from the optical axis.  The corrected position is::

        x_corrected = cx + (x - cx) * (H - h) / H

    where *cx, cy* is the arena center (midpoint of *arena_bounds*),
    *H* is the camera height, and *h* is the tracking height.

    Parameters
    ----------
    positions:
        DataFrame with columns ``x``, ``y``.
    arena_bounds:
        (x_min, x_max, y_min, y_max) in pixels.
    camera_height_mm:
        Camera height above floor in mm.
    tracking_height_mm:
        Tracked point height above floor in mm.

    Returns
    -------
    DataFrame with corrected ``x``, ``y``.
    """
    x_min, x_max, y_min, y_max = arena_bounds
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    factor = (camera_height_mm - tracking_height_mm) / camera_height_mm

    df = positions.copy()
    df["x"] = cx + (df["x"] - cx) * factor
    df["y"] = cy + (df["y"] - cy) * factor
    return df


def clip_to_arena(
    positions: pd.DataFrame,
    arena_bounds: tuple[float, float, float, float],
) -> pd.DataFrame:
    """Clip positions to arena boundaries.

    Points outside the arena (from detection errors) are clamped to the
    nearest boundary edge.

    Parameters
    ----------
    positions:
        DataFrame with columns ``x``, ``y``.
    arena_bounds:
        (x_min, x_max, y_min, y_max) in pixels.

    Returns
    -------
    DataFrame with ``x``, ``y`` clipped to arena bounds.
    """
    x_min, x_max, y_min, y_max = arena_bounds
    df = positions.copy()
    df["x"] = df["x"].clip(x_min, x_max)
    df["y"] = df["y"].clip(y_min, y_max)
    return df


def recompute_speed(
    trajectory: pd.DataFrame,
    window_frames: int,
) -> pd.DataFrame:
    """Recompute speed on a trajectory that already has ``x``, ``y``, ``unix_time``.

    Use this after spatial corrections (jump removal, perspective, clipping)
    to update the ``speed`` column from the corrected positions.

    Parameters
    ----------
    trajectory:
        DataFrame with columns ``x``, ``y``, ``unix_time``.
    window_frames:
        Number of frames to look ahead for speed calculation.

    Returns
    -------
    DataFrame with ``speed`` column overwritten (in position-units / s).
    """
    df = trajectory.sort_values("frame_index")
    x_vals = df["x"].values
    y_vals = df["y"].values
    t_vals = df["unix_time"].values
    n = len(df)
    speed = np.zeros(n)
    for i in range(n):
        end_idx = min(i + window_frames, n - 1)
        if end_idx > i:
            dx = x_vals[end_idx] - x_vals[i]
            dy = y_vals[end_idx] - y_vals[i]
            dt = t_vals[end_idx] - t_vals[i]
            if dt > 0:
                speed[i] = np.sqrt(dx**2 + dy**2) / dt
    trajectory = trajectory.copy()
    trajectory.loc[df.index, "speed"] = speed
    return trajectory


def filter_by_speed(
    trajectory: pd.DataFrame,
    speed_threshold: float,
) -> pd.DataFrame:
    """Filter trajectory to frames above a speed threshold.

    Parameters
    ----------
    trajectory:
        DataFrame with columns ``frame_index`` and ``speed``.
    speed_threshold:
        Minimum speed to keep.

    Returns
    -------
    Filtered copy, sorted by frame index, with ``frame_index`` renamed
    to ``beh_frame_index``.
    """
    filtered = trajectory[trajectory["speed"] >= speed_threshold].copy()
    filtered = filtered.sort_values("frame_index")
    filtered = filtered.rename(columns={"frame_index": "beh_frame_index"})
    return filtered


def compute_behavior_speed(
    positions: pd.DataFrame,
    timestamps: pd.DataFrame,
    window_frames: int = 10,
) -> pd.DataFrame:
    """Compute speed from behavior positions and timestamps using a window.

    Speed is calculated over a window of frames for stability, especially
    useful at high frame rates where consecutive frame differences are noisy.

    Parameters
    ----------
    positions:
        DataFrame with columns `frame_index`, `x`, `y` in pixels.
    timestamps:
        DataFrame with columns `frame_index`, `unix_time` in seconds.
    window_frames:
        Number of frames to use for speed calculation. Speed is computed
        as distance traveled over this window divided by time elapsed.

    Returns
    -------
    DataFrame with speed in pixels/s.
    """

    df = positions.merge(timestamps, on="frame_index", how="inner").sort_values("frame_index")

    x_vals = df["x"].values
    y_vals = df["y"].values
    t_vals = df["unix_time"].values

    n = len(df)
    distances = np.zeros(n)
    time_diffs = np.zeros(n)

    for i in range(n):
        # Look ahead by window_frames, but don't go past the end
        end_idx = min(i + window_frames, n - 1)
        if end_idx > i:
            dx = x_vals[end_idx] - x_vals[i]
            dy = y_vals[end_idx] - y_vals[i]
            dt = t_vals[end_idx] - t_vals[i]

            dist = np.sqrt(dx**2 + dy**2)
            distances[i] = dist
            time_diffs[i] = dt if dt > 0 else np.nan
        else:
            distances[i] = 0.0
            time_diffs[i] = np.nan

    speed = distances / time_diffs
    df["speed"] = pd.Series(speed).fillna(0.0)
    return df


def build_event_place_dataframe(
    event_index: pd.DataFrame,
    neural_timestamp_path: Path,
    behavior_with_speed: pd.DataFrame,
    behavior_fps: float,
    speed_threshold: float = 50.0,
) -> pd.DataFrame:
    """Match neural events to behavior positions for place-cell analysis.

    For each event, finds the closest behavior frame in time, filters out
    matches where the timestamp difference exceeds 0.5 / behavior_fps, and
    drops events below the speed threshold.

    Parameters
    ----------
    event_index:
        DataFrame with columns: unit_id, frame, s.
    neural_timestamp_path:
        Path to neural timestamp CSV (columns: frame, timestamp_first, timestamp_last).
    behavior_with_speed:
        Trajectory DataFrame with columns: frame_index, x, y, unix_time, speed.
    behavior_fps:
        Behavior sampling rate (Hz).
    speed_threshold:
        Minimum running speed to keep events (pixels/s).

    Returns
    -------
    DataFrame with columns: unit_id, frame, s, neural_time,
    beh_frame_index, beh_time, x, y, speed.
    """
    neural_ts = pd.read_csv(neural_timestamp_path)
    neural_ts = neural_ts.rename(columns={"timestamp_last": "neural_time"})[
        ["frame", "neural_time"]
    ]

    beh = behavior_with_speed.rename(
        columns={"frame_index": "beh_frame_index", "unix_time": "beh_time"}
    )

    events = event_index.merge(neural_ts, on="frame", how="left")

    beh_times = beh[["beh_frame_index", "beh_time", "x", "y", "speed"]]
    beh_times = beh_times.sort_values("beh_time").reset_index(drop=True)

    event_times = events["neural_time"].to_numpy()
    beh_time_arr = beh_times["beh_time"].to_numpy()

    time_threshold = 0.5 / behavior_fps

    idx = np.searchsorted(beh_time_arr, event_times, side="left")
    idx_clipped = np.clip(idx, 0, len(beh_time_arr) - 1)

    idx_left = idx_clipped
    idx_right = np.clip(idx_clipped + 1, 0, len(beh_time_arr) - 1)

    time_diff_left = np.abs(event_times - beh_time_arr[idx_left])
    time_diff_right = np.abs(event_times - beh_time_arr[idx_right])

    use_right = time_diff_right < time_diff_left
    idx_final = np.where(use_right, idx_right, idx_left)
    time_diff_final = np.where(use_right, time_diff_right, time_diff_left)

    beh_matched = beh_times.iloc[idx_final].reset_index(drop=True)
    out = pd.concat([events.reset_index(drop=True), beh_matched], axis=1)

    out = out[time_diff_final <= time_threshold].reset_index(drop=True)
    out = out[out["speed"] >= float(speed_threshold)].reset_index(drop=True)
    return out
