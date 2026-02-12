"""Behavior data loading and processing."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_curated_unit_ids(curation_csv: Path) -> list[int]:
    """Load curated unit IDs from a curation results CSV.

    Parameters
    ----------
    curation_csv:
        Path to CSV file with columns 'unit_id' and 'keep'.
        Units with keep=1 are included.

    Returns
    -------
    List of unit IDs to keep, sorted.
    """
    df = pd.read_csv(curation_csv)
    if "unit_id" not in df.columns or "keep" not in df.columns:
        raise ValueError(
            f"Curation CSV must have 'unit_id' and 'keep' columns, " f"got: {list(df.columns)}"
        )
    keep_ids = df.loc[df["keep"] == 1, "unit_id"].tolist()
    return sorted(int(uid) for uid in keep_ids)


def _load_behavior_xy(csv_path: Path, bodypart: str) -> pd.DataFrame:
    """Load DeepLabCut-style behavior CSV and return x/y coordinates per frame.

    Parameters
    ----------
    csv_path:
        Path to DeepLabCut CSV file with multi-index header.
    bodypart:
        Body part name to extract (e.g. 'LED').
    """

    # Read CSV with multi-index header (scorer, bodypart, coord)
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    scorer = None
    for col in df.columns[1:]:
        if col[1] == bodypart and col[2] == "x":
            scorer = col[0]
            break

    if scorer is None:
        available_bodyparts = {col[1] for col in df.columns[1:]}
        raise ValueError(
            f"Bodypart '{bodypart}' not found in CSV. "
            f"Available bodyparts: {sorted(available_bodyparts)}"
        )

    x = df[(scorer, bodypart, "x")]
    y = df[(scorer, bodypart, "y")]
    frame_index = df.iloc[:, 0]

    out = pd.DataFrame({"frame_index": frame_index, "x": x, "y": y})
    return out


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
