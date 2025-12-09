"""Analysis functions for place cells."""

from pathlib import Path

import numpy as np
import pandas as pd


def _load_behavior_xy(csv_path: Path, bodypart: str) -> pd.DataFrame:
    """Load DeepLabCut-style behavior CSV and return x/y coordinates per frame.

    Parameters
    ----------
    csv_path:
        Path to DeepLabCut CSV file with multi-index header.
    bodypart:
        Body part name to extract (e.g. 'LED').
    """

    df = pd.read_csv(csv_path, header=[0, 1, 2])
    scorer = df.columns[1][0]
    x = df[(scorer, bodypart, "x")]
    y = df[(scorer, bodypart, "y")]
    out = pd.DataFrame({"frame_index": df.index, "x": x, "y": y})
    return out


def compute_behavior_speed(
    positions: pd.DataFrame,
    timestamps: pd.DataFrame,
    window_frames: int = 5,
) -> pd.DataFrame:
    """Compute speed from behavior positions and timestamps using a window.

    Speed is calculated over a window of frames for stability, especially
    useful at high frame rates where consecutive frame differences are noisy.

    Parameters
    ----------
    positions:
        DataFrame with columns ``frame_index``, ``x``, ``y`` in pixels.
    timestamps:
        DataFrame with columns ``frame_index``, ``unix_time`` in seconds.
    window_frames:
        Number of frames to use for speed calculation. Speed is computed
        as distance traveled over this window divided by time elapsed.
        Default is 5 frames (0.25s at 20 fps).

    Returns
    -------
    DataFrame with speed in pixels/s.
    """

    df = positions.merge(timestamps, on="frame_index", how="inner").sort_values("frame_index")

    # Calculate distance and time over the window
    # For each frame, look ahead by window_frames
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


def build_spike_place_dataframe(
    spike_index_path: Path,
    neural_timestamp_path: Path,
    behavior_position_path: Path,
    behavior_timestamp_path: Path,
    *,
    bodypart: str,
    speed_threshold: float = 50.0,
    use_neural_last_timestamp: bool = True,
    speed_window_frames: int = 5,
    behavior_fps: float,
) -> pd.DataFrame:
    """Match spikes to behavior positions for place-cell analysis.

    This:
    - Reads ``spike_index.csv`` (unit_id, frame, s)
    - Reads neural frame timestamps (``timestamp.csv``)
    - Reads behavior positions + timestamps
    - For each spike, finds the closest behavior frame in time
    - Filters out matches where timestamp difference exceeds threshold (0.5 / behavior_fps)
    - Filters out samples where running speed is below ``speed_threshold``

    Parameters
    ----------
    behavior_fps:
        Frames per second for behavior data. Required. Used to set timestamp
        difference threshold (0.5 / behavior_fps) for matching spikes to behavior frames.

    Returns
    -------
    DataFrame with at least:
      - unit_id
      - frame (neural)
      - s
      - neural_time
      - beh_frame_index
      - beh_time
      - x, y
      - speed
    """

    spike_df = pd.read_csv(spike_index_path)

    neural_ts = pd.read_csv(neural_timestamp_path)
    ts_col = "timestamp_last" if use_neural_last_timestamp else "timestamp_first"
    neural_ts = neural_ts.rename(columns={"frame": "frame", ts_col: "neural_time"})[
        ["frame", "neural_time"]
    ]

    beh_pos = _load_behavior_xy(behavior_position_path, bodypart=bodypart)
    beh_ts = pd.read_csv(behavior_timestamp_path)  # frame_index, unix_time

    beh = compute_behavior_speed(
        positions=beh_pos,
        timestamps=beh_ts,
        window_frames=speed_window_frames,
    )

    # Rename columns for clarity
    beh = beh.rename(
        columns={
            "frame_index": "beh_frame_index",
            "unix_time": "beh_time",
        }
    )

    # Merge spikes with neural timestamps
    spikes = spike_df.merge(neural_ts, on="frame", how="left")

    # For each spike, find nearest behavior frame in time
    beh_times = beh[["beh_frame_index", "beh_time", "x", "y", "speed"]]
    beh_times = beh_times.sort_values("beh_time").reset_index(drop=True)

    spike_times = spikes["neural_time"].to_numpy()
    beh_time_arr = beh_times["beh_time"].to_numpy()

    # Timestamp difference threshold: half the sampling time
    time_threshold = 0.5 / behavior_fps

    # Find nearest behavior frame for each spike
    idx = np.searchsorted(beh_time_arr, spike_times, side="left")
    idx_clipped = np.clip(idx, 0, len(beh_time_arr) - 1)

    # Check both left and right neighbors to find the closest
    idx_left = idx_clipped
    idx_right = np.clip(idx_clipped + 1, 0, len(beh_time_arr) - 1)

    time_diff_left = np.abs(spike_times - beh_time_arr[idx_left])
    time_diff_right = np.abs(spike_times - beh_time_arr[idx_right])

    # Choose the closer neighbor
    use_right = time_diff_right < time_diff_left
    idx_final = np.where(use_right, idx_right, idx_left)
    time_diff_final = np.where(use_right, time_diff_right, time_diff_left)

    beh_matched = beh_times.iloc[idx_final].reset_index(drop=True)
    out = pd.concat([spikes.reset_index(drop=True), beh_matched], axis=1)

    # Filter by timestamp difference threshold
    out = out[time_diff_final <= time_threshold].reset_index(drop=True)

    # Apply speed threshold
    out = out[out["speed"] >= float(speed_threshold)].reset_index(drop=True)
    return out
