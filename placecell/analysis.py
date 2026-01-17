"""Analysis functions for place cells."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter


def gaussian_filter_normalized(
    data: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Apply Gaussian smoothing with adaptive normalization at boundaries.

    Uses zero-padding and normalizes by the kernel weight sum so that
    edge bins are not penalized. This is the standard approach for
    place cell rate map smoothing.

    Parameters
    ----------
    data:
        Input 2D array to smooth.
    sigma:
        Gaussian smoothing sigma in bins.

    Returns
    -------
    np.ndarray
        Smoothed array with normalized edges.
    """
    if sigma <= 0:
        return data.copy()

    # Smooth data with zero padding
    smoothed = gaussian_filter(data, sigma=sigma, mode="constant", cval=0)
    # Smooth a mask of ones to get normalization weights
    norm = gaussian_filter(np.ones_like(data), sigma=sigma, mode="constant", cval=0)
    # Avoid division by zero
    norm[norm == 0] = 1
    return smoothed / norm


def compute_occupancy_map(
    trajectory_df: pd.DataFrame,
    bins: int,
    behavior_fps: float,
    occupancy_sigma: float = 1.0,
    min_occupancy: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute occupancy map from speed-filtered trajectory.

    Parameters
    ----------
    trajectory_df:
        Speed-filtered trajectory with x, y columns.
    bins:
        Number of spatial bins.
    behavior_fps:
        Behavior sampling rate.
    occupancy_sigma:
        Gaussian smoothing sigma for occupancy map.
    min_occupancy:
        Minimum occupancy time in seconds.

    Returns
    -------
    tuple
        (occupancy_time, valid_mask, x_edges, y_edges)
    """
    x_edges = np.linspace(trajectory_df["x"].min(), trajectory_df["x"].max(), bins + 1)
    y_edges = np.linspace(trajectory_df["y"].min(), trajectory_df["y"].max(), bins + 1)
    time_per_frame = 1.0 / behavior_fps

    occupancy_counts, _, _ = np.histogram2d(
        trajectory_df["x"], trajectory_df["y"], bins=[x_edges, y_edges]
    )
    occupancy_time = occupancy_counts * time_per_frame

    if occupancy_sigma > 0:
        occupancy_time = gaussian_filter_normalized(occupancy_time, sigma=occupancy_sigma)

    valid_mask = occupancy_time >= min_occupancy

    return occupancy_time, valid_mask, x_edges, y_edges


def compute_rate_map(
    unit_events: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    activity_sigma: float = 1.0,
) -> np.ndarray:
    """Compute smoothed and normalized rate map for a unit.

    Parameters
    ----------
    unit_events:
        DataFrame with x, y, s columns for a single unit.
    occupancy_time:
        Occupancy time map.
    valid_mask:
        Valid occupancy mask.
    x_edges, y_edges:
        Spatial bin edges.
    activity_sigma:
        Gaussian smoothing sigma for rate map.

    Returns
    -------
    np.ndarray
        Smoothed rate map normalized to 0-1 range.
    """
    if unit_events.empty:
        rate_map = np.full_like(occupancy_time, np.nan)
        return rate_map

    event_weights, _, _ = np.histogram2d(
        unit_events["x"],
        unit_events["y"],
        bins=[x_edges, y_edges],
        weights=unit_events["s"],
    )
    rate_map = np.zeros_like(occupancy_time)
    rate_map[valid_mask] = event_weights[valid_mask] / occupancy_time[valid_mask]
    rate_map_smooth = gaussian_filter_normalized(rate_map, sigma=activity_sigma)

    # Normalize to 0-1 range
    valid_rate_values = rate_map_smooth[valid_mask]
    if len(valid_rate_values) > 0 and np.nanmax(valid_rate_values) > 0:
        rate_map_smooth[valid_mask] = (
            rate_map_smooth[valid_mask] - np.nanmin(valid_rate_values)
        ) / (np.nanmax(valid_rate_values) - np.nanmin(valid_rate_values))
    rate_map_smooth[~valid_mask] = np.nan

    return rate_map_smooth


def compute_spatial_information(
    unit_events: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    n_shuffles: int = 100,
    random_seed: int | None = None,
) -> tuple[float, float, np.ndarray]:
    """Compute spatial information and significance via shuffling.

    Parameters
    ----------
    unit_events:
        DataFrame with x, y, s, beh_frame_index columns for a single unit.
    trajectory_df:
        Speed-filtered trajectory with beh_frame_index column.
    occupancy_time:
        Occupancy time map.
    valid_mask:
        Valid occupancy mask.
    x_edges, y_edges:
        Spatial bin edges.
    n_shuffles:
        Number of shuffles for significance test.
    random_seed:
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (spatial_info, p_value, shuffled_sis)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Compute rate map (unsmoothed for SI calculation)
    if unit_events.empty:
        return 0.0, 1.0, np.zeros(n_shuffles)

    event_weights, _, _ = np.histogram2d(
        unit_events["x"],
        unit_events["y"],
        bins=[x_edges, y_edges],
        weights=unit_events["s"],
    )
    rate_map = np.zeros_like(occupancy_time)
    rate_map[valid_mask] = event_weights[valid_mask] / occupancy_time[valid_mask]

    total_time = np.sum(occupancy_time[valid_mask])
    total_events = np.sum(event_weights[valid_mask])

    if total_time <= 0 or total_events <= 0:
        return 0.0, 1.0, np.zeros(n_shuffles)

    overall_lambda = total_events / total_time
    P_i = np.zeros_like(occupancy_time)
    P_i[valid_mask] = occupancy_time[valid_mask] / total_time

    valid_si = (rate_map > 0) & valid_mask
    if np.any(valid_si):
        si_term = P_i[valid_si] * rate_map[valid_si] * np.log2(rate_map[valid_si] / overall_lambda)
        actual_si = float(np.sum(si_term))
    else:
        actual_si = 0.0

    # Shuffling test
    traj_frames = trajectory_df["beh_frame_index"].values
    u_grouped = unit_events.groupby("beh_frame_index")["s"].sum()
    aligned_events = u_grouped.reindex(traj_frames, fill_value=0).values

    shuffled_sis = []
    for _ in range(n_shuffles):
        shift = np.random.randint(len(aligned_events))
        s_shuffled = np.roll(aligned_events, shift)

        event_w_shuf, _, _ = np.histogram2d(
            trajectory_df["x"],
            trajectory_df["y"],
            bins=[x_edges, y_edges],
            weights=s_shuffled,
        )
        rate_shuf = np.zeros_like(occupancy_time)
        rate_shuf[valid_mask] = event_w_shuf[valid_mask] / occupancy_time[valid_mask]

        valid_s = (rate_shuf > 0) & valid_mask
        if np.any(valid_s):
            si_shuf = np.sum(
                P_i[valid_s] * rate_shuf[valid_s] * np.log2(rate_shuf[valid_s] / overall_lambda)
            )
        else:
            si_shuf = 0.0
        shuffled_sis.append(si_shuf)

    shuffled_sis = np.array(shuffled_sis)
    p_val = np.sum(shuffled_sis >= actual_si) / n_shuffles

    return actual_si, p_val, shuffled_sis


def compute_stability_score(
    unit_events: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    activity_sigma: float = 1.0,
    behavior_fps: float = 20.0,
    min_occupancy: float = 0.1,
    split_method: str = "half",
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute stability score by comparing rate maps from split data.

    Splits the recording into two halves based on behavior frame index,
    computes rate maps for each half, and returns the Pearson correlation
    between them (using valid bins in both halves).

    Parameters
    ----------
    unit_events:
        DataFrame with x, y, s, beh_frame_index columns for a single unit.
    trajectory_df:
        Speed-filtered trajectory with beh_frame_index column.
    occupancy_time:
        Occupancy time map (full session, for reference).
    valid_mask:
        Valid occupancy mask (full session).
    x_edges, y_edges:
        Spatial bin edges.
    activity_sigma:
        Gaussian smoothing sigma for rate maps.
    behavior_fps:
        Behavior sampling rate.
    min_occupancy:
        Minimum occupancy time in seconds for a bin to be valid.
    split_method:
        Splitting method. Currently only "half" is supported.
        Future options: "odd_even", "thirds", etc.

    Returns
    -------
    tuple
        (correlation, fisher_z, rate_map_first, rate_map_second)
        correlation: Pearson correlation between the two rate maps
        fisher_z: Fisher z-transformed correlation
        rate_map_first: Rate map from first half
        rate_map_second: Rate map from second half
    """
    if unit_events.empty or trajectory_df.empty:
        nan_map = np.full_like(occupancy_time, np.nan)
        return np.nan, np.nan, nan_map, nan_map

    # Split trajectory by frame index
    all_frames = trajectory_df["beh_frame_index"].values
    mid_frame = np.median(all_frames)

    traj_first = trajectory_df[trajectory_df["beh_frame_index"] <= mid_frame]
    traj_second = trajectory_df[trajectory_df["beh_frame_index"] > mid_frame]

    events_first = unit_events[unit_events["beh_frame_index"] <= mid_frame]
    events_second = unit_events[unit_events["beh_frame_index"] > mid_frame]

    # Compute occupancy maps for each half
    time_per_frame = 1.0 / behavior_fps

    def compute_half_occupancy(traj_half: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Compute occupancy and valid mask for a trajectory half."""
        if traj_half.empty:
            occ = np.zeros_like(occupancy_time)
            mask = np.zeros_like(valid_mask, dtype=bool)
            return occ, mask
        counts, _, _ = np.histogram2d(
            traj_half["x"], traj_half["y"], bins=[x_edges, y_edges]
        )
        occ = counts * time_per_frame
        occ_smooth = gaussian_filter_normalized(occ, sigma=activity_sigma)
        mask = occ_smooth >= min_occupancy
        return occ_smooth, mask

    occ_first, valid_first = compute_half_occupancy(traj_first)
    occ_second, valid_second = compute_half_occupancy(traj_second)

    # Compute rate maps for each half (unnormalized for fair comparison)
    def compute_half_rate_map(
        events: pd.DataFrame, occ: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Compute rate map for a half."""
        rate_map = np.full_like(occ, np.nan)
        if events.empty or not np.any(mask):
            return rate_map
        event_weights, _, _ = np.histogram2d(
            events["x"], events["y"],
            bins=[x_edges, y_edges],
            weights=events["s"],
        )
        rate_map[mask] = event_weights[mask] / occ[mask]
        rate_map_smooth = gaussian_filter_normalized(rate_map, sigma=activity_sigma)
        rate_map_smooth[~mask] = np.nan
        return rate_map_smooth

    rate_map_first = compute_half_rate_map(events_first, occ_first, valid_first)
    rate_map_second = compute_half_rate_map(events_second, occ_second, valid_second)

    # Compute correlation only on bins valid in both halves
    both_valid = valid_first & valid_second
    if not np.any(both_valid):
        return np.nan, np.nan, rate_map_first, rate_map_second

    vals_first = rate_map_first[both_valid]
    vals_second = rate_map_second[both_valid]

    # Remove any remaining NaN values
    finite_mask = np.isfinite(vals_first) & np.isfinite(vals_second)
    if np.sum(finite_mask) < 3:  # Need at least 3 points for correlation
        return np.nan, np.nan, rate_map_first, rate_map_second

    vals_first = vals_first[finite_mask]
    vals_second = vals_second[finite_mask]

    # Pearson correlation
    corr = np.corrcoef(vals_first, vals_second)[0, 1]

    # Fisher z-transform: z = 0.5 * ln((1+r)/(1-r)) = arctanh(r)
    # Clip to avoid infinity at r=1 or r=-1
    corr_clipped = np.clip(corr, -0.9999, 0.9999)
    fisher_z = np.arctanh(corr_clipped)

    return corr, fisher_z, rate_map_first, rate_map_second


def compute_unit_analysis(
    unit_id: int,
    df_filtered: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    activity_sigma: float = 1.0,
    event_threshold_sigma: float = 2.0,
    n_shuffles: int = 100,
    random_seed: int | None = None,
    behavior_fps: float = 20.0,
    min_occupancy: float = 0.1,
    stability_threshold: float = 0.5,
) -> dict:
    """Compute rate map, spatial information, stability, and thresholded events for a unit.

    Parameters
    ----------
    unit_id:
        Unit identifier.
    df_filtered:
        Speed-filtered event data with columns unit_id, x, y, s, beh_frame_index.
    trajectory_df:
        Speed-filtered trajectory with beh_frame_index column.
    occupancy_time:
        Occupancy time map.
    valid_mask:
        Valid occupancy mask.
    x_edges, y_edges:
        Spatial bin edges.
    activity_sigma:
        Gaussian smoothing sigma for rate map.
    event_threshold_sigma:
        Sigma multiplier for event amplitude threshold.
    n_shuffles:
        Number of shuffles for significance test.
    random_seed:
        Random seed for reproducibility.
    behavior_fps:
        Behavior sampling rate for stability computation.
    min_occupancy:
        Minimum occupancy time for stability computation.
    stability_threshold:
        Correlation threshold for stability test pass/fail.

    Returns
    -------
    dict
        Analysis results with keys: rate_map, si, p_val, shuffled_sis,
        events_above_threshold, vis_threshold, stability_corr, stability_z,
        rate_map_first, rate_map_second.
    """
    unit_data = (
        df_filtered[df_filtered["unit_id"] == unit_id] if not df_filtered.empty else pd.DataFrame()
    )

    # Rate map
    rate_map = compute_rate_map(
        unit_data, occupancy_time, valid_mask, x_edges, y_edges, activity_sigma
    )

    # Spatial information
    si, p_val, shuffled_sis = compute_spatial_information(
        unit_data,
        trajectory_df,
        occupancy_time,
        valid_mask,
        x_edges,
        y_edges,
        n_shuffles,
        random_seed=random_seed,
    )

    # Event threshold for visualization
    if not unit_data.empty and len(unit_data) > 1:
        vis_threshold = unit_data["s"].mean() + event_threshold_sigma * unit_data["s"].std()
        events_above = unit_data[unit_data["s"] > vis_threshold]
    else:
        vis_threshold = 0.0
        events_above = pd.DataFrame()

    # Stability test
    stability_corr, stability_z, rate_map_first, rate_map_second = compute_stability_score(
        unit_data,
        trajectory_df,
        occupancy_time,
        valid_mask,
        x_edges,
        y_edges,
        activity_sigma=activity_sigma,
        behavior_fps=behavior_fps,
        min_occupancy=min_occupancy,
    )

    return {
        "rate_map": rate_map,
        "si": si,
        "p_val": p_val,
        "shuffled_sis": shuffled_sis,
        "events_above_threshold": events_above,
        "vis_threshold": vis_threshold,
        "unit_data": unit_data,
        "stability_corr": stability_corr,
        "stability_z": stability_z,
        "rate_map_first": rate_map_first,
        "rate_map_second": rate_map_second,
    }


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
    event_index_path: Path,
    neural_timestamp_path: Path,
    behavior_position_path: Path,
    behavior_timestamp_path: Path,
    bodypart: str,
    behavior_fps: float,
    speed_threshold: float = 50.0,
    speed_window_frames: int = 5,
    use_neural_last_timestamp: bool = True,
) -> pd.DataFrame:
    """Match neural events to behavior positions for place-cell analysis.

    This function:
    - Reads event index CSV (columns: unit_id, frame, s)
    - Reads neural frame timestamps CSV (columns: frame, timestamp_first, timestamp_last)
    - Reads behavior position CSV (DeepLabCut format with multi-index header)
    - Reads behavior timestamp CSV (columns: frame_index, unix_time)
    - For each event, finds the closest behavior frame in time
    - Filters out matches where timestamp difference exceeds threshold (0.5 / behavior_fps)
    - Filters out samples where running speed is below `speed_threshold`

    Parameters
    ----------
    event_index_path:
        Path to event index CSV file (columns: unit_id, frame, s).
    neural_timestamp_path:
        Path to neural timestamp CSV file (columns: frame, timestamp_first, timestamp_last).
    behavior_position_path:
        Path to behavior position CSV file (DeepLabCut format with multi-index header).
    behavior_timestamp_path:
        Path to behavior timestamp CSV file (columns: frame_index, unix_time).
    bodypart:
        Body part name to use for position tracking (e.g. 'LED').
    behavior_fps:
        Frames per second for behavior data. Required. Used to set timestamp
        difference threshold (0.5 / behavior_fps) for matching events to behavior frames.
    speed_threshold:
        Minimum running speed to keep events (pixels/s).
    speed_window_frames:
        Number of frames to use for speed calculation window.
    use_neural_last_timestamp:
        Whether to use the last neural timestamp for each frame.

    Returns
    -------
    DataFrame with columns:
      - unit_id: Unit identifier
      - frame: Neural frame number
      - s: Event amplitude
      - neural_time: Neural timestamp (seconds)
      - beh_frame_index: Behavior frame index
      - beh_time: Behavior timestamp (seconds, unix time)
      - x: X position (pixels)
      - y: Y position (pixels)
      - speed: Running speed (pixels/s)
    """

    event_df = pd.read_csv(event_index_path)

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

    beh = beh.rename(
        columns={
            "frame_index": "beh_frame_index",
            "unix_time": "beh_time",
        }
    )

    # Merge events with neural timestamps
    events = event_df.merge(neural_ts, on="frame", how="left")

    # For each event, find nearest behavior frame in time
    beh_times = beh[["beh_frame_index", "beh_time", "x", "y", "speed"]]
    beh_times = beh_times.sort_values("beh_time").reset_index(drop=True)

    event_times = events["neural_time"].to_numpy()
    beh_time_arr = beh_times["beh_time"].to_numpy()

    # Timestamp difference threshold: half the sampling time
    time_threshold = 0.5 / behavior_fps

    # Find nearest behavior frame for each event
    idx = np.searchsorted(beh_time_arr, event_times, side="left")
    idx_clipped = np.clip(idx, 0, len(beh_time_arr) - 1)

    # Check both left and right neighbors to find the closest
    idx_left = idx_clipped
    idx_right = np.clip(idx_clipped + 1, 0, len(beh_time_arr) - 1)

    time_diff_left = np.abs(event_times - beh_time_arr[idx_left])
    time_diff_right = np.abs(event_times - beh_time_arr[idx_right])

    # Choose the closer neighbor
    use_right = time_diff_right < time_diff_left
    idx_final = np.where(use_right, idx_right, idx_left)
    time_diff_final = np.where(use_right, time_diff_right, time_diff_left)

    beh_matched = beh_times.iloc[idx_final].reset_index(drop=True)
    out = pd.concat([events.reset_index(drop=True), beh_matched], axis=1)

    # Filter by timestamp difference threshold
    out = out[time_diff_final <= time_threshold].reset_index(drop=True)

    # Apply speed threshold
    out = out[out["speed"] >= float(speed_threshold)].reset_index(drop=True)
    return out


def load_traces(
    neural_path: Path,
    trace_name: str = "C",
) -> xr.DataArray:
    """Load traces from a Minian-style zarr store as a DataArray.

    Parameters
    ----------
    neural_path:
        Directory containing ``<trace_name>.zarr``.
    trace_name:
        Base name of the zarr group (e.g. ``"C"`` or ``"C_lp"``).
        Also used as the variable name if the zarr contains a Dataset.

    Returns
    -------
    xr.DataArray
        DataArray with dimensions ('unit_id', 'frame').
    """
    zarr_path = neural_path / f"{trace_name}.zarr"
    ds_or_da = xr.open_zarr(zarr_path, consolidated=False)

    if isinstance(ds_or_da, xr.Dataset):
        if trace_name not in ds_or_da:
            raise KeyError(
                f"Variable {trace_name!r} not found in dataset; "
                f"available: {list(ds_or_da.data_vars)}"
            )
        C = ds_or_da[trace_name]
    else:
        C = ds_or_da

    if "unit_id" not in C.dims or "frame" not in C.dims:
        raise ValueError(f"Expected dims ('unit_id','frame'), got {C.dims}")

    # Validate coordinates are unique
    unit_ids = C.coords["unit_id"].values
    if len(unit_ids) != len(np.unique(unit_ids)):
        raise ValueError(
            f"unit_id coordinates must be unique, but found {len(np.unique(unit_ids))} "
            f"unique values for {len(unit_ids)} units. "
            f"The zarr file has corrupted coordinates."
        )

    return C
