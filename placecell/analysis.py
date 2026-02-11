"""Spatial analysis functions for place cells."""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, label


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
        rate_map_smooth[valid_mask] = rate_map_smooth[valid_mask] / np.nanmax(valid_rate_values)

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
    min_shift_seconds: float = 0.0,
    behavior_fps: float = 20.0,
    si_weight_mode: str = "amplitude",
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
    min_shift_seconds:
        Minimum circular shift in seconds. Shifts smaller than this are
        re-drawn to ensure the temporal-spatial association is broken.
        Default 0.0 (no minimum).
    behavior_fps:
        Behavior sampling rate, used to convert min_shift_seconds to frames.
    si_weight_mode:
        ``"amplitude"`` weights events by their ``s`` value;
        ``"binary"`` counts each event as 1 regardless of amplitude,
        which is more robust to bursty firing patterns.

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

    # Choose weights based on mode
    if si_weight_mode == "binary":
        weights = np.ones(len(unit_events))
    else:
        weights = unit_events["s"].values

    event_weights, _, _ = np.histogram2d(
        unit_events["x"],
        unit_events["y"],
        bins=[x_edges, y_edges],
        weights=weights,
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
    if si_weight_mode == "binary":
        u_grouped = unit_events.groupby("beh_frame_index").size()
    else:
        u_grouped = unit_events.groupby("beh_frame_index")["s"].sum()
    aligned_events = u_grouped.reindex(traj_frames, fill_value=0).values.astype(float)

    n_frames = len(aligned_events)
    min_shift_frames = int(min_shift_seconds * behavior_fps)
    # Clamp so valid range exists (shift between min_shift and n_frames - min_shift)
    min_shift_frames = min(min_shift_frames, n_frames // 2)

    shuffled_sis = []
    for _ in range(n_shuffles):
        if min_shift_frames > 0:
            shift = np.random.randint(min_shift_frames, n_frames - min_shift_frames)
        else:
            shift = np.random.randint(n_frames)
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


def compute_shuffled_rate_percentile(
    unit_events: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    activity_sigma: float = 1.0,
    n_shuffles: int = 100,
    min_shift_seconds: float = 0.0,
    behavior_fps: float = 20.0,
    si_weight_mode: str = "amplitude",
    random_seed: int | None = None,
    percentile: float = 95.0,
) -> np.ndarray:
    """Compute per-bin percentile of shuffled smoothed rate maps.

    Used for the seed detection step of the Guo et al. 2023 place field
    algorithm.  For each shuffle iteration, circularly shifts event times
    relative to the trajectory, computes a smoothed and normalized (0-1)
    rate map, then returns the requested percentile across shuffles at
    each spatial bin.

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
    activity_sigma:
        Gaussian smoothing sigma for rate maps.
    n_shuffles:
        Number of shuffle iterations.
    min_shift_seconds:
        Minimum circular shift in seconds.
    behavior_fps:
        Behavior sampling rate.
    si_weight_mode:
        ``"amplitude"`` or ``"binary"``.
    random_seed:
        Random seed for reproducibility.
    percentile:
        Percentile to compute (default 95).

    Returns
    -------
    np.ndarray
        Per-bin percentile of the shuffled normalized rate maps.
    """
    result = np.zeros_like(occupancy_time)
    if unit_events.empty:
        return result

    # Use a local RNG to avoid disturbing the global state.
    # Offset seed by a large prime so shuffles differ from SI shuffles.
    rng = np.random.RandomState(random_seed + 104729 if random_seed is not None else None)

    traj_frames = trajectory_df["beh_frame_index"].values
    if si_weight_mode == "binary":
        u_grouped = unit_events.groupby("beh_frame_index").size()
    else:
        u_grouped = unit_events.groupby("beh_frame_index")["s"].sum()
    aligned_events = u_grouped.reindex(traj_frames, fill_value=0).values.astype(float)

    n_frames = len(aligned_events)
    min_shift_frames = int(min_shift_seconds * behavior_fps)
    min_shift_frames = min(min_shift_frames, n_frames // 2)

    traj_x = trajectory_df["x"].values
    traj_y = trajectory_df["y"].values

    shuffled_rates = np.zeros((n_shuffles, *occupancy_time.shape))

    for i in range(n_shuffles):
        if min_shift_frames > 0:
            shift = rng.randint(min_shift_frames, n_frames - min_shift_frames)
        else:
            shift = rng.randint(n_frames)
        s_shuffled = np.roll(aligned_events, shift)

        event_w_shuf, _, _ = np.histogram2d(
            traj_x,
            traj_y,
            bins=[x_edges, y_edges],
            weights=s_shuffled,
        )
        rate_shuf = np.zeros_like(occupancy_time)
        rate_shuf[valid_mask] = event_w_shuf[valid_mask] / occupancy_time[valid_mask]
        rate_shuf_smooth = gaussian_filter_normalized(rate_shuf, sigma=activity_sigma)

        # Normalize 0-1 (same as actual rate map)
        peak = np.nanmax(rate_shuf_smooth[valid_mask]) if np.any(valid_mask) else 0
        if peak > 0:
            rate_shuf_smooth[valid_mask] /= peak
        rate_shuf_smooth[~valid_mask] = 0

        shuffled_rates[i] = rate_shuf_smooth

    return np.percentile(shuffled_rates, percentile, axis=0)


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
    occupancy_sigma: float = 0.0,
    split_method: str = "half",
    n_shuffles: int = 0,
    random_seed: int | None = None,
    min_shift_seconds: float = 0.0,
    si_weight_mode: str = "amplitude",
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Compute stability score by comparing rate maps from split data.

    Splits the recording into two halves based on behavior frame index,
    computes rate maps for each half, and returns the Pearson correlation
    between them (using valid bins in both halves).

    Optionally runs a shuffle significance test (Shuman et al. 2020):
    circularly shifts events and computes the split-half correlation for
    each shuffle to build a null distribution.

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
    occupancy_sigma:
        Gaussian smoothing sigma for occupancy maps (default 0.0 = no smoothing).
    split_method:
        Splitting method. Currently only "half" is supported.
    n_shuffles:
        Number of shuffles for stability significance test.
        0 means no shuffle test (return NaN for p-value).
    random_seed:
        Random seed for reproducibility.
    min_shift_seconds:
        Minimum circular shift in seconds for shuffle test.
    si_weight_mode:
        ``"amplitude"`` weights events by their ``s`` value;
        ``"binary"`` counts each event as 1.

    Returns
    -------
    tuple
        (correlation, fisher_z, stability_p_val, rate_map_first, rate_map_second)
        correlation: Pearson correlation between the two rate maps
        fisher_z: Fisher z-transformed correlation
        stability_p_val: Shuffle-based p-value (NaN if n_shuffles=0)
        rate_map_first: Rate map from first half
        rate_map_second: Rate map from second half
    """
    if unit_events.empty or trajectory_df.empty:
        nan_map = np.full_like(occupancy_time, np.nan)
        return np.nan, np.nan, np.nan, nan_map, nan_map

    # Binary mode: use event counts instead of amplitudes
    if si_weight_mode == "binary":
        unit_events = unit_events.copy()
        unit_events["s"] = 1.0

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
        counts, _, _ = np.histogram2d(traj_half["x"], traj_half["y"], bins=[x_edges, y_edges])
        occ = counts * time_per_frame
        occ_smooth = gaussian_filter_normalized(occ, sigma=occupancy_sigma)
        mask = occ_smooth >= min_occupancy
        return occ_smooth, mask

    occ_first, valid_first = compute_half_occupancy(traj_first)
    occ_second, valid_second = compute_half_occupancy(traj_second)

    # Compute rate maps for each half (unnormalized for fair comparison)
    def compute_half_rate_map(
        events: pd.DataFrame, occ: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Compute rate map for a half."""
        if events.empty or not np.any(mask):
            return np.full_like(occ, np.nan)

        event_weights, _, _ = np.histogram2d(
            events["x"],
            events["y"],
            bins=[x_edges, y_edges],
            weights=events["s"],
        )

        # Compute rate map - use 0 for invalid bins to prevent NaN propagation
        rate_map = np.zeros_like(occ)
        rate_map[mask] = event_weights[mask] / occ[mask]

        # Smooth the rate map (zeros in invalid bins won't propagate NaN)
        rate_map_smooth = gaussian_filter_normalized(rate_map, sigma=activity_sigma)

        # Set invalid bins to NaN after smoothing
        rate_map_smooth[~mask] = np.nan
        return rate_map_smooth

    rate_map_first = compute_half_rate_map(events_first, occ_first, valid_first)
    rate_map_second = compute_half_rate_map(events_second, occ_second, valid_second)

    # Compute correlation only on bins valid in both halves
    both_valid = valid_first & valid_second
    if not np.any(both_valid):
        return np.nan, np.nan, np.nan, rate_map_first, rate_map_second

    vals_first = rate_map_first[both_valid]
    vals_second = rate_map_second[both_valid]

    # Remove any remaining NaN values
    finite_mask = np.isfinite(vals_first) & np.isfinite(vals_second)
    if np.sum(finite_mask) < 3:  # Need at least 3 points for correlation
        return np.nan, np.nan, np.nan, rate_map_first, rate_map_second

    vals_first = vals_first[finite_mask]
    vals_second = vals_second[finite_mask]

    # Pearson correlation
    corr = np.corrcoef(vals_first, vals_second)[0, 1]

    # Fisher z-transform: z = 0.5 * ln((1+r)/(1-r)) = arctanh(r)
    # Clip to avoid infinity at r=1 or r=-1
    corr_clipped = np.clip(corr, -0.9999, 0.9999)
    fisher_z = np.arctanh(corr_clipped)

    # Normalize split rate maps to 0-1 for display (doesn't affect correlation)
    for rm, mask in [(rate_map_first, valid_first), (rate_map_second, valid_second)]:
        valid_vals = rm[mask]
        if len(valid_vals) > 0 and np.nanmax(valid_vals) > 0:
            rm[mask] = rm[mask] / np.nanmax(valid_vals)

    # Shuffle-based stability significance test
    if n_shuffles > 0:
        if random_seed is not None:
            np.random.seed(random_seed)

        traj_frames = trajectory_df["beh_frame_index"].values
        u_grouped = unit_events.groupby("beh_frame_index")["s"].sum()
        aligned_events = u_grouped.reindex(traj_frames, fill_value=0).values.astype(float)

        all_frames = trajectory_df["beh_frame_index"].values
        mid_frame = np.median(all_frames)
        first_half = all_frames <= mid_frame
        second_half = all_frames > mid_frame

        traj_x = trajectory_df["x"].values
        traj_y = trajectory_df["y"].values

        n_frames = len(aligned_events)
        min_shift_frames = int(min_shift_seconds * behavior_fps)
        min_shift_frames = min(min_shift_frames, n_frames // 2)

        shuffled_corrs = np.empty(n_shuffles)
        for i in range(n_shuffles):
            if min_shift_frames > 0:
                shift = np.random.randint(min_shift_frames, n_frames - min_shift_frames)
            else:
                shift = np.random.randint(n_frames)
            shifted = np.roll(aligned_events, shift)

            ew1, _, _ = np.histogram2d(
                traj_x[first_half], traj_y[first_half],
                bins=[x_edges, y_edges], weights=shifted[first_half],
            )
            rm1 = np.zeros_like(occ_first)
            rm1[valid_first] = ew1[valid_first] / occ_first[valid_first]
            rm1 = gaussian_filter_normalized(rm1, sigma=activity_sigma)

            ew2, _, _ = np.histogram2d(
                traj_x[second_half], traj_y[second_half],
                bins=[x_edges, y_edges], weights=shifted[second_half],
            )
            rm2 = np.zeros_like(occ_second)
            rm2[valid_second] = ew2[valid_second] / occ_second[valid_second]
            rm2 = gaussian_filter_normalized(rm2, sigma=activity_sigma)

            bv = valid_first & valid_second
            if not np.any(bv):
                shuffled_corrs[i] = 0.0
                continue
            v1, v2 = rm1[bv], rm2[bv]
            fm = np.isfinite(v1) & np.isfinite(v2)
            if np.sum(fm) < 3:
                shuffled_corrs[i] = 0.0
                continue
            shuffled_corrs[i] = np.corrcoef(v1[fm], v2[fm])[0, 1]

        stability_p_val = float(np.sum(shuffled_corrs >= corr) / n_shuffles)
    else:
        stability_p_val = np.nan

    return corr, fisher_z, stability_p_val, rate_map_first, rate_map_second


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
    occupancy_sigma: float = 0.0,
    stability_threshold: float = 0.5,
    stability_method: str = "shuffle",
    min_shift_seconds: float = 0.0,
    si_weight_mode: str = "amplitude",
    place_field_seed_percentile: float | None = 95.0,
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
    occupancy_sigma:
        Gaussian smoothing sigma for occupancy maps in stability computation.
    stability_threshold:
        Correlation threshold for stability test pass/fail.
    stability_method:
        ``"shuffle"`` runs circular-shift significance test for stability;
        ``"threshold"`` uses a fixed correlation threshold only.
    min_shift_seconds:
        Minimum circular shift in seconds for shuffle significance test.
    si_weight_mode:
        Weight mode for SI: ``"amplitude"`` or ``"binary"``.
    place_field_seed_percentile:
        Percentile of shuffled rate maps for seed detection (Guo et al.
        2023).  ``None`` skips the computation and uses the simplified
        threshold-only algorithm (faster).

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

    # Rate map (smoothed + normalized for display, raw for place field detection)
    rate_map = compute_rate_map(
        unit_data, occupancy_time, valid_mask, x_edges, y_edges, activity_sigma
    )
    rate_map_raw = compute_raw_rate_map(
        unit_data, occupancy_time, valid_mask, x_edges, y_edges
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
        min_shift_seconds=min_shift_seconds,
        behavior_fps=behavior_fps,
        si_weight_mode=si_weight_mode,
    )

    # Event threshold for visualization
    if not unit_data.empty and len(unit_data) > 1:
        vis_threshold = unit_data["s"].mean() + event_threshold_sigma * unit_data["s"].std()
        events_above = unit_data[unit_data["s"] > vis_threshold]
    else:
        vis_threshold = 0.0
        events_above = pd.DataFrame()

    # Shuffled rate percentile for place field seed detection (Guo et al. 2023)
    if place_field_seed_percentile is not None:
        shuffled_rate_p95 = compute_shuffled_rate_percentile(
            unit_data,
            trajectory_df,
            occupancy_time,
            valid_mask,
            x_edges,
            y_edges,
            activity_sigma=activity_sigma,
            n_shuffles=n_shuffles,
            min_shift_seconds=min_shift_seconds,
            behavior_fps=behavior_fps,
            si_weight_mode=si_weight_mode,
            random_seed=random_seed,
            percentile=place_field_seed_percentile,
        )
    else:
        shuffled_rate_p95 = None

    # Stability test
    stab_shuffles = n_shuffles if stability_method == "shuffle" else 0
    stability_corr, stability_z, stability_p_val, rate_map_first, rate_map_second = (
        compute_stability_score(
            unit_data,
            trajectory_df,
            occupancy_time,
            valid_mask,
            x_edges,
            y_edges,
            activity_sigma=activity_sigma,
            behavior_fps=behavior_fps,
            min_occupancy=min_occupancy,
            occupancy_sigma=occupancy_sigma,
            n_shuffles=stab_shuffles,
            random_seed=random_seed,
            min_shift_seconds=min_shift_seconds,
            si_weight_mode=si_weight_mode,
        )
    )

    return {
        "rate_map": rate_map,
        "rate_map_raw": rate_map_raw,
        "si": si,
        "p_val": p_val,
        "shuffled_sis": shuffled_sis,
        "shuffled_rate_p95": shuffled_rate_p95,
        "events_above_threshold": events_above,
        "vis_threshold": vis_threshold,
        "unit_data": unit_data,
        "stability_corr": stability_corr,
        "stability_z": stability_z,
        "stability_p_val": stability_p_val,
        "rate_map_first": rate_map_first,
        "rate_map_second": rate_map_second,
    }


def compute_raw_rate_map(
    unit_events: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> np.ndarray:
    """Compute unsmoothed binned spatial activity rate.

    This is the raw event-weight / occupancy-time rate without Gaussian
    smoothing or min-max normalization.  It is used for place field
    boundary detection where smoothing would artificially widen the field.

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

    Returns
    -------
    np.ndarray
        Unsmoothed rate map.  Invalid bins are set to NaN.
    """
    if unit_events.empty:
        return np.full_like(occupancy_time, np.nan)

    event_weights, _, _ = np.histogram2d(
        unit_events["x"],
        unit_events["y"],
        bins=[x_edges, y_edges],
        weights=unit_events["s"],
    )
    rate_map = np.full_like(occupancy_time, np.nan)
    rate_map[valid_mask] = event_weights[valid_mask] / occupancy_time[valid_mask]
    return rate_map


def compute_place_field_mask(
    rate_map: np.ndarray,
    threshold: float = 0.05,
    min_bins: int = 5,
    shuffled_rate_p95: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a binary mask of the place field from a rate map.

    Implements the place field detection algorithm from Guo et al. 2023
    (Science Advances, Supplementary Methods lines 1013-1020):

    1. **Seed detection**: find bins where the actual rate exceeds the
       95th percentile of the shuffled rate (``shuffled_rate_p95``).
       Only contiguous seed regions with >= ``min_bins`` bins are kept.
    2. **Extension**: from each seed region, extend to all contiguous
       bins whose rate >= ``threshold`` × (seed region's peak rate).

    If ``shuffled_rate_p95`` is not provided, falls back to the
    simplified threshold-only algorithm (threshold + connected-component
    size filter).

    Parameters
    ----------
    rate_map:
        Smoothed, normalized (0-1) rate map.  NaN bins are treated as
        outside the field.
    threshold:
        Fraction of peak rate for field extension (step 2).
    min_bins:
        Minimum number of contiguous bins for a seed region (step 1)
        or for a component in the simplified fallback.
    shuffled_rate_p95:
        Per-bin 95th percentile of shuffled normalized rate maps.
        When provided, enables the full seed-extension algorithm.

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates the place field.
    """
    mask = np.zeros_like(rate_map, dtype=bool)
    valid = np.isfinite(rate_map)
    if not np.any(valid):
        return mask
    peak = np.nanmax(rate_map)
    if peak <= 0:
        return mask

    if shuffled_rate_p95 is None:
        # Simplified fallback: threshold + connected-component size filter
        raw_mask = valid & (rate_map >= threshold * peak)
        if min_bins <= 1:
            return raw_mask
        labeled, n_components = label(raw_mask)
        for comp_id in range(1, n_components + 1):
            comp = labeled == comp_id
            if comp.sum() >= min_bins:
                mask |= comp
        return mask

    # --- Full Guo et al. 2023 algorithm ---

    # Step 1: Identify significant seed bins (rate > 95th percentile of shuffle)
    sig_bins = valid & (rate_map > shuffled_rate_p95)
    seed_labeled, n_seeds = label(sig_bins)

    # Keep only seed regions with >= min_bins contiguous bins
    seed_mask = np.zeros_like(rate_map, dtype=bool)
    for comp_id in range(1, n_seeds + 1):
        comp = seed_labeled == comp_id
        if comp.sum() >= min_bins:
            seed_mask |= comp

    if not np.any(seed_mask):
        return mask

    # Step 2: For each seed region, extend to contiguous bins
    # >= threshold × (seed's peak rate)
    seed_labeled_clean, n_seed_regions = label(seed_mask)

    for seed_id in range(1, n_seed_regions + 1):
        seed_region = seed_labeled_clean == seed_id
        field_peak = np.nanmax(rate_map[seed_region])
        if field_peak <= 0:
            continue
        extension_cutoff = threshold * field_peak

        # Find all candidate bins above the extension threshold
        candidate = valid & (rate_map >= extension_cutoff)
        candidate_labeled, _ = label(candidate)

        # Keep the candidate component(s) that overlap with this seed
        overlapping_ids = set(candidate_labeled[seed_region]) - {0}
        for cand_id in overlapping_ids:
            mask |= candidate_labeled == cand_id

    return mask


def compute_coverage_map(
    unit_results: dict,
    threshold: float = 0.05,
    min_bins: int = 5,
) -> np.ndarray:
    """Compute combined place field coverage across all units.

    For each unit, thresholds the smoothed rate map to define the place
    field, then sums all binary masks to get the number of overlapping
    fields at each spatial bin.

    Parameters
    ----------
    unit_results:
        Dictionary mapping unit_id to analysis results
        (must contain 'rate_map').
    threshold:
        Fraction of peak rate to define place field boundary.
    min_bins:
        Minimum contiguous bins for a connected component to count.

    Returns
    -------
    np.ndarray
        Integer array of place field overlap counts at each bin.
    """
    coverage = None
    for uid, result in unit_results.items():
        rm = result["rate_map"]
        if coverage is None:
            coverage = np.zeros_like(rm, dtype=int)
        field_mask = compute_place_field_mask(
            rm,
            threshold=threshold,
            min_bins=min_bins,
            shuffled_rate_p95=result.get("shuffled_rate_p95"),
        )
        coverage += field_mask.astype(int)
    if coverage is None:
        return np.zeros((1, 1), dtype=int)
    return coverage


def compute_coverage_curve(
    unit_results: dict,
    valid_mask: np.ndarray,
    threshold: float = 0.05,
    min_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative coverage curve: fraction of environment covered vs number of cells.

    Cells are added one at a time (largest field first). At each step,
    the fraction of valid bins covered by at least one place field is recorded.

    Parameters
    ----------
    unit_results:
        Dictionary mapping unit_id to analysis results
        (must contain 'rate_map').
    valid_mask:
        Boolean mask of valid spatial bins.
    threshold:
        Fraction of peak rate to define place field boundary.
    min_bins:
        Minimum contiguous bins for a connected component to count.

    Returns
    -------
    tuple
        (n_cells_array, coverage_fraction_array) where n_cells goes from 0 to N.
    """
    n_valid = int(np.sum(valid_mask))
    if n_valid == 0:
        return np.array([0]), np.array([0.0])

    # Collect per-unit field masks and sort by field size (largest first)
    masks = []
    for uid, result in unit_results.items():
        m = compute_place_field_mask(
            result["rate_map"],
            threshold=threshold,
            min_bins=min_bins,
            shuffled_rate_p95=result.get("shuffled_rate_p95"),
        )
        masks.append(m)
    masks.sort(key=lambda m: m.sum(), reverse=True)

    n_cells = np.arange(len(masks) + 1)
    fractions = np.zeros(len(masks) + 1)
    cumulative = np.zeros_like(valid_mask, dtype=bool)
    for i, m in enumerate(masks):
        cumulative |= m
        covered = np.sum(cumulative & valid_mask)
        fractions[i + 1] = covered / n_valid

    return n_cells, fractions
