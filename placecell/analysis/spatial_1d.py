"""1D spatial analysis functions for place cells in linear tracks / arms."""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from placecell.config import SpatialMap1DConfig


def gaussian_filter_normalized_1d(
    data: np.ndarray,
    sigma: float,
    segment_bins: list[int] | None = None,
) -> np.ndarray:
    """Apply 1D Gaussian smoothing with boundary normalization.

    Uses zero-padding and normalizes by the kernel weight sum so that
    edge bins are not penalized.

    Parameters
    ----------
    data:
        1D array to smooth.
    sigma:
        Gaussian smoothing sigma in bins.
    segment_bins:
        Bin boundary indices for independent segments (e.g.
        ``[0, 50, 100, 200, 300]`` for 4 segments of varying length).
        Each segment ``[segment_bins[i]:segment_bins[i+1]]`` is smoothed
        independently.  When *None* the whole array is smoothed as one.
    """
    if sigma <= 0:
        return data.copy()
    if segment_bins is None or len(segment_bins) <= 2:
        smoothed = gaussian_filter1d(data, sigma=sigma, mode="constant", cval=0)
        norm = gaussian_filter1d(np.ones_like(data), sigma=sigma, mode="constant", cval=0)
        norm[norm == 0] = 1
        return smoothed / norm

    # Smooth each segment independently
    result = np.empty_like(data)
    for i in range(len(segment_bins) - 1):
        s = segment_bins[i]
        e = segment_bins[i + 1]
        seg = data[s:e]
        smoothed = gaussian_filter1d(seg, sigma=sigma, mode="constant", cval=0)
        norm = gaussian_filter1d(np.ones_like(seg), sigma=sigma, mode="constant", cval=0)
        norm[norm == 0] = 1
        result[s:e] = smoothed / norm
    return result


def compute_occupancy_map_1d(
    trajectory_df: pd.DataFrame,
    n_bins: int,
    pos_range: tuple[float, float],
    behavior_fps: float,
    spatial_sigma: float = 1.0,
    min_occupancy: float = 0.1,
    pos_column: str = "pos_1d",
    segment_bins: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 1D occupancy histogram.

    Parameters
    ----------
    trajectory_df:
        Speed-filtered trajectory with pos_column.
    n_bins:
        Total number of spatial bins across all arms.
    pos_range:
        (min, max) of the 1D position axis.
    behavior_fps:
        Behavior sampling rate.
    spatial_sigma:
        Gaussian smoothing sigma in bins.
    min_occupancy:
        Minimum occupancy in seconds.
    pos_column:
        Column name for position values.
    segment_bins:
        Bin boundary indices for per-segment smoothing.

    Returns
    -------
    tuple
        (occupancy_time, valid_mask, edges) -- all 1D arrays.
    """
    edges = np.linspace(pos_range[0], pos_range[1], n_bins + 1)
    time_per_frame = 1.0 / behavior_fps

    counts, _ = np.histogram(trajectory_df[pos_column], bins=edges)
    occupancy_time = counts.astype(float) * time_per_frame

    if spatial_sigma > 0:
        occupancy_time = gaussian_filter_normalized_1d(
            occupancy_time, sigma=spatial_sigma, segment_bins=segment_bins
        )

    valid_mask = occupancy_time >= min_occupancy

    return occupancy_time, valid_mask, edges


def compute_rate_map_1d(
    unit_events: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    edges: np.ndarray,
    spatial_sigma: float = 1.0,
    pos_column: str = "pos_1d",
    segment_bins: list[int] | None = None,
) -> np.ndarray:
    """Compute smoothed and normalized 1D rate map.

    Returns
    -------
    np.ndarray
        1D rate map normalized to 0-1 range. Invalid bins = NaN.
    """
    if unit_events.empty:
        return np.full_like(occupancy_time, np.nan)

    event_weights, _ = np.histogram(
        unit_events[pos_column],
        bins=edges,
        weights=unit_events["s"],
    )
    event_weights = event_weights.astype(float)
    # Smooth numerator and denominator independently (Skaggs et al. 1996)
    event_smooth = gaussian_filter_normalized_1d(
        event_weights, sigma=spatial_sigma, segment_bins=segment_bins
    )
    occ_smooth = gaussian_filter_normalized_1d(
        occupancy_time, sigma=spatial_sigma, segment_bins=segment_bins
    )
    rate_map_smooth = np.zeros_like(occupancy_time)
    rate_map_smooth[valid_mask] = event_smooth[valid_mask] / occ_smooth[valid_mask]

    valid_rate_values = rate_map_smooth[valid_mask]
    if len(valid_rate_values) > 0 and np.nanmax(valid_rate_values) > 0:
        rate_map_smooth[valid_mask] = rate_map_smooth[valid_mask] / np.nanmax(valid_rate_values)

    rate_map_smooth[~valid_mask] = np.nan
    return rate_map_smooth


def compute_raw_rate_map_1d(
    unit_events: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    edges: np.ndarray,
    pos_column: str = "pos_1d",
) -> np.ndarray:
    """Compute unsmoothed 1D rate map."""
    if unit_events.empty:
        return np.full_like(occupancy_time, np.nan)
    event_weights, _ = np.histogram(
        unit_events[pos_column],
        bins=edges,
        weights=unit_events["s"],
    )
    event_weights = event_weights.astype(float)
    rate_map = np.full_like(occupancy_time, np.nan)
    rate_map[valid_mask] = event_weights[valid_mask] / occupancy_time[valid_mask]
    return rate_map


def compute_spatial_information_1d(
    unit_events: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    edges: np.ndarray,
    n_shuffles: int = 100,
    random_seed: int | None = None,
    min_shift_seconds: float = 0.0,
    behavior_fps: float = 20.0,
    si_weight_mode: str = "amplitude",
    spatial_sigma: float = 0.0,
    pos_column: str = "pos_1d",
    segment_bins: list[int] | None = None,
) -> tuple[float, float, np.ndarray]:
    """Compute 1D spatial information with shuffle significance test.

    Same SI formula as 2D, applied to 1D bins. Circular shift shuffle.

    Returns
    -------
    tuple: (spatial_info, p_value, shuffled_sis)
    """
    rng = np.random.RandomState(random_seed)

    if unit_events.empty:
        return 0.0, 1.0, np.zeros(n_shuffles)

    weights = np.ones(len(unit_events)) if si_weight_mode == "binary" else unit_events["s"].values

    event_weights, _ = np.histogram(
        unit_events[pos_column],
        bins=edges,
        weights=weights,
    )
    event_weights = event_weights.astype(float)
    # Smooth numerator and denominator independently (Skaggs et al. 1996).
    # occ_smooth is loop-invariant and is reused inside the shuffle loop below.
    if spatial_sigma > 0:
        occ_smooth = gaussian_filter_normalized_1d(
            occupancy_time, sigma=spatial_sigma, segment_bins=segment_bins
        )
        event_smooth = gaussian_filter_normalized_1d(
            event_weights, sigma=spatial_sigma, segment_bins=segment_bins
        )
        rate_map = np.zeros_like(occupancy_time)
        rate_map[valid_mask] = event_smooth[valid_mask] / occ_smooth[valid_mask]
    else:
        rate_map = np.zeros_like(occupancy_time)
        rate_map[valid_mask] = event_weights[valid_mask] / occupancy_time[valid_mask]

    total_time = np.sum(occupancy_time[valid_mask])

    if total_time <= 0 or np.sum(event_weights[valid_mask]) <= 0:
        return 0.0, 1.0, np.zeros(n_shuffles)

    # Compute overall rate from smoothed map so Σ P_i × (λ_i/λ̄) = 1
    overall_lambda = np.sum(rate_map[valid_mask] * occupancy_time[valid_mask]) / total_time
    P_i = np.zeros_like(occupancy_time)
    P_i[valid_mask] = occupancy_time[valid_mask] / total_time

    if overall_lambda <= 0:
        return 0.0, 1.0, np.zeros(n_shuffles)

    valid_si = (rate_map > 0) & valid_mask
    if np.any(valid_si):
        ratio = rate_map[valid_si] / overall_lambda
        si_term = P_i[valid_si] * ratio * np.log2(ratio)
        actual_si = float(np.sum(si_term))
    else:
        actual_si = 0.0

    # Shuffling test
    traj_frames = trajectory_df["frame_index"].values
    if si_weight_mode == "binary":
        u_grouped = unit_events.groupby("frame_index").size()
    else:
        u_grouped = unit_events.groupby("frame_index")["s"].sum()
    aligned_events = u_grouped.reindex(traj_frames, fill_value=0).values.astype(float)

    n_frames = len(aligned_events)
    min_shift_frames = int(min_shift_seconds * behavior_fps)
    min_shift_frames = min(min_shift_frames, n_frames // 2)

    traj_pos = trajectory_df[pos_column].values

    shuffled_sis = []
    for _ in range(n_shuffles):
        if min_shift_frames > 0:
            shift = rng.randint(min_shift_frames, n_frames - min_shift_frames)
        else:
            shift = rng.randint(1, n_frames)
        s_shuffled = np.roll(aligned_events, shift)

        event_w_shuf, _ = np.histogram(traj_pos, bins=edges, weights=s_shuffled)
        event_w_shuf = event_w_shuf.astype(float)

        # Smooth numerator only; occ_smooth was hoisted out of the loop above.
        if spatial_sigma > 0:
            event_smooth = gaussian_filter_normalized_1d(
                event_w_shuf, sigma=spatial_sigma, segment_bins=segment_bins
            )
            rate_shuf = np.zeros_like(occupancy_time)
            rate_shuf[valid_mask] = event_smooth[valid_mask] / occ_smooth[valid_mask]
        else:
            rate_shuf = np.zeros_like(occupancy_time)
            rate_shuf[valid_mask] = event_w_shuf[valid_mask] / occupancy_time[valid_mask]

        # Use smoothed overall rate for each shuffle (consistent SI)
        shuf_lambda = np.sum(rate_shuf[valid_mask] * occupancy_time[valid_mask]) / total_time
        valid_s = (rate_shuf > 0) & valid_mask & (shuf_lambda > 0)
        if np.any(valid_s) and shuf_lambda > 0:
            ratio_s = rate_shuf[valid_s] / shuf_lambda
            si_shuf = np.sum(P_i[valid_s] * ratio_s * np.log2(ratio_s))
        else:
            si_shuf = 0.0
        shuffled_sis.append(si_shuf)

    shuffled_sis = np.array(shuffled_sis)
    p_val = (np.sum(shuffled_sis >= actual_si) + 1) / (n_shuffles + 1)

    return actual_si, p_val, shuffled_sis


def compute_stability_score_1d(
    unit_events: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    edges: np.ndarray,
    spatial_sigma: float = 1.0,
    behavior_fps: float = 20.0,
    min_occupancy: float = 0.1,
    n_split_blocks: int = 10,
    block_shift: float = 0.0,
    n_shuffles: int = 0,
    random_seed: int | None = None,
    min_shift_seconds: float = 0.0,
    si_weight_mode: str = "amplitude",
    pos_column: str = "pos_1d",
    segment_bins: list[int] | None = None,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 1D split-half stability test.

    Same interleaved block-splitting approach as 2D, with 1D rate maps.

    Returns
    -------
    tuple: (correlation, fisher_z, stability_p_val,
            rate_map_first, rate_map_second, shuffled_corrs)
    """
    if unit_events.empty or trajectory_df.empty:
        nan_map = np.full_like(occupancy_time, np.nan)
        return np.nan, np.nan, np.nan, nan_map, nan_map, np.array([])

    if si_weight_mode == "binary":
        unit_events = unit_events.copy()
        unit_events["s"] = 1.0

    # Split trajectory into interleaved temporal blocks
    all_frames = trajectory_df["frame_index"].values
    frame_min = all_frames.min()
    frame_max = all_frames.max()
    span = frame_max - frame_min
    if span == 0:
        nan_map = np.full_like(occupancy_time, np.nan)
        return np.nan, np.nan, np.nan, nan_map, nan_map, np.array([])
    block_width = span / n_split_blocks
    offset = block_shift * block_width

    traj_block_ids = np.floor((all_frames - frame_min - offset) / block_width).astype(int)
    traj_block_ids = np.clip(traj_block_ids, 0, n_split_blocks - 1)
    traj_first_mask = traj_block_ids % 2 == 0
    traj_second_mask = ~traj_first_mask

    event_frames = unit_events["frame_index"].values
    event_block_ids = np.floor((event_frames - frame_min - offset) / block_width).astype(int)
    event_block_ids = np.clip(event_block_ids, 0, n_split_blocks - 1)
    events_first_mask = event_block_ids % 2 == 0
    events_second_mask = ~events_first_mask

    traj_first = trajectory_df[traj_first_mask]
    traj_second = trajectory_df[traj_second_mask]
    events_first = unit_events[events_first_mask]
    events_second = unit_events[events_second_mask]

    time_per_frame = 1.0 / behavior_fps

    def compute_half_occupancy(traj_half: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # Returns raw occupancy; the validity mask is derived from smoothed
        # occupancy so noisy edge bins are excluded, but the rate-map and
        # shuffle paths apply Skaggs (1996) single smoothing to the raw occ.
        if traj_half.empty:
            return np.zeros_like(occupancy_time), np.zeros_like(valid_mask, dtype=bool)
        counts, _ = np.histogram(traj_half[pos_column], bins=edges)
        occ = counts.astype(float) * time_per_frame
        occ_smooth = gaussian_filter_normalized_1d(
            occ, sigma=spatial_sigma, segment_bins=segment_bins
        )
        mask = occ_smooth >= min_occupancy
        return occ, mask

    occ_first, valid_first = compute_half_occupancy(traj_first)
    occ_second, valid_second = compute_half_occupancy(traj_second)

    def compute_half_rate_map(
        events: pd.DataFrame, occ: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        if events.empty or not np.any(mask):
            return np.full_like(occ, np.nan)
        event_weights, _ = np.histogram(
            events[pos_column],
            bins=edges,
            weights=events["s"],
        )
        event_weights = event_weights.astype(float)
        # Smooth numerator and denominator independently
        event_smooth = gaussian_filter_normalized_1d(
            event_weights, sigma=spatial_sigma, segment_bins=segment_bins
        )
        occ_smooth_half = gaussian_filter_normalized_1d(
            occ, sigma=spatial_sigma, segment_bins=segment_bins
        )
        rate_map_smooth = np.zeros_like(occ)
        rate_map_smooth[mask] = event_smooth[mask] / occ_smooth_half[mask]
        rate_map_smooth[~mask] = np.nan
        return rate_map_smooth

    rate_map_first = compute_half_rate_map(events_first, occ_first, valid_first)
    rate_map_second = compute_half_rate_map(events_second, occ_second, valid_second)

    both_valid = valid_first & valid_second
    if not np.any(both_valid):
        return np.nan, np.nan, np.nan, rate_map_first, rate_map_second, np.array([])

    vals_first = rate_map_first[both_valid]
    vals_second = rate_map_second[both_valid]

    finite_mask = np.isfinite(vals_first) & np.isfinite(vals_second)
    if np.sum(finite_mask) < 3:
        return np.nan, np.nan, np.nan, rate_map_first, rate_map_second, np.array([])

    vals_first = vals_first[finite_mask]
    vals_second = vals_second[finite_mask]

    corr = np.corrcoef(vals_first, vals_second)[0, 1]
    corr_clipped = np.clip(corr, -0.9999, 0.9999)
    fisher_z = np.arctanh(corr_clipped)

    # Half rate maps are returned unnormalized (in firing-rate units).
    # Display code should normalize all three maps (first, second, full)
    # to a shared scale for honest visual comparison.

    # Shuffle-based stability significance test
    if n_shuffles > 0:
        # Offset seed so stability shuffles are independent of SI shuffles
        stab_seed = random_seed + 224737 if random_seed is not None else None
        rng = np.random.RandomState(stab_seed)

        traj_frames = trajectory_df["frame_index"].values
        u_grouped = unit_events.groupby("frame_index")["s"].sum()
        aligned_events = u_grouped.reindex(traj_frames, fill_value=0).values.astype(float)

        traj_pos = trajectory_df[pos_column].values

        n_frames = len(aligned_events)
        min_shift_frames = int(min_shift_seconds * behavior_fps)
        min_shift_frames = min(min_shift_frames, n_frames // 2)

        # Smoothed half-occupancies are loop-invariant.
        os1 = gaussian_filter_normalized_1d(
            occ_first, sigma=spatial_sigma, segment_bins=segment_bins
        )
        os2 = gaussian_filter_normalized_1d(
            occ_second, sigma=spatial_sigma, segment_bins=segment_bins
        )

        shuffled_corrs = np.empty(n_shuffles)
        for i in range(n_shuffles):
            if min_shift_frames > 0:
                shift = rng.randint(min_shift_frames, n_frames - min_shift_frames)
            else:
                shift = rng.randint(1, n_frames)
            shifted = np.roll(aligned_events, shift)

            ew1, _ = np.histogram(
                traj_pos[traj_first_mask], bins=edges, weights=shifted[traj_first_mask]
            )
            es1 = gaussian_filter_normalized_1d(
                ew1.astype(float), sigma=spatial_sigma, segment_bins=segment_bins
            )
            rm1 = np.zeros_like(occ_first)
            rm1[valid_first] = es1[valid_first] / os1[valid_first]

            ew2, _ = np.histogram(
                traj_pos[traj_second_mask], bins=edges, weights=shifted[traj_second_mask]
            )
            es2 = gaussian_filter_normalized_1d(
                ew2.astype(float), sigma=spatial_sigma, segment_bins=segment_bins
            )
            rm2 = np.zeros_like(occ_second)
            rm2[valid_second] = es2[valid_second] / os2[valid_second]

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

        stability_p_val = float((np.sum(shuffled_corrs >= corr) + 1) / (n_shuffles + 1))
    else:
        stability_p_val = np.nan
        shuffled_corrs = np.array([])

    return corr, fisher_z, stability_p_val, rate_map_first, rate_map_second, shuffled_corrs


def compute_unit_analysis_1d(
    unit_id: int,
    df_filtered: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    edges: np.ndarray,
    scfg: SpatialMap1DConfig,
    behavior_fps: float,
    random_seed: int | None = None,
    pos_column: str = "pos_1d",
    segment_bins: list[int] | None = None,
) -> dict:
    """Compute 1D rate map, SI, stability, and place field for a unit.

    Returns same dict keys as the 2D ``compute_unit_analysis``.
    """
    unit_data = (
        df_filtered[df_filtered["unit_id"] == unit_id] if not df_filtered.empty else pd.DataFrame()
    )

    rate_map = compute_rate_map_1d(
        unit_data,
        occupancy_time,
        valid_mask,
        edges,
        scfg.spatial_sigma,
        pos_column,
        segment_bins=segment_bins,
    )
    rate_map_raw = compute_raw_rate_map_1d(unit_data, occupancy_time, valid_mask, edges, pos_column)

    # Overall rate: amplitude-weighted and binary event count
    valid_occ = valid_mask & (occupancy_time > 0)
    total_time = float(np.sum(occupancy_time[valid_occ])) if np.any(valid_occ) else 0.0
    if total_time > 0:
        overall_rate = float(
            np.sum(rate_map_raw[valid_occ] * occupancy_time[valid_occ]) / total_time
        )
        event_count_rate = float(len(unit_data)) / total_time
    else:
        overall_rate = 0.0
        event_count_rate = 0.0

    si, p_val, shuffled_sis = compute_spatial_information_1d(
        unit_data,
        trajectory_df,
        occupancy_time,
        valid_mask,
        edges,
        scfg.n_shuffles,
        random_seed=random_seed,
        min_shift_seconds=scfg.min_shift_seconds,
        behavior_fps=behavior_fps,
        si_weight_mode=scfg.si_weight_mode,
        spatial_sigma=scfg.spatial_sigma,
        pos_column=pos_column,
        segment_bins=segment_bins,
    )

    # Visualization events: speed-filtered only (no amplitude threshold)
    events_above = unit_data
    vis_threshold = 0.0

    # Stability tests — one per configured split.
    stability_splits: list[dict] = []
    full_peak = float(np.nanmax(rate_map_raw[valid_mask])) if valid_mask.any() else 0.0
    for n_splits in scfg.stability_splits:
        (
            s_corr,
            s_z,
            s_p,
            rm_first,
            rm_second,
            shuffled_s,
        ) = compute_stability_score_1d(
            unit_data,
            trajectory_df,
            occupancy_time,
            valid_mask,
            edges,
            spatial_sigma=scfg.spatial_sigma,
            behavior_fps=behavior_fps,
            min_occupancy=scfg.min_occupancy,
            n_split_blocks=n_splits,
            block_shift=scfg.block_shift,
            n_shuffles=scfg.n_shuffles,
            random_seed=random_seed,
            min_shift_seconds=scfg.min_shift_seconds,
            si_weight_mode=scfg.si_weight_mode,
            pos_column=pos_column,
            segment_bins=segment_bins,
        )
        if full_peak > 0:
            for rm_half in (rm_first, rm_second):
                finite = np.isfinite(rm_half)
                if finite.any():
                    rm_half[finite] = rm_half[finite] / full_peak
        stability_splits.append({
            "n_split_blocks": n_splits,
            "corr": s_corr,
            "fisher_z": s_z,
            "p_val": s_p,
            "shuffled_corrs": shuffled_s,
            "rate_map_first": rm_first,
            "rate_map_second": rm_second,
        })

    return {
        "rate_map": rate_map,
        "rate_map_raw": rate_map_raw,
        "overall_rate": overall_rate,
        "event_count_rate": event_count_rate,
        "si": si,
        "p_val": p_val,
        "shuffled_sis": shuffled_sis,
        "events_above_threshold": events_above,
        "vis_threshold": vis_threshold,
        "unit_data": unit_data,
        "stability_splits": stability_splits,
    }
