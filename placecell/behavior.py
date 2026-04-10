"""Behavior data loading and processing."""

from pathlib import Path

import numpy as np
import pandas as pd

from placecell.log import init_logger

logger = init_logger(__name__)

# If neural_fps exceeds behavior_fps by more than this factor, refuse to
# interpolate behavior onto neural timestamps: the resulting per-frame
# table would be dominated by tracking jitter rather than animal motion.
MAX_NEURAL_TO_BEHAVIOR_FPS_RATIO = 5.0


def _infer_fps(timestamps: np.ndarray) -> float:
    """Estimate sampling rate from a monotonic timestamp array."""
    if timestamps.size < 2:
        raise ValueError("Need at least two timestamps to infer fps.")
    diffs = np.diff(timestamps)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise ValueError("Timestamps must be strictly increasing.")
    return float(1.0 / np.median(diffs))


def validate_neural_timestamps(
    neural_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate neural timestamps and return the clean subset.

    Checks (in order):

    1. **NaN** — hard error.
    2. **< 2 entries** — hard error.
    3. **Hampel outliers** — frames whose timestamp deviates from the local
       trend (rolling median, window=11, 3σ MAD) are excluded.
    4. **Residual backward jumps** — any remaining non-monotonic frames
       after Hampel are excluded.
    5. **Large forward gaps** — warned but NOT excluded (recording stalls
       have valid timestamps on both sides; only interpolated behavior
       within the gap is unreliable).

    Parameters
    ----------
    neural_time:
        Raw neural timestamps (one per frame).

    Returns
    -------
    (clean_time, original_indices):
        ``clean_time`` is the monotonic subset of ``neural_time``.
        ``original_indices`` maps each surviving row back to its
        position in the input array (for S_list lookup).
    """
    neural_time = np.asarray(neural_time, dtype=float)

    # 1. NaN check
    if np.any(np.isnan(neural_time)):
        n_nan = int(np.isnan(neural_time).sum())
        raise ValueError(
            f"neural_time contains {n_nan} NaN value(s). " "Inspect the neural_timestamp CSV."
        )

    # 2. Length check
    if len(neural_time) < 2:
        raise ValueError("neural_time must have at least 2 entries.")

    # 3. Hampel filter on the timestamp signal
    original_idx = np.arange(len(neural_time), dtype=np.int64)
    ts_series = pd.Series(neural_time)
    hampel_window = 11
    min_periods = hampel_window // 2 + 1
    t_med = ts_series.rolling(hampel_window, center=True, min_periods=min_periods).median()
    deviation = (ts_series - t_med).abs()
    mad = deviation.rolling(hampel_window, center=True, min_periods=min_periods).median()
    bad = (deviation > 3.0 * 1.4826 * mad).fillna(False).to_numpy().copy()

    # 4. Residual backward jumps after Hampel
    clean_idx = np.where(~bad)[0]
    clean_t = neural_time[clean_idx]
    if len(clean_t) > 1:
        residual_back = np.zeros(len(clean_t), dtype=bool)
        residual_back[1:] = np.diff(clean_t) < 0
        if residual_back.any():
            bad[clean_idx[residual_back]] = True

    n_bad = int(bad.sum())
    if n_bad:
        logger.warning(
            "Excluding %d neural frames (%.2f%%) with anomalous timestamps "
            "(Hampel outliers + residual backward jumps).",
            n_bad,
            100 * n_bad / len(neural_time),
        )
        neural_time = neural_time[~bad]
        original_idx = original_idx[~bad]

    # 5. Large forward gaps — warn only
    diffs = np.diff(neural_time)
    median_dt = float(np.median(diffs[diffs > 0])) if (diffs > 0).any() else 1.0
    gap_threshold = max(1.0, 10 * median_dt)
    n_gap = int((diffs > gap_threshold).sum())
    if n_gap:
        max_gap = float(diffs[diffs > gap_threshold].max())
        logger.warning(
            "%d large forward gap(s) in neural timestamps "
            "(max %.1fs, expected ~%.3fs). Possible recording stall; "
            "interpolated behavior within these gaps may be unreliable.",
            n_gap,
            max_gap,
            median_dt,
        )

    return neural_time, original_idx


def validate_temporal_overlap(
    beh_time: np.ndarray,
    neural_time: np.ndarray,
    behavior_fps: float | None = None,
    neural_fps: float | None = None,
) -> np.ndarray:
    """Check behavior/neural overlap and return an in-range mask.

    Checks:

    1. **Zero overlap** — hard error.
    2. **Partial overlap** — logged (neural frames outside behavior time
       window will be dropped downstream).
    3. **FPS ratio** — hard error if neural_fps > 5× behavior_fps.

    Parameters
    ----------
    beh_time:
        Sorted behavior timestamps.
    neural_time:
        Sorted neural timestamps (after validation).
    behavior_fps, neural_fps:
        Inferred sampling rates. Pass ``None`` to skip the FPS ratio check.

    Returns
    -------
    Boolean mask (one per neural frame): ``True`` for frames within
    the behavior time window.
    """
    in_range = (neural_time >= beh_time[0]) & (neural_time <= beh_time[-1])
    n_covered = int(in_range.sum())

    # 1. Zero overlap
    if n_covered == 0:
        raise ValueError(
            "No overlap between behavior and neural timestamps. "
            f"Behavior: [{beh_time[0]:.3f}, {beh_time[-1]:.3f}], "
            f"Neural: [{neural_time[0]:.3f}, {neural_time[-1]:.3f}]."
        )

    # 2. Partial overlap
    n_uncovered = len(neural_time) - n_covered
    if n_uncovered > 0:
        logger.info(
            "Behavior coverage: %d/%d neural frames (%.1f%% uncovered, "
            "outside behavior time window, will be dropped).",
            n_covered,
            len(neural_time),
            100 * n_uncovered / len(neural_time),
        )

    # 3. FPS ratio
    if (
        behavior_fps is not None
        and neural_fps is not None
        and neural_fps > MAX_NEURAL_TO_BEHAVIOR_FPS_RATIO * behavior_fps
    ):
        raise ValueError(
            "Refusing to interpolate: "
            f"neural_fps ({neural_fps:.1f} Hz) exceeds "
            f"{MAX_NEURAL_TO_BEHAVIOR_FPS_RATIO:.0f}× behavior_fps "
            f"({behavior_fps:.1f} Hz)."
        )

    return in_range


def interpolate_behavior_onto_neural(
    behavior: pd.DataFrame,
    neural_time: np.ndarray,
    *,
    columns: list[str] | None = None,
    fps_check: bool = True,
) -> pd.DataFrame:
    """Linearly interpolate per-frame behavior columns onto neural timestamps.

    Runs :func:`validate_neural_timestamps` and
    :func:`validate_temporal_overlap` first, then interpolates each
    requested column via ``np.interp``.

    Parameters
    ----------
    behavior:
        DataFrame at behavior rate with a ``unix_time`` column.
    neural_time:
        Raw neural timestamps (one per frame). Anomalous entries are
        excluded automatically by :func:`validate_neural_timestamps`.
    columns:
        Behavior columns to interpolate. ``None`` = all except
        ``unix_time`` and ``frame_index``.
    fps_check:
        If ``True`` (default), error when neural_fps > 5× behavior_fps.

    Returns
    -------
    DataFrame with one row per surviving neural frame and columns
    ``[frame_index, neural_time, *interpolated_columns]``. Frames
    outside the behavior time window get NaN in every interpolated column.
    """
    if "unix_time" not in behavior.columns:
        raise ValueError("behavior DataFrame must contain a 'unix_time' column.")
    beh_time = behavior["unix_time"].to_numpy()
    if not np.all(np.diff(beh_time) >= 0):
        order = np.argsort(beh_time)
        behavior = behavior.iloc[order].reset_index(drop=True)
        beh_time = behavior["unix_time"].to_numpy()

    neural_time, original_idx = validate_neural_timestamps(neural_time)

    behavior_fps = _infer_fps(beh_time) if fps_check else None
    neural_fps = _infer_fps(neural_time) if fps_check else None
    in_range = validate_temporal_overlap(beh_time, neural_time, behavior_fps, neural_fps)

    if columns is None:
        skip = {"unix_time", "frame_index"}
        columns = [c for c in behavior.columns if c not in skip]

    out: dict[str, np.ndarray] = {
        "frame_index": original_idx,
        "neural_time": neural_time,
    }
    for col in columns:
        raw = behavior[col]
        values = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
        n_coerced = int(raw.notna().sum() - np.isfinite(values).sum())
        if n_coerced:
            logger.warning(
                "Column '%s': %d non-numeric value(s) coerced to NaN.",
                col,
                n_coerced,
            )
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            interp = np.full(len(neural_time), np.nan, dtype=float)
        else:
            interp = np.interp(neural_time, beh_time[finite_mask], values[finite_mask])
        interp = np.where(in_range, interp, np.nan)
        out[col] = interp

    return pd.DataFrame(out)


def build_canonical_table(
    behavior_at_neural: pd.DataFrame,
    deconv_traces: dict[int, np.ndarray] | pd.DataFrame,
    *,
    drop_uncovered: bool = True,
) -> pd.DataFrame:
    """Combine the neural-rate behavior table with deconvolved traces.

    Parameters
    ----------
    behavior_at_neural:
        Output of :func:`interpolate_behavior_onto_neural` (or any DataFrame
        with ``frame_index`` and ``neural_time`` columns).
    deconv_traces:
        Per-unit deconvolved spike trains. Either a ``{unit_id: array}``
        mapping or a wide DataFrame indexed by neural frame with one
        column per unit.
    drop_uncovered:
        If ``True`` (default) drop neural frames where the behavior
        interpolation produced NaN (i.e. outside behavior coverage).

    Returns
    -------
    DataFrame with one row per surviving neural frame and columns
    ``[frame_index, neural_time, *behavior_cols, s_unit_<id>...]``.
    """
    df = behavior_at_neural.copy()

    if isinstance(deconv_traces, pd.DataFrame):
        traces_df = deconv_traces
    else:
        traces_df = pd.DataFrame(
            {f"s_unit_{int(uid)}": np.asarray(s) for uid, s in deconv_traces.items()}
        )
    # ``frame_index`` in ``behavior_at_neural`` maps each row to the
    # *original* neural frame index (before any timestamp-sorting).
    # Use it to look up the correct spike train entry for each row.
    orig_idx = df["frame_index"].to_numpy()
    if orig_idx.max() >= len(traces_df):
        raise ValueError(
            "deconv_traces length does not match neural-frame count "
            f"({len(traces_df)} traces vs max frame_index {orig_idx.max()})."
        )
    reindexed = traces_df.iloc[orig_idx].reset_index(drop=True)
    df = df.reset_index(drop=True)
    out = pd.concat([df, reindexed], axis=1)

    if drop_uncovered:
        beh_cols = [c for c in df.columns if c not in ("frame_index", "neural_time")]
        if beh_cols:
            out = out.dropna(subset=beh_cols).reset_index(drop=True)

    return out


def filter_canonical_by_speed(
    canonical: pd.DataFrame,
    *,
    speed_column: str,
    speed_threshold: float,
    drop_below_threshold: bool = True,
) -> pd.DataFrame:
    """Return a speed-filtered view of the canonical neural-rate table.

    When ``drop_below_threshold`` is ``True`` (default) rows below
    ``speed_threshold`` are dropped. When ``False``, every row is kept
    (useful for building unfiltered notebook views).
    """
    if speed_column not in canonical.columns:
        raise ValueError(f"speed column {speed_column!r} missing from canonical table.")
    if drop_below_threshold:
        speeds = canonical[speed_column]
        n_nan = int(speeds.isna().sum())
        if n_nan:
            logger.info(
                "Speed filter: %d frame(s) with NaN %s excluded.",
                n_nan,
                speed_column,
            )
        keep = canonical[speeds.fillna(-np.inf) >= speed_threshold]
    else:
        keep = canonical
    return keep.reset_index(drop=True)


def derive_event_place_from_canonical(
    canonical_filtered: pd.DataFrame,
    *,
    position_columns: tuple[str, ...] = ("x", "y"),
    speed_column: str = "speed",
    extra_columns: tuple[str, ...] = (),
    event_threshold: float = 1e-12,
) -> pd.DataFrame:
    """Melt a (speed-filtered) canonical table into a long-format event table.

    For each ``s_unit_<uid>`` column, emit one row per frame whose
    amplitude exceeds ``event_threshold``. The output schema matches the
    long-format ``event_place`` shape (``unit_id, frame_index, x, y, s,
    speed, ...``) so that ``compute_unit_analysis*`` can consume it
    without modification.

    Parameters
    ----------
    canonical_filtered:
        Output of :func:`filter_canonical_by_speed`. Must have a
        ``frame_index`` column.
    position_columns:
        Position columns to copy into the long-format output.
    speed_column:
        Speed column to copy into the long-format output.
    extra_columns:
        Any additional per-frame columns to keep (e.g. ``("pos_1d",
        "arm_index", "direction")`` for the maze pipeline).
    event_threshold:
        Minimum spike amplitude to consider an event. Smaller values are
        skipped (deconvolution can produce ~1e-17 ghost events).
    """
    if "frame_index" not in canonical_filtered.columns:
        raise ValueError(
            "canonical_filtered must have a frame_index column. "
            "Did you forget to call filter_canonical_by_speed first?"
        )
    keep = canonical_filtered

    unit_cols = [c for c in keep.columns if c.startswith("s_unit_")]
    if not unit_cols:
        return pd.DataFrame()

    base_cols = ["frame_index"]
    for c in (*position_columns, speed_column, *extra_columns):
        if c in keep.columns and c not in base_cols:
            base_cols.append(c)
    base = keep[base_cols]

    pieces = []
    for col in unit_cols:
        unit_id = int(col.removeprefix("s_unit_"))
        amplitudes = keep[col].to_numpy()
        mask = amplitudes > event_threshold
        if not mask.any():
            continue
        slice_df = base.loc[mask].copy()
        slice_df["unit_id"] = unit_id
        slice_df["s"] = amplitudes[mask]
        pieces.append(slice_df)

    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    return out


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
    window_frames: int = 7,
    n_sigmas: float = 3.0,
) -> tuple[pd.DataFrame, int]:
    """Replace implausible position jumps with linear interpolation (Hampel filter).

    For each frame the local centroid is the (median x, median y) over a
    centered window of ``window_frames`` frames. The deviation is the
    Euclidean distance from the frame to its centroid; the local scale is
    the rolling median of those deviations. A frame is flagged when its
    deviation exceeds ``n_sigmas * 1.4826 * scale`` — the standard Hampel
    rule (Hampel 1974) generalized to 2D via the spatial median.

    Flagged frames have their x/y replaced by linear interpolation from the
    surrounding good frames.

    Parameters
    ----------
    positions:
        DataFrame with columns ``x``, ``y`` (and any others, preserved).
    window_frames:
        Window size for the rolling median centroid and MAD.  Should be odd
        and large enough to span typical glitch durations.
    n_sigmas:
        Number of (MAD-based) standard deviations beyond which a frame is
        treated as an outlier.  3.0 corresponds to a ~99.7% Gaussian band.

    Returns
    -------
    tuple of (DataFrame with jumps interpolated, number of frames fixed).
    """
    if window_frames < 3:
        raise ValueError("window_frames must be >= 3.")

    df = positions.copy()
    x = df["x"].astype(float)
    y = df["y"].astype(float)

    min_periods = window_frames // 2 + 1
    x_med = x.rolling(window_frames, center=True, min_periods=min_periods).median()
    y_med = y.rolling(window_frames, center=True, min_periods=min_periods).median()
    deviation = np.hypot(x - x_med, y - y_med)
    scale = deviation.rolling(window_frames, center=True, min_periods=min_periods).median()

    threshold = n_sigmas * 1.4826 * scale
    bad = (deviation > threshold).fillna(False).to_numpy()

    n_bad = int(bad.sum())
    if n_bad > 0:
        x_clean = x.to_numpy(copy=True)
        y_clean = y.to_numpy(copy=True)
        x_clean[bad] = np.nan
        y_clean[bad] = np.nan
        df["x"] = pd.Series(x_clean).interpolate(limit_direction="both").to_numpy()
        df["y"] = pd.Series(y_clean).interpolate(limit_direction="both").to_numpy()

    return df, n_bad


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


def compute_speed_2d(
    trajectory: pd.DataFrame,
    *,
    window_seconds: float,
    sample_rate_hz: float,
    time_column: str = "neural_time",
) -> pd.DataFrame:
    """Add a ``speed`` column to a 2D trajectory using a centered window.

    Parameters
    ----------
    trajectory:
        DataFrame with ``x``, ``y``, and a monotonic time column.
    window_seconds:
        Centered window span (seconds).
    sample_rate_hz:
        Trajectory sampling rate, used to convert ``window_seconds`` to
        a frame count.
    time_column:
        Name of the time column to use for the dt denominator. Defaults
        to ``"neural_time"`` (the canonical neural-rate clock).

    Returns
    -------
    DataFrame with a fresh ``speed`` column (position-units / s).
    """
    window_frames = max(1, int(round(window_seconds * sample_rate_hz)))
    df = trajectory.copy()
    x_vals = df["x"].to_numpy()
    y_vals = df["y"].to_numpy()
    t_vals = df[time_column].to_numpy()
    n = len(df)
    half = window_frames // 2

    start_indices = np.clip(np.arange(n) - half, 0, n - 1)
    end_indices = np.clip(np.arange(n) + half, 0, n - 1)

    dx = x_vals[end_indices] - x_vals[start_indices]
    dy = y_vals[end_indices] - y_vals[start_indices]
    dt = t_vals[end_indices] - t_vals[start_indices]

    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, np.sqrt(dx**2 + dy**2) / dt, np.nan)

    df["speed"] = speed
    return df
