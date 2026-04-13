"""Interpolate behavior onto neural timestamps and build the canonical table."""

import numpy as np
import pandas as pd

from placecell.dataset_validation import (
    infer_fps,
    validate_neural_timestamps,
    validate_temporal_overlap,
)
from placecell.log import init_logger

logger = init_logger(__name__)


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

    behavior_fps = infer_fps(beh_time) if fps_check else None
    neural_fps = infer_fps(neural_time) if fps_check else None
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
    # *original* neural frame index (before any timestamp exclusion).
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
