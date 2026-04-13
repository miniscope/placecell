"""Timestamp and temporal-overlap validation for neural/behavior recordings."""

import numpy as np
import pandas as pd

from placecell.log import init_logger

logger = init_logger(__name__)

# If neural_fps exceeds behavior_fps by more than this factor, refuse to
# interpolate behavior onto neural timestamps: the resulting per-frame
# table would be dominated by tracking jitter rather than animal motion.
MAX_NEURAL_TO_BEHAVIOR_FPS_RATIO = 5.0


def hampel_mask(
    deviation: pd.Series,
    window: int = 11,
    n_sigmas: float = 3.0,
) -> np.ndarray:
    """Flag outliers using the MAD-based Hampel rule (Hampel 1974).

    The constant 1.4826 scales MAD to standard deviation for Gaussian data.

    Parameters
    ----------
    deviation:
        Per-sample deviation from the local trend (e.g. ``|t - median(t)|``
        for timestamps, or ``hypot(x - median(x), y - median(y))`` for 2D
        positions).
    window:
        Centered rolling window for the MAD estimate.
    n_sigmas:
        Threshold in MAD-scaled standard deviations.

    Returns
    -------
    Boolean array — ``True`` for outlier samples.
    """
    min_periods = window // 2 + 1
    mad = deviation.rolling(window, center=True, min_periods=min_periods).median()
    return (deviation > n_sigmas * 1.4826 * mad).fillna(False).to_numpy()


def infer_fps(timestamps: np.ndarray) -> float:
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

    1. **Hampel outliers** — frames whose timestamp deviates from the local
       trend (rolling median, window=11, 3σ MAD) are excluded.
    2. **Residual backward jumps** — any remaining non-monotonic frames
       after Hampel are excluded.
    3. **Large forward gaps** — warned but NOT excluded (recording stalls
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

    # Sanity checks (corrupted CSV, not a real recording failure mode).
    if np.any(np.isnan(neural_time)):
        raise ValueError("neural_time contains NaN — check the CSV file.")
    if len(neural_time) < 2:
        raise ValueError("neural_time must have at least 2 entries.")

    # 1. Hampel filter on the timestamp signal
    original_idx = np.arange(len(neural_time), dtype=np.int64)
    ts_series = pd.Series(neural_time)
    t_med = ts_series.rolling(11, center=True, min_periods=6).median()
    bad = hampel_mask((ts_series - t_med).abs()).copy()

    # 2. Residual backward jumps after Hampel
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

    # 3. Large forward gaps — warn only
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
