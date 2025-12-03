"""Event detection and kinetics utilities derived from ``WL25_oasis.ipynb``.

These helpers are factored out so they can be reused from scripts and notebooks.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

ArrayLike = Sequence[float] | np.ndarray


def robust_baseline(y: ArrayLike, q: float = 10) -> float:
    """Robust baseline estimate using a lower percentile."""

    y_arr = np.asarray(y, dtype=float)
    return float(np.percentile(y_arr, q))


def simple_lowpass(y: ArrayLike, wlen_sec: float = 0.3, fps: float = 20.0) -> np.ndarray:
    """Simple moving-average low-pass filter."""

    y_arr = np.asarray(y, dtype=float)
    w = max(3, int(round(wlen_sec * fps)) | 1)  # odd window
    k = np.ones(w, dtype=float) / w
    return np.convolve(y_arr, k, mode="same")


def detect_isolated_events(
    y: ArrayLike,
    fps: float,
    min_prom_sigma: float = 3.5,
    min_separation_s: float = 1.2,
    pre_s: float = 0.5,
    post_s: float = 2.0,
    smooth_sec: float | None = None,
) -> Tuple[List[Tuple[int, int, int]], float]:
    """Detect isolated events in a 1D trace.

    Parameters
    ----------
    y:
        1D fluorescence trace.
    fps:
        Sampling rate (frames per second).
    min_prom_sigma:
        Minimum peak prominence in units of robust noise (MAD-based).
    min_separation_s:
        Minimum separation between events in seconds.
    pre_s, post_s:
        Amount of data (in seconds) to include before/after the detected peak.
    smooth_sec:
        If not ``None``, apply a simple moving-average low-pass before detection.

    Returns
    -------
    events:
        List of ``(start_idx, peak_idx, end_idx)`` tuples.
    baseline:
        Robust baseline level used for z-scoring.
    """

    y_arr = np.asarray(y, dtype=float)

    if smooth_sec is not None:
        y_use = simple_lowpass(y_arr, wlen_sec=smooth_sec, fps=fps)
    else:
        y_use = y_arr

    base = robust_baseline(y_use, q=10)
    z = y_use - base

    # robust noise estimate
    mad = np.median(np.abs(z - np.median(z))) + 1e-12
    sig = 1.4826 * mad
    prom = max(1e-6, float(min_prom_sigma) * sig)

    distance = int(round(min_separation_s * fps))
    peaks, _ = find_peaks(z, prominence=prom, distance=distance)

    events: List[Tuple[int, int, int]] = []
    pre = int(round(pre_s * fps))
    post = int(round(post_s * fps))
    n = len(y_arr)
    for p in peaks:
        s = max(0, p - pre)
        e = min(n, p + post)
        if e - s >= 5:
            events.append((s, p, e))
    return events, float(base)


def frac_time_to_level(t: ArrayLike, y: ArrayLike, level: float) -> float:
    """Time at which ``y`` first crosses ``level`` (linear interpolation)."""

    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    above = y_arr >= level
    if not np.any(above):
        return float("nan")
    idx = int(np.argmax(above))
    if idx == 0:
        return float(t_arr[0])
    t0, t1 = t_arr[idx - 1], t_arr[idx]
    y0, y1 = y_arr[idx - 1], y_arr[idx]
    if y1 == y0:
        return float(t1)
    return float(t0 + (level - y0) * (t1 - t0) / (y1 - y0))


def _exp_decay(t: np.ndarray, A: float, tau: float, B: float) -> np.ndarray:
    """Single-exponential decay model with offset."""

    t0 = float(t[0])
    return A * np.exp(-(t - t0) / max(1e-9, tau)) + B


@dataclass
class EventKinetics:
    """Container for single-event kinetics."""

    rise_t10_90_s: float
    tau_decay_s: float
    A_peak: float
    ok_rise: bool
    ok_tau: bool


def measure_event_kinetics(
    t: ArrayLike,
    y: ArrayLike,
    peak_idx: int,
    base_val: float,
    *,
    rise_lo: float = 0.10,
    rise_hi: float = 0.90,
    tail_to: str = "baseline",  # "baseline" or "frac"
    tail_frac: float = 0.10,  # only used if tail_to == "frac"
    min_tail_pts: int = 8,  # need enough points to fit a line
    r2_min: float = 0.85,  # require decent log-linear fit
    noise_k: float = 2.0,  # exclude tail once below noise floor = k * MAD
    max_tau_s: float | None = None,  # e.g., 3.0 to drop >3s, or None to keep all
) -> EventKinetics:
    """
    Measure rise time (10–90%) and decay time constant τ for one event window.

    This is a refactored version of ``measure_event_kinetics`` from
    ``WL25_oasis.ipynb``.

    Parameters
    ----------
    t, y:
        1D arrays for the event window (seconds and fluorescence).
    peak_idx:
        Index (in this window) of the event peak.
    base_val:
        Baseline value for this unit/segment (same units as ``y``).

    Returns
    -------
    EventKinetics
        Dataclass with fields: ``rise_t10_90_s``, ``tau_decay_s``,
        ``A_peak``, ``ok_rise``, ``ok_tau``.
    """

    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    # amplitude at peak
    A = float(y_arr[peak_idx] - base_val)
    if not np.isfinite(A) or A <= 1e-9:
        return EventKinetics(
            rise_t10_90_s=float("nan"),
            tau_decay_s=float("nan"),
            A_peak=0.0,
            ok_rise=False,
            ok_tau=False,
        )

    # ---------- Rise (10→90%) ----------
    yn = (y_arr - base_val) / A
    tn = t_arr

    def _t_at_level(tseg: np.ndarray, yseg: np.ndarray, level: float) -> float:
        above = yseg >= level
        if not np.any(above):
            return float("nan")
        i = int(np.argmax(above))
        if i == 0:
            return float(tseg[0])
        # linear interpolation between i-1 and i
        t0, t1 = tseg[i - 1], tseg[i]
        y0, y1 = yseg[i - 1], yseg[i]
        if y1 == y0:
            return float(t1)
        return float(t0 + (level - y0) * (t1 - t0) / (y1 - y0))

    t10 = _t_at_level(tn[: peak_idx + 1], yn[: peak_idx + 1], rise_lo)
    t90 = _t_at_level(tn[: peak_idx + 1], yn[: peak_idx + 1], rise_hi)
    ok_rise = np.isfinite(t10) and np.isfinite(t90) and (t90 > t10)
    rise_t = (t90 - t10) if ok_rise else float("nan")

    # ---------- Decay τ (baseline-to-peak tail) ----------
    # Use the original (not normalized) trace for stability.
    z = np.maximum(y_arr - base_val, 0.0)

    # noise floor from pre-peak tail: MAD × 1.4826
    pre = z[max(0, peak_idx - 20) : peak_idx]
    if pre.size:
        mad = np.median(np.abs(pre - np.median(pre))) * 1.4826
    else:
        mad = 0.0
    noise_floor = noise_k * mad

    # decide tail stop value
    if tail_to == "baseline":
        stop_level = max(noise_floor, tail_frac * A)  # don't go below noise
    else:
        stop_level = max(tail_frac * A, noise_floor)

    # tail region: from peak_idx forward until z <= stop_level
    end = peak_idx + 1
    while end < len(z) and z[end] > stop_level:
        end += 1

    # need enough samples
    if (end - peak_idx) < min_tail_pts:
        return EventKinetics(
            rise_t10_90_s=rise_t,
            tau_decay_s=float("nan"),
            A_peak=A,
            ok_rise=ok_rise,
            ok_tau=False,
        )

    tt = t_arr[peak_idx:end]
    zz = z[peak_idx:end]

    # ensure strictly positive for log
    mask = zz > max(noise_floor, 1e-9)
    if np.count_nonzero(mask) < min_tail_pts:
        return EventKinetics(
            rise_t10_90_s=rise_t,
            tau_decay_s=float("nan"),
            A_peak=A,
            ok_rise=ok_rise,
            ok_tau=False,
        )

    tt = tt[mask]
    zz = zz[mask]

    try:
        p0 = (A, 1.0, base_val)
        popt, _ = curve_fit(_exp_decay, tt, zz, p0=p0, maxfev=5000)
        tau = float(popt[1])
        if max_tau_s is not None and np.isfinite(tau) and tau > max_tau_s:
            ok_tau = False
        else:
            # quick R^2 check
            yhat = _exp_decay(tt, *popt)
            ss_res = float(np.sum((zz - yhat) ** 2))
            ss_tot = float(np.sum((zz - np.mean(zz)) ** 2) + 1e-12)
            r2 = 1.0 - ss_res / ss_tot
            ok_tau = bool(r2 >= r2_min)
    except Exception:
        tau = float("nan")
        ok_tau = False

    return EventKinetics(
        rise_t10_90_s=rise_t,
        tau_decay_s=tau,
        A_peak=A,
        ok_rise=ok_rise,
        ok_tau=ok_tau,
    )
