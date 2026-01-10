"""Filtering utilities for calcium imaging traces."""

import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt


def butter_lowpass_xr(
    C_da: xr.DataArray,
    *,
    fps: float,
    cutoff_hz: float = 2.0,
    order: int = 2,
    padlen: int | None = None,
) -> xr.DataArray:
    """
    Apply a Butterworth low-pass filter to each unit's 1D trace along ``frame``.

    This is a generalized, notebook-free version of the low-pass helper
    defined in ``WL25_20251201.ipynb``.

    Parameters
    ----------
    C_da:
        2D DataArray with dims including ``("unit_id", "frame")``.
    fps:
        Sampling rate in frames per second.
    cutoff_hz:
        Cutoff frequency of the low-pass filter in Hz.
    order:
        Filter order.
    padlen:
        Optional override for the ``padlen`` argument of :func:`scipy.signal.filtfilt`.
        If ``None`` (default), a safe value based on the trace length and filter
        order is chosen per trace.

    Returns
    -------
    xr.DataArray
        Low-pass filtered traces with the same shape/dims as ``C_da`` and
        a few filter parameters attached as attributes.
    """
    if not {"unit_id", "frame"} <= set(C_da.dims):
        raise ValueError("C_da must have dims including ('unit_id', 'frame')")

    nyq = 0.5 * float(fps)
    Wn = min(0.999, float(cutoff_hz) / nyq)  # clamp just in case
    b, a = butter(order, Wn, btype="low", analog=False)

    def _filt_1d(tr: np.ndarray) -> np.ndarray:
        tr = np.asarray(tr, dtype=np.float32)

        # Fill NaNs (if any) to avoid filtfilt errors
        if np.any(np.isnan(tr)):
            idx = np.arange(tr.size)
            good = ~np.isnan(tr)
            if good.any():
                tr = tr.copy()
                tr[~good] = np.interp(idx[~good], idx[good], tr[good])
            else:
                return tr  # all-NaN safeguard

        # Safe pad length for short traces
        L = tr.size
        pl_default = 3 * max(len(a), len(b))
        pl = min(pl_default, max(1, L - 1)) if padlen is None else min(int(padlen), max(1, L - 1))

        try:
            y = filtfilt(b, a, tr, method="pad", padlen=pl)
        except ValueError:
            # If trace is too short for chosen padlen, fall back to no filtering
            y = tr
        return y.astype(np.float32)

    out = xr.apply_ufunc(
        _filt_1d,
        C_da,
        input_core_dims=[["frame"]],
        output_core_dims=[["frame"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
    )
    return out.assign_attrs(
        {
            "filter": "butterworth_lowpass",
            "cutoff_hz": float(cutoff_hz),
            "order": int(order),
            "fps": float(fps),
        }
    )
