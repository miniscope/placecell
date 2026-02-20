"""Neural data loading and deconvolution."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from placecell.logging import init_logger

logger = init_logger(__name__)


def load_calcium_traces(
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


def run_deconvolution(
    C_da: Any,
    unit_ids: list[int],
    g: tuple[float, float],
    baseline: float | str,
    penalty: float,
    s_min: float,
    progress_bar: Any = None,
) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    """Run OASIS deconvolution on calcium traces.

    Parameters
    ----------
    C_da : xarray.DataArray
        Calcium traces with dimensions (unit_id, frame).
    unit_ids : list[int]
        List of unit IDs to process.
    g : tuple[float, float]
        AR(2) coefficients for OASIS.
    baseline : float or str
        Baseline correction. Use 'pXX' for percentile (e.g., 'p10') or numeric value.
    penalty : float
        Sparsity penalty for OASIS.
    s_min : float
        Minimum event size threshold.
    progress_bar : optional
        tqdm progress bar wrapper (e.g., tqdm.notebook.tqdm).

    Returns
    -------
    good_unit_ids : list[int]
        Unit IDs that were successfully deconvolved.
    S_list : list[np.ndarray]
        Spike trains.
    """
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from oasis.oasis_methods import oasisAR2
    for w in caught:
        logger.warning(str(w.message))

    good_unit_ids: list[int] = []
    S_list: list[np.ndarray] = []

    iterator = progress_bar(unit_ids) if progress_bar else unit_ids

    for uid in iterator:
        y = np.ascontiguousarray(C_da.sel(unit_id=uid).values, dtype=np.float64)

        # Baseline correction
        if isinstance(baseline, str) and baseline.startswith("p"):
            p = float(baseline[1:])
            b = float(np.percentile(y, p))
        else:
            b = float(baseline)

        y_corrected = y - b

        try:
            c, s = oasisAR2(y_corrected, g1=g[0], g2=g[1], lam=penalty, s_min=s_min)
            good_unit_ids.append(int(uid))
            S_list.append(np.asarray(s, dtype=float))
        except Exception:
            continue

    return good_unit_ids, S_list


def build_event_index_dataframe(
    unit_ids: list[int],
    S_list: list[np.ndarray],
) -> pd.DataFrame:
    """Build event index DataFrame from spike trains.

    Parameters
    ----------
    unit_ids : list[int]
        Unit IDs corresponding to each spike train.
    S_list : list[np.ndarray]
        List of spike train arrays.

    Returns
    -------
    pd.DataFrame
        Event index with columns: unit_id, frame, s.
    """
    S_arr = np.stack(S_list, axis=0)
    event_rows = []

    for i, uid in enumerate(unit_ids):
        s_vec = S_arr[i]
        frames = np.nonzero(s_vec > 0)[0]
        for fr in frames:
            event_rows.append({"unit_id": uid, "frame": int(fr), "s": float(s_vec[fr])})

    return pd.DataFrame(event_rows)
