"""Simple interactive trace curation helpers."""

from pathlib import Path

import xarray as xr


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
    """

    zarr_path = neural_path / f"{trace_name}.zarr"
    ds_or_da = xr.open_zarr(zarr_path)

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

    return C
