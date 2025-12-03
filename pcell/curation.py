"""Simple interactive trace curation helpers."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def load_traces(
    minian_path: Path,
    trace_name: str = "C",
    var_name: str = "C",
) -> xr.DataArray:
    """Load traces from a Minian-style zarr store as a DataArray.

    Parameters
    ----------
    minian_path:
        Directory containing ``<trace_name>.zarr``.
    trace_name:
        Base name of the zarr group (e.g. ``"C"`` or ``"C_lp"``).
    var_name:
        If the zarr contains a Dataset, the variable to extract (default: ``"C"``).
    """

    zarr_path = minian_path / f"{trace_name}.zarr"
    ds_or_da = xr.open_zarr(zarr_path)

    if isinstance(ds_or_da, xr.Dataset):
        if var_name not in ds_or_da:
            raise KeyError(
                f"Variable {var_name!r} not found in dataset; available: {list(ds_or_da.data_vars)}"
            )
        C = ds_or_da[var_name]
    else:
        C = ds_or_da

    if "unit_id" not in C.dims or "frame" not in C.dims:
        raise ValueError(f"Expected dims ('unit_id','frame'), got {C.dims}")

    return C


def interactive_select_units(
    C: xr.DataArray,
    *,
    fps: float,
    out_file: Path,
    max_units: int | None = None,
) -> List[int]:
    """Very simple matplotlib-based trace curation.

    For each unit, shows its full trace and asks in the terminal:

        keep this unit? [y/N/q]

    Selected unit IDs are written one per line to ``out_file``.
    """

    unit_ids = list(map(int, C["unit_id"].values))
    if max_units is not None:
        unit_ids = unit_ids[:max_units]

    selected: List[int] = []

    for uid in unit_ids:
        y = np.asarray(C.sel(unit_id=uid).values, dtype=float)
        t = np.arange(len(y)) / float(fps)

        plt.figure(figsize=(10, 3))
        plt.plot(t, y, lw=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Fluorescence (a.u.)")
        plt.title(f"Unit {uid}")
        plt.tight_layout()
        plt.show(block=True)

        ans = input(f"Keep unit {uid}? [y/N/q]: ").strip().lower()
        if ans == "q":
            break
        if ans == "y":
            selected.append(uid)

        plt.close("all")

    out_file = out_file.resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for uid in selected:
            f.write(f"{uid}\n")

    return selected
