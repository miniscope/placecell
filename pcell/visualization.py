"""Visualization functions for place cell data."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

try:
    import matplotlib.pyplot as plt

    if TYPE_CHECKING:
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure
except ImportError:
    plt = None
    Axes = None
    Figure = None


def plot_trajectory(
    csv_path: str | Path,
    bodypart: str,
    ax: "Axes | None" = None,
    linewidth: float = 0.5,
    alpha: float = 0.7,
) -> "Axes":
    """
    Plot trajectory from DeepLabCut CSV file.

    Parameters
    ----------
    csv_path : str | Path
        Path to the DLC CSV file
    bodypart : str
        Body part to plot (default: "LED")
    ax : matplotlib.Axes, optional
        Matplotlib axes to plot on
    linewidth : float
        Line width for trajectory plot (default: 0.5)
    alpha : float
        Alpha (transparency) for trajectory plot (default: 0.7)

    Returns
    -------
    matplotlib.Axes
        The axes object

    Raises
    ------
    ImportError
        If matplotlib is not installed
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required for trajectory plotting. "
            "Install it with: pip install matplotlib"
        )

    # Read CSV with multi-index header
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    # Find scorer name by searching for a column with the bodypart
    # Skip the first column which is the header row identifier
    scorer_name = None
    for col in df.columns[1:]:
        if col[1] == bodypart and col[2] == "x":
            scorer_name = col[0]
            break

    if scorer_name is None:
        available_bodyparts = {col[1] for col in df.columns[1:]}
        raise ValueError(
            f"Bodypart '{bodypart}' not found in CSV. "
            f"Available bodyparts: {sorted(available_bodyparts)}"
        )

    # Extract x and y coordinates for the specified bodypart
    x_col = (scorer_name, bodypart, "x")
    y_col = (scorer_name, bodypart, "y")

    x = df[x_col].values
    y = df[y_col].values

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(x, y, linewidth=linewidth, alpha=alpha)
    ax.set_xlabel("X position (pixels)")
    ax.set_ylabel("Y position (pixels)")
    ax.set_title(f"{bodypart} trajectory")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return ax


def plot_max_projection_with_unit_footprint(
    neural_path: Path,
    unit_id: int,
    max_proj_name: str = "max_proj",
    A_name: str = "A",
    figsize: tuple[float, float] = (6, 6),
    dpi: int = 150,
    contour_level: float = 0.3,
) -> "Figure":
    """Plot max projection with overlaid spatial footprint for a single unit.

    Parameters
    ----------
    neural_path:
        Directory containing max_proj.zarr and A.zarr files.
    unit_id:
        Unit ID to plot footprint for.
    max_proj_name:
        Name of the max projection zarr file (default: "max_proj").
    A_name:
        Name of the spatial footprint zarr file (default: "A").
    figsize:
        Figure size (width, height) in inches.
    dpi:
        Resolution for saved figures.
    contour_level:
        Contour level for footprint visualization (default: 0.3).

    Returns
    -------
    matplotlib.Figure
        The figure object with max projection and single unit footprint overlay.

    Raises
    ------
    ImportError
        If matplotlib is not installed
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required for plotting. " "Install it with: pip install matplotlib"
        )

    max_proj_path = neural_path / f"{max_proj_name}.zarr"
    A_path = neural_path / f"{A_name}.zarr"

    if not max_proj_path.exists():
        raise FileNotFoundError(f"Max projection file not found: {max_proj_path}")
    if not A_path.exists():
        raise FileNotFoundError(f"Spatial footprint file not found: {A_path}")

    # Load max projection
    max_proj_ds = xr.open_zarr(max_proj_path)
    if "max_proj" in max_proj_ds:
        max_proj = max_proj_ds["max_proj"]
    else:
        # Try to get the first data variable
        max_proj = max_proj_ds[list(max_proj_ds.data_vars)[0]]

    # Handle quantile dimension if present
    if "quantile" in max_proj.dims:
        max_proj = max_proj.isel(quantile=0)  # Use first quantile

    max_proj_data = np.asarray(max_proj.values, dtype=float)

    # Load spatial footprints
    A_ds = xr.open_zarr(A_path)
    A = A_ds[A_name] if A_name in A_ds else A_ds[list(A_ds.data_vars)[0]]

    # Create single figure with overlay
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot max projection as background
    ax.imshow(max_proj_data, cmap="gray", origin="upper", alpha=0.7)

    # Overlay footprint for the specified unit
    try:
        footprint = np.asarray(A.sel(unit_id=unit_id).values, dtype=float)
        if footprint.max() > 0:
            footprint_norm = footprint / footprint.max()
            # Use filled contour (contourf) instead of just contour lines
            ax.contourf(
                footprint_norm,
                levels=[contour_level, 1.0],
                colors=["red"],
                alpha=0.4,
            )
            # Also add a contour line for better visibility
            ax.contour(
                footprint_norm,
                levels=[contour_level],
                colors=["red"],
                linewidths=2,
                alpha=0.8,
            )
    except Exception:
        # Unit not found or can't be loaded
        pass

    ax.axis("off")

    plt.tight_layout()
    return fig
