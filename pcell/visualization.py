"""Visualization functions for place cell data."""

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

try:
    import matplotlib.pyplot as plt

    if TYPE_CHECKING:
        from matplotlib.axes import Axes
except ImportError:
    plt = None
    Axes = None


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

    # Find the scorer name (first non-header column's level 0)
    scorer_name = df.columns[1][0]

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
