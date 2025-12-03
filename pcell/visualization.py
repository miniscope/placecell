"""Visualization functions for place cell data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_rate_map(
    rate_map: np.ndarray,
    ax: plt.Axes | None = None,
    cmap: str = "hot",
    **kwargs,
) -> plt.Axes:
    """
    Plot a firing rate map.

    Parameters
    ----------
    rate_map : np.ndarray
        Firing rate map to plot
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    cmap : str
        Colormap to use (default: "hot")
    **kwargs
        Additional arguments passed to imshow

    Returns
    -------
    plt.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(rate_map, cmap=cmap, origin="lower", **kwargs)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    plt.colorbar(im, ax=ax, label="Firing rate (Hz)")

    return ax


def plot_trajectory(
    csv_path: str | Path,
    bodypart: str = "LED",
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot trajectory from DeepLabCut CSV file.

    Parameters
    ----------
    csv_path : str | Path
        Path to the DLC CSV file
    bodypart : str
        Body part to plot (default: "LED")
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    **kwargs
        Additional arguments passed to plot

    Returns
    -------
    plt.Axes
        The axes object
    """
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

    ax.plot(x, y, **kwargs)
    ax.set_xlabel("X position (pixels)")
    ax.set_ylabel("Y position (pixels)")
    ax.set_title(f"{bodypart} trajectory")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return ax
