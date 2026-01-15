"""Visualization functions for place cell data."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from mio.logging import init_logger

try:
    import matplotlib.pyplot as plt

    if TYPE_CHECKING:
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure
except ImportError:
    plt = None
    Axes = None
    Figure = None

logger = init_logger(__name__)


def _gaussian_filter_normalized(
    data: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Apply Gaussian smoothing with adaptive normalization at boundaries.

    Uses zero-padding and normalizes by the kernel weight sum so that
    edge bins are not penalized. This seems to be the standard approach for
    place cell rate map smoothing.


    Parameters
    ----------
    data : np.ndarray
        Input 2D array to smooth.
    sigma : float
        Gaussian smoothing sigma in bins.

    Returns
    -------
    np.ndarray
        Smoothed array with normalized edges.
    """
    from scipy.ndimage import gaussian_filter

    if sigma <= 0:
        return data.copy()

    # Smooth data with zero padding
    smoothed = gaussian_filter(data, sigma=sigma, mode="constant", cval=0)
    # Smooth a mask of ones to get normalization weights
    norm = gaussian_filter(np.ones_like(data), sigma=sigma, mode="constant", cval=0)
    # Avoid division by zero
    norm[norm == 0] = 1
    return smoothed / norm


def _load_behavior_data(
    behavior_path: Path,
    bodypart: str,
    speed_window_frames: int,
    speed_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load behavior data and compute speed-filtered trajectory.

    Parameters
    ----------
    behavior_path : Path
        Path to behavior data directory.
    bodypart : str
        Body part name to use for trajectory.
    speed_window_frames : int
        Window size for speed computation.
    speed_threshold : float
        Minimum speed threshold.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (trajectory_with_speed, trajectory_filtered) - full trajectory with speed
        and speed-filtered trajectory.
    """
    from placecell.analysis import _load_behavior_xy, compute_behavior_speed

    behavior_position_path = behavior_path / "behavior_position.csv"
    behavior_timestamp_path = behavior_path / "behavior_timestamp.csv"

    if not behavior_position_path.exists():
        raise FileNotFoundError(
            f"Behavior position file not found: {behavior_position_path}. "
            "This is required for full trajectory plotting and occupancy calculation."
        )
    if not behavior_timestamp_path.exists():
        raise FileNotFoundError(
            f"Behavior timestamp file not found: {behavior_timestamp_path}. "
            "This is required for speed calculation."
        )

    full_trajectory = _load_behavior_xy(behavior_position_path, bodypart=bodypart)
    behavior_timestamps = pd.read_csv(behavior_timestamp_path)

    trajectory_with_speed = compute_behavior_speed(
        positions=full_trajectory,
        timestamps=behavior_timestamps,
        window_frames=speed_window_frames,
    )

    trajectory_filtered = trajectory_with_speed[trajectory_with_speed["speed"] >= speed_threshold]
    trajectory_filtered = trajectory_filtered.sort_values("frame_index")
    trajectory_filtered = trajectory_filtered.rename(columns={"frame_index": "beh_frame_index"})

    return trajectory_with_speed, trajectory_filtered


def _load_neural_data(
    neural_path: Path | None,
    trace_name: str,
) -> tuple[Any, np.ndarray | None, Any]:
    """
    Load neural data including traces, max projection, and footprints.

    Parameters
    ----------
    neural_path : Path or None
        Path to neural data directory.
    trace_name : str
        Name of trace zarr to load.

    Returns
    -------
    tuple
        (traces, max_proj, footprints) - xarray DataArray or None for each.
    """
    traces = None
    max_proj = None
    footprints = None

    if neural_path is None:
        return traces, max_proj, footprints

    neural_path = Path(neural_path)

    # Load traces
    try:
        from placecell.analysis import load_traces

        traces = load_traces(neural_path, trace_name=trace_name)
    except (FileNotFoundError, KeyError, ValueError):
        pass

    # Load max projection and footprints
    try:
        max_proj_path = neural_path / "max_proj.zarr"
        if max_proj_path.exists():
            max_proj_ds = xr.open_zarr(max_proj_path, consolidated=False)
            if "max_proj" in max_proj_ds:
                mp = max_proj_ds["max_proj"]
            else:
                mp = max_proj_ds[list(max_proj_ds.data_vars)[0]]
            if "quantile" in mp.dims:
                mp = mp.isel(quantile=0)
            max_proj = np.asarray(mp.values, dtype=float)

        a_path = neural_path / "A.zarr"
        if a_path.exists():
            A_ds = xr.open_zarr(a_path, consolidated=False)
            footprints = A_ds["A"] if "A" in A_ds else A_ds[list(A_ds.data_vars)[0]]
    except (FileNotFoundError, KeyError, ValueError, OSError):
        pass

    return traces, max_proj, footprints


def _compute_occupancy_map(
    trajectory_df: pd.DataFrame,
    bins: int,
    behavior_fps: float,
    occupancy_sigma: float,
    min_occupancy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute occupancy map from speed-filtered trajectory.

    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Speed-filtered trajectory with x, y columns.
    bins : int
        Number of spatial bins.
    behavior_fps : float
        Behavior sampling rate.
    occupancy_sigma : float
        Gaussian smoothing sigma for occupancy map.
    min_occupancy : float
        Minimum occupancy time in seconds.

    Returns
    -------
    tuple
        (occupancy_time, valid_mask, x_edges, y_edges)
    """
    x_edges = np.linspace(trajectory_df["x"].min(), trajectory_df["x"].max(), bins + 1)
    y_edges = np.linspace(trajectory_df["y"].min(), trajectory_df["y"].max(), bins + 1)
    time_per_frame = 1.0 / behavior_fps

    occupancy_counts, _, _ = np.histogram2d(
        trajectory_df["x"], trajectory_df["y"], bins=[x_edges, y_edges]
    )
    occupancy_time = occupancy_counts * time_per_frame

    if occupancy_sigma > 0:
        occupancy_time = _gaussian_filter_normalized(occupancy_time, sigma=occupancy_sigma)

    valid_mask = occupancy_time >= min_occupancy

    return occupancy_time, valid_mask, x_edges, y_edges


def _display_occupancy_preview(
    trajectory_df: pd.DataFrame,
    trajectory_with_speed: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    occupancy_sigma: float,
    min_occupancy: float,
    speed_threshold: float,
) -> None:
    """
    Display occupancy map preview with trajectory and speed histogram.

    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Speed-filtered trajectory.
    trajectory_with_speed : pd.DataFrame
        Full trajectory with speed.
    occupancy_time : np.ndarray
        Occupancy time map.
    valid_mask : np.ndarray
        Valid occupancy mask.
    x_edges, y_edges : np.ndarray
        Spatial bin edges.
    occupancy_sigma : float
        Gaussian smoothing sigma used.
    min_occupancy : float
        Minimum occupancy threshold.
    speed_threshold : float
        Speed threshold used.
    """
    fig_occ, axes_occ = plt.subplots(1, 3, figsize=(14, 4))

    # Left: trajectory
    axes_occ[0].plot(trajectory_df["x"], trajectory_df["y"], "k-", alpha=0.5, linewidth=0.5)
    axes_occ[0].set_title("Trajectory (filtered)")
    axes_occ[0].set_aspect("equal")
    axes_occ[0].axis("off")

    # Middle: occupancy map
    im = axes_occ[1].imshow(
        occupancy_time.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap="hot",
        aspect="equal",
    )
    axes_occ[1].contour(
        valid_mask.T.astype(float),
        levels=[0.5],
        colors="white",
        linewidths=1.5,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        origin="lower",
    )
    axes_occ[1].set_title(f"Occupancy (sigma={occupancy_sigma}, min={min_occupancy}s)")
    plt.colorbar(im, ax=axes_occ[1], label="Time (s)")

    # Right: speed histogram
    all_speeds = trajectory_with_speed["speed"].dropna()
    speed_max = np.percentile(all_speeds, 99)
    axes_occ[2].hist(
        all_speeds.clip(upper=speed_max),
        bins=50,
        color="gray",
        edgecolor="black",
        alpha=0.7,
    )
    axes_occ[2].axvline(
        speed_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {speed_threshold}",
    )
    axes_occ[2].set_xlim(0, speed_max)
    axes_occ[2].set_xlabel("Speed (px/s)")
    axes_occ[2].set_ylabel("Count")
    axes_occ[2].set_title("Speed Distribution")
    axes_occ[2].legend()

    fig_occ.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def _compute_unit_analysis(
    unit_id: int,
    df_filtered: pd.DataFrame,
    df_all_spikes: pd.DataFrame | None,
    trajectory_df: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    activity_sigma: float,
    spike_threshold_sigma: float,
    n_shuffles: int,
    traces: Any,
    trace_fps: float,
) -> dict:
    """
    Compute rate map and spatial information for a single unit.

    Parameters
    ----------
    unit_id : int
        Unit identifier.
    df_filtered : pd.DataFrame
        Speed-filtered spike data.
    df_all_spikes : pd.DataFrame or None
        All spikes data (for trace visualization).
    trajectory_df : pd.DataFrame
        Speed-filtered trajectory.
    occupancy_time : np.ndarray
        Occupancy time map.
    valid_mask : np.ndarray
        Valid occupancy mask.
    x_edges, y_edges : np.ndarray
        Spatial bin edges.
    activity_sigma : float
        Gaussian smoothing sigma for rate map.
    spike_threshold_sigma : float
        Sigma multiplier for visualization threshold.
    n_shuffles : int
        Number of shuffles for significance test.
    traces : xarray.DataArray or None
        Neural traces.
    trace_fps : float
        Trace sampling rate.

    Returns
    -------
    dict
        Unit analysis results including rate_map, si, p_val, etc.
    """
    unit_data = (
        df_filtered[df_filtered["unit_id"] == unit_id] if not df_filtered.empty else pd.DataFrame()
    )

    # Rate map
    spike_weights, _, _ = np.histogram2d(
        unit_data["x"],
        unit_data["y"],
        bins=[x_edges, y_edges],
        weights=unit_data["s"],
    )
    rate_map = np.zeros_like(occupancy_time)
    rate_map[valid_mask] = spike_weights[valid_mask] / occupancy_time[valid_mask]
    rate_map_smooth = _gaussian_filter_normalized(rate_map, sigma=activity_sigma)

    # Normalize to 0-1 range
    valid_rate_values = rate_map_smooth[valid_mask]
    if len(valid_rate_values) > 0 and np.nanmax(valid_rate_values) > 0:
        rate_map_smooth[valid_mask] = (
            rate_map_smooth[valid_mask] - np.nanmin(valid_rate_values)
        ) / (np.nanmax(valid_rate_values) - np.nanmin(valid_rate_values))
    rate_map_smooth[~valid_mask] = np.nan

    # Visualization threshold
    vis_thresh = unit_data["s"].mean() + spike_threshold_sigma * unit_data["s"].std()
    total_time = np.sum(occupancy_time[valid_mask])
    total_spikes = np.sum(spike_weights[valid_mask])

    # Spatial information calculation
    if total_time > 0 and total_spikes > 0:
        overall_lambda = total_spikes / total_time
        P_i = np.zeros_like(occupancy_time)
        P_i[valid_mask] = occupancy_time[valid_mask] / total_time

        valid_si = (rate_map > 0) & valid_mask
        if np.any(valid_si):
            si_term = (
                P_i[valid_si] * rate_map[valid_si] * np.log2(rate_map[valid_si] / overall_lambda)
            )
            actual_si = float(np.sum(si_term))
        else:
            actual_si = 0.0

        # Shuffling test
        traj_frames = trajectory_df["beh_frame_index"].values
        u_grouped = unit_data.groupby("beh_frame_index")["s"].sum()
        aligned_spikes = u_grouped.reindex(traj_frames, fill_value=0).values

        shuffled_sis = []
        for _ in range(n_shuffles):
            shift = np.random.randint(len(aligned_spikes))
            s_shuffled = np.roll(aligned_spikes, shift)

            spike_w_shuf, _, _ = np.histogram2d(
                trajectory_df["x"],
                trajectory_df["y"],
                bins=[x_edges, y_edges],
                weights=s_shuffled,
            )
            rate_shuf = np.zeros_like(occupancy_time)
            rate_shuf[valid_mask] = spike_w_shuf[valid_mask] / occupancy_time[valid_mask]

            valid_s = (rate_shuf > 0) & valid_mask
            if np.any(valid_s):
                si_shuf = np.sum(
                    P_i[valid_s] * rate_shuf[valid_s] * np.log2(rate_shuf[valid_s] / overall_lambda)
                )
            else:
                si_shuf = 0.0
            shuffled_sis.append(si_shuf)

        shuffled_sis = np.array(shuffled_sis)
        p_val = np.sum(shuffled_sis >= actual_si) / n_shuffles
    else:
        actual_si = 0.0
        shuffled_sis = np.zeros(n_shuffles)
        p_val = 1.0

    # Visualization data
    vis_data_above = unit_data[unit_data["s"] > vis_thresh]
    vis_data_below = pd.DataFrame()
    if df_all_spikes is not None:
        unit_all_spikes = df_all_spikes[df_all_spikes["unit_id"] == unit_id]
        vis_data_below = unit_all_spikes[unit_all_spikes["s"] > vis_thresh]

    # Trace data
    trace_data = None
    trace_times = None
    if traces is not None:
        try:
            trace_data = traces.sel(unit_id=int(unit_id)).values
            trace_times = np.arange(len(trace_data)) / trace_fps
        except (KeyError, IndexError):
            pass

    return {
        "rate_map": rate_map_smooth,
        "si": actual_si,
        "shuffled_sis": shuffled_sis,
        "p_val": p_val,
        "vis_data_above": vis_data_above,
        "vis_data_below": vis_data_below,
        "unit_data": unit_data,
        "trace_data": trace_data,
        "trace_times": trace_times,
    }


def plot_trajectory(
    csv_path: str | Path,
    bodypart: str,
    ax: "Axes | None" = None,
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

    ax.plot(x, y, linewidth=0.5, alpha=0.7)
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
    """
    Plot max projection with overlaid spatial footprint for a single unit.

    Parameters
    ----------
    neural_path : Path
        Directory containing max_proj.zarr and A.zarr files.
    unit_id : int
        Unit ID to plot footprint for.
    max_proj_name : str
        Name of the max projection zarr file (default: "max_proj").
    A_name : str
        Name of the spatial footprint zarr file (default: "A").
    figsize : tuple[float, float]
        Figure size (width, height) in inches.
    dpi : int
        Resolution for saved figures.
    contour_level : float
        Contour level for footprint visualization (default: 0.3).

    Returns
    -------
    Figure
        The figure object with max projection and single unit footprint overlay.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
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
    max_proj_ds = xr.open_zarr(max_proj_path, consolidated=False)
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
    A_ds = xr.open_zarr(A_path, consolidated=False)
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
    except (KeyError, IndexError, ValueError):
        # Unit not found or can't be loaded
        pass

    ax.axis("off")

    plt.tight_layout()
    return fig


def browse_place_cells(
    spike_place_csv: str | Path,
    behavior_path: str | Path,
    bodypart: str,
    neural_path: str | Path | None = None,
    spike_index_csv: str | Path | None = None,
    trace_name: str = "C",
    speed_threshold: float = 1.0,
    min_occupancy: float = 0.1,
    bins: int = 30,
    occupancy_sigma: float = 0.0,
    activity_sigma: float = 1.0,
    behavior_fps: float = 20.0,
    neural_fps: float = 20.0,
    speed_window_frames: int = 5,
    n_shuffles: int = 100,
    random_seed: int | None = None,
    trace_time_window: float = 600.0,
    spike_threshold_sigma: float = 2.0,
    p_value_threshold: float | None = None,
) -> None:
    """
    Interactive browser for place cell analysis with keyboard navigation.

    Layout:
    - Top row (4 quarters): Max projection, Trajectory+spikes, Rate map, SI histogram
    - Bottom row (full width): Trace

    Navigation:
    - Left/Right arrows or A/D: Previous/Next unit
    - Q: Quit

    Parameters
    ----------
    spike_place_csv : str or Path
        Path to spike_place CSV file (speed-filtered).
    behavior_path : str or Path
        Path to behavior data directory (must contain behavior_position.csv).
    bodypart : str
        Body part name to use for trajectory (e.g. "LED").
    neural_path : str or Path, optional
        Path to neural data directory (for traces and max projection).
    spike_index_csv : str or Path, optional
        Path to spike_index CSV (all spikes). If provided, shows all spikes on trace plot (gray).
    trace_name : str
        Name of trace zarr to load (default "C").
    speed_threshold : float
        Minimum speed threshold (default 1.0).
    min_occupancy : float
        Minimum occupancy time in seconds (default 0.1).
    bins : int
        Number of spatial bins (default 30).
    occupancy_sigma : float
        Gaussian smoothing sigma for occupancy map (default 0.0 = no smoothing).
    activity_sigma : float
        Gaussian smoothing sigma for spatial activity map (default 1.0).
    behavior_fps : float
        Behavior sampling rate (default 20.0).
    neural_fps : float
        Neural data sampling rate (default 20.0).
    n_shuffles : int
        Number of shuffles for significance test (default 100).
    trace_time_window : float
        Time window for trace display in seconds (default 600.0, i.e., 10 minutes).
    random_seed : int, optional
        Random seed for reproducible shuffling. If None, results vary between runs.
    spike_threshold_sigma : float
        Sigma multiplier for spike amplitude threshold in trajectory visualization (default 2.0).
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        )

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Load spike data
    df = pd.read_csv(spike_place_csv)
    df_filtered = df[df["speed"] > speed_threshold].copy()

    # Load all spikes from spike_index if provided (for trace plot only)
    df_all_spikes = None
    if spike_index_csv is not None:
        df_all_spikes = pd.read_csv(spike_index_csv)

    # Load behavior data
    behavior_path = Path(behavior_path)
    trajectory_with_speed, trajectory_df = _load_behavior_data(
        behavior_path=behavior_path,
        bodypart=bodypart,
        speed_window_frames=speed_window_frames,
        speed_threshold=speed_threshold,
    )

    # Compute occupancy map
    occupancy_time, valid_mask, x_edges, y_edges = _compute_occupancy_map(
        trajectory_df=trajectory_df,
        bins=bins,
        behavior_fps=behavior_fps,
        occupancy_sigma=occupancy_sigma,
        min_occupancy=min_occupancy,
    )

    # Display occupancy preview
    _display_occupancy_preview(
        trajectory_df=trajectory_df,
        trajectory_with_speed=trajectory_with_speed,
        occupancy_time=occupancy_time,
        valid_mask=valid_mask,
        x_edges=x_edges,
        y_edges=y_edges,
        occupancy_sigma=occupancy_sigma,
        min_occupancy=min_occupancy,
        speed_threshold=speed_threshold,
    )

    # Load neural data
    traces, max_proj, footprints = _load_neural_data(
        neural_path=Path(neural_path) if neural_path else None,
        trace_name=trace_name,
    )
    trace_fps = neural_fps

    # Compute analysis for each unit
    unique_units = sorted(df_filtered["unit_id"].unique())
    n_units = len(unique_units)
    logger.info(f"Loaded {n_units} units, computing analysis...")

    unit_results = {}
    for unit_id in unique_units:
        unit_results[unit_id] = _compute_unit_analysis(
            unit_id=unit_id,
            df_filtered=df_filtered,
            df_all_spikes=df_all_spikes,
            trajectory_df=trajectory_df,
            occupancy_time=occupancy_time,
            valid_mask=valid_mask,
            x_edges=x_edges,
            y_edges=y_edges,
            activity_sigma=activity_sigma,
            spike_threshold_sigma=spike_threshold_sigma,
            n_shuffles=n_shuffles,
            traces=traces,
            trace_fps=trace_fps,
        )

    # Filter units by p-value threshold if specified
    if p_value_threshold is not None and p_value_threshold < 1.0:
        filtered_units = [
            uid for uid in unique_units if unit_results[uid]["p_val"] < p_value_threshold
        ]
        if len(filtered_units) == 0:
            raise ValueError(
                f"No units found with p-value < {p_value_threshold:.3f}. "
                "Try adjusting the p_value_threshold or check your data."
            )
        unique_units = filtered_units
        n_units = len(unique_units)  # Update n_units after filtering
        logger.info(f"Filtered to {n_units} units with p-value < {p_value_threshold:.3f}")

    logger.info("Ready.")

    # Create figure
    from matplotlib.widgets import Slider

    fig = plt.figure(figsize=(18, 9))
    current_idx = [0]  # Use list to allow modification in nested function
    trace_scroll_pos = [0.0]  # Current scroll position for trace

    # Unit info at top
    min_uid, max_uid = min(unique_units), max(unique_units)

    # Create axes - unit info at top, more space for trace
    ax1 = fig.add_axes([0.03, 0.42, 0.18, 0.45])  # Max projection
    ax2 = fig.add_axes([0.25, 0.42, 0.18, 0.45])  # Trajectory
    ax3 = fig.add_axes([0.47, 0.42, 0.16, 0.45])  # Rate map (slightly smaller for colorbar)
    ax3_cbar = fig.add_axes([0.635, 0.42, 0.015, 0.45])  # Dedicated colorbar axes
    ax4 = fig.add_axes([0.74, 0.42, 0.18, 0.45])  # SI histogram (square)
    ax5 = fig.add_axes([0.05, 0.20, 0.90, 0.20])  # Trace
    ax_trace_slider = fig.add_axes(
        [0.15, 0.12, 0.70, 0.02]
    )  # Trace scrollbar (moved up to avoid x label)

    # Create trace scrollbar (will be initialized in render function)
    trace_slider = None

    def render(idx: int) -> None:
        nonlocal trace_slider
        # Clamp index to valid range
        idx = max(0, min(idx, n_units - 1))
        unit_id = unique_units[idx]
        result = unit_results[unit_id]

        # Reset trace scroll position when unit changes
        if idx != current_idx[0]:
            trace_scroll_pos[0] = 0.0
            if trace_slider is not None:
                trace_slider.set_val(0.0)

        # Clear all axes
        for ax in [ax1, ax2, ax3, ax3_cbar, ax4, ax5]:
            ax.clear()

        # 1. Max projection with neuron position
        if max_proj is not None:
            ax1.imshow(max_proj, cmap="gray", aspect="equal")
            if footprints is not None:
                try:
                    unit_fp = footprints.sel(unit_id=int(unit_id)).values
                    if unit_fp.max() > 0:
                        ax1.contour(
                            unit_fp, levels=[unit_fp.max() * 0.3], colors="red", linewidths=1.5
                        )
                except (KeyError, IndexError):
                    pass
            ax1.set_title(f"Unit {unit_id}")
        else:
            ax1.text(
                0.5, 0.5, "No max projection", ha="center", va="center", transform=ax1.transAxes
            )
            ax1.set_title(f"Unit {unit_id}")
        ax1.axis("off")

        # 2. Trajectory + spikes (only above speed threshold)
        vis_data_above = result["vis_data_above"]
        ax2.plot(trajectory_df["x"], trajectory_df["y"], "k-", alpha=1.0, linewidth=1, zorder=1)

        # Plot above-threshold spikes in red (on top of trajectory)
        if not vis_data_above.empty:
            ax2.scatter(vis_data_above["x"], vis_data_above["y"], c="red", s=10, zorder=2)

        ax2.set_title(f"Trajectory ({len(vis_data_above)} spikes)")
        ax2.set_aspect("equal")
        ax2.axis("off")

        # 3. Rate map (NaN values will appear white - bins below min_occupancy)
        rate_map_data = result["rate_map"].T
        im = ax3.imshow(
            rate_map_data,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect="equal",
            cmap="jet",
        )
        ax3.set_title("Rate map")
        ax3.axis("off")

        # Add colorbar for rate map using dedicated axes
        im.set_clim(0.0, 1.0)
        plt.colorbar(im, cax=ax3_cbar)
        ax3_cbar.set_ylabel("Norm. rate", rotation=270, labelpad=10)

        # 4. SI histogram
        ax4.hist(result["shuffled_sis"], bins=15, color="gray", alpha=0.7, edgecolor="black")
        ax4.axvline(result["si"], color="red", linestyle="--", linewidth=2)
        ax4.set_title(f"SI: {result['si']:.2f}, p={result['p_val']:.3f}")
        ax4.set_xlabel("SI (bits/s)")
        ax4.set_ylabel("Count")
        ax4.set_aspect("auto")
        ax4.set_box_aspect(1)  # Square

        # 5. Trace (scrollable window)
        if result["trace_data"] is not None and result["trace_times"] is not None:
            trace = result["trace_data"]
            t_full = result["trace_times"]

            # Calculate visible window
            t_max = t_full[-1] if len(t_full) > 0 else trace_time_window
            scroll_pos = trace_scroll_pos[0]
            t_start = max(0, scroll_pos)
            t_end = min(t_max, t_start + trace_time_window)

            # Get indices for visible window
            mask = (t_full >= t_start) & (t_full <= t_end)
            t_visible = t_full[mask]
            trace_visible = trace[mask]

            # Plot trace
            ax5.plot(t_visible, trace_visible, "b-", linewidth=0.5, label="Fluorescence")

            # Collect deconvolved event data for plotting as dots
            event_times_gray = []
            event_amplitudes_gray = []
            event_times_red = []
            event_amplitudes_red = []

            # Gray: all deconvolved events from spike_index (below speed threshold)
            if df_all_spikes is not None:
                unit_all_spikes = df_all_spikes[df_all_spikes["unit_id"] == unit_id]
                has_frame = "frame" in unit_all_spikes.columns
                has_amp = "s" in unit_all_spikes.columns
                if has_frame and has_amp and not unit_all_spikes.empty:
                    spike_frames = unit_all_spikes["frame"].values
                    spike_amplitudes = unit_all_spikes["s"].values
                    spike_times = spike_frames / trace_fps
                    spike_mask = (spike_times >= t_start) & (spike_times <= t_end)
                    if np.any(spike_mask):
                        event_times_gray = spike_times[spike_mask]
                        event_amplitudes_gray = spike_amplitudes[spike_mask]

            # Red: deconvolved events from spike_place (above speed threshold)
            vis_data_above = result["vis_data_above"]
            has_required_cols = "frame" in vis_data_above.columns and "s" in vis_data_above.columns
            if has_required_cols and not vis_data_above.empty:
                spike_frames = vis_data_above["frame"].values
                spike_amplitudes = vis_data_above["s"].values
                spike_times = spike_frames / trace_fps
                spike_mask = (spike_times >= t_start) & (spike_times <= t_end)
                if np.any(spike_mask):
                    event_times_red = spike_times[spike_mask]
                    event_amplitudes_red = spike_amplitudes[spike_mask]

            # Plot deconvolved events as vertical spikes with height proportional to amplitude
            y_min, y_max = ax5.get_ylim()
            baseline = y_min  # Start spikes from bottom

            # Combine all amplitudes for normalization
            all_amps = np.concatenate([
                event_amplitudes_gray if len(event_amplitudes_gray) > 0 else [],
                event_amplitudes_red if len(event_amplitudes_red) > 0 else [],
            ])
            if len(all_amps) > 0:
                amp_max = np.max(all_amps)
            else:
                amp_max = 1.0

            # Scale spike heights to use ~30% of the y-axis range
            y_range = y_max - y_min
            max_spike_height = y_range * 0.3

            def scale_height(amp: float) -> float:
                if amp_max > 0:
                    return (amp / amp_max) * max_spike_height
                return 0.0

            # Draw gray spikes (all events)
            if len(event_times_gray) > 0:
                for t, amp in zip(event_times_gray, event_amplitudes_gray):
                    h = scale_height(amp)
                    ax5.plot([t, t], [baseline, baseline + h], color="gray", lw=1.5)

            # Draw red spikes (speed-filtered events) on top
            if len(event_times_red) > 0:
                for t, amp in zip(event_times_red, event_amplitudes_red):
                    h = scale_height(amp)
                    ax5.plot([t, t], [baseline, baseline + h], color="red", lw=1.5)

            ax5.set_xlim(t_start, t_end)
            ax5.set_xlabel("Time (s)")
            ax5.set_ylabel(trace_name)

            # Legend
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], color="blue", linewidth=0.5, label="Fluorescence"),
            ]
            has_gray = len(event_times_gray) > 0
            has_red = len(event_times_red) > 0
            if has_gray:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="gray",
                        linewidth=1.5,
                        label=f"Deconv events (< {speed_threshold:.1f} px/s)",
                    )
                )
            if has_red:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="red",
                        linewidth=1.5,
                        label=f"Deconv events (≥ {speed_threshold:.1f} px/s)",
                    )
                )
            ax5.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.9)

            # Update trace slider range if needed
            if trace_slider is None and t_max > trace_time_window:
                trace_slider_max = max(0, t_max - trace_time_window)
                trace_slider = Slider(
                    ax_trace_slider,
                    "Time",
                    0.0,
                    trace_slider_max,
                    valinit=0.0,
                    valfmt="%.0f s",
                )

                def on_trace_slider(val: float) -> None:
                    trace_scroll_pos[0] = val
                    render(current_idx[0])

                trace_slider.on_changed(on_trace_slider)
            elif trace_slider is not None:
                if t_max > trace_time_window:
                    trace_slider_max = max(0, t_max - trace_time_window)
                    trace_slider.valmax = trace_slider_max
                    trace_slider.ax.set_xlim(0, trace_slider_max)
                    trace_slider.ax.set_visible(True)
                else:
                    trace_slider.ax.set_visible(False)
        else:
            ax5.text(0.5, 0.5, "No trace data", ha="center", va="center", transform=ax5.transAxes)
            ax5.set_xlabel("Time (s)")
            if trace_slider is not None:
                trace_slider.ax.set_visible(False)

        # Update unit info in suptitle
        fig.suptitle(
            f"Unit ID: {unit_id} ({idx + 1}/{n_units}) | Range: {min_uid}-{max_uid}"
            f" | Use ←/→ keys to navigate",
            fontsize=11,
            y=0.98,
        )
        fig.canvas.draw_idle()

    def change_unit(new_idx: int) -> None:
        """Change to a new unit index."""
        # Loop around the list
        num_units = len(unique_units)
        new_idx = new_idx % num_units
        current_idx[0] = new_idx
        render(new_idx)

    def on_key(event: Any) -> None:
        """Keyboard navigation handler."""
        num_units = len(unique_units)
        if event.key in ["right", "d"]:
            new_idx = (current_idx[0] + 1) % num_units
            change_unit(new_idx)
        elif event.key in ["left", "a"]:
            new_idx = (current_idx[0] - 1) % num_units
            change_unit(new_idx)
        elif event.key == "q":
            plt.close(fig)

    # Connect keyboard
    fig.canvas.mpl_connect("key_press_event", on_key)
    render(0)
    plt.show()
