"""Visualization functions for place cell data."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from mio.logging import init_logger
from tqdm import tqdm

from placecell.analysis import (
    compute_occupancy_map,
    compute_unit_analysis,
)
from placecell.io import compute_overlap_time_range, load_behavior_data, load_neural_data

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


def plot_trajectory(
    csv_path: str | Path,
    bodypart: str,
    ax: "Axes | None" = None,
    x_col: str = "x",
    y_col: str = "y",
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
    x_col : str
        Coordinate column name for x-axis (default "x").
    y_col : str
        Coordinate column name for y-axis (default "y").

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
        if col[1] == bodypart and col[2] == x_col:
            scorer_name = col[0]
            break

    if scorer_name is None:
        available_bodyparts = {col[1] for col in df.columns[1:]}
        raise ValueError(
            f"Bodypart '{bodypart}' not found in CSV. "
            f"Available bodyparts: {sorted(available_bodyparts)}"
        )

    # Extract x and y coordinates for the specified bodypart
    x_csv_col = (scorer_name, bodypart, x_col)
    y_csv_col = (scorer_name, bodypart, y_col)

    x = df[x_csv_col].values
    y = df[y_csv_col].values

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
    except (KeyError, IndexError, ValueError) as e:
        logger.warning(f"Failed to load footprint for unit {unit_id} from A.zarr: {e}")

    ax.axis("off")

    plt.tight_layout()
    return fig


def plot_summary_scatter(
    unit_results: dict,
    p_value_threshold: float = 0.05,
    stability_threshold: float = 0.5,
) -> "Figure":
    """Create side-by-side scatter plots: significance vs stability and SI vs Fisher Z.

    Parameters
    ----------
    unit_results : dict
        Dictionary mapping unit_id to analysis results.
    p_value_threshold : float
        Threshold for significance test (default 0.05).
    stability_threshold : float
        Threshold for stability test (default 0.5).

    Returns
    -------
    Figure
        Matplotlib figure with two scatter plots side by side.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    unit_ids = list(unit_results.keys())
    p_vals = [unit_results[uid]["p_val"] for uid in unit_ids]
    stab_corrs = [unit_results[uid]["stability_corr"] for uid in unit_ids]
    fisher_z = [unit_results[uid]["stability_z"] for uid in unit_ids]
    si_vals = [unit_results[uid]["si"] for uid in unit_ids]

    # Determine colors based on pass/fail both tests
    colors = []
    for p, s in zip(p_vals, stab_corrs):
        sig_pass = p < p_value_threshold
        stab_pass = not np.isnan(s) and s >= stability_threshold
        if sig_pass and stab_pass:
            colors.append("green")  # Both pass
        elif sig_pass and not stab_pass:
            colors.append("orange")  # Only significance passes
        elif not sig_pass and stab_pass:
            colors.append("blue")  # Only stability passes
        else:
            colors.append("red")  # Both fail

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Significance vs Stability
    ax1.scatter(p_vals, stab_corrs, c=colors, s=50, alpha=0.7, edgecolors="black", linewidths=0.5)

    # Add threshold lines
    ax1.axvline(
        p_value_threshold,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"p={p_value_threshold}",
    )
    ax1.axhline(
        stability_threshold,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label=f"r={stability_threshold}",
    )

    # Shade quadrants
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    # Top-left (sig pass, stab pass) - green
    ax1.fill_between([0, p_value_threshold], stability_threshold, ylim[1], alpha=0.1, color="green")
    # Top-right (sig fail, stab pass) - blue
    ax1.fill_between(
        [p_value_threshold, xlim[1]], stability_threshold, ylim[1], alpha=0.1, color="blue"
    )
    # Bottom-left (sig pass, stab fail) - orange
    ax1.fill_between(
        [0, p_value_threshold], ylim[0], stability_threshold, alpha=0.1, color="orange"
    )
    # Bottom-right (sig fail, stab fail) - red
    ax1.fill_between(
        [p_value_threshold, xlim[1]], ylim[0], stability_threshold, alpha=0.1, color="red"
    )

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    ax1.set_xlabel("P-value (significance test)", fontsize=12)
    ax1.set_ylabel("Correlation (stability test)", fontsize=12)
    ax1.set_title("Significance vs Stability", fontsize=12)

    # Count units in each quadrant
    n_both = sum(1 for c in colors if c == "green")
    n_sig_only = sum(1 for c in colors if c == "orange")
    n_stab_only = sum(1 for c in colors if c == "blue")
    n_neither = sum(1 for c in colors if c == "red")

    # Add legend with counts
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", edgecolor="black", label=f"Both pass: {n_both}"),
        Patch(facecolor="orange", edgecolor="black", label=f"Sig only: {n_sig_only}"),
        Patch(facecolor="blue", edgecolor="black", label=f"Stab only: {n_stab_only}"),
        Patch(facecolor="red", edgecolor="black", label=f"Neither: {n_neither}"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=10)

    # Right plot: SI vs Fisher Z (same colors as left plot)
    ax2.scatter(
        si_vals,
        fisher_z,
        s=50,
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
        c=colors,
    )

    # Linear regression
    si_arr = np.array(si_vals)
    z_arr = np.array(fisher_z)
    # Filter out NaN values for regression
    valid_mask = ~(np.isnan(si_arr) | np.isnan(z_arr))
    si_valid = si_arr[valid_mask]
    z_valid = z_arr[valid_mask]

    if len(si_valid) > 1:
        # Compute linear regression
        slope, intercept = np.polyfit(si_valid, z_valid, 1)
        # Compute R²
        y_pred = slope * si_valid + intercept
        ss_res = np.sum((z_valid - y_pred) ** 2)
        ss_tot = np.sum((z_valid - np.mean(z_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Plot regression line
        x_line = np.array([si_valid.min(), si_valid.max()])
        y_line = slope * x_line + intercept
        ax2.plot(
            x_line,
            y_line,
            color="red",
            linestyle="-",
            linewidth=2,
            label=f"$R^2$ = {r_squared:.3f}",
        )
        ax2.legend(loc="upper right", fontsize=10)

    ax2.set_xlabel("Spatial Information (bits/s)", fontsize=12)
    ax2.set_ylabel("Fisher Z score (stability)", fontsize=12)
    ax2.set_title("Spatial Information vs Stability (Fisher Z)", fontsize=12)

    # Add grid for readability
    ax2.grid(True, alpha=0.3, linestyle="--")

    # Add zero line for Fisher Z reference
    ax2.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    plt.tight_layout()
    return fig


def browse_place_cells(
    event_place_csv: str | Path,
    behavior_position: str | Path,
    behavior_timestamp: str | Path,
    bodypart: str,
    neural_path: str | Path | None = None,
    neural_timestamp: str | Path | None = None,
    event_index_csv: str | Path | None = None,
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
    event_threshold_sigma: float = 2.0,
    p_value_threshold: float | None = None,
    stability_threshold: float = 0.5,
    x_col: str = "x",
    y_col: str = "y",
) -> None:
    """
    Interactive browser for place cell analysis with keyboard navigation.

    Layout:
    - Top row (4 quarters): Max projection, Trajectory+events, Rate map, SI histogram
    - Bottom row (full width): Trace

    Navigation:
    - Left/Right arrows or A/D: Previous/Next unit
    - Q: Quit

    Parameters
    ----------
    event_place_csv : str or Path
        Path to event_place CSV file (speed-filtered).
    behavior_position : str or Path
        Path to behavior position CSV file (behavior_position.csv).
    behavior_timestamp : str or Path
        Path to behavior timestamp CSV file (behavior_timestamp.csv).
    bodypart : str
        Body part name to use for trajectory (e.g. "LED").
    neural_path : str or Path, optional
        Path to neural data directory (for traces and max projection).
    neural_timestamp : str or Path, optional
        Path to neural timestamp CSV. If provided, behavior data is trimmed to the
        overlapping time range between neural and behavior recordings.
    event_index_csv : str or Path, optional
        Path to event_index CSV (all events). If provided, shows all events on trace plot (gray).
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
    event_threshold_sigma : float
        Sigma multiplier for event amplitude threshold in trajectory visualization (default 2.0).
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        )

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Load event data
    df = pd.read_csv(event_place_csv)
    df_filtered = df[df["speed"] > speed_threshold].copy()

    # Load all events from event_index if provided (for trace plot only)
    df_all_events = None
    if event_index_csv is not None:
        df_all_events = pd.read_csv(event_index_csv)

    # Compute overlap time range if neural timestamp is provided
    time_range = None
    if neural_timestamp is not None:
        time_range = compute_overlap_time_range(
            neural_timestamp=Path(neural_timestamp),
            behavior_timestamp=Path(behavior_timestamp),
        )

    # Load behavior data
    trajectory_with_speed, trajectory_df = load_behavior_data(
        behavior_position=Path(behavior_position),
        behavior_timestamp=Path(behavior_timestamp),
        bodypart=bodypart,
        speed_window_frames=speed_window_frames,
        speed_threshold=speed_threshold,
        time_range=time_range,
        x_col=x_col,
        y_col=y_col,
    )

    # Compute occupancy map
    occupancy_time, valid_mask, x_edges, y_edges = compute_occupancy_map(
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
    traces, max_proj, footprints = load_neural_data(
        neural_path=Path(neural_path) if neural_path else None,
        trace_name=trace_name,
    )
    trace_fps = neural_fps

    # Compute analysis for each unit
    unique_units = sorted(df_filtered["unit_id"].unique())
    n_units = len(unique_units)
    logger.info(f"Loaded {n_units} units, computing analysis...")

    unit_results = {}
    for unit_id in tqdm(unique_units, desc="Computing unit analysis", unit="unit"):
        # Core analysis from analysis module
        result = compute_unit_analysis(
            unit_id=unit_id,
            df_filtered=df_filtered,
            trajectory_df=trajectory_df,
            occupancy_time=occupancy_time,
            valid_mask=valid_mask,
            x_edges=x_edges,
            y_edges=y_edges,
            activity_sigma=activity_sigma,
            event_threshold_sigma=event_threshold_sigma,
            n_shuffles=n_shuffles,
            behavior_fps=behavior_fps,
            min_occupancy=min_occupancy,
            stability_threshold=stability_threshold,
        )

        # Visualization-specific: events above threshold for this unit
        vis_data_above = result["events_above_threshold"]

        # Visualization-specific: below-threshold events from all events (non-speed-filtered)
        vis_data_below = pd.DataFrame()
        if df_all_events is not None:
            unit_all_events = df_all_events[df_all_events["unit_id"] == unit_id]
            vis_data_below = unit_all_events[unit_all_events["s"] > result["vis_threshold"]]

        # Visualization-specific: trace data
        trace_data = None
        trace_times = None
        if traces is not None:
            try:
                trace_data = traces.sel(unit_id=int(unit_id)).values
                trace_times = np.arange(len(trace_data)) / trace_fps
            except (KeyError, IndexError) as e:
                logger.warning(f"Failed to load trace for unit {unit_id}: {e}")

        unit_results[unit_id] = {
            "rate_map": result["rate_map"],
            "si": result["si"],
            "shuffled_sis": result["shuffled_sis"],
            "p_val": result["p_val"],
            "stability_corr": result["stability_corr"],
            "stability_z": result["stability_z"],
            "rate_map_first": result["rate_map_first"],
            "rate_map_second": result["rate_map_second"],
            "vis_data_above": vis_data_above,
            "vis_data_below": vis_data_below,
            "unit_data": result["unit_data"],
            "trace_data": trace_data,
            "trace_times": trace_times,
        }

    # p_value_threshold is used to determine pass/fail status (not for filtering)
    # All units are shown, with significance test result displayed

    # Show summary scatter plots (side by side)
    threshold_p = p_value_threshold if p_value_threshold is not None else 0.05
    fig_scatter = plot_summary_scatter(
        unit_results,
        p_value_threshold=threshold_p,
        stability_threshold=stability_threshold,
    )
    fig_scatter.show()

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
    ax3_cbar = fig.add_axes([0.635, 0.49, 0.015, 0.315])  # Colorbar (70% height, centered)
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
                    unit_fp = footprints.sel(unit_id=unit_id).values
                    if unit_fp.max() > 0:
                        ax1.contour(
                            unit_fp, levels=[unit_fp.max() * 0.3], colors="red", linewidths=1.5
                        )
                except (KeyError, IndexError, ValueError) as e:
                    logger.warning(f"Failed to load footprint for unit {unit_id}: {e}")
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

        # Plot above-threshold spikes with alpha proportional to amplitude
        if not vis_data_above.empty:
            amps = vis_data_above["s"].values
            amp_max = np.max(amps) if len(amps) > 0 and np.max(amps) > 0 else 1.0
            # Linear alpha from 0 to 1
            alphas = amps / amp_max
            ax2.scatter(
                vis_data_above["x"],
                vis_data_above["y"],
                c="red",
                s=30,
                alpha=alphas,
                zorder=2,
            )

        ax2.set_title(f"Trajectory ({len(vis_data_above)} events)")
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

            # Gray: all deconvolved events from event_index (below speed threshold)
            if df_all_events is not None:
                unit_all_events = df_all_events[df_all_events["unit_id"] == unit_id]
                has_frame = "frame" in unit_all_events.columns
                has_amp = "s" in unit_all_events.columns
                if has_frame and has_amp and not unit_all_events.empty:
                    event_frames = unit_all_events["frame"].values
                    event_amplitudes = unit_all_events["s"].values
                    event_times = event_frames / trace_fps
                    event_mask = (event_times >= t_start) & (event_times <= t_end)
                    if np.any(event_mask):
                        event_times_gray = event_times[event_mask]
                        event_amplitudes_gray = event_amplitudes[event_mask]

            # Red: deconvolved events from event_place (above speed threshold)
            vis_data_above = result["vis_data_above"]
            has_required_cols = "frame" in vis_data_above.columns and "s" in vis_data_above.columns
            if has_required_cols and not vis_data_above.empty:
                event_frames = vis_data_above["frame"].values
                event_amplitudes = vis_data_above["s"].values
                event_times = event_frames / trace_fps
                event_mask = (event_times >= t_start) & (event_times <= t_end)
                if np.any(event_mask):
                    event_times_red = event_times[event_mask]
                    event_amplitudes_red = event_amplitudes[event_mask]

            # Plot deconvolved events as vertical spikes with height proportional to amplitude
            y_min, y_max = ax5.get_ylim()
            baseline = y_min  # Start spikes from bottom

            # Combine all amplitudes for normalization
            all_amps = np.concatenate(
                [
                    event_amplitudes_gray if len(event_amplitudes_gray) > 0 else [],
                    event_amplitudes_red if len(event_amplitudes_red) > 0 else [],
                ]
            )
            amp_max = np.max(all_amps) if len(all_amps) > 0 else 1.0

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

        # Determine significance test pass/fail
        p_val = result["p_val"]
        threshold = p_value_threshold if p_value_threshold is not None else 0.05
        sig_pass = p_val < threshold
        sig_text = "pass" if sig_pass else "fail"
        sig_color = "green" if sig_pass else "red"

        # Determine stability test pass/fail
        stab_corr = result["stability_corr"]
        if np.isnan(stab_corr):
            stab_pass = None  # N/A
            stab_text = "N/A"
            stab_color = "gray"
        else:
            stab_pass = stab_corr >= stability_threshold
            stab_text = "pass" if stab_pass else "fail"
            stab_color = "green" if stab_pass else "red"

        # Clear any existing text annotations for test results
        for txt in fig.texts[:]:
            if hasattr(txt, "_is_test_status"):
                txt.remove()

        # Get event count for this unit
        n_events = len(result["unit_data"]) if not result["unit_data"].empty else 0

        # Unit info at top left (with event count)
        unit_info = fig.text(
            0.02,
            0.98,
            (
                f"Unit ID: {unit_id} ({idx + 1}/{n_units}) | N={n_events} events | "
                f"Range: {min_uid}-{max_uid} | Use ←/→ keys"
            ),
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            transform=fig.transFigure,
        )
        unit_info._is_test_status = True

        # Significance test (stacked below unit info)
        sig_label = fig.text(
            0.02,
            0.95,
            f"Significance test (p={p_val:.3f}): ",
            ha="left",
            va="top",
            fontsize=10,
            transform=fig.transFigure,
        )
        sig_label._is_test_status = True

        sig_status = fig.text(
            0.185,
            0.95,
            sig_text,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=sig_color,
            transform=fig.transFigure,
        )
        sig_status._is_test_status = True

        # Stability test (stacked below significance test)
        stab_corr_str = f"r={stab_corr:.2f}" if not np.isnan(stab_corr) else ""
        stab_label = fig.text(
            0.02,
            0.92,
            f"Stability test ({stab_corr_str}): ",
            ha="left",
            va="top",
            fontsize=10,
            transform=fig.transFigure,
        )
        stab_label._is_test_status = True

        stab_status = fig.text(
            0.175,
            0.92,
            stab_text,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=stab_color,
            transform=fig.transFigure,
        )
        stab_status._is_test_status = True

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
