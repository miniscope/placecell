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
    except Exception:
        # Unit not found or can't be loaded
        pass

    ax.axis("off")

    plt.tight_layout()
    return fig


def browse_place_cells(
    spike_place_csv: str | Path,
    neural_path: str | Path | None = None,
    min_speed: float = 1.0,
    min_occupancy: float = 0.1,
    bins: int = 30,
    smooth_sigma: float = 1.0,
    behavior_fps: float = 20.0,
    neural_fps: float = 20.0,
    n_shuffles: int = 100,
    random_seed: int | None = None,
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
        Path to spike_place CSV file.
    neural_path : str or Path, optional
        Path to neural data directory (for traces and max projection).
    min_speed : float
        Minimum speed threshold (default 1.0).
    min_occupancy : float
        Minimum occupancy time in seconds (default 0.1).
    bins : int
        Number of spatial bins (default 30).
    smooth_sigma : float
        Gaussian smoothing sigma (default 1.0).
    behavior_fps : float
        Behavior sampling rate (default 20.0).
    neural_fps : float
        Neural data sampling rate (default 20.0).
    n_shuffles : int
        Number of shuffles for significance test (default 100).
    random_seed : int, optional
        Random seed for reproducible shuffling. If None, results vary between runs.
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        )

    from scipy.ndimage import gaussian_filter

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Load spike data
    df = pd.read_csv(spike_place_csv)
    df_filtered = df[df["speed"] > min_speed].copy()

    # Build trajectory (deduplicated by frame)
    trajectory_df = (
        df_filtered[["beh_frame_index", "x", "y"]]
        .drop_duplicates(subset=["beh_frame_index"])
        .sort_values("beh_frame_index")
    )

    # Spatial grid
    x_edges = np.linspace(trajectory_df["x"].min(), trajectory_df["x"].max(), bins + 1)
    y_edges = np.linspace(trajectory_df["y"].min(), trajectory_df["y"].max(), bins + 1)
    time_per_frame = 1.0 / behavior_fps

    # Occupancy map
    occupancy_counts, _, _ = np.histogram2d(
        trajectory_df["x"], trajectory_df["y"], bins=[x_edges, y_edges]
    )
    occupancy_time = occupancy_counts * time_per_frame
    valid_mask = occupancy_time >= min_occupancy

    # Load traces if available
    traces = None
    trace_fps = neural_fps
    if neural_path is not None:
        neural_path = Path(neural_path)
        try:
            from placecell.analysis import load_traces

            C = load_traces(neural_path, trace_name="C")
            traces = C
        except Exception:
            pass

    # Load max projection if available
    max_proj = None
    footprints = None
    if neural_path is not None:
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
        except Exception:
            pass

    # Get unique units
    unique_units = sorted(df_filtered["unit_id"].unique())
    traj_frames = trajectory_df["beh_frame_index"].values
    n_units = len(unique_units)
    logger.info(f"Loaded {n_units} units, computing analysis...")
    unit_results = {}
    for unit_id in unique_units:
        unit_data = df_filtered[df_filtered["unit_id"] == unit_id]

        # Rate map
        spike_weights, _, _ = np.histogram2d(
            unit_data["x"],
            unit_data["y"],
            bins=[x_edges, y_edges],
            weights=unit_data["s"],
        )
        rate_map = np.zeros_like(occupancy_time)
        rate_map[valid_mask] = spike_weights[valid_mask] / occupancy_time[valid_mask]
        rate_map_smooth = gaussian_filter(rate_map, sigma=smooth_sigma)

        # Spatial Information
        total_time = np.sum(occupancy_time[valid_mask])
        total_spikes = np.sum(spike_weights[valid_mask])

        if total_time > 0 and total_spikes > 0:
            overall_lambda = total_spikes / total_time
            P_i = np.zeros_like(occupancy_time)
            P_i[valid_mask] = occupancy_time[valid_mask] / total_time

            valid_si = (rate_map > 0) & valid_mask
            if np.any(valid_si):
                si_term = (
                    P_i[valid_si]
                    * rate_map[valid_si]
                    * np.log2(rate_map[valid_si] / overall_lambda)
                )
                actual_si = float(np.sum(si_term))
            else:
                actual_si = 0.0

            # Shuffling test
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
                        P_i[valid_s]
                        * rate_shuf[valid_s]
                        * np.log2(rate_shuf[valid_s] / overall_lambda)
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

        # 2-sigma threshold for visualization
        vis_thresh = unit_data["s"].mean() + 2 * unit_data["s"].std()
        vis_data = unit_data[unit_data["s"] > vis_thresh]

        unit_results[unit_id] = {
            "rate_map": rate_map_smooth,
            "si": actual_si,
            "shuffled_sis": shuffled_sis,
            "p_val": p_val,
            "vis_data": vis_data,
            "unit_data": unit_data,
        }

    logger.info("Ready.")

    # Create figure
    from matplotlib.widgets import Slider

    fig = plt.figure(figsize=(16, 9))
    current_idx = [0]  # Use list to allow modification in nested function

    # Create axes - adjusted for slider at bottom, all top plots square
    ax1 = fig.add_axes([0.02, 0.45, 0.22, 0.50])  # Max projection
    ax2 = fig.add_axes([0.27, 0.45, 0.22, 0.50])  # Trajectory
    ax3 = fig.add_axes([0.52, 0.45, 0.22, 0.50])  # Rate map
    ax4 = fig.add_axes([0.77, 0.45, 0.22, 0.50])  # SI histogram (square)
    ax5 = fig.add_axes([0.05, 0.12, 0.90, 0.26])  # Trace
    ax_slider = fig.add_axes([0.15, 0.02, 0.70, 0.03])  # Slider

    # Create slider
    min_uid, max_uid = min(unique_units), max(unique_units)
    slider = Slider(
        ax_slider,
        f"ID (units: {min_uid}-{max_uid})",
        0,
        n_units - 1,
        valinit=0,
        valstep=1,
    )
    slider.valtext.set_visible(False)  # Hide default value text
    # Add text showing actual unit ID and position
    unit_text = ax_slider.text(
        1.02, 0.5, f"{unique_units[0]} ({1}/{n_units})", transform=ax_slider.transAxes, va="center"
    )

    def render(idx: int) -> None:
        unit_id = unique_units[idx]
        result = unit_results[unit_id]

        # Clear all axes
        for ax in [ax1, ax2, ax3, ax4, ax5]:
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
                except Exception:
                    pass
            ax1.set_title(f"Unit {unit_id}")
        else:
            ax1.text(
                0.5, 0.5, "No max projection", ha="center", va="center", transform=ax1.transAxes
            )
            ax1.set_title(f"Unit {unit_id}")
        ax1.axis("off")

        # 2. Trajectory + spikes
        vis_data = result["vis_data"]
        ax2.plot(trajectory_df["x"], trajectory_df["y"], "k-", alpha=0.1, linewidth=0.5)
        if not vis_data.empty:
            ax2.scatter(vis_data["x"], vis_data["y"], c="red", s=10, alpha=0.6)
        ax2.set_title(f"Trajectory ({len(vis_data)} spikes)")
        ax2.set_aspect("equal")
        ax2.axis("off")

        # 3. Rate map
        ax3.imshow(
            result["rate_map"].T,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect="equal",
            cmap="hot",
        )
        ax3.set_title("Rate map")
        ax3.axis("off")

        # 4. SI histogram
        ax4.hist(result["shuffled_sis"], bins=15, color="gray", alpha=0.7, edgecolor="black")
        ax4.axvline(result["si"], color="red", linestyle="--", linewidth=2)
        ax4.set_title(f"SI: {result['si']:.2f}, p={result['p_val']:.3f}")
        ax4.set_xlabel("SI (bits/s)")
        ax4.set_ylabel("Count")
        ax4.set_aspect("auto")
        ax4.set_box_aspect(1)  # Square

        # 5. Trace
        if traces is not None:
            try:
                trace = traces.sel(unit_id=int(unit_id)).values
                t = np.arange(len(trace)) / trace_fps
                ax5.plot(t, trace, "b-", linewidth=0.5)

                if "frame" in vis_data.columns and not vis_data.empty:
                    spike_frames = vis_data["frame"].values
                    spike_times = spike_frames / trace_fps
                    spike_vals = trace[spike_frames.astype(int).clip(0, len(trace) - 1)]
                    ax5.scatter(spike_times, spike_vals, c="red", s=20, zorder=5)

                ax5.set_xlabel("Time (s)")
                ax5.set_ylabel("Fluorescence")
            except Exception:
                ax5.text(
                    0.5,
                    0.5,
                    "Could not plot trace",
                    ha="center",
                    va="center",
                    transform=ax5.transAxes,
                )
        else:
            ax5.text(0.5, 0.5, "No trace data", ha="center", va="center", transform=ax5.transAxes)

        fig.suptitle(f"Unit {unit_id} ({idx + 1}/{n_units})", fontsize=12)
        fig.canvas.draw_idle()

    def on_slider(val: float) -> None:
        idx = int(val)
        current_idx[0] = idx
        unit_text.set_text(f"{unique_units[idx]} ({idx + 1}/{n_units})")
        render(idx)

    def on_key(event: Any) -> None:
        if event.key in ["right", "d"]:
            new_idx = (current_idx[0] + 1) % n_units
            slider.set_val(new_idx)
        elif event.key in ["left", "a"]:
            new_idx = (current_idx[0] - 1) % n_units
            slider.set_val(new_idx)
        elif event.key == "q":
            plt.close(fig)

    slider.on_changed(on_slider)
    fig.canvas.mpl_connect("key_press_event", on_key)
    render(0)
    plt.show()
