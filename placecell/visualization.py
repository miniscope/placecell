"""Visualization functions for place cell analysis."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from placecell.logging import init_logger

try:
    import matplotlib.pyplot as plt

    if TYPE_CHECKING:
        from matplotlib.figure import Figure
except ImportError:
    plt = None
    Figure = None

logger = init_logger(__name__)


def plot_session_summary(summary_df: "pd.DataFrame") -> "Figure":
    """Across-session counts and proportions of place cell classifications.

    Parameters
    ----------
    summary_df:
        DataFrame with columns ``dataset``, ``n_total``, ``n_sig``,
        ``n_stable``, ``n_place_cells``.  One row per session.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    df = summary_df.copy()
    session = range(1, len(df) + 1)

    # Add proportion columns
    for col in ["n_sig", "n_stable", "n_place_cells"]:
        df[col.replace("n_", "pct_")] = (df[col] / df["n_total"] * 100).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    # Left: counts
    ax = axes[0]
    ax.plot(session, df["n_sig"], "o-", label="Significant", color="tab:orange")
    ax.plot(session, df["n_stable"], "s-", label="Stable", color="tab:blue")
    ax.plot(session, df["n_place_cells"], "D-", label="Place cells", color="tab:green")
    ax.set_ylabel("Number of units")
    ax.set_xlabel("Session")
    ax.set_title("Counts")
    ax.legend(fontsize=8)
    ax.set_xticks(list(session))
    ax.set_xticklabels(df["dataset"], rotation=45, ha="right", fontsize=7)

    # Right: proportions
    ax = axes[1]
    ax.plot(session, df["pct_sig"], "o-", label="Significant", color="tab:orange")
    ax.plot(session, df["pct_stable"], "s-", label="Stable", color="tab:blue")
    ax.plot(session, df["pct_place_cells"], "D-", label="Place cells", color="tab:green")
    ax.set_ylabel("Proportion (%)")
    ax.set_xlabel("Session")
    ax.set_title("Proportions")
    ax.legend(fontsize=8)
    ax.set_xticks(list(session))
    ax.set_xticklabels(df["dataset"], rotation=45, ha="right", fontsize=7)

    fig.tight_layout()
    return fig


def plot_summary_scatter(
    unit_results: dict,
    p_value_threshold: float = 0.05,
) -> "Figure":
    """Scatter plots: significance vs stability and SI vs Fisher Z.

    Parameters
    ----------
    unit_results:
        Dictionary mapping unit_id to analysis results.
    p_value_threshold:
        Threshold for significance test.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    unit_ids = list(unit_results.keys())
    p_vals = [unit_results[uid].p_val for uid in unit_ids]
    stab_pvals = [unit_results[uid].stability_p_val for uid in unit_ids]
    fisher_z = [unit_results[uid].stability_z for uid in unit_ids]
    si_vals = [unit_results[uid].si for uid in unit_ids]

    colors = []
    for p, sp in zip(p_vals, stab_pvals):
        sig_pass = p < p_value_threshold
        stab_pass = not np.isnan(sp) and sp < p_value_threshold
        if sig_pass and stab_pass:
            colors.append("green")
        elif sig_pass and not stab_pass:
            colors.append("orange")
        elif not sig_pass and stab_pass:
            colors.append("blue")
        else:
            colors.append("red")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.scatter(p_vals, stab_pvals, c=colors, s=50, alpha=0.7, edgecolors="black", linewidths=0.5)
    ax1.axvline(
        p_value_threshold,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"p={p_value_threshold}",
    )
    ax1.axhline(
        p_value_threshold,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label=f"stab p={p_value_threshold}",
    )

    ax1.set_xlabel("P-value (significance test)", fontsize=12)
    ax1.set_ylabel("P-value (stability test)", fontsize=12)
    ax1.set_title("Significance vs Stability", fontsize=12)

    n_both = sum(1 for c in colors if c == "green")
    n_sig_only = sum(1 for c in colors if c == "orange")
    n_stab_only = sum(1 for c in colors if c == "blue")
    n_neither = sum(1 for c in colors if c == "red")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", edgecolor="black", label=f"Both pass: {n_both}"),
        Patch(facecolor="orange", edgecolor="black", label=f"Sig only: {n_sig_only}"),
        Patch(facecolor="blue", edgecolor="black", label=f"Stab only: {n_stab_only}"),
        Patch(facecolor="red", edgecolor="black", label=f"Neither: {n_neither}"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax2.scatter(si_vals, fisher_z, s=50, alpha=0.7, edgecolors="black", linewidths=0.5, c=colors)

    si_arr = np.array(si_vals)
    z_arr = np.array(fisher_z)
    valid_mask = ~(np.isnan(si_arr) | np.isnan(z_arr))
    si_valid = si_arr[valid_mask]
    z_valid = z_arr[valid_mask]

    if len(si_valid) > 1:
        slope, intercept = np.polyfit(si_valid, z_valid, 1)
        y_pred = slope * si_valid + intercept
        ss_res = np.sum((z_valid - y_pred) ** 2)
        ss_tot = np.sum((z_valid - np.mean(z_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

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
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    plt.tight_layout()
    return fig


def plot_diagnostics(
    unit_results: dict,
    p_value_threshold: float = 0.05,
) -> "Figure":
    """Event count diagnostics: histogram, SI vs events, p-value vs events.

    Parameters
    ----------
    unit_results:
        Dictionary mapping unit_id to analysis results.
    p_value_threshold:
        Threshold for significance test.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    uids = list(unit_results.keys())
    n_events = [len(unit_results[u].unit_data) for u in uids]
    si_vals = [unit_results[u].si for u in uids]
    p_vals = [unit_results[u].p_val for u in uids]

    fig, (ax_hist, ax_si, ax_pv) = plt.subplots(1, 3, figsize=(12, 3.5))

    ax_hist.hist(n_events, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    ax_hist.set_xlabel("Event count")
    ax_hist.set_ylabel("Units")
    ax_hist.axvline(
        np.median(n_events),
        color="red",
        linestyle="--",
        lw=1.5,
        label=f"Median: {int(np.median(n_events))}",
    )
    ax_hist.legend(fontsize=8)

    sig_mask = np.array([p < p_value_threshold for p in p_vals])
    ns_mask = ~sig_mask
    n_ev = np.array(n_events)
    si = np.array(si_vals)
    ax_si.scatter(
        n_ev[ns_mask],
        si[ns_mask],
        c="gray",
        s=20,
        alpha=0.5,
        edgecolors="black",
        linewidths=0.3,
        label=f"Not sig (p>={p_value_threshold})",
    )
    ax_si.scatter(
        n_ev[sig_mask],
        si[sig_mask],
        c="green",
        s=20,
        alpha=0.6,
        edgecolors="black",
        linewidths=0.3,
        label=f"Significant (p<{p_value_threshold})",
    )
    ax_si.set_xlabel("Event count")
    ax_si.set_ylabel("Spatial Information (bits/s)")
    ax_si.set_xscale("log")
    ax_si.legend(fontsize=7)

    ax_pv.scatter(
        n_events,
        p_vals,
        c="steelblue",
        s=20,
        alpha=0.6,
        edgecolors="black",
        linewidths=0.3,
    )
    ax_pv.axhline(
        p_value_threshold,
        color="red",
        linestyle="--",
        lw=1.5,
        label=f"p={p_value_threshold}",
    )
    ax_pv.set_xlabel("Event count")
    ax_pv.set_ylabel("P-value")
    ax_pv.set_xscale("log")
    ax_pv.legend(fontsize=8)

    fig.tight_layout()

    n_sig = int(sig_mask.sum())
    logger.info("Total units: %d", len(uids))
    logger.info(
        "Significant (p<%s): %d (%.1f%%)", p_value_threshold, n_sig, 100 * n_sig / len(uids)
    )
    logger.info(
        "Event count: median=%d, min=%d, max=%d",
        int(np.median(n_events)),
        min(n_events),
        max(n_events),
    )

    return fig


def plot_behavior_preview(
    trajectory: pd.DataFrame,
    trajectory_filtered: pd.DataFrame,
    speed_threshold: float,
    speed_unit: str = "mm/s",
) -> "Figure":
    """Raw vs filtered trajectory and speed histogram.

    Parameters
    ----------
    trajectory:
        Full trajectory with columns x, y, speed.
    trajectory_filtered:
        Speed-filtered trajectory.
    speed_threshold:
        Speed cutoff used for filtering.
    speed_unit:
        Label for speed axis (e.g. 'mm/s' or 'px/s').
    """
    fig, (ax_raw, ax_filt, ax_hist) = plt.subplots(1, 3, figsize=(10, 3.5))

    ax_raw.plot(trajectory["x"], trajectory["y"], "k-", linewidth=0.3, alpha=0.5)
    ax_raw.set_title(f"All frames ({len(trajectory)})")
    ax_raw.set_aspect("equal")
    ax_raw.axis("off")

    ax_filt.plot(
        trajectory_filtered["x"],
        trajectory_filtered["y"],
        "k-",
        linewidth=0.3,
        alpha=0.5,
    )
    ax_filt.set_title(f"Speed > {speed_threshold} {speed_unit} ({len(trajectory_filtered)})")
    ax_filt.set_aspect("equal")
    ax_filt.axis("off")

    all_speeds = trajectory["speed"].dropna()
    speed_max = np.percentile(all_speeds, 99)
    ax_hist.hist(
        all_speeds.clip(upper=speed_max),
        bins=50,
        color="gray",
        edgecolor="black",
        alpha=0.7,
    )
    ax_hist.axvline(
        speed_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {speed_threshold}",
    )
    ax_hist.set_xlabel(f"Speed ({speed_unit})")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Speed Distribution")
    ax_hist.legend()

    fig.tight_layout()
    return fig


def plot_occupancy_preview(
    trajectory_filtered: pd.DataFrame,
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> "Figure":
    """Filtered trajectory alongside occupancy heatmap.

    Parameters
    ----------
    trajectory_filtered:
        Speed-filtered trajectory.
    occupancy_time:
        Occupancy time map (bins x bins).
    valid_mask:
        Boolean mask of valid spatial bins.
    x_edges, y_edges:
        Spatial bin edges.
    """
    ext = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    fig, (ax_traj, ax_occ) = plt.subplots(1, 2, figsize=(8, 3.5))

    ax_traj.plot(
        trajectory_filtered["x"],
        trajectory_filtered["y"],
        "k-",
        alpha=0.5,
        linewidth=0.3,
    )
    if np.any(valid_mask) and not np.all(valid_mask):
        ax_traj.contour(
            valid_mask.T.astype(float),
            levels=[0.5],
            colors="white",
            linewidths=2,
            extent=ext,
            origin="lower",
        )
    ax_traj.set_title("Trajectory (filtered)")
    ax_traj.set_aspect("equal")
    ax_traj.axis("off")

    im = ax_occ.imshow(
        occupancy_time.T,
        origin="lower",
        extent=ext,
        cmap="hot",
        aspect="equal",
    )
    if np.any(valid_mask) and not np.all(valid_mask):
        ax_occ.contour(
            valid_mask.T.astype(float),
            levels=[0.5],
            colors="white",
            linewidths=2,
            extent=ext,
            origin="lower",
        )
    ax_occ.set_title(f"Occupancy ({valid_mask.sum()}/{valid_mask.size} valid bins)")
    plt.colorbar(im, ax=ax_occ, label="Time (s)")

    fig.tight_layout()
    return fig


def plot_footprints(
    max_proj: np.ndarray,
    footprints: xr.DataArray,
) -> "Figure":
    """Max projection, cell footprint contours, and overlay.

    Parameters
    ----------
    max_proj:
        Max-projection image.
    footprints:
        Spatial footprints DataArray with unit_id coordinate.
    """
    unit_ids = footprints.coords["unit_id"].values
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, (ax_mp, ax_fp, ax_ov) = plt.subplots(1, 3, figsize=(10, 3.5))

    ax_mp.imshow(max_proj, cmap="gray", aspect="equal")
    ax_mp.set_title("Max Projection")
    ax_mp.axis("off")

    ax_fp.imshow(np.zeros_like(max_proj), cmap="gray", aspect="equal")
    for i, uid in enumerate(unit_ids):
        fp = footprints.sel(unit_id=uid).values
        if fp.max() > 0:
            ax_fp.contour(
                fp,
                levels=[fp.max() * 0.3],
                colors=[colors[i % len(colors)]],
                linewidths=1,
            )
    ax_fp.set_title(f"Cell Footprints ({len(unit_ids)})")
    ax_fp.axis("off")

    ax_ov.imshow(max_proj, cmap="gray", aspect="equal")
    for i, uid in enumerate(unit_ids):
        fp = footprints.sel(unit_id=uid).values
        if fp.max() > 0:
            ax_ov.contour(
                fp,
                levels=[fp.max() * 0.3],
                colors=[colors[i % len(colors)]],
                linewidths=1,
            )
    ax_ov.set_title("Overlay")
    ax_ov.axis("off")

    fig.tight_layout()
    return fig


def plot_coverage(
    coverage_map: np.ndarray,
    n_cells_arr: np.ndarray,
    coverage_frac: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    valid_mask: np.ndarray,
    n_place_cells: int,
) -> "Figure":
    """Place field coverage map and cumulative coverage curve.

    Parameters
    ----------
    coverage_map:
        Overlap count per spatial bin.
    n_cells_arr:
        Number of place cells at each step of the cumulative curve.
    coverage_frac:
        Fraction of environment covered at each step.
    x_edges, y_edges:
        Spatial bin edges.
    valid_mask:
        Boolean mask of valid spatial bins.
    n_place_cells:
        Total number of place cells.
    """
    ext = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    fig, (ax_map, ax_curve) = plt.subplots(1, 2, figsize=(10, 4))

    im = ax_map.imshow(
        coverage_map.T,
        origin="lower",
        extent=ext,
        cmap="hot",
        aspect="equal",
    )
    if np.any(coverage_map > 0):
        ax_map.contour(
            (coverage_map > 0).T.astype(float),
            levels=[0.5],
            colors="white",
            linewidths=1.5,
            extent=ext,
            origin="lower",
        )
    plt.colorbar(im, ax=ax_map, label="Overlapping fields")
    ax_map.set_title(f"Place Field Coverage ({n_place_cells} cells)")
    ax_map.axis("off")

    ax_curve.plot(n_cells_arr, coverage_frac * 100, "k-", linewidth=2)
    ax_curve.fill_between(
        n_cells_arr,
        0,
        coverage_frac * 100,
        alpha=0.15,
        color="steelblue",
    )
    ax_curve.set_xlabel("Number of place cells (sorted by field size)")
    ax_curve.set_ylabel("Environment coverage (%)")
    ax_curve.set_title("Cumulative Coverage")
    ax_curve.set_ylim(0, 105)
    ax_curve.set_xlim(0, n_place_cells)
    ax_curve.axhline(100, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_curve.grid(True, alpha=0.3, linestyle="--")

    for pct in [90, 100]:
        idx = np.searchsorted(coverage_frac, pct / 100.0)
        if idx < len(n_cells_arr):
            ax_curve.axvline(
                n_cells_arr[idx],
                color="red",
                linestyle=":",
                linewidth=1,
                alpha=0.7,
            )
            ax_curve.text(
                n_cells_arr[idx] + 1,
                pct - 5,
                f"{pct}% at {n_cells_arr[idx]} cells",
                fontsize=8,
                color="red",
            )

    fig.tight_layout()

    n_valid = int(valid_mask.sum())
    n_covered = int(np.sum((coverage_map > 0) & valid_mask))
    logger.info("Covered: %d/%d bins (%.1f%%)", n_covered, n_valid, 100 * n_covered / n_valid)
    if np.any(coverage_map > 0):
        logger.info(
            "Max overlap: %d, mean overlap: %.1f",
            coverage_map.max(),
            coverage_map[coverage_map > 0].mean(),
        )

    return fig


def plot_arena_calibration(
    trajectory: "pd.DataFrame",
    arena_bounds: tuple[float, float, float, float],
    arena_size_mm: tuple[float, float] | None = None,
    mm_per_px: float | None = None,
    video_frame: "np.ndarray | None" = None,
) -> "Figure":
    """Plot arena calibration overlay on trajectory and optional video frame.

    Shows the arena bounding box overlaid on the raw trajectory.  If a
    video frame is provided, a second panel shows it with the same overlay.

    Parameters
    ----------
    trajectory:
        DataFrame with columns ``x``, ``y``.
    arena_bounds:
        (x_min, x_max, y_min, y_max) in pixels.
    arena_size_mm:
        (width, height) in mm.  Used for the title only.
    mm_per_px:
        mm-per-pixel scale.  Used for the title only.
    video_frame:
        RGB image array (H, W, 3) from the behavior video.
        If provided, shown alongside the trajectory.

    Returns
    -------
    Figure
    """
    import matplotlib.patches as patches

    x_min, x_max, y_min, y_max = arena_bounds

    ncols = 2 if video_frame is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    col = 0
    if video_frame is not None:
        ax = axes[col]
        ax.imshow(video_frame)
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.set_title("Behavior video frame + arena bounds")
        col += 1

    ax = axes[col]
    ax.plot(trajectory["x"], trajectory["y"], lw=0.3, alpha=0.5, color="steelblue")
    rect2 = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect2)
    ax.set_aspect("equal")
    ax.set_title("Raw trajectory + arena bounds")
    ax.invert_yaxis()

    title_parts = []
    if arena_size_mm is not None:
        title_parts.append(f"Arena: {arena_size_mm} mm")
    if mm_per_px is not None:
        title_parts.append(f"Scale: {mm_per_px:.3f} mm/px")
    if title_parts:
        fig.suptitle(" | ".join(title_parts))

    fig.tight_layout()
    return fig


def plot_preprocess_steps(
    steps: "dict[str, pd.DataFrame]",
    arena_bounds: tuple[float, float, float, float],
) -> "Figure":
    """Plot trajectory at each behavior preprocessing stage.

    Parameters
    ----------
    steps:
        Ordered dict mapping step name → DataFrame with ``x``, ``y``.
        Typically from ``ds._preprocess_steps``.
    arena_bounds:
        (x_min, x_max, y_min, y_max) in pixels.

    Returns
    -------
    Figure
    """
    import matplotlib.patches as patches

    x_min, x_max, y_min, y_max = arena_bounds
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    # Compute shared limits across all steps
    all_x = np.concatenate([df["x"].values for df in steps.values()])
    all_y = np.concatenate([df["y"].values for df in steps.values()])
    pad = 20
    xlim = (min(all_x.min(), x_min) - pad, max(all_x.max(), x_max) + pad)
    ylim = (min(all_y.min(), y_min) - pad, max(all_y.max(), y_max) + pad)

    for ax, (title, df) in zip(axes, steps.items()):
        ax.plot(df["x"], df["y"], lw=0.3, alpha=0.5, color="steelblue")
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim[1], ylim[0])  # inverted y
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    return fig
