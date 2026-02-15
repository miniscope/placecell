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
    """Summary scatter plots: p-value, SI vs Z, and density contour.

    Panel 1: Significance p-value vs stability p-value scatter.
    Panel 2: SI vs Fisher Z scatter.
    Panel 3: SI vs stability density contour (Guo et al. style) with
             place cells vs non-place cells as separate contour groups
             and marginal KDE histograms.

    Parameters
    ----------
    unit_results:
        Dictionary mapping unit_id to analysis results.
    p_value_threshold:
        Threshold for significance test.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    from matplotlib.patches import Patch
    from scipy.stats import gaussian_kde

    unit_ids = list(unit_results.keys())
    p_vals = np.array([unit_results[uid].p_val for uid in unit_ids])
    stab_pvals = np.array([unit_results[uid].stability_p_val for uid in unit_ids])
    fisher_z = np.array([unit_results[uid].stability_z for uid in unit_ids])
    si_vals = np.array([unit_results[uid].si for uid in unit_ids])

    # Classify units
    is_sig = p_vals < p_value_threshold
    is_stable = np.array(
        [not np.isnan(sp) and sp < p_value_threshold for sp in stab_pvals]
    )
    is_place_cell = is_sig & is_stable

    colors = []
    for s, st in zip(is_sig, is_stable):
        if s and st:
            colors.append("green")
        elif s:
            colors.append("orange")
        elif st:
            colors.append("blue")
        else:
            colors.append("red")

    n_both = int(np.sum(is_place_cell))
    n_sig_only = int(np.sum(is_sig & ~is_stable))
    n_stab_only = int(np.sum(~is_sig & is_stable))
    n_neither = int(np.sum(~is_sig & ~is_stable))

    # ── Figure layout: 3 panels ───────────────────────────────────
    fig = plt.figure(figsize=(15, 4.5))

    ax1 = fig.add_axes([0.04, 0.14, 0.26, 0.78])
    ax2 = fig.add_axes([0.37, 0.14, 0.26, 0.78])

    # Panel 3: density contour with marginals
    # Main scatter area + marginal axes
    left3 = 0.72
    ax3 = fig.add_axes([left3, 0.14, 0.22, 0.64])
    ax3_top = fig.add_axes([left3, 0.80, 0.22, 0.14])
    ax3_right = fig.add_axes([left3 + 0.23, 0.14, 0.04, 0.64])

    # ── Panel 1: P-value scatter ──────────────────────────────────
    ax1.scatter(
        p_vals, stab_pvals, c=colors, s=50, alpha=0.7,
        edgecolors="black", linewidths=0.5,
    )
    ax1.axvline(p_value_threshold, color="gray", linestyle="--", linewidth=1.5)
    ax1.axhline(p_value_threshold, color="gray", linestyle=":", linewidth=1.5)
    ax1.set_xlabel("P-value (significance)", fontsize=10)
    ax1.set_ylabel("P-value (stability)", fontsize=10)
    ax1.set_title("Significance vs Stability", fontsize=11)
    ax1.set_aspect("equal", adjustable="datalim")

    legend_elements = [
        Patch(facecolor="green", edgecolor="black", label=f"Both pass: {n_both}"),
        Patch(facecolor="orange", edgecolor="black", label=f"Sig only: {n_sig_only}"),
        Patch(facecolor="blue", edgecolor="black", label=f"Stab only: {n_stab_only}"),
        Patch(facecolor="red", edgecolor="black", label=f"Neither: {n_neither}"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # ── Panel 2: SI vs Fisher Z (no regression) ──────────────────
    ax2.scatter(
        si_vals, fisher_z, s=50, alpha=0.7,
        edgecolors="black", linewidths=0.5, c=colors,
    )
    ax2.set_xlabel("Spatial Information (bits/s)", fontsize=10)
    ax2.set_ylabel("Fisher Z (stability)", fontsize=10)
    ax2.set_title("SI vs Stability", fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax2.set_aspect("equal", adjustable="datalim")

    # ── Panel 3: Density contour (Guo et al. style) ──────────────
    valid = np.isfinite(si_vals) & np.isfinite(fisher_z)
    si_v = si_vals[valid]
    z_v = fisher_z[valid]
    pc_mask = is_place_cell[valid]
    npc_mask = ~pc_mask

    def _contour_group(ax_: "Axes", x: np.ndarray, y: np.ndarray,
                       color: str, label: str) -> None:
        if len(x) < 5:
            ax_.scatter(x, y, c=color, s=20, alpha=0.5, label=label)
            return
        xy = np.vstack([x, y])
        try:
            kde = gaussian_kde(xy)
        except np.linalg.LinAlgError:
            ax_.scatter(x, y, c=color, s=20, alpha=0.5, label=label)
            return

        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        pad_x = (xmax - xmin) * 0.15 or 0.5
        pad_y = (ymax - ymin) * 0.15 or 0.5
        xi = np.linspace(xmin - pad_x, xmax + pad_x, 80)
        yi = np.linspace(ymin - pad_y, ymax + pad_y, 80)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

        ax_.contourf(Xi, Yi, Zi, levels=6, cmap=None, colors=None,
                     alpha=0.0)  # invisible, just for structure
        ax_.contour(Xi, Yi, Zi, levels=6, colors=color, linewidths=0.8, alpha=0.8)
        ax_.contourf(Xi, Yi, Zi, levels=6,
                     colors=[(*plt.cm.colors.to_rgba(color)[:3], a)
                             for a in np.linspace(0.0, 0.35, 7)])

    _contour_group(ax3, si_v[npc_mask], z_v[npc_mask],
                   "darkorange", f"Non-place cells ({int(npc_mask.sum())})")
    _contour_group(ax3, si_v[pc_mask], z_v[pc_mask],
                   "green", f"Place cells ({int(pc_mask.sum())})")

    ax3.set_xlabel("Spatial Information (bits/s)", fontsize=10)
    ax3.set_ylabel("Stability score (Fisher Z)", fontsize=10)
    ax3.set_title("SI vs Stability density", fontsize=11)
    ax3.legend(fontsize=7, loc="upper right")

    # Marginal KDE: top (SI)
    ax3_top.set_xlim(ax3.get_xlim())
    if np.sum(npc_mask) >= 2:
        kde_npc = gaussian_kde(si_v[npc_mask])
        xs = np.linspace(*ax3.get_xlim(), 200)
        ax3_top.fill_between(xs, kde_npc(xs), alpha=0.3, color="darkorange")
    if np.sum(pc_mask) >= 2:
        kde_pc = gaussian_kde(si_v[pc_mask])
        xs = np.linspace(*ax3.get_xlim(), 200)
        ax3_top.fill_between(xs, kde_pc(xs), alpha=0.3, color="green")
    ax3_top.set_yticks([])
    ax3_top.set_xticks([])
    ax3_top.spines["top"].set_visible(False)
    ax3_top.spines["right"].set_visible(False)
    ax3_top.spines["left"].set_visible(False)

    # Marginal KDE: right (stability)
    ax3_right.set_ylim(ax3.get_ylim())
    if np.sum(npc_mask) >= 2:
        kde_npc_z = gaussian_kde(z_v[npc_mask])
        ys = np.linspace(*ax3.get_ylim(), 200)
        ax3_right.fill_betweenx(ys, kde_npc_z(ys), alpha=0.3, color="darkorange")
    if np.sum(pc_mask) >= 2:
        kde_pc_z = gaussian_kde(z_v[pc_mask])
        ys = np.linspace(*ax3.get_ylim(), 200)
        ax3_right.fill_betweenx(ys, kde_pc_z(ys), alpha=0.3, color="green")
    ax3_right.set_xticks([])
    ax3_right.set_yticks([])
    ax3_right.spines["top"].set_visible(False)
    ax3_right.spines["right"].set_visible(False)
    ax3_right.spines["bottom"].set_visible(False)

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
    arena_size_mm: tuple[float, float],
) -> "Figure":
    """Plot trajectory at each behavior preprocessing stage.

    All snapshots are expected to be in mm coordinates.

    Parameters
    ----------
    steps:
        Ordered dict mapping step name → DataFrame with ``x``, ``y``
        in mm.  Typically from ``ds._preprocess_steps``.
    arena_size_mm:
        (width, height) in mm.

    Returns
    -------
    Figure
    """
    import matplotlib.patches as patches

    w_mm, h_mm = arena_size_mm
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    # Compute shared limits across all steps
    all_x = np.concatenate([df["x"].values for df in steps.values()])
    all_y = np.concatenate([df["y"].values for df in steps.values()])
    pad = 20
    xlim = (min(all_x.min(), 0) - pad, max(all_x.max(), w_mm) + pad)
    ylim = (min(all_y.min(), 0) - pad, max(all_y.max(), h_mm) + pad)

    for ax, (title, df) in zip(axes, steps.items()):
        ax.plot(df["x"], df["y"], lw=0.3, alpha=0.5, color="steelblue")
        rect = patches.Rectangle(
            (0, 0),
            w_mm,
            h_mm,
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


# ── 1D / maze visualization ────────────────────────────────────────────


def plot_graph_overlay(
    graph_polylines: dict[str, list[list[float]]],
    mm_per_pixel: float,
    tube_order: list[str],
    video_frame: "np.ndarray | None" = None,
) -> "Figure":
    """Overlay behavior graph polylines on a video frame.

    Each zone's polyline is drawn in pixel coordinates on the video frame.
    Tubes in ``tube_order`` are drawn with distinct colors; other zones
    (rooms, etc.) are drawn in gray.

    Parameters
    ----------
    graph_polylines:
        Dict mapping zone name to list of [x, y] waypoints in pixels.
    mm_per_pixel:
        Scale factor (for title annotation).
    tube_order:
        Ordered list of tube zone names (drawn with distinct colors).
    video_frame:
        RGB image (H, W, 3). If None, polylines are drawn on a white
        background.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if video_frame is not None:
        ax.imshow(video_frame)
    else:
        # Determine extent from polylines
        all_pts = [pt for wps in graph_polylines.values() for pt in wps]
        if all_pts:
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            pad = 20
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
            ax.set_ylim(max(ys) + pad, min(ys) - pad)
        ax.set_facecolor("white")

    tube_set = set(tube_order)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    tube_color_map = {t: colors[i % len(colors)] for i, t in enumerate(tube_order)}

    for zone_name, waypoints in graph_polylines.items():
        xs = [p[0] for p in waypoints]
        ys = [p[1] for p in waypoints]

        if zone_name in tube_set:
            color = tube_color_map[zone_name]
            lw = 2.5
            alpha = 0.9
        else:
            color = "gray"
            lw = 1.5
            alpha = 0.5

        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, solid_capstyle="round")

        # Label at midpoint of polyline
        mid_idx = len(waypoints) // 2
        mx, my = waypoints[mid_idx]
        ax.text(
            mx, my, zone_name, fontsize=7, fontweight="bold",
            color=color, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    # Scale bar (100 mm)
    scale_mm = 100.0
    scale_px = scale_mm / mm_per_pixel
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Bottom-right corner with padding
    bar_x = xlim[1] - scale_px - 20
    bar_y = ylim[0] - 20 if ylim[0] > ylim[1] else ylim[0] + 20  # handle inverted y
    ax.plot(
        [bar_x, bar_x + scale_px], [bar_y, bar_y],
        color="white", linewidth=4, solid_capstyle="butt",
    )
    ax.plot(
        [bar_x, bar_x + scale_px], [bar_y, bar_y],
        color="black", linewidth=2, solid_capstyle="butt",
    )
    ax.text(
        bar_x + scale_px / 2, bar_y, f"{scale_mm:.0f} mm",
        ha="center", va="top", fontsize=8, fontweight="bold",
        color="black",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none"),
    )

    ax.set_title(f"Behavior graph (mm_per_pixel={mm_per_pixel:.2f})", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_rate_map_1d(
    rate_map: np.ndarray,
    edges: np.ndarray,
    tube_boundaries: list[float] | None = None,
    tube_labels: list[str] | None = None,
    title: str = "",
    ax: "Any | None" = None,
) -> "Figure":
    """Plot a 1D rate map as a filled line plot with tube boundaries.

    Parameters
    ----------
    rate_map:
        1D rate map array (n_bins,).
    edges:
        Bin edges (n_bins + 1,).
    tube_boundaries:
        Position values at tube boundaries (vertical lines).
    tube_labels:
        Labels for each tube segment.
    title:
        Plot title.
    ax:
        Optional axes to plot on.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    else:
        fig = ax.figure

    centers = 0.5 * (edges[:-1] + edges[1:])
    rm = rate_map.copy()
    valid = np.isfinite(rm)
    rm[~valid] = 0

    # Fill only valid regions so invalid bins appear as gaps
    ax.fill_between(centers, rm, where=valid, alpha=0.3, color="steelblue")
    rm_line = rate_map.copy()  # keep NaN for line gaps
    ax.plot(centers, rm_line, color="steelblue", linewidth=1.5)

    if tube_boundaries:
        for b in tube_boundaries:
            ax.axvline(b, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    if tube_labels and tube_boundaries and len(tube_labels) == len(tube_boundaries) - 1:
        for i, label_text in enumerate(tube_labels):
            mid = (tube_boundaries[i] + tube_boundaries[i + 1]) / 2
            ax.text(mid, ax.get_ylim()[1] * 0.95, label_text, ha="center", fontsize=8, alpha=0.7)

    ax.set_xlabel("1D position")
    ax.set_ylabel("Normalized rate")
    ax.set_xlim(edges[0], edges[-1])
    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


def plot_shuffle_test_1d(
    unit_results: dict,
    edges: np.ndarray,
    p_value_threshold: float = 0.05,
    tube_boundaries: "list[float] | None" = None,
    tube_labels: "list[str] | None" = None,
) -> "Figure":
    """Population rate map heatmap for all place cells (Guo et al. 2023 style).

    Each row is a place cell sorted by peak position, columns are spatial
    bins.  Invalid (low-occupancy) bins are excluded so there are no gaps.

    Parameters
    ----------
    unit_results:
        Dictionary mapping unit_id to UnitResult.
    edges:
        1D bin edges array.
    p_value_threshold:
        Threshold for classifying place cells.
    tube_boundaries:
        Tube boundary positions for vertical markers.
    tube_labels:
        Labels for each tube segment.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    # Identify place cells: significant SI AND stable
    place_cell_ids = []
    for uid, res in unit_results.items():
        is_sig = res.p_val < p_value_threshold
        is_stable = not np.isnan(res.stability_p_val) and res.stability_p_val < p_value_threshold
        if is_sig and is_stable:
            place_cell_ids.append(uid)

    if not place_cell_ids:
        fig, ax = plt.subplots(1, 1, figsize=(8, 2))
        ax.text(0.5, 0.5, "No place cells found", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Collect rate maps and sort by peak position
    centers = 0.5 * (edges[:-1] + edges[1:])
    rate_maps_all = np.array([unit_results[uid].rate_map for uid in place_cell_ids])

    # Find valid columns: bins that are finite in at least one cell
    valid_cols = np.any(np.isfinite(rate_maps_all), axis=0)
    n_excluded = int((~valid_cols).sum())
    n_total = len(valid_cols)
    logger.info(
        "Peak position map: %d/%d bins valid, %d excluded",
        int(valid_cols.sum()), n_total, n_excluded,
    )

    # Sort by peak position (using only valid columns for peak detection)
    rate_maps_valid = np.where(np.isfinite(rate_maps_all), rate_maps_all, 0.0)
    peak_positions = np.array([centers[np.argmax(rm)] for rm in rate_maps_valid])
    sort_order = np.argsort(peak_positions)
    sorted_ids = [place_cell_ids[i] for i in sort_order]

    # Build compressed heatmap: exclude invalid columns
    rate_maps_sorted = rate_maps_all[sort_order][:, valid_cols]
    rate_maps_sorted = np.where(np.isfinite(rate_maps_sorted), rate_maps_sorted, 0.0)
    valid_centers = centers[valid_cols]

    # Map tube boundaries to compressed column indices
    compressed_boundaries = []
    if tube_boundaries:
        for b in tube_boundaries:
            # Find how many valid bins are to the left of this boundary
            idx = int(np.sum(valid_centers < b))
            compressed_boundaries.append(idx)

    has_labels = tube_labels and tube_boundaries and len(tube_labels) == len(tube_boundaries) - 1

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    cmap = plt.cm.jet.copy()
    n_valid = int(valid_cols.sum())
    im = ax.imshow(
        rate_maps_sorted,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        extent=[0, n_valid, len(sorted_ids), 0],
        vmin=0,
        vmax=1,
    )
    ax.set_ylabel("Cell number")
    ax.set_xticklabels([])
    ax.set_xticks([])
    plt.colorbar(im, ax=ax, label="Normalized rate")

    if compressed_boundaries:
        for b in compressed_boundaries:
            ax.axvline(b, color="white", linestyle="--", linewidth=0.8, alpha=0.7)
    if has_labels and compressed_boundaries:
        for i, lbl in enumerate(tube_labels):
            mid = (compressed_boundaries[i] + compressed_boundaries[i + 1]) / 2
            ax.text(
                mid, 0, lbl, ha="center", va="top", fontsize=8,
                rotation=45, clip_on=False,
                transform=ax.get_xaxis_transform(),
            )

    fig.tight_layout(rect=[0, 0, 1, 1])
    return fig


def plot_occupancy_preview_1d(
    trajectory_1d_filtered: "pd.DataFrame",
    occupancy_time: np.ndarray,
    valid_mask: np.ndarray,
    edges: np.ndarray,
    trajectory_1d: "pd.DataFrame | None" = None,
    tube_boundaries: list[float] | None = None,
    tube_labels: list[str] | None = None,
) -> "Figure":
    """1D position time series and occupancy bar chart.

    Parameters
    ----------
    trajectory_1d_filtered:
        Speed-filtered 1D trajectory with pos_1d and unix_time columns.
    occupancy_time:
        1D occupancy histogram.
    valid_mask:
        Boolean mask of valid bins.
    edges:
        Bin edges.
    trajectory_1d:
        Unfiltered 1D trajectory (before speed filter). If provided,
        plotted as a light background layer under the filtered trajectory.
    tube_boundaries:
        Position values at tube boundaries.
    tube_labels:
        Labels for each tube segment.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    fig, (ax_traj, ax_occ) = plt.subplots(1, 2, figsize=(12, 3.5))

    # Left: position vs time
    pos_col = "pos_1d"
    time_col = "unix_time"

    # Unfiltered trajectory as background
    if trajectory_1d is not None and time_col in trajectory_1d.columns:
        t0 = trajectory_1d[time_col].iloc[0]
        t_all = trajectory_1d[time_col] - t0
        ax_traj.scatter(
            t_all,
            trajectory_1d[pos_col],
            s=0.5,
            alpha=0.15,
            color="lightcoral",
            label=f"All ({len(trajectory_1d)})",
            rasterized=True,
        )

    # Filtered trajectory on top
    if time_col in trajectory_1d_filtered.columns:
        t0 = (
            trajectory_1d[time_col].iloc[0]
            if trajectory_1d is not None and time_col in trajectory_1d.columns
            else trajectory_1d_filtered[time_col].iloc[0]
        )
        t = trajectory_1d_filtered[time_col] - t0
        ax_traj.scatter(
            t,
            trajectory_1d_filtered[pos_col],
            s=0.5,
            alpha=0.3,
            color="steelblue",
            label=f"Speed-filtered ({len(trajectory_1d_filtered)})",
            rasterized=True,
        )
        ax_traj.set_xlabel("Time (s)")
    else:
        ax_traj.scatter(
            range(len(trajectory_1d_filtered)),
            trajectory_1d_filtered[pos_col],
            s=0.5,
            alpha=0.3,
            color="steelblue",
        )
        ax_traj.set_xlabel("Frame")
    ax_traj.set_ylabel("1D position")
    ax_traj.legend(markerscale=8, fontsize=7, loc="upper right")
    ax_traj.set_title("Tube trajectory")

    if tube_boundaries:
        for b in tube_boundaries:
            ax_traj.axhline(b, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # Right: occupancy bar chart
    centers = 0.5 * (edges[:-1] + edges[1:])
    bar_width = edges[1] - edges[0]
    colors = ["steelblue" if v else "lightgray" for v in valid_mask]
    ax_occ.bar(centers, occupancy_time, width=bar_width, color=colors, edgecolor="none")
    ax_occ.set_xlabel("1D position")
    ax_occ.set_ylabel("Time (s)")
    ax_occ.set_title(f"Occupancy ({valid_mask.sum()}/{valid_mask.size} valid bins)")

    if tube_boundaries:
        for b in tube_boundaries:
            ax_occ.axvline(b, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    fig.tight_layout()
    return fig
