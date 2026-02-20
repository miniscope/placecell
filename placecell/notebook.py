"""Notebook utilities for interactive place cell visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from placecell.analysis import compute_place_field_mask

if TYPE_CHECKING:
    from placecell.dataset import ArenaDataset


def create_deconv_browser(
    good_unit_ids: list[int],
    C_da: Any,
    S_list: list[np.ndarray],
    neural_fps: float,
    trace_name: str = "C",
    time_window: float = 600.0,
) -> tuple[plt.Figure, widgets.VBox]:
    """Interactive browser for deconvolution results.

    Parameters
    ----------
    good_unit_ids:
        Unit IDs that were successfully deconvolved.
    C_da:
        Calcium traces DataArray with dims (unit_id, frame).
    S_list:
        Spike trains from deconvolution, one per unit.
    neural_fps:
        Neural sampling rate.
    trace_name:
        Label for y-axis.
    time_window:
        Visible time window in seconds.
    """
    n_good = len(good_unit_ids)
    max_time = C_da.sizes["frame"] / neural_fps

    fig, ax = plt.subplots(1, 1, figsize=(10, 2.5))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.layout.width = "100%"

    def _render(unit_idx: int, t_start: float) -> None:
        ax.clear()
        uid = good_unit_ids[unit_idx]
        trace = C_da.sel(unit_id=uid).values
        t_full = np.arange(len(trace)) / neural_fps
        spikes = S_list[unit_idx]

        t_end = min(max_time, t_start + time_window)
        mask = (t_full >= t_start) & (t_full <= t_end)

        ax.plot(t_full[mask], trace[mask], "b-", linewidth=0.5, alpha=0.7)

        spike_frames = np.nonzero(spikes > 0)[0]
        spike_times = spike_frames / neural_fps
        spike_mask = (spike_times >= t_start) & (spike_times <= t_end)
        if np.any(spike_mask):
            st = spike_times[spike_mask]
            sa = spikes[spike_frames[spike_mask]]
            y_min, y_max = ax.get_ylim()
            amp_max = sa.max() if sa.max() > 0 else 1.0
            max_spike_h = (y_max - y_min) * 0.3
            for t_s, a_s in zip(st, sa):
                h = (a_s / amp_max) * max_spike_h
                ax.plot([t_s, t_s], [y_min, y_min + h], color="red", lw=0.8)

        ax.set_xlim(t_start, t_end)
        ax.set_ylabel(trace_name, fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_title(
            f"Deconvolution Preview — Unit {uid} ({unit_idx + 1}/{n_good})",
            fontsize=10,
        )
        ax.legend(
            handles=[
                Line2D([0], [0], color="blue", linewidth=0.5, label="Fluorescence"),
                Line2D([0], [0], color="red", linewidth=1.5, label="Deconvolved spikes"),
            ],
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
        )
        fig.canvas.draw_idle()

    unit_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=n_good - 1,
        step=1,
        description="Unit:",
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
    )
    time_slider = widgets.FloatSlider(
        value=0,
        min=0,
        max=max(0, max_time - time_window),
        step=10,
        description="Time (s):",
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
    )
    prev_btn = widgets.Button(description="< Prev", layout=widgets.Layout(width="70px"))
    next_btn = widgets.Button(description="Next >", layout=widgets.Layout(width="70px"))

    def _on_prev(_: Any) -> None:
        unit_slider.value = (unit_slider.value - 1) % n_good

    def _on_next(_: Any) -> None:
        unit_slider.value = (unit_slider.value + 1) % n_good

    prev_btn.on_click(_on_prev)
    next_btn.on_click(_on_next)

    def _update(_: Any = None) -> None:
        _render(unit_slider.value, time_slider.value)

    unit_slider.observe(_update, names="value")
    time_slider.observe(_update, names="value")

    nav = widgets.HBox([prev_btn, unit_slider, next_btn], layout=widgets.Layout(width="100%"))
    controls = widgets.VBox([nav, time_slider], layout=widgets.Layout(width="100%"))

    _render(0, 0)
    return fig, controls


def create_unit_browser(
    unit_results: dict[int, dict],
    unique_units: list[int],
    trajectory_df: pd.DataFrame,
    df_all_events: pd.DataFrame | None,
    max_proj: np.ndarray | None,
    footprints: Any | None,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    occupancy_time: np.ndarray,
    trace_name: str,
    neural_fps: float,
    speed_threshold: float,
    p_value_threshold: float,
    trace_time_window: float = 600.0,
    place_field_threshold: float = 0.05,
    place_field_min_bins: int = 5,
    speed_unit: str = "mm/s",
) -> tuple[plt.Figure, widgets.VBox]:
    """Create interactive place cell browser widget.

    Parameters
    ----------
    unit_results : dict
        Dictionary mapping unit_id to analysis results.
    unique_units : list[int]
        Sorted list of unique unit IDs.
    trajectory_df : pd.DataFrame
        Trajectory data with x, y columns.
    df_all_events : pd.DataFrame or None
        All events (for trace visualization).
    max_proj : np.ndarray or None
        Maximum projection image.
    footprints : xarray.DataArray or None
        Unit footprints.
    x_edges, y_edges : np.ndarray
        Spatial bin edges.
    occupancy_time : np.ndarray
        Occupancy time map for normalizing event alpha.
    trace_name : str
        Name of trace for y-axis label.
    neural_fps : float
        Neural data sampling rate.
    speed_threshold : float
        Speed threshold for event filtering.
    p_value_threshold : float
        P-value threshold for significance.
    trace_time_window : float
        Time window for trace display in seconds.
    place_field_threshold : float
        Fraction of peak rate to define place field boundary for red contour.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure.
    controls : widgets.VBox
        Widget controls (sliders and buttons).
    """
    n_units = len(unique_units)

    # Create figure with 3 rows
    fig = plt.figure(figsize=(10, 8))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.layout.width = "100%"

    # Create axes - 3 rows: top (unit/traj/SI), middle (3 rate maps), bottom (trace)
    # Top row (leave space above for text annotations)
    ax1 = fig.add_axes([0, 0.65, 0.25, 0.27])  # Unit footprint
    ax2 = fig.add_axes([0.33, 0.65, 0.25, 0.27])  # Trajectory
    ax4 = fig.add_axes([0.66, 0.65, 0.25, 0.27])  # SI histogram

    # Middle row - rate maps (with better spacing)
    ax3a = fig.add_axes([0, 0.3, 0.24, 0.26])  # First half
    ax3b = fig.add_axes([0.33, 0.3, 0.24, 0.26])  # Second half
    ax3c = fig.add_axes([0.66, 0.3, 0.24, 0.26])  # Full session

    # Bottom row - trace
    ax5 = fig.add_axes([0.08, 0.05, 0.8, 0.18])

    text_annotations: list[Any] = []

    def render_unit(unit_idx: int, trace_start: float) -> None:
        nonlocal text_annotations

        unit_id = unique_units[unit_idx]
        result = unit_results[unit_id]

        for ax in [ax1, ax2, ax3a, ax3b, ax3c, ax4, ax5]:
            ax.clear()

        for txt in text_annotations:
            txt.remove()
        text_annotations = []

        # 1. Max projection
        if max_proj is not None:
            ax1.imshow(max_proj, cmap="gray", aspect="equal")
            if footprints is not None:
                try:
                    unit_fp = footprints.sel(unit_id=unit_id).values
                    if unit_fp.max() > 0:
                        ax1.contour(
                            unit_fp, levels=[unit_fp.max() * 0.3], colors="red", linewidths=1.5
                        )
                except (KeyError, IndexError, ValueError):
                    pass
            ax1.set_title(f"Unit {unit_id}", fontsize=9)
        else:
            ax1.text(0.5, 0.5, "No max projection", ha="center", va="center", fontsize=8)
            ax1.set_title(f"Unit {unit_id}", fontsize=9)
        ax1.axis("off")

        # 2. Trajectory + events
        vis_data_above = result.vis_data_above
        ax2.plot(trajectory_df["x"], trajectory_df["y"], "k-", alpha=1.0, linewidth=0.8, zorder=1)

        if not vis_data_above.empty:
            amps = vis_data_above["s"].values
            x_vals = vis_data_above["x"].values
            y_vals = vis_data_above["y"].values

            # Find spatial bin index for each event
            x_bin_idx = np.digitize(x_vals, x_edges) - 1
            y_bin_idx = np.digitize(y_vals, y_edges) - 1

            # Clip to valid bin indices
            x_bin_idx = np.clip(x_bin_idx, 0, len(x_edges) - 2)
            y_bin_idx = np.clip(y_bin_idx, 0, len(y_edges) - 2)

            # Look up occupancy at each event location
            event_occupancy = occupancy_time[x_bin_idx, y_bin_idx]

            # Normalize amplitude by occupancy (avoid division by zero)
            normalized_amps = amps / np.maximum(event_occupancy, 0.01)

            # Scale to 0-1 range
            has_amps = len(normalized_amps) > 0 and np.max(normalized_amps) > 0
            norm_max = np.max(normalized_amps) if has_amps else 1.0
            alphas = normalized_amps / norm_max

            ax2.scatter(x_vals, y_vals, c="red", s=20, alpha=alphas, zorder=2)

        ax2.set_title(f"Trajectory ({len(vis_data_above)} events)", fontsize=9)
        ax2.set_aspect("equal")
        ax2.axis("off")

        # 3. Rate maps (first half, second half, full)
        rate_map_first = result.rate_map_first
        rate_map_second = result.rate_map_second
        stab_corr = result.stability_corr

        # Plot first half
        im1 = ax3a.imshow(rate_map_first.T, origin="lower", cmap="inferno", aspect="equal")
        ax3a.set_title("1st half", fontsize=9)
        ax3a.axis("off")

        # Plot second half
        im2 = ax3b.imshow(rate_map_second.T, origin="lower", cmap="inferno", aspect="equal")
        ax3b.set_title("2nd half", fontsize=9)
        ax3b.axis("off")

        # Plot full session with red contour
        im3 = ax3c.imshow(result.rate_map.T, origin="lower", cmap="inferno", aspect="equal")
        field_mask_full = compute_place_field_mask(
            result.rate_map,
            threshold=place_field_threshold,
            min_bins=place_field_min_bins,
            shuffled_rate_p95=result.shuffled_rate_p95,
        )
        if np.any(field_mask_full):
            ax3c.contour(
                field_mask_full.T.astype(float), levels=[0.5], colors="red", linewidths=1.5
            )
        title = f"Full (r={stab_corr:.2f})" if not np.isnan(stab_corr) else "Full"
        ax3c.set_title(title, fontsize=9)
        ax3c.axis("off")

        # Set same color scale for all three
        im1.set_clim(0.0, 1.0)
        im2.set_clim(0.0, 1.0)
        im3.set_clim(0.0, 1.0)

        # 4. SI histogram
        ax4.hist(result.shuffled_sis, bins=15, color="gray", alpha=0.7, edgecolor="black")
        ax4.axvline(result.si, color="red", linestyle="--", linewidth=2)
        ax4.set_title(f"SI: {result.si:.2f}, p={result.p_val:.3f}", fontsize=9)
        ax4.set_xlabel("SI (bits/s)", fontsize=8)
        ax4.set_ylabel("Count", fontsize=8)
        ax4.tick_params(labelsize=7)
        ax4.set_box_aspect(1)

        # 5. Trace
        if result.trace_data is not None and result.trace_times is not None:
            trace = result.trace_data
            t_full = result.trace_times

            t_max = t_full[-1] if len(t_full) > 0 else trace_time_window
            t_start = max(0, trace_start)
            t_end = min(t_max, t_start + trace_time_window)

            mask = (t_full >= t_start) & (t_full <= t_end)
            ax5.plot(t_full[mask], trace[mask], "b-", linewidth=0.5)

            # Event spikes
            event_times_gray, event_amps_gray = [], []
            event_times_red, event_amps_red = [], []

            if df_all_events is not None:
                unit_all = df_all_events[df_all_events["unit_id"] == unit_id]
                if "frame" in unit_all.columns and "s" in unit_all.columns and not unit_all.empty:
                    event_t = unit_all["frame"].values / neural_fps
                    event_a = unit_all["s"].values
                    m = (event_t >= t_start) & (event_t <= t_end)
                    if np.any(m):
                        event_times_gray = event_t[m]
                        event_amps_gray = event_a[m]

            if (
                "frame" in vis_data_above.columns
                and "s" in vis_data_above.columns
                and not vis_data_above.empty
            ):
                event_t = vis_data_above["frame"].values / neural_fps
                event_a = vis_data_above["s"].values
                m = (event_t >= t_start) & (event_t <= t_end)
                if np.any(m):
                    event_times_red = event_t[m]
                    event_amps_red = event_a[m]

            y_min, y_max = ax5.get_ylim()
            baseline_y = y_min
            all_amps = np.concatenate(
                [
                    event_amps_gray if len(event_amps_gray) > 0 else [],
                    event_amps_red if len(event_amps_red) > 0 else [],
                ]
            )
            amp_max = np.max(all_amps) if len(all_amps) > 0 else 1.0
            y_range = y_max - y_min
            max_spike_height = y_range * 0.3

            def scale_h(a: float) -> float:
                return (a / amp_max) * max_spike_height if amp_max > 0 else 0

            for t, a in zip(event_times_gray, event_amps_gray):
                ax5.plot([t, t], [baseline_y, baseline_y + scale_h(a)], color="gray", lw=1)
            for t, a in zip(event_times_red, event_amps_red):
                ax5.plot([t, t], [baseline_y, baseline_y + scale_h(a)], color="red", lw=1)

            ax5.set_xlim(t_start, t_end)
            ax5.set_xlabel("Time (s)", fontsize=8)
            ax5.set_ylabel(trace_name, fontsize=8)
            ax5.tick_params(labelsize=7)

            legend_elements = [Line2D([0], [0], color="blue", linewidth=0.5, label="Fluorescence")]
            if len(event_times_gray) > 0:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="gray",
                        linewidth=1.5,
                        label=f"Events (< {speed_threshold:.0f} {speed_unit})",
                    )
                )
            if len(event_times_red) > 0:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="red",
                        linewidth=1.5,
                        label=f"Events (>= {speed_threshold:.0f} {speed_unit})",
                    )
                )
            ax5.legend(handles=legend_elements, loc="upper left", fontsize=7, framealpha=0.9)
        else:
            ax5.text(0.5, 0.5, "No trace data", ha="center", va="center", fontsize=8)

        # Status text
        n_events = len(result.unit_data) if not result.unit_data.empty else 0
        p_val = result.p_val
        stab_corr = result.stability_corr
        stab_p = result.stability_p_val

        sig_pass = p_val < p_value_threshold
        sig_text = "pass" if sig_pass else "fail"
        sig_color = "green" if sig_pass else "red"

        if np.isnan(stab_p):
            stab_text, stab_color = "N/A", "gray"
        else:
            stab_pass = stab_p < p_value_threshold
            stab_text = "pass" if stab_pass else "fail"
            stab_color = "green" if stab_pass else "red"

        txt = fig.text(
            0.02,
            0.97,
            f"Unit {unit_id} ({unit_idx + 1}/{n_units}) | N={n_events}",
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
        )
        text_annotations.append(txt)

        txt = fig.text(0.28, 0.97, f"Sig (p={p_val:.3f}): ", ha="left", va="top", fontsize=8)
        text_annotations.append(txt)
        txt = fig.text(
            0.40,
            0.97,
            sig_text,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=sig_color,
        )
        text_annotations.append(txt)

        stab_parts = []
        if not np.isnan(stab_corr):
            stab_parts.append(f"r={stab_corr:.2f}")
        if not np.isnan(stab_p):
            stab_parts.append(f"p={stab_p:.3f}")
        stab_str = ", ".join(stab_parts)
        txt = fig.text(0.48, 0.97, f"Stab ({stab_str}): ", ha="left", va="top", fontsize=8)
        text_annotations.append(txt)
        txt = fig.text(
            0.62,
            0.97,
            stab_text,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=stab_color,
        )
        text_annotations.append(txt)

        fig.canvas.draw_idle()

    # Get max trace time
    max_trace_time = 0.0
    for r in unit_results.values():
        if r.trace_times is not None and len(r.trace_times) > 0:
            max_trace_time = max(max_trace_time, r.trace_times[-1])

    # Widgets
    unit_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=n_units - 1,
        step=1,
        description="Unit:",
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
    )

    trace_slider = widgets.FloatSlider(
        value=0,
        min=0,
        max=max(0, max_trace_time - trace_time_window),
        step=10,
        description="Time (s):",
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
    )

    prev_btn = widgets.Button(description="< Prev", layout=widgets.Layout(width="70px"))
    next_btn = widgets.Button(description="Next >", layout=widgets.Layout(width="70px"))

    def on_prev(_: Any) -> None:
        unit_slider.value = (unit_slider.value - 1) % n_units

    def on_next(_: Any) -> None:
        unit_slider.value = (unit_slider.value + 1) % n_units

    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)

    def update(_: Any = None) -> None:
        render_unit(unit_slider.value, trace_slider.value)

    unit_slider.observe(update, names="value")
    trace_slider.observe(update, names="value")

    nav_box = widgets.HBox([prev_btn, unit_slider, next_btn], layout=widgets.Layout(width="100%"))
    controls = widgets.VBox([nav_box, trace_slider], layout=widgets.Layout(width="100%"))

    # Render initial unit
    render_unit(0, 0)

    return fig, controls


def browse_units(
    ds: ArenaDataset,
    unit_results: dict[int, dict] | None = None,
    place_field_threshold: float | None = None,
) -> tuple[plt.Figure, widgets.VBox]:
    """Create a unit browser from an ArenaDataset.

    Parameters
    ----------
    ds:
        Dataset with completed analysis (analyze_units must have been called).
    unit_results:
        Subset of results to browse. Defaults to ds.unit_results.
    place_field_threshold:
        Override for place field contour threshold.
    """
    results = unit_results if unit_results is not None else ds.unit_results
    scfg = ds.spatial

    return create_unit_browser(
        unit_results=results,
        unique_units=sorted(results.keys()),
        trajectory_df=ds.trajectory_filtered,
        df_all_events=ds.event_index,
        max_proj=ds.max_proj,
        footprints=ds.footprints,
        x_edges=ds.x_edges,
        y_edges=ds.y_edges,
        occupancy_time=ds.occupancy_time,
        trace_name=ds.cfg.neural.trace_name,
        neural_fps=ds.neural_fps,
        speed_threshold=ds.cfg.behavior.speed_threshold,
        p_value_threshold=scfg.p_value_threshold,
        trace_time_window=scfg.trace_time_window,
        place_field_threshold=place_field_threshold or scfg.place_field_threshold,
        place_field_min_bins=scfg.place_field_min_bins,
        speed_unit="mm/s" if ds.mm_per_px else "px/s",
    )


def create_shuffle_browser_1d(
    unit_results: dict[int, Any],
    edges: np.ndarray,
    p_value_threshold: float = 0.05,
    arm_boundaries: list[float] | None = None,
    arm_labels: list[str] | None = None,
) -> tuple[plt.Figure, widgets.VBox]:
    """Per-unit shuffle distribution browser for 1D maze analysis.

    Shows three panels per unit:
      - Rate map (1D)
      - SI shuffle histogram with observed SI line
      - Stability shuffle histogram with observed correlation line

    Parameters
    ----------
    unit_results:
        Dictionary mapping unit_id to UnitResult.
    edges:
        1D bin edges array.
    p_value_threshold:
        Threshold for significance / stability classification.
    arm_boundaries:
        Arm boundary positions for vertical markers.
    arm_labels:
        Labels for each arm segment.
    """
    sorted_ids = sorted(unit_results.keys())
    n_units = len(sorted_ids)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.layout.width = "100%"

    text_annotations: list[Any] = []

    def _render(unit_idx: int) -> None:
        nonlocal text_annotations

        uid = sorted_ids[unit_idx]
        res = unit_results[uid]

        for ax in axes:
            ax.clear()
        for txt in text_annotations:
            txt.remove()
        text_annotations = []

        # ── Panel 1: Rate map ──────────────────────────────────────
        ax = axes[0]
        rm = res.rate_map
        valid = np.isfinite(rm)
        ax.fill_between(
            centers,
            0,
            np.where(valid, rm, 0),
            alpha=0.3,
            color="steelblue",
            where=valid,
        )
        ax.plot(centers, rm, color="steelblue", linewidth=1.0)

        if arm_boundaries:
            for b in arm_boundaries:
                ax.axvline(b, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
        if arm_labels and arm_boundaries and len(arm_labels) == len(arm_boundaries) - 1:
            ymax = ax.get_ylim()[1]
            for i, lbl in enumerate(arm_labels):
                mid = (arm_boundaries[i] + arm_boundaries[i + 1]) / 2
                ax.text(
                    mid,
                    ymax * 1.02,
                    lbl,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=45,
                    clip_on=False,
                )

        ax.set_xlabel("1D position", fontsize=8)
        ax.set_ylabel("Rate (norm.)", fontsize=8)
        ax.set_title("Rate map", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")

        # ── Panel 2: SI shuffle distribution ───────────────────────
        ax = axes[1]
        if len(res.shuffled_sis) > 0:
            ax.hist(
                res.shuffled_sis,
                bins=30,
                color="gray",
                alpha=0.7,
                edgecolor="none",
                density=True,
                label="Shuffled",
            )
            pct95 = np.percentile(res.shuffled_sis, 95)
            ax.axvline(
                pct95, color="black", linewidth=1, linestyle="--", label=f"95th pctl ({pct95:.3f})"
            )
        ax.axvline(res.si, color="red", linewidth=2, label=f"Observed ({res.si:.3f})")
        ax.set_xlabel("Spatial information (bits/s)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        sig_str = "PASS" if res.p_val < p_value_threshold else "fail"
        ax.set_title(f"SI test (p={res.p_val:.3f}, {sig_str})", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)

        # ── Panel 3: Stability shuffle distribution ────────────────
        ax = axes[2]
        if len(res.shuffled_stability) > 0:
            ax.hist(
                res.shuffled_stability,
                bins=30,
                color="gray",
                alpha=0.7,
                edgecolor="none",
                density=True,
                label="Shuffled",
            )
            pct95_stab = np.percentile(res.shuffled_stability, 95)
            ax.axvline(
                pct95_stab,
                color="black",
                linewidth=1,
                linestyle="--",
                label=f"95th pctl ({pct95_stab:.3f})",
            )
        if not np.isnan(res.stability_corr):
            ax.axvline(
                res.stability_corr,
                color="blue",
                linewidth=2,
                label=f"Observed (r={res.stability_corr:.3f})",
            )

        ax.set_xlabel("Split-half correlation", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        if np.isnan(res.stability_p_val):
            stab_str = "N/A"
        elif res.stability_p_val < p_value_threshold:
            stab_str = "PASS"
        else:
            stab_str = "fail"
        stab_p_str = (
            f"p={res.stability_p_val:.3f}" if not np.isnan(res.stability_p_val) else "p=N/A"
        )
        ax.set_title(f"Stability test ({stab_p_str}, {stab_str})", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)

        # ── Status text ────────────────────────────────────────────
        is_sig = res.p_val < p_value_threshold
        is_stable = not np.isnan(res.stability_p_val) and res.stability_p_val < p_value_threshold
        if is_sig and is_stable:
            status = "PLACE CELL"
            color = "green"
        elif is_sig:
            status = "Significant only"
            color = "orange"
        else:
            status = "Not significant"
            color = "gray"

        txt = fig.text(
            0.02,
            0.98,
            f"Unit {uid} ({unit_idx + 1}/{n_units}) — {status}",
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=color,
            transform=fig.transFigure,
        )
        text_annotations.append(txt)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.canvas.draw_idle()

    # ── Widgets ────────────────────────────────────────────────────
    unit_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=max(0, n_units - 1),
        step=1,
        description="Unit:",
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
    )
    prev_btn = widgets.Button(description="< Prev", layout=widgets.Layout(width="70px"))
    next_btn = widgets.Button(description="Next >", layout=widgets.Layout(width="70px"))

    def _on_prev(_: Any) -> None:
        unit_slider.value = (unit_slider.value - 1) % n_units

    def _on_next(_: Any) -> None:
        unit_slider.value = (unit_slider.value + 1) % n_units

    prev_btn.on_click(_on_prev)
    next_btn.on_click(_on_next)

    def _update(_: Any = None) -> None:
        _render(unit_slider.value)

    unit_slider.observe(_update, names="value")

    nav = widgets.HBox([prev_btn, unit_slider, next_btn], layout=widgets.Layout(width="100%"))
    controls = widgets.VBox([nav], layout=widgets.Layout(width="100%"))

    _render(0)
    return fig, controls


def create_unit_browser_1d(
    unit_results: dict[int, Any],
    edges: np.ndarray,
    df_all_events: pd.DataFrame | None,
    trace_name: str,
    neural_fps: float,
    speed_threshold: float,
    p_value_threshold: float,
    trace_time_window: float = 600.0,
    arm_boundaries: list[float] | None = None,
    arm_labels: list[str] | None = None,
    speed_unit: str = "pos/s",
) -> tuple[plt.Figure, widgets.VBox]:
    """Interactive unit browser for 1D maze analysis.

    Layout:
      Top-left:  Overlaid rate maps (1st half, 2nd half, full)
      Top-right: SI shuffle histogram (top) and stability shuffle histogram (bottom)
      Bottom:    Calcium trace with event spikes

    Parameters
    ----------
    unit_results:
        Dictionary mapping unit_id to UnitResult.
    edges:
        1D bin edges array.
    df_all_events:
        All events DataFrame for trace overlay.
    trace_name:
        Label for y-axis on trace panel.
    neural_fps:
        Neural sampling rate.
    p_value_threshold:
        Threshold for significance / stability classification.
    trace_time_window:
        Visible time window in seconds for trace panel.
    arm_boundaries:
        Arm boundary positions for vertical markers.
    arm_labels:
        Labels for each arm segment.
    """
    sorted_ids = sorted(unit_results.keys())
    n_units = len(sorted_ids)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig = plt.figure(figsize=(14, 6))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.layout.width = "100%"

    # Layout: rate map (left), SI shuffle (top-right),
    # stability shuffle (bottom-right), trace (bottom)
    ax_rm = fig.add_axes([0.05, 0.42, 0.55, 0.45])
    ax_si_shuf = fig.add_axes([0.68, 0.65, 0.28, 0.22])
    ax_stab_shuf = fig.add_axes([0.68, 0.36, 0.28, 0.22])
    ax_trace = fig.add_axes([0.07, 0.07, 0.88, 0.22])

    text_annotations: list[Any] = []

    def _render(unit_idx: int, trace_start: float) -> None:
        nonlocal text_annotations

        uid = sorted_ids[unit_idx]
        res = unit_results[uid]

        for ax in [ax_rm, ax_si_shuf, ax_stab_shuf, ax_trace]:
            ax.clear()
        for txt in text_annotations:
            txt.remove()
        text_annotations = []

        # ── Overlaid rate maps ────────────────────────────────────
        valid_full = np.isfinite(res.rate_map)
        valid_1st = np.isfinite(res.rate_map_first)
        valid_2nd = np.isfinite(res.rate_map_second)

        ax_rm.fill_between(
            centers,
            0,
            np.where(valid_full, res.rate_map, 0),
            alpha=0.15,
            color="black",
            where=valid_full,
        )
        ax_rm.plot(centers, res.rate_map, color="black", linewidth=1.5, label="Full")
        ax_rm.plot(
            centers,
            np.where(valid_1st, res.rate_map_first, np.nan),
            color="steelblue",
            linewidth=1.0,
            alpha=0.8,
            label="1st half",
        )
        ax_rm.plot(
            centers,
            np.where(valid_2nd, res.rate_map_second, np.nan),
            color="coral",
            linewidth=1.0,
            alpha=0.8,
            label="2nd half",
        )

        if arm_boundaries:
            for b in arm_boundaries:
                ax_rm.axvline(b, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
        if arm_labels and arm_boundaries and len(arm_labels) == len(arm_boundaries) - 1:
            for i, lbl in enumerate(arm_labels):
                mid = (arm_boundaries[i] + arm_boundaries[i + 1]) / 2
                ax_rm.text(
                    mid,
                    -0.06,
                    lbl,
                    ha="center",
                    va="top",
                    fontsize=5,
                    rotation=45,
                    clip_on=False,
                    transform=ax_rm.get_xaxis_transform(),
                )

        ax_rm.set_xlim(edges[0], edges[-1])
        ax_rm.set_xticklabels([])
        ax_rm.set_ylabel("Rate (norm.)", fontsize=7)
        ax_rm.tick_params(labelsize=6)
        ax_rm.legend(fontsize=6, loc="upper right", framealpha=0.8)

        # ── SI shuffle histogram ──────────────────────────────────
        if res.shuffled_sis is not None and len(res.shuffled_sis) > 0:
            ax_si_shuf.hist(
                res.shuffled_sis,
                bins=30,
                color="gray",
                alpha=0.6,
                edgecolor="none",
            )
            ax_si_shuf.axvline(res.si, color="red", linewidth=1.5, label=f"SI={res.si:.3f}")
            ax_si_shuf.set_title(f"SI shuffle (p={res.p_val:.3f})", fontsize=8)
        else:
            ax_si_shuf.text(0.5, 0.5, "No SI shuffle", ha="center", va="center", fontsize=8)
        ax_si_shuf.tick_params(labelsize=5)
        ax_si_shuf.set_ylabel("Count", fontsize=6)

        # ── Stability shuffle histogram ───────────────────────────
        if res.shuffled_stability is not None and len(res.shuffled_stability) > 0:
            ax_stab_shuf.hist(
                res.shuffled_stability,
                bins=30,
                color="gray",
                alpha=0.6,
                edgecolor="none",
            )
            corr_val = res.stability_corr if not np.isnan(res.stability_corr) else 0
            ax_stab_shuf.axvline(
                corr_val,
                color="red",
                linewidth=1.5,
                label=f"r={corr_val:.2f}",
            )
            p_str = f"p={res.stability_p_val:.3f}" if not np.isnan(res.stability_p_val) else "p=N/A"
            ax_stab_shuf.set_title(f"Stability shuffle ({p_str})", fontsize=8)
        else:
            ax_stab_shuf.text(
                0.5,
                0.5,
                "No stability shuffle",
                ha="center",
                va="center",
                fontsize=8,
            )
        ax_stab_shuf.tick_params(labelsize=5)
        ax_stab_shuf.set_ylabel("Count", fontsize=6)

        # ── Trace with events ─────────────────────────────────────
        if res.trace_data is not None and res.trace_times is not None:
            trace = res.trace_data
            t_full = res.trace_times
            t_max = t_full[-1] if len(t_full) > 0 else trace_time_window
            t_start = max(0.0, trace_start)
            t_end = min(t_max, t_start + trace_time_window)
            mask = (t_full >= t_start) & (t_full <= t_end)
            ax_trace.plot(t_full[mask], trace[mask], "b-", linewidth=0.5)

            event_times_gray: Any = []
            event_amps_gray: Any = []
            event_times_red: Any = []
            event_amps_red: Any = []

            if df_all_events is not None:
                unit_all = df_all_events[df_all_events["unit_id"] == uid]
                if "frame" in unit_all.columns and "s" in unit_all.columns and not unit_all.empty:
                    et = unit_all["frame"].values / neural_fps
                    ea = unit_all["s"].values
                    m = (et >= t_start) & (et <= t_end)
                    if np.any(m):
                        event_times_gray = et[m]
                        event_amps_gray = ea[m]

            vis = res.vis_data_above
            if "frame" in vis.columns and "s" in vis.columns and not vis.empty:
                et = vis["frame"].values / neural_fps
                ea = vis["s"].values
                m = (et >= t_start) & (et <= t_end)
                if np.any(m):
                    event_times_red = et[m]
                    event_amps_red = ea[m]

            y_min, y_max = ax_trace.get_ylim()
            all_a = np.concatenate(
                [
                    event_amps_gray if len(event_amps_gray) > 0 else [],
                    event_amps_red if len(event_amps_red) > 0 else [],
                ]
            )
            amp_max = float(np.max(all_a)) if len(all_a) > 0 else 1.0
            max_h = (y_max - y_min) * 0.3

            def _h(a: float) -> float:
                return (a / amp_max) * max_h if amp_max > 0 else 0

            for t, a in zip(event_times_gray, event_amps_gray):
                ax_trace.plot([t, t], [y_min, y_min + _h(a)], color="gray", lw=0.8)
            for t, a in zip(event_times_red, event_amps_red):
                ax_trace.plot([t, t], [y_min, y_min + _h(a)], color="red", lw=0.8)

            ax_trace.set_xlim(t_start, t_end)
            ax_trace.set_xlabel("Time (s)", fontsize=8)
            ax_trace.set_ylabel(trace_name, fontsize=8)
            ax_trace.tick_params(labelsize=7)

            legend_el = [Line2D([0], [0], color="blue", linewidth=0.5, label="Fluorescence")]
            if len(event_times_gray) > 0:
                legend_el.append(
                    Line2D(
                        [0],
                        [0],
                        color="gray",
                        linewidth=1.5,
                        label=f"Events (< {speed_threshold:.1f} {speed_unit})",
                    )
                )
            if len(event_times_red) > 0:
                legend_el.append(
                    Line2D(
                        [0],
                        [0],
                        color="red",
                        linewidth=1.5,
                        label=f"Events (>= {speed_threshold:.1f} {speed_unit})",
                    )
                )
            ax_trace.legend(handles=legend_el, loc="upper left", fontsize=6, framealpha=0.9)
        else:
            ax_trace.text(0.5, 0.5, "No trace data", ha="center", va="center", fontsize=8)

        # ── Status text ───────────────────────────────────────────
        is_sig = res.p_val < p_value_threshold
        is_stable = not np.isnan(res.stability_p_val) and res.stability_p_val < p_value_threshold
        n_events = len(res.unit_data) if not res.unit_data.empty else 0

        sig_color = "green" if is_sig else "red"
        sig_text = "pass" if is_sig else "fail"
        if np.isnan(res.stability_p_val):
            stab_color, stab_text = "gray", "N/A"
        else:
            stab_color = "green" if is_stable else "red"
            stab_text = "pass" if is_stable else "fail"

        txt = fig.text(
            0.02,
            0.97,
            f"Unit {uid} ({unit_idx + 1}/{n_units}) | N={n_events}",
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
        )
        text_annotations.append(txt)
        txt = fig.text(
            0.30,
            0.97,
            f"SI={res.si:.3f}, p={res.p_val:.3f}: ",
            ha="left",
            va="top",
            fontsize=8,
        )
        text_annotations.append(txt)
        txt = fig.text(
            0.48,
            0.97,
            sig_text,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=sig_color,
        )
        text_annotations.append(txt)

        stab_parts = []
        if not np.isnan(res.stability_corr):
            stab_parts.append(f"r={res.stability_corr:.2f}")
        if not np.isnan(res.stability_p_val):
            stab_parts.append(f"p={res.stability_p_val:.3f}")
        txt = fig.text(
            0.54,
            0.97,
            f"Stab ({', '.join(stab_parts)}): ",
            ha="left",
            va="top",
            fontsize=8,
        )
        text_annotations.append(txt)
        txt = fig.text(
            0.72,
            0.97,
            stab_text,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=stab_color,
        )
        text_annotations.append(txt)

        fig.canvas.draw_idle()

    # ── Max trace time ────────────────────────────────────────────
    max_trace_time = 0.0
    for r in unit_results.values():
        if r.trace_times is not None and len(r.trace_times) > 0:
            max_trace_time = max(max_trace_time, r.trace_times[-1])

    # ── Widgets ───────────────────────────────────────────────────
    unit_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=max(0, n_units - 1),
        step=1,
        description="Unit:",
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
    )
    trace_slider = widgets.FloatSlider(
        value=0,
        min=0,
        max=max(0, max_trace_time - trace_time_window),
        step=10,
        description="Time (s):",
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
    )
    prev_btn = widgets.Button(description="< Prev", layout=widgets.Layout(width="70px"))
    next_btn = widgets.Button(description="Next >", layout=widgets.Layout(width="70px"))

    def _on_prev(_: Any) -> None:
        unit_slider.value = (unit_slider.value - 1) % n_units

    def _on_next(_: Any) -> None:
        unit_slider.value = (unit_slider.value + 1) % n_units

    prev_btn.on_click(_on_prev)
    next_btn.on_click(_on_next)

    def _update(_: Any = None) -> None:
        _render(unit_slider.value, trace_slider.value)

    unit_slider.observe(_update, names="value")
    trace_slider.observe(_update, names="value")

    nav = widgets.HBox([prev_btn, unit_slider, next_btn], layout=widgets.Layout(width="100%"))
    controls = widgets.VBox([nav, trace_slider], layout=widgets.Layout(width="100%"))

    _render(0, 0)
    return fig, controls


def browse_units_1d(
    ds: Any,
    unit_results: dict[int, Any] | None = None,
) -> tuple[plt.Figure, widgets.VBox]:
    """Create a 1D unit browser from a MazeDataset.

    Parameters
    ----------
    ds:
        MazeDataset with completed analysis.
    unit_results:
        Subset of results to browse. Defaults to ds.unit_results.
    """
    results = unit_results if unit_results is not None else ds.unit_results
    scfg = ds.spatial_1d

    return create_unit_browser_1d(
        unit_results=results,
        edges=ds.edges_1d,
        df_all_events=ds.event_index,
        trace_name=ds.cfg.neural.trace_name,
        neural_fps=ds.neural_fps,
        speed_threshold=ds.cfg.behavior.speed_threshold,
        p_value_threshold=scfg.p_value_threshold,
        trace_time_window=scfg.trace_time_window,
        arm_boundaries=ds.arm_boundaries,
        arm_labels=ds.effective_arm_order,
        speed_unit="mm/s" if ds.arm_lengths is not None else "pos/s",
    )
