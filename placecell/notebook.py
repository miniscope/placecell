"""Notebook utilities for interactive place cell visualization."""

from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def run_deconvolution(
    C_da: Any,
    unit_ids: list[int],
    g: tuple[float, float],
    baseline: float | str,
    penalty: float,
    s_min: float,
    progress_bar: Any = None,
) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    """Run OASIS deconvolution on calcium traces.

    Parameters
    ----------
    C_da : xarray.DataArray
        Calcium traces with dimensions (unit_id, frame).
    unit_ids : list[int]
        List of unit IDs to process.
    g : tuple[float, float]
        AR(2) coefficients for OASIS.
    baseline : float or str
        Baseline correction. Use 'pXX' for percentile (e.g., 'p10') or numeric value.
    penalty : float
        Sparsity penalty for OASIS.
    s_min : float
        Minimum event size threshold.
    progress_bar : optional
        tqdm progress bar wrapper (e.g., tqdm.notebook.tqdm).

    Returns
    -------
    good_unit_ids : list[int]
        Unit IDs that were successfully deconvolved.
    C_list : list[np.ndarray]
        Deconvolved calcium traces.
    S_list : list[np.ndarray]
        Spike trains.
    """
    from oasis.oasis_methods import oasisAR2

    good_unit_ids: list[int] = []
    C_list: list[np.ndarray] = []
    S_list: list[np.ndarray] = []

    iterator = progress_bar(unit_ids) if progress_bar else unit_ids

    for uid in iterator:
        y = np.ascontiguousarray(C_da.sel(unit_id=uid).values, dtype=np.float64)

        # Baseline correction
        if isinstance(baseline, str) and baseline.startswith("p"):
            p = float(baseline[1:])
            b = float(np.percentile(y, p))
        else:
            b = float(baseline)

        y_corrected = y - b

        try:
            c, s = oasisAR2(y_corrected, g1=g[0], g2=g[1], lam=penalty, s_min=s_min)
            good_unit_ids.append(int(uid))
            C_list.append(np.asarray(c, dtype=float))
            S_list.append(np.asarray(s, dtype=float))
        except Exception:
            continue

    return good_unit_ids, C_list, S_list


def build_event_index_dataframe(
    unit_ids: list[int],
    S_list: list[np.ndarray],
) -> pd.DataFrame:
    """Build event index DataFrame from spike trains.

    Parameters
    ----------
    unit_ids : list[int]
        Unit IDs corresponding to each spike train.
    S_list : list[np.ndarray]
        List of spike train arrays.

    Returns
    -------
    pd.DataFrame
        Event index with columns: unit_id, frame, s.
    """
    S_arr = np.stack(S_list, axis=0)
    event_rows = []

    for i, uid in enumerate(unit_ids):
        s_vec = S_arr[i]
        frames = np.nonzero(s_vec > 0)[0]
        for fr in frames:
            event_rows.append({"unit_id": uid, "frame": int(fr), "s": float(s_vec[fr])})

    return pd.DataFrame(event_rows)


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
    stability_threshold: float,
    trace_time_window: float = 600.0,
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
    stability_threshold : float
        Correlation threshold for stability.
    trace_time_window : float
        Time window for trace display in seconds.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure.
    controls : widgets.VBox
        Widget controls (sliders and buttons).
    """
    n_units = len(unique_units)

    # Create figure with 3 rows
    fig = plt.figure(figsize=(8, 7))
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
        vis_data_above = result["vis_data_above"]
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
            norm_max = np.max(normalized_amps) if len(normalized_amps) > 0 and np.max(normalized_amps) > 0 else 1.0
            alphas = normalized_amps / norm_max

            ax2.scatter(
                x_vals, y_vals, c="red", s=20, alpha=alphas, zorder=2
            )

        ax2.set_title(f"Trajectory ({len(vis_data_above)} events)", fontsize=9)
        ax2.set_aspect("equal")
        ax2.axis("off")

        # 3. Rate maps (first half, second half, full)
        rate_map_first = result.get("rate_map_first", np.full_like(result["rate_map"], np.nan))
        rate_map_second = result.get("rate_map_second", np.full_like(result["rate_map"], np.nan))
        stab_corr = result.get("stability_corr", np.nan)

        # Plot first half
        im1 = ax3a.imshow(rate_map_first.T, origin="lower", cmap="jet", aspect="equal")
        ax3a.set_title("1st half", fontsize=9)
        ax3a.axis("off")

        # Plot second half
        im2 = ax3b.imshow(rate_map_second.T, origin="lower", cmap="jet", aspect="equal")
        ax3b.set_title("2nd half", fontsize=9)
        ax3b.axis("off")

        # Plot full session
        im3 = ax3c.imshow(result["rate_map"].T, origin="lower", cmap="jet", aspect="equal")
        ax3c.set_title(f"Full (r={stab_corr:.2f})" if not np.isnan(stab_corr) else "Full", fontsize=9)
        ax3c.axis("off")

        # Set same color scale for all three
        im1.set_clim(0.0, 1.0)
        im2.set_clim(0.0, 1.0)
        im3.set_clim(0.0, 1.0)

        # 4. SI histogram
        ax4.hist(result["shuffled_sis"], bins=15, color="gray", alpha=0.7, edgecolor="black")
        ax4.axvline(result["si"], color="red", linestyle="--", linewidth=2)
        ax4.set_title(f"SI: {result['si']:.2f}, p={result['p_val']:.3f}", fontsize=9)
        ax4.set_xlabel("SI (bits/s)", fontsize=8)
        ax4.set_ylabel("Count", fontsize=8)
        ax4.tick_params(labelsize=7)
        ax4.set_box_aspect(1)

        # 5. Trace
        if result["trace_data"] is not None and result["trace_times"] is not None:
            trace = result["trace_data"]
            t_full = result["trace_times"]

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
                        label=f"Events (< {speed_threshold:.0f} px/s)",
                    )
                )
            if len(event_times_red) > 0:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="red",
                        linewidth=1.5,
                        label=f"Events (>= {speed_threshold:.0f} px/s)",
                    )
                )
            ax5.legend(handles=legend_elements, loc="upper left", fontsize=7, framealpha=0.9)
        else:
            ax5.text(0.5, 0.5, "No trace data", ha="center", va="center", fontsize=8)

        # Status text
        n_events = len(result["unit_data"]) if not result["unit_data"].empty else 0
        p_val = result["p_val"]
        stab_corr = result["stability_corr"]

        sig_pass = p_val < p_value_threshold
        sig_text = "pass" if sig_pass else "fail"
        sig_color = "green" if sig_pass else "red"

        if np.isnan(stab_corr):
            stab_text, stab_color = "N/A", "gray"
        else:
            stab_pass = stab_corr >= stability_threshold
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

        stab_str = f"r={stab_corr:.2f}" if not np.isnan(stab_corr) else ""
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
        if r["trace_times"] is not None and len(r["trace_times"]) > 0:
            max_trace_time = max(max_trace_time, r["trace_times"][-1])

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
