"""Helpers for cell-event overlay and zone-occupancy plots.

These functions operate on a fully-loaded ``BasePlaceCellDataset`` (typically
restored from a ``.pcellbundle``) and are kept generic enough to be reused
between the per-session maze viewer and cross-session analyses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from placecell.dataset import BasePlaceCellDataset

UNKNOWN_LABELS = {"unknown", "Unknown", "UNKNOWN"}


@dataclass
class ViewConfig:
    """3D view orientation knobs."""

    elev: float = 15.0
    azim: float = -60.0
    box_aspect: tuple[float, float, float] = (1.0, 1.0, 2.0)
    invert_y: bool = True


@dataclass
class ProjectionConfig:
    """Flat 2D-projection-beneath-3D knobs. ``z=None`` disables the projection."""

    z: float | None = None
    box: bool = True
    dot_alpha: float = 0.5
    dot_size_frac: float = 0.6


# Saturated palette from ColorBrewer Dark2 (gray + too-green index dropped) +
# Set1 (gray dropped). 14 distinct colors that read on white.
_DARK2 = plt.get_cmap("Dark2")
_SET1 = plt.get_cmap("Set1")
_SKIP_DARK2 = {4, 7}
DEFAULT_PALETTE: list[Any] = [_DARK2(i) for i in range(8) if i not in _SKIP_DARK2] + [
    _SET1(i) for i in range(8)
]
ROOM_COLOR = "0.6"


def _resolve_arm_lists(ds: BasePlaceCellDataset) -> tuple[list[str], list[str]]:
    """Return (raw_arms, effective_arm_order)."""
    raw_arms = list((ds.data_cfg.arm_order if ds.data_cfg and ds.data_cfg.arm_order else []) or [])
    arm_order = list(getattr(ds, "effective_arm_order", []) or raw_arms)
    return raw_arms, arm_order


def arm_color_map(arm_order: Sequence[str]) -> dict[str, Any]:
    """Stable arm → color mapping from :data:`DEFAULT_PALETTE` (cycles)."""
    return {name: DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)] for i, name in enumerate(arm_order)}


# ---------------------------------------------------------------------------
# Occupancy
# ---------------------------------------------------------------------------


def compute_zone_occupancy(ds: BasePlaceCellDataset) -> tuple[pd.Series, pd.Series, float]:
    """Return ``(arm_seconds, room_seconds, unknown_seconds)`` for ``ds``.

    Arm frames are labeled ``{zone}_{direction}`` when the canonical table
    carries a direction column (so the result aligns with
    ``effective_arm_order``); otherwise the raw zone label is used. Room
    frames are zone labels that aren't in ``data_cfg.arm_order`` and aren't
    in ``UNKNOWN_LABELS``.
    """
    if ds.data_cfg is None:
        raise ValueError("ds.data_cfg is required for zone_occupancy.")
    zone_col = getattr(ds.data_cfg, "zone_column", "zone") or "zone"
    fps = float(getattr(ds.data_cfg, "behavior_fps", 20.0))

    traj_df = ds.trajectory if ds.trajectory is not None else ds.trajectory_raw
    if traj_df is None or zone_col not in traj_df.columns:
        empty = pd.Series(dtype=float)
        return empty, empty, 0.0

    raw_arms, arm_order = _resolve_arm_lists(ds)

    canonical = ds.canonical
    if (
        canonical is not None
        and "direction" in canonical.columns
        and "frame_index" in traj_df.columns
        and canonical["direction"].notna().any()
    ):
        dir_map = canonical.set_index("frame_index")["direction"]
        traj_dir = dir_map.reindex(traj_df["frame_index"].astype(int).values).to_numpy()
    else:
        traj_dir = np.array([None] * len(traj_df))

    zone_array = traj_df[zone_col].astype(str).to_numpy()
    valid = traj_df[zone_col].notna().to_numpy()
    is_unknown = np.array([z in UNKNOWN_LABELS for z in zone_array]) & valid
    unknown_s = float(is_unknown.sum()) / fps

    is_arm = np.array([z in raw_arms for z in zone_array]) & valid

    labels = [
        f"{z}_{d}" if d is not None else z for z, d in zip(zone_array[is_arm], traj_dir[is_arm])
    ]
    arm_counts = pd.Series(labels).value_counts() if labels else pd.Series(dtype=int)
    arm_s = (arm_counts / fps).reindex(arm_order).fillna(0.0)

    is_room = valid & ~is_arm & ~is_unknown
    room_counts = pd.Series(zone_array[is_room]).value_counts()
    room_s = (room_counts / fps).sort_index()

    return arm_s, room_s, unknown_s


def plot_zone_occupancy(
    ds: BasePlaceCellDataset,
    *,
    proportion: bool = False,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Single-panel bar chart of zone occupancy (arms + rooms)."""
    arm_s, room_s, unknown_s = compute_zone_occupancy(ds)
    if arm_s.empty and room_s.empty:
        raise ValueError("No zone data on dataset — cannot plot.")

    if proportion:
        total = arm_s.sum() + room_s.sum()
        arm_v = arm_s / total if total > 0 else arm_s * 0
        room_v = room_s / total if total > 0 else room_s * 0
        ylabel = "proportion of total time"
        label_fmt = lambda v: f"{v:.2f}"  # noqa: E731
    else:
        arm_v = arm_s
        room_v = room_s
        ylabel = "time (s)"
        label_fmt = lambda v: f"{v:.0f}s"  # noqa: E731

    arm_colors = arm_color_map(arm_s.index)
    labels = list(arm_v.index) + list(room_v.index)
    values = list(arm_v.values) + list(room_v.values)
    bar_colors = [arm_colors.get(a, DEFAULT_PALETTE[0]) for a in arm_v.index] + [ROOM_COLOR] * len(
        room_v
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(labels) + 2), 3.8))
    else:
        fig = ax.figure

    x = np.arange(len(labels))
    ax.bar(x, values, color=bar_colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if proportion and values:
        ax.set_ylim(0, max(max(values) * 1.15, 0.1))
    for xi, v in zip(x, values):
        ax.text(xi, v, label_fmt(v), ha="center", va="bottom", fontsize=8)

    suffix = (
        f"(arms {arm_s.sum():.0f} s, rooms {room_s.sum():.0f} s, " f"unknown {unknown_s:.0f} s)"
    )
    ax.set_title(title or f"zone occupancy {suffix}")
    fig.tight_layout()
    return fig


def plot_cross_session_occupancy(
    datasets: dict[str, BasePlaceCellDataset],
    *,
    title: str | None = None,
) -> plt.Figure:
    """Mean ± SD proportional zone occupancy across multiple datasets."""
    if not datasets:
        raise ValueError("datasets is empty.")

    sessions: dict[str, dict] = {}
    for name, ds in datasets.items():
        arm_s, room_s, _ = compute_zone_occupancy(ds)
        total = arm_s.sum() + room_s.sum()
        sessions[name] = {
            "arms_prop": arm_s / total if total > 0 else arm_s * 0,
            "rooms_prop": room_s / total if total > 0 else room_s * 0,
        }

    all_arms: list[str] = []
    for s in sessions.values():
        for a in s["arms_prop"].index:
            if a not in all_arms:
                all_arms.append(a)
    all_rooms = sorted({r for s in sessions.values() for r in s["rooms_prop"].index})

    if not all_arms and not all_rooms:
        raise ValueError("No zone data found across sessions.")

    rows_arms = (
        np.stack(
            [s["arms_prop"].reindex(all_arms).fillna(0.0).to_numpy() for s in sessions.values()]
        )
        if all_arms
        else np.empty((len(sessions), 0))
    )
    rows_rooms = (
        np.stack(
            [s["rooms_prop"].reindex(all_rooms).fillna(0.0).to_numpy() for s in sessions.values()]
        )
        if all_rooms
        else np.empty((len(sessions), 0))
    )
    arr = np.concatenate([rows_arms, rows_rooms], axis=1)
    mean = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)

    arm_colors = arm_color_map(all_arms)
    labels = all_arms + all_rooms
    bar_colors = [arm_colors.get(a, DEFAULT_PALETTE[0]) for a in all_arms] + [ROOM_COLOR] * len(
        all_rooms
    )

    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(labels) + 2), 4.0))
    x = np.arange(len(labels))
    ax.bar(x, mean, yerr=sd, color=bar_colors, capsize=3, ecolor="0.4")
    for j in range(arr.shape[0]):
        ax.plot(x, arr[j], "o", color="0.4", alpha=0.5, markersize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("proportion of total time")
    ax.set_ylim(0, max((mean + sd).max(), 0.1) * 1.2)
    ax.set_title(title or f"Cross-session occupancy — mean ± SD, n={len(sessions)}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cell event overlay
# ---------------------------------------------------------------------------


def select_top_place_cells(ds: BasePlaceCellDataset, n: int) -> list[int]:
    """Top-N place cell unit_ids ranked by SI."""
    pcs = ds.place_cells()
    return [uid for uid, _ in sorted(pcs.items(), key=lambda kv: -kv[1].si)[:n]]


def gather_events(
    ds: BasePlaceCellDataset,
    unit_ids: Sequence[int],
    *,
    event_percentile: float = 0.0,
    event_abs_min: float = 0.0,
) -> dict[int, pd.DataFrame]:
    """Per-unit event tables filtered by amplitude (percentile and/or absolute)."""
    out: dict[int, pd.DataFrame] = {}
    for uid in unit_ids:
        ev = ds.event_place[ds.event_place["unit_id"] == uid].copy()
        if "s" in ev.columns and len(ev):
            thr = event_abs_min
            if event_percentile > 0:
                thr = max(thr, float(np.percentile(ev["s"].to_numpy(), event_percentile)))
            if thr > 0:
                ev = ev[ev["s"] >= thr]
        out[uid] = ev
    return out


def _arm_masked_canonical(canonical: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``canonical`` with x/y NaN'd on non-arm rows."""
    if "pos_1d" not in canonical.columns:
        return canonical.copy()
    out = canonical.copy()
    out.loc[out["pos_1d"].isna(), ["x", "y"]] = np.nan
    return out


def _strip_3d_background(ax: Any) -> None:
    """Transparent panes, no grid; hide x/y axes (keep z/time)."""
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((1, 1, 1, 0))
        pane.set_edgecolor((1, 1, 1, 0))
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis):
        axis.line.set_color((1, 1, 1, 0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)


def _directions_in(canonical: pd.DataFrame) -> list[Any]:
    if "direction" not in canonical.columns or not canonical["direction"].notna().any():
        return [None]
    return sorted(canonical["direction"].dropna().unique(), key=str)


def plot_event_overlay_2d(
    ds: BasePlaceCellDataset,
    unit_ids: Sequence[int],
    *,
    events_by_cell: dict[int, pd.DataFrame] | None = None,
    cell_colors: dict[int, Any] | None = None,
    dot_size: int = 16,
    dot_alpha: float = 0.85,
    invert_y: bool = True,
    title: str | None = None,
) -> plt.Figure:
    """2D x/y trajectory with colored event dots, split by direction if available."""
    canonical = ds.canonical
    arm_canonical = _arm_masked_canonical(canonical)
    directions = _directions_in(arm_canonical)

    events_by_cell = events_by_cell or gather_events(ds, unit_ids)
    cmap = plt.get_cmap("tab10")
    cell_colors = cell_colors or {uid: cmap(i % cmap.N) for i, uid in enumerate(unit_ids)}

    fig, axes = plt.subplots(
        1,
        len(directions),
        figsize=(5 * len(directions), 5),
        squeeze=False,
    )
    axes = axes[0]

    graph_polylines = getattr(ds, "graph_polylines", None)

    for ax, direction in zip(axes, directions):
        sub = arm_canonical if direction is None else arm_canonical.copy()
        if direction is not None:
            sub.loc[sub["direction"] != direction, ["x", "y"]] = np.nan
        ax.plot(sub["x"], sub["y"], color="0", lw=0.3, zorder=1)
        if graph_polylines:
            for waypoints in graph_polylines.values():
                pts = np.asarray(waypoints, dtype=float)
                ax.plot(pts[:, 0], pts[:, 1], color="0.6", lw=0.8, zorder=0)

        for uid in unit_ids:
            ev = events_by_cell.get(uid, pd.DataFrame())
            if direction is not None and "direction" in ev.columns:
                ev = ev[ev["direction"] == direction]
            if ev.empty:
                continue
            ax.scatter(
                ev["x"],
                ev["y"],
                s=dot_size,
                c=[cell_colors[uid]],
                edgecolors="none",
                alpha=dot_alpha,
                label=f"cell {uid}",
                zorder=2,
            )
        ax.set_aspect("equal")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        if invert_y:
            ax.invert_yaxis()
        ax.set_title(str(direction) if direction is not None else "All")

    axes[0].legend(loc="upper right", fontsize=8, markerscale=1.2, framealpha=0.9)
    fig.suptitle(title or f"event overlay ({len(unit_ids)} cells)")
    fig.tight_layout()
    return fig


def plot_event_overlay_3d(
    ds: BasePlaceCellDataset,
    unit_ids: Sequence[int],
    *,
    events_by_cell: dict[int, pd.DataFrame] | None = None,
    cell_colors: dict[int, Any] | None = None,
    dot_size: int = 16,
    dot_alpha: float = 1.0,
    view: ViewConfig | None = None,
    projection: ProjectionConfig | None = None,
    graph_z: float | None = -20.0,
    traversal_subset: bool = False,
    show_trajectory: bool = True,
    title: str | None = None,
) -> plt.Figure:
    """3D (x, y, time) overlay with all selected cells on shared axes per direction.

    Parameters
    ----------
    view:
        Camera orientation knobs; see :class:`ViewConfig`.
    projection:
        Flat 2D-beneath-3D event projection; see :class:`ProjectionConfig`.
        ``projection.z is None`` (the default) disables the projection.
    graph_z:
        z-plane for the maze-graph polyline overlay (``None`` to disable).
    show_trajectory:
        Draw the trajectory line (possibly subset).
    traversal_subset:
        If True, restrict the trajectory (and dots) to arm traversals that
        contain at least one event from ``events_by_cell``.
    """
    view = view or ViewConfig()
    projection = projection or ProjectionConfig()
    canonical = ds.canonical
    arm_canonical = _arm_masked_canonical(canonical)
    directions = _directions_in(arm_canonical)

    traj = arm_canonical.copy()
    t0_series = traj["neural_time"].dropna()
    if t0_series.empty:
        raise ValueError("canonical has no neural_time values")
    t0 = float(t0_series.iloc[0])
    traj["t_min"] = (traj["neural_time"] - t0) / 60.0

    frame_to_time = canonical.set_index("frame_index")["neural_time"]

    events_by_cell = events_by_cell or gather_events(ds, unit_ids)

    # Restrict events to arm frames (consistent with the arm-masked trajectory).
    if "pos_1d" in canonical.columns:
        _arm_frames = set(
            canonical.loc[canonical["pos_1d"].notna(), "frame_index"].astype(int).tolist()
        )
        events_by_cell = {
            uid: (
                ev[ev["frame_index"].astype(int).isin(_arm_frames)]
                if "frame_index" in ev.columns and not ev.empty
                else ev
            )
            for uid, ev in events_by_cell.items()
        }

    cmap = plt.get_cmap("tab10")
    cell_colors = cell_colors or {uid: cmap(i % cmap.N) for i, uid in enumerate(unit_ids)}
    graph_polylines = getattr(ds, "graph_polylines", None)

    # Optional trajectory subset: expand each event frame in ``events_by_cell``
    # to its full arm traversal (maximal contiguous run on the same arm_index
    # in the canonical table). This keeps every dot shown in the overlay backed
    # by a traversal in the subset view — no extra percentile filter applied
    # here; ``gather_events`` is the single source of truth for which events
    # are "kept".
    traj_frame_set: set[int] | None = None
    if traversal_subset:
        pooled_parts = [
            ev for ev in events_by_cell.values() if not ev.empty and "frame_index" in ev.columns
        ]
        if pooled_parts:
            pooled = pd.concat(pooled_parts, ignore_index=True)
            event_frames = set(pooled["frame_index"].astype(int).tolist())

            # Build traversal IDs per row in canonical: each maximal contiguous
            # block of equal non-NaN arm_index is one traversal.
            if "arm_index" in canonical.columns and "frame_index" in canonical.columns:
                cf = canonical["frame_index"].astype(int).to_numpy()
                arm = canonical["arm_index"].to_numpy()
                arm_s = pd.to_numeric(pd.Series(arm), errors="coerce")
                arm_isna = arm_s.isna().to_numpy()
                arm_filled = arm_s.fillna(-1).to_numpy()
                changes = np.concatenate([[True], arm_filled[1:] != arm_filled[:-1]])
                traversal_id = np.cumsum(changes) - 1
                traversal_id[arm_isna] = -1

                event_rows = np.array([i for i, f in enumerate(cf) if int(f) in event_frames])
                if event_rows.size:
                    hit_traversals = {int(t) for t in traversal_id[event_rows] if t >= 0}
                    keep_rows = np.isin(traversal_id, list(hit_traversals))
                    traj_frame_set = {int(f) for f in cf[keep_rows]}
                else:
                    traj_frame_set = set()
            else:
                traj_frame_set = event_frames

    fig = plt.figure(figsize=(4.5 * len(directions), 5))
    for i, direction in enumerate(directions):
        ax = fig.add_subplot(
            1,
            len(directions),
            i + 1,
            projection="3d",
            computed_zorder=False,
        )
        _strip_3d_background(ax)
        ax.set_box_aspect(view.box_aspect)
        ax.view_init(elev=view.elev, azim=view.azim)

        sub = traj if direction is None else traj.copy()
        if direction is not None:
            sub.loc[sub["direction"] != direction, ["x", "y", "t_min"]] = np.nan
        if show_trajectory:
            if traj_frame_set is not None and "frame_index" in sub.columns:
                sub = sub.copy()
                keep_mask = sub["frame_index"].astype(int).isin(traj_frame_set)
                sub.loc[~keep_mask, ["x", "y", "t_min"]] = np.nan
            ax.plot(sub["x"], sub["y"], sub["t_min"], color="0", lw=0.3, zorder=1)

        if graph_polylines and graph_z is not None:
            for waypoints in graph_polylines.values():
                pts = np.asarray(waypoints, dtype=float)
                ax.plot(
                    pts[:, 0], pts[:, 1], np.full(len(pts), graph_z), color="0.5", lw=0.8, zorder=0
                )

        # Optional flat 2D projection of events at a fixed z plane.
        if projection.z is not None:
            for uid in unit_ids:
                ev = events_by_cell.get(uid, pd.DataFrame())
                if direction is not None and "direction" in ev.columns:
                    ev = ev[ev["direction"] == direction]
                if traj_frame_set is not None and "frame_index" in ev.columns:
                    ev = ev[ev["frame_index"].astype(int).isin(traj_frame_set)]
                if ev.empty:
                    continue
                ax.scatter(
                    ev["x"].to_numpy(),
                    ev["y"].to_numpy(),
                    np.full(len(ev), projection.z),
                    s=dot_size * projection.dot_size_frac,
                    c=[cell_colors[uid]],
                    edgecolors="none",
                    alpha=projection.dot_alpha,
                    depthshade=False,
                    zorder=2,
                )
            if projection.box:
                # Bounding box from projected event points + 5% margin.
                proj_xs, proj_ys = [], []
                for uid in unit_ids:
                    ev = events_by_cell.get(uid, pd.DataFrame())
                    if direction is not None and "direction" in ev.columns:
                        ev = ev[ev["direction"] == direction]
                    if traj_frame_set is not None and "frame_index" in ev.columns:
                        ev = ev[ev["frame_index"].astype(int).isin(traj_frame_set)]
                    if not ev.empty:
                        proj_xs.append(ev["x"].to_numpy())
                        proj_ys.append(ev["y"].to_numpy())
                if proj_xs:
                    all_x = np.concatenate(proj_xs)
                    all_y = np.concatenate(proj_ys)
                    xmin, xmax = float(np.nanmin(all_x)), float(np.nanmax(all_x))
                    ymin, ymax = float(np.nanmin(all_y)), float(np.nanmax(all_y))
                    mx = (xmax - xmin) * 0.05
                    my = (ymax - ymin) * 0.05
                    xmin -= mx
                    xmax += mx
                    ymin -= my
                    ymax += my
                    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

                    pz = projection.z
                    quad = [
                        [
                            (xmin, ymin, pz),
                            (xmax, ymin, pz),
                            (xmax, ymax, pz),
                            (xmin, ymax, pz),
                        ]
                    ]
                    ax.add_collection3d(
                        Poly3DCollection(
                            quad,
                            facecolor="0.85",
                            edgecolor="none",
                            zorder=0,
                        )
                    )

        for uid in unit_ids:
            ev = events_by_cell.get(uid, pd.DataFrame())
            if direction is not None and "direction" in ev.columns:
                ev = ev[ev["direction"] == direction]
            if traj_frame_set is not None and "frame_index" in ev.columns:
                ev = ev[ev["frame_index"].astype(int).isin(traj_frame_set)]
            if ev.empty:
                continue
            ev_time = frame_to_time.reindex(ev["frame_index"]).to_numpy()
            keep = ~np.isnan(ev_time)
            if not keep.any():
                continue
            t_ev = (ev_time[keep] - t0) / 60.0
            ax.scatter(
                ev["x"].to_numpy()[keep],
                ev["y"].to_numpy()[keep],
                t_ev,
                s=dot_size,
                c=[cell_colors[uid]],
                edgecolors="none",
                alpha=dot_alpha,
                depthshade=False,
                zorder=10,
                label=f"cell {uid}",
            )

        if view.invert_y:
            ax.invert_yaxis()
        # Start time axis at 0; projection lives in negative z space.
        t_max = float(traj["t_min"].max())
        z_lo = projection.z if projection.z is not None else 0.0
        ax.set_zlim(z_lo, t_max * 1.02)
        zticks = [t for t in ax.get_zticks() if t >= 0]
        ax.set_zticks(zticks)
        ax.set_zlabel("time (min)")
        ax.set_title(str(direction) if direction is not None else "All")
        if i == 0:
            ax.legend(loc="upper left", fontsize=8, markerscale=1.2)

    fig.suptitle(title or f"event overlay 3D ({len(unit_ids)} cells)")
    fig.tight_layout()
    return fig
