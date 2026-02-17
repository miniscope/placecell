"""Maze behavior processing for 1D tube analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from placecell.logging import init_logger

logger = init_logger(__name__)


def load_graph_polylines(graph_path: Path) -> tuple[dict[str, list[list[float]]], float]:
    """Load behavior graph YAML and return raw polyline waypoints.

    Parameters
    ----------
    graph_path:
        Path to YAML file in combined zone format (``zones`` key with
        ``type``, ``points``, and optionally ``connections`` per zone).

    Returns
    -------
    tuple of (polylines, mm_per_pixel)
        polylines maps zone name to list of [x, y] waypoints in pixel coords.
    """
    with open(graph_path) as f:
        data = yaml.safe_load(f)

    mm_per_pixel = float(data.get("mm_per_pixel", 1.0))

    polylines: dict[str, list[list[float]]] = {}
    for zone_name, zone_data in data["zones"].items():
        points = zone_data.get("points", [])
        if isinstance(points, list) and len(points) >= 2:
            polylines[zone_name] = points

    return polylines, mm_per_pixel


def compute_tube_lengths(graph_path: Path) -> tuple[dict[str, float], float]:
    """Load behavior graph YAML and compute polyline lengths for each zone.

    Parameters
    ----------
    graph_path:
        Path to YAML file with ``mm_per_pixel`` and zone polylines.

    Returns
    -------
    tuple of (zone_lengths, mm_per_pixel)
        zone_lengths maps zone name to physical length in mm.
    """
    polylines, mm_per_pixel = load_graph_polylines(graph_path)

    zone_lengths: dict[str, float] = {}
    for key, waypoints in polylines.items():
        length = 0.0
        for i in range(1, len(waypoints)):
            dx = waypoints[i][0] - waypoints[i - 1][0]
            dy = waypoints[i][1] - waypoints[i - 1][1]
            length += (dx**2 + dy**2) ** 0.5
        zone_lengths[key] = length * mm_per_pixel

    logger.info(
        "Loaded behavior graph: %d zones, mm_per_pixel=%.2f, tube lengths: %s",
        len(zone_lengths),
        mm_per_pixel,
        {k: f"{v:.1f}" for k, v in zone_lengths.items()},
    )
    return zone_lengths, mm_per_pixel


def serialize_tube_position(
    trajectory: pd.DataFrame,
    tube_order: list[str],
    zone_column: str = "zone",
    tube_position_column: str = "tube_position",
    tube_lengths: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Convert zone + tube_position to a concatenated 1D position.

    Filters to tube-only frames and maps each tube's [0,1] position
    onto a concatenated 1D axis.  When *tube_lengths* is provided the
    axis is in physical units (mm) so that each tube spans its real
    length; otherwise each tube spans exactly 1 unit.

    Parameters
    ----------
    trajectory:
        DataFrame with zone, tube_position, and standard columns.
    tube_order:
        Ordered list of tube zone names.
    zone_column:
        Column containing zone labels.
    tube_position_column:
        Column containing within-tube position (0-1).
    tube_lengths:
        Optional dict mapping zone name to physical length.  When
        provided, ``pos_1d`` is in physical units.

    Returns
    -------
    pd.DataFrame
        Filtered to tube frames only, with new columns ``pos_1d``
        and ``tube_index``.
    """
    tube_set = set(tube_order)
    tp = pd.to_numeric(trajectory[tube_position_column], errors="coerce")
    mask = trajectory[zone_column].isin(tube_set) & tp.notna()
    df = trajectory[mask].copy()
    df[tube_position_column] = tp[mask]

    df["tube_index"] = df[zone_column].map({z: i for i, z in enumerate(tube_order)})

    if tube_lengths is not None:
        lengths = [tube_lengths[z] for z in tube_order]
        offsets = np.concatenate([[0.0], np.cumsum(lengths[:-1])])
        zone_to_offset = {z: offsets[i] for i, z in enumerate(tube_order)}
        zone_to_length = {z: tube_lengths[z] for z in tube_order}
        df["pos_1d"] = df[tube_position_column].astype(float) * df[zone_column].map(
            zone_to_length
        ) + df[zone_column].map(zone_to_offset)
    else:
        zone_to_offset = {zone: float(i) for i, zone in enumerate(tube_order)}
        df["pos_1d"] = df[tube_position_column].astype(float) + df[zone_column].map(zone_to_offset)

    logger.info(
        "Serialized %d/%d frames to 1D (tubes: %s, physical=%s)",
        len(df),
        len(trajectory),
        tube_order,
        tube_lengths is not None,
    )
    return df


def assign_traversal_direction(
    trajectory: pd.DataFrame,
    tube_order: list[str],
    zone_column: str = "zone",
    tube_position_column: str = "tube_position",
    direction_threshold: float = 0.5,
    tube_lengths: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Split tube traversals into forward/reverse directional segments.

    Each contiguous run of frames within the same tube is a "traversal".
    Direction is determined by the first ``tube_position`` value:
    < threshold → forward, >= threshold → reverse.

    Parameters
    ----------
    trajectory:
        DataFrame from ``serialize_tube_position`` with columns
        ``zone_column``, ``tube_position_column``, ``tube_index``,
        ``pos_1d``, ``frame_index``.
    tube_order:
        Original tube order (e.g. ["Tube_1", "Tube_2", ...]).
    zone_column:
        Column with zone labels.
    tube_position_column:
        Column with within-tube position (0-1).
    direction_threshold:
        Position threshold for direction classification.
    tube_lengths:
        Optional dict mapping zone name to physical length.  When
        provided, ``pos_1d`` is recomputed in physical units.

    Returns
    -------
    tuple of (DataFrame with direction columns and recomputed pos_1d,
              effective_tube_order list e.g. ["Tube_1_fwd", "Tube_1_rev", ...])
    """
    df = trajectory.sort_values("frame_index").copy()

    # Detect traversal boundaries: zone changes or frame_index gaps > 1
    zone_changed = df[zone_column] != df[zone_column].shift(1)
    frame_gap = df["frame_index"].diff() > 1
    boundary = zone_changed | frame_gap
    # First row is always a boundary
    boundary.iloc[0] = True
    df["traversal_id"] = boundary.cumsum()

    # Classify each traversal by its first tube_position
    first_pos = df.groupby("traversal_id")[tube_position_column].first()
    traversal_dir = (first_pos < direction_threshold).map({True: "fwd", False: "rev"})
    df["direction"] = df["traversal_id"].map(traversal_dir)

    # Build effective tube order: [Tube_1_fwd, Tube_1_rev, Tube_2_fwd, ...]
    effective_order = []
    for tube in tube_order:
        effective_order.append(f"{tube}_fwd")
        effective_order.append(f"{tube}_rev")

    # Create directional zone label and recompute tube_index + pos_1d
    df["zone_dir"] = df[zone_column] + "_" + df["direction"]
    zone_dir_to_idx = {name: i for i, name in enumerate(effective_order)}
    df["tube_index"] = df["zone_dir"].map(zone_dir_to_idx)

    if tube_lengths is not None:
        # Each directional segment has the same physical length as its parent tube
        effective_lengths = {}
        for tube in tube_order:
            L = tube_lengths[tube]
            effective_lengths[f"{tube}_fwd"] = L
            effective_lengths[f"{tube}_rev"] = L
        seg_lengths = [effective_lengths[t] for t in effective_order]
        offsets = np.concatenate([[0.0], np.cumsum(seg_lengths[:-1])])
        zone_dir_to_offset = {name: offsets[i] for i, name in enumerate(effective_order)}
        zone_dir_to_length = {name: effective_lengths[name] for name in effective_order}
        df["pos_1d"] = df[tube_position_column].astype(float) * df["zone_dir"].map(
            zone_dir_to_length
        ) + df["zone_dir"].map(zone_dir_to_offset)
    else:
        df["pos_1d"] = df[tube_position_column].astype(float) + df["tube_index"].astype(float)

    logger.info(
        "Direction splitting: %d traversals, effective segments: %s",
        df["traversal_id"].nunique(),
        effective_order,
    )
    return df, effective_order


def filter_complete_traversals(
    trajectory: pd.DataFrame,
    full_trajectory: pd.DataFrame,
    tube_order: list[str],
    zone_column: str = "zone",
) -> pd.DataFrame:
    """Remove traversals where the animal did not cross from one room to another.

    For each traversal, the zone immediately before entering and after
    leaving the tube is looked up in *full_trajectory*.  A traversal is
    "complete" when both zones are rooms (not tubes) and they differ
    (i.e. the animal went Room A → Tube → Room B, not Room A → Tube → Room A).

    If ``traversal_id`` is not yet present (e.g. ``split_by_direction``
    is False), traversal boundaries are detected automatically.

    Parameters
    ----------
    trajectory:
        Tube-only DataFrame with ``frame_index`` column (and optionally
        ``traversal_id`` from ``assign_traversal_direction``).
    full_trajectory:
        Complete trajectory including room frames, with ``frame_index``
        and *zone_column*.
    tube_order:
        List of tube zone names (anything else is treated as a room).
    zone_column:
        Column containing zone labels in *full_trajectory*.

    Returns
    -------
    pd.DataFrame
        Filtered to only complete (room-to-room) traversals.
    """
    df = trajectory.sort_values("frame_index").copy()

    # Assign traversal_id if not already present
    if "traversal_id" not in df.columns:
        zone_changed = df[zone_column] != df[zone_column].shift(1)
        frame_gap = df["frame_index"].diff() > 1
        boundary = zone_changed | frame_gap
        boundary.iloc[0] = True
        df["traversal_id"] = boundary.cumsum()

    tube_set = set(tube_order)

    # Build sorted lookup from full trajectory
    full_sorted = full_trajectory.sort_values("frame_index")
    full_fi = full_sorted["frame_index"].values
    full_zone = full_sorted[zone_column].values

    # For each traversal, check entry/exit rooms
    bounds = df.groupby("traversal_id")["frame_index"].agg(["first", "last"])
    complete_ids = []
    for trav_id, (first_frame, last_frame) in bounds.iterrows():
        idx_before = np.searchsorted(full_fi, first_frame, side="left") - 1
        idx_after = np.searchsorted(full_fi, last_frame, side="right")

        if idx_before < 0 or idx_after >= len(full_fi):
            continue

        entry_zone = full_zone[idx_before]
        exit_zone = full_zone[idx_after]

        if (
            entry_zone not in tube_set
            and exit_zone not in tube_set
            and entry_zone != exit_zone
        ):
            complete_ids.append(trav_id)

    n_total = bounds.shape[0]
    n_kept = len(complete_ids)
    result = df[df["traversal_id"].isin(complete_ids)].copy()
    logger.info(
        "Complete-traversal filter (room-based): kept %d/%d traversals",
        n_kept,
        n_total,
    )
    return result


def compute_speed_1d(
    trajectory: pd.DataFrame,
    window_frames: int = 5,
) -> pd.DataFrame:
    """Compute 1D speed along the tube axis.

    Speed is computed as |delta(pos_1d)| / delta(time) over a window
    of *actual* frame indices, not entry offsets.  Because room frames
    are absent from the tube-only trajectory, consecutive entries can
    have large frame_index gaps (room visits).  Using entry offsets
    would compute distance/time over artificially long intervals,
    systematically underestimating speed in frequently-exited tubes.

    Parameters
    ----------
    trajectory:
        DataFrame with pos_1d, tube_index, unix_time, frame_index columns.
        Must be tube-only (output of ``serialize_tube_position``).
    window_frames:
        Look-ahead window in *real* frame numbers.

    Returns
    -------
    pd.DataFrame with ``speed_1d`` column added.
    """
    df = trajectory.sort_values("frame_index").copy()
    pos = df["pos_1d"].values
    t = df["unix_time"].values
    tube_idx = df["tube_index"].values
    frame_idx = df["frame_index"].values
    n = len(df)

    # For each entry, find the entry whose frame_index is closest to
    # frame_index[i] + window_frames (the real look-ahead target).
    target_frames = frame_idx + window_frames
    end_indices = np.searchsorted(frame_idx, target_frames, side="left")
    end_indices = np.clip(end_indices, 0, n - 1)

    # Vectorized speed computation
    dt = t[end_indices] - t
    dpos = np.abs(pos[end_indices] - pos)
    same_tube = tube_idx[end_indices] == tube_idx
    frame_gap = frame_idx[end_indices] - frame_idx
    # Valid: same tube, positive time, and frame gap not absurdly large
    # (allow up to 3x window to tolerate a few missing frames)
    valid = (end_indices > np.arange(n)) & same_tube & (dt > 0) & (frame_gap <= window_frames * 3)

    speed = np.zeros(n)
    speed[valid] = dpos[valid] / dt[valid]
    df["speed_1d"] = speed
    return df


def filter_tube_by_speed(
    trajectory: pd.DataFrame,
    speed_threshold: float,
) -> pd.DataFrame:
    """Filter 1D trajectory to frames above speed threshold.

    Parameters
    ----------
    trajectory:
        DataFrame with speed_1d and frame_index columns.
    speed_threshold:
        Minimum 1D speed to keep.

    Returns
    -------
    Filtered DataFrame with frame_index renamed to beh_frame_index.
    """
    filtered = trajectory[trajectory["speed_1d"] >= speed_threshold].copy()
    filtered = filtered.sort_values("frame_index")
    filtered = filtered.rename(columns={"frame_index": "beh_frame_index"})
    return filtered
