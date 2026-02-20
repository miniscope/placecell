"""Maze behavior processing for 1D arm analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from placecell.logging import init_logger

logger = init_logger(__name__)


def load_graph_polylines(graph_path: Path) -> dict[str, list[list[float]]]:
    """Load behavior graph YAML and return raw polyline waypoints.

    Parameters
    ----------
    graph_path:
        Path to YAML file in combined zone format (``zones`` key with
        ``type``, ``points``, and optionally ``connections`` per zone).

    Returns
    -------
    dict[str, list[list[float]]]
        Maps zone name to list of [x, y] waypoints in pixel coords.
    """
    with open(graph_path) as f:
        data = yaml.safe_load(f)

    polylines: dict[str, list[list[float]]] = {}
    for zone_name, zone_data in data["zones"].items():
        points = zone_data.get("points", [])
        if isinstance(points, list) and len(points) >= 2:
            polylines[zone_name] = points

    return polylines


def compute_arm_lengths(
    polylines: dict[str, list[list[float]]], mm_per_pixel: float,
) -> dict[str, float]:
    """Compute polyline lengths for each zone.

    Parameters
    ----------
    polylines:
        Maps zone name to list of [x, y] waypoints in pixel coords.
    mm_per_pixel:
        Scale factor for converting pixel distances to mm.

    Returns
    -------
    dict[str, float]
        Maps zone name to physical length in mm.
    """
    zone_lengths: dict[str, float] = {}
    for key, waypoints in polylines.items():
        length = 0.0
        for i in range(1, len(waypoints)):
            dx = waypoints[i][0] - waypoints[i - 1][0]
            dy = waypoints[i][1] - waypoints[i - 1][1]
            length += (dx**2 + dy**2) ** 0.5
        zone_lengths[key] = length * mm_per_pixel

    logger.info(
        "Arm lengths (mm_per_pixel=%.2f): %s",
        mm_per_pixel,
        {k: f"{v:.1f}" for k, v in zone_lengths.items()},
    )
    return zone_lengths


def serialize_arm_position(
    trajectory: pd.DataFrame,
    arm_order: list[str],
    zone_column: str = "zone",
    arm_position_column: str = "arm_position",
    arm_lengths: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Convert zone + arm_position to a concatenated 1D position.

    Filters to arm-only frames and maps each arm's [0,1] position
    onto a concatenated 1D axis.  When *arm_lengths* is provided the
    axis is in physical units (mm) so that each arm spans its real
    length; otherwise each arm spans exactly 1 unit.

    Parameters
    ----------
    trajectory:
        DataFrame with zone, arm_position, and standard columns.
    arm_order:
        Ordered list of arm zone names.
    zone_column:
        Column containing zone labels.
    arm_position_column:
        Column containing within-arm position (0-1).
    arm_lengths:
        Optional dict mapping zone name to physical length.  When
        provided, ``pos_1d`` is in physical units.

    Returns
    -------
    pd.DataFrame
        Filtered to arm frames only, with new columns ``pos_1d``
        and ``arm_index``.
    """
    arm_set = set(arm_order)
    tp = pd.to_numeric(trajectory[arm_position_column], errors="coerce")
    mask = trajectory[zone_column].isin(arm_set) & tp.notna()
    df = trajectory[mask].copy()
    df[arm_position_column] = tp[mask]

    df["arm_index"] = df[zone_column].map({z: i for i, z in enumerate(arm_order)})

    if arm_lengths is not None:
        lengths = [arm_lengths[z] for z in arm_order]
        offsets = np.concatenate([[0.0], np.cumsum(lengths[:-1])])
        zone_to_offset = {z: offsets[i] for i, z in enumerate(arm_order)}
        zone_to_length = {z: arm_lengths[z] for z in arm_order}
        df["pos_1d"] = df[arm_position_column].astype(float) * df[zone_column].map(
            zone_to_length
        ) + df[zone_column].map(zone_to_offset)
    else:
        zone_to_offset = {zone: float(i) for i, zone in enumerate(arm_order)}
        df["pos_1d"] = df[arm_position_column].astype(float) + df[zone_column].map(zone_to_offset)

    logger.info(
        "Serialized %d/%d frames to 1D (arms: %s, physical=%s)",
        len(df),
        len(trajectory),
        arm_order,
        arm_lengths is not None,
    )
    return df


def assign_traversal_direction(
    trajectory: pd.DataFrame,
    arm_order: list[str],
    zone_column: str = "zone",
    arm_position_column: str = "arm_position",
    direction_threshold: float = 0.5,
    arm_lengths: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Split arm traversals into forward/reverse directional segments.

    Each contiguous run of frames within the same arm is a "traversal".
    Direction is determined by the first ``arm_position`` value:
    < threshold → forward, >= threshold → reverse.

    Parameters
    ----------
    trajectory:
        DataFrame from ``serialize_arm_position`` with columns
        ``zone_column``, ``arm_position_column``, ``arm_index``,
        ``pos_1d``, ``frame_index``.
    arm_order:
        Original arm order (e.g. ["Arm_1", "Arm_2", ...]).
    zone_column:
        Column with zone labels.
    arm_position_column:
        Column with within-arm position (0-1).
    direction_threshold:
        Position threshold for direction classification.
    arm_lengths:
        Optional dict mapping zone name to physical length.  When
        provided, ``pos_1d`` is recomputed in physical units.

    Returns
    -------
    tuple of (DataFrame with direction columns and recomputed pos_1d,
              effective_arm_order list e.g. ["Arm_1_fwd", "Arm_1_rev", ...])
    """
    df = trajectory.sort_values("frame_index").copy()

    # Detect traversal boundaries: zone changes or frame_index gaps > 1
    zone_changed = df[zone_column] != df[zone_column].shift(1)
    frame_gap = df["frame_index"].diff() > 1
    boundary = zone_changed | frame_gap
    # First row is always a boundary
    boundary.iloc[0] = True
    df["traversal_id"] = boundary.cumsum()

    # Classify each traversal by its first arm_position
    first_pos = df.groupby("traversal_id")[arm_position_column].first()
    traversal_dir = (first_pos < direction_threshold).map({True: "fwd", False: "rev"})
    df["direction"] = df["traversal_id"].map(traversal_dir)

    # Build effective arm order: [Arm_1_fwd, Arm_1_rev, Arm_2_fwd, ...]
    effective_order = []
    for arm in arm_order:
        effective_order.append(f"{arm}_fwd")
        effective_order.append(f"{arm}_rev")

    # Create directional zone label and recompute arm_index + pos_1d
    df["zone_dir"] = df[zone_column] + "_" + df["direction"]
    zone_dir_to_idx = {name: i for i, name in enumerate(effective_order)}
    df["arm_index"] = df["zone_dir"].map(zone_dir_to_idx)

    if arm_lengths is not None:
        # Each directional segment has the same physical length as its parent arm
        effective_lengths = {}
        for arm in arm_order:
            L = arm_lengths[arm]
            effective_lengths[f"{arm}_fwd"] = L
            effective_lengths[f"{arm}_rev"] = L
        seg_lengths = [effective_lengths[t] for t in effective_order]
        offsets = np.concatenate([[0.0], np.cumsum(seg_lengths[:-1])])
        zone_dir_to_offset = {name: offsets[i] for i, name in enumerate(effective_order)}
        zone_dir_to_length = {name: effective_lengths[name] for name in effective_order}
        df["pos_1d"] = df[arm_position_column].astype(float) * df["zone_dir"].map(
            zone_dir_to_length
        ) + df["zone_dir"].map(zone_dir_to_offset)
    else:
        df["pos_1d"] = df[arm_position_column].astype(float) + df["arm_index"].astype(float)

    logger.info(
        "Direction splitting: %d traversals, effective segments: %s",
        df["traversal_id"].nunique(),
        effective_order,
    )
    return df, effective_order


def filter_complete_traversals(
    trajectory: pd.DataFrame,
    full_trajectory: pd.DataFrame,
    arm_order: list[str],
    zone_column: str = "zone",
) -> pd.DataFrame:
    """Remove traversals where the animal did not cross from one room to another.

    For each traversal, the zone immediately before entering and after
    leaving the arm is looked up in *full_trajectory*.  A traversal is
    "complete" when both zones are rooms (not arms) and they differ
    (i.e. the animal went Room A → Arm → Room B, not Room A → Arm → Room A).

    If ``traversal_id`` is not yet present (e.g. ``split_by_direction``
    is False), traversal boundaries are detected automatically.

    Parameters
    ----------
    trajectory:
        Arm-only DataFrame with ``frame_index`` column (and optionally
        ``traversal_id`` from ``assign_traversal_direction``).
    full_trajectory:
        Complete trajectory including room frames, with ``frame_index``
        and *zone_column*.
    arm_order:
        List of arm zone names (anything else is treated as a room).
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

    arm_set = set(arm_order)

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
            entry_zone not in arm_set
            and exit_zone not in arm_set
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
    """Compute 1D speed along the arm axis.

    Speed is computed as |delta(pos_1d)| / delta(time) over a window
    of *actual* frame indices, not entry offsets.  Because room frames
    are absent from the arm-only trajectory, consecutive entries can
    have large frame_index gaps (room visits).  Using entry offsets
    would compute distance/time over artificially long intervals,
    systematically underestimating speed in frequently-exited arms.

    Parameters
    ----------
    trajectory:
        DataFrame with pos_1d, arm_index, unix_time, frame_index columns.
        Must be arm-only (output of ``serialize_arm_position``).
    window_frames:
        Look-ahead window in *real* frame numbers.

    Returns
    -------
    pd.DataFrame with ``speed_1d`` column added.
    """
    df = trajectory.sort_values("frame_index").copy()
    pos = df["pos_1d"].values
    t = df["unix_time"].values
    arm_idx = df["arm_index"].values
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
    same_arm = arm_idx[end_indices] == arm_idx
    frame_gap = frame_idx[end_indices] - frame_idx
    # Valid: same arm, positive time, and frame gap not absurdly large
    # (allow up to 3x window to tolerate a few missing frames)
    valid = (end_indices > np.arange(n)) & same_arm & (dt > 0) & (frame_gap <= window_frames * 3)

    speed = np.zeros(n)
    speed[valid] = dpos[valid] / dt[valid]
    df["speed_1d"] = speed
    return df


def filter_arm_by_speed(
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
