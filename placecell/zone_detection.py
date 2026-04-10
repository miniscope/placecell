"""Zone detection from position tracking data."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from placecell.behavior import interpolate_behavior_onto_neural, remove_position_jumps
from placecell.geometry import (
    closest_point_on_polyline_prepared,
    get_zone_probabilities,
    is_valid_transition,
    load_zone_config,
    position_along_polyline_prepared,
    prepare_zone_geometry,
)
from placecell.log import init_logger

logger = init_logger(__name__)
_EMPTY_TRANSITIONS: frozenset[str] = frozenset()


def detect_zones(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    zone_polygons: dict[str, np.ndarray],
    zone_types: dict[str, str],
    zone_graph: dict[str, list[str]],
    sample_rate_hz: float,
    arm_max_distance: float = 60.0,
    min_confidence: float = 0.5,
    min_confidence_forbidden: float = 0.8,
    min_seconds_same: float = 0.05,
    min_seconds_forbidden: float = 0.15,
    room_decay_power: float = 2.0,
    arm_decay_power: float = 0.5,
    soft_boundary: bool = True,
    progress_bar: type | None = None,
) -> pd.DataFrame:
    """Detect zones from x/y position arrays.

    Parameters
    ----------
    x_coords, y_coords:
        Position arrays (one value per frame).
    zone_polygons:
        Dict mapping zone name to polygon/polyline array.
    zone_types:
        Dict mapping zone name to "room" or "arm".
    zone_graph:
        Bidirectional adjacency dict (zone -> list of connected zones).
    sample_rate_hz:
        Sampling rate of ``x_coords``/``y_coords`` (Hz). Used to convert
        ``min_seconds_*`` thresholds into frame counts internally.
    arm_max_distance:
        Maximum distance from arm centerline for classification.
    min_confidence:
        Minimum confidence for zone transitions.
    min_confidence_forbidden:
        Minimum confidence for forbidden (non-adjacent) transitions.
    min_seconds_same:
        Minimum dwell time (seconds) in the current zone before allowing
        a transition.
    min_seconds_forbidden:
        Minimum dwell time (seconds) of consecutive forbidden-zone
        evidence before accepting a forbidden transition.
    room_decay_power:
        Exponent for room boundary probability decay.
    arm_decay_power:
        Exponent for arm boundary probability decay.
    soft_boundary:
        Use fuzzy distance-based boundaries.

    Returns
    -------
    DataFrame with columns: zone, x_pinned, y_pinned, arm_position.
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    min_frames_same = max(1, int(round(min_seconds_same * sample_rate_hz)))
    min_frames_forbidden = max(1, int(round(min_seconds_forbidden * sample_rate_hz)))
    prepared_geometry = prepare_zone_geometry(zone_polygons, zone_types)
    valid_transitions = {
        zone_name: {
            other_zone
            for other_zone in zone_polygons
            if is_valid_transition(zone_name, other_zone, zone_graph, zone_types)
        }
        for zone_name in zone_polygons
    }

    current_zone = None
    frames_in_current_zone = 0
    forbidden_candidate_zone = None
    forbidden_consecutive_frames = 0

    zones = []
    x_pinned_list = []
    y_pinned_list = []
    arm_positions = []

    iterator = zip(x_coords, y_coords)
    if progress_bar is not None:
        iterator = progress_bar(iterator, total=len(x_coords), desc="Detecting zones")

    for x, y in iterator:
        zone_probs = get_zone_probabilities(
            x,
            y,
            zone_polygons,
            zone_types,
            arm_max_distance=arm_max_distance,
            soft_boundary=soft_boundary,
            room_decay_power=room_decay_power,
            arm_decay_power=arm_decay_power,
            normalize=True,
            prepared_geometry=prepared_geometry,
        )

        if zone_probs:
            new_zone = max(zone_probs, key=zone_probs.get)
            pred_confidence = zone_probs[new_zone]
        else:
            new_zone = None
            pred_confidence = 0.0

        # Zone transition logic with graph constraints
        if (
            current_zone is not None
            and new_zone is not None
            and new_zone not in valid_transitions.get(current_zone, _EMPTY_TRANSITIONS)
        ):
            best_alt_zone = None
            best_alt_prob = 0.0
            for alt_zone, alt_prob in zone_probs.items():
                if (
                    alt_prob > best_alt_prob
                    and alt_zone != new_zone
                    and alt_zone in valid_transitions.get(current_zone, _EMPTY_TRANSITIONS)
                ):
                    best_alt_zone = alt_zone
                    best_alt_prob = alt_prob

            if (
                best_alt_zone is not None
                and best_alt_prob >= min_confidence
                and best_alt_prob > pred_confidence
            ):
                new_zone = best_alt_zone
                pred_confidence = best_alt_prob

        if current_zone is None:
            if new_zone and pred_confidence >= min_confidence:
                current_zone = new_zone
                frames_in_current_zone = 1
        elif new_zone == current_zone:
            frames_in_current_zone += 1
            forbidden_candidate_zone = None
            forbidden_consecutive_frames = 0
        else:
            is_valid = new_zone in valid_transitions.get(current_zone, _EMPTY_TRANSITIONS)

            if is_valid:
                if pred_confidence >= min_confidence and frames_in_current_zone >= min_frames_same:
                    current_zone = new_zone
                    frames_in_current_zone = 1
                    forbidden_candidate_zone = None
                    forbidden_consecutive_frames = 0
                else:
                    frames_in_current_zone += 1
            else:
                if pred_confidence >= min_confidence_forbidden:
                    if new_zone == forbidden_candidate_zone:
                        forbidden_consecutive_frames += 1
                    else:
                        forbidden_candidate_zone = new_zone
                        forbidden_consecutive_frames = 1

                    if (
                        forbidden_consecutive_frames >= min_frames_forbidden
                        and frames_in_current_zone >= min_frames_same
                    ):
                        current_zone = new_zone
                        frames_in_current_zone = 1
                        forbidden_candidate_zone = None
                        forbidden_consecutive_frames = 0
                    else:
                        frames_in_current_zone += 1
                else:
                    forbidden_candidate_zone = None
                    forbidden_consecutive_frames = 0
                    frames_in_current_zone += 1

        pred_label = current_zone if current_zone else "Unknown"

        arm_position = None
        pinned_x, pinned_y = x, y
        if zone_types.get(pred_label) == "arm" and pred_label in zone_polygons:
            geometry = prepared_geometry[pred_label]
            arm_position = position_along_polyline_prepared((x, y), geometry)
            pinned_pt = closest_point_on_polyline_prepared((x, y), geometry)
            pinned_x, pinned_y = float(pinned_pt[0]), float(pinned_pt[1])

        zones.append(pred_label)
        x_pinned_list.append(pinned_x)
        y_pinned_list.append(pinned_y)
        arm_positions.append(arm_position)

    return pd.DataFrame(
        {
            "zone": zones,
            "x_pinned": x_pinned_list,
            "y_pinned": y_pinned_list,
            "arm_position": arm_positions,
        }
    )


def backup_file(path: str | Path) -> Path:
    """Move *path* to a ``backup/`` subdirectory with an incrementing suffix.

    Example: ``data.csv`` → ``backup/data_1.csv`` (then ``_2``, ``_3``, …).

    Returns
    -------
    Path to the backup copy.
    """
    path = Path(path)
    backup_dir = path.parent / "backup"
    backup_dir.mkdir(exist_ok=True)
    stem, suffix = path.stem, path.suffix
    i = 1
    while (backup_dir / f"{stem}_{i}{suffix}").exists():
        i += 1
    dest = backup_dir / f"{stem}_{i}{suffix}"
    shutil.move(str(path), str(dest))
    logger.info("Backed up %s → %s", path.name, dest)
    return dest


# Zone color palette (BGR for OpenCV)
_ZONE_COLORS = [
    (100, 200, 100),  # green
    (100, 200, 200),  # yellow-green
    (200, 200, 100),  # cyan-ish
    (100, 100, 255),  # red
    (255, 100, 100),  # blue
    (100, 255, 255),  # yellow
    (255, 200, 100),  # light blue
    (200, 100, 200),  # purple
    (100, 255, 100),  # bright green
    (255, 255, 100),  # cyan
]


def export_zone_video(
    video_path: str | Path,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    result: pd.DataFrame,
    zone_polygons: dict[str, np.ndarray],
    zone_types: dict[str, str],
    output_path: str | Path,
    fps: float | None = None,
    interpolate: int = 5,
    playback_speed: float = 10.0,
    progress_bar: type | None = None,
) -> None:
    """Render zone detection results as an annotated video.

    Only the currently detected zone is overlaid on each frame.
    The animal's position (red dot) and arm-pinned position (green dot)
    are drawn on top.

    Parameters
    ----------
    video_path:
        Path to the behavior video.
    x_coords, y_coords:
        Raw position arrays (one per frame).
    result:
        Output of ``detect_zones()`` with columns
        ``zone``, ``x_pinned``, ``y_pinned``, ``arm_position``.
    zone_polygons:
        Dict mapping zone name to polygon/polyline ndarray.
    zone_types:
        Dict mapping zone name to ``"room"`` or ``"arm"``.
    output_path:
        Output video file path.
    fps:
        Output video frame rate. If None, uses the source video's fps.
    interpolate:
        Frame subsampling factor. Only every *interpolate*-th frame is
        written to the output video (default 5).
    playback_speed:
        Playback speed multiplier relative to the source video timeline.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV is required for video export. " "Install with: pip install opencv-python"
        ) from None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    step = max(interpolate, 1)
    out_fps = fps * playback_speed / step

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (width, height))

    overlay_color = (200, 200, 100)  # cyan-ish BGR

    zones = result["zone"].values
    x_pinned = result["x_pinned"].values
    y_pinned = result["y_pinned"].values
    n_frames = len(zones)
    zone_draw_data = {
        zone_name: (np.asarray(points, dtype=np.int32), zone_types.get(zone_name) == "room")
        for zone_name, points in zone_polygons.items()
    }

    frame_iter = range(n_frames)
    if progress_bar is not None:
        frame_iter = progress_bar(frame_iter, desc="Exporting video")

    for frame_idx in frame_iter:
        if frame_idx % step != 0:
            ret = cap.grab()
            if not ret:
                break
            continue
        ret, frame = cap.read()
        if not ret:
            break

        zone_label = zones[frame_idx]
        x, y = x_coords[frame_idx], y_coords[frame_idx]
        xp, yp = x_pinned[frame_idx], y_pinned[frame_idx]

        # Draw only the current zone overlay
        if zone_label in zone_draw_data:
            pts, is_room = zone_draw_data[zone_label]
            if is_room and len(pts) >= 3:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], overlay_color)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.polylines(frame, [pts], True, overlay_color, 3)
            else:
                cv2.polylines(frame, [pts], False, overlay_color, 3)

        # Green dot at pinned position (arm regression)
        if zone_types.get(zone_label) == "arm" and not (np.isnan(xp) or np.isnan(yp)):
            cv2.circle(frame, (int(xp), int(yp)), 12, (0, 255, 0), -1)

        # Red dot at raw position
        if not (np.isnan(x) or np.isnan(y)):
            cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), -1)

        # Zone label text
        cv2.putText(
            frame,
            zone_label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            overlay_color,
            2,
        )

        # Frame number
        cv2.putText(
            frame,
            str(frame_idx),
            (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

    cap.release()
    writer.release()
    n_written = (n_frames + step - 1) // step
    logger.info(
        "Zone video saved to %s (%d frames, source_fps=%.3f, out_fps=%.3f, "
        "step=%d, playback=%.2fx)",
        output_path,
        n_written,
        fps,
        out_fps,
        step,
        playback_speed,
    )


def detect_zones_from_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    zone_config_path: str | Path,
    behavior_timestamp_csv: str | Path,
    neural_timestamp_csv: str | Path,
    bodypart: str | None = None,
    arm_max_distance: float = 60.0,
    min_confidence: float = 0.5,
    min_confidence_forbidden: float = 0.8,
    min_seconds_same: float = 0.05,
    min_seconds_forbidden: float = 0.15,
    room_decay_power: float = 2.0,
    arm_decay_power: float = 0.5,
    soft_boundary: bool = True,
    hampel_window_frames: int = 7,
    hampel_n_sigmas: float = 3.0,
    zone_connections: dict[str, list[str]] | None = None,
    video_path: str | Path | None = None,
    video_output: str | Path | None = None,
    interpolate: int = 5,
    playback_speed: float = 10.0,
    progress_bar: type | None = None,
) -> None:
    """Detect zones from a DLC-format tracking CSV.

    The pipeline is: read raw behavior → Hampel jump removal → linear
    interpolation onto the neural timestamp grid → project the
    interpolated coordinates onto the maze graph. The resulting CSV is
    indexed by **neural frame** so that all downstream analysis lives on
    a single common clock.

    Parameters
    ----------
    input_csv:
        Input behavior CSV with x, y coordinates (DLC multi-index format).
    output_csv:
        Output CSV with zone columns added (neural-rate).
    zone_config_path:
        Path to combined zone config YAML.
    behavior_timestamp_csv:
        CSV with ``frame_index, unix_time`` columns for the behavior CSV.
    neural_timestamp_csv:
        CSV with ``frame, timestamp_first, timestamp_last`` columns; the
        midpoint of those two timestamps is used as the neural sample time.
    bodypart:
        Body part name to use. If ``None``, uses the first bodypart found.
    arm_max_distance:
        Maximum distance from arm centerline for arm classification.
    min_confidence:
        Minimum confidence for zone transitions.
    min_confidence_forbidden:
        Minimum confidence for forbidden (non-adjacent) transitions.
    min_seconds_same:
        Minimum dwell time (seconds) in the current zone before allowing
        a transition. Converted to frames at the neural sample rate.
    min_seconds_forbidden:
        Minimum dwell time (seconds) of consecutive forbidden-zone
        evidence before accepting a forbidden transition.
    room_decay_power, arm_decay_power, soft_boundary:
        Soft-boundary tuning for the per-zone probability function.
    hampel_window_frames:
        Centered rolling window (frames at behavior rate) for the
        raw-position Hampel filter applied before interpolation.
    hampel_n_sigmas:
        MAD-scaled threshold for the Hampel filter.
    zone_connections:
        Zone adjacency graph override.
    video_path, video_output, interpolate, playback_speed:
        Optional zone-detection validation video parameters.
    """
    logger.info("Loading position data from %s", input_csv)

    # Read DLC-format behavior CSV
    df = pd.read_csv(input_csv, header=[0, 1, 2], index_col=0)
    scorer = df.columns.get_level_values(0)[0]
    if bodypart is None:
        bodypart = df.columns.get_level_values(1)[0]
    x_col = (scorer, bodypart, "x")
    y_col = (scorer, bodypart, "y")
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(
            f"Could not find x, y columns for bodypart '{bodypart}'. "
            f"Available: {sorted(set(df.columns.get_level_values(1)))}"
        )

    behavior_frame_index = df.index.to_numpy()
    x_raw = df[x_col].values.astype(float)
    y_raw = df[y_col].values.astype(float)

    # Hampel-filter the raw behavior trajectory at behavior rate, then
    # linearly interpolate onto the neural timestamp grid.
    cleaned_xy, n_jumps = remove_position_jumps(
        pd.DataFrame({"x": x_raw, "y": y_raw}),
        window_frames=hampel_window_frames,
        n_sigmas=hampel_n_sigmas,
    )
    logger.info(
        "Hampel jump removal: %d frames interpolated (window=%d, n_sigmas=%.1f)",
        n_jumps,
        hampel_window_frames,
        hampel_n_sigmas,
    )

    behavior_ts = pd.read_csv(behavior_timestamp_csv)
    if "frame_index" not in behavior_ts.columns or "unix_time" not in behavior_ts.columns:
        raise ValueError(
            f"behavior_timestamp_csv must have 'frame_index' and 'unix_time' columns. "
            f"Got: {list(behavior_ts.columns)}"
        )
    behavior_at_beh = pd.DataFrame(
        {
            "frame_index": behavior_frame_index,
            "x": cleaned_xy["x"].to_numpy(),
            "y": cleaned_xy["y"].to_numpy(),
        }
    ).merge(behavior_ts[["frame_index", "unix_time"]], on="frame_index", how="left")

    neural_ts = pd.read_csv(neural_timestamp_csv)
    if not {"frame", "timestamp_first"}.issubset(neural_ts.columns):
        raise ValueError(
            f"neural_timestamp_csv must have 'frame' and 'timestamp_first' "
            f"columns. Got: {list(neural_ts.columns)}"
        )
    # timestamp_first is the canonical neural sample time; timestamp_last
    # is the end-of-exposure stamp and is occasionally noisy, so we ignore
    # it here.
    neural_time = neural_ts["timestamp_first"].to_numpy()

    interpolated = interpolate_behavior_onto_neural(
        behavior_at_beh,
        neural_time,
        columns=["x", "y"],
    )
    # After validation inside interpolate_behavior_onto_neural, frame_index
    # contains the original neural frame numbers of surviving (non-excluded)
    # frames, and neural_time is the corresponding monotonic timestamp array.
    neural_time = interpolated["neural_time"].to_numpy()
    kept_frame_idx = interpolated["frame_index"].to_numpy()
    n_uncovered = int(interpolated[["x", "y"]].isna().any(axis=1).sum())
    if n_uncovered:
        logger.info(
            "Interpolation: %d/%d neural frames have no behavior coverage",
            n_uncovered,
            len(interpolated),
        )
    x_all = interpolated["x"].to_numpy(copy=True)
    y_all = interpolated["y"].to_numpy(copy=True)
    n_all = len(x_all)

    # Only run the state machine on frames that have valid (non-NaN)
    # coordinates. Frames with no behavior coverage are left as
    # Unknown / NaN in the output — no fabricated positions.
    valid_mask = np.isfinite(x_all) & np.isfinite(y_all)
    x_valid = x_all[valid_mask]
    y_valid = y_all[valid_mask]

    zone_polygons, zone_types, file_graph = load_zone_config(zone_config_path)
    if zone_connections is not None:
        from placecell.geometry import build_zone_graph

        zone_graph = build_zone_graph(zone_connections)
    else:
        zone_graph = file_graph

    valid_neural_time = neural_time[valid_mask]
    sample_rate_hz = (
        float(1.0 / np.median(np.diff(valid_neural_time))) if len(valid_neural_time) > 1 else 20.0
    )
    logger.info(
        "Loaded %d neural samples (%d with behavior coverage, ~%.2f Hz), %d zones",
        n_all,
        len(x_valid),
        sample_rate_hz,
        len(zone_polygons),
    )
    result_valid = detect_zones(
        x_valid,
        y_valid,
        zone_polygons,
        zone_types,
        zone_graph,
        sample_rate_hz=sample_rate_hz,
        arm_max_distance=arm_max_distance,
        min_confidence=min_confidence,
        min_confidence_forbidden=min_confidence_forbidden,
        min_seconds_same=min_seconds_same,
        min_seconds_forbidden=min_seconds_forbidden,
        room_decay_power=room_decay_power,
        arm_decay_power=arm_decay_power,
        soft_boundary=soft_boundary,
        progress_bar=progress_bar,
    )

    # Expand valid-only results back to full length, filling gaps with
    # NaN / "Unknown" so no data is fabricated for uncovered frames.
    zone_out = np.full(n_all, "Unknown", dtype=object)
    x_pinned_out = np.full(n_all, np.nan)
    y_pinned_out = np.full(n_all, np.nan)
    arm_pos_out = np.full(n_all, np.nan, dtype=object)
    zone_out[valid_mask] = result_valid["zone"].to_numpy()
    x_pinned_out[valid_mask] = result_valid["x_pinned"].to_numpy()
    y_pinned_out[valid_mask] = result_valid["y_pinned"].to_numpy()
    arm_pos_out[valid_mask] = result_valid["arm_position"].to_numpy()

    output_data = {
        (scorer, bodypart, "x"): x_all,
        (scorer, bodypart, "y"): y_all,
        (scorer, bodypart, "x_pinned"): x_pinned_out,
        (scorer, bodypart, "y_pinned"): y_pinned_out,
        (scorer, bodypart, "zone"): zone_out,
        (scorer, bodypart, "arm_position"): arm_pos_out,
        (scorer, bodypart, "neural_time"): neural_time,
    }
    output_df = pd.DataFrame(output_data, index=kept_frame_idx)
    output_df.to_csv(output_csv)
    logger.info("Zone detection results saved to %s", output_csv)

    zone_counts = pd.Series(zone_out).value_counts()
    logger.info("Zone distribution:")
    for zone, count in zone_counts.items():
        pct = 100 * count / len(zone_out)
        logger.info("  %s: %d frames (%.1f%%)", zone, count, pct)

    if video_path is not None:
        if video_output is None:
            video_output = Path(output_csv).parent / "zone_detection.mp4"
        # Video export needs full-length arrays (one per neural frame).
        result_full = pd.DataFrame(
            {
                "zone": zone_out,
                "x_pinned": x_pinned_out,
                "y_pinned": y_pinned_out,
                "arm_position": arm_pos_out,
            }
        )
        export_zone_video(
            video_path=video_path,
            x_coords=x_all,
            y_coords=y_all,
            result=result_full,
            zone_polygons=zone_polygons,
            zone_types=zone_types,
            output_path=video_output,
            interpolate=interpolate,
            playback_speed=playback_speed,
            progress_bar=progress_bar,
        )
