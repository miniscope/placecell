"""Zone detection from position tracking data."""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from placecell.geometry import (
    closest_point_on_polyline,
    get_zone_probabilities,
    is_valid_transition,
    load_zone_config,
    position_along_polyline,
)
from placecell.logging import init_logger

logger = init_logger(__name__)


def detect_zones(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    zone_polygons: dict[str, np.ndarray],
    zone_types: dict[str, str],
    zone_graph: dict[str, list[str]],
    arm_max_distance: float = 60.0,
    min_confidence: float = 0.5,
    min_confidence_forbidden: float = 0.8,
    min_frames_same: int = 1,
    min_frames_forbidden: int = 3,
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
    arm_max_distance:
        Maximum distance from arm centerline for classification.
    min_confidence:
        Minimum confidence for zone transitions.
    min_confidence_forbidden:
        Minimum confidence for forbidden (non-adjacent) transitions.
    min_frames_same:
        Minimum frames in current zone before allowing transition.
    min_frames_forbidden:
        Minimum consecutive frames for forbidden transition override.
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
        )

        if zone_probs:
            new_zone = max(zone_probs, key=zone_probs.get)
            pred_confidence = zone_probs[new_zone]
        else:
            new_zone = None
            pred_confidence = 0.0

        # Zone transition logic with graph constraints
        if current_zone is not None and new_zone is not None:
            if not is_valid_transition(current_zone, new_zone, zone_graph, zone_types):
                # Check for valid alternatives
                valid_alternatives = []
                for alt_zone, alt_prob in zone_probs.items():
                    if (
                        alt_prob > 0
                        and alt_zone != new_zone
                        and is_valid_transition(current_zone, alt_zone, zone_graph, zone_types)
                    ):
                        valid_alternatives.append((alt_zone, alt_prob))

                if valid_alternatives:
                    best_alt_zone = max(valid_alternatives, key=lambda x: x[1])
                    if best_alt_zone[1] >= min_confidence and best_alt_zone[1] > pred_confidence:
                        new_zone = best_alt_zone[0]
                        pred_confidence = best_alt_zone[1]

        if current_zone is None:
            if new_zone and pred_confidence >= min_confidence:
                current_zone = new_zone
                frames_in_current_zone = 1
        elif new_zone == current_zone:
            frames_in_current_zone += 1
            forbidden_candidate_zone = None
            forbidden_consecutive_frames = 0
        else:
            is_valid = is_valid_transition(current_zone, new_zone, zone_graph, zone_types)

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

        # Calculate arm-pinned coordinates
        arm_position = None
        pinned_x, pinned_y = x, y
        if zone_types.get(pred_label) == "arm" and pred_label in zone_polygons:
            arm_position = position_along_polyline((x, y), zone_polygons[pred_label])
            pinned_pt = closest_point_on_polyline((x, y), zone_polygons[pred_label])
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
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV is required for video export. "
            "Install with: pip install opencv-python"
        ) from None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    # Adjust output fps to preserve real-time playback speed
    out_fps = fps / max(interpolate, 1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (width, height))

    overlay_color = (200, 200, 100)  # cyan-ish BGR

    zones = result["zone"].values
    x_pinned = result["x_pinned"].values
    y_pinned = result["y_pinned"].values
    n_frames = len(zones)

    frame_iter = range(n_frames)
    if progress_bar is not None:
        frame_iter = progress_bar(frame_iter, desc="Exporting video")

    for frame_idx in frame_iter:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames for subsampling
        if frame_idx % max(interpolate, 1) != 0:
            continue

        zone_label = zones[frame_idx]
        x, y = x_coords[frame_idx], y_coords[frame_idx]
        xp, yp = x_pinned[frame_idx], y_pinned[frame_idx]

        # Draw only the current zone overlay
        if zone_label in zone_polygons:
            pts = np.array(zone_polygons[zone_label], dtype=np.int32)
            if zone_types.get(zone_label) == "room" and len(pts) >= 3:
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
            frame, zone_label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, overlay_color, 2,
        )

        # Frame number
        cv2.putText(
            frame, str(frame_idx), (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        writer.write(frame)

    cap.release()
    writer.release()
    n_written = n_frames // max(interpolate, 1)
    logger.info("Zone video saved to %s (%d frames)", output_path, n_written)


def detect_zones_from_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    zone_config_path: str | Path,
    bodypart: str | None = None,
    arm_max_distance: float = 60.0,
    min_confidence: float = 0.5,
    min_confidence_forbidden: float = 0.8,
    min_frames_same: int = 1,
    min_frames_forbidden: int = 3,
    room_decay_power: float = 2.0,
    arm_decay_power: float = 0.5,
    soft_boundary: bool = True,
    zone_connections: dict[str, list[str]] | None = None,
    video_path: str | Path | None = None,
    video_output: str | Path | None = None,
    interpolate: int = 5,
    progress_bar: type | None = None,
) -> None:
    """Detect zones from a DLC-format tracking CSV.

    Parameters
    ----------
    input_csv:
        Input CSV with x, y coordinates (DLC multi-index format).
    output_csv:
        Output CSV with zone information added.
    zone_config_path:
        Path to combined zone config YAML.
    bodypart:
        Body part name to use. If None, uses the first bodypart found.
    arm_max_distance:
        Maximum distance from arm centerline for classification.
    min_confidence:
        Minimum confidence for zone transitions.
    min_confidence_forbidden:
        Minimum confidence for forbidden transitions.
    min_frames_same:
        Minimum frames in zone before allowing transition.
    min_frames_forbidden:
        Minimum consecutive frames for forbidden transition.
    room_decay_power:
        Exponent for room boundary probability decay.
    arm_decay_power:
        Exponent for arm boundary probability decay.
    soft_boundary:
        Use fuzzy distance-based boundaries.
    zone_connections:
        Zone adjacency graph (room → list of arms).  If provided,
        overrides any connections in the zone config file.
    video_path:
        Path to behavior video. If provided, exports an annotated video.
    video_output:
        Output video path. Defaults to ``zone_detection.mp4`` next to output CSV.
    interpolate:
        Frame subsampling factor for video export (default 5).
    """
    logger.info("Loading position data from %s", input_csv)

    # Read DLC-format CSV
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

    x_coords = df[x_col].values.astype(float)
    y_coords = df[y_col].values.astype(float)
    # Load zone config (geometry + optional file-level connections)
    zone_polygons, zone_types, file_graph = load_zone_config(zone_config_path)

    # Data config connections override file connections
    if zone_connections is not None:
        from placecell.geometry import build_zone_graph

        zone_graph = build_zone_graph(zone_connections)
    else:
        zone_graph = file_graph

    logger.info("Loaded %d position samples, %d zones", len(x_coords), len(zone_polygons))
    result = detect_zones(
        x_coords,
        y_coords,
        zone_polygons,
        zone_types,
        zone_graph,
        arm_max_distance=arm_max_distance,
        min_confidence=min_confidence,
        min_confidence_forbidden=min_confidence_forbidden,
        min_frames_same=min_frames_same,
        min_frames_forbidden=min_frames_forbidden,
        room_decay_power=room_decay_power,
        arm_decay_power=arm_decay_power,
        soft_boundary=soft_boundary,
        progress_bar=progress_bar,
    )

    # Build output in DLC multi-index format
    output_data = {
        (scorer, bodypart, "x"): x_coords,
        (scorer, bodypart, "y"): y_coords,
        (scorer, bodypart, "x_pinned"): result["x_pinned"].values,
        (scorer, bodypart, "y_pinned"): result["y_pinned"].values,
        (scorer, bodypart, "zone"): result["zone"].values,
        (scorer, bodypart, "arm_position"): result["arm_position"].values,
    }

    # Include likelihood if it exists
    likelihood_col = (scorer, bodypart, "likelihood")
    if likelihood_col in df.columns:
        output_data[likelihood_col] = df[likelihood_col].values

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv)
    logger.info("Zone detection results saved to %s", output_csv)

    # Print zone statistics
    zone_counts = result["zone"].value_counts()
    logger.info("Zone distribution:")
    for zone, count in zone_counts.items():
        pct = 100 * count / len(result)
        logger.info("  %s: %d frames (%.1f%%)", zone, count, pct)

    # Export annotated video
    if video_path is not None:
        if video_output is None:
            video_output = Path(output_csv).parent / "zone_detection.mp4"
        export_zone_video(
            video_path=video_path,
            x_coords=x_coords,
            y_coords=y_coords,
            result=result,
            zone_polygons=zone_polygons,
            zone_types=zone_types,
            output_path=video_output,
            interpolate=interpolate,
            progress_bar=progress_bar,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect zones from tracking CSV")
    parser.add_argument("--input", "-i", required=True, help="Input tracking CSV")
    parser.add_argument("--output", "-o", required=True, help="Output CSV")
    parser.add_argument(
        "--zone-config", "-z", required=True, help="Zone config YAML"
    )
    parser.add_argument("--bodypart", "-b", default=None, help="Body part name")
    parser.add_argument(
        "--arm-max-distance", type=float, default=60.0, help="Max arm distance (px)"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.5, help="Min transition confidence"
    )
    parser.add_argument(
        "--min-confidence-forbidden",
        type=float,
        default=0.8,
        help="Min forbidden transition confidence",
    )
    parser.add_argument(
        "--min-frames-same", type=int, default=1, help="Min frames before transition"
    )
    parser.add_argument(
        "--min-frames-forbidden",
        type=int,
        default=3,
        help="Min consecutive frames for forbidden transition",
    )

    args = parser.parse_args()

    detect_zones_from_csv(
        input_csv=args.input,
        output_csv=args.output,
        zone_config_path=args.zone_config,
        bodypart=args.bodypart,
        arm_max_distance=args.arm_max_distance,
        min_confidence=args.min_confidence,
        min_confidence_forbidden=args.min_confidence_forbidden,
        min_frames_same=args.min_frames_same,
        min_frames_forbidden=args.min_frames_forbidden,
    )
