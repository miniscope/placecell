"""Interactive zone definition tool using OpenCV.

Requires ``opencv-python`` (install with ``pip install placecell[zones]``).
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from placecell.logging import init_logger

logger = init_logger(__name__)

# Default color palette for zones (BGR for OpenCV)
_DEFAULT_COLORS = [
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


def define_zones(
    video_path: str,
    output_file: str = "zone_config.yaml",
    zone_names: list[str] | None = None,
    zone_types: dict[str, str] | None = None,
) -> None:
    """Interactive tool to define zone polygons by clicking points on video.

    Zone connections (adjacency graph) are stored separately in the data
    config YAML under ``zone_connections``, so redefining polygons here
    will not lose the graph topology.

    Parameters
    ----------
    video_path:
        Path to input video file.
    output_file:
        Path to output YAML file (combined format).
    zone_names:
        Ordered list of zone names to define. If None, uses default maze zones.
    zone_types:
        Dict mapping zone name to "room" or "arm". If None, inferred from name.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV is required for define_zones. "
            "Install with: pip install opencv-python"
        ) from None

    if zone_names is None:
        zone_names = [
            "Room_1",
            "Room_2",
            "Room_3",
            "Arm_1",
            "Arm_2",
            "Arm_3",
            "Arm_4",
        ]

    if zone_types is None:
        zone_types = {}
        for name in zone_names:
            if name.startswith("Room"):
                zone_types[name] = "room"
            else:
                zone_types[name] = "arm"

    zone_colors = {
        name: _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i, name in enumerate(zone_names)
    }

    # State
    current_zone_idx = 0
    current_polygon: list[list[int]] = []
    polygons: dict[str, list[list[int]]] = {zone: [] for zone in zone_names}

    frame = None

    def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal current_polygon
        if event == cv2.EVENT_LBUTTONDOWN:
            current_polygon.append([x, y])
            logger.info("  Added point (%d, %d) - %d points total", x, y, len(current_polygon))
            redraw_frame()

    def redraw_frame() -> None:
        nonlocal frame, current_polygon, current_zone_idx
        display_frame = frame.copy()
        current_zone = zone_names[current_zone_idx]

        # Draw all completed polygons/polylines
        for zone, poly in polygons.items():
            if len(poly) >= 2:
                color = zone_colors.get(zone, (128, 128, 128))
                poly_array = np.array(poly, dtype=np.int32)
                if zone_types.get(zone) == "room" and len(poly) >= 3:
                    cv2.fillPoly(display_frame, [poly_array], color)
                    cv2.polylines(display_frame, [poly_array], True, color, 2)
                else:
                    cv2.polylines(display_frame, [poly_array], False, color, 2)
                # Label
                cx = int(np.mean([p[0] for p in poly]))
                cy = int(np.mean([p[1] for p in poly]))
                cv2.putText(
                    display_frame,
                    zone,
                    (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

        # Draw current polygon being defined
        if len(current_polygon) >= 2:
            color = zone_colors.get(current_zone, (128, 128, 128))
            points = np.array(current_polygon, dtype=np.int32)
            is_closed = zone_types.get(current_zone) == "room"
            cv2.polylines(display_frame, [points], is_closed, color, 2)
        for pt in current_polygon:
            color = zone_colors.get(current_zone, (128, 128, 128))
            cv2.circle(display_frame, tuple(pt), 5, color, -1)

        cv2.putText(
            display_frame,
            f"Defining: {current_zone} ({zone_types.get(current_zone, '?')})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_frame,
            f"Points: {len(current_polygon)}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_frame,
            "CLICK: Add | ENTER: Finish | n/p: Next/Prev | r: Reset | s: Save | q: Quit",
            (10, display_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.imshow("Zone Definition Tool", display_frame)

    def save_zones() -> None:
        save_data: dict = {}

        # Build zones dict
        zones_dict: dict = {}
        for zone in zone_names:
            poly = polygons[zone]
            min_points = 3 if zone_types.get(zone) == "room" else 2
            if len(poly) < min_points:
                continue

            # Remove duplicate closing point for arms
            final_poly = poly.copy()
            if zone_types.get(zone) == "arm" and len(final_poly) > 1:
                if final_poly[-1] == final_poly[0]:
                    final_poly = final_poly[:-1]

            zone_entry: dict = {
                "type": zone_types.get(zone, "arm"),
                "points": final_poly,
            }
            zones_dict[zone] = zone_entry

        save_data["zones"] = zones_dict

        with open(output_file, "w") as f:
            yaml.dump(save_data, f, default_flow_style=False, sort_keys=False)

        logger.info("Saved %d zones to %s", len(zones_dict), output_file)

    # Load existing zones from file
    try:
        with open(output_file) as f:
            existing = yaml.safe_load(f) or {}

        for zone in zone_names:
            if zone in existing.get("zones", {}):
                zone_info = existing["zones"][zone]
                pts = zone_info.get("points", [])
                if pts:
                    polygons[zone] = [list(p) for p in pts]
                    logger.info("Loaded existing %s (%d points)", zone, len(pts))
    except FileNotFoundError:
        logger.info("No existing file found. Starting fresh.")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Error opening video: %s", video_path)
        return

    ret, frame = cap.read()
    if not ret:
        logger.error("Error reading video frame")
        cap.release()
        return

    cv2.namedWindow("Zone Definition Tool", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Zone Definition Tool", mouse_callback)

    current_zone = zone_names[current_zone_idx]
    current_polygon = polygons[current_zone].copy() if polygons[current_zone] else []

    logger.info("=== Zone Definition Tool ===")
    logger.info("Current zone: %s", current_zone)
    logger.info("  CLICK: Add point | ENTER: Finish zone | n/p: Next/Prev")
    logger.info("  r: Reset | s: Save | q: Quit")

    redraw_frame()

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == 13 or key == 10:  # ENTER
            min_points = 3 if zone_types.get(current_zone) == "room" else 2
            if len(current_polygon) >= min_points:
                final_poly = current_polygon.copy()
                if zone_types.get(current_zone) == "arm" and len(final_poly) > 1:
                    if final_poly[-1] == final_poly[0]:
                        final_poly = final_poly[:-1]
                polygons[current_zone] = final_poly
                logger.info("Finished %s with %d points", current_zone, len(final_poly))
                current_zone_idx = (current_zone_idx + 1) % len(zone_names)
                current_zone = zone_names[current_zone_idx]
                current_polygon = polygons[current_zone].copy() if polygons[current_zone] else []
                logger.info("Now defining: %s", current_zone)
            else:
                logger.info(
                    "Need at least %d points! Currently have %d",
                    min_points,
                    len(current_polygon),
                )
            redraw_frame()

        elif key == ord("n"):
            current_zone_idx = (current_zone_idx + 1) % len(zone_names)
            current_zone = zone_names[current_zone_idx]
            current_polygon = polygons[current_zone].copy() if polygons[current_zone] else []
            logger.info("Switched to: %s", current_zone)
            redraw_frame()

        elif key == ord("p"):
            current_zone_idx = (current_zone_idx - 1) % len(zone_names)
            current_zone = zone_names[current_zone_idx]
            current_polygon = polygons[current_zone].copy() if polygons[current_zone] else []
            logger.info("Switched to: %s", current_zone)
            redraw_frame()

        elif key == ord("r"):
            current_polygon = []
            logger.info("Reset %s", current_zone)
            redraw_frame()

        elif key == ord("s"):
            # Commit current work before saving
            min_pts = 3 if zone_types.get(current_zone) == "room" else 2
            if len(current_polygon) >= min_pts:
                polygons[current_zone] = current_polygon.copy()
            save_zones()

        elif key == ord("q"):
            break

    # Save on exit
    min_pts = 3 if zone_types.get(current_zone) == "room" else 2
    if len(current_polygon) >= min_pts:
        polygons[current_zone] = current_polygon.copy()
    save_zones()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive zone polygon definition tool")
    parser.add_argument("--video", "-v", required=True, help="Video file")
    parser.add_argument(
        "--output",
        "-o",
        default="zone_config.yaml",
        help="Output YAML file (default: zone_config.yaml)",
    )
    parser.add_argument(
        "--zones",
        nargs="+",
        default=None,
        help="Zone names (default: Room_1 Room_2 Room_3 Arm_1..4)",
    )
    args = parser.parse_args()

    # Infer zone types from names
    ztypes = None
    if args.zones:
        ztypes = {}
        for name in args.zones:
            ztypes[name] = "room" if name.startswith("Room") else "arm"

    define_zones(
        video_path=args.video,
        output_file=args.output,
        zone_names=args.zones,
        zone_types=ztypes,
    )
