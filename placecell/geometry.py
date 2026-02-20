"""Pure-numpy geometric utilities for zone classification."""

from pathlib import Path

import numpy as np
import yaml

from placecell.logging import init_logger

logger = init_logger(__name__)


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    """Check if a point is inside a polygon using ray-casting algorithm.

    Parameters
    ----------
    point:
        (x, y) coordinate to test.
    polygon:
        Array of shape (N, 2) defining the polygon vertices.

    Returns
    -------
    True if the point is inside (or on the boundary of) the polygon.
    """
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _point_to_segment_distance(
    point: tuple[float, float],
    seg_start: np.ndarray,
    seg_end: np.ndarray,
) -> float:
    """Calculate distance from a point to a line segment.

    Parameters
    ----------
    point:
        (x, y) coordinate.
    seg_start, seg_end:
        Endpoints of the segment.
    """
    point = np.array(point, dtype=np.float64)
    seg_start = np.array(seg_start, dtype=np.float64)
    seg_end = np.array(seg_end, dtype=np.float64)

    seg_vec = seg_end - seg_start
    point_vec = point - seg_start

    seg_len_sq = np.dot(seg_vec, seg_vec)
    if seg_len_sq == 0:
        return float(np.linalg.norm(point - seg_start))

    t = np.clip(np.dot(point_vec, seg_vec) / seg_len_sq, 0.0, 1.0)
    closest = seg_start + t * seg_vec
    return float(np.linalg.norm(point - closest))


def signed_distance_to_polygon(
    point: tuple[float, float],
    polygon: np.ndarray,
) -> float:
    """Signed distance from a point to a polygon boundary.

    Parameters
    ----------
    point:
        (x, y) coordinate.
    polygon:
        Array of shape (N, 2) defining the polygon vertices.

    Returns
    -------
    Positive if the point is inside, negative if outside.
    Magnitude is the minimum distance to any edge.
    """
    n = len(polygon)
    min_dist = float("inf")
    for i in range(n):
        j = (i + 1) % n
        d = _point_to_segment_distance(point, polygon[i], polygon[j])
        if d < min_dist:
            min_dist = d

    if point_in_polygon(point, polygon):
        return min_dist
    return -min_dist


def distance_to_polyline(
    point: tuple[float, float],
    polyline: np.ndarray,
    max_distance: float = 20.0,
) -> bool:
    """Check if a point is near a polyline (within max_distance).

    Parameters
    ----------
    point:
        (x, y) coordinate.
    polyline:
        Array of shape (N, 2) defining the polyline.
    max_distance:
        Maximum distance threshold.
    """
    min_dist = float("inf")
    for i in range(len(polyline) - 1):
        d = _point_to_segment_distance(point, polyline[i], polyline[i + 1])
        if d < min_dist:
            min_dist = d
    return min_dist <= max_distance


def min_distance_to_polyline(
    point: tuple[float, float],
    polyline: np.ndarray,
) -> float:
    """Compute minimum distance from a point to a polyline.

    Parameters
    ----------
    point:
        (x, y) coordinate.
    polyline:
        Array of shape (N, 2) defining the polyline.
    """
    min_dist = float("inf")
    for i in range(len(polyline) - 1):
        d = _point_to_segment_distance(point, polyline[i], polyline[i + 1])
        if d < min_dist:
            min_dist = d
    return min_dist


def closest_point_on_polyline(
    point: tuple[float, float],
    polyline: np.ndarray,
) -> np.ndarray:
    """Find the closest point on a polyline to a given point.

    Parameters
    ----------
    point:
        (x, y) coordinate.
    polyline:
        Array of shape (N, 2) defining the polyline.

    Returns
    -------
    Closest point on the polyline as (x, y) numpy array.
    """
    point = np.array(point, dtype=np.float64)
    polyline = np.array(polyline, dtype=np.float64)

    min_dist = float("inf")
    closest = None

    for i in range(len(polyline) - 1):
        p1 = polyline[i]
        p2 = polyline[i + 1]

        seg_vec = p2 - p1
        point_vec = point - p1

        seg_len_sq = np.dot(seg_vec, seg_vec)
        if seg_len_sq == 0:
            candidate = p1
        else:
            t = np.clip(np.dot(point_vec, seg_vec) / seg_len_sq, 0.0, 1.0)
            candidate = p1 + t * seg_vec

        dist = np.linalg.norm(point - candidate)
        if dist < min_dist:
            min_dist = dist
            closest = candidate

    if closest is None and len(polyline) > 0:
        closest = polyline[0].astype(np.float64)

    return closest.astype(np.int32)


def position_along_polyline(
    point: tuple[float, float],
    polyline: np.ndarray,
) -> float:
    """Calculate normalized position (0-1) along a polyline for a given point.

    Parameters
    ----------
    point:
        (x, y) coordinate.
    polyline:
        Array of shape (N, 2) defining the polyline.

    Returns
    -------
    Position along the polyline from 0 (start) to 1 (end).
    """
    point = np.array(point, dtype=np.float64)
    polyline = np.array(polyline, dtype=np.float64)

    if len(polyline) < 2:
        return 0.0

    segment_lengths = []
    for i in range(len(polyline) - 1):
        seg_len = float(np.linalg.norm(polyline[i + 1] - polyline[i]))
        segment_lengths.append(seg_len)

    total_length = sum(segment_lengths)
    if total_length == 0:
        return 0.0

    min_dist = float("inf")
    best_segment_idx = 0
    best_t = 0.0

    for i in range(len(polyline) - 1):
        p1 = polyline[i]
        p2 = polyline[i + 1]

        seg_vec = p2 - p1
        point_vec = point - p1

        seg_len_sq = np.dot(seg_vec, seg_vec)
        if seg_len_sq == 0:
            t = 0.0
            candidate = p1
        else:
            t = float(np.clip(np.dot(point_vec, seg_vec) / seg_len_sq, 0.0, 1.0))
            candidate = p1 + t * seg_vec

        dist = float(np.linalg.norm(point - candidate))
        if dist < min_dist:
            min_dist = dist
            best_segment_idx = i
            best_t = t

    arc_length = sum(segment_lengths[:best_segment_idx])
    arc_length += best_t * segment_lengths[best_segment_idx]

    return arc_length / total_length


def get_zone_probabilities(
    x: float,
    y: float,
    zone_polygons: dict[str, np.ndarray],
    zone_types: dict[str, str],
    tube_max_distance: float = 40.0,
    soft_boundary: bool = True,
    normalize: bool = True,
) -> dict[str, float]:
    """Compute probability of belonging to each zone.

    Parameters
    ----------
    x, y:
        Coordinates in pixel space.
    zone_polygons:
        Dict mapping zone name to polygon/polyline array.
    zone_types:
        Dict mapping zone name to type ("room" or "tube").
    tube_max_distance:
        Maximum distance for tube classification.
    soft_boundary:
        Use soft boundaries with distance decay.
    normalize:
        Normalize probabilities to sum to 1.
    """
    point = (float(x), float(y))
    probs: dict[str, float] = {zone: 0.0 for zone in zone_polygons}

    for zone_name, polygon in zone_polygons.items():
        if zone_types.get(zone_name) == "room":
            if point_in_polygon(point, polygon):
                probs[zone_name] = 1.0
            elif soft_boundary:
                dist = signed_distance_to_polygon(point, polygon)
                # dist is negative when outside
                if dist > -tube_max_distance:
                    normalized_dist = max(0.0, 1.0 + dist / tube_max_distance)
                    probs[zone_name] = normalized_dist * normalized_dist

    for zone_name, polyline in zone_polygons.items():
        if zone_types.get(zone_name) == "tube":
            min_dist = min_distance_to_polyline(point, polyline)
            if min_dist <= tube_max_distance:
                if soft_boundary:
                    normalized_dist = min_dist / tube_max_distance
                    probs[zone_name] = max(0.0, 1.0 - normalized_dist**0.5)
                else:
                    probs[zone_name] = 1.0

    if normalize:
        total = sum(probs.values())
        if total > 0:
            for zone in probs:
                probs[zone] /= total

    return probs


def is_valid_transition(
    current_zone: str,
    new_zone: str,
    zone_graph: dict[str, list[str]],
    zone_types: dict[str, str],
) -> bool:
    """Check if a zone transition is valid according to the graph.

    Parameters
    ----------
    current_zone:
        Current zone name.
    new_zone:
        Proposed new zone name.
    zone_graph:
        Dict mapping zone name to list of connected zone names.
    zone_types:
        Dict mapping zone name to type ("room" or "tube").
    """
    if current_zone == new_zone:
        return True

    if not zone_graph:
        # Without a graph, forbid same-type transitions
        cur_type = zone_types.get(current_zone, "")
        new_type = zone_types.get(new_zone, "")
        if cur_type == new_type:
            return False
        return True

    if current_zone in zone_graph:
        return new_zone in zone_graph[current_zone]

    return True


def load_zone_config(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, list[str]]]:
    """Load zone configuration from combined YAML format.

    Expected format::

        zones:
          Room_1:
            type: room
            points: [[x,y], ...]
            connections: [Tube_1, Tube_2]
          Tube_1:
            type: tube
            points: [[x,y], ...]

    Parameters
    ----------
    path:
        Path to YAML file.

    Returns
    -------
    tuple of (zone_polygons, zone_types, zone_graph)
        - zone_polygons: dict mapping zone name to np.ndarray of points.
        - zone_types: dict mapping zone name to "room" or "tube".
        - zone_graph: bidirectional adjacency dict (zone -> list of connected zones).
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    zone_polygons: dict[str, np.ndarray] = {}
    zone_types: dict[str, str] = {}
    zone_graph: dict[str, list[str]] = {}

    zones_data = data.get("zones", {})
    for zone_name, zone_info in zones_data.items():
        points = zone_info.get("points", [])
        if points:
            zone_polygons[zone_name] = np.array(points, dtype=np.int32)

        zone_types[zone_name] = zone_info.get("type", "tube")

        connections = zone_info.get("connections", [])
        if connections:
            zone_graph[zone_name] = list(connections)

    # Make connections bidirectional
    for zone_name, connections in list(zone_graph.items()):
        for connected in connections:
            if connected not in zone_graph:
                zone_graph[connected] = []
            if zone_name not in zone_graph[connected]:
                zone_graph[connected].append(zone_name)

    logger.info("Loaded zone config: %d zones", len(zone_polygons))
    return zone_polygons, zone_types, zone_graph
