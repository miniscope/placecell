"""Pure-numpy geometric utilities for zone classification."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from placecell.log import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class PathGeometry:
    """Precomputed per-path geometry for repeated point queries."""

    points: np.ndarray
    seg_starts: np.ndarray
    seg_vecs: np.ndarray
    seg_len_sq: np.ndarray
    seg_lengths: np.ndarray
    cumulative_lengths: np.ndarray
    total_length: float


@dataclass(frozen=True)
class PolygonGeometry(PathGeometry):
    """Precomputed polygon geometry including ray-casting edges."""

    edge_start_x: np.ndarray
    edge_start_y: np.ndarray
    edge_end_x: np.ndarray
    edge_end_y: np.ndarray


def prepare_polyline_geometry(polyline: np.ndarray) -> PathGeometry:
    """Precompute segment data for a polyline."""
    points = np.asarray(polyline, dtype=np.float64)
    if len(points) < 2:
        empty = np.empty((0, 2), dtype=np.float64)
        empty_1d = np.empty(0, dtype=np.float64)
        return PathGeometry(
            points=points,
            seg_starts=empty,
            seg_vecs=empty,
            seg_len_sq=empty_1d,
            seg_lengths=empty_1d,
            cumulative_lengths=np.array([0.0], dtype=np.float64),
            total_length=0.0,
        )

    seg_starts = points[:-1]
    seg_ends = points[1:]
    seg_vecs = seg_ends - seg_starts
    seg_len_sq = np.einsum("ij,ij->i", seg_vecs, seg_vecs)
    seg_lengths = np.sqrt(seg_len_sq)
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_length = float(cumulative_lengths[-1])
    return PathGeometry(
        points=points,
        seg_starts=seg_starts,
        seg_vecs=seg_vecs,
        seg_len_sq=seg_len_sq,
        seg_lengths=seg_lengths,
        cumulative_lengths=cumulative_lengths,
        total_length=total_length,
    )


def prepare_polygon_geometry(polygon: np.ndarray) -> PolygonGeometry:
    """Precompute segment and edge data for a polygon."""
    base = prepare_polyline_geometry(np.vstack([polygon, polygon[0]]))
    points = np.asarray(polygon, dtype=np.float64)
    edge_start = points
    edge_end = np.roll(points, -1, axis=0)
    return PolygonGeometry(
        points=base.points,
        seg_starts=base.seg_starts,
        seg_vecs=base.seg_vecs,
        seg_len_sq=base.seg_len_sq,
        seg_lengths=base.seg_lengths,
        cumulative_lengths=base.cumulative_lengths,
        total_length=base.total_length,
        edge_start_x=edge_start[:, 0],
        edge_start_y=edge_start[:, 1],
        edge_end_x=edge_end[:, 0],
        edge_end_y=edge_end[:, 1],
    )


def prepare_zone_geometry(
    zone_polygons: dict[str, np.ndarray],
    zone_types: dict[str, str],
) -> dict[str, PathGeometry | PolygonGeometry]:
    """Precompute repeated geometry for every configured zone."""
    prepared: dict[str, PathGeometry | PolygonGeometry] = {}
    for zone_name, points in zone_polygons.items():
        if zone_types.get(zone_name) == "room":
            prepared[zone_name] = prepare_polygon_geometry(points)
        else:
            prepared[zone_name] = prepare_polyline_geometry(points)
    return prepared


def _project_point_to_path_geometry(
    point: tuple[float, float],
    geometry: PathGeometry,
) -> tuple[np.ndarray, float, int, float]:
    """Project a point onto the closest segment of a prepared path."""
    if len(geometry.seg_starts) == 0:
        if len(geometry.points) == 0:
            origin = np.zeros(2, dtype=np.float64)
            return origin, float("inf"), 0, 0.0
        closest = geometry.points[0].astype(np.float64)
        dist = float(np.linalg.norm(np.asarray(point, dtype=np.float64) - closest))
        return closest, dist, 0, 0.0

    point_arr = np.asarray(point, dtype=np.float64)
    point_vecs = point_arr - geometry.seg_starts
    dot = np.einsum("ij,ij->i", point_vecs, geometry.seg_vecs)
    t = np.divide(
        dot,
        geometry.seg_len_sq,
        out=np.zeros_like(dot),
        where=geometry.seg_len_sq > 0,
    )
    t = np.clip(t, 0.0, 1.0)
    candidates = geometry.seg_starts + t[:, None] * geometry.seg_vecs
    diff = candidates - point_arr
    dist_sq = np.einsum("ij,ij->i", diff, diff)
    best_idx = int(np.argmin(dist_sq))
    closest = candidates[best_idx]
    return closest, float(np.sqrt(dist_sq[best_idx])), best_idx, float(t[best_idx])


def point_in_polygon_prepared(point: tuple[float, float], geometry: PolygonGeometry) -> bool:
    """Check whether a point lies inside a prepared polygon."""
    x, y = float(point[0]), float(point[1])
    edge_crosses_y = (geometry.edge_start_y > y) != (geometry.edge_end_y > y)
    x_intersections = np.empty_like(geometry.edge_start_x)
    np.divide(
        (geometry.edge_end_x - geometry.edge_start_x) * (y - geometry.edge_start_y),
        geometry.edge_end_y - geometry.edge_start_y,
        out=x_intersections,
        where=edge_crosses_y,
    )
    x_intersections += geometry.edge_start_x
    crosses = edge_crosses_y & (x < x_intersections)
    return bool(np.count_nonzero(crosses) % 2)


def signed_distance_to_polygon_prepared(
    point: tuple[float, float],
    geometry: PolygonGeometry,
) -> float:
    """Signed distance from a point to a prepared polygon boundary."""
    _, min_dist, _, _ = _project_point_to_path_geometry(point, geometry)
    if point_in_polygon_prepared(point, geometry):
        return min_dist
    return -min_dist


def min_distance_to_polyline_prepared(
    point: tuple[float, float],
    geometry: PathGeometry,
) -> float:
    """Minimum distance from a point to a prepared polyline."""
    _, min_dist, _, _ = _project_point_to_path_geometry(point, geometry)
    return min_dist


def closest_point_on_polyline_prepared(
    point: tuple[float, float],
    geometry: PathGeometry,
) -> np.ndarray:
    """Find the closest point on a prepared polyline."""
    closest, _, _, _ = _project_point_to_path_geometry(point, geometry)
    return closest.astype(np.int32)


def position_along_polyline_prepared(
    point: tuple[float, float],
    geometry: PathGeometry,
) -> float:
    """Calculate normalized position along a prepared polyline."""
    if geometry.total_length == 0:
        return 0.0
    _, _, best_segment_idx, best_t = _project_point_to_path_geometry(point, geometry)
    arc_length = geometry.cumulative_lengths[best_segment_idx]
    arc_length += best_t * geometry.seg_lengths[best_segment_idx]
    return float(arc_length / geometry.total_length)


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
    return point_in_polygon_prepared(point, prepare_polygon_geometry(polygon))


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
    return signed_distance_to_polygon_prepared(point, prepare_polygon_geometry(polygon))


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
    return min_distance_to_polyline_prepared(point, prepare_polyline_geometry(polyline))


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
    return closest_point_on_polyline_prepared(point, prepare_polyline_geometry(polyline))


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
    return position_along_polyline_prepared(point, prepare_polyline_geometry(polyline))


def get_zone_probabilities(
    x: float,
    y: float,
    zone_polygons: dict[str, np.ndarray],
    zone_types: dict[str, str],
    arm_max_distance: float = 40.0,
    soft_boundary: bool = True,
    room_decay_power: float = 2.0,
    arm_decay_power: float = 0.5,
    normalize: bool = True,
    prepared_geometry: dict[str, PathGeometry | PolygonGeometry] | None = None,
) -> dict[str, float]:
    """Compute probability of belonging to each zone.

    Parameters
    ----------
    x, y:
        Coordinates in pixel space.
    zone_polygons:
        Dict mapping zone name to polygon/polyline array.
    zone_types:
        Dict mapping zone name to type ("room" or "arm").
    arm_max_distance:
        Maximum distance for arm classification.
    soft_boundary:
        Use soft boundaries with distance decay.
    room_decay_power:
        Exponent for room boundary decay.  Higher values produce a
        steeper drop-off near room edges (from both inside and outside).
    arm_decay_power:
        Exponent for arm boundary decay.  Lower values produce a more
        gradual (fuzzy) drop-off away from the arm centerline.
    normalize:
        Normalize probabilities to sum to 1.
    """
    point = (float(x), float(y))
    probs: dict[str, float] = {zone: 0.0 for zone in zone_polygons}

    for zone_name, polygon in zone_polygons.items():
        geometry = prepared_geometry.get(zone_name) if prepared_geometry is not None else None
        if zone_types.get(zone_name) == "room":
            if soft_boundary:
                # signed dist: positive inside, negative outside
                if isinstance(geometry, PolygonGeometry):
                    dist = signed_distance_to_polygon_prepared(point, geometry)
                else:
                    dist = signed_distance_to_polygon(point, polygon)
                if dist >= arm_max_distance:
                    probs[zone_name] = 1.0
                elif dist > -arm_max_distance:
                    # Smooth decay across the boundary band
                    normalized = (dist + arm_max_distance) / (2 * arm_max_distance)
                    probs[zone_name] = normalized**room_decay_power
                # else: 0.0 (too far outside)
            else:
                probs[zone_name] = 1.0 if point_in_polygon(point, polygon) else 0.0

    for zone_name, polyline in zone_polygons.items():
        if zone_types.get(zone_name) == "arm":
            geometry = prepared_geometry.get(zone_name) if prepared_geometry is not None else None
            if isinstance(geometry, PathGeometry):
                min_dist = min_distance_to_polyline_prepared(point, geometry)
            else:
                min_dist = min_distance_to_polyline(point, polyline)
            if min_dist <= arm_max_distance:
                if soft_boundary:
                    normalized_dist = min_dist / arm_max_distance
                    probs[zone_name] = max(0.0, 1.0 - normalized_dist**arm_decay_power)
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
        Dict mapping zone name to type ("room" or "arm").
    """
    if current_zone == new_zone:
        return True

    if not zone_graph:
        # Without a graph, forbid same-type transitions
        cur_type = zone_types.get(current_zone, "")
        new_type = zone_types.get(new_zone, "")
        return cur_type != new_type

    if current_zone in zone_graph:
        return new_zone in zone_graph[current_zone]

    return True


def build_zone_graph(connections: dict[str, list[str]]) -> dict[str, list[str]]:
    """Build a bidirectional adjacency dict from a connections mapping.

    Parameters
    ----------
    connections:
        Dict mapping room names to lists of connected arm names.
        Example: ``{"Room_1": ["Arm_1", "Arm_2"]}``.

    Returns
    -------
    Bidirectional adjacency dict where every zone lists its neighbors.
    """
    graph: dict[str, list[str]] = {}
    for zone, neighbors in connections.items():
        graph.setdefault(zone, [])
        for neighbor in neighbors:
            if neighbor not in graph[zone]:
                graph[zone].append(neighbor)
            graph.setdefault(neighbor, [])
            if zone not in graph[neighbor]:
                graph[neighbor].append(zone)
    return graph


def load_zone_config(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, list[str]]]:
    """Load zone configuration from combined YAML format.

    Expected format::

        zones:
          Room_1:
            type: room
            points: [[x,y], ...]
            connections: [Arm_1, Arm_2]
          Arm_1:
            type: arm
            points: [[x,y], ...]

    Parameters
    ----------
    path:
        Path to YAML file.

    Returns
    -------
    tuple of (zone_polygons, zone_types, zone_graph)
        - zone_polygons: dict mapping zone name to np.ndarray of points.
        - zone_types: dict mapping zone name to "room" or "arm".
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

        zone_types[zone_name] = zone_info.get("type", "arm")

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
