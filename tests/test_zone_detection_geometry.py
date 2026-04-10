"""Tests for cached zone geometry helpers."""

import numpy as np

from placecell.geometry import (
    closest_point_on_polyline_prepared,
    get_zone_probabilities,
    min_distance_to_polyline_prepared,
    point_in_polygon_prepared,
    position_along_polyline_prepared,
    prepare_polygon_geometry,
    prepare_polyline_geometry,
    prepare_zone_geometry,
    signed_distance_to_polygon_prepared,
)


def test_prepared_polygon_helpers_on_known_rectangle() -> None:
    # Axis-aligned rectangle [0,0]-[5,4]; signed distance is positive inside.
    polygon = np.array([[0, 0], [5, 0], [5, 4], [0, 4]], dtype=np.int32)
    geometry = prepare_polygon_geometry(polygon)

    inside = (2.0, 1.5)
    outside = (7.0, 1.0)

    assert point_in_polygon_prepared(inside, geometry) is True
    assert point_in_polygon_prepared(outside, geometry) is False
    # Inside: closest edge is the bottom (y=0), distance 1.5.
    np.testing.assert_allclose(signed_distance_to_polygon_prepared(inside, geometry), 1.5)
    # Outside: closest edge is the right side (x=5) at y=1, distance 2.0.
    np.testing.assert_allclose(signed_distance_to_polygon_prepared(outside, geometry), -2.0)


def test_prepared_polyline_helpers_on_known_path() -> None:
    # Two-segment polyline: (0,0)->(4,0)->(8,4). Total length = 4 + sqrt(32).
    polyline = np.array([[0, 0], [4, 0], [8, 4]], dtype=np.int32)
    geometry = prepare_polyline_geometry(polyline)
    point = (5.0, 2.0)

    # Closest point on the second segment at parameter t=0.375 is (5.5, 1.5);
    # min distance = sqrt(0.5).
    np.testing.assert_allclose(min_distance_to_polyline_prepared(point, geometry), np.sqrt(0.5))
    np.testing.assert_allclose(closest_point_on_polyline_prepared(point, geometry), [5.5, 1.5])
    expected_arc = 4.0 + 0.375 * np.sqrt(32.0)
    expected_norm = expected_arc / (4.0 + np.sqrt(32.0))
    np.testing.assert_allclose(position_along_polyline_prepared(point, geometry), expected_norm)


def test_get_zone_probabilities_matches_with_prepared_geometry() -> None:
    zone_polygons = {
        "Room_1": np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32),
        "Arm_1": np.array([[10, 5], [20, 5]], dtype=np.int32),
    }
    zone_types = {"Room_1": "room", "Arm_1": "arm"}
    prepared = prepare_zone_geometry(zone_polygons, zone_types)

    uncached = get_zone_probabilities(
        9.0,
        5.0,
        zone_polygons,
        zone_types,
        arm_max_distance=6.0,
        room_decay_power=2.0,
        arm_decay_power=0.5,
        normalize=True,
    )
    cached = get_zone_probabilities(
        9.0,
        5.0,
        zone_polygons,
        zone_types,
        arm_max_distance=6.0,
        room_decay_power=2.0,
        arm_decay_power=0.5,
        normalize=True,
        prepared_geometry=prepared,
    )

    assert cached.keys() == uncached.keys()
    for zone_name in cached:
        np.testing.assert_allclose(cached[zone_name], uncached[zone_name])
