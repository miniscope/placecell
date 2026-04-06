"""Tests for cached zone geometry helpers."""

import numpy as np

from placecell.geometry import (
    closest_point_on_polyline,
    closest_point_on_polyline_prepared,
    get_zone_probabilities,
    min_distance_to_polyline,
    min_distance_to_polyline_prepared,
    point_in_polygon,
    point_in_polygon_prepared,
    position_along_polyline,
    position_along_polyline_prepared,
    prepare_polygon_geometry,
    prepare_polyline_geometry,
    prepare_zone_geometry,
    signed_distance_to_polygon,
    signed_distance_to_polygon_prepared,
)


def test_prepared_polygon_helpers_match_uncached_versions() -> None:
    polygon = np.array([[0, 0], [5, 0], [5, 4], [0, 4]], dtype=np.int32)
    geometry = prepare_polygon_geometry(polygon)

    inside_point = (2.0, 1.5)
    outside_point = (7.0, 1.0)

    assert point_in_polygon_prepared(inside_point, geometry) == point_in_polygon(inside_point, polygon)
    assert point_in_polygon_prepared(outside_point, geometry) == point_in_polygon(
        outside_point, polygon
    )
    assert signed_distance_to_polygon_prepared(
        inside_point, geometry
    ) == signed_distance_to_polygon(inside_point, polygon)
    assert signed_distance_to_polygon_prepared(
        outside_point, geometry
    ) == signed_distance_to_polygon(outside_point, polygon)


def test_prepared_polyline_helpers_match_uncached_versions() -> None:
    polyline = np.array([[0, 0], [4, 0], [8, 4]], dtype=np.int32)
    geometry = prepare_polyline_geometry(polyline)
    point = (5.0, 2.0)

    assert min_distance_to_polyline_prepared(point, geometry) == min_distance_to_polyline(
        point, polyline
    )
    np.testing.assert_array_equal(
        closest_point_on_polyline_prepared(point, geometry),
        closest_point_on_polyline(point, polyline),
    )
    assert position_along_polyline_prepared(point, geometry) == position_along_polyline(
        point, polyline
    )


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
        assert cached[zone_name] == uncached[zone_name]
