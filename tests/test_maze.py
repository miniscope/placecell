"""Tests for maze behavior processing."""

import numpy as np
import pandas as pd
import pytest

from placecell.maze import (
    assign_traversal_direction,
    compute_speed_1d,
    compute_tube_lengths,
    filter_tube_by_speed,
    serialize_tube_position,
)


def _make_trajectory(n_frames: int = 100) -> pd.DataFrame:
    """Create a synthetic trajectory DataFrame with zone/tube_position columns."""
    rng = np.random.RandomState(42)
    zones = ["Tube_1"] * 30 + ["Room_1"] * 10 + ["Tube_2"] * 30 + ["Room_2"] * 10 + ["Tube_3"] * 20
    zones = zones[:n_frames]
    tube_pos = []
    for z in zones:
        if z.startswith("Tube"):
            tube_pos.append(rng.uniform(0, 1))
        else:
            tube_pos.append(np.nan)
    return pd.DataFrame(
        {
            "frame_index": np.arange(n_frames),
            "x": rng.uniform(0, 500, n_frames),
            "y": rng.uniform(0, 500, n_frames),
            "unix_time": np.linspace(1000, 1000 + n_frames / 20, n_frames),
            "speed": rng.uniform(0, 100, n_frames),
            "zone": zones,
            "tube_position": tube_pos,
        }
    )


class TestSerializeTubePosition:
    def test_filters_to_tubes_only(self):
        traj = _make_trajectory()
        result = serialize_tube_position(traj, tube_order=["Tube_1", "Tube_2", "Tube_3"])
        assert all(result["zone"].str.startswith("Tube"))
        assert result["pos_1d"].notna().all()

    def test_correct_offsets(self):
        traj = _make_trajectory()
        result = serialize_tube_position(traj, tube_order=["Tube_1", "Tube_2", "Tube_3"])
        tube1 = result[result["zone"] == "Tube_1"]["pos_1d"]
        tube2 = result[result["zone"] == "Tube_2"]["pos_1d"]
        tube3 = result[result["zone"] == "Tube_3"]["pos_1d"]
        assert (tube1 >= 0).all() and (tube1 < 1).all()
        assert (tube2 >= 1).all() and (tube2 < 2).all()
        assert (tube3 >= 2).all() and (tube3 < 3).all()

    def test_tube_index_assigned(self):
        traj = _make_trajectory()
        result = serialize_tube_position(traj, tube_order=["Tube_1", "Tube_2", "Tube_3"])
        assert (result[result["zone"] == "Tube_1"]["tube_index"] == 0).all()
        assert (result[result["zone"] == "Tube_2"]["tube_index"] == 1).all()
        assert (result[result["zone"] == "Tube_3"]["tube_index"] == 2).all()

    def test_room_frames_excluded(self):
        traj = _make_trajectory()
        n_rooms = traj["zone"].str.startswith("Room").sum()
        result = serialize_tube_position(traj, tube_order=["Tube_1", "Tube_2", "Tube_3"])
        assert len(result) == len(traj) - n_rooms


class TestComputeSpeed1D:
    def test_adds_speed_column(self):
        traj = _make_trajectory()
        traj_1d = serialize_tube_position(traj, tube_order=["Tube_1", "Tube_2", "Tube_3"])
        result = compute_speed_1d(traj_1d, window_frames=3)
        assert "speed_1d" in result.columns

    def test_cross_tube_speed_zero(self):
        """Speed across tube boundaries should be zero when tubes differ."""
        # Create trajectory with tube change at boundary
        df = pd.DataFrame(
            {
                "frame_index": [0, 1, 2, 3, 4],
                "pos_1d": [0.8, 0.9, 1.1, 1.2, 1.3],
                "tube_index": [0, 0, 1, 1, 1],
                "unix_time": [0.0, 0.05, 0.10, 0.15, 0.20],
            }
        )
        result = compute_speed_1d(df, window_frames=3)
        # Frame 0: window to frame 3, but tube_index differs (0 vs 1) → speed=0
        # Frame 1: window to frame 4, tube_index differs (0 vs 1) → speed=0
        assert result.iloc[0]["speed_1d"] == 0.0
        assert result.iloc[1]["speed_1d"] == 0.0


def _make_directional_trajectory() -> pd.DataFrame:
    """Create trajectory with known traversal directions.

    Layout: Tube_1 (fwd), Room_1, Tube_1 (rev), Room_1, Tube_2 (fwd), Room_2, Tube_2 (rev)
    Forward = starts near 0, Reverse = starts near 1.
    """
    rows = []
    t = 0.0
    fi = 0

    # Tube_1 forward: pos 0.1 -> 0.9
    for p in np.linspace(0.1, 0.9, 10):
        rows.append({"frame_index": fi, "zone": "Tube_1", "tube_position": p, "unix_time": t})
        fi += 1
        t += 0.05

    # Room_1 gap
    for _ in range(5):
        rows.append({"frame_index": fi, "zone": "Room_1", "tube_position": np.nan, "unix_time": t})
        fi += 1
        t += 0.05

    # Tube_1 reverse: pos 0.9 -> 0.1
    for p in np.linspace(0.9, 0.1, 10):
        rows.append({"frame_index": fi, "zone": "Tube_1", "tube_position": p, "unix_time": t})
        fi += 1
        t += 0.05

    # Room_1 gap
    for _ in range(5):
        rows.append({"frame_index": fi, "zone": "Room_1", "tube_position": np.nan, "unix_time": t})
        fi += 1
        t += 0.05

    # Tube_2 forward: pos 0.05 -> 0.95
    for p in np.linspace(0.05, 0.95, 10):
        rows.append({"frame_index": fi, "zone": "Tube_2", "tube_position": p, "unix_time": t})
        fi += 1
        t += 0.05

    # Room_2 gap
    for _ in range(5):
        rows.append({"frame_index": fi, "zone": "Room_2", "tube_position": np.nan, "unix_time": t})
        fi += 1
        t += 0.05

    # Tube_2 reverse: pos 0.8 -> 0.2
    for p in np.linspace(0.8, 0.2, 10):
        rows.append({"frame_index": fi, "zone": "Tube_2", "tube_position": p, "unix_time": t})
        fi += 1
        t += 0.05

    df = pd.DataFrame(rows)
    df["x"] = 0.0
    df["y"] = 0.0
    df["speed"] = 1.0
    return df


class TestAssignTraversalDirection:
    def test_creates_correct_effective_order(self):
        traj = _make_directional_trajectory()
        tube_order = ["Tube_1", "Tube_2"]
        traj_1d = serialize_tube_position(traj, tube_order=tube_order)
        result, eff_order = assign_traversal_direction(traj_1d, tube_order=tube_order)
        assert eff_order == ["Tube_1_fwd", "Tube_1_rev", "Tube_2_fwd", "Tube_2_rev"]

    def test_direction_labels(self):
        traj = _make_directional_trajectory()
        tube_order = ["Tube_1", "Tube_2"]
        traj_1d = serialize_tube_position(traj, tube_order=tube_order)
        result, _ = assign_traversal_direction(traj_1d, tube_order=tube_order)
        assert "direction" in result.columns
        assert set(result["direction"].unique()) == {"fwd", "rev"}

    def test_pos_1d_ranges(self):
        """Each directional segment should occupy [i, i+1] on the 1D axis."""
        traj = _make_directional_trajectory()
        tube_order = ["Tube_1", "Tube_2"]
        traj_1d = serialize_tube_position(traj, tube_order=tube_order)
        result, eff_order = assign_traversal_direction(traj_1d, tube_order=tube_order)

        for i, seg_name in enumerate(eff_order):
            seg_data = result[result["zone_dir"] == seg_name]
            if len(seg_data) > 0:
                assert seg_data["pos_1d"].min() >= i, f"{seg_name} pos_1d below {i}"
                assert seg_data["pos_1d"].max() < i + 1, f"{seg_name} pos_1d above {i+1}"

    def test_tube_index_matches_effective_order(self):
        traj = _make_directional_trajectory()
        tube_order = ["Tube_1", "Tube_2"]
        traj_1d = serialize_tube_position(traj, tube_order=tube_order)
        result, eff_order = assign_traversal_direction(traj_1d, tube_order=tube_order)

        for i, seg_name in enumerate(eff_order):
            seg_data = result[result["zone_dir"] == seg_name]
            if len(seg_data) > 0:
                assert (seg_data["tube_index"] == i).all()

    def test_traversal_count(self):
        """Should detect 4 traversals (Tube_1 fwd, Tube_1 rev, Tube_2 fwd, Tube_2 rev)."""
        traj = _make_directional_trajectory()
        tube_order = ["Tube_1", "Tube_2"]
        traj_1d = serialize_tube_position(traj, tube_order=tube_order)
        result, _ = assign_traversal_direction(traj_1d, tube_order=tube_order)
        assert result["traversal_id"].nunique() == 4

    def test_frame_count_preserved(self):
        """Direction splitting should not drop any frames."""
        traj = _make_directional_trajectory()
        tube_order = ["Tube_1", "Tube_2"]
        traj_1d = serialize_tube_position(traj, tube_order=tube_order)
        n_before = len(traj_1d)
        result, _ = assign_traversal_direction(traj_1d, tube_order=tube_order)
        assert len(result) == n_before


class TestFilterTubeBySpeed:
    def test_filters_by_threshold(self):
        traj = _make_trajectory()
        traj_1d = serialize_tube_position(traj, tube_order=["Tube_1", "Tube_2", "Tube_3"])
        traj_1d = compute_speed_1d(traj_1d, window_frames=3)
        result = filter_tube_by_speed(traj_1d, speed_threshold=0.5)
        assert all(result["speed_1d"] >= 0.5)

    def test_renames_frame_index(self):
        traj = _make_trajectory()
        traj_1d = serialize_tube_position(traj, tube_order=["Tube_1", "Tube_2", "Tube_3"])
        traj_1d = compute_speed_1d(traj_1d, window_frames=3)
        result = filter_tube_by_speed(traj_1d, speed_threshold=0.0)
        assert "beh_frame_index" in result.columns
        assert "frame_index" not in result.columns


class TestComputeTubeLengths:
    def test_simple_straight_tubes(self, tmp_path):
        """Straight-line polylines with known lengths."""
        graph = {
            "mm_per_pixel": 2.0,
            "zones": {
                "Tube_1": {"type": "tube", "points": [[0, 0], [100, 0]]},  # 100 px → 200 mm
                "Tube_2": {"type": "tube", "points": [[0, 0], [0, 50]]},  # 50 px → 100 mm
                "Room_1": {"type": "room", "points": [[0, 0], [10, 10]]},  # ~14.14 px → ~28.28 mm
            },
        }
        import yaml

        p = tmp_path / "graph.yaml"
        p.write_text(yaml.dump(graph))
        lengths, mm_pp = compute_tube_lengths(p)
        assert mm_pp == 2.0
        assert lengths["Tube_1"] == pytest.approx(200.0)
        assert lengths["Tube_2"] == pytest.approx(100.0)
        assert "Room_1" in lengths  # rooms are also returned

    def test_polyline_with_corners(self, tmp_path):
        """L-shaped polyline: 3→0→4 in pixel coords."""
        graph = {
            "mm_per_pixel": 1.0,
            "zones": {
                "Tube_1": {"type": "tube", "points": [[0, 0], [3, 0], [3, 4]]},  # 3 + 4 = 7 px
            },
        }
        import yaml

        p = tmp_path / "graph.yaml"
        p.write_text(yaml.dump(graph))
        lengths, _ = compute_tube_lengths(p)
        assert lengths["Tube_1"] == pytest.approx(7.0)

    def test_default_mm_per_pixel(self, tmp_path):
        """mm_per_pixel defaults to 1 if not in YAML."""
        graph = {
            "zones": {
                "Tube_1": {"type": "tube", "points": [[0, 0], [10, 0]]},
            },
        }
        import yaml

        p = tmp_path / "graph.yaml"
        p.write_text(yaml.dump(graph))
        lengths, mm_pp = compute_tube_lengths(p)
        assert mm_pp == 1.0
        assert lengths["Tube_1"] == pytest.approx(10.0)


class TestSerializeTubePositionPhysical:
    """Tests for serialize_tube_position with tube_lengths (physical scaling)."""

    def test_pos_1d_spans_physical_lengths(self):
        """Each tube's pos_1d should span its physical length."""
        tube_order = ["Tube_1", "Tube_2"]
        tube_lengths = {"Tube_1": 200.0, "Tube_2": 500.0}
        traj = pd.DataFrame(
            {
                "frame_index": np.arange(20),
                "zone": ["Tube_1"] * 10 + ["Tube_2"] * 10,
                "tube_position": np.tile(np.linspace(0, 0.99, 10), 2),
                "unix_time": np.linspace(0, 1, 20),
                "x": np.zeros(20),
                "y": np.zeros(20),
                "speed": np.ones(20),
            }
        )
        result = serialize_tube_position(traj, tube_order, tube_lengths=tube_lengths)
        t1 = result[result["zone"] == "Tube_1"]["pos_1d"]
        t2 = result[result["zone"] == "Tube_2"]["pos_1d"]
        # Tube_1: [0, 200), Tube_2: [200, 700)
        assert t1.min() >= 0.0
        assert t1.max() < 200.0
        assert t2.min() >= 200.0
        assert t2.max() < 700.0

    def test_total_range_equals_sum_of_lengths(self):
        tube_order = ["Tube_1", "Tube_2", "Tube_3"]
        tube_lengths = {"Tube_1": 100.0, "Tube_2": 300.0, "Tube_3": 200.0}
        traj = pd.DataFrame(
            {
                "frame_index": np.arange(30),
                "zone": ["Tube_1"] * 10 + ["Tube_2"] * 10 + ["Tube_3"] * 10,
                "tube_position": np.tile(np.linspace(0, 1, 10), 3),
                "unix_time": np.linspace(0, 1.5, 30),
                "x": np.zeros(30),
                "y": np.zeros(30),
                "speed": np.ones(30),
            }
        )
        result = serialize_tube_position(traj, tube_order, tube_lengths=tube_lengths)
        assert result["pos_1d"].max() == pytest.approx(600.0)  # 100+300+200
        assert result["pos_1d"].min() == pytest.approx(0.0)

    def test_backward_compat_no_lengths(self):
        """Without tube_lengths, each tube spans exactly 1 unit (original behavior)."""
        traj = _make_trajectory()
        result = serialize_tube_position(traj, tube_order=["Tube_1", "Tube_2", "Tube_3"])
        tube1 = result[result["zone"] == "Tube_1"]["pos_1d"]
        assert (tube1 >= 0).all() and (tube1 < 1).all()


class TestDirectionSplitPhysical:
    """Tests for assign_traversal_direction with physical tube_lengths."""

    def test_directional_segments_span_physical_lengths(self):
        """Each fwd/rev segment should span its parent tube's physical length."""
        traj = _make_directional_trajectory()
        tube_order = ["Tube_1", "Tube_2"]
        tube_lengths = {"Tube_1": 200.0, "Tube_2": 400.0}
        traj_1d = serialize_tube_position(traj, tube_order, tube_lengths=tube_lengths)
        result, eff_order = assign_traversal_direction(
            traj_1d, tube_order, tube_lengths=tube_lengths
        )
        assert eff_order == ["Tube_1_fwd", "Tube_1_rev", "Tube_2_fwd", "Tube_2_rev"]

        # Expected offsets: fwd1=[0,200), rev1=[200,400), fwd2=[400,800), rev2=[800,1200)
        seg_data = {name: result[result["zone_dir"] == name]["pos_1d"] for name in eff_order}
        assert seg_data["Tube_1_fwd"].min() >= 0.0
        assert seg_data["Tube_1_fwd"].max() < 200.0
        assert seg_data["Tube_1_rev"].min() >= 200.0
        assert seg_data["Tube_1_rev"].max() < 400.0
        assert seg_data["Tube_2_fwd"].min() >= 400.0
        assert seg_data["Tube_2_fwd"].max() < 800.0
        assert seg_data["Tube_2_rev"].min() >= 800.0
        assert seg_data["Tube_2_rev"].max() < 1200.0

    def test_total_range_physical(self):
        """Total pos_1d range = 2 * (length_1 + length_2) with direction split."""
        traj = _make_directional_trajectory()
        tube_order = ["Tube_1", "Tube_2"]
        tube_lengths = {"Tube_1": 150.0, "Tube_2": 350.0}
        traj_1d = serialize_tube_position(traj, tube_order, tube_lengths=tube_lengths)
        result, _ = assign_traversal_direction(traj_1d, tube_order, tube_lengths=tube_lengths)
        # 4 segments: 150 + 150 + 350 + 350 = 1000
        assert result["pos_1d"].max() < 1000.0
        assert result["pos_1d"].min() >= 0.0

    def test_backward_compat_no_lengths(self):
        """Without tube_lengths, each directional segment spans 1 unit."""
        traj = _make_directional_trajectory()
        tube_order = ["Tube_1", "Tube_2"]
        traj_1d = serialize_tube_position(traj, tube_order)
        result, eff_order = assign_traversal_direction(traj_1d, tube_order)
        for i, seg_name in enumerate(eff_order):
            seg = result[result["zone_dir"] == seg_name]
            if len(seg) > 0:
                assert seg["pos_1d"].min() >= i
                assert seg["pos_1d"].max() < i + 1
