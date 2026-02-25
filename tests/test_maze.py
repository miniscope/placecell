"""Tests for maze behavior processing."""

import numpy as np
import pandas as pd
import pytest

from placecell.maze_helper import (
    assign_traversal_direction,
    compute_speed_1d,
    compute_arm_lengths,
    filter_arm_by_speed,
    remove_position_jumps_1d,
    serialize_arm_position,
)


def _make_trajectory(n_frames: int = 100) -> pd.DataFrame:
    """Create a synthetic trajectory DataFrame with zone/arm_position columns."""
    rng = np.random.RandomState(42)
    zones = ["Arm_1"] * 30 + ["Room_1"] * 10 + ["Arm_2"] * 30 + ["Room_2"] * 10 + ["Arm_3"] * 20
    zones = zones[:n_frames]
    arm_pos = []
    for z in zones:
        if z.startswith("Arm"):
            arm_pos.append(rng.uniform(0, 1))
        else:
            arm_pos.append(np.nan)
    return pd.DataFrame(
        {
            "frame_index": np.arange(n_frames),
            "x": rng.uniform(0, 500, n_frames),
            "y": rng.uniform(0, 500, n_frames),
            "unix_time": np.linspace(1000, 1000 + n_frames / 20, n_frames),
            "speed": rng.uniform(0, 100, n_frames),
            "zone": zones,
            "arm_position": arm_pos,
        }
    )


class TestSerializeArmPosition:
    def test_filters_to_arms_only(self):
        traj = _make_trajectory()
        result = serialize_arm_position(traj, arm_order=["Arm_1", "Arm_2", "Arm_3"])
        assert all(result["zone"].str.startswith("Arm"))
        assert result["pos_1d"].notna().all()

    def test_correct_offsets(self):
        traj = _make_trajectory()
        result = serialize_arm_position(traj, arm_order=["Arm_1", "Arm_2", "Arm_3"])
        arm1 = result[result["zone"] == "Arm_1"]["pos_1d"]
        arm2 = result[result["zone"] == "Arm_2"]["pos_1d"]
        arm3 = result[result["zone"] == "Arm_3"]["pos_1d"]
        assert (arm1 >= 0).all() and (arm1 < 1).all()
        assert (arm2 >= 1).all() and (arm2 < 2).all()
        assert (arm3 >= 2).all() and (arm3 < 3).all()

    def test_arm_index_assigned(self):
        traj = _make_trajectory()
        result = serialize_arm_position(traj, arm_order=["Arm_1", "Arm_2", "Arm_3"])
        assert (result[result["zone"] == "Arm_1"]["arm_index"] == 0).all()
        assert (result[result["zone"] == "Arm_2"]["arm_index"] == 1).all()
        assert (result[result["zone"] == "Arm_3"]["arm_index"] == 2).all()

    def test_room_frames_excluded(self):
        traj = _make_trajectory()
        n_rooms = traj["zone"].str.startswith("Room").sum()
        result = serialize_arm_position(traj, arm_order=["Arm_1", "Arm_2", "Arm_3"])
        assert len(result) == len(traj) - n_rooms


class TestComputeSpeed1D:
    def test_adds_speed_column(self):
        traj = _make_trajectory()
        traj_1d = serialize_arm_position(traj, arm_order=["Arm_1", "Arm_2", "Arm_3"])
        result = compute_speed_1d(traj_1d, window_frames=3)
        assert "speed_1d" in result.columns

    def test_cross_arm_speed_zero(self):
        """Speed across arm boundaries should be zero when arms differ."""
        # Create trajectory with arm change at boundary
        df = pd.DataFrame(
            {
                "frame_index": [0, 1, 2, 3, 4],
                "pos_1d": [0.8, 0.9, 1.1, 1.2, 1.3],
                "arm_index": [0, 0, 1, 1, 1],
                "unix_time": [0.0, 0.05, 0.10, 0.15, 0.20],
            }
        )
        result = compute_speed_1d(df, window_frames=3)
        # Frame 0: window to frame 3, but arm_index differs (0 vs 1) → speed=0
        # Frame 1: window to frame 4, arm_index differs (0 vs 1) → speed=0
        assert result.iloc[0]["speed_1d"] == 0.0
        assert result.iloc[1]["speed_1d"] == 0.0


def _make_directional_trajectory() -> pd.DataFrame:
    """Create trajectory with known traversal directions.

    Layout: Arm_1 (fwd), Room_1, Arm_1 (rev), Room_1, Arm_2 (fwd), Room_2, Arm_2 (rev)
    Forward = starts near 0, Reverse = starts near 1.
    """
    rows = []
    t = 0.0
    fi = 0

    # Arm_1 forward: pos 0.1 -> 0.9
    for p in np.linspace(0.1, 0.9, 10):
        rows.append({"frame_index": fi, "zone": "Arm_1", "arm_position": p, "unix_time": t})
        fi += 1
        t += 0.05

    # Room_1 gap
    for _ in range(5):
        rows.append({"frame_index": fi, "zone": "Room_1", "arm_position": np.nan, "unix_time": t})
        fi += 1
        t += 0.05

    # Arm_1 reverse: pos 0.9 -> 0.1
    for p in np.linspace(0.9, 0.1, 10):
        rows.append({"frame_index": fi, "zone": "Arm_1", "arm_position": p, "unix_time": t})
        fi += 1
        t += 0.05

    # Room_1 gap
    for _ in range(5):
        rows.append({"frame_index": fi, "zone": "Room_1", "arm_position": np.nan, "unix_time": t})
        fi += 1
        t += 0.05

    # Arm_2 forward: pos 0.05 -> 0.95
    for p in np.linspace(0.05, 0.95, 10):
        rows.append({"frame_index": fi, "zone": "Arm_2", "arm_position": p, "unix_time": t})
        fi += 1
        t += 0.05

    # Room_2 gap
    for _ in range(5):
        rows.append({"frame_index": fi, "zone": "Room_2", "arm_position": np.nan, "unix_time": t})
        fi += 1
        t += 0.05

    # Arm_2 reverse: pos 0.8 -> 0.2
    for p in np.linspace(0.8, 0.2, 10):
        rows.append({"frame_index": fi, "zone": "Arm_2", "arm_position": p, "unix_time": t})
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
        arm_order = ["Arm_1", "Arm_2"]
        traj_1d = serialize_arm_position(traj, arm_order=arm_order)
        result, eff_order = assign_traversal_direction(traj_1d, arm_order=arm_order)
        assert eff_order == ["Arm_1_fwd", "Arm_1_rev", "Arm_2_fwd", "Arm_2_rev"]

    def test_direction_labels(self):
        traj = _make_directional_trajectory()
        arm_order = ["Arm_1", "Arm_2"]
        traj_1d = serialize_arm_position(traj, arm_order=arm_order)
        result, _ = assign_traversal_direction(traj_1d, arm_order=arm_order)
        assert "direction" in result.columns
        assert set(result["direction"].unique()) == {"fwd", "rev"}

    def test_pos_1d_ranges(self):
        """Each directional segment should occupy [i, i+1] on the 1D axis."""
        traj = _make_directional_trajectory()
        arm_order = ["Arm_1", "Arm_2"]
        traj_1d = serialize_arm_position(traj, arm_order=arm_order)
        result, eff_order = assign_traversal_direction(traj_1d, arm_order=arm_order)

        for i, seg_name in enumerate(eff_order):
            seg_data = result[result["zone_dir"] == seg_name]
            if len(seg_data) > 0:
                assert seg_data["pos_1d"].min() >= i, f"{seg_name} pos_1d below {i}"
                assert seg_data["pos_1d"].max() < i + 1, f"{seg_name} pos_1d above {i+1}"

    def test_arm_index_matches_effective_order(self):
        traj = _make_directional_trajectory()
        arm_order = ["Arm_1", "Arm_2"]
        traj_1d = serialize_arm_position(traj, arm_order=arm_order)
        result, eff_order = assign_traversal_direction(traj_1d, arm_order=arm_order)

        for i, seg_name in enumerate(eff_order):
            seg_data = result[result["zone_dir"] == seg_name]
            if len(seg_data) > 0:
                assert (seg_data["arm_index"] == i).all()

    def test_traversal_count(self):
        """Should detect 4 traversals (Arm_1 fwd, Arm_1 rev, Arm_2 fwd, Arm_2 rev)."""
        traj = _make_directional_trajectory()
        arm_order = ["Arm_1", "Arm_2"]
        traj_1d = serialize_arm_position(traj, arm_order=arm_order)
        result, _ = assign_traversal_direction(traj_1d, arm_order=arm_order)
        assert result["traversal_id"].nunique() == 4

    def test_frame_count_preserved(self):
        """Direction splitting should not drop any frames."""
        traj = _make_directional_trajectory()
        arm_order = ["Arm_1", "Arm_2"]
        traj_1d = serialize_arm_position(traj, arm_order=arm_order)
        n_before = len(traj_1d)
        result, _ = assign_traversal_direction(traj_1d, arm_order=arm_order)
        assert len(result) == n_before


class TestFilterArmBySpeed:
    def test_filters_by_threshold(self):
        traj = _make_trajectory()
        traj_1d = serialize_arm_position(traj, arm_order=["Arm_1", "Arm_2", "Arm_3"])
        traj_1d = compute_speed_1d(traj_1d, window_frames=3)
        result = filter_arm_by_speed(traj_1d, speed_threshold=0.5)
        assert all(result["speed_1d"] >= 0.5)

    def test_renames_frame_index(self):
        traj = _make_trajectory()
        traj_1d = serialize_arm_position(traj, arm_order=["Arm_1", "Arm_2", "Arm_3"])
        traj_1d = compute_speed_1d(traj_1d, window_frames=3)
        result = filter_arm_by_speed(traj_1d, speed_threshold=0.0)
        assert "beh_frame_index" in result.columns
        assert "frame_index" not in result.columns


class TestRemovePositionJumps1D:
    def test_interpolates_within_arm(self):
        """Jumps within the same arm should be detected and interpolated."""
        df = pd.DataFrame({
            "frame_index": [0, 1, 2, 3, 4],
            "pos_1d": [10.0, 20.0, 500.0, 30.0, 40.0],  # frame 2 is a jump
            "arm_index": [0, 0, 0, 0, 0],
        })
        result, n_fixed = remove_position_jumps_1d(df, threshold_mm=100.0)
        assert n_fixed == 2  # frames 2 and 3 both arrive via jumps
        # Both NaN'd, interpolated linearly between 20.0 (frame 1) and 40.0 (frame 4)
        assert result["pos_1d"].iloc[2] == pytest.approx(26.67, abs=0.1)
        assert result["pos_1d"].iloc[3] == pytest.approx(33.33, abs=0.1)

    def test_ignores_cross_arm_jumps(self):
        """Jumps across arm boundaries should NOT be flagged."""
        df = pd.DataFrame({
            "frame_index": [0, 1, 2, 3],
            "pos_1d": [10.0, 12.0, 500.0, 502.0],  # big cross-arm jump at frame 2
            "arm_index": [0, 0, 1, 1],  # arm change at frame 2
        })
        result, n_fixed = remove_position_jumps_1d(df, threshold_mm=100.0)
        assert n_fixed == 0

    def test_no_jumps_returns_copy(self):
        """When no jumps exist, return unchanged copy."""
        df = pd.DataFrame({
            "frame_index": [0, 1, 2],
            "pos_1d": [10.0, 15.0, 20.0],
            "arm_index": [0, 0, 0],
        })
        result, n_fixed = remove_position_jumps_1d(df, threshold_mm=100.0)
        assert n_fixed == 0
        np.testing.assert_array_equal(result["pos_1d"].values, [10.0, 15.0, 20.0])


class TestComputeArmLengths:
    def test_simple_straight_arms(self):
        """Straight-line polylines with known lengths."""
        polylines = {
            "Arm_1": [[0, 0], [100, 0]],  # 100 px → 200 mm
            "Arm_2": [[0, 0], [0, 50]],  # 50 px → 100 mm
            "Room_1": [[0, 0], [10, 10]],  # ~14.14 px → ~28.28 mm
        }
        lengths = compute_arm_lengths(polylines, mm_per_pixel=2.0)
        assert lengths["Arm_1"] == pytest.approx(200.0)
        assert lengths["Arm_2"] == pytest.approx(100.0)
        assert "Room_1" in lengths  # rooms are also returned

    def test_polyline_with_corners(self):
        """L-shaped polyline: 3→0→4 in pixel coords."""
        polylines = {
            "Arm_1": [[0, 0], [3, 0], [3, 4]],  # 3 + 4 = 7 px
        }
        lengths = compute_arm_lengths(polylines, mm_per_pixel=1.0)
        assert lengths["Arm_1"] == pytest.approx(7.0)


class TestSerializeArmPositionPhysical:
    """Tests for serialize_arm_position with arm_lengths (physical scaling)."""

    def test_pos_1d_spans_physical_lengths(self):
        """Each arm's pos_1d should span its physical length."""
        arm_order = ["Arm_1", "Arm_2"]
        arm_lengths = {"Arm_1": 200.0, "Arm_2": 500.0}
        traj = pd.DataFrame(
            {
                "frame_index": np.arange(20),
                "zone": ["Arm_1"] * 10 + ["Arm_2"] * 10,
                "arm_position": np.tile(np.linspace(0, 0.99, 10), 2),
                "unix_time": np.linspace(0, 1, 20),
                "x": np.zeros(20),
                "y": np.zeros(20),
                "speed": np.ones(20),
            }
        )
        result = serialize_arm_position(traj, arm_order, arm_lengths=arm_lengths)
        t1 = result[result["zone"] == "Arm_1"]["pos_1d"]
        t2 = result[result["zone"] == "Arm_2"]["pos_1d"]
        # Arm_1: [0, 200), Arm_2: [200, 700)
        assert t1.min() >= 0.0
        assert t1.max() < 200.0
        assert t2.min() >= 200.0
        assert t2.max() < 700.0

    def test_total_range_equals_sum_of_lengths(self):
        arm_order = ["Arm_1", "Arm_2", "Arm_3"]
        arm_lengths = {"Arm_1": 100.0, "Arm_2": 300.0, "Arm_3": 200.0}
        traj = pd.DataFrame(
            {
                "frame_index": np.arange(30),
                "zone": ["Arm_1"] * 10 + ["Arm_2"] * 10 + ["Arm_3"] * 10,
                "arm_position": np.tile(np.linspace(0, 1, 10), 3),
                "unix_time": np.linspace(0, 1.5, 30),
                "x": np.zeros(30),
                "y": np.zeros(30),
                "speed": np.ones(30),
            }
        )
        result = serialize_arm_position(traj, arm_order, arm_lengths=arm_lengths)
        assert result["pos_1d"].max() == pytest.approx(600.0)  # 100+300+200
        assert result["pos_1d"].min() == pytest.approx(0.0)

    def test_backward_compat_no_lengths(self):
        """Without arm_lengths, each arm spans exactly 1 unit (original behavior)."""
        traj = _make_trajectory()
        result = serialize_arm_position(traj, arm_order=["Arm_1", "Arm_2", "Arm_3"])
        arm1 = result[result["zone"] == "Arm_1"]["pos_1d"]
        assert (arm1 >= 0).all() and (arm1 < 1).all()


class TestDirectionSplitPhysical:
    """Tests for assign_traversal_direction with physical arm_lengths."""

    def test_directional_segments_span_physical_lengths(self):
        """Each fwd/rev segment should span its parent arm's physical length."""
        traj = _make_directional_trajectory()
        arm_order = ["Arm_1", "Arm_2"]
        arm_lengths = {"Arm_1": 200.0, "Arm_2": 400.0}
        traj_1d = serialize_arm_position(traj, arm_order, arm_lengths=arm_lengths)
        result, eff_order = assign_traversal_direction(
            traj_1d, arm_order, arm_lengths=arm_lengths
        )
        assert eff_order == ["Arm_1_fwd", "Arm_1_rev", "Arm_2_fwd", "Arm_2_rev"]

        # Expected offsets: fwd1=[0,200), rev1=[200,400), fwd2=[400,800), rev2=[800,1200)
        seg_data = {name: result[result["zone_dir"] == name]["pos_1d"] for name in eff_order}
        assert seg_data["Arm_1_fwd"].min() >= 0.0
        assert seg_data["Arm_1_fwd"].max() < 200.0
        assert seg_data["Arm_1_rev"].min() >= 200.0
        assert seg_data["Arm_1_rev"].max() < 400.0
        assert seg_data["Arm_2_fwd"].min() >= 400.0
        assert seg_data["Arm_2_fwd"].max() < 800.0
        assert seg_data["Arm_2_rev"].min() >= 800.0
        assert seg_data["Arm_2_rev"].max() < 1200.0

    def test_total_range_physical(self):
        """Total pos_1d range = 2 * (length_1 + length_2) with direction split."""
        traj = _make_directional_trajectory()
        arm_order = ["Arm_1", "Arm_2"]
        arm_lengths = {"Arm_1": 150.0, "Arm_2": 350.0}
        traj_1d = serialize_arm_position(traj, arm_order, arm_lengths=arm_lengths)
        result, _ = assign_traversal_direction(traj_1d, arm_order, arm_lengths=arm_lengths)
        # 4 segments: 150 + 150 + 350 + 350 = 1000
        assert result["pos_1d"].max() < 1000.0
        assert result["pos_1d"].min() >= 0.0

    def test_backward_compat_no_lengths(self):
        """Without arm_lengths, each directional segment spans 1 unit."""
        traj = _make_directional_trajectory()
        arm_order = ["Arm_1", "Arm_2"]
        traj_1d = serialize_arm_position(traj, arm_order)
        result, eff_order = assign_traversal_direction(traj_1d, arm_order)
        for i, seg_name in enumerate(eff_order):
            seg = result[result["zone_dir"] == seg_name]
            if len(seg) > 0:
                assert seg["pos_1d"].min() >= i
                assert seg["pos_1d"].max() < i + 1
