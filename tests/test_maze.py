"""Tests for maze behavior processing."""

import numpy as np
import pandas as pd
import pytest

from placecell.maze import compute_speed_1d, filter_tube_by_speed, serialize_tube_position


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
