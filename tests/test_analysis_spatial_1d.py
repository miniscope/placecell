"""Tests for 1D analysis functions."""

import numpy as np
import pandas as pd
import pytest

from placecell.analysis.spatial_1d import (
    compute_occupancy_map_1d,
    compute_rate_map_1d,
    compute_raw_rate_map_1d,
    compute_spatial_information_1d,
    compute_stability_score_1d,
    compute_unit_analysis_1d,
    gaussian_filter_normalized_1d,
)


class TestGaussianFilter1D:
    def test_preserves_uniform(self):
        """Boundary normalization should preserve sum for uniform input."""
        data = np.ones(20)
        result = gaussian_filter_normalized_1d(data, sigma=2.0)
        np.testing.assert_allclose(result, 1.0, rtol=1e-5)

    def test_zero_sigma_returns_copy(self):
        """Sigma=0 should return a copy of the input."""
        data = np.array([1.0, 2.0, 3.0])
        result = gaussian_filter_normalized_1d(data, sigma=0.0)
        np.testing.assert_array_equal(result, data)
        assert result is not data # just to make sure it's a copy, not the same array. probably redundant.

    def test_segment_bins_independent_smoothing(self):
        """Segment boundaries should prevent smoothing across segments."""
        data = np.zeros(20)
        data[9] = 10.0  # Peak at boundary between seg 0 and seg 1
        # With segment_bins, the peak should not leak into the next segment
        result = gaussian_filter_normalized_1d(data, sigma=2.0, segment_bins=[0, 10, 20])
        assert result[10] == 0.0  # First bin of segment 2 should be zero

    def test_segment_bins_none_smooths_whole(self):
        """segment_bins=None should smooth the whole array."""
        data = np.zeros(20)
        data[9] = 10.0
        result = gaussian_filter_normalized_1d(data, sigma=2.0, segment_bins=None)
        # Without segment boundaries, the peak leaks across
        assert result[10] > 0.0

    def test_segment_bins_preserves_uniform_per_segment(self):
        """Uniform data should stay uniform within each segment."""
        data = np.ones(30)
        result = gaussian_filter_normalized_1d(data, sigma=2.0, segment_bins=[0, 10, 20, 30])
        np.testing.assert_allclose(result, 1.0, rtol=1e-5)


class TestComputeOccupancy1D:
    def test_total_time_matches(self):
        """Occupancy bins should sum to total time (without smoothing)."""
        rng = np.random.RandomState(123)
        n_frames = 1000
        df = pd.DataFrame({"pos_1d": rng.uniform(0, 4, n_frames)})
        fps = 20.0
        occ, valid, edges = compute_occupancy_map_1d(
            df, n_bins=40, pos_range=(0, 4), behavior_fps=fps, spatial_sigma=0, min_occupancy=0
        )
        expected_time = n_frames / fps
        np.testing.assert_allclose(occ.sum(), expected_time, rtol=1e-10)

    def test_shape(self):
        df = pd.DataFrame({"pos_1d": np.linspace(0, 4, 100)})
        occ, valid, edges = compute_occupancy_map_1d(
            df, n_bins=20, pos_range=(0, 4), behavior_fps=20.0, spatial_sigma=0, min_occupancy=0
        )
        assert occ.shape == (20,)
        assert valid.shape == (20,)
        assert edges.shape == (21,)

    def test_min_occupancy_filters(self):
        df = pd.DataFrame({"pos_1d": np.full(100, 1.5)})  # All in one bin
        occ, valid, edges = compute_occupancy_map_1d(
            df, n_bins=40, pos_range=(0, 4), behavior_fps=20.0, spatial_sigma=0, min_occupancy=1.0
        )
        # Only 1 bin should have data (5 seconds), rest should be invalid
        assert valid.sum() <= 5  # At most a few bins near the peak


class TestComputeRateMap1D:
    def test_shape_matches_occupancy(self):
        n_bins = 20
        occ = np.ones(n_bins)
        valid = np.ones(n_bins, dtype=bool)
        edges = np.linspace(0, 4, n_bins + 1)
        events = pd.DataFrame({"pos_1d": [1.0, 2.0, 3.0], "s": [1.0, 1.0, 1.0]})
        rm = compute_rate_map_1d(events, occ, valid, edges, spatial_sigma=0)
        assert rm.shape == (n_bins,)

    def test_empty_events_returns_nan(self):
        n_bins = 10
        occ = np.ones(n_bins)
        valid = np.ones(n_bins, dtype=bool)
        edges = np.linspace(0, 2, n_bins + 1)
        rm = compute_rate_map_1d(pd.DataFrame(), occ, valid, edges)
        assert np.all(np.isnan(rm))


class TestSpatialInformation1D:
    def test_concentrated_events_high_si(self):
        """Events concentrated at one location should have higher SI."""
        rng = np.random.RandomState(42)
        n_frames = 500
        traj = pd.DataFrame({
            "frame_index": np.arange(n_frames),
            "pos_1d": rng.uniform(0, 4, n_frames),
        })
        occ, valid, edges = compute_occupancy_map_1d(
            traj, n_bins=40, pos_range=(0, 4), behavior_fps=20.0, spatial_sigma=0, min_occupancy=0
        )
        # Events all at one location
        events_conc = pd.DataFrame({
            "pos_1d": np.full(50, 1.5),
            "s": np.ones(50),
            "frame_index": rng.choice(n_frames, 50, replace=False),
        })
        si_conc, _, _ = compute_spatial_information_1d(
            events_conc, traj, occ, valid, edges, n_shuffles=10, random_seed=1
        )
        # Events spread uniformly
        events_spread = pd.DataFrame({
            "pos_1d": rng.uniform(0, 4, 50),
            "s": np.ones(50),
            "frame_index": rng.choice(n_frames, 50, replace=False),
        })
        si_spread, _, _ = compute_spatial_information_1d(
            events_spread, traj, occ, valid, edges, n_shuffles=10, random_seed=1
        )
        assert si_conc > si_spread


class TestMinEventsGate1D:
    def _make_scfg(self, min_events: int):
        from placecell.config import SpatialMap1DConfig
        return SpatialMap1DConfig(
            bin_width_mm=1.0,
            min_occupancy=0.0,
            spatial_sigma=0.0,
            n_shuffles=10,
            random_seed=0,
            min_shift_seconds=0.0,
            si_weight_mode="amplitude",
            stability_splits=[2],
            min_events=min_events,
        )

    def test_gate_blocks_sparse_unit(self):
        """A unit with too few events should get p_val=1 for SI and stability."""
        rng = np.random.RandomState(0)
        n_frames = 400
        traj = pd.DataFrame({
            "frame_index": np.arange(n_frames),
            "pos_1d": rng.uniform(0, 4, n_frames),
        })
        occ, valid, edges = compute_occupancy_map_1d(
            traj, n_bins=40, pos_range=(0, 4), behavior_fps=20.0,
            spatial_sigma=0, min_occupancy=0,
        )
        # 3 events — well below typical gate
        events = pd.DataFrame({
            "unit_id": np.full(3, 1),
            "pos_1d": [1.0, 1.1, 1.2],
            "s": [1.0, 1.0, 1.0],
            "frame_index": [50, 51, 52],
        })

        scfg = self._make_scfg(min_events=20)
        result = compute_unit_analysis_1d(
            unit_id=1,
            df_filtered=events,
            trajectory_df=traj,
            occupancy_time=occ,
            valid_mask=valid,
            edges=edges,
            scfg=scfg,
            behavior_fps=20.0,
            random_seed=0,
        )
        assert result["p_val"] == 1.0
        assert result["si"] == 0.0
        assert all(s["p_val"] == 1.0 for s in result["stability_splits"])
        # Rate map is still produced for inspection.
        assert result["rate_map"].shape == occ.shape

    def test_gate_disabled_by_default(self):
        """min_events=0 should leave behavior unchanged."""
        rng = np.random.RandomState(0)
        n_frames = 400
        traj = pd.DataFrame({
            "frame_index": np.arange(n_frames),
            "pos_1d": rng.uniform(0, 4, n_frames),
        })
        occ, valid, edges = compute_occupancy_map_1d(
            traj, n_bins=40, pos_range=(0, 4), behavior_fps=20.0,
            spatial_sigma=0, min_occupancy=0,
        )
        events = pd.DataFrame({
            "unit_id": np.full(3, 1),
            "pos_1d": [1.0, 1.1, 1.2],
            "s": [1.0, 1.0, 1.0],
            "frame_index": [50, 51, 52],
        })
        scfg = self._make_scfg(min_events=0)
        result = compute_unit_analysis_1d(
            unit_id=1,
            df_filtered=events,
            trajectory_df=traj,
            occupancy_time=occ,
            valid_mask=valid,
            edges=edges,
            scfg=scfg,
            behavior_fps=20.0,
            random_seed=0,
        )
        # With the gate disabled the SI test runs normally; a 3-event
        # unit with no shuffles above observed SI can hit the p=1/(N+1)
        # floor, but the key invariant is that p_val is real, not the
        # hard 1.0 sentinel.
        assert result["p_val"] < 1.0 or result["si"] > 0.0


class TestStabilityShuffleSeeds1D:
    def test_different_n_split_blocks_use_different_shuffles(self):
        """Different stability splits on the same unit+seed must draw
        different shuffle sequences, so their null p-values are not
        pathologically correlated replicas of each other."""
        rng = np.random.RandomState(0)
        n_frames = 800
        traj = pd.DataFrame({
            "frame_index": np.arange(n_frames),
            "pos_1d": rng.uniform(0, 4, n_frames),
        })
        occ, valid, edges = compute_occupancy_map_1d(
            traj, n_bins=40, pos_range=(0, 4), behavior_fps=20.0,
            spatial_sigma=0, min_occupancy=0,
        )
        events = pd.DataFrame({
            "pos_1d": rng.uniform(0, 4, 120),
            "s": np.ones(120),
            "frame_index": rng.choice(n_frames, 120, replace=False),
        })

        _, _, _, _, _, shuf_2 = compute_stability_score_1d(
            events, traj, occ, valid, edges,
            spatial_sigma=0, n_split_blocks=2,
            n_shuffles=50, random_seed=42,
        )
        _, _, _, _, _, shuf_10 = compute_stability_score_1d(
            events, traj, occ, valid, edges,
            spatial_sigma=0, n_split_blocks=10,
            n_shuffles=50, random_seed=42,
        )
        assert shuf_2.size == 50
        assert shuf_10.size == 50
        # If the seed did not depend on n_split_blocks, the two shuffle
        # draws would use identical shift sequences and their null
        # correlation distributions would be highly correlated (often
        # identical up to the block structure). Require the two null
        # distributions to be meaningfully different.
        assert not np.allclose(np.sort(shuf_2), np.sort(shuf_10), atol=1e-6)
