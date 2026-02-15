"""Tests for 1D analysis functions."""

import numpy as np
import pandas as pd
import pytest

from placecell.analysis_1d import (
    compute_occupancy_map_1d,
    compute_place_field_mask_1d,
    compute_rate_map_1d,
    compute_raw_rate_map_1d,
    compute_spatial_information_1d,
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
        data = np.array([1.0, 2.0, 3.0])
        result = gaussian_filter_normalized_1d(data, sigma=0.0)
        np.testing.assert_array_equal(result, data)
        assert result is not data


class TestComputeOccupancy1D:
    def test_total_time_matches(self):
        """Occupancy bins should sum to total time (without smoothing)."""
        rng = np.random.RandomState(123)
        n_frames = 1000
        df = pd.DataFrame({"pos_1d": rng.uniform(0, 4, n_frames)})
        fps = 20.0
        occ, valid, edges = compute_occupancy_map_1d(
            df, n_bins=40, pos_range=(0, 4), behavior_fps=fps, occupancy_sigma=0, min_occupancy=0
        )
        expected_time = n_frames / fps
        np.testing.assert_allclose(occ.sum(), expected_time, rtol=1e-10)

    def test_shape(self):
        df = pd.DataFrame({"pos_1d": np.linspace(0, 4, 100)})
        occ, valid, edges = compute_occupancy_map_1d(
            df, n_bins=20, pos_range=(0, 4), behavior_fps=20.0, occupancy_sigma=0, min_occupancy=0
        )
        assert occ.shape == (20,)
        assert valid.shape == (20,)
        assert edges.shape == (21,)

    def test_min_occupancy_filters(self):
        df = pd.DataFrame({"pos_1d": np.full(100, 1.5)})  # All in one bin
        occ, valid, edges = compute_occupancy_map_1d(
            df, n_bins=40, pos_range=(0, 4), behavior_fps=20.0, occupancy_sigma=0, min_occupancy=1.0
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
        rm = compute_rate_map_1d(events, occ, valid, edges, activity_sigma=0)
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
            "beh_frame_index": np.arange(n_frames),
            "pos_1d": rng.uniform(0, 4, n_frames),
        })
        occ, valid, edges = compute_occupancy_map_1d(
            traj, n_bins=40, pos_range=(0, 4), behavior_fps=20.0, occupancy_sigma=0, min_occupancy=0
        )
        # Events all at one location
        events_conc = pd.DataFrame({
            "pos_1d": np.full(50, 1.5),
            "s": np.ones(50),
            "beh_frame_index": rng.choice(n_frames, 50, replace=False),
        })
        si_conc, _, _ = compute_spatial_information_1d(
            events_conc, traj, occ, valid, edges, n_shuffles=10, random_seed=1
        )
        # Events spread uniformly
        events_spread = pd.DataFrame({
            "pos_1d": rng.uniform(0, 4, 50),
            "s": np.ones(50),
            "beh_frame_index": rng.choice(n_frames, 50, replace=False),
        })
        si_spread, _, _ = compute_spatial_information_1d(
            events_spread, traj, occ, valid, edges, n_shuffles=10, random_seed=1
        )
        assert si_conc > si_spread


class TestPlaceFieldMask1D:
    def test_detects_peak(self):
        """Should detect a clear peak above shuffled threshold."""
        n_bins = 40
        rate_map = np.zeros(n_bins)
        rate_map[15:20] = 0.8  # Clear peak
        shuffled_p95 = np.full(n_bins, 0.3)
        mask = compute_place_field_mask_1d(rate_map, shuffled_p95, threshold=0.1, min_bins=3)
        assert mask[15:20].all()

    def test_no_field_below_threshold(self):
        """No field when all bins below shuffled threshold."""
        n_bins = 20
        rate_map = np.full(n_bins, 0.1)
        shuffled_p95 = np.full(n_bins, 0.5)
        mask = compute_place_field_mask_1d(rate_map, shuffled_p95, threshold=0.05, min_bins=2)
        assert not mask.any()

    def test_min_bins_filter(self):
        """Seed regions smaller than min_bins should be excluded."""
        n_bins = 20
        rate_map = np.zeros(n_bins)
        rate_map[5] = 0.9  # Single bin peak
        shuffled_p95 = np.full(n_bins, 0.3)
        mask = compute_place_field_mask_1d(rate_map, shuffled_p95, threshold=0.1, min_bins=3)
        assert not mask.any()
