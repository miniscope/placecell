"""Tests for arm-by-arm PVO utilities."""

from types import SimpleNamespace

import numpy as np
import pytest
from placecell.analysis.pvo_1d import compute_arm_pvo


def _res(rate_map: list[float]) -> SimpleNamespace:
    """Build a unit-result stub. ``rate_map`` is stored as
    ``rate_map_smoothed`` since that's the only field PVO consumes."""
    return SimpleNamespace(rate_map_smoothed=np.asarray(rate_map, dtype=float))


def _res_smoothed(rate_map_smoothed: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        rate_map_smoothed=np.asarray(rate_map_smoothed, dtype=float)
    )


def test_compute_arm_pvo_shape_and_labels() -> None:
    unit_results = {
        1: _res([1, 2, 1, 0, 0, 1]),
        2: _res([0, 1, 2, 1, 1, 0]),
        3: _res([2, 1, 0, 1, 2, 1]),
    }

    results = compute_arm_pvo(
        unit_results,
        segment_bins=[0, 3, 6],
        segment_labels=["Arm_A", "Arm_B"],
    )

    assert set(results) == {
        ("Arm_A", "Arm_A"),
        ("Arm_A", "Arm_B"),
        ("Arm_B", "Arm_A"),
        ("Arm_B", "Arm_B"),
    }
    assert results[("Arm_A", "Arm_B")].pvo.shape == (3, 3)
    assert results[("Arm_A", "Arm_B")].n_units == 3


def test_same_pattern_across_arms_has_high_diagonal() -> None:
    unit_results = {
        1: _res([1, 2, 3, 1, 2, 3]),
        2: _res([3, 2, 1, 3, 2, 1]),
        3: _res([2, 3, 1, 2, 3, 1]),
    }

    results = compute_arm_pvo(
        unit_results,
        segment_bins=[0, 3, 6],
        segment_labels=["Arm_A", "Arm_B"],
    )

    pvo = results[("Arm_A", "Arm_B")].pvo
    diag_mean = float(np.nanmean(np.diag(pvo)))
    assert diag_mean == pytest.approx(1.0)


def test_pvo_is_firing_rate_weighted_not_peak_normalized() -> None:
    """PVO must reflect true rate magnitudes, not peak-normalized shapes.

    When two cells have very different peak rates (10 Hz vs 0.1 Hz) and
    are co-active at the same bins, PVO should be dominated by the
    high-rate cell. Pre-normalized inputs would give a different, less
    meaningful answer.
    """
    # Full concatenated maps. segment_bins=[0, 2, 4] → Arm_A=bins[0:2],
    # Arm_B=bins[2:4]. Both arms have the same tuning shape.
    # unit 1 peaks at 10 Hz; unit 2 peaks at 0.1 Hz.
    smoothed_results = {
        1: _res_smoothed([10.0, 5.0, 10.0, 5.0]),
        2: _res_smoothed([0.1, 0.1, 0.1, 0.1]),
    }
    # What you'd get if each cell were independently peak-normalized
    # before stacking (the behavior we explicitly rejected).
    normalized_results = {
        1: _res_smoothed([1.0, 0.5, 1.0, 0.5]),
        2: _res_smoothed([1.0, 1.0, 1.0, 1.0]),
    }

    smoothed = compute_arm_pvo(
        smoothed_results,
        segment_bins=[0, 2, 4],
        segment_labels=["Arm_A", "Arm_B"],
    )[("Arm_A", "Arm_B")].pvo
    normalized = compute_arm_pvo(
        normalized_results,
        segment_bins=[0, 2, 4],
        segment_labels=["Arm_A", "Arm_B"],
    )[("Arm_A", "Arm_B")].pvo

    # Rate-weighted off-diagonal is dominated by the 10 Hz cell, so
    # cross-bin cosine stays close to 1. Peak-normalized treats both
    # cells equally, pulling the cosine noticeably below 1.
    assert smoothed[0, 1] > 0.999
    assert normalized[0, 1] < 0.96
    assert not np.allclose(smoothed, normalized, atol=1e-3), (
        "PVO must differ between rate-weighted and peak-normalized inputs."
    )


def test_overlap_is_bounded_between_zero_and_one() -> None:
    unit_results = {
        1: _res([1, 0, 0, 1]),
        2: _res([0, 1, 1, 0]),
        3: _res([1, 1, 1, 1]),
    }

    results = compute_arm_pvo(
        unit_results,
        segment_bins=[0, 2, 4],
        segment_labels=["Arm_A", "Arm_B"],
    )

    pvo = results[("Arm_A", "Arm_B")].pvo
    finite = pvo[np.isfinite(pvo)]
    assert np.all(finite >= 0.0)
    assert np.all(finite <= 1.0)
