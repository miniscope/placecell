"""Tests for arm-by-arm PVO utilities."""

from types import SimpleNamespace

import numpy as np
import pytest
from placecell.analysis.pvo_1d import compute_arm_pvo


def _res(rate_map: list[float]) -> SimpleNamespace:
    return SimpleNamespace(rate_map=np.asarray(rate_map, dtype=float))


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
