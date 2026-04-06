"""Tests for arm-by-arm PVO utilities."""

from types import SimpleNamespace

import numpy as np
from placecell.analysis.pvo_1d import arm_pvo_summary, compute_arm_pvo


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
        n_position_bins=5,
    )

    assert set(results) == {
        ("Arm_A", "Arm_A"),
        ("Arm_A", "Arm_B"),
        ("Arm_B", "Arm_A"),
        ("Arm_B", "Arm_B"),
    }
    assert results[("Arm_A", "Arm_B")].pvo.shape == (5, 5)
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
        n_position_bins=3,
    )

    diag_mean = results[("Arm_A", "Arm_B")].mean_diagonal
    assert diag_mean == 1.0


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
        n_position_bins=4,
    )

    pvo = results[("Arm_A", "Arm_B")].pvo
    finite = pvo[np.isfinite(pvo)]
    assert np.all(finite >= 0.0)
    assert np.all(finite <= 1.0)


def test_summary_table_contains_all_pairs() -> None:
    unit_results = {
        1: _res([1, 0, 0, 1]),
        2: _res([0, 1, 1, 0]),
    }
    results = compute_arm_pvo(
        unit_results,
        segment_bins=[0, 2, 4],
        segment_labels=["Arm_A", "Arm_B"],
        n_position_bins=4,
    )

    summary = arm_pvo_summary(results)
    assert len(summary) == 4
    assert set(summary.columns) == {
        "arm_a",
        "arm_b",
        "mean_diagonal",
        "n_units",
        "n_bins_per_arm",
    }
