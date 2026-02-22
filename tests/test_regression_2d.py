"""Full-pipeline regression test for the 2D arena analysis.

Runs the complete ArenaDataset pipeline on a small data subset and
compares every output against a saved reference bundle.  The reference
was generated once from a real dataset (requires ProcData drive).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from placecell.dataset.arena import ArenaDataset
from placecell.dataset.base import BasePlaceCellDataset


REGRESSION_DIR = Path(__file__).parent / "assets" / "regression_2d"


@pytest.fixture(scope="module")
def pipeline_result() -> ArenaDataset:
    """Run the full pipeline once for all tests in this module."""
    ds = BasePlaceCellDataset.from_yaml(
        REGRESSION_DIR / "analysis_config.yaml",
        REGRESSION_DIR / "data_paths.yaml",
    )
    ds.load()
    ds.preprocess_behavior()
    ds.deconvolve()
    ds.match_events()
    ds.compute_occupancy()
    ds.analyze_units()
    return ds


@pytest.fixture(scope="module")
def reference() -> ArenaDataset:
    """Load the reference bundle."""
    return BasePlaceCellDataset.load_bundle(REGRESSION_DIR / "reference.pcellbundle")


# ── Summary ──────────────────────────────────────────────────────────


@pytest.mark.timeout(120)
def test_summary_counts(
    pipeline_result: ArenaDataset,
    reference: ArenaDataset,
) -> None:
    """Pipeline summary counts must match the reference."""
    assert pipeline_result.summary() == reference.summary()


# ── Deconvolution ────────────────────────────────────────────────────


def test_good_unit_ids(
    pipeline_result: ArenaDataset,
    reference: ArenaDataset,
) -> None:
    """Deconvolved unit IDs must match."""
    assert pipeline_result.good_unit_ids == reference.good_unit_ids


def test_event_index_shape(
    pipeline_result: ArenaDataset,
    reference: ArenaDataset,
) -> None:
    """Non-trivial event count must match.

    OASIS deconvolution can produce near-zero ghost events (s ~ 1e-17)
    whose presence varies across platforms.  Filter these out before
    comparing counts.
    """
    threshold = 1e-10
    got = (pipeline_result.event_index["s"] > threshold).sum()
    ref = (reference.event_index["s"] > threshold).sum()
    assert got == ref


# ── Event–place matching ─────────────────────────────────────────────


def test_event_place_shape(
    pipeline_result: ArenaDataset,
    reference: ArenaDataset,
) -> None:
    """Non-trivial matched event count must match (see event_index note)."""
    threshold = 1e-10
    got = (pipeline_result.event_place["s"] > threshold).sum()
    ref = (reference.event_place["s"] > threshold).sum()
    assert got == ref


# ── Occupancy ────────────────────────────────────────────────────────


def test_occupancy_map(
    pipeline_result: ArenaDataset,
    reference: ArenaDataset,
) -> None:
    """Occupancy map must match reference."""
    np.testing.assert_allclose(
        pipeline_result.occupancy_time,
        reference.occupancy_time,
        rtol=1e-5,
    )


def test_valid_mask(
    pipeline_result: ArenaDataset,
    reference: ArenaDataset,
) -> None:
    """Valid mask must match reference."""
    np.testing.assert_array_equal(
        pipeline_result.valid_mask,
        reference.valid_mask,
    )


# ── Per-unit analysis results ────────────────────────────────────────


def test_unit_result_ids(
    pipeline_result: ArenaDataset,
    reference: ArenaDataset,
) -> None:
    """Analyzed unit IDs must match."""
    assert sorted(pipeline_result.unit_results.keys()) == sorted(
        reference.unit_results.keys()
    )


def test_unit_scalars(
    pipeline_result: ArenaDataset,
    reference: ArenaDataset,
) -> None:
    """Per-unit scalar metrics (SI, p_val, stability) must match."""
    for uid in reference.unit_results:
        ref = reference.unit_results[uid]
        got = pipeline_result.unit_results[uid]

        assert got.si == pytest.approx(ref.si, rel=1e-5), f"unit {uid} SI"
        assert got.p_val == pytest.approx(ref.p_val, rel=1e-5), f"unit {uid} p_val"
        assert got.stability_corr == pytest.approx(
            ref.stability_corr, nan_ok=True, rel=1e-5
        ), f"unit {uid} stability_corr"
        assert got.stability_z == pytest.approx(
            ref.stability_z, nan_ok=True, rel=1e-5
        ), f"unit {uid} stability_z"
        assert got.stability_p_val == pytest.approx(
            ref.stability_p_val, nan_ok=True, rel=1e-5
        ), f"unit {uid} stability_p_val"


def test_rate_maps(
    pipeline_result: ArenaDataset,
    reference: ArenaDataset,
) -> None:
    """Per-unit rate maps must match."""
    for uid in reference.unit_results:
        ref_map = reference.unit_results[uid].rate_map
        got_map = pipeline_result.unit_results[uid].rate_map
        assert got_map.shape == ref_map.shape, f"unit {uid} rate_map shape"
        np.testing.assert_allclose(
            got_map,
            ref_map,
            rtol=1e-5,
            atol=1e-10,
            equal_nan=True,
            err_msg=f"unit {uid} rate_map",
        )
