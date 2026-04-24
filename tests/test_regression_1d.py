"""Full-pipeline regression test for the 1D maze analysis.

Runs the complete MazeDataset pipeline on a small data subset and
compares every output against a saved reference bundle.

To regenerate the reference bundle with low ``n_shuffles`` in the config::

    placecell analysis -c config.yaml -d data.yaml -o tests/assets/regression_1d/reference -y --subset-units 10 --subset-frames 10000
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from placecell.dataset.base import BasePlaceCellDataset
from placecell.dataset.maze import MazeDataset

REGRESSION_DIR = Path(__file__).parent / "assets" / "regression_1d"


@pytest.fixture(scope="module")
def pipeline_result() -> MazeDataset:
    """Run the full 1D maze pipeline once for all tests in this module."""
    ds = BasePlaceCellDataset.from_yaml(
        REGRESSION_DIR / "analysis_config.yaml",
        REGRESSION_DIR / "data_paths.yaml",
    )
    assert isinstance(ds, MazeDataset)
    ds.load()
    ds.preprocess_behavior()
    ds.deconvolve()
    ds.match_events()
    ds.compute_occupancy()
    ds.analyze_units()
    return ds


@pytest.fixture(scope="module")
def reference() -> MazeDataset:
    """Load the reference bundle."""
    ds = BasePlaceCellDataset.load_bundle(REGRESSION_DIR / "reference.pcellbundle")
    assert isinstance(ds, MazeDataset)
    return ds


@pytest.mark.timeout(120)
def test_summary_counts(
    pipeline_result: MazeDataset,
    reference: MazeDataset,
) -> None:
    """Pipeline summary counts must match the reference."""
    assert pipeline_result.summary() == reference.summary()


def test_good_unit_ids(
    pipeline_result: MazeDataset,
    reference: MazeDataset,
) -> None:
    """Deconvolved unit IDs must match."""
    assert pipeline_result.good_unit_ids == reference.good_unit_ids


def test_event_index_shape(
    pipeline_result: MazeDataset,
    reference: MazeDataset,
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


def test_event_place_shape(
    pipeline_result: MazeDataset,
    reference: MazeDataset,
) -> None:
    """Non-trivial matched event count must match (see event_index note)."""
    threshold = 1e-10
    got = (pipeline_result.event_place["s"] > threshold).sum()
    ref = (reference.event_place["s"] > threshold).sum()
    assert got == ref


def test_event_place_has_pos_1d(
    pipeline_result: MazeDataset,
) -> None:
    """Matched events must have pos_1d column from 1D matching."""
    assert "pos_1d" in pipeline_result.event_place.columns
    assert pipeline_result.event_place["pos_1d"].notna().all()


def test_trajectory_1d_shape(
    pipeline_result: MazeDataset,
    reference: MazeDataset,
) -> None:
    """1D trajectory frame count must match."""
    assert len(pipeline_result.trajectory_1d) == len(reference.trajectory_1d)


def test_trajectory_1d_filtered_shape(
    pipeline_result: MazeDataset,
    reference: MazeDataset,
) -> None:
    """Speed-filtered 1D trajectory frame count must match."""
    assert len(pipeline_result.trajectory_1d_filtered) == len(
        reference.trajectory_1d_filtered
    )


def test_arm_boundaries(
    pipeline_result: MazeDataset,
    reference: MazeDataset,
) -> None:
    """Arm boundaries must match reference."""
    np.testing.assert_allclose(
        pipeline_result.arm_boundaries,
        reference.arm_boundaries,
        rtol=1e-5,
    )


def test_effective_arm_order(
    pipeline_result: MazeDataset,
    reference: MazeDataset,
) -> None:
    """Effective arm order (with direction splits) must match."""
    assert pipeline_result.effective_arm_order == reference.effective_arm_order


def test_pos_range(
    pipeline_result: MazeDataset,
    reference: MazeDataset,
) -> None:
    """Position range must match reference."""
    np.testing.assert_allclose(
        pipeline_result.pos_range,
        reference.pos_range,
        rtol=1e-5,
    )


def test_occupancy_map(
    pipeline_result: MazeDataset,
    reference: MazeDataset,
) -> None:
    """1D occupancy map must match reference."""
    np.testing.assert_allclose(
        pipeline_result.occupancy_time,
        reference.occupancy_time,
        rtol=1e-5,
    )


def test_valid_mask(
    pipeline_result: MazeDataset,
    reference: MazeDataset,
) -> None:
    """Valid mask must match reference."""
    np.testing.assert_array_equal(
        pipeline_result.valid_mask,
        reference.valid_mask,
    )


def test_edges_1d(
    pipeline_result: MazeDataset,
    reference: MazeDataset,
) -> None:
    """1D bin edges must match reference."""
    np.testing.assert_allclose(
        pipeline_result.edges_1d,
        reference.edges_1d,
        rtol=1e-5,
    )


def test_unit_result_ids(
    pipeline_result: MazeDataset,
    reference: MazeDataset,
) -> None:
    """Analyzed unit IDs must match."""
    assert sorted(pipeline_result.unit_results.keys()) == sorted(
        reference.unit_results.keys()
    )


def test_unit_scalars(
    pipeline_result: MazeDataset,
    reference: MazeDataset,
) -> None:
    """Per-unit scalar metrics (SI, p_val, stability) must match."""
    for uid in reference.unit_results:
        ref = reference.unit_results[uid]
        got = pipeline_result.unit_results[uid]

        assert got.si == pytest.approx(ref.si, rel=1e-5), f"unit {uid} SI"
        assert got.p_val == pytest.approx(ref.p_val, rel=1e-5), f"unit {uid} p_val"
        assert len(got.stability_splits) == len(ref.stability_splits), (
            f"unit {uid} stability_splits length"
        )
        for i, (g, r) in enumerate(zip(got.stability_splits, ref.stability_splits)):
            assert g.n_split_blocks == r.n_split_blocks, f"unit {uid} split {i} n_split_blocks"
            assert g.corr == pytest.approx(r.corr, nan_ok=True, rel=1e-5), (
                f"unit {uid} split {i} corr"
            )
            assert g.fisher_z == pytest.approx(r.fisher_z, nan_ok=True, rel=1e-5), (
                f"unit {uid} split {i} fisher_z"
            )
            assert g.p_val == pytest.approx(r.p_val, nan_ok=True, rel=1e-5), (
                f"unit {uid} split {i} p_val"
            )


def test_rate_maps(
    pipeline_result: MazeDataset,
    reference: MazeDataset,
) -> None:
    """Per-unit 1D rate maps must match."""
    for uid in reference.unit_results:
        ref_map = reference.unit_results[uid].rate_map_smoothed
        got_map = pipeline_result.unit_results[uid].rate_map_smoothed
        assert got_map.shape == ref_map.shape, f"unit {uid} rate_map shape"
        np.testing.assert_allclose(
            got_map,
            ref_map,
            rtol=1e-5,
            atol=1e-10,
            equal_nan=True,
            err_msg=f"unit {uid} rate_map_smoothed",
        )


def test_save_load_bundle_roundtrip(
    pipeline_result: MazeDataset,
) -> None:
    """save_bundle → load_bundle must round-trip without error and preserve results."""
    with tempfile.TemporaryDirectory() as tmp:
        bundle_path = pipeline_result.save_bundle(
            Path(tmp) / "test", save_figures=False
        )
        reloaded = BasePlaceCellDataset.load_bundle(bundle_path)

    assert isinstance(reloaded, MazeDataset)
    assert reloaded.summary() == pipeline_result.summary()
    assert sorted(reloaded.unit_results.keys()) == sorted(
        pipeline_result.unit_results.keys()
    )
    for uid in pipeline_result.unit_results:
        np.testing.assert_allclose(
            reloaded.unit_results[uid].rate_map_smoothed,
            pipeline_result.unit_results[uid].rate_map_smoothed,
            rtol=1e-5,
            equal_nan=True,
            err_msg=f"unit {uid} rate_map_smoothed round-trip",
        )


def _copy_regression_to_tmp(tmp_dir: Path) -> Path:
    """Copy the maze regression assets into a tmpdir for mutation tests.

    Returns the path to the copied data_paths.yaml.
    """
    for entry in REGRESSION_DIR.iterdir():
        if entry.name == "reference.pcellbundle":
            continue
        dst = tmp_dir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dst)
        else:
            shutil.copy2(entry, dst)
    return tmp_dir / "data_paths.yaml"


def test_load_auto_runs_detect_zones_when_missing() -> None:
    """MazeDataset.load() should auto-run detect-zones when the cached CSV is gone."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        data_path = _copy_regression_to_tmp(tmp_dir)
        zone_csv = tmp_dir / "zone_tracking.csv"
        zone_csv.unlink()  # delete the cached projection
        assert not zone_csv.exists()

        ds = BasePlaceCellDataset.from_yaml(
            REGRESSION_DIR / "analysis_config.yaml",
            data_path,
        )
        assert isinstance(ds, MazeDataset)
        ds.load()

        assert zone_csv.exists(), "zone_tracking.csv should be regenerated by load()"
        assert "zone" in ds.trajectory.columns
        assert "arm_position" in ds.trajectory.columns


def test_zone_tracking_path_defaults_when_unset() -> None:
    """Omitting zone_tracking from the data config falls back to a stable default."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        data_path = _copy_regression_to_tmp(tmp_dir)

        # Strip zone_tracking from the data config and remove the cached CSV.
        text = data_path.read_text()
        text = "\n".join(
            line for line in text.splitlines() if not line.startswith("zone_tracking:")
        )
        data_path.write_text(text + "\n")
        (tmp_dir / "zone_tracking.csv").unlink()

        ds = BasePlaceCellDataset.from_yaml(
            REGRESSION_DIR / "analysis_config.yaml",
            data_path,
        )
        assert isinstance(ds, MazeDataset)
        # Default lives next to the data config: zone_tracking_{data_path.stem}.csv.
        expected = tmp_dir / f"zone_tracking_{data_path.stem}.csv"
        assert ds.zone_tracking_path == expected
        ds.load()  # auto-runs detect-zones into the default location
        assert expected.exists()


def test_load_force_redetect_overwrites_existing_csv() -> None:
    """force_redetect=True should re-run detect-zones even if the file exists."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        data_path = _copy_regression_to_tmp(tmp_dir)
        zone_csv = tmp_dir / "zone_tracking.csv"
        assert zone_csv.exists()
        before_mtime = zone_csv.stat().st_mtime_ns

        ds = BasePlaceCellDataset.from_yaml(
            REGRESSION_DIR / "analysis_config.yaml",
            data_path,
        )
        assert isinstance(ds, MazeDataset)
        ds.load(force_redetect=True)

        after_mtime = zone_csv.stat().st_mtime_ns
        assert after_mtime > before_mtime, "force_redetect should rewrite the CSV"


def test_save_bundle_includes_maze_summary_figures(
    pipeline_result: MazeDataset,
) -> None:
    """Maze figure export should include the fused global PVO matrix."""
    with tempfile.TemporaryDirectory() as tmp:
        bundle_path = pipeline_result.save_bundle(Path(tmp) / "test", save_figures=True)

        assert (bundle_path / "figures" / "occupancy.pdf").exists()
        assert (bundle_path / "figures" / "population_rate_map.pdf").exists()
        assert (bundle_path / "figures" / "graph_overlay.pdf").exists()
        # PVO matrix is only generated when the dataset has place cells
        has_place_cells = bool(pipeline_result.place_cells())
        assert (bundle_path / "figures" / "global_pvo_matrix.pdf").exists() == has_place_cells
