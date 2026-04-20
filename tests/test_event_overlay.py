"""Smoke tests for the cell-event overlay and zone-occupancy helpers.

Shares the 1D maze regression fixture so the pipeline runs once.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")

from placecell.dataset.base import BasePlaceCellDataset
from placecell.dataset.maze import MazeDataset
from placecell.event_overlay import (
    ProjectionConfig,
    ViewConfig,
    gather_events,
    plot_cross_session_occupancy,
    plot_event_overlay_2d,
    plot_event_overlay_3d,
    plot_zone_occupancy,
)

REGRESSION_DIR = Path(__file__).parent / "assets" / "regression_1d"


@pytest.fixture(scope="module")
def ds() -> MazeDataset:
    """Run the 1D maze pipeline once for this module."""
    dataset = BasePlaceCellDataset.from_yaml(
        REGRESSION_DIR / "analysis_config.yaml",
        REGRESSION_DIR / "data_paths.yaml",
    )
    assert isinstance(dataset, MazeDataset)
    dataset.load()
    dataset.preprocess_behavior()
    dataset.deconvolve()
    dataset.match_events()
    dataset.compute_occupancy()
    dataset.analyze_units()
    return dataset


@pytest.fixture
def selected_units(ds: MazeDataset) -> list[int]:
    """Return up to 3 place-cell (or unit) ids for overlay tests."""
    pcs = ds.place_cells()
    if pcs:
        return list(pcs.keys())[:3]
    return list(ds.unit_results.keys())[:3]


def test_gather_events_percentile_reduces_count(
    ds: MazeDataset, selected_units: list[int]
) -> None:
    """Higher percentile threshold never keeps more events than a lower one."""
    lenient = gather_events(ds, selected_units, event_percentile=0)
    strict = gather_events(ds, selected_units, event_percentile=95.0)
    for uid in selected_units:
        assert len(strict[uid]) <= len(lenient[uid])


def test_plot_zone_occupancy_runs(ds: MazeDataset) -> None:
    """Zone occupancy plot runs in both absolute and proportional modes."""
    fig = plot_zone_occupancy(ds, proportion=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    fig = plot_zone_occupancy(ds, proportion=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_cross_session_occupancy_runs(ds: MazeDataset) -> None:
    """Cross-session aggregation renders without error."""
    fig = plot_cross_session_occupancy({"a": ds, "b": ds})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_event_overlay_2d_runs(
    ds: MazeDataset, selected_units: list[int]
) -> None:
    """2D overlay returns a Figure."""
    fig = plot_event_overlay_2d(ds, selected_units)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_event_overlay_3d_traversal_subset(
    ds: MazeDataset, selected_units: list[int]
) -> None:
    """3D overlay with ViewConfig, ProjectionConfig, and traversal subset."""
    fig = plot_event_overlay_3d(
        ds,
        selected_units,
        view=ViewConfig(elev=20, azim=-45, box_aspect=(1, 1, 1)),
        projection=ProjectionConfig(z=-10.0, box=True),
        traversal_subset=True,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
