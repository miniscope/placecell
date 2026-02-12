"""Central dataset class for place cell analysis."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from mio.logging import init_logger

from placecell.analysis import (
    compute_coverage_curve,
    compute_coverage_map,
    compute_occupancy_map,
    compute_unit_analysis,
)
from placecell.behavior import build_event_place_dataframe, load_curated_unit_ids
from placecell.config import AnalysisConfig, DataPathsConfig, SpatialMapConfig
from placecell.io import load_behavior_data, load_visualization_data
from placecell.neural import build_event_index_dataframe, load_calcium_traces, run_deconvolution

logger = init_logger(__name__)


class PlaceCellDataset:
    """Container for a single recording session's place cell analysis.

    Pipeline (each step populates attributes for the next)::

        ds = PlaceCellDataset.from_yaml(config_path, data_path)
        ds.load()                            # traces, trajectory, footprints
        ds.deconvolve(progress_bar=tqdm)     # good_unit_ids, S_list, event_index
        ds.match_events()                    # event_place
        ds.compute_occupancy()               # occupancy_time, valid_mask, edges
        ds.analyze_units(progress_bar=tqdm)  # unit_results

    Parameters
    ----------
    cfg : AnalysisConfig
        Merged analysis config (pipeline + data-specific overrides).
    neural_path : Path
        Directory containing neural zarr files.
    neural_timestamp_path : Path
        Path to neural timestamp CSV.
    behavior_position_path : Path
        Path to behavior position CSV.
    behavior_timestamp_path : Path
        Path to behavior timestamp CSV.
    curation_csv_path : Path or None
        Path to curation CSV, or None to use all units.
    """

    def __init__(
        self,
        cfg: AnalysisConfig,
        *,
        neural_path: Path,
        neural_timestamp_path: Path,
        behavior_position_path: Path,
        behavior_timestamp_path: Path,
        curation_csv_path: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.neural_path = neural_path
        self.neural_timestamp_path = neural_timestamp_path
        self.behavior_position_path = behavior_position_path
        self.behavior_timestamp_path = behavior_timestamp_path
        self.curation_csv_path = curation_csv_path

        # Neural data
        self.traces: xr.DataArray | None = None
        self.good_unit_ids: list[int] = []
        self.C_list: list[np.ndarray] = []
        self.S_list: list[np.ndarray] = []

        # Event data
        self.event_index: pd.DataFrame | None = None
        self.event_place: pd.DataFrame | None = None

        # Behavior data
        self.trajectory: pd.DataFrame | None = None
        self.trajectory_filtered: pd.DataFrame | None = None

        # Occupancy
        self.occupancy_time: np.ndarray | None = None
        self.valid_mask: np.ndarray | None = None
        self.x_edges: np.ndarray | None = None
        self.y_edges: np.ndarray | None = None

        # Visualization assets
        self.max_proj: np.ndarray | None = None
        self.footprints: xr.DataArray | None = None

        # Results
        self.unit_results: dict[int, dict] = {}

    @classmethod
    def from_yaml(cls, config_path: str | Path, data_path: str | Path) -> "PlaceCellDataset":
        """Create dataset from YAML config and data paths files."""
        config_path = Path(config_path)
        data_path = Path(data_path)
        data_dir = data_path.parent

        cfg = AnalysisConfig.from_yaml(config_path)
        data_cfg = DataPathsConfig.from_yaml(data_path)
        cfg = cfg.with_data_overrides(data_cfg)

        return cls(
            cfg=cfg,
            neural_path=data_dir / data_cfg.neural_path,
            neural_timestamp_path=data_dir / data_cfg.neural_timestamp,
            behavior_position_path=data_dir / data_cfg.behavior_position,
            behavior_timestamp_path=data_dir / data_cfg.behavior_timestamp,
            curation_csv_path=(
                data_dir / data_cfg.curation_csv if data_cfg.curation_csv else None
            ),
        )

    @property
    def spatial(self) -> SpatialMapConfig:
        """Shortcut to spatial map config."""
        return self.cfg.behavior.spatial_map

    @property
    def neural_fps(self) -> float:
        """Neural sampling rate in Hz."""
        return self.cfg.neural.fps

    def load(self) -> None:
        """Load neural traces, behavior data, and visualization assets."""
        ncfg = self.cfg.neural
        bcfg = self.cfg.behavior

        # Traces
        self.traces = load_calcium_traces(self.neural_path, trace_name=ncfg.trace_name)
        logger.info(
            "Loaded traces: %d units, %d frames",
            self.traces.sizes["unit_id"],
            self.traces.sizes["frame"],
        )

        # Behavior
        self.trajectory, self.trajectory_filtered = load_behavior_data(
            behavior_position=self.behavior_position_path,
            behavior_timestamp=self.behavior_timestamp_path,
            bodypart=bcfg.bodypart,
            speed_window_frames=bcfg.speed_window_frames,
            speed_threshold=bcfg.speed_threshold,
        )
        logger.info(
            "Trajectory: %d frames (%d after speed filter)",
            len(self.trajectory),
            len(self.trajectory_filtered),
        )

        # Visualization assets (max projection, footprints)
        self.traces, self.max_proj, self.footprints = load_visualization_data(
            neural_path=self.neural_path,
            trace_name=ncfg.trace_name,
        )

    def deconvolve(
        self,
        unit_ids: list[int] | None = None,
        progress_bar: Any = None,
    ) -> None:
        """Run OASIS deconvolution on calcium traces.

        Parameters
        ----------
        unit_ids:
            Specific unit IDs to process. None = all (respecting curation + max_units).
        progress_bar:
            Progress bar wrapper, e.g. ``tqdm``.
        """
        if self.traces is None:
            raise RuntimeError("Call load() first.")

        ncfg = self.cfg.neural
        oasis = ncfg.oasis

        all_unit_ids = list(map(int, self.traces["unit_id"].values))

        # Curation filter
        if self.curation_csv_path is not None and self.curation_csv_path.exists():
            curated = set(load_curated_unit_ids(self.curation_csv_path))
            all_unit_ids = [uid for uid in all_unit_ids if uid in curated]
            logger.info("After curation filter: %d units", len(all_unit_ids))

        # User-specified subset
        if unit_ids is not None:
            available = set(all_unit_ids)
            all_unit_ids = [uid for uid in unit_ids if uid in available]
            missing = set(unit_ids) - available
            if missing:
                logger.warning("Unit IDs not found: %s", sorted(missing))
            logger.info("Selected %d units", len(all_unit_ids))
        elif ncfg.max_units is not None and len(all_unit_ids) > ncfg.max_units:
            all_unit_ids = all_unit_ids[: ncfg.max_units]
            logger.info("Limited to first %d units", ncfg.max_units)

        logger.info("Deconvolving %d units (g=%s)...", len(all_unit_ids), oasis.g)

        self.good_unit_ids, self.C_list, self.S_list = run_deconvolution(
            C_da=self.traces,
            unit_ids=all_unit_ids,
            g=oasis.g,
            baseline=oasis.baseline,
            penalty=oasis.penalty,
            s_min=oasis.s_min,
            progress_bar=progress_bar,
        )

        self.event_index = build_event_index_dataframe(self.good_unit_ids, self.S_list)
        logger.info(
            "Deconvolved %d units, %d events", len(self.good_unit_ids), len(self.event_index)
        )

    def match_events(self) -> None:
        """Match neural events to behavior positions."""
        if self.event_index is None:
            raise RuntimeError("Call deconvolve() first.")
        if self.trajectory is None:
            raise RuntimeError("Call load() first.")

        bcfg = self.cfg.behavior

        self.event_place = build_event_place_dataframe(
            event_index=self.event_index,
            neural_timestamp_path=self.neural_timestamp_path,
            behavior_with_speed=self.trajectory,
            behavior_fps=bcfg.behavior_fps,
            speed_threshold=bcfg.speed_threshold,
        )
        logger.info(
            "Matched %d events (%d units)",
            len(self.event_place),
            self.event_place["unit_id"].nunique(),
        )

    def compute_occupancy(self) -> None:
        """Compute occupancy map from speed-filtered trajectory."""
        if self.trajectory_filtered is None:
            raise RuntimeError("Call load() first.")

        scfg = self.spatial

        self.occupancy_time, self.valid_mask, self.x_edges, self.y_edges = compute_occupancy_map(
            trajectory_df=self.trajectory_filtered,
            bins=scfg.bins,
            behavior_fps=self.cfg.behavior.behavior_fps,
            occupancy_sigma=scfg.occupancy_sigma,
            min_occupancy=scfg.min_occupancy,
        )
        logger.info(
            "Occupancy map: %s, %d/%d valid bins",
            self.occupancy_time.shape,
            self.valid_mask.sum(),
            self.valid_mask.size,
        )

    def analyze_units(self, progress_bar: Any = None) -> None:
        """Run spatial analysis for all deconvolved units with events.

        Parameters
        ----------
        progress_bar:
            Progress bar wrapper, e.g. ``tqdm``.
        """
        if self.event_place is None:
            raise RuntimeError("Call match_events() first.")
        if self.occupancy_time is None:
            raise RuntimeError("Call compute_occupancy() first.")

        scfg = self.spatial
        bcfg = self.cfg.behavior

        if scfg.random_seed is not None:
            np.random.seed(scfg.random_seed)

        df_filtered = self.event_place[self.event_place["speed"] >= bcfg.speed_threshold].copy()

        unique_units = sorted(df_filtered["unit_id"].unique())
        unique_units = [uid for uid in unique_units if uid in self.good_unit_ids]
        n_units = len(unique_units)
        logger.info("Analyzing %d units...", n_units)

        iterator = unique_units
        if progress_bar is not None:
            iterator = progress_bar(iterator)

        self.unit_results = {}
        for unit_id in iterator:
            result = compute_unit_analysis(
                unit_id=unit_id,
                df_filtered=df_filtered,
                trajectory_df=self.trajectory_filtered,
                occupancy_time=self.occupancy_time,
                valid_mask=self.valid_mask,
                x_edges=self.x_edges,
                y_edges=self.y_edges,
                activity_sigma=scfg.activity_sigma,
                event_threshold_sigma=scfg.event_threshold_sigma,
                n_shuffles=scfg.n_shuffles,
                random_seed=scfg.random_seed,
                behavior_fps=bcfg.behavior_fps,
                min_occupancy=scfg.min_occupancy,
                occupancy_sigma=scfg.occupancy_sigma,
                stability_threshold=scfg.stability_threshold,
                stability_method=scfg.stability_method,
                min_shift_seconds=scfg.min_shift_seconds,
                si_weight_mode=scfg.si_weight_mode,
                place_field_seed_percentile=scfg.place_field_seed_percentile,
            )

            # Attach visualization data
            vis_data_above = result["events_above_threshold"]
            vis_data_below = pd.DataFrame()
            if self.event_index is not None:
                unit_all = self.event_index[self.event_index["unit_id"] == unit_id]
                vis_data_below = unit_all[unit_all["s"] > result["vis_threshold"]]

            trace_data = None
            trace_times = None
            if self.traces is not None:
                try:
                    trace_data = self.traces.sel(unit_id=int(unit_id)).values
                    trace_times = np.arange(len(trace_data)) / self.neural_fps
                except (KeyError, IndexError):
                    pass

            self.unit_results[unit_id] = {
                "rate_map": result["rate_map"],
                "rate_map_raw": result["rate_map_raw"],
                "si": result["si"],
                "shuffled_sis": result["shuffled_sis"],
                "shuffled_rate_p95": result["shuffled_rate_p95"],
                "p_val": result["p_val"],
                "stability_corr": result["stability_corr"],
                "stability_z": result["stability_z"],
                "stability_p_val": result["stability_p_val"],
                "rate_map_first": result["rate_map_first"],
                "rate_map_second": result["rate_map_second"],
                "vis_data_above": vis_data_above,
                "vis_data_below": vis_data_below,
                "unit_data": result["unit_data"],
                "trace_data": trace_data,
                "trace_times": trace_times,
            }

        logger.info("Done. %d units analyzed.", len(self.unit_results))

    def place_cells(self) -> dict[int, dict]:
        """Return units passing both significance and stability tests."""
        p_thresh = self.spatial.p_value_threshold or 0.05
        stab_thresh = self.spatial.stability_threshold

        out = {}
        for uid, res in self.unit_results.items():
            if res["p_val"] >= p_thresh:
                continue
            stab_corr = res.get("stability_corr", np.nan)
            stab_p = res.get("stability_p_val", np.nan)
            if np.isnan(stab_corr):
                continue
            if not np.isnan(stab_p):
                if stab_p >= p_thresh:
                    continue
            else:
                if stab_corr < stab_thresh:
                    continue
            out[uid] = res
        return out

    def coverage(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute place field coverage map and curve.

        Returns (coverage_map, n_cells_array, coverage_fraction_array).
        """
        scfg = self.spatial
        pc = self.place_cells()

        coverage_map = compute_coverage_map(
            pc,
            threshold=scfg.place_field_threshold,
            min_bins=scfg.place_field_min_bins,
        )
        n_cells, fractions = compute_coverage_curve(
            pc,
            self.valid_mask,
            threshold=scfg.place_field_threshold,
            min_bins=scfg.place_field_min_bins,
        )
        return coverage_map, n_cells, fractions
