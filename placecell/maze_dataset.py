"""Dataset class for maze/tube 1D place cell analysis."""

from typing import Any

import numpy as np
import pandas as pd

from placecell.analysis_1d import (
    compute_occupancy_map_1d,
    compute_unit_analysis_1d,
)
from placecell.config import SpatialMap1DConfig
from placecell.dataset import PlaceCellDataset, UnitResult
from placecell.logging import init_logger
from placecell.maze import compute_speed_1d, filter_tube_by_speed, serialize_tube_position

logger = init_logger(__name__)


class MazeDataset(PlaceCellDataset):
    """Extension of PlaceCellDataset for 1D tube/maze analysis.

    Overrides the behavior preprocessing, occupancy computation, and
    unit analysis steps to work on a concatenated 1D axis.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trajectory_1d: pd.DataFrame | None = None
        self.trajectory_1d_filtered: pd.DataFrame | None = None
        self.pos_range: tuple[float, float] | None = None
        self.edges_1d: np.ndarray | None = None
        self.tube_boundaries: list[float] = []

    @property
    def maze_cfg(self):
        """Shortcut to maze config."""
        return self.cfg.behavior.maze

    @property
    def spatial_1d(self) -> SpatialMap1DConfig:
        """Shortcut to 1D spatial map config."""
        return self.cfg.behavior.spatial_map_1d

    def preprocess_behavior(self) -> None:
        """Load behavior, serialize to 1D, compute speed, and filter.

        1. Calls super() to load standard trajectory (needed for timestamps).
        2. Serializes tube_position to concatenated 1D axis.
        3. Computes 1D speed within tubes.
        4. Applies speed filter.
        """
        super().preprocess_behavior()

        if self.trajectory is None:
            raise RuntimeError("Behavior data not loaded.")

        mcfg = self.maze_cfg
        if mcfg is None:
            raise RuntimeError("MazeDataset requires maze config.")

        # Load extra columns (zone, tube_position) from the behavior CSV
        self._load_maze_columns()

        # Serialize to 1D
        self.trajectory_1d = serialize_tube_position(
            self.trajectory,
            tube_order=mcfg.tube_order,
            zone_column=mcfg.zone_column,
            tube_position_column=mcfg.tube_position_column,
        )

        # Compute 1D speed
        bcfg = self.cfg.behavior
        self.trajectory_1d = compute_speed_1d(
            self.trajectory_1d,
            window_frames=bcfg.speed_window_frames,
        )

        # Speed filter
        self.trajectory_1d_filtered = filter_tube_by_speed(
            self.trajectory_1d,
            speed_threshold=bcfg.speed_threshold,
        )

        n_tubes = len(mcfg.tube_order)
        self.pos_range = (0.0, float(n_tubes))
        self.tube_boundaries = [float(i) for i in range(n_tubes + 1)]

        logger.info(
            "1D trajectory: %d frames (%d after speed filter), %d tubes",
            len(self.trajectory_1d),
            len(self.trajectory_1d_filtered),
            n_tubes,
        )

    def _load_maze_columns(self) -> None:
        """Load zone and tube_position columns from behavior CSV into trajectory."""
        mcfg = self.maze_cfg
        zone_col = mcfg.zone_column
        tp_col = mcfg.tube_position_column

        if zone_col in self.trajectory.columns and tp_col in self.trajectory.columns:
            return  # Already present

        # Re-read the behavior CSV to get the extra columns
        import pandas as pd

        df = pd.read_csv(self.behavior_position_path, header=[0, 1, 2])
        scorer = df.columns[1][0]
        bodypart = self.cfg.behavior.bodypart

        # Extract zone and tube_position from multi-index
        zone_key = (scorer, bodypart, zone_col)
        tp_key = (scorer, bodypart, tp_col)

        if zone_key in df.columns:
            self.trajectory[zone_col] = df[zone_key].values[: len(self.trajectory)]
        else:
            raise ValueError(
                f"Column '{zone_col}' not found for bodypart '{bodypart}' in behavior CSV."
            )

        if tp_key in df.columns:
            self.trajectory[tp_col] = pd.to_numeric(
                df[tp_key].values[: len(self.trajectory)], errors="coerce"
            )
        else:
            raise ValueError(
                f"Column '{tp_col}' not found for bodypart '{bodypart}' in behavior CSV."
            )

    def match_events(self) -> None:
        """Match events to behavior, then add 1D position.

        Calls the parent match_events (which matches neural events to x, y),
        then adds pos_1d by joining on beh_frame_index from trajectory_1d.
        """
        super().match_events()

        if self.trajectory_1d is None:
            raise RuntimeError("Call preprocess_behavior() first.")

        # Add pos_1d to event_place by joining on behavior frame
        pos_lookup = self.trajectory_1d[["frame_index", "pos_1d", "tube_index"]].rename(
            columns={"frame_index": "beh_frame_index"}
        )
        self.event_place = self.event_place.merge(pos_lookup, on="beh_frame_index", how="left")

        # Drop events not in tubes (pos_1d will be NaN)
        n_before = len(self.event_place)
        self.event_place = self.event_place.dropna(subset=["pos_1d"]).reset_index(drop=True)
        logger.info(
            "1D event matching: %d/%d events in tubes",
            len(self.event_place),
            n_before,
        )

    def compute_occupancy(self) -> None:
        """Compute 1D occupancy from speed-filtered tube trajectory."""
        if self.trajectory_1d_filtered is None:
            raise RuntimeError("Call preprocess_behavior() first.")

        scfg = self.spatial_1d
        n_tubes = len(self.maze_cfg.tube_order)
        n_bins = scfg.bins_per_tube * n_tubes

        self.occupancy_time, self.valid_mask, self.edges_1d = compute_occupancy_map_1d(
            trajectory_df=self.trajectory_1d_filtered,
            n_bins=n_bins,
            pos_range=self.pos_range,
            behavior_fps=self.cfg.behavior.behavior_fps,
            occupancy_sigma=scfg.occupancy_sigma,
            min_occupancy=scfg.min_occupancy,
        )

        # Store in x_edges for compatibility with save_bundle
        self.x_edges = self.edges_1d
        self.y_edges = None

        logger.info(
            "1D occupancy: %d bins, %d/%d valid",
            n_bins,
            self.valid_mask.sum(),
            self.valid_mask.size,
        )

    def analyze_units(self, progress_bar: Any = None) -> None:
        """Run 1D spatial analysis for all deconvolved units."""
        if self.event_place is None:
            raise RuntimeError("Call match_events() first.")
        if self.occupancy_time is None:
            raise RuntimeError("Call compute_occupancy() first.")

        scfg = self.spatial_1d
        bcfg = self.cfg.behavior

        if scfg.random_seed is not None:
            np.random.seed(scfg.random_seed)

        df_filtered = self.event_place[self.event_place["speed"] >= bcfg.speed_threshold].copy()

        unique_units = sorted(df_filtered["unit_id"].unique())
        unique_units = [uid for uid in unique_units if uid in self.good_unit_ids]
        logger.info("Analyzing %d units (1D)...", len(unique_units))

        iterator = unique_units
        if progress_bar is not None:
            iterator = progress_bar(iterator)

        self.unit_results = {}
        for unit_id in iterator:
            result = compute_unit_analysis_1d(
                unit_id=unit_id,
                df_filtered=df_filtered,
                trajectory_df=self.trajectory_1d_filtered,
                occupancy_time=self.occupancy_time,
                valid_mask=self.valid_mask,
                edges=self.edges_1d,
                activity_sigma=scfg.activity_sigma,
                event_threshold_sigma=scfg.event_threshold_sigma,
                n_shuffles=scfg.n_shuffles,
                random_seed=scfg.random_seed,
                behavior_fps=bcfg.behavior_fps,
                min_occupancy=scfg.min_occupancy,
                occupancy_sigma=scfg.occupancy_sigma,
                min_shift_seconds=scfg.min_shift_seconds,
                si_weight_mode=scfg.si_weight_mode,
                place_field_seed_percentile=scfg.place_field_seed_percentile,
                n_split_blocks=scfg.n_split_blocks,
                block_shifts=scfg.block_shifts,
            )

            trace_data = None
            trace_times = None
            if self.traces is not None:
                try:
                    trace_data = self.traces.sel(unit_id=int(unit_id)).values
                    trace_times = np.arange(len(trace_data)) / self.neural_fps
                except (KeyError, IndexError):
                    pass

            self.unit_results[unit_id] = UnitResult(
                rate_map=result["rate_map"],
                rate_map_raw=result["rate_map_raw"],
                si=result["si"],
                p_val=result["p_val"],
                shuffled_sis=result["shuffled_sis"],
                shuffled_rate_p95=result["shuffled_rate_p95"],
                stability_corr=result["stability_corr"],
                stability_z=result["stability_z"],
                stability_p_val=result["stability_p_val"],
                shuffled_stability=result["shuffled_stability"],
                rate_map_first=result["rate_map_first"],
                rate_map_second=result["rate_map_second"],
                vis_data_above=result["events_above_threshold"],
                unit_data=result["unit_data"],
                trace_data=trace_data,
                trace_times=trace_times,
            )

        logger.info("Done. %d units analyzed (1D).", len(self.unit_results))

    def summary(self) -> dict[str, int]:
        """Compute summary counts using 1D config thresholds."""
        p_thresh = self.spatial_1d.p_value_threshold
        n_sig = 0
        n_stable = 0
        n_place_cells = 0
        for res in self.unit_results.values():
            is_sig = res.p_val < p_thresh
            is_stable = not np.isnan(res.stability_p_val) and res.stability_p_val < p_thresh
            if is_sig:
                n_sig += 1
            if is_stable:
                n_stable += 1
            if is_sig and is_stable:
                n_place_cells += 1
        return {
            "n_total": len(self.unit_results),
            "n_sig": n_sig,
            "n_stable": n_stable,
            "n_place_cells": n_place_cells,
        }

    def save_bundle(self, path) -> "Path":
        """Save bundle, including 1D trajectory data."""
        from pathlib import Path

        result = super().save_bundle(path)

        # Also save 1D-specific data
        if self.trajectory_1d is not None:
            self.trajectory_1d.to_parquet(result / "trajectory_1d.parquet")
        if self.trajectory_1d_filtered is not None:
            self.trajectory_1d_filtered.to_parquet(result / "trajectory_1d_filtered.parquet")

        return result
