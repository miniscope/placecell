"""Dataset class for maze/tube 1D place cell analysis."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from placecell.analysis_1d import (
    compute_occupancy_map_1d,
    compute_unit_analysis_1d,
)
from placecell.config import MazeConfig, SpatialMap1DConfig
from placecell.dataset import BasePlaceCellDataset, UnitResult
from placecell.logging import init_logger
from placecell.maze import (
    assign_traversal_direction,
    compute_speed_1d,
    compute_tube_lengths,
    filter_tube_by_speed,
    load_graph_polylines,
    serialize_tube_position,
)

logger = init_logger(__name__)


class MazeDataset(BasePlaceCellDataset):
    """Dataset for 1D tube/maze place cell analysis.

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
        self.effective_tube_order: list[str] = []
        self.tube_lengths: dict[str, float] | None = None
        self.segment_bins: list[int] | None = None
        self.graph_polylines: dict[str, list[list[float]]] | None = None
        self.graph_mm_per_pixel: float | None = None

    @property
    def maze_cfg(self) -> "MazeConfig | None":
        """Shortcut to maze config."""
        return self.cfg.behavior.maze

    @property
    def spatial_1d(self) -> SpatialMap1DConfig:
        """Shortcut to 1D spatial map config."""
        return self.cfg.behavior.spatial_map_1d

    @property
    def p_value_threshold(self) -> float:
        """P-value threshold from 1D spatial map config."""
        return self.spatial_1d.p_value_threshold

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

        # Load behavior graph for physical tube lengths (optional)
        if self.behavior_graph_path is not None and self.behavior_graph_path.exists():
            self.graph_polylines, self.graph_mm_per_pixel = load_graph_polylines(
                self.behavior_graph_path
            )
            zone_lengths, _ = compute_tube_lengths(self.behavior_graph_path)
            # Keep only the tubes we're using
            self.tube_lengths = {t: zone_lengths[t] for t in mcfg.tube_order if t in zone_lengths}
        else:
            self.tube_lengths = None

        # Serialize to 1D
        self.trajectory_1d = serialize_tube_position(
            self.trajectory,
            tube_order=mcfg.tube_order,
            zone_column=mcfg.zone_column,
            tube_position_column=mcfg.tube_position_column,
            tube_lengths=self.tube_lengths,
        )

        # Optionally split by traversal direction (doubles segments)
        if mcfg.split_by_direction:
            self.trajectory_1d, self.effective_tube_order = assign_traversal_direction(
                self.trajectory_1d,
                tube_order=mcfg.tube_order,
                zone_column=mcfg.zone_column,
                tube_position_column=mcfg.tube_position_column,
                tube_lengths=self.tube_lengths,
            )
        else:
            self.effective_tube_order = list(mcfg.tube_order)

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

        # Compute pos_range and tube_boundaries
        n_segments = len(self.effective_tube_order)
        if self.tube_lengths is not None:
            # Physical scaling: each segment spans its real length
            seg_lengths = []
            for seg_name in self.effective_tube_order:
                # Strip _fwd/_rev suffix to find parent tube
                base = (
                    seg_name.rsplit("_", 1)[0] if seg_name.endswith(("_fwd", "_rev")) else seg_name
                )
                seg_lengths.append(self.tube_lengths.get(base, 1.0))
            cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            self.tube_boundaries = cumulative.tolist()
            self.pos_range = (0.0, cumulative[-1])
        else:
            self.pos_range = (0.0, float(n_segments))
            self.tube_boundaries = [float(i) for i in range(n_segments + 1)]

        logger.info(
            "1D trajectory: %d frames (%d after speed filter), %d segments%s, "
            "pos_range=(%.1f, %.1f)",
            len(self.trajectory_1d),
            len(self.trajectory_1d_filtered),
            n_segments,
            " (direction split)" if mcfg.split_by_direction else "",
            self.pos_range[0],
            self.pos_range[1],
        )

    def _load_maze_columns(self) -> None:
        """Load zone and tube_position columns from behavior CSV into trajectory.

        Joins on ``frame_index`` to handle trajectories that have been
        trimmed or reordered by speed computation / time-range filtering.
        """
        mcfg = self.maze_cfg
        zone_col = mcfg.zone_column
        tp_col = mcfg.tube_position_column

        if zone_col in self.trajectory.columns and tp_col in self.trajectory.columns:
            return  # Already present

        # Re-read the behavior CSV to get the extra columns
        df = pd.read_csv(self.behavior_position_path, header=[0, 1, 2])
        scorer = df.columns[1][0]
        bodypart = self.cfg.behavior.bodypart

        # Extract zone and tube_position from multi-index
        zone_key = (scorer, bodypart, zone_col)
        tp_key = (scorer, bodypart, tp_col)

        if zone_key not in df.columns:
            raise ValueError(
                f"Column '{zone_col}' not found for bodypart '{bodypart}' in behavior CSV."
            )
        if tp_key not in df.columns:
            raise ValueError(
                f"Column '{tp_col}' not found for bodypart '{bodypart}' in behavior CSV."
            )

        # Build a lookup keyed by frame_index (first column of the CSV)
        frame_index = df.iloc[:, 0].values
        lookup = pd.DataFrame(
            {
                "frame_index": frame_index,
                zone_col: df[zone_key].values,
                tp_col: pd.to_numeric(df[tp_key].values, errors="coerce"),
            }
        )

        # Merge on frame_index so alignment survives trimming / reordering
        self.trajectory = self.trajectory.merge(lookup, on="frame_index", how="left")
        logger.info(
            "Loaded maze columns via frame_index join: %d/%d frames matched",
            self.trajectory[zone_col].notna().sum(),
            len(self.trajectory),
        )

    def match_events(self) -> None:
        """Match events to behavior, then add 1D position.

        Calls the parent match_events (which matches neural events to x, y),
        then adds pos_1d by joining on beh_frame_index from trajectory_1d.
        """
        super().match_events()

        if self.trajectory_1d is None:
            raise RuntimeError("Call preprocess_behavior() first.")

        # Add pos_1d and speed_1d to event_place by joining on behavior frame
        lookup_cols = ["frame_index", "pos_1d", "tube_index", "speed_1d"]
        if "direction" in self.trajectory_1d.columns:
            lookup_cols.append("direction")
        pos_lookup = self.trajectory_1d[lookup_cols].rename(
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
        n_segments = len(self.effective_tube_order)

        total_length = self.pos_range[1] - self.pos_range[0]
        n_bins = max(n_segments, round(total_length / scfg.bin_width_mm))

        # Compute segment bin boundaries from tube boundaries
        edges_tmp = np.linspace(self.pos_range[0], self.pos_range[1], n_bins + 1)
        self.segment_bins = [int(np.searchsorted(edges_tmp, b)) for b in self.tube_boundaries]

        self.occupancy_time, self.valid_mask, self.edges_1d = compute_occupancy_map_1d(
            trajectory_df=self.trajectory_1d_filtered,
            n_bins=n_bins,
            pos_range=self.pos_range,
            behavior_fps=self.cfg.behavior.behavior_fps,
            occupancy_sigma=scfg.occupancy_sigma,
            min_occupancy=scfg.min_occupancy,
            segment_bins=self.segment_bins,
        )

        # Store in x_edges for compatibility with save_bundle
        self.x_edges = self.edges_1d
        self.y_edges = None

        logger.info(
            "1D occupancy: %d bins (%d segments), %d/%d valid",
            n_bins,
            n_segments,
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

        # Use 1D tube speed (same criterion as occupancy), not 2D camera speed
        df_filtered = self.event_place[self.event_place["speed_1d"] >= bcfg.speed_threshold].copy()

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
                n_shuffles=scfg.n_shuffles,
                random_seed=scfg.random_seed,
                behavior_fps=bcfg.behavior_fps,
                min_occupancy=scfg.min_occupancy,
                occupancy_sigma=scfg.occupancy_sigma,
                min_shift_seconds=scfg.min_shift_seconds,
                si_weight_mode=scfg.si_weight_mode,
                n_split_blocks=scfg.n_split_blocks,
                block_shifts=scfg.block_shifts,
                segment_bins=self.segment_bins,
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
                shuffled_rate_p95=None,
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

    def save_bundle(self, path: "str | Path") -> "Path":
        """Save bundle, including 1D trajectory and maze metadata."""

        result = super().save_bundle(path)

        # 1D trajectories
        if self.trajectory_1d is not None:
            self.trajectory_1d.to_parquet(result / "trajectory_1d.parquet")
        if self.trajectory_1d_filtered is not None:
            self.trajectory_1d_filtered.to_parquet(result / "trajectory_1d_filtered.parquet")

        # Maze metadata needed for visualization
        maze_meta: dict[str, Any] = {
            "tube_boundaries": self.tube_boundaries,
            "effective_tube_order": self.effective_tube_order,
            "tube_lengths": self.tube_lengths,
            "segment_bins": self.segment_bins,
            "pos_range": list(self.pos_range) if self.pos_range else None,
            "graph_polylines": self.graph_polylines,
            "graph_mm_per_pixel": self.graph_mm_per_pixel,
        }
        (result / "maze_meta.json").write_text(json.dumps(maze_meta, indent=2))

        return result

    @classmethod
    def load_bundle(cls, path: str | Path) -> "MazeDataset":
        """Load a saved ``.pcellbundle`` that contains 1D maze data.

        Restores all base attributes via the parent loader, then adds
        1D-specific state (trajectories, tube boundaries, etc.).
        """
        path = Path(path)

        # Use the parent loader to get a base dataset, then upgrade
        base = BasePlaceCellDataset.load_bundle.__func__(cls, path)

        # Restore 1D trajectories
        t1d_path = path / "trajectory_1d.parquet"
        if t1d_path.exists():
            base.trajectory_1d = pd.read_parquet(t1d_path)
        t1df_path = path / "trajectory_1d_filtered.parquet"
        if t1df_path.exists():
            base.trajectory_1d_filtered = pd.read_parquet(t1df_path)

        # Restore maze metadata
        meta_path = path / "maze_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            base.tube_boundaries = meta.get("tube_boundaries", [])
            base.effective_tube_order = meta.get("effective_tube_order", [])
            base.tube_lengths = meta.get("tube_lengths")
            base.segment_bins = meta.get("segment_bins")
            pr = meta.get("pos_range")
            base.pos_range = tuple(pr) if pr else None
            base.graph_polylines = meta.get("graph_polylines")
            base.graph_mm_per_pixel = meta.get("graph_mm_per_pixel")

        # edges_1d is stored as x_edges
        base.edges_1d = base.x_edges

        return base
