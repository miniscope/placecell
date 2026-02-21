"""Dataset class for maze/arm 1D place cell analysis."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from placecell.analysis_1d import (
    compute_occupancy_map_1d,
    compute_unit_analysis_1d,
)
from placecell.behavior import build_event_place_dataframe
from placecell.config import SpatialMap1DConfig
from placecell.dataset import BasePlaceCellDataset, UnitResult
from placecell.logging import init_logger
from placecell.maze import (
    assign_traversal_direction,
    compute_arm_lengths,
    compute_speed_1d,
    filter_arm_by_speed,
    filter_complete_traversals,
    load_graph_polylines,
    serialize_arm_position,
)

logger = init_logger(__name__)


class MazeDataset(BasePlaceCellDataset):
    """Dataset for 1D arm/maze place cell analysis.

    Overrides the behavior preprocessing, occupancy computation, and
    unit analysis steps to work on a concatenated 1D axis.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trajectory_1d: pd.DataFrame | None = None
        self.trajectory_1d_all: pd.DataFrame | None = None
        self.trajectory_1d_filtered: pd.DataFrame | None = None
        self.pos_range: tuple[float, float] | None = None
        self.edges_1d: np.ndarray | None = None
        self.arm_boundaries: list[float] = []
        self.effective_arm_order: list[str] = []
        self.arm_lengths: dict[str, float] | None = None
        self.segment_bins: list[int] | None = None
        self.graph_polylines: dict[str, list[list[float]]] | None = None
        self.graph_mm_per_pixel: float | None = None

    @property
    def spatial_1d(self) -> SpatialMap1DConfig:
        """Shortcut to 1D spatial map config."""
        return self.cfg.behavior.spatial_map_1d

    @property
    def p_value_threshold(self) -> float:
        """P-value threshold from 1D spatial map config."""
        return self.spatial_1d.p_value_threshold

    def load(self) -> None:
        """Load neural traces, behavior from zone_tracking CSV, and vis assets.

        Unlike ArenaDataset, this does NOT load the raw behavior_position CSV.
        The maze pipeline only needs the zone_tracking CSV (which already
        contains x, y, zone, arm_position) plus behavior timestamps.
        """
        self._load_neural_and_viz()

        dcfg = self.data_cfg
        if dcfg is None or dcfg.bodypart is None:
            raise RuntimeError("bodypart must be set in data config")

        # Behavior: load from zone_tracking CSV (not behavior_position)
        zone_csv = self.zone_tracking_path
        if zone_csv is None or not zone_csv.exists():
            raise FileNotFoundError(
                "zone_tracking CSV not found. Run 'placecell detect-zones' first "
                f"to generate it. Expected: {zone_csv}"
            )

        df = pd.read_csv(zone_csv, header=[0, 1, 2])
        scorer = df.columns[1][0]
        bp = dcfg.bodypart
        zone_col = dcfg.zone_column
        tp_col = dcfg.arm_position_column

        frame_index = df.iloc[:, 0].values
        x = df[(scorer, bp, "x")].values.astype(float)
        y = df[(scorer, bp, "y")].values.astype(float)
        zone = df[(scorer, bp, zone_col)].values
        arm_pos = pd.to_numeric(df[(scorer, bp, tp_col)].values, errors="coerce")

        # Merge with behavior timestamps for unix_time
        timestamps = pd.read_csv(self.behavior_timestamp_path)
        ts_lookup = timestamps.set_index("frame_index")["unix_time"]
        unix_time = ts_lookup.reindex(frame_index).values

        self.trajectory = pd.DataFrame(
            {
                "frame_index": frame_index,
                "x": x,
                "y": y,
                "unix_time": unix_time,
                "speed": 0.0,  # placeholder; maze uses speed_1d instead
                zone_col: zone,
                tp_col: arm_pos,
            }
        )
        logger.info("Loaded trajectory from zone_tracking: %d frames", len(self.trajectory))

    def preprocess_behavior(self) -> None:
        """Serialize to 1D, compute speed, and filter.

        Zone and arm_position columns are already in self.trajectory
        from load(), so no extra CSV loading is needed.
        """
        if self.trajectory is None:
            raise RuntimeError("Behavior data not loaded. Call load() first.")

        dcfg = self.data_cfg
        if dcfg is None or dcfg.arm_order is None:
            raise RuntimeError("arm_order must be set in data config for maze analysis.")

        bcfg = self.cfg.behavior

        # Load behavior graph for physical arm lengths (optional)
        if self.behavior_graph_path is not None and self.behavior_graph_path.exists():
            self.graph_polylines = load_graph_polylines(self.behavior_graph_path)
            self.graph_mm_per_pixel = dcfg.mm_per_pixel or 1.0
            zone_lengths = compute_arm_lengths(self.graph_polylines, self.graph_mm_per_pixel)
            # Keep only the arms we're using
            self.arm_lengths = {t: zone_lengths[t] for t in dcfg.arm_order if t in zone_lengths}
        else:
            self.arm_lengths = None

        # Serialize to 1D
        self.trajectory_1d = serialize_arm_position(
            self.trajectory,
            arm_order=dcfg.arm_order,
            zone_column=dcfg.zone_column,
            arm_position_column=dcfg.arm_position_column,
            arm_lengths=self.arm_lengths,
        )

        # Optionally split by traversal direction (doubles segments)
        if bcfg.split_by_direction:
            self.trajectory_1d, self.effective_arm_order = assign_traversal_direction(
                self.trajectory_1d,
                arm_order=dcfg.arm_order,
                zone_column=dcfg.zone_column,
                arm_position_column=dcfg.arm_position_column,
                arm_lengths=self.arm_lengths,
            )
        else:
            self.effective_arm_order = list(dcfg.arm_order)

        # Optionally filter to complete traversals only (room-to-room)
        # Keep pre-filter copy for visualization
        self.trajectory_1d_all = self.trajectory_1d.copy()
        if bcfg.require_complete_traversal:
            self.trajectory_1d = filter_complete_traversals(
                self.trajectory_1d,
                full_trajectory=self.trajectory,
                arm_order=dcfg.arm_order,
                zone_column=dcfg.zone_column,
            )

        # Compute 1D speed
        self.trajectory_1d = compute_speed_1d(
            self.trajectory_1d,
            window_frames=bcfg.speed_window_frames,
        )

        # Speed filter
        self.trajectory_1d_filtered = filter_arm_by_speed(
            self.trajectory_1d,
            speed_threshold=bcfg.speed_threshold,
        )

        # Compute pos_range and arm_boundaries
        n_segments = len(self.effective_arm_order)
        if self.arm_lengths is not None:
            # Physical scaling: each segment spans its real length
            seg_lengths = []
            for seg_name in self.effective_arm_order:
                # Strip _fwd/_rev suffix to find parent arm
                base = (
                    seg_name.rsplit("_", 1)[0] if seg_name.endswith(("_fwd", "_rev")) else seg_name
                )
                seg_lengths.append(self.arm_lengths.get(base, 1.0))
            cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            self.arm_boundaries = cumulative.tolist()
            self.pos_range = (0.0, cumulative[-1])
        else:
            self.pos_range = (0.0, float(n_segments))
            self.arm_boundaries = [float(i) for i in range(n_segments + 1)]

        logger.info(
            "1D trajectory: %d frames (%d after speed filter), %d segments%s, "
            "pos_range=(%.1f, %.1f)",
            len(self.trajectory_1d),
            len(self.trajectory_1d_filtered),
            n_segments,
            " (direction split)" if bcfg.split_by_direction else "",
            self.pos_range[0],
            self.pos_range[1],
        )

    def match_events(self) -> None:
        """Match neural events to behavior frames, then add 1D position.

        Unlike the parent, does NOT filter by 2D speed — the maze pipeline
        filters by 1D arm speed in analyze_units() instead.
        """
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
            speed_threshold=0.0,  # no 2D speed filter; use speed_1d later
        )

        if self.trajectory_1d is None:
            raise RuntimeError("Call preprocess_behavior() first.")

        # Add pos_1d and speed_1d to event_place by joining on behavior frame
        lookup_cols = ["frame_index", "pos_1d", "arm_index", "speed_1d"]
        if "direction" in self.trajectory_1d.columns:
            lookup_cols.append("direction")
        pos_lookup = self.trajectory_1d[lookup_cols].rename(
            columns={"frame_index": "beh_frame_index"}
        )
        self.event_place = self.event_place.merge(pos_lookup, on="beh_frame_index", how="left")

        # Drop events not in arms (pos_1d will be NaN)
        n_before = len(self.event_place)
        self.event_place = self.event_place.dropna(subset=["pos_1d"]).reset_index(drop=True)
        logger.info(
            "1D event matching: %d/%d events in arms",
            len(self.event_place),
            n_before,
        )

    def compute_occupancy(self) -> None:
        """Compute 1D occupancy from speed-filtered arm trajectory."""
        if self.trajectory_1d_filtered is None:
            raise RuntimeError("Call preprocess_behavior() first.")

        scfg = self.spatial_1d
        n_segments = len(self.effective_arm_order)

        total_length = self.pos_range[1] - self.pos_range[0]
        n_bins = max(n_segments, round(total_length / scfg.bin_width_mm))

        # Compute segment bin boundaries from arm boundaries
        edges_tmp = np.linspace(self.pos_range[0], self.pos_range[1], n_bins + 1)
        self.segment_bins = [int(np.searchsorted(edges_tmp, b)) for b in self.arm_boundaries]

        self.occupancy_time, self.valid_mask, self.edges_1d = compute_occupancy_map_1d(
            trajectory_df=self.trajectory_1d_filtered,
            n_bins=n_bins,
            pos_range=self.pos_range,
            behavior_fps=self.cfg.behavior.behavior_fps,
            spatial_sigma=scfg.spatial_sigma,
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

    def analyze_units(self, progress_bar: Any = None, n_workers: int = 1) -> None:
        """Run 1D spatial analysis for all deconvolved units.

        Parameters
        ----------
        progress_bar:
            Progress bar wrapper, e.g. ``tqdm``.
        n_workers:
            Number of parallel worker processes.  ``1`` (default) runs
            sequentially with no multiprocessing overhead.
        """
        if self.event_place is None:
            raise RuntimeError("Call match_events() first.")
        if self.occupancy_time is None:
            raise RuntimeError("Call compute_occupancy() first.")

        scfg = self.spatial_1d
        bcfg = self.cfg.behavior
        base_seed = scfg.random_seed

        # Use 1D arm speed (same criterion as occupancy), not 2D camera speed
        df_filtered = self.event_place[self.event_place["speed_1d"] >= bcfg.speed_threshold].copy()

        unique_units = sorted(df_filtered["unit_id"].unique())
        unique_units = [uid for uid in unique_units if uid in self.good_unit_ids]
        n_units = len(unique_units)
        logger.info("Analyzing %d units (1D)...", n_units)

        common_kwargs: dict[str, Any] = dict(
            df_filtered=df_filtered,
            trajectory_df=self.trajectory_1d_filtered,
            occupancy_time=self.occupancy_time,
            valid_mask=self.valid_mask,
            edges=self.edges_1d,
            spatial_sigma=scfg.spatial_sigma,
            n_shuffles=scfg.n_shuffles,
            behavior_fps=bcfg.behavior_fps,
            min_occupancy=scfg.min_occupancy,
            min_shift_seconds=scfg.min_shift_seconds,
            si_weight_mode=scfg.si_weight_mode,
            n_split_blocks=scfg.n_split_blocks,
            block_shifts=scfg.block_shifts,
            segment_bins=self.segment_bins,
        )

        results: dict[int, dict] = {}

        if n_workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        compute_unit_analysis_1d,
                        unit_id=uid,
                        random_seed=(base_seed + uid if base_seed is not None else None),
                        **common_kwargs,
                    ): uid
                    for uid in unique_units
                }
                iterator = as_completed(futures)
                if progress_bar is not None:
                    iterator = progress_bar(iterator, total=n_units)
                for future in iterator:
                    uid = futures[future]
                    results[uid] = future.result()
        else:
            if base_seed is not None:
                np.random.seed(base_seed)
            iterator = unique_units
            if progress_bar is not None:
                iterator = progress_bar(iterator)
            for uid in iterator:
                results[uid] = compute_unit_analysis_1d(
                    unit_id=uid,
                    random_seed=base_seed,
                    **common_kwargs,
                )

        self.unit_results = {}
        for uid in unique_units:
            result = results[uid]

            trace_data = None
            trace_times = None
            if self.traces is not None:
                try:
                    trace_data = self.traces.sel(unit_id=int(uid)).values
                    trace_times = np.arange(len(trace_data)) / self.neural_fps
                except (KeyError, IndexError):
                    pass

            self.unit_results[uid] = UnitResult(
                rate_map=result["rate_map"],
                rate_map_raw=result["rate_map_raw"],
                si=result["si"],
                p_val=result["p_val"],
                shuffled_sis=result["shuffled_sis"],
                shuffled_rate_p95=None,
                overall_rate=result["overall_rate"],
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

    def _save_summary_figures(self, figures_dir: "Path") -> list[str]:
        """Maze-specific figures on top of the base set."""
        saved = super()._save_summary_figures(figures_dir)

        try:
            import matplotlib
            import matplotlib.pyplot as _plt
        except ImportError:
            return saved

        from placecell.visualization import (
            plot_graph_overlay,
            plot_occupancy_preview_1d,
            plot_position_and_traces_1d,
            plot_shuffle_test_1d,
        )

        rc = {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "Arial",
        }

        with matplotlib.rc_context(rc):
            if self.trajectory_1d_filtered is not None and self.occupancy_time is not None:
                try:
                    fig = plot_occupancy_preview_1d(
                        self.trajectory_1d_filtered,
                        self.occupancy_time,
                        self.valid_mask,
                        self.edges_1d,
                        trajectory_1d=self.trajectory_1d,
                        trajectory_1d_all=self.trajectory_1d_all,
                        arm_boundaries=self.arm_boundaries,
                        arm_labels=self.effective_arm_order,
                    )
                    fig.savefig(figures_dir / "occupancy.pdf", bbox_inches="tight")
                    _plt.close(fig)
                    saved.append("occupancy.pdf")
                except Exception:
                    logger.warning("Failed to save occupancy.pdf", exc_info=True)

            if self.unit_results and self.edges_1d is not None:
                try:
                    fig = plot_shuffle_test_1d(
                        self.unit_results,
                        self.edges_1d,
                        p_value_threshold=self.p_value_threshold,
                        arm_boundaries=self.arm_boundaries,
                        arm_labels=self.effective_arm_order,
                    )
                    fig.savefig(figures_dir / "population_rate_map.pdf", bbox_inches="tight")
                    _plt.close(fig)
                    saved.append("population_rate_map.pdf")
                except Exception:
                    logger.warning("Failed to save population_rate_map.pdf", exc_info=True)

            place_cell_results = self.place_cells()
            if place_cell_results and self.trajectory_1d is not None:
                try:
                    fig = plot_position_and_traces_1d(
                        self.trajectory_1d,
                        place_cell_results,
                        self.edges_1d,
                        behavior_fps=self.cfg.behavior.behavior_fps,
                        speed_threshold=self.cfg.behavior.speed_threshold,
                        trajectory_1d_filtered=self.trajectory_1d_filtered,
                        arm_boundaries=self.arm_boundaries,
                        arm_labels=self.effective_arm_order,
                    )
                    fig.savefig(figures_dir / "position_traces.pdf", bbox_inches="tight")
                    _plt.close(fig)
                    saved.append("position_traces.pdf")
                except Exception:
                    logger.warning("Failed to save position_traces.pdf", exc_info=True)

            if self.graph_polylines is not None:
                try:
                    fig = plot_graph_overlay(
                        self.graph_polylines,
                        self.graph_mm_per_pixel,
                        arm_order=self.data_cfg.arm_order,
                        video_frame=self.behavior_video_frame,
                    )
                    fig.savefig(figures_dir / "graph_overlay.pdf", bbox_inches="tight")
                    _plt.close(fig)
                    saved.append("graph_overlay.pdf")
                except Exception:
                    logger.warning("Failed to save graph_overlay.pdf", exc_info=True)

        return saved

    def save_bundle(self, path: "str | Path", *, save_figures: bool = True) -> "Path":
        """Save bundle, including 1D trajectory and maze metadata."""

        result = super().save_bundle(path, save_figures=save_figures)

        # 1D trajectories
        if self.trajectory_1d is not None:
            self.trajectory_1d.to_parquet(result / "trajectory_1d.parquet")
        if self.trajectory_1d_all is not None:
            self.trajectory_1d_all.to_parquet(result / "trajectory_1d_all.parquet")
        if self.trajectory_1d_filtered is not None:
            self.trajectory_1d_filtered.to_parquet(result / "trajectory_1d_filtered.parquet")

        # Maze metadata needed for visualization
        maze_meta: dict[str, Any] = {
            "arm_boundaries": self.arm_boundaries,
            "effective_arm_order": self.effective_arm_order,
            "arm_lengths": self.arm_lengths,
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
        1D-specific state (trajectories, arm boundaries, etc.).
        """
        path = Path(path)

        base = cls._load_bundle_data(path)

        # Restore 1D trajectories
        t1d_path = path / "trajectory_1d.parquet"
        if t1d_path.exists():
            base.trajectory_1d = pd.read_parquet(t1d_path)
        t1da_path = path / "trajectory_1d_all.parquet"
        if t1da_path.exists():
            base.trajectory_1d_all = pd.read_parquet(t1da_path)
        t1df_path = path / "trajectory_1d_filtered.parquet"
        if t1df_path.exists():
            base.trajectory_1d_filtered = pd.read_parquet(t1df_path)

        # Restore maze metadata
        meta_path = path / "maze_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            base.arm_boundaries = meta.get("arm_boundaries", [])
            base.effective_arm_order = meta.get("effective_arm_order", [])
            base.arm_lengths = meta.get("arm_lengths")
            base.segment_bins = meta.get("segment_bins")
            pr = meta.get("pos_range")
            base.pos_range = tuple(pr) if pr else None
            base.graph_polylines = meta.get("graph_polylines")
            base.graph_mm_per_pixel = meta.get("graph_mm_per_pixel")

        # edges_1d is stored as x_edges
        base.edges_1d = base.x_edges

        return base
