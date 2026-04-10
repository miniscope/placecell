"""Dataset class for maze/arm 1D place cell analysis."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from placecell.analysis.spatial_1d import (
    compute_occupancy_map_1d,
    compute_unit_analysis_1d,
)
from placecell.behavior import (
    build_canonical_table,
    derive_event_place_from_canonical,
    filter_canonical_by_speed,
)
from placecell.config import SpatialMap1DConfig, ZoneDetectionConfig
from placecell.dataset.base import BasePlaceCellDataset, UnitResult
from placecell.log import init_logger
from placecell.maze_helper import (
    assign_traversal_direction,
    compute_arm_lengths,
    compute_speed_1d,
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

    def _run_zone_detection(self, *, force: bool = False) -> None:
        """Run :func:`detect_zones_from_csv` for this dataset.

        Called automatically by :meth:`load` when the cached
        ``zone_tracking`` CSV is missing or when ``force_redetect=True``.
        Assumes the caller has already validated ``data_cfg`` and
        ``bodypart``; only checks the conditions unique to detection.
        """
        from placecell.zone_detection import detect_zones_from_csv

        dcfg = self.data_cfg
        if self.behavior_graph_path is None or not self.behavior_graph_path.exists():
            raise RuntimeError(
                "behavior_graph is required to run detect-zones automatically. "
                "Set it in the data config or run 'placecell detect-zones' manually."
            )

        zone_csv = self.zone_tracking_path
        zd = dcfg.zone_detection or ZoneDetectionConfig()
        action = "Re-running" if force else "zone_tracking CSV not found; running"
        logger.info(
            "%s detect-zones (%s -> %s)",
            action,
            self.behavior_position_path.name,
            zone_csv.name,
        )
        zone_csv.parent.mkdir(parents=True, exist_ok=True)
        detect_zones_from_csv(
            input_csv=self.behavior_position_path,
            output_csv=zone_csv,
            zone_config_path=self.behavior_graph_path,
            behavior_timestamp_csv=self.behavior_timestamp_path,
            neural_timestamp_csv=self.neural_timestamp_path,
            bodypart=dcfg.bodypart,
            arm_max_distance=zd.arm_max_distance,
            min_confidence=zd.min_confidence,
            min_confidence_forbidden=zd.min_confidence_forbidden,
            min_seconds_same=zd.min_seconds_same,
            min_seconds_forbidden=zd.min_seconds_forbidden,
            room_decay_power=zd.room_decay_power,
            arm_decay_power=zd.arm_decay_power,
            soft_boundary=zd.soft_boundary,
            hampel_window_frames=zd.hampel_window_frames,
            hampel_n_sigmas=zd.hampel_n_sigmas,
            zone_connections=dcfg.zone_connections,
        )

    def load(self, *, force_redetect: bool = False) -> None:
        """Load neural traces, behavior from zone_tracking CSV, and vis assets.

        ``MazeDataset`` reads the zone-detected ``zone_tracking`` CSV directly.
        If the CSV is missing, :meth:`_run_zone_detection` is invoked first
        to project the raw ``behavior_position`` CSV onto the maze graph.

        Parameters
        ----------
        force_redetect:
            If ``True``, re-run :func:`detect_zones_from_csv` even when
            ``zone_tracking_path`` already exists. Useful when zone-detection
            parameters have changed and the cached output is stale.
        """
        self._load_neural_and_viz()

        dcfg = self.data_cfg
        if dcfg is None or dcfg.bodypart is None:
            raise RuntimeError("bodypart must be set in data config")

        # Auto-run detect-zones when the cached output is missing or forced.
        zone_csv = self.zone_tracking_path
        if zone_csv is None:
            raise RuntimeError(
                "zone_tracking_path is unset; this should not happen for a "
                "MazeDataset constructed via from_yaml()."
            )
        if force_redetect or not zone_csv.exists():
            self._run_zone_detection(force=force_redetect)

        df = pd.read_csv(zone_csv, header=[0, 1, 2])
        scorer = df.columns[1][0]
        bp = dcfg.bodypart
        zone_col = dcfg.zone_column
        tp_col = dcfg.arm_position_column

        # zone_tracking is now indexed by frame_index (one row per neural
        # frame) with x, y, zone, arm_position, neural_time columns derived
        # from the Hampel-filtered raw trajectory interpolated onto neural
        # timestamps inside detect_zones_from_csv.
        frame_index = df.iloc[:, 0].values.astype(np.int64)
        x = df[(scorer, bp, "x")].values.astype(float)
        y = df[(scorer, bp, "y")].values.astype(float)
        zone = df[(scorer, bp, zone_col)].values
        arm_pos_raw = df[(scorer, bp, tp_col)]
        arm_pos = pd.to_numeric(arm_pos_raw.values, errors="coerce")
        n_coerced = int(arm_pos_raw.notna().sum() - np.isfinite(arm_pos).sum())
        if n_coerced:
            logger.warning(
                "%d non-numeric '%s' value(s) coerced to NaN in zone_tracking CSV.",
                n_coerced,
                tp_col,
            )
        neural_time = df[(scorer, bp, "neural_time")].values.astype(float)

        self.trajectory = pd.DataFrame(
            {
                "frame_index": frame_index,
                "x": x,
                "y": y,
                "unix_time": neural_time,
                "speed": 0.0,  # placeholder; maze uses speed_1d instead
                zone_col: zone,
                tp_col: arm_pos,
            }
        )
        logger.info(
            "Loaded trajectory from zone_tracking: %d neural frames",
            len(self.trajectory),
        )

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
        scfg = self.spatial_1d

        if self.behavior_graph_path is not None and self.behavior_graph_path.exists():
            self.graph_polylines = load_graph_polylines(self.behavior_graph_path)
            self.graph_mm_per_pixel = dcfg.mm_per_pixel or 1.0
            zone_lengths = compute_arm_lengths(self.graph_polylines, self.graph_mm_per_pixel)
            # Keep only the arms we're using
            self.arm_lengths = {t: zone_lengths[t] for t in dcfg.arm_order if t in zone_lengths}
        else:
            self.arm_lengths = None

        self.trajectory_1d = serialize_arm_position(
            self.trajectory,
            arm_order=dcfg.arm_order,
            zone_column=dcfg.zone_column,
            arm_position_column=dcfg.arm_position_column,
            arm_lengths=self.arm_lengths,
        )

        # Optionally split by traversal direction (doubles segments)
        if scfg.split_by_direction:
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
        if scfg.require_complete_traversal:
            self.trajectory_1d = filter_complete_traversals(
                self.trajectory_1d,
                full_trajectory=self.trajectory,
                arm_order=dcfg.arm_order,
                zone_column=dcfg.zone_column,
            )

        # Compute 1D speed at the neural sample rate (the trajectory has
        # already been interpolated onto neural timestamps inside
        # detect_zones_from_csv).
        self.trajectory_1d = compute_speed_1d(
            self.trajectory_1d,
            window_seconds=bcfg.speed_window_seconds,
            sample_rate_hz=self.neural_fps,
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
            "1D trajectory: %d frames, %d segments%s, pos_range=(%.1f, %.1f)",
            len(self.trajectory_1d),
            n_segments,
            " (direction split)" if scfg.split_by_direction else "",
            self.pos_range[0],
            self.pos_range[1],
        )

    def match_events(self) -> None:
        """Build the canonical neural-rate table for the maze pipeline.

        After this call:
            - ``self.canonical`` holds one row per neural frame with
              columns ``frame_index, neural_time, x, y, pos_1d,
              arm_index, [direction], speed_1d, s_unit_*``.
            - ``self.trajectory_1d_filtered`` is the speed-filtered
              canonical view restricted to arm frames, with
              ``frame_index`` aliased to ``frame_index``.
            - ``self.event_place`` is the long-format event table
              derived from the same speed-filtered view.
        """
        if not self.S_list:
            raise RuntimeError("Call deconvolve() first.")
        if self.trajectory_1d is None:
            raise RuntimeError("Call preprocess_behavior() first.")

        bcfg = self.cfg.behavior

        # The 1D trajectory is already at neural rate (one row per
        # arm-frame, indexed by frame_index). Build a behavior-at-neural
        # table from it; non-arm frames are filled with NaN.
        n_neural = len(self.S_list[0])
        beh_cols = ["x", "y", "pos_1d", "arm_index", "speed_1d"]
        if "direction" in self.trajectory_1d.columns:
            beh_cols.append("direction")
        behavior_at_neural = pd.DataFrame(
            {
                "frame_index": np.arange(n_neural, dtype=np.int64),
                "neural_time": np.full(n_neural, np.nan, dtype=float),
            }
        )
        for col in beh_cols:
            behavior_at_neural[col] = np.nan if col != "direction" else None
        # Fill in the arm-frame rows.
        idx = self.trajectory_1d["frame_index"].to_numpy().astype(np.int64)
        in_range = (idx >= 0) & (idx < n_neural)
        idx = idx[in_range]
        traj_arm = self.trajectory_1d.loc[in_range].reset_index(drop=True)
        behavior_at_neural.loc[idx, "neural_time"] = traj_arm["unix_time"].to_numpy()
        for col in beh_cols:
            if col in traj_arm.columns:
                behavior_at_neural.loc[idx, col] = traj_arm[col].to_numpy()

        traces = {int(uid): self.S_list[i] for i, uid in enumerate(self.good_unit_ids)}
        self.canonical = build_canonical_table(
            behavior_at_neural,
            traces,
            drop_uncovered=False,
        )
        logger.info(
            "Canonical neural-rate table: %d frames (%d arm), %d units",
            len(self.canonical),
            int(self.canonical["pos_1d"].notna().sum()),
            len(self.good_unit_ids),
        )

        # Long-format event table over the *unfiltered* canonical, used by
        # the notebook browser (which shows all events including low-speed
        # and non-arm frames).
        self.event_index = derive_event_place_from_canonical(
            filter_canonical_by_speed(
                self.canonical,
                speed_column="speed_1d",
                speed_threshold=0.0,
                drop_below_threshold=False,
            ),
            position_columns=("x", "y"),
            speed_column="speed_1d",
        )

        # Speed-filtered view, restricted to arm frames.
        arm_canonical = self.canonical.dropna(subset=["pos_1d"]).reset_index(drop=True)
        self.trajectory_1d_filtered = filter_canonical_by_speed(
            arm_canonical,
            speed_column="speed_1d",
            speed_threshold=bcfg.speed_threshold,
        )

        extras = ("pos_1d", "arm_index")
        if "direction" in self.canonical.columns:
            extras = (*extras, "direction")
        self.event_place = derive_event_place_from_canonical(
            self.trajectory_1d_filtered,
            position_columns=("x", "y"),
            speed_column="speed_1d",
            extra_columns=extras,
        )
        logger.info(
            "1D speed filter (%.1f mm/s): %d/%d arm frames; %d events",
            bcfg.speed_threshold,
            len(self.trajectory_1d_filtered),
            len(arm_canonical),
            len(self.event_place),
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
            behavior_fps=self.data_cfg.behavior_fps,
            spatial_sigma=scfg.spatial_sigma,
            min_occupancy=scfg.min_occupancy,
            segment_bins=self.segment_bins,
        )

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
            scfg=scfg,
            behavior_fps=self.data_cfg.behavior_fps,
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
                event_count_rate=result["event_count_rate"],
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

        from placecell.analysis.pvo_1d import compute_dataset_arm_pvo, plot_arm_pvo_grid
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

                try:
                    pvo_results = compute_dataset_arm_pvo(self, use_place_cells=True)
                    fig = plot_arm_pvo_grid(
                        pvo_results,
                        self.effective_arm_order,
                    )
                    fig.savefig(figures_dir / "global_pvo_matrix.pdf", bbox_inches="tight")
                    _plt.close(fig)
                    saved.append("global_pvo_matrix.pdf")
                except Exception:
                    logger.warning("Failed to save global_pvo_matrix.pdf", exc_info=True)

            place_cell_results = self.place_cells()
            if place_cell_results and self.trajectory_1d is not None:
                try:
                    fig = plot_position_and_traces_1d(
                        self.trajectory_1d,
                        place_cell_results,
                        self.edges_1d,
                        behavior_fps=self.data_cfg.behavior_fps,
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
