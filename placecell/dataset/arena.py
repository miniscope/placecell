"""Dataset class for 2D open-field arena place cell analysis."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from placecell.analysis.spatial_2d import (
    compute_coverage_curve,
    compute_coverage_map,
    compute_occupancy_map,
    compute_unit_analysis,
)
from placecell.behavior import (
    build_event_place_dataframe,
    clip_to_arena,
    correct_perspective,
    filter_by_speed,
    recompute_speed,
    remove_position_jumps,
)
from placecell.config import SpatialMapConfig
from placecell.dataset.base import BasePlaceCellDataset, UnitResult
from placecell.loaders import load_behavior_data
from placecell.log import init_logger

logger = init_logger(__name__)


class ArenaDataset(BasePlaceCellDataset):
    """Dataset for 2D open-field arena place cell analysis.

    Adds arena-specific functionality: perspective correction scale
    properties, 2D occupancy computation, and 2D spatial analysis.
    """

    @property
    def spatial(self) -> "SpatialMapConfig":
        """Shortcut to 2D spatial map config."""
        return self.cfg.behavior.spatial_map_2d

    @property
    def p_value_threshold(self) -> float:
        """P-value threshold from 2D spatial map config."""
        return self.spatial.p_value_threshold

    def load(self) -> None:
        """Load neural traces, arena behavior data, and visualization assets."""
        self._load_neural_and_viz()

        bcfg = self.cfg.behavior
        dcfg = self.data_cfg
        if dcfg is None or dcfg.bodypart is None:
            raise RuntimeError("bodypart must be set in data config")
        self.trajectory, _ = load_behavior_data(
            behavior_position=self.behavior_position_path,
            behavior_timestamp=self.behavior_timestamp_path,
            bodypart=dcfg.bodypart,
            speed_window_frames=bcfg.speed_window_frames,
            speed_threshold=0.0,
            x_col=dcfg.x_col,
            y_col=dcfg.y_col,
        )
        logger.info("Loaded trajectory: %d frames", len(self.trajectory))

    @property
    def mm_per_px(self) -> float | None:
        """Averaged mm-per-pixel scale, or None if arena is not calibrated."""
        scales = self.mm_per_px_xy
        if scales is None:
            return None
        return (scales[0] + scales[1]) / 2.0

    @property
    def mm_per_px_xy(self) -> tuple[float, float] | None:
        """Per-axis (scale_x, scale_y) mm-per-pixel, or None if not calibrated."""
        if self.data_cfg is None:
            return None
        bounds = self.data_cfg.arena_bounds
        size_mm = self.data_cfg.arena_size_mm
        if bounds is None or size_mm is None:
            return None
        x_min, x_max, y_min, y_max = bounds
        scale_x = size_mm[0] / (x_max - x_min)
        scale_y = size_mm[1] / (y_max - y_min)
        return (scale_x, scale_y)

    def preprocess_behavior(self) -> None:
        """Apply data-integrity corrections, convert units, and speed-filter.

        When ``arena_bounds`` is configured:
            jump removal → perspective correction → boundary clipping
            → recompute speed (mm/s) → speed filter.

        When ``arena_bounds`` is **not** configured:
            warnings are logged and speed filtering is applied in px/s.

        Requires ``load()`` to have been called first.
        """
        if self.trajectory is None:
            raise RuntimeError("Call load() first.")

        bcfg = self.cfg.behavior

        # Preserve the raw trajectory before any corrections
        self.trajectory_raw = self.trajectory.copy()

        dcfg = self.data_cfg
        has_arena = dcfg is not None and dcfg.arena_bounds is not None

        if has_arena:
            missing = [
                name
                for name, val in [
                    ("arena_size_mm", dcfg.arena_size_mm),
                    ("camera_height_mm", dcfg.camera_height_mm),
                    ("tracking_height_mm", dcfg.tracking_height_mm),
                ]
                if val is None
            ]
            if missing:
                raise ValueError(
                    f"arena_bounds is set but missing required fields: {', '.join(missing)}"
                )

            scale_x, scale_y = self.mm_per_px_xy

            # Store intermediate snapshots for visualization
            self._preprocess_steps: dict[str, pd.DataFrame] = {}
            self._preprocess_steps["Raw"] = self.trajectory[["x", "y"]].copy()

            # 1. Jump removal (threshold in mm → convert to px using averaged scale)
            scale_avg = (scale_x + scale_y) / 2.0
            jump_px = bcfg.jump_threshold_mm / scale_avg
            self.trajectory, n_jumps = remove_position_jumps(self.trajectory, threshold_px=jump_px)
            logger.info(
                "Jump removal: %d frames interpolated (threshold %.0f mm = %.1f px)",
                n_jumps,
                bcfg.jump_threshold_mm,
                jump_px,
            )
            self._preprocess_steps["Jump removal"] = self.trajectory[["x", "y"]].copy()

            # 2. Perspective correction
            self.trajectory = correct_perspective(
                self.trajectory,
                arena_bounds=dcfg.arena_bounds,
                camera_height_mm=dcfg.camera_height_mm,
                tracking_height_mm=dcfg.tracking_height_mm,
            )
            factor = (dcfg.camera_height_mm - dcfg.tracking_height_mm) / dcfg.camera_height_mm
            logger.info(
                "Perspective correction: factor=%.3f (H=%.0f mm, h=%.0f mm)",
                factor,
                dcfg.camera_height_mm,
                dcfg.tracking_height_mm,
            )
            self._preprocess_steps["Perspective"] = self.trajectory[["x", "y"]].copy()

            # 3. Boundary clipping
            self.trajectory = clip_to_arena(self.trajectory, arena_bounds=dcfg.arena_bounds)
            logger.info("Boundary clipping to arena_bounds=%s", dcfg.arena_bounds)
            self._preprocess_steps["Clipped"] = self.trajectory[["x", "y"]].copy()

            # 4. Convert positions from pixels to mm (per-axis)
            x_min = dcfg.arena_bounds[0]
            y_min = dcfg.arena_bounds[2]
            self.trajectory["x"] = (self.trajectory["x"] - x_min) * scale_x
            self.trajectory["y"] = (self.trajectory["y"] - y_min) * scale_y
            logger.info(
                "Converted to mm (scale_x=%.4f, scale_y=%.4f mm/px)",
                scale_x,
                scale_y,
            )

            # Convert all preprocess snapshots to mm for consistent visualization
            for step_name in self._preprocess_steps:
                df = self._preprocess_steps[step_name]
                df["x"] = (df["x"] - x_min) * scale_x
                df["y"] = (df["y"] - y_min) * scale_y
            self._preprocess_steps["Converted"] = self.trajectory[["x", "y"]].copy()

            # 5. Recompute speed on mm-space positions (natively mm/s)
            self.trajectory = recompute_speed(
                self.trajectory, window_frames=bcfg.speed_window_frames
            )
            logger.info("Speed recomputed after coordinate correction (mm/s)")
            speed_unit = "mm/s"
        else:
            logger.warning(
                "No arena_bounds — skipping spatial corrections; "
                "speed and position remain in pixels"
            )
            speed_unit = "px/s"

        # Speed filter (always applied)
        self.trajectory_filtered = filter_by_speed(self.trajectory, bcfg.speed_threshold)
        logger.info(
            "Trajectory: %d frames (%d after speed filter at %.1f %s)",
            len(self.trajectory),
            len(self.trajectory_filtered),
            bcfg.speed_threshold,
            speed_unit,
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
            behavior_fps=self.data_cfg.behavior_fps,
            speed_threshold=bcfg.speed_threshold,
        )
        logger.info(
            "Matched %d events (%d units)",
            len(self.event_place),
            self.event_place["unit_id"].nunique(),
        )

    def compute_occupancy(self) -> None:
        """Compute 2D occupancy map from speed-filtered trajectory."""
        if self.trajectory_filtered is None:
            raise RuntimeError("Call preprocess_behavior() first.")

        scfg = self.spatial

        self.occupancy_time, self.valid_mask, self.x_edges, self.y_edges = compute_occupancy_map(
            trajectory_df=self.trajectory_filtered,
            bins=scfg.bins,
            behavior_fps=self.data_cfg.behavior_fps,
            spatial_sigma=scfg.spatial_sigma,
            min_occupancy=scfg.min_occupancy,
        )
        logger.info(
            "Occupancy map: %s, %d/%d valid bins",
            self.occupancy_time.shape,
            self.valid_mask.sum(),
            self.valid_mask.size,
        )

    def analyze_units(self, progress_bar: Any = None, n_workers: int = 1) -> None:
        """Run 2D spatial analysis for all deconvolved units with events.

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

        scfg = self.spatial
        bcfg = self.cfg.behavior
        base_seed = scfg.random_seed

        df_filtered = self.event_place[self.event_place["speed"] >= bcfg.speed_threshold].copy()

        unique_units = sorted(df_filtered["unit_id"].unique())
        unique_units = [uid for uid in unique_units if uid in self.good_unit_ids]
        n_units = len(unique_units)
        logger.info("Analyzing %d units...", n_units)

        common_kwargs: dict[str, Any] = dict(
            df_filtered=df_filtered,
            trajectory_df=self.trajectory_filtered,
            occupancy_time=self.occupancy_time,
            valid_mask=self.valid_mask,
            x_edges=self.x_edges,
            y_edges=self.y_edges,
            scfg=scfg,
            behavior_fps=self.data_cfg.behavior_fps,
        )

        results: dict[int, dict] = {}

        if n_workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        compute_unit_analysis,
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
                results[uid] = compute_unit_analysis(
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
                shuffled_sis=result["shuffled_sis"],
                shuffled_rate_p95=result["shuffled_rate_p95"],
                p_val=result["p_val"],
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

        logger.info("Done. %d units analyzed.", len(self.unit_results))

    def coverage(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute place field coverage map and curve.

        Returns (coverage_map, n_cells_array, coverage_fraction_array).
        """
        scfg = self.cfg.behavior.spatial_map_2d
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

    def _save_summary_figures(self, figures_dir: Path) -> list[str]:
        """Arena-specific figures on top of the base set."""
        saved = super()._save_summary_figures(figures_dir)

        try:
            import matplotlib
            import matplotlib.pyplot as _plt
        except ImportError:
            return saved

        from placecell.visualization import (
            plot_arena_calibration,
            plot_coverage,
            plot_occupancy_preview,
            plot_preprocess_steps,
        )

        rc = {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "Arial",
        }

        with matplotlib.rc_context(rc):
            if self.trajectory_filtered is not None and self.occupancy_time is not None:
                try:
                    fig = plot_occupancy_preview(
                        self.trajectory_filtered,
                        self.occupancy_time,
                        self.valid_mask,
                        self.x_edges,
                        self.y_edges,
                    )
                    fig.savefig(figures_dir / "occupancy.pdf", bbox_inches="tight")
                    _plt.close(fig)
                    saved.append("occupancy.pdf")
                except Exception:
                    logger.warning("Failed to save occupancy.pdf", exc_info=True)

            dcfg = self.data_cfg
            if (
                self.trajectory_raw is not None
                and dcfg is not None
                and dcfg.arena_bounds is not None
            ):
                try:
                    fig = plot_arena_calibration(
                        self.trajectory_raw,
                        dcfg.arena_bounds,
                        arena_size_mm=dcfg.arena_size_mm,
                        mm_per_px=self.mm_per_px,
                        video_frame=self.behavior_video_frame,
                    )
                    fig.savefig(figures_dir / "arena_calibration.pdf", bbox_inches="tight")
                    _plt.close(fig)
                    saved.append("arena_calibration.pdf")
                except Exception:
                    logger.warning("Failed to save arena_calibration.pdf", exc_info=True)

            if (
                hasattr(self, "_preprocess_steps")
                and self._preprocess_steps
                and dcfg is not None
                and dcfg.arena_size_mm is not None
            ):
                try:
                    fig = plot_preprocess_steps(self._preprocess_steps, dcfg.arena_size_mm)
                    fig.savefig(figures_dir / "preprocess_steps.pdf", bbox_inches="tight")
                    _plt.close(fig)
                    saved.append("preprocess_steps.pdf")
                except Exception:
                    logger.warning("Failed to save preprocess_steps.pdf", exc_info=True)

            place_cell_results = self.place_cells()
            if place_cell_results:
                try:
                    coverage_map, _, _ = self.coverage()
                    fig = plot_coverage(
                        coverage_map,
                        self.x_edges,
                        self.y_edges,
                        self.valid_mask,
                        len(place_cell_results),
                    )
                    fig.savefig(figures_dir / "coverage.pdf", bbox_inches="tight")
                    _plt.close(fig)
                    saved.append("coverage.pdf")
                except Exception:
                    logger.warning("Failed to save coverage.pdf", exc_info=True)

        return saved
