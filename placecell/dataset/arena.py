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
from placecell.behavior import clip_to_arena, correct_perspective, remove_position_jumps
from placecell.config import BaseSpatialMapConfig
from placecell.dataset.base import BasePlaceCellDataset, UnitResult
from placecell.loaders import load_behavior_data
from placecell.log import init_logger
from placecell.temporal_alignment import (
    build_canonical_table,
    compute_speed_2d,
    derive_event_place_from_canonical,
    filter_canonical_by_speed,
    interpolate_behavior_onto_neural,
)

logger = init_logger(__name__)


class ArenaDataset(BasePlaceCellDataset):
    """Dataset for 2D open-field arena place cell analysis.

    Adds arena-specific functionality: perspective correction scale
    properties, 2D occupancy computation, and 2D spatial analysis.
    """

    @property
    def spatial(self) -> "BaseSpatialMapConfig":
        """Shortcut to 2D spatial map config."""
        return self.cfg.behavior.spatial_map_2d

    @property
    def p_value_threshold(self) -> float:
        """P-value threshold from 2D spatial map config."""
        return self.spatial.p_value_threshold

    def load(self) -> None:
        """Load neural traces, arena behavior data, and visualization assets."""
        self._load_neural_and_viz()

        dcfg = self.data_cfg
        if dcfg is None or dcfg.bodypart is None:
            raise RuntimeError("bodypart must be set in data config")
        self.trajectory = load_behavior_data(
            behavior_position=self.behavior_position_path,
            behavior_timestamp=self.behavior_timestamp_path,
            bodypart=dcfg.bodypart,
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
        """Apply geometric corrections to the behavior trajectory.

        When ``arena_bounds`` is configured:
            jump removal → perspective correction → boundary clipping →
            unit conversion (px → mm).

        When ``arena_bounds`` is **not** configured:
            warnings are logged and the trajectory remains in pixels.

        Speed is computed later at the neural sample rate inside
        :meth:`match_events`. Requires :meth:`load` to have been called.
        """
        if self.trajectory is None:
            raise RuntimeError("Call load() first.")

        bcfg = self.cfg.behavior

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

            self._preprocess_steps: dict[str, pd.DataFrame] = {}
            self._preprocess_steps["Raw"] = self.trajectory[["x", "y"]].copy()

            # 1. Hampel-filter outlier removal on the raw 2D trajectory.
            self.trajectory, n_jumps = remove_position_jumps(
                self.trajectory,
                window_frames=bcfg.hampel_window_frames,
                n_sigmas=bcfg.hampel_n_sigmas,
            )
            logger.info(
                "Hampel jump removal: %d frames interpolated (window=%d, n_sigmas=%.1f)",
                n_jumps,
                bcfg.hampel_window_frames,
                bcfg.hampel_n_sigmas,
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

            for step_name in self._preprocess_steps:
                df = self._preprocess_steps[step_name]
                df["x"] = (df["x"] - x_min) * scale_x
                df["y"] = (df["y"] - y_min) * scale_y
            self._preprocess_steps["Converted"] = self.trajectory[["x", "y"]].copy()
        else:
            logger.warning(
                "No arena_bounds — skipping spatial corrections; "
                "speed and position remain in pixels"
            )

        logger.info("Trajectory ready: %d behavior-rate frames", len(self.trajectory))

    def match_events(self) -> None:
        """Build the canonical neural-rate table and derive analysis views.

        After this call:
            - ``self.canonical`` — one row per neural frame with columns
              ``frame_index, neural_time, x, y, speed, s_unit_*``.
            - ``self.trajectory_filtered`` — speed-filtered view of
              ``canonical`` for occupancy and spatial analysis.
            - ``self.event_place`` — long-format event table derived
              from the speed-filtered view.
        """
        if not self.S_list:
            raise RuntimeError("Call deconvolve() first.")
        if self.trajectory is None:
            raise RuntimeError("Call load() first.")

        bcfg = self.cfg.behavior
        # Use the timestamps already validated during load().
        neural_time = self._neural_time_raw

        n_neural = len(neural_time)
        if any(len(s) != n_neural for s in self.S_list):
            raise RuntimeError(
                "Deconvolved trace length does not match neural-timestamp count "
                f"({len(self.S_list[0])} vs {n_neural})."
            )

        # Interpolate (x, y) onto neural timestamps, then compute speed
        # at the neural sample rate so the canonical table is internally
        # consistent (one clock for everything).
        behavior_at_neural = interpolate_behavior_onto_neural(
            self.trajectory,
            neural_time,
            columns=["x", "y"],
        )
        behavior_at_neural = compute_speed_2d(
            behavior_at_neural,
            window_seconds=bcfg.speed_window_seconds,
            sample_rate_hz=self.neural_fps,
            time_column="neural_time",
        )
        traces = {int(uid): self.S_list[i] for i, uid in enumerate(self.good_unit_ids)}
        self.canonical = build_canonical_table(behavior_at_neural, traces)
        logger.info(
            "Canonical neural-rate table: %d frames, %d units",
            len(self.canonical),
            len(self.good_unit_ids),
        )

        self._refresh_views_from_canonical()

    def _refresh_views_from_canonical(self) -> None:
        """Rebuild ``event_index``, ``trajectory_filtered``, ``event_place`` from ``canonical``.

        Shared by :meth:`match_events` and :meth:`apply_time_window` so the two
        paths stay in lockstep.
        """
        bcfg = self.cfg.behavior

        # Long-format event table over the *unfiltered* canonical, used by
        # the notebook browser (which shows all events including low-speed).
        self.event_index = derive_event_place_from_canonical(
            filter_canonical_by_speed(
                self.canonical,
                speed_column="speed",
                speed_threshold=0.0,
                drop_below_threshold=False,
            ),
            position_columns=("x", "y"),
            speed_column="speed",
        )

        self.trajectory_filtered = filter_canonical_by_speed(
            self.canonical,
            speed_column="speed",
            speed_threshold=bcfg.speed_threshold,
        )
        self.event_place = derive_event_place_from_canonical(
            self.trajectory_filtered,
            position_columns=("x", "y"),
            speed_column="speed",
        )
        logger.info(
            "Speed filter (%.1f mm/s): %d/%d frames; %d events across %d units",
            bcfg.speed_threshold,
            len(self.trajectory_filtered),
            len(self.canonical),
            len(self.event_place),
            self.event_place["unit_id"].nunique() if not self.event_place.empty else 0,
        )

    def apply_time_window(self, start_s: float, end_s: float) -> None:
        """Restrict the canonical table to ``[start_s, end_s)`` relative to the first neural frame.

        Call after :meth:`match_events`. The first call snapshots the full
        canonical table; subsequent calls always re-slice from that snapshot,
        so iterating windows is idempotent. Re-runs the view derivations but
        not deconvolution — the caller should rerun :meth:`compute_occupancy`
        and :meth:`analyze_units` to complete the analysis for the new window.
        """
        if self.canonical is None:
            raise RuntimeError("Call match_events() first.")

        if not hasattr(self, "_canonical_full") or self._canonical_full is None:
            self._canonical_full = self.canonical.copy()

        full = self._canonical_full
        t0 = float(full["neural_time"].iloc[0])
        rel = full["neural_time"].to_numpy() - t0
        mask = (rel >= start_s) & (rel < end_s)
        self.canonical = full.loc[mask].reset_index(drop=True)
        logger.info(
            "Time window [%.1f, %.1f)s: %d/%d frames",
            start_s,
            end_s,
            len(self.canonical),
            len(full),
        )
        self._refresh_views_from_canonical()

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
            plot_behavior_preview,
            plot_coverage,
            plot_occupancy_preview,
            plot_position_and_traces_2d,
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
                    scfg = self.spatial
                    fig = plot_occupancy_preview(
                        self.trajectory_filtered,
                        self.occupancy_time,
                        self.valid_mask,
                        self.x_edges,
                        self.y_edges,
                        behavior_fps=self.neural_fps,
                        n_split_blocks=scfg.n_split_blocks,
                        block_shift=scfg.block_shift,
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

            # Speed distribution (behavior preview with speed histogram)
            if self.canonical is not None and self.trajectory_filtered is not None:
                try:
                    fig = plot_behavior_preview(
                        self.canonical,
                        self.trajectory_filtered,
                        self.cfg.behavior.speed_threshold,
                        speed_unit="mm/s" if self.mm_per_px else "px/s",
                    )
                    fig.savefig(figures_dir / "behavior_preview.pdf", bbox_inches="tight")
                    _plt.close(fig)
                    saved.append("behavior_preview.pdf")
                except Exception:
                    logger.warning("Failed to save behavior_preview.pdf", exc_info=True)

            place_cell_results = self.place_cells()

            # Speed + place cell traces
            if place_cell_results and self.canonical is not None:
                try:
                    fig = plot_position_and_traces_2d(
                        self.canonical,
                        place_cell_results,
                        behavior_fps=self.neural_fps,
                        speed_threshold=self.cfg.behavior.speed_threshold,
                        trajectory_filtered=self.trajectory_filtered,
                        speed_unit="mm/s" if self.mm_per_px else "px/s",
                    )
                    fig.savefig(figures_dir / "speed_traces.pdf", bbox_inches="tight")
                    _plt.close(fig)
                    saved.append("speed_traces.pdf")
                except Exception:
                    logger.warning("Failed to save speed_traces.pdf", exc_info=True)

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
