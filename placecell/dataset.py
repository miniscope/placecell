"""Central dataset class for place cell analysis."""

import abc
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from placecell.analysis import (
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
from placecell.config import AnalysisConfig, DataPathsConfig, SpatialMap2DConfig
from placecell.io import load_behavior_data, load_visualization_data
from placecell.logging import init_logger
from placecell.neural import build_event_index_dataframe, load_calcium_traces, run_deconvolution

logger = init_logger(__name__)

_BUNDLE_VERSION = 1


def unique_bundle_path(bundle_dir: str | Path, stem: str) -> Path:
    """Return a bundle path, appending ``_1``, ``_2``, ... if it already exists.

    Parameters
    ----------
    bundle_dir:
        Directory where bundles are stored.
    stem:
        Base name for the bundle (without extension).

    Returns
    -------
    Path
        A path like ``bundle_dir/stem.pcellbundle`` (or ``stem_1``, ``stem_2``, ...).
    """
    bundle_dir = Path(bundle_dir)
    candidate = bundle_dir / f"{stem}.pcellbundle"
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate = bundle_dir / f"{stem}_{i}.pcellbundle"
        if not candidate.exists():
            return candidate
        i += 1


@dataclass
class UnitResult:
    """Analysis results for a single unit.

    Parameters
    ----------
    rate_map:
        Smoothed rate map (e.g. 2D array for arena dataset, or 1D array for maze dataset).
    rate_map_raw:
        Raw (unsmoothed) rate map.
    si:
        Spatial information (bits/spike).
    p_val:
        P-value from spatial information significance test.
    shuffled_sis:
        Spatial information values from shuffled data (for significance test).
    shuffled_rate_p95:
        95th percentile of shuffled rate maps (for place field thresholding).
    stability_corr:
        Correlation between rate maps from first vs. second half of session.
    stability_z:
        Fisher z-score corresponding to stability_corr.
    stability_p_val:
        P-value from stability significance test.
    shuffled_stability:
        Stability correlations from shuffled data (for significance test).
    rate_map_first:
        Rate map for first half of session.
    rate_map_second:
        Rate map for second half of session.
    vis_data_above:
        Subset of unit_data where event amplitude exceeds the threshold
        (used for plotting event dots on rate maps).
    unit_data:
        Speed-filtered deconvolved events for this unit (subset of event_place).
    overall_rate:
        Overall firing rate (total events / total time), i.e. lambda.
    trace_data:
        Raw calcium trace for this unit (None if traces unavailable).
    trace_times:
        Time axis in seconds corresponding to trace_data.

    """

    rate_map: np.ndarray
    rate_map_raw: np.ndarray
    si: float
    p_val: float
    shuffled_sis: np.ndarray
    shuffled_rate_p95: np.ndarray
    stability_corr: float
    stability_z: float
    stability_p_val: float
    shuffled_stability: np.ndarray
    rate_map_first: np.ndarray
    rate_map_second: np.ndarray
    vis_data_above: pd.DataFrame
    unit_data: pd.DataFrame
    overall_rate: float
    trace_data: np.ndarray | None
    trace_times: np.ndarray | None


class BasePlaceCellDataset(abc.ABC):
    """Base class for place cell analysis datasets.

    Shared pipeline (each step populates attributes for the next)::

        ds = BasePlaceCellDataset.from_yaml(config_path, data_path)
        ds.load()                            # traces, trajectory, footprints
        ds.preprocess_behavior()             # corrections + speed filter
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
    """

    def __init__(
        self,
        cfg: AnalysisConfig,
        *,
        neural_path: Path | None = None,
        neural_timestamp_path: Path | None = None,
        behavior_position_path: Path | None = None,
        behavior_timestamp_path: Path | None = None,
        behavior_video_path: Path | None = None,
        behavior_graph_path: Path | None = None,
        data_cfg: DataPathsConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.neural_path = neural_path
        self.neural_timestamp_path = neural_timestamp_path
        self.behavior_position_path = behavior_position_path
        self.behavior_timestamp_path = behavior_timestamp_path
        self.behavior_video_path = behavior_video_path
        self.behavior_graph_path = behavior_graph_path
        self.data_cfg = data_cfg

        # Neural data
        self.traces: xr.DataArray | None = None
        self.good_unit_ids: list[int] = []
        self.S_list: list[np.ndarray] = []

        # Event data
        self.event_index: pd.DataFrame | None = None
        self.event_place: pd.DataFrame | None = None

        # Behavior data
        self.trajectory_raw: pd.DataFrame | None = None
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
        self.behavior_video_frame: np.ndarray | None = None

        # Results
        self.unit_results: dict[int, UnitResult] = {}

    @classmethod
    def from_yaml(cls, config: str | Path, data_path: str | Path) -> "BasePlaceCellDataset":
        """Create dataset from analysis config and data paths file.

        Parameters
        ----------
        config:
            Analysis config — either a file path or a config id
            (resolved via ``AnalysisConfig.from_id``).
        data_path:
            Path to the per-session data paths YAML file.
        """
        config_p = Path(config)
        if config_p.exists():
            cfg = AnalysisConfig.from_yaml(config_p)
        else:
            cfg = AnalysisConfig.from_id(str(config))

        data_path = Path(data_path)
        data_dir = data_path.parent
        data_cfg = DataPathsConfig.from_yaml(data_path)
        cfg = cfg.with_data_overrides(data_cfg)

        # Auto-select subclass based on behavior type
        klass = cls
        if cfg.behavior and cfg.behavior.type == "maze":
            from placecell.maze_dataset import MazeDataset

            klass = MazeDataset
        elif cls is BasePlaceCellDataset:
            klass = ArenaDataset

        return klass(
            cfg=cfg,
            neural_path=data_dir / data_cfg.neural_path,
            neural_timestamp_path=data_dir / data_cfg.neural_timestamp,
            behavior_position_path=data_dir / data_cfg.behavior_position,
            behavior_timestamp_path=data_dir / data_cfg.behavior_timestamp,
            behavior_video_path=(
                data_dir / data_cfg.behavior_video if data_cfg.behavior_video else None
            ),
            behavior_graph_path=(
                data_dir / data_cfg.behavior_graph if data_cfg.behavior_graph else None
            ),
            data_cfg=data_cfg,
        )

    @property
    @abc.abstractmethod
    def p_value_threshold(self) -> float:
        """P-value threshold from the appropriate spatial config."""
        ...

    @property
    def _spatial_cfg(self) -> "SpatialMap2DConfig | SpatialMap1DConfig | None":
        """Return whichever spatial map config is set."""
        bcfg = self.cfg.behavior
        return bcfg.spatial_map_2d or bcfg.spatial_map_1d

    @property
    def _shuffle_n(self) -> int | None:
        """Number of shuffles from the spatial config."""
        sc = self._spatial_cfg
        return sc.n_shuffles if sc else None

    @property
    def _shuffle_shift(self) -> float | None:
        """Minimum shift in seconds from the spatial config."""
        sc = self._spatial_cfg
        return sc.min_shift_seconds if sc else None

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

        # Behavior — load positions and compute speed (px/s)
        self.trajectory, _ = load_behavior_data(
            behavior_position=self.behavior_position_path,
            behavior_timestamp=self.behavior_timestamp_path,
            bodypart=bcfg.bodypart,
            speed_window_frames=bcfg.speed_window_frames,
            speed_threshold=0.0,
            x_col=bcfg.x_col,
            y_col=bcfg.y_col,
        )
        logger.info("Loaded trajectory: %d frames", len(self.trajectory))

        # Visualization assets (max projection, footprints)
        self.traces, self.max_proj, self.footprints = load_visualization_data(
            neural_path=self.neural_path,
            trace_name=ncfg.trace_name,
        )

        # Behavior video frame (single frame for calibration overlay)
        if self.behavior_video_path is not None and self.behavior_video_path.exists():
            try:
                import cv2

                cap = cv2.VideoCapture(str(self.behavior_video_path))
                ret, frame = cap.read()
                cap.release()
                if ret:
                    self.behavior_video_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    logger.info(
                        "Loaded behavior video frame from %s", self.behavior_video_path.name
                    )
                else:
                    logger.warning("Could not read frame from %s", self.behavior_video_path)
            except ImportError:
                logger.warning("cv2 not installed — skipping behavior video frame")

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
            logger.info("Speed recomputed in mm/s")
            speed_unit = "mm/s"
        else:
            logger.warning(
                "No arena_bounds — skipping spatial corrections "
                "(jump removal, perspective correction, boundary clipping)"
            )
            logger.warning("No arena calibration — speed and position remain in pixels")
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

    def deconvolve(
        self,
        progress_bar: Any = None,
    ) -> None:
        """Run OASIS deconvolution on calcium traces.

        Parameters
        ----------
        progress_bar:
            Progress bar wrapper, e.g. ``tqdm``.
        """
        if self.traces is None:
            raise RuntimeError("Call load() first.")

        oasis = self.cfg.neural.oasis

        all_unit_ids = list(map(int, self.traces["unit_id"].values))
        logger.info("Deconvolving %d units (g=%s)...", len(all_unit_ids), oasis.g)

        self.good_unit_ids, self.S_list = run_deconvolution(
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

    @abc.abstractmethod
    def compute_occupancy(self) -> None:
        """Compute occupancy map from speed-filtered trajectory."""
        ...

    @abc.abstractmethod
    def analyze_units(self, progress_bar: Any = None) -> None:
        """Run spatial analysis for all deconvolved units."""
        ...

    def place_cells(self) -> dict[int, UnitResult]:
        """Return units passing both significance and stability tests."""
        p_thresh = self.p_value_threshold

        out: dict[int, UnitResult] = {}
        for uid, res in self.unit_results.items():
            if res.p_val >= p_thresh:
                continue
            if np.isnan(res.stability_p_val) or res.stability_p_val >= p_thresh:
                continue
            out[uid] = res
        return out

    def summary(self) -> dict[str, int]:
        """Compute summary counts of significant and stable units.

        Returns
        -------
        dict
            Keys: ``n_total``, ``n_sig``, ``n_stable``, ``n_place_cells``.
        """
        p_thresh = self.p_value_threshold

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

    # ── Bundle I/O ──────────────────────────────────────────────────────

    def save_bundle(self, path: str | Path, *, save_figures: bool = True) -> Path:
        """Save all analysis results to a portable ``.pcellbundle`` directory.

        The bundle is self-contained: it stores config, behavior, neural,
        and per-unit analysis results so that visualizations can be
        recreated without access to the original raw data.

        Parameters
        ----------
        path:
            Output directory. ``.pcellbundle`` is appended if not present.

        Returns
        -------
        Path
            The bundle directory that was created.
        """
        path = Path(path)
        if path.suffix != ".pcellbundle":
            path = path.with_suffix(".pcellbundle")

        # Avoid overwriting: append _1, _2, ... if path already exists
        path = unique_bundle_path(path.parent, path.stem)

        path.mkdir(parents=True, exist_ok=True)

        # Metadata
        meta = {
            "version": _BUNDLE_VERSION,
            "created": datetime.now(UTC).isoformat(),
        }
        (path / "metadata.json").write_text(json.dumps(meta, indent=2))

        # Config
        self.cfg.to_yaml(path / "config.yaml")

        # Spatial arrays
        spatial_kw: dict[str, np.ndarray] = {}
        if self.occupancy_time is not None:
            spatial_kw["occupancy_time"] = self.occupancy_time
        if self.valid_mask is not None:
            spatial_kw["valid_mask"] = self.valid_mask
        if self.x_edges is not None:
            spatial_kw["x_edges"] = self.x_edges
        if self.y_edges is not None:
            spatial_kw["y_edges"] = self.y_edges
        if self.max_proj is not None:
            spatial_kw["max_proj"] = self.max_proj
        if self.behavior_video_frame is not None:
            spatial_kw["behavior_video_frame"] = self.behavior_video_frame
        if spatial_kw:
            np.savez_compressed(path / "spatial.npz", **spatial_kw)

        # xarray DataArrays
        if self.footprints is not None:
            self.footprints.to_netcdf(path / "footprints.nc")
        if self.traces is not None:
            self.traces.to_netcdf(path / "traces.nc")

        # DataFrames
        for name, df in [
            ("trajectory_raw", self.trajectory_raw),
            ("trajectory", self.trajectory),
            ("trajectory_filtered", self.trajectory_filtered),
            ("event_index", self.event_index),
            ("event_place", self.event_place),
        ]:
            if df is not None:
                df.to_parquet(path / f"{name}.parquet")

        # Deconvolution data
        deconv_kw: dict[str, np.ndarray] = {}
        if self.good_unit_ids:
            deconv_kw["good_unit_ids"] = np.array(self.good_unit_ids)
        for i, s in enumerate(self.S_list):
            deconv_kw[f"S_{i}"] = s
        if deconv_kw:
            np.savez_compressed(path / "deconv.npz", **deconv_kw)

        # Unit results
        if self.unit_results:
            ur_dir = path / "unit_results"
            ur_dir.mkdir(exist_ok=True)
            self._save_unit_results(ur_dir)

        # Summary figures
        if save_figures:
            figures_dir = path / "figures"
            figures_dir.mkdir(exist_ok=True)
            saved = self._save_summary_figures(figures_dir)
            if saved:
                logger.info("Saved %d summary figures to %s", len(saved), figures_dir)

        logger.info("Bundle saved to %s", path)
        return path

    def _save_summary_figures(self, figures_dir: Path) -> list[str]:
        """Generate and save key summary figures as PDFs into *figures_dir*."""
        try:
            import matplotlib
            import matplotlib.pyplot as _plt
        except ImportError:
            logger.warning("matplotlib not available — skipping figure export")
            return []

        from placecell.visualization import (
            plot_diagnostics,
            plot_footprints_filled,
            plot_summary_scatter,
        )

        saved: list[str] = []
        rc = {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "Arial",
        }

        with matplotlib.rc_context(rc):
            if self.unit_results:
                for name, fn in [
                    ("diagnostics.pdf", lambda: plot_diagnostics(
                        self.unit_results, p_value_threshold=self.p_value_threshold)),
                    ("summary_scatter.pdf", lambda: plot_summary_scatter(
                        self.unit_results, p_value_threshold=self.p_value_threshold,
                        n_shuffles=self._shuffle_n, min_shift_seconds=self._shuffle_shift)),
                ]:
                    try:
                        fig = fn()
                        fig.savefig(figures_dir / name, bbox_inches="tight")
                        _plt.close(fig)
                        saved.append(name)
                    except Exception:
                        logger.warning("Failed to save %s", name, exc_info=True)

            if self.max_proj is not None and self.footprints is not None:
                try:
                    fig = plot_footprints_filled(self.max_proj, self.footprints)
                    fig.savefig(figures_dir / "footprints.pdf", bbox_inches="tight")
                    _plt.close(fig)
                    saved.append("footprints.pdf")
                except Exception:
                    logger.warning("Failed to save footprints.pdf", exc_info=True)

        return saved

    def _save_unit_results(self, ur_dir: Path) -> None:
        """Serialize unit_results into *ur_dir*."""
        scalar_fields = [
            "si",
            "p_val",
            "overall_rate",
            "stability_corr",
            "stability_z",
            "stability_p_val",
        ]
        array_fields = [
            "rate_map",
            "rate_map_raw",
            "shuffled_sis",
            "shuffled_rate_p95",
            "shuffled_stability",
            "rate_map_first",
            "rate_map_second",
            "trace_data",
            "trace_times",
        ]
        df_fields = ["vis_data_above", "unit_data"]

        # Scalars → CSV
        rows = []
        for uid, res in self.unit_results.items():
            row = {"unit_id": uid}
            for f in scalar_fields:
                row[f] = getattr(res, f)
            rows.append(row)
        pd.DataFrame(rows).to_csv(ur_dir / "scalars.csv", index=False)

        # Arrays → NPZ
        arrays: dict[str, np.ndarray] = {}
        for uid, res in self.unit_results.items():
            for f in array_fields:
                val = getattr(res, f)
                if val is not None:
                    arrays[f"{uid}_{f}"] = val
        if arrays:
            np.savez_compressed(ur_dir / "arrays.npz", **arrays)

        # DataFrames → single parquet (concatenated with identifiers)
        parts = []
        for uid, res in self.unit_results.items():
            for f in df_fields:
                df = getattr(res, f)
                if df is not None and not df.empty:
                    chunk = df.copy()
                    chunk["_unit_id"] = uid
                    chunk["_field"] = f
                    parts.append(chunk)
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(ur_dir / "events.parquet")

    @classmethod
    def load_bundle(cls, path: str | Path) -> "BasePlaceCellDataset":
        """Load a previously saved ``.pcellbundle`` directory.

        Parameters
        ----------
        path:
            Path to the ``.pcellbundle`` directory.

        Returns
        -------
        BasePlaceCellDataset
            Dataset with all attributes restored. Recomputation methods
            (``load``, ``deconvolve``, etc.) are unavailable since the
            original raw data paths are not preserved.
        """
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"Bundle not found: {path}")

        # Metadata
        meta = json.loads((path / "metadata.json").read_text())
        if meta["version"] > _BUNDLE_VERSION:
            raise ValueError(
                f"Bundle version {meta['version']} is newer than supported ({_BUNDLE_VERSION})"
            )

        # Config
        cfg = AnalysisConfig.from_yaml(path / "config.yaml")

        # Auto-select subclass based on behavior type
        if cls is BasePlaceCellDataset:
            if cfg.behavior and cfg.behavior.type == "maze":
                from placecell.maze_dataset import MazeDataset

                return MazeDataset.load_bundle(path)
            cls = ArenaDataset

        ds = cls(cfg)

        # Spatial arrays
        spatial_path = path / "spatial.npz"
        if spatial_path.exists():
            spatial = np.load(spatial_path)
            ds.occupancy_time = spatial.get("occupancy_time")
            ds.valid_mask = spatial.get("valid_mask")
            ds.x_edges = spatial.get("x_edges")
            ds.y_edges = spatial.get("y_edges")
            ds.max_proj = spatial.get("max_proj")
            ds.behavior_video_frame = spatial.get("behavior_video_frame")

        # xarray DataArrays
        fp_path = path / "footprints.nc"
        if fp_path.exists():
            ds.footprints = xr.open_dataarray(fp_path).load()
        tr_path = path / "traces.nc"
        if tr_path.exists():
            ds.traces = xr.open_dataarray(tr_path).load()

        # DataFrames
        df_names = [
            "trajectory_raw",
            "trajectory",
            "trajectory_filtered",
            "event_index",
            "event_place",
        ]
        for name in df_names:
            pq = path / f"{name}.parquet"
            if pq.exists():
                setattr(ds, name, pd.read_parquet(pq))

        # Deconvolution data
        deconv_path = path / "deconv.npz"
        if deconv_path.exists():
            deconv = np.load(deconv_path)
            if "good_unit_ids" in deconv:
                ds.good_unit_ids = list(deconv["good_unit_ids"])
            i = 0
            while f"S_{i}" in deconv:
                ds.S_list.append(deconv[f"S_{i}"])
                i += 1

        # Unit results
        ur_dir = path / "unit_results"
        if ur_dir.is_dir():
            ds.unit_results = cls._load_unit_results(ur_dir)

        logger.info(
            "Loaded bundle: %d units, %d results",
            len(ds.good_unit_ids),
            len(ds.unit_results),
        )
        return ds

    @staticmethod
    def _load_unit_results(ur_dir: Path) -> dict[int, "UnitResult"]:
        """Reconstruct unit_results from saved files."""
        scalar_fields = [
            "si",
            "p_val",
            "overall_rate",
            "stability_corr",
            "stability_z",
            "stability_p_val",
        ]
        # Read scalars
        scalars_df = pd.read_csv(ur_dir / "scalars.csv")
        unit_ids = scalars_df["unit_id"].tolist()

        scalar_defaults = {"overall_rate": 0.0}
        scalars_by_uid: dict[int, dict] = {}
        for _, row in scalars_df.iterrows():
            uid = int(row["unit_id"])
            scalars_by_uid[uid] = {
                f: row[f] if f in row.index else scalar_defaults.get(f, np.nan)
                for f in scalar_fields
            }

        # Read arrays
        arrays_by_uid: dict[int, dict] = {uid: {} for uid in unit_ids}
        arrays_path = ur_dir / "arrays.npz"
        if arrays_path.exists():
            arrays_npz = np.load(arrays_path)
            for key in arrays_npz.files:
                # key format: "{uid}_{field}"
                parts = key.split("_", 1)
                uid = int(parts[0])
                field = parts[1]
                if uid in arrays_by_uid:
                    arrays_by_uid[uid][field] = arrays_npz[key]

        # Read events
        events_by_uid: dict[int, dict] = {uid: {} for uid in unit_ids}
        events_path = ur_dir / "events.parquet"
        if events_path.exists():
            events_df = pd.read_parquet(events_path)
            for (uid, field), group in events_df.groupby(["_unit_id", "_field"]):
                uid = int(uid)
                if uid in events_by_uid:
                    events_by_uid[uid][field] = group.drop(
                        columns=["_unit_id", "_field"]
                    ).reset_index(drop=True)

        # Assemble UnitResult objects
        results: dict[int, UnitResult] = {}
        for uid in unit_ids:
            sc = scalars_by_uid[uid]
            ar = arrays_by_uid.get(uid, {})
            ev = events_by_uid.get(uid, {})
            results[uid] = UnitResult(
                rate_map=ar.get("rate_map", np.array([])),
                rate_map_raw=ar.get("rate_map_raw", np.array([])),
                si=float(sc["si"]),
                p_val=float(sc["p_val"]),
                overall_rate=float(sc.get("overall_rate", 0.0)),
                shuffled_sis=ar.get("shuffled_sis", np.array([])),
                shuffled_rate_p95=ar.get("shuffled_rate_p95", np.array([])),
                stability_corr=float(sc["stability_corr"]),
                stability_z=float(sc["stability_z"]),
                stability_p_val=float(sc["stability_p_val"]),
                shuffled_stability=ar.get("shuffled_stability", np.array([])),
                rate_map_first=ar.get("rate_map_first", np.array([])),
                rate_map_second=ar.get("rate_map_second", np.array([])),
                vis_data_above=ev.get("vis_data_above", pd.DataFrame()),
                unit_data=ev.get("unit_data", pd.DataFrame()),
                trace_data=ar.get("trace_data"),
                trace_times=ar.get("trace_times"),
            )
        return results


class ArenaDataset(BasePlaceCellDataset):
    """Dataset for 2D open-field arena place cell analysis.

    Adds arena-specific functionality: perspective correction scale
    properties, 2D occupancy computation, and 2D spatial analysis.
    """

    @property
    def spatial(self) -> SpatialMap2DConfig:
        """Shortcut to 2D spatial map config."""
        return self.cfg.behavior.spatial_map_2d

    @property
    def p_value_threshold(self) -> float:
        """P-value threshold from 2D spatial map config."""
        return self.spatial.p_value_threshold

    def compute_occupancy(self) -> None:
        """Compute 2D occupancy map from speed-filtered trajectory."""
        if self.trajectory_filtered is None:
            raise RuntimeError("Call preprocess_behavior() first.")

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
        """Run 2D spatial analysis for all deconvolved units with events.

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
                min_shift_seconds=scfg.min_shift_seconds,
                si_weight_mode=scfg.si_weight_mode,
                place_field_seed_percentile=scfg.place_field_seed_percentile,
                n_split_blocks=scfg.n_split_blocks,
                block_shifts=scfg.block_shifts,
            )

            # Attach visualization data
            vis_data_above = result["events_above_threshold"]

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
                vis_data_above=vis_data_above,
                unit_data=result["unit_data"],
                trace_data=trace_data,
                trace_times=trace_times,
            )

        logger.info("Done. %d units analyzed.", len(self.unit_results))

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
                        self.trajectory_filtered, self.occupancy_time,
                        self.valid_mask, self.x_edges, self.y_edges,
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
                        self.x_edges, self.y_edges, self.valid_mask,
                        len(place_cell_results),
                    )
                    fig.savefig(figures_dir / "coverage.pdf", bbox_inches="tight")
                    _plt.close(fig)
                    saved.append("coverage.pdf")
                except Exception:
                    logger.warning("Failed to save coverage.pdf", exc_info=True)

        return saved
