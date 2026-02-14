"""Central dataset class for place cell analysis."""

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
    load_curated_unit_ids,
    remove_position_jumps,
)
from placecell.config import AnalysisConfig, DataPathsConfig, SpatialMapConfig
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
    """Analysis results for a single unit."""

    rate_map: np.ndarray
    rate_map_raw: np.ndarray
    si: float
    p_val: float
    shuffled_sis: np.ndarray
    shuffled_rate_p95: np.ndarray | None
    stability_corr: float
    stability_z: float
    stability_p_val: float
    rate_map_first: np.ndarray
    rate_map_second: np.ndarray
    vis_data_above: pd.DataFrame
    vis_data_below: pd.DataFrame
    unit_data: pd.DataFrame
    trace_data: np.ndarray | None
    trace_times: np.ndarray | None


class PlaceCellDataset:
    """Container for a single recording session's place cell analysis.

    Pipeline (each step populates attributes for the next)::

        ds = PlaceCellDataset.from_yaml(config_path, data_path)
        ds.load()                            # traces, trajectory, footprints
        ds.preprocess_behavior()             # jump/perspective/clip corrections
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
        neural_path: Path | None = None,
        neural_timestamp_path: Path | None = None,
        behavior_position_path: Path | None = None,
        behavior_timestamp_path: Path | None = None,
        behavior_video_path: Path | None = None,
        curation_csv_path: Path | None = None,
        data_cfg: DataPathsConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.neural_path = neural_path
        self.neural_timestamp_path = neural_timestamp_path
        self.behavior_position_path = behavior_position_path
        self.behavior_timestamp_path = behavior_timestamp_path
        self.behavior_video_path = behavior_video_path
        self.curation_csv_path = curation_csv_path
        self.data_cfg = data_cfg

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
        self.behavior_video_frame: np.ndarray | None = None

        # Results
        self.unit_results: dict[int, UnitResult] = {}

    @classmethod
    def from_yaml(cls, config: str | Path, data_path: str | Path) -> "PlaceCellDataset":
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

        return cls(
            cfg=cfg,
            neural_path=data_dir / data_cfg.neural_path,
            neural_timestamp_path=data_dir / data_cfg.neural_timestamp,
            behavior_position_path=data_dir / data_cfg.behavior_position,
            behavior_timestamp_path=data_dir / data_cfg.behavior_timestamp,
            behavior_video_path=(
                data_dir / data_cfg.behavior_video if data_cfg.behavior_video else None
            ),
            curation_csv_path=(data_dir / data_cfg.curation_csv if data_cfg.curation_csv else None),
            data_cfg=data_cfg,
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

        # Behavior — load positions and compute speed
        self.trajectory, _ = load_behavior_data(
            behavior_position=self.behavior_position_path,
            behavior_timestamp=self.behavior_timestamp_path,
            bodypart=bcfg.bodypart,
            speed_window_frames=bcfg.speed_window_frames,
            speed_threshold=0.0,  # no filtering here; filter below after unit conversion
        )

        # Convert speed from px/s to mm/s when arena calibration is available
        scale = self.mm_per_px
        if scale is not None:
            self.trajectory["speed"] = self.trajectory["speed"] * scale
            logger.info("Speed converted to mm/s (%.3f mm/px)", scale)

        # Apply speed filter (mm/s if calibrated, px/s otherwise)
        self.trajectory_filtered = self.trajectory[
            self.trajectory["speed"] >= bcfg.speed_threshold
        ].copy()
        self.trajectory_filtered = self.trajectory_filtered.sort_values("frame_index").rename(
            columns={"frame_index": "beh_frame_index"}
        )
        logger.info(
            "Trajectory: %d frames (%d after speed filter at %.1f %s)",
            len(self.trajectory),
            len(self.trajectory_filtered),
            bcfg.speed_threshold,
            "mm/s" if scale else "px/s",
        )

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
        """Mm-per-pixel scale, or None if arena is not calibrated."""
        if self.data_cfg is None:
            return None
        bounds = self.data_cfg.arena_bounds
        size_mm = self.data_cfg.arena_size_mm
        if bounds is None or size_mm is None:
            return None
        x_min, x_max, y_min, y_max = bounds
        scale_x = size_mm[0] / (x_max - x_min)
        scale_y = size_mm[1] / (y_max - y_min)
        return (scale_x + scale_y) / 2.0

    def preprocess_behavior(self) -> None:
        """Apply data-integrity corrections and recompute speed.

        Pipeline: jump removal → perspective correction → boundary clipping
        → recompute speed (mm/s) → re-filter by speed threshold.

        Requires ``load()`` to have been called first.  Skipped automatically
        when ``arena_bounds`` is not set in the data config.
        """
        if self.trajectory is None:
            raise RuntimeError("Call load() first.")

        dcfg = self.data_cfg
        if dcfg is None or dcfg.arena_bounds is None:
            logger.info("No arena_bounds configured — skipping behavior preprocessing")
            return

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

        bcfg = self.cfg.behavior
        scale = self.mm_per_px

        # Store intermediate snapshots for visualization
        self._preprocess_steps: dict[str, pd.DataFrame] = {}
        self._preprocess_steps["Raw"] = self.trajectory[["x", "y"]].copy()

        # 1. Jump removal (threshold in mm → convert to px)
        jump_px = bcfg.jump_threshold_mm / scale
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

        # 4. Recompute speed in-place on corrected positions (preserves index)
        traj = self.trajectory.sort_values("frame_index")
        x_vals = traj["x"].values
        y_vals = traj["y"].values
        t_vals = traj["unix_time"].values
        n = len(traj)
        w = bcfg.speed_window_frames
        speed = np.zeros(n)
        for i in range(n):
            end_idx = min(i + w, n - 1)
            if end_idx > i:
                dx = x_vals[end_idx] - x_vals[i]
                dy = y_vals[end_idx] - y_vals[i]
                dt = t_vals[end_idx] - t_vals[i]
                if dt > 0:
                    speed[i] = np.sqrt(dx**2 + dy**2) / dt
        # Scale pixel speed to mm/s
        self.trajectory.loc[traj.index, "speed"] = speed * scale
        logger.info("Speed recomputed in mm/s (%.3f mm/px)", scale)

        # 5. Re-filter with mm/s threshold
        self.trajectory_filtered = self.trajectory[
            self.trajectory["speed"] >= bcfg.speed_threshold
        ].copy()
        self.trajectory_filtered = self.trajectory_filtered.sort_values("frame_index")
        self.trajectory_filtered = self.trajectory_filtered.rename(
            columns={"frame_index": "beh_frame_index"}
        )
        logger.info(
            "Trajectory after preprocessing: %d frames (%d after speed filter at %.1f mm/s)",
            len(self.trajectory),
            len(self.trajectory_filtered),
            bcfg.speed_threshold,
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

            self.unit_results[unit_id] = UnitResult(
                rate_map=result["rate_map"],
                rate_map_raw=result["rate_map_raw"],
                si=result["si"],
                shuffled_sis=result["shuffled_sis"],
                shuffled_rate_p95=result["shuffled_rate_p95"],
                p_val=result["p_val"],
                stability_corr=result["stability_corr"],
                stability_z=result["stability_z"],
                stability_p_val=result["stability_p_val"],
                rate_map_first=result["rate_map_first"],
                rate_map_second=result["rate_map_second"],
                vis_data_above=vis_data_above,
                vis_data_below=vis_data_below,
                unit_data=result["unit_data"],
                trace_data=trace_data,
                trace_times=trace_times,
            )

        logger.info("Done. %d units analyzed.", len(self.unit_results))

    def place_cells(self) -> dict[int, UnitResult]:
        """Return units passing both significance and stability tests."""
        p_thresh = self.spatial.p_value_threshold or 0.05
        stab_thresh = self.spatial.stability_threshold

        out: dict[int, UnitResult] = {}
        for uid, res in self.unit_results.items():
            if res.p_val >= p_thresh:
                continue
            stab_corr = res.stability_corr
            stab_p = res.stability_p_val
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

    def summary(self) -> dict[str, int]:
        """Compute summary counts of significant and stable units.

        Returns
        -------
        dict
            Keys: ``n_total``, ``n_sig``, ``n_stable_thresh``,
            ``n_stable_shuffle``, ``n_both_thresh``, ``n_both_shuffle``.
        """
        p_thresh = self.spatial.p_value_threshold or 0.05
        stab_thresh = self.spatial.stability_threshold

        n_sig = 0
        n_stable_thresh = 0
        n_stable_shuffle = 0
        n_both_thresh = 0
        n_both_shuffle = 0

        for res in self.unit_results.values():
            is_sig = res.p_val < p_thresh
            corr = res.stability_corr
            stab_p = res.stability_p_val

            is_stable_thresh = not np.isnan(corr) and corr >= stab_thresh
            is_stable_shuffle = not np.isnan(stab_p) and stab_p < p_thresh

            if is_sig:
                n_sig += 1
            if is_stable_thresh:
                n_stable_thresh += 1
            if is_stable_shuffle:
                n_stable_shuffle += 1
            if is_sig and is_stable_thresh:
                n_both_thresh += 1
            if is_sig and is_stable_shuffle:
                n_both_shuffle += 1

        return {
            "n_total": len(self.unit_results),
            "n_sig": n_sig,
            "n_stable_thresh": n_stable_thresh,
            "n_stable_shuffle": n_stable_shuffle,
            "n_both_thresh": n_both_thresh,
            "n_both_shuffle": n_both_shuffle,
        }

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

    # ── Bundle I/O ──────────────────────────────────────────────────────

    def save_bundle(self, path: str | Path) -> Path:
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
        for i, c in enumerate(self.C_list):
            deconv_kw[f"C_{i}"] = c
        for i, s in enumerate(self.S_list):
            deconv_kw[f"S_{i}"] = s
        if deconv_kw:
            np.savez_compressed(path / "deconv.npz", **deconv_kw)

        # Unit results
        if self.unit_results:
            ur_dir = path / "unit_results"
            ur_dir.mkdir(exist_ok=True)
            self._save_unit_results(ur_dir)

        logger.info("Bundle saved to %s", path)
        return path

    def _save_unit_results(self, ur_dir: Path) -> None:
        """Serialize unit_results into *ur_dir*."""
        scalar_fields = [
            "si",
            "p_val",
            "stability_corr",
            "stability_z",
            "stability_p_val",
        ]
        array_fields = [
            "rate_map",
            "rate_map_raw",
            "shuffled_sis",
            "shuffled_rate_p95",
            "rate_map_first",
            "rate_map_second",
            "trace_data",
            "trace_times",
        ]
        df_fields = ["vis_data_above", "vis_data_below", "unit_data"]

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
    def load_bundle(cls, path: str | Path) -> "PlaceCellDataset":
        """Load a previously saved ``.pcellbundle`` directory.

        Parameters
        ----------
        path:
            Path to the ``.pcellbundle`` directory.

        Returns
        -------
        PlaceCellDataset
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
        for name in ["trajectory", "trajectory_filtered", "event_index", "event_place"]:
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
            while f"C_{i}" in deconv:
                ds.C_list.append(deconv[f"C_{i}"])
                i += 1
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
            "stability_corr",
            "stability_z",
            "stability_p_val",
        ]
        # Read scalars
        scalars_df = pd.read_csv(ur_dir / "scalars.csv")
        unit_ids = scalars_df["unit_id"].tolist()

        scalars_by_uid: dict[int, dict] = {}
        for _, row in scalars_df.iterrows():
            uid = int(row["unit_id"])
            scalars_by_uid[uid] = {f: row[f] for f in scalar_fields}

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
                shuffled_sis=ar.get("shuffled_sis", np.array([])),
                shuffled_rate_p95=ar.get("shuffled_rate_p95"),
                stability_corr=float(sc["stability_corr"]),
                stability_z=float(sc["stability_z"]),
                stability_p_val=float(sc["stability_p_val"]),
                rate_map_first=ar.get("rate_map_first", np.array([])),
                rate_map_second=ar.get("rate_map_second", np.array([])),
                vis_data_above=ev.get("vis_data_above", pd.DataFrame()),
                vis_data_below=ev.get("vis_data_below", pd.DataFrame()),
                unit_data=ev.get("unit_data", pd.DataFrame()),
                trace_data=ar.get("trace_data"),
                trace_times=ar.get("trace_times"),
            )
        return results
