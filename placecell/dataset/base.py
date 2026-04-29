"""Base classes for place cell analysis datasets."""

import abc
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from placecell import __version__ as _PACKAGE_VERSION
from placecell.config import (
    CONFIG_DIR,
    AnalysisConfig,
    BaseDataConfig,
    BaseSpatialMapConfig,
    MazeDataConfig,
)
from placecell.loaders import load_visualization_data
from placecell.log import init_logger
from placecell.neural import load_calcium_traces, run_deconvolution

logger = init_logger(__name__)

_BUNDLE_VERSION = 2


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
class StabilitySplitResult:
    """Result of one stability shuffle test at a given ``n_split_blocks``.

    Parameters
    ----------
    n_split_blocks:
        Number of interleaved temporal blocks used to split the session
        (2 = classic first/second half).
    corr:
        Pearson correlation between first-half and second-half rate maps.
    fisher_z:
        Fisher z-transform of ``corr``.
    p_val:
        Shuffle-based p-value (NaN if ``n_shuffles == 0``).
    shuffled_corrs:
        Correlations from shuffled data (used for significance).
    rate_map_first:
        Smoothed rate map for the first half of the split, in firing-rate
        units (same scale as ``UnitResult.rate_map_smoothed``).
    rate_map_second:
        Smoothed rate map for the second half of the split, in firing-rate
        units (same scale as ``UnitResult.rate_map_smoothed``).
    """

    n_split_blocks: int
    corr: float
    fisher_z: float
    p_val: float
    shuffled_corrs: np.ndarray
    rate_map_first: np.ndarray
    rate_map_second: np.ndarray


@dataclass
class UnitResult:
    """Analysis results for a single unit.

    Parameters
    ----------
    rate_map_smoothed:
        Smoothed rate map in firing-rate units (events·s⁻¹ per bin).
        This is the authoritative rate map for quantitative analyses.
    rate_map_raw:
        Unsmoothed rate map in firing-rate units (events·s⁻¹ per bin).
    rate_map_peak_normalized:
        (Property, derived) Smoothed rate map divided by its peak so
        values span 0-1. Used for display-friendly colorbars. Computed
        on demand from ``rate_map_smoothed``.
    si:
        Spatial information (bits/spike).
    p_val:
        P-value from spatial information significance test.
    shuffled_sis:
        Spatial information values from shuffled data (for significance test).
    shuffled_rate_p95:
        Per-bin 95th percentile of smoothed shuffled rate maps, in the same
        firing-rate units as ``rate_map_smoothed`` (used for place-field
        seed detection).
    stability_splits:
        One :class:`StabilitySplitResult` per entry in
        ``spatial_map.stability_splits``. A cell is considered stable only if
        every split's ``p_val`` is below the threshold.
    vis_data_above:
        Subset of unit_data where event amplitude exceeds the threshold
        (used for plotting event dots on rate maps).
    unit_data:
        Speed-filtered deconvolved events for this unit (subset of event_place).
    overall_rate:
        Activity rate in a.u./s (sum of deconvolved amplitudes / total time).
    event_count_rate:
        Event count rate in 1/s (number of events / total time).
    trace_data:
        Neural trace for this unit (None if traces unavailable).
    trace_times:
        Time axis in seconds corresponding to trace_data.

    """

    rate_map_smoothed: np.ndarray
    rate_map_raw: np.ndarray
    si: float
    p_val: float
    shuffled_sis: np.ndarray
    shuffled_rate_p95: np.ndarray
    stability_splits: list[StabilitySplitResult]
    vis_data_above: pd.DataFrame
    unit_data: pd.DataFrame
    overall_rate: float
    event_count_rate: float
    trace_data: np.ndarray | None
    trace_times: np.ndarray | None

    @property
    def rate_map_peak_normalized(self) -> np.ndarray:
        """Smoothed rate map divided by its peak (0-1 range, for display).

        NaN bins pass through; zero-peak maps are returned unchanged
        (all zeros/NaN).
        """
        rm = np.asarray(self.rate_map_smoothed, dtype=float).copy()
        finite = np.isfinite(rm)
        if not finite.any():
            return rm
        peak = float(np.nanmax(rm[finite]))
        if peak > 0:
            rm[finite] = rm[finite] / peak
        return rm

    def is_stable(self, p_threshold: float) -> bool:
        """True iff every stability split's p-value is below ``p_threshold``."""
        if not self.stability_splits:
            return False
        return all(not np.isnan(s.p_val) and s.p_val < p_threshold for s in self.stability_splits)


class BasePlaceCellDataset(abc.ABC):
    """Base class for place cell analysis datasets.

    Shared pipeline (each step populates attributes for the next)::

        ds = BasePlaceCellDataset.from_yaml(config_path, data_path)
        ds.load()                            # traces, trajectory, footprints
        ds.preprocess_behavior()             # corrections + speed filter
        ds.deconvolve(progress_bar=tqdm)     # good_unit_ids, S_list
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
        zone_tracking_path: Path | None = None,
        data_cfg: BaseDataConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.neural_path = neural_path
        self.neural_timestamp_path = neural_timestamp_path
        self.behavior_position_path = behavior_position_path
        self.behavior_timestamp_path = behavior_timestamp_path
        self.behavior_video_path = behavior_video_path
        self.behavior_graph_path = behavior_graph_path
        self.zone_tracking_path = zone_tracking_path
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

        # Canonical neural-rate table: one row per neural frame, columns
        # frame_index, neural_time, x, y, speed, [pos_1d, ...], s_unit_<id>...
        self.canonical: pd.DataFrame | None = None

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
            Path to analysis config YAML, or a stem name matching a
            bundled config in ``placecell/config/`` (e.g. ``"example_arena_config"``).
        data_path:
            Path to the per-session data paths YAML file.
        """
        config_path = Path(config)
        if not config_path.suffix and not config_path.exists():
            config_path = CONFIG_DIR / f"{config}.yaml"
        cfg = AnalysisConfig.from_yaml(config_path)

        data_path = Path(data_path)
        data_dir = data_path.parent
        data_cfg = BaseDataConfig.from_yaml(data_path)
        cfg = cfg.with_data_overrides(data_cfg)

        # Auto-select subclass based on behavior type
        klass = cls
        if cfg.behavior and cfg.behavior.type == "maze":
            from placecell.dataset.maze import MazeDataset

            klass = MazeDataset
        elif cls is BasePlaceCellDataset:
            from placecell.dataset.arena import ArenaDataset

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
                data_dir / data_cfg.behavior_graph
                if isinstance(data_cfg, MazeDataConfig) and data_cfg.behavior_graph
                else None
            ),
            zone_tracking_path=(
                data_dir / (data_cfg.zone_tracking or f"zone_tracking_{data_path.stem}.csv")
                if isinstance(data_cfg, MazeDataConfig)
                else None
            ),
            data_cfg=data_cfg,
        )

    @property
    @abc.abstractmethod
    def p_value_threshold(self) -> float:
        """P-value threshold from the appropriate spatial config."""
        ...

    @property
    def _spatial_cfg(self) -> "BaseSpatialMapConfig | None":
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

    def _validate_neural_timestamps(self) -> None:
        """Read and validate neural timestamps; log a quality report.

        Called by :meth:`_load_neural_and_viz` so the user sees timestamp
        quality immediately after ``load()``, before waiting for
        deconvolution.
        """
        from placecell.dataset_validation import infer_fps, validate_neural_timestamps

        ts_df = pd.read_csv(self.neural_timestamp_path)
        if "timestamp_first" not in ts_df.columns:
            raise ValueError(
                "neural_timestamp CSV must contain a 'timestamp_first' column. "
                f"Got: {list(ts_df.columns)}"
            )
        neural_time = ts_df["timestamp_first"].to_numpy()

        # Run the same validation that match_events will run later;
        # log results now so the user sees quality before deconvolution.
        clean_time, _ = validate_neural_timestamps(neural_time)
        fps = infer_fps(clean_time)
        n_excluded = len(neural_time) - len(clean_time)
        if n_excluded == 0:
            logger.info(
                "Neural timestamps: %d frames, inferred %.1f Hz",
                len(neural_time),
                fps,
            )

        # Store raw timestamps for downstream use by match_events.
        self._neural_time_raw = neural_time

    def _load_neural_and_viz(self) -> None:
        """Load neural traces and visualization assets (shared by all subclasses)."""
        ncfg = self.cfg.neural

        # Traces
        self.traces = load_calcium_traces(self.neural_path, trace_name=ncfg.trace_name)
        logger.info(
            "Loaded traces: %d units, %d frames",
            self.traces.sizes["unit_id"],
            self.traces.sizes["frame"],
        )

        # Neural timestamp quality check — runs early so the user sees
        # warnings before waiting for deconvolution.
        self._validate_neural_timestamps()

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
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                idx = min(self.data_cfg.overlay_frame_index, total - 1) if total > 0 else 0
                if idx > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    self.behavior_video_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    logger.info(
                        "Loaded behavior video frame %d from %s", idx, self.behavior_video_path.name
                    )
                else:
                    logger.warning("Could not read frame from %s", self.behavior_video_path)
            except ImportError:
                logger.warning("cv2 not installed — skipping behavior video frame")

    @abc.abstractmethod
    def load(self) -> None:
        """Load neural traces, behavior data, and visualization assets."""
        ...

    def subset(self, n_units: int | None = None, n_frames: int | None = None) -> None:
        """Trim loaded data to the first *n_units* units and *n_frames* frames.

        Must be called after :meth:`load` and before :meth:`preprocess_behavior`.
        """
        if self.traces is None:
            raise RuntimeError("Call load() before subset()")
        if n_units is not None:
            unit_ids = self.traces.coords["unit_id"].values[:n_units]
            self.traces = self.traces.sel(unit_id=unit_ids)
            if self.footprints is not None:
                self.footprints = self.footprints.sel(unit_id=unit_ids)
        if n_frames is not None:
            frame_ids = self.traces.coords["frame"].values[:n_frames]
            self.traces = self.traces.sel(frame=frame_ids)
            if self.trajectory is not None:
                self.trajectory = self.trajectory.iloc[:n_frames].reset_index(drop=True)
        logger.info(
            "Subset: %d units, %d frames",
            self.traces.sizes["unit_id"],
            self.traces.sizes["frame"],
        )

    @abc.abstractmethod
    def preprocess_behavior(self) -> None:
        """Preprocess behavior data. Subclass-specific pipelines."""
        ...

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
        logger.info("Deconvolved %d units", len(self.good_unit_ids))

    @abc.abstractmethod
    def match_events(self) -> None:
        """Match neural events to behavior positions."""
        ...

    @abc.abstractmethod
    def compute_occupancy(self) -> None:
        """Compute occupancy map from speed-filtered trajectory."""
        ...

    @abc.abstractmethod
    def analyze_units(self, progress_bar: Any = None) -> None:
        """Run spatial analysis for all deconvolved units."""
        ...

    def place_cells(self) -> dict[int, UnitResult]:
        """Return units passing both significance and all stability tests."""
        p_thresh = self.p_value_threshold

        out: dict[int, UnitResult] = {}
        for uid, res in self.unit_results.items():
            if res.p_val >= p_thresh:
                continue
            if not res.is_stable(p_thresh):
                continue
            out[uid] = res
        return out

    def summary(self) -> dict:
        """Compute summary counts and percentages of significant and stable units.

        Returns
        -------
        dict
            Keys: ``n_total``, ``n_sig``, ``n_stable``, ``n_place_cells``,
            ``pct_sig``, ``pct_stable``, ``pct_place_cells``.
        """
        p_thresh = self.p_value_threshold

        n_sig = 0
        n_stable = 0
        n_place_cells = 0

        for res in self.unit_results.values():
            is_sig = res.p_val < p_thresh
            is_stable = res.is_stable(p_thresh)

            if is_sig:
                n_sig += 1
            if is_stable:
                n_stable += 1
            if is_sig and is_stable:
                n_place_cells += 1

        n_total = len(self.unit_results)

        def pct(k: int) -> float:
            return round(100 * k / n_total, 1) if n_total else 0.0

        return {
            "n_total": n_total,
            "n_sig": n_sig,
            "pct_sig": pct(n_sig),
            "n_stable": n_stable,
            "pct_stable": pct(n_stable),
            "n_place_cells": n_place_cells,
            "pct_place_cells": pct(n_place_cells),
        }

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

        # ``version`` is the schema version (enforced on load)
        meta = {
            "version": _BUNDLE_VERSION,
            "placecell_version": _PACKAGE_VERSION,
            "created": datetime.now(UTC).isoformat(),
        }
        (path / "metadata.json").write_text(json.dumps(meta, indent=2))

        # Config
        self.cfg.to_yaml(path / "config.yaml")

        # Data config
        if self.data_cfg is not None:
            with open(path / "data_config.yaml", "w") as f:
                yaml.dump(
                    self.data_cfg.model_dump(mode="json"),
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                )

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
            ("canonical", self.canonical),
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
            plot_stability_splits_summary,
            plot_summary_scatter,
            plot_timestamp_diagnostics,
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
                    (
                        "diagnostics.pdf",
                        lambda: plot_diagnostics(
                            self.unit_results, p_value_threshold=self.p_value_threshold
                        ),
                    ),
                    (
                        "summary_scatter.pdf",
                        lambda: plot_summary_scatter(
                            self.unit_results,
                            p_value_threshold=self.p_value_threshold,
                            n_shuffles=self._shuffle_n,
                            min_shift_seconds=self._shuffle_shift,
                        ),
                    ),
                    (
                        "stability_splits.pdf",
                        lambda: plot_stability_splits_summary(
                            self.unit_results,
                            p_value_threshold=self.p_value_threshold,
                        ),
                    ),
                ]:
                    try:
                        fig = fn()
                        fig.savefig(figures_dir / name, bbox_inches="tight")
                        _plt.close(fig)
                        saved.append(name)
                    except Exception:
                        logger.warning("Failed to save %s", name, exc_info=True)

            try:
                fig = plot_timestamp_diagnostics(self.trajectory_raw, self.canonical)
                fig.savefig(figures_dir / "timestamp_diagnostics.pdf", bbox_inches="tight")
                _plt.close(fig)
                saved.append("timestamp_diagnostics.pdf")
            except Exception:
                logger.warning("Failed to save timestamp_diagnostics.pdf", exc_info=True)

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
        """Serialize unit_results into *ur_dir*.

        Scalar per-unit fields go to ``scalars.csv``.
        Shared arrays (rate_map, shuffled_sis, etc.) go to ``arrays.npz``.
        Per-split stability results (corr, p_val, shuffled_corrs, half maps)
        go to ``stability_scalars.csv`` + ``stability_arrays.npz`` keyed by
        ``{uid}_{split_index}_{field}``. ``stability_scalars.csv`` also
        records ``split_index`` and ``n_split_blocks`` so order and split
        identity survive round-trip.
        """
        scalar_fields = [
            "si",
            "p_val",
            "overall_rate",
            "event_count_rate",
        ]
        array_fields = [
            "rate_map_smoothed",
            "rate_map_raw",
            "shuffled_sis",
            "shuffled_rate_p95",
            "trace_data",
            "trace_times",
        ]
        df_fields = ["vis_data_above", "unit_data"]
        split_scalar_fields = ["n_split_blocks", "corr", "fisher_z", "p_val"]
        split_array_fields = ["shuffled_corrs", "rate_map_first", "rate_map_second"]

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

        # Stability splits: scalars CSV + arrays NPZ
        stab_rows = []
        stab_arrays: dict[str, np.ndarray] = {}
        for uid, res in self.unit_results.items():
            for i, s in enumerate(res.stability_splits):
                row = {"unit_id": uid, "split_index": i}
                for f in split_scalar_fields:
                    row[f] = getattr(s, f)
                stab_rows.append(row)
                for f in split_array_fields:
                    val = getattr(s, f)
                    if val is not None:
                        stab_arrays[f"{uid}_{i}_{f}"] = val
        if stab_rows:
            pd.DataFrame(stab_rows).to_csv(ur_dir / "stability_scalars.csv", index=False)
        if stab_arrays:
            np.savez_compressed(ur_dir / "stability_arrays.npz", **stab_arrays)

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
    def _load_bundle_data(cls, path: Path) -> "BasePlaceCellDataset":
        """Load bundle data into a *cls* instance (no subclass auto-selection).

        Handles metadata validation, config loading, and restoration of all
        shared attributes.  Subclasses that override ``load_bundle`` should
        call this helper instead of the ``__func__`` workaround.
        """
        if not path.is_dir():
            raise FileNotFoundError(f"Bundle not found: {path}")

        # Metadata
        meta = json.loads((path / "metadata.json").read_text())
        if meta["version"] != _BUNDLE_VERSION:
            raise ValueError(
                f"Bundle version {meta['version']} is not supported "
                f"(expected {_BUNDLE_VERSION}). Legacy bundles must be rerun "
                f"through the pipeline to upgrade."
            )

        # Config
        cfg = AnalysisConfig.from_yaml(path / "config.yaml")
        data_cfg = None
        data_cfg_path = path / "data_config.yaml"
        if data_cfg_path.exists():
            data_cfg = BaseDataConfig.from_yaml(data_cfg_path)
        ds = cls(cfg, data_cfg=data_cfg)

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
            "canonical",
        ]
        for name in df_names:
            pq = path / f"{name}.parquet"
            if pq.exists():
                setattr(ds, name, pd.read_parquet(pq))

        # Derive event_index/event_place from the canonical table so
        # downstream consumers (notebook browser, regression tests) see
        # the same DataFrames as a freshly-run pipeline.
        if ds.canonical is not None:
            from placecell.temporal_alignment import (
                derive_event_place_from_canonical,
                filter_canonical_by_speed,
            )

            speed_col = "speed_1d" if "speed_1d" in ds.canonical.columns else "speed"
            unfiltered = filter_canonical_by_speed(
                ds.canonical,
                speed_column=speed_col,
                speed_threshold=0.0,
                drop_below_threshold=False,
            )
            ds.event_index = derive_event_place_from_canonical(
                unfiltered,
                position_columns=("x", "y"),
                speed_column=speed_col,
            )
            speed_threshold = cfg.behavior.speed_threshold if cfg.behavior is not None else 0.0
            extras = tuple(
                c for c in ("pos_1d", "arm_index", "direction") if c in ds.canonical.columns
            )
            filtered = filter_canonical_by_speed(
                ds.canonical,
                speed_column=speed_col,
                speed_threshold=speed_threshold,
            )
            if extras:
                filtered = filtered.dropna(subset=["pos_1d"]).reset_index(drop=True)
            ds.event_place = derive_event_place_from_canonical(
                filtered,
                position_columns=("x", "y"),
                speed_column=speed_col,
                extra_columns=extras,
            )

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

        # Auto-select subclass based on behavior type
        if cls is BasePlaceCellDataset:
            if not path.is_dir():
                raise FileNotFoundError(f"Bundle not found: {path}")
            cfg = AnalysisConfig.from_yaml(path / "config.yaml")
            if cfg.behavior and cfg.behavior.type == "maze":
                from placecell.dataset.maze import MazeDataset

                return MazeDataset.load_bundle(path)

            from placecell.dataset.arena import ArenaDataset

            return ArenaDataset._load_bundle_data(path)

        return cls._load_bundle_data(path)

    @staticmethod
    def _load_unit_results(ur_dir: Path) -> dict[int, "UnitResult"]:
        """Reconstruct unit_results from saved files."""
        scalar_fields = [
            "si",
            "p_val",
            "overall_rate",
            "event_count_rate",
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

        # Read stability splits (one row per (uid, split_index)).
        stab_arrays_by_key: dict[tuple[int, int, str], np.ndarray] = {}
        stab_arrays_path = ur_dir / "stability_arrays.npz"
        if stab_arrays_path.exists():
            stab_npz = np.load(stab_arrays_path)
            for key in stab_npz.files:
                uid_str, idx_str, field = key.split("_", 2)
                stab_arrays_by_key[(int(uid_str), int(idx_str), field)] = stab_npz[key]

        stability_by_uid: dict[int, list[StabilitySplitResult]] = {uid: [] for uid in unit_ids}
        stab_csv = ur_dir / "stability_scalars.csv"
        if stab_csv.exists():
            stab_df = pd.read_csv(stab_csv).sort_values(["unit_id", "split_index"])
            for _, row in stab_df.iterrows():
                uid = int(row["unit_id"])
                idx = int(row["split_index"])
                stability_by_uid[uid].append(
                    StabilitySplitResult(
                        n_split_blocks=int(row["n_split_blocks"]),
                        corr=float(row["corr"]),
                        fisher_z=float(row["fisher_z"]),
                        p_val=float(row["p_val"]),
                        shuffled_corrs=stab_arrays_by_key.get(
                            (uid, idx, "shuffled_corrs"), np.array([])
                        ),
                        rate_map_first=stab_arrays_by_key.get(
                            (uid, idx, "rate_map_first"), np.array([])
                        ),
                        rate_map_second=stab_arrays_by_key.get(
                            (uid, idx, "rate_map_second"), np.array([])
                        ),
                    )
                )

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
                rate_map_smoothed=ar.get("rate_map_smoothed", np.array([])),
                rate_map_raw=ar.get("rate_map_raw", np.array([])),
                si=float(sc["si"]),
                p_val=float(sc["p_val"]),
                overall_rate=float(sc["overall_rate"]),
                event_count_rate=float(sc["event_count_rate"]),
                shuffled_sis=ar.get("shuffled_sis", np.array([])),
                shuffled_rate_p95=ar.get("shuffled_rate_p95", np.array([])),
                stability_splits=stability_by_uid.get(uid, []),
                vis_data_above=ev.get("vis_data_above", pd.DataFrame()),
                unit_data=ev.get("unit_data", pd.DataFrame()),
                trace_data=ar.get("trace_data"),
                trace_times=ar.get("trace_times"),
            )
        return results
