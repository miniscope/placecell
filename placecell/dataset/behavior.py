"""Behavior-only dataset (no neural recording).

Loads behavior tracking and, when configured, the zone-detected output of
``placecell detect-zones``. The neural pipeline (deconvolution, occupancy,
unit analysis) is not supported and the corresponding methods raise
``NotImplementedError``.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from placecell.config import BehaviorDataConfig
from placecell.dataset.base import BasePlaceCellDataset
from placecell.loaders import load_behavior_data
from placecell.log import init_logger

logger = init_logger(__name__)


def _behavior_only(method: str) -> str:
    return (
        f"BehaviorDataset does not support {method}() — this dataset has no "
        "neural recording. Use ArenaDataset or MazeDataset for the neural pipeline."
    )


class BehaviorDataset(BasePlaceCellDataset):
    """Dataset for behavior-only sessions (no neural data).

    Provides ``load()`` for the behavior trajectory (and zone tracking when
    available). Neural-pipeline methods raise ``NotImplementedError``.
    """

    @classmethod
    def from_data_config(cls, data_path: str | Path) -> "BehaviorDataset":
        """Construct a ``BehaviorDataset`` from a data config YAML alone.

        No analysis config is required because the neural pipeline is not
        used. Resolves all paths relative to the data config directory.
        """
        from placecell.config import BaseDataConfig

        data_path = Path(data_path)
        data_dir = data_path.parent
        data_cfg = BaseDataConfig.from_yaml(data_path)
        if not isinstance(data_cfg, BehaviorDataConfig):
            raise ValueError(
                f"BehaviorDataset.from_data_config requires 'type: behavior'; "
                f"got '{getattr(data_cfg, 'type', None)!r}'"
            )

        return cls(
            cfg=None,
            behavior_position_path=data_dir / data_cfg.behavior_position,
            behavior_timestamp_path=data_dir / data_cfg.behavior_timestamp,
            behavior_video_path=(
                data_dir / data_cfg.behavior_video if data_cfg.behavior_video else None
            ),
            behavior_graph_path=(
                data_dir / data_cfg.behavior_graph if data_cfg.behavior_graph else None
            ),
            zone_tracking_path=(
                data_dir / (data_cfg.zone_tracking or f"zone_tracking_{data_path.stem}.csv")
            ),
            data_cfg=data_cfg,
        )

    @property
    def p_value_threshold(self) -> float:
        """Not available for behavior-only datasets."""
        raise NotImplementedError(_behavior_only("p_value_threshold"))

    def load(self) -> None:
        """Load behavior trajectory and, when present, the zone tracking CSV.

        If ``zone_tracking`` exists, it is read (one row per behavior frame
        with ``x, y, zone, arm_position, neural_time`` columns). Otherwise
        the raw ``behavior_position`` CSV is loaded.
        """
        self._load_behavior_video_frame()

        dcfg = self.data_cfg
        if dcfg is None or dcfg.bodypart is None:
            raise RuntimeError("bodypart must be set in data config")

        zone_csv = self.zone_tracking_path
        if zone_csv is not None and zone_csv.exists():
            self._load_zone_tracking(zone_csv, dcfg)
        else:
            self.trajectory = load_behavior_data(
                behavior_position=self.behavior_position_path,
                behavior_timestamp=self.behavior_timestamp_path,
                bodypart=dcfg.bodypart,
                x_col=dcfg.x_col,
                y_col=dcfg.y_col,
            )
            logger.info(
                "Loaded raw behavior trajectory: %d frames (no zone tracking)",
                len(self.trajectory),
            )

    def _load_zone_tracking(self, zone_csv: Path, dcfg: BehaviorDataConfig) -> None:
        df = pd.read_csv(zone_csv, header=[0, 1, 2])
        scorer = df.columns[1][0]
        bp = dcfg.bodypart
        zone_col = dcfg.zone_column
        ap_col = dcfg.arm_position_column

        frame_index = df.iloc[:, 0].values.astype(np.int64)
        x = df[(scorer, bp, "x")].values.astype(float)
        y = df[(scorer, bp, "y")].values.astype(float)
        zone = df[(scorer, bp, zone_col)].values
        unix_time = df[(scorer, bp, "neural_time")].values.astype(float)

        cols: dict[str, Any] = {
            "frame_index": frame_index,
            "x": x,
            "y": y,
            "unix_time": unix_time,
            zone_col: zone,
        }
        if (scorer, bp, ap_col) in df.columns:
            cols[ap_col] = pd.to_numeric(df[(scorer, bp, ap_col)].values, errors="coerce")

        self.trajectory = pd.DataFrame(cols)
        logger.info(
            "Loaded zone tracking: %d frames, %d zones",
            len(self.trajectory),
            int(pd.Series(zone).nunique()),
        )

    def preprocess_behavior(self) -> None:
        """No-op for behavior-only datasets; trajectory is used as loaded."""
        if self.trajectory is None:
            raise RuntimeError("Call load() first.")

    def deconvolve(self, progress_bar: Any = None) -> None:
        """Not available for behavior-only datasets."""
        raise NotImplementedError(_behavior_only("deconvolve"))

    def match_events(self) -> None:
        """Not available for behavior-only datasets."""
        raise NotImplementedError(_behavior_only("match_events"))

    def compute_occupancy(self) -> None:
        """Not available for behavior-only datasets."""
        raise NotImplementedError(_behavior_only("compute_occupancy"))

    def analyze_units(self, progress_bar: Any = None) -> None:
        """Not available for behavior-only datasets."""
        raise NotImplementedError(_behavior_only("analyze_units"))
