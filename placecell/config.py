"""Configuration models loaded from YAML."""

from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from placecell.log import init_logger

logger = init_logger(__name__)

CONFIG_DIR = Path(__file__).parent / "config"


class OasisConfig(BaseModel):
    """OASIS AR(2) deconvolution parameters.

    ``g``, ``penalty``, ``s_min`` are passed directly to oasisAR2.
    ``baseline`` is applied before deconvolution.
    """

    g: tuple[float, float] = Field(
        ...,
        description="AR(2) coefficients (g1, g2).",
    )
    baseline: str | float = Field(
        "p10",
        description="'pXX' for percentile, or numeric value (0 = no baseline).",
    )
    penalty: float = Field(
        0.0,
        ge=0.0,
        description="Sparsity penalty (L0 norm).",
    )
    s_min: float = Field(
        0.0,
        ge=0.0,
        description="Minimum event size threshold.",
    )


class NeuralConfig(BaseModel):
    """Neural data paths and deconvolution settings."""

    fps: float = Field(..., description="Sampling rate (fps).")
    oasis: OasisConfig = Field(...)
    trace_name: str = Field("C", description="Zarr group name (e.g. 'C' or 'C_lp').")


class BaseSpatialMapConfig(BaseModel):
    """Shared spatial map config: occupancy, smoothing, shuffle testing, stability.

    Subclasses add approach-specific binning and place field parameters.
    """

    min_occupancy: float = Field(
        ...,
        ge=0.0,
        description="Minimum occupancy time (seconds) per bin.",
    )
    spatial_sigma: float = Field(
        ...,
        ge=0.0,
        description="Gaussian smoothing sigma (in bins).",
    )
    n_shuffles: int = Field(
        ...,
        ge=1,
        le=10000,
        description="Number of circular-shift shuffles.",
    )
    random_seed: int | None = Field(
        None,
        description="Random seed for reproducible shuffling.",
    )
    p_value_threshold: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="P-value threshold for SI and stability significance tests.",
    )
    min_shift_seconds: float = Field(
        20.0,
        ge=0.0,
        description="Minimum circular shift (seconds) for shuffle test.",
    )
    si_weight_mode: Literal["amplitude", "binary"] = Field(
        ...,
        description="'amplitude' uses event amplitudes, 'binary' uses event counts.",
    )
    stability_splits: list[int] = Field(
        default_factory=lambda: [2, 10],
        min_length=1,
        description=(
            "Block counts for stability tests. n=2 is a classic "
            "first-half/second-half split (sensitive to session-long "
            "drift); n>=4 interleaves odd/even blocks so within-session "
            "drift averages out. One shuffle test is run per entry and "
            "a cell is 'stable' only if every test passes "
            "(p < p_value_threshold). Default [2, 10] requires stability "
            "at both the session-wide and within-session timescales."
        ),
    )
    min_events: int = Field(
        0,
        ge=0,
        description=(
            "Minimum number of speed-filtered events required to run the "
            "significance and stability tests for a unit. Units with fewer "
            "events keep a rate map for inspection but receive p_val=1.0 so "
            "they cannot be classified as place cells. Set to 0 (default) "
            "to disable the gate; typical published calcium-imaging values "
            "are 20-50."
        ),
    )

    @model_validator(mode="after")
    def _validate_stability_splits(self) -> "BaseSpatialMapConfig":
        bad = [n for n in self.stability_splits if n < 2 or n > 100]
        if bad:
            raise ValueError(f"stability_splits entries must be in [2, 100]; got {bad}.")
        if len(set(self.stability_splits)) != len(self.stability_splits):
            raise ValueError(f"stability_splits must be unique; got {self.stability_splits}.")
        return self

    block_shift: float = Field(
        0.0,
        ge=0.0,
        lt=1.0,
        description="Block boundary shift as fraction of block width (0.0 to <1.0).",
    )


class SpatialMap2DConfig(BaseSpatialMapConfig):
    """2D arena spatial map settings."""

    bins: int = Field(
        ...,
        ge=5,
        le=200,
        description="Number of spatial bins per axis.",
    )
    event_threshold_sigma: float = Field(
        0.0,
        description="Event amplitude threshold in SDs above mean (trajectory plot only).",
    )
    place_field_threshold: float = Field(
        0.35,
        gt=0.0,
        lt=1.0,
        description="Fraction of peak rate for place field boundary.",
    )
    place_field_min_bins: int = Field(
        5,
        ge=1,
        description="Minimum contiguous bins for a place field.",
    )
    place_field_seed_percentile: float = Field(
        95.0,
        description="Percentile of shuffled rate maps for place field seed detection.",
    )


class SpatialMap1DConfig(BaseSpatialMapConfig):
    """1D maze spatial map settings."""

    bin_width_mm: float = Field(
        ...,
        gt=0.0,
        description="Bin width in mm.",
    )
    split_by_direction: bool = Field(
        True,
        description="Split each arm into forward/reverse segments.",
    )
    require_complete_traversal: bool = Field(
        False,
        description="Keep only room-to-room traversals, discard partial entries.",
    )


class ZoneDetectionConfig(BaseModel):
    """Parameters for the zone detection state machine."""

    arm_max_distance: float = Field(
        60.0,
        gt=0.0,
        description="Max distance (px) from arm centerline.",
    )
    min_confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for zone transitions.",
    )
    min_confidence_forbidden: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for forbidden (non-adjacent) transitions.",
    )
    min_seconds_same: float = Field(
        0.05,
        gt=0.0,
        description=(
            "Minimum dwell time (seconds) in the current zone before "
            "allowing a transition.  Converted to frames internally using "
            "the projection sampling rate."
        ),
    )
    min_seconds_forbidden: float = Field(
        0.15,
        gt=0.0,
        description=(
            "Minimum dwell time (seconds) of consecutive forbidden-zone "
            "evidence before the state machine accepts a forbidden "
            "transition.  Converted to frames internally using the "
            "projection sampling rate."
        ),
    )
    room_decay_power: float = Field(
        2.0,
        gt=0.0,
        description="Exponent for room boundary probability decay.",
    )
    arm_decay_power: float = Field(
        0.5,
        gt=0.0,
        description="Exponent for arm boundary probability decay.",
    )
    soft_boundary: bool = Field(
        True,
        description="Use fuzzy distance-based boundaries instead of hard cutoffs.",
    )
    hampel_window_frames: int = Field(
        7,
        ge=3,
        description=(
            "Hampel-filter window for raw position outlier removal "
            "applied before zone projection (centered, odd, >= 3)."
        ),
    )
    hampel_n_sigmas: float = Field(
        3.0,
        gt=0.0,
        description=(
            "Hampel-filter threshold in MAD-scaled standard deviations "
            "(3.0 ~ 99.7% Gaussian band)."
        ),
    )
    interpolate: int = Field(
        5,
        ge=1,
        description="Frame subsampling factor for video export.",
    )
    playback_speed: float = Field(
        10.0,
        gt=0.0,
        description="Playback speed multiplier for exported zone videos.",
    )


class BehaviorConfig(BaseModel):
    """Behavior filtering and spatial map selection."""

    type: Literal["arena", "maze"] = Field(
        ...,
        description="'arena' for 2D open-field, 'maze' for 1D arm analysis.",
    )
    speed_threshold: float = Field(
        ...,
        description="Minimum running speed (mm/s).",
    )
    speed_window_seconds: float = Field(
        0.25,
        gt=0.0,
        description=(
            "Centered window (seconds) for finite-difference speed "
            "computation.  Converted to frames internally using the "
            "trajectory sampling rate."
        ),
    )
    hampel_window_frames: int = Field(
        7,
        ge=3,
        description=(
            "Hampel-filter window for position outlier removal "
            "applied to the raw (x, y) trajectory in the arena pipeline. "
            "Maze datasets configure this in ZoneDetectionConfig instead."
        ),
    )
    hampel_n_sigmas: float = Field(
        3.0,
        gt=0.0,
        description=(
            "Hampel-filter threshold in MAD-scaled standard deviations "
            "(3.0 ~ 99.7% Gaussian band). Arena pipeline only."
        ),
    )
    spatial_map_2d: SpatialMap2DConfig | None = Field(
        None,
        description="Required when type='arena'.",
    )
    spatial_map_1d: SpatialMap1DConfig | None = Field(
        None,
        description="Required when type='maze'.",
    )

    @model_validator(mode="after")
    def _validate_type_fields(self) -> "BehaviorConfig":
        if self.type == "arena":
            if self.spatial_map_2d is None:
                raise ValueError("spatial_map_2d is required when type='arena'")
        elif self.type == "maze" and self.spatial_map_1d is None:
            raise ValueError("spatial_map_1d is required when type='maze'")
        return self


class BaseDataConfig(BaseModel):
    """Per-session data paths and calibration. Subclassed by arena/maze."""

    @model_validator(mode="before")
    @classmethod
    def _warn_unknown_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            known = set(cls.model_fields)
            for key in data:
                if key not in known:
                    logger.warning(
                        "BaseDataConfig: unknown key '%s' (valid: %s)",
                        key,
                        ", ".join(sorted(cls.model_fields)),
                    )
        return data

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ArenaDataConfig | MazeDataConfig":
        """Load from YAML, dispatching to the correct subclass via ``type``."""
        with open(path) as f:
            data = yaml.safe_load(f)
        t = data.get("type")
        if t == "maze":
            return MazeDataConfig(**data)
        elif t == "arena":
            return ArenaDataConfig(**data)
        else:
            raise ValueError(f"DataConfig requires 'type: arena' or 'type: maze', got: {t!r}")

    behavior_fps: float = Field(..., gt=0.0, description="Behavior camera fps.")
    neural_path: str = Field(..., description="Directory containing neural zarr files.")
    neural_timestamp: str = Field(..., description="Path to neural timestamp CSV.")
    behavior_position: str = Field(..., description="Path to behavior position CSV.")
    behavior_timestamp: str = Field(..., description="Path to behavior timestamp CSV.")
    behavior_video: str | None = Field(None, description="Path to behavior video file.")
    bodypart: str | None = Field(None, description="DLC bodypart name (e.g. 'LED').")
    overlay_frame_index: int = Field(
        1000,
        ge=0,
        description="Video frame index for overlay figures (clamped to video length).",
    )
    x_col: str = Field("x", description="X-axis column name in behavior CSV.")
    y_col: str = Field("y", description="Y-axis column name in behavior CSV.")
    config_override: dict[str, Any] | None = Field(
        None,
        description="Deep-merged overrides for AnalysisConfig.",
    )


class ArenaDataConfig(BaseDataConfig):
    """Arena (2D open-field) data config."""

    type: Literal["arena"] = "arena"
    arena_bounds: tuple[float, float, float, float] | None = Field(
        None,
        description="Arena bounding box in pixels: (x_min, x_max, y_min, y_max).",
    )
    arena_size_mm: tuple[float, float] | None = Field(
        None,
        description="Physical arena dimensions in mm: (width, height).",
    )
    camera_height_mm: float | None = Field(
        None, gt=0.0, description="Camera height above arena floor (mm)."
    )
    tracking_height_mm: float | None = Field(
        None, ge=0.0, description="Tracked point height above arena floor (mm)."
    )


class MazeDataConfig(BaseDataConfig):
    """Maze (1D arm) data config."""

    type: Literal["maze"] = "maze"
    behavior_graph: str | None = Field(
        None, description="Path to behavior graph YAML with zone polylines."
    )
    mm_per_pixel: float | None = Field(
        None, gt=0.0, description="Scale factor for graph coordinates to mm."
    )
    arm_order: list[str] | None = Field(
        None, description="Ordered arm zone names for 1D concatenation."
    )
    zone_column: str = Field("zone", description="Zone label column in behavior CSV.")
    arm_position_column: str = Field(
        "arm_position", description="Within-arm position column (0-1)."
    )
    zone_tracking: str | None = Field(
        None, description="Path to zone-detected tracking CSV (output of detect-zones)."
    )
    zone_connections: dict[str, list[str]] | None = Field(
        None,
        description="Zone adjacency graph. Example: {Room_1: [Arm_1, Arm_2]}",
    )
    zone_detection: ZoneDetectionConfig | None = Field(
        None, description="Zone detection parameters for detect-zones CLI."
    )


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in place."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def _validate_override_keys(
    model_cls: type[BaseModel],
    override: dict,
    path: str = "",
) -> None:
    """Warn on keys in *override* that don't match *model_cls* fields."""
    for key, val in override.items():
        if key not in model_cls.model_fields:
            logger.warning(
                "config_override: unknown key '%s' (valid: %s)",
                f"{path}{key}",
                ", ".join(model_cls.model_fields),
            )
            continue
        if isinstance(val, dict):
            ann = model_cls.model_fields[key].annotation
            if get_origin(ann) is Union:
                args = [a for a in get_args(ann) if a is not type(None)]
                if len(args) == 1:
                    ann = args[0]
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                _validate_override_keys(ann, val, f"{path}{key}.")


class AnalysisConfig(BaseModel):
    """Top-level analysis config combining neural and behavior settings."""

    model_config = ConfigDict(extra="forbid")

    neural: NeuralConfig
    behavior: BehaviorConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnalysisConfig":
        """Load from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Serialize to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)

    def with_data_overrides(self, data_cfg: BaseDataConfig) -> "AnalysisConfig":
        """Return a new config with ``data_cfg.config_override`` deep-merged in."""
        if not data_cfg.config_override:
            return self
        _validate_override_keys(type(self), data_cfg.config_override)
        for key, val in data_cfg.config_override.items():
            logger.info("Overriding %s = %s", key, val)
        base = self.model_dump()
        _deep_merge(base, data_cfg.config_override)
        return type(self)(**base)
