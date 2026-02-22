"""Configuration models for pcell, loaded from YAML."""

from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

import yaml
from mio.models import MiniscopeConfig
from mio.models.mixins import ConfigYAMLMixin
from pydantic import BaseModel, ConfigDict, Field, model_validator

from placecell.log import init_logger

logger = init_logger(__name__)

CONFIG_DIR = Path(__file__).parent / "config"


class OasisConfig(BaseModel):
    """OASIS deconvolution parameters.
    `g, penalty, s_min` are directly passed to the oasisAR2 deconvolution function.
    `baseline` is applied before deconvolution.
    """

    g: tuple[float, float] = Field(
        ...,
        description="AR(2) coefficients (g1, g2). Required - must be provided in config.",
    )
    baseline: str | float = Field(
        "p10",
        description="Baseline mode: 'pXX' for percentile, or numeric value (0 = no baseline).",
    )
    penalty: float = Field(
        0.0,
        ge=0.0,
        description="Sparsity penalty (L0 norm). Default 0 (no penalty).",
    )
    s_min: float = Field(
        0.0,
        ge=0.0,
        description="Minimum event size threshold. Default 0 (no minimum).",
    )


class NeuralConfig(BaseModel):
    """Neural data configuration."""

    fps: float = Field(
        20.0,
        description="Sampling rate in frames per second.",
    )
    oasis: OasisConfig = Field(..., description="OASIS deconvolution parameters.")
    trace_name: str = Field(
        "C",
        description="Base name of the zarr group (e.g. 'C' or 'C_lp').",
    )


class SpatialMapConfig(BaseModel):
    """Base spatial map configuration shared by all analysis approaches.

    Defines the common analysis contract: occupancy filtering, smoothing,
    shuffle-based significance testing, and stability splitting.
    Subclasses add approach-specific binning and place field parameters.
    """

    min_occupancy: float = Field(
        ...,
        ge=0.0,
        description="Minimum occupancy time (seconds) for a bin to be included.",
    )
    spatial_sigma: float = Field(
        ...,
        ge=0.0,
        description="Gaussian smoothing sigma (in bins) for occupancy and rate maps.",
    )
    n_shuffles: int = Field(
        ...,
        ge=1,
        le=10000,
        description="Number of circular-shift shuffles for significance testing.",
    )
    random_seed: int | None = Field(
        None,
        description="Random seed for reproducible shuffling. If None, results vary between runs.",
    )
    p_value_threshold: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description=(
            "P-value threshold for shuffle-based significance tests "
            "(spatial information and stability). "
            "Units with p-value < threshold pass."
        ),
    )
    min_shift_seconds: float = Field(
        20.0,
        ge=0.0,
        description=(
            "Minimum circular shift in seconds for shuffle significance test. "
            "Ensures shuffled data breaks the temporal-spatial association. "
            "Set to 0 to allow any shift size."
        ),
    )
    si_weight_mode: Literal["amplitude", "binary"] = Field(
        ...,
        description=(
            "Weight mode for spatial information calculation: "
            "'amplitude' uses event amplitudes (s values), "
            "'binary' uses event counts (1 per event, ignoring amplitude)."
        ),
    )
    n_split_blocks: int = Field(
        10,
        ge=2,
        le=100,
        description=(
            "Number of temporal blocks for interleaved stability splitting. "
            "The session is divided into this many equal-duration blocks, "
            "and odd/even blocks are assigned to each half."
        ),
    )
    block_shifts: list[float] = Field(
        [0.0],
        description=(
            "List of block boundary shifts as fractions of one block width. "
            "Each value produces an independent split; results are Fisher "
            "z-averaged. Use [0] for a single split, [0, 0.5] for two "
            "shifted arrangements, etc. Values are circular with period 1.0."
        ),
    )


class SpatialMap2DConfig(SpatialMapConfig):
    """Spatial map configuration for 2D arena analysis."""

    bins: int = Field(
        ...,
        ge=5,
        le=200,
        description="Number of spatial bins per axis.",
    )
    event_threshold_sigma: float = Field(
        0.0,
        description="Event amplitude threshold as number of standard deviations above the mean "
        "(threshold = mean + sigma * SD). Only affects trajectory plot visualization. "
        "Can be negative to include lower-amplitude events.",
    )
    place_field_threshold: float = Field(
        0.35,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of peak rate to define the place field boundary "
            "(red contour on rate maps and coverage analysis). "
            "Applied to the smoothed, normalized rate map. "
            "E.g. 0.35 means bins >= 35%% of peak are inside the field."
        ),
    )
    place_field_min_bins: int = Field(
        5,
        ge=1,
        description=(
            "Minimum number of contiguous bins for a connected componentto count as a place field. "
            "Smaller disconnected regions are discarded. Set to 1 to disable."
        ),
    )
    place_field_seed_percentile: float = Field(
        95.0,
        description=(
            "Percentile of shuffled rate maps for place field seed detection. "
            "Bins exceeding this percentile form seeds; "
            "seeds extend to contiguous bins above place_field_threshold."
        ),
    )


class SpatialMap1DConfig(SpatialMapConfig):
    """Spatial map configuration for 1D arm analysis."""

    bin_width_mm: float = Field(
        ...,
        gt=0.0,
        description="Bin width in mm. Total bins = round(total_length / bin_width_mm).",
    )
    split_by_direction: bool = Field(
        True,
        description="Split each arm into forward/reverse segments based on traversal direction. "
        "Doubles total segments (e.g. 4 arms -> 8 directional segments).",
    )
    require_complete_traversal: bool = Field(
        False,
        description="If True, keep only traversals where the animal crosses "
        "from one room to a different room. Partial entries (animal enters "
        "an arm and returns to the same room) are discarded.",
    )


class ZoneDetectionConfig(BaseModel):
    """Parameters for zone detection state machine."""

    arm_max_distance: float = Field(
        60.0,
        gt=0.0,
        description="Maximum distance (pixels) from arm centerline for zone classification.",
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
    min_frames_same: int = Field(
        1,
        ge=1,
        description="Minimum frames in current zone before allowing transition.",
    )
    min_frames_forbidden: int = Field(
        3,
        ge=1,
        description="Minimum consecutive frames for forbidden transition override.",
    )
    room_decay_power: float = Field(
        2.0,
        gt=0.0,
        description=(
            "Exponent for room boundary probability decay. "
            "Controls how quickly room probability drops near edges. "
            "Higher = steeper drop-off."
        ),
    )
    arm_decay_power: float = Field(
        0.5,
        gt=0.0,
        description=(
            "Exponent for arm boundary probability decay. "
            "Controls how quickly arm probability drops with distance from centerline. "
            "Lower = more fuzzy/gradual."
        ),
    )
    soft_boundary: bool = Field(
        True,
        description="Use fuzzy distance-based boundaries instead of hard inside/outside.",
    )
    interpolate: int = Field(
        5,
        ge=1,
        description="Frame subsampling factor for video export.",
    )


class BehaviorConfig(BaseModel):
    """Behavior / place-field configuration."""

    type: Literal["arena", "maze"] = Field(
        ...,
        description="Analysis type: 'arena' for 2D open-field, 'maze' for 1D arm analysis.",
    )
    speed_threshold: float = Field(
        25.0,
        description="Minimum running speed to keep events (mm/s).",
    )
    speed_window_frames: int = Field(
        5,
        description="Number of frames to use for speed calculation window. "
        "Larger values give more stable speed estimates but less temporal resolution. "
        "Default 5 frames (0.25s at 20 fps).",
    )
    jump_threshold_mm: float = Field(
        100.0,
        gt=0.0,
        description="Maximum plausible frame-to-frame displacement in mm. "
        "Larger jumps are treated as tracking errors and interpolated.",
    )
    spatial_map_2d: SpatialMap2DConfig | None = Field(
        None,
        description="Spatial map settings for 2D arena analysis. Required when type='arena'.",
    )
    spatial_map_1d: SpatialMap1DConfig | None = Field(
        None,
        description="Spatial map settings for 1D analysis. Required when type='maze'.",
    )

    @model_validator(mode="after")
    def _validate_type_fields(self) -> "BehaviorConfig":
        if self.type == "arena":
            if self.spatial_map_2d is None:
                raise ValueError("spatial_map_2d is required when type='arena'")
        elif self.type == "maze" and self.spatial_map_1d is None:
            raise ValueError("spatial_map_1d is required when type='maze'")
        return self


_MIO_METADATA_KEYS = frozenset({"id", "mio_model", "mio_version"})


class DataConfig(BaseModel):
    """Per-session data configuration (file paths, calibration, overrides)."""

    @model_validator(mode="before")
    @classmethod
    def _warn_unknown_keys(cls, data: Any) -> Any:
        """Warn on unrecognised keys (catches typos like ``config_ovrride``)."""
        if isinstance(data, dict):
            known = set(cls.model_fields) | _MIO_METADATA_KEYS
            for key in data:
                if key not in known:
                    logger.warning(
                        "DataConfig: unknown key '%s' (valid: %s)",
                        key,
                        ", ".join(sorted(cls.model_fields)),
                    )
        return data

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataConfig":
        """Load from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    behavior_fps: float = Field(
        ...,
        gt=0.0,
        description=(
            "Behavior data sampling rate in frames per second. "
            "Property of the recording setup, used for event-place matching."
        ),
    )
    neural_path: str = Field(
        ...,
        description="Directory containing neural data (C.zarr, max_proj.zarr, A.zarr).",
    )
    neural_timestamp: str = Field(
        ...,
        description="Path to neural timestamp CSV file (neural_timestamp.csv).",
    )
    behavior_position: str = Field(
        ...,
        description="Path to behavior position CSV file (behavior_position.csv).",
    )
    behavior_timestamp: str = Field(
        ...,
        description="Path to behavior timestamp CSV file (behavior_timestamp.csv).",
    )
    behavior_video: str | None = Field(
        None,
        description="Path to behavior video file (e.g. .mp4). Used for arena bounds verification.",
    )
    arena_bounds: tuple[float, float, float, float] | None = Field(
        None,
        description=(
            "Arena bounding box in pixels: (x_min, x_max, y_min, y_max). "
            "Used for perspective correction center and boundary clipping. "
            "If None, no preprocessing (perspective/clipping) is applied."
        ),
    )
    arena_size_mm: tuple[float, float] | None = Field(
        None,
        description=(
            "Physical arena dimensions in mm: (width, height). "
            "Required when arena_bounds is set. Used to derive mm/pixel scale."
        ),
    )
    camera_height_mm: float | None = Field(
        None,
        gt=0.0,
        description="Camera height above arena floor in mm. Required when arena_bounds is set.",
    )
    tracking_height_mm: float | None = Field(
        None,
        ge=0.0,
        description="Height of tracked point (e.g. LED) above arena floor in mm. "
        "Required when arena_bounds is set.",
    )
    behavior_graph: str | None = Field(
        None,
        description="Path to behavior graph YAML with zone polylines.",
    )
    mm_per_pixel: float | None = Field(
        None,
        gt=0.0,
        description="Scale factor (mm per pixel) for converting graph coordinates to mm.",
    )
    bodypart: str | None = Field(
        None,
        description="Body part name to use for position tracking (e.g. 'LED').",
    )
    x_col: str = Field(
        "x",
        description="Coordinate column name for the x-axis in the behavior CSV.",
    )
    y_col: str = Field(
        "y",
        description="Coordinate column name for the y-axis in the behavior CSV.",
    )
    arm_order: list[str] | None = Field(
        None,
        description="Ordered list of arm zone names for maze analysis. "
        "Determines concatenation order on 1D axis.",
    )
    zone_column: str = Field(
        "zone",
        description="Column name in behavior CSV containing zone labels.",
    )
    arm_position_column: str = Field(
        "arm_position",
        description="Column name in behavior CSV containing within-arm position (0-1).",
    )
    zone_tracking: str | None = Field(
        None,
        description="Path to zone-detected tracking CSV (output of detect-zones). "
        "Contains zone, x_pinned, y_pinned, arm_position columns. "
        "Read by the maze analysis pipeline for 1D serialization.",
    )
    zone_connections: dict[str, list[str]] | None = Field(
        None,
        description="Zone adjacency graph mapping each room to its connected arms. "
        "Defines which transitions are legal vs forbidden. "
        "Example: {Room_1: [Arm_1, Arm_2], Room_2: [Arm_2, Arm_3]}",
    )
    zone_detection: ZoneDetectionConfig | None = Field(
        None,
        description="Zone detection algorithm parameters. "
        "Required for the detect-zones CLI command.",
    )
    config_override: dict[str, Any] | None = Field(
        None,
        description=(
            "Arbitrary overrides for the analysis config. "
            "Keys mirror the AnalysisConfig structure and are deep-merged. "
            "Example: {neural: {oasis: {penalty: 0.5}}, behavior: {speed_threshold: 30}}"
        ),
    )


class _PlacecellConfigMixin(ConfigYAMLMixin):
    """Override config sources to include placecell bundled configs."""

    @classmethod
    def config_sources(cls) -> list[Path]:
        from mio import CONFIG_DIR as MIO_CONFIG_DIR
        from mio import Config

        return [Config().config_dir, CONFIG_DIR, MIO_CONFIG_DIR]


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
            # Unwrap Optional[X] / X | None to get the inner model type
            ann = model_cls.model_fields[key].annotation
            if get_origin(ann) is Union:
                args = [a for a in get_args(ann) if a is not type(None)]
                if len(args) == 1:
                    ann = args[0]
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                _validate_override_keys(ann, val, f"{path}{key}.")


class AnalysisConfig(MiniscopeConfig, _PlacecellConfigMixin):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="ignore")

    neural: NeuralConfig
    behavior: BehaviorConfig

    def with_data_overrides(self, data_cfg: DataConfig) -> "AnalysisConfig":
        """Create a new config with data-specific overrides applied.

        Parameters
        ----------
        data_cfg : DataConfig
            Data configuration that may contain a ``config_override`` dict.

        Returns
        -------
        AnalysisConfig
            New config with overrides deep-merged. Original is unchanged.
        """
        if not data_cfg.config_override:
            return self
        _validate_override_keys(type(self), data_cfg.config_override)
        for key, val in data_cfg.config_override.items():
            logger.info("Overriding %s = %s", key, val)
        base = self.model_dump()
        _deep_merge(base, data_cfg.config_override)
        return type(self)(**base)
