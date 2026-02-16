"""Configuration models for pcell, loaded from YAML."""

from pathlib import Path
from typing import Literal

import yaml
from mio.models import MiniscopeConfig
from mio.models.mixins import ConfigYAMLMixin
from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_DIR = Path(__file__).parent / "config"


class OasisConfig(BaseModel):
    """OASIS deconvolution parameters."""

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
    oasis: OasisConfig = Field(default_factory=OasisConfig)
    trace_name: str = Field(
        "C",
        description="Base name of the zarr group (e.g. 'C' or 'C_lp').",
    )


class SpatialMap2DConfig(BaseModel):
    """Spatial map visualization configuration for 2D arena analysis."""

    bins: int = Field(
        ...,
        ge=5,
        le=200,
        description="Number of spatial bins.",
    )
    min_occupancy: float = Field(
        ...,
        ge=0.0,
        description="Minimum occupancy time (seconds) for a bin to be included.",
    )
    occupancy_sigma: float = Field(
        ...,
        ge=0.0,
        description="Gaussian smoothing sigma (in bins) for the occupancy map. "
        "Smoothing reduces noise from undersampled bins. Use 0 for no smoothing.",
    )
    activity_sigma: float = Field(
        ...,
        ge=0.0,
        description="Gaussian smoothing sigma (in bins) for the spatial activity map. "
        "Use 0 for no smoothing.",
    )
    n_shuffles: int = Field(
        ...,
        ge=1,
        le=10000,
        description="Number of shuffles for spatial information significance test.",
    )
    random_seed: int | None = Field(
        None,
        description="Random seed for reproducible shuffling. If None, results vary between runs.",
    )
    event_threshold_sigma: float = Field(
        0.0,
        description="Sigma multiplier for event amplitude threshold in trajectory plot. "
        "Can be negative to include lower-amplitude events.",
    )
    p_value_threshold: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description=(
            "P-value threshold for significance test pass/fail. "
            "Units with p-value < threshold pass."
        ),
    )
    min_shift_seconds: float = Field(
        20.0,
        ge=0.0,
        description=(
            "Minimum circular shift in seconds for shuffle significance test. "
            "Ensures shuffled data breaks the temporal-spatial association. "
            "Set to 0 to allow any shift size (original behavior)."
        ),
    )
    si_weight_mode: str = Field(
        "binary",
        description=(
            "Weight mode for spatial information calculation: "
            "'amplitude' uses event amplitudes (s values), "
            "'binary' uses event counts (1 per event, ignoring amplitude). "
            "Binary mode is more robust to bursty firing patterns."
        ),
    )
    place_field_threshold: float = Field(
        0.05,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of peak rate to define the place field boundary "
            "(red contour on rate maps and coverage analysis). "
            "Applied to the smoothed, normalized rate map. "
            "E.g. 0.05 means bins >= 5%% of peak are inside the field."
        ),
    )
    place_field_min_bins: int = Field(
        5,
        ge=1,
        description=(
            "Minimum number of contiguous bins for a connected component "
            "to count as a place field (Guo et al. 2023). Smaller "
            "disconnected regions are discarded. Set to 1 to disable."
        ),
    )
    place_field_seed_percentile: float = Field(
        95.0,
        description=(
            "Percentile of shuffled rate maps for place field seed detection "
            "(Guo et al. 2023). Bins exceeding this percentile form seeds; "
            "seeds extend to contiguous bins above place_field_threshold."
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
    trace_time_window: float = Field(
        600.0,
        gt=0.0,
        description="Time window in seconds for trace display in the interactive browser.",
    )


class MazeConfig(BaseModel):
    """Configuration for maze/tube-based 1D analysis."""

    tube_order: list[str] = Field(
        ["Tube_1", "Tube_2", "Tube_3", "Tube_4"],
        description="Ordered list of tube zone names. Determines concatenation order on 1D axis.",
    )
    zone_column: str = Field(
        "zone",
        description="Column name in behavior CSV containing zone labels.",
    )
    tube_position_column: str = Field(
        "tube_position",
        description="Column name in behavior CSV containing within-tube position (0-1).",
    )
    split_by_direction: bool = Field(
        True,
        description="Split each tube into forward/reverse segments based on traversal direction. "
        "Doubles total segments (e.g. 4 tubes -> 8 directional segments).",
    )


class SpatialMap1DConfig(BaseModel):
    """Spatial map settings for 1D tube analysis."""

    bin_width_mm: float = Field(
        10.0,
        gt=0.0,
        description="Bin width in mm. Total bins = round(total_length / bin_width_mm).",
    )
    min_occupancy: float = Field(
        0.025,
        ge=0.0,
        description="Minimum occupancy time (seconds) for a bin to be included.",
    )
    occupancy_sigma: float = Field(
        2.0,
        ge=0.0,
        description="Gaussian smoothing sigma (in bins) for the 1D occupancy histogram.",
    )
    activity_sigma: float = Field(
        2.0,
        ge=0.0,
        description="Gaussian smoothing sigma (in bins) for the 1D rate map.",
    )
    n_shuffles: int = Field(
        1000,
        ge=1,
        le=10000,
        description="Number of shuffles for significance test.",
    )
    random_seed: int | None = Field(
        None,
        description="Random seed for reproducible shuffling.",
    )
    p_value_threshold: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="P-value threshold for significance test.",
    )
    min_shift_seconds: float = Field(
        20.0,
        ge=0.0,
        description="Minimum circular shift in seconds for shuffle test.",
    )
    si_weight_mode: str = Field(
        "amplitude",
        description="Weight mode for spatial information: 'amplitude' or 'binary'.",
    )
    n_split_blocks: int = Field(
        10,
        ge=2,
        le=100,
        description="Number of temporal blocks for stability splitting.",
    )
    block_shifts: list[float] = Field(
        [0.0],
        description="Block boundary shifts as fractions of one block width.",
    )
    trace_time_window: float = Field(
        600.0,
        gt=0.0,
        description="Time window in seconds for trace display.",
    )


class BehaviorConfig(BaseModel):
    """Behavior / place-field configuration."""

    type: Literal["arena", "maze"] = Field(
        "arena",
        description="Analysis type: 'arena' for 2D open-field, 'maze' for 1D tube analysis.",
    )
    behavior_fps: float = Field(
        ...,
        gt=0.0,
        description=(
            "Behavior data sampling rate in frames per second. "
            "Required for event-place matching."
        ),
    )
    speed_threshold: float = Field(
        50.0,
        description="Minimum running speed to keep events (mm/s).",
    )
    speed_window_frames: int = Field(
        5,
        description="Number of frames to use for speed calculation window. "
        "Larger values give more stable speed estimates but less temporal resolution. "
        "Default 5 frames (0.25s at 20 fps).",
    )
    bodypart: str = Field(
        ...,
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
    maze: MazeConfig | None = Field(
        None,
        description="Maze configuration for 1D tube analysis. Required when type='maze'.",
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
        elif self.type == "maze":
            if self.maze is None:
                raise ValueError("maze is required when type='maze'")
            if self.spatial_map_1d is None:
                raise ValueError("spatial_map_1d is required when type='maze'")
        return self


class DataPathsConfig(BaseModel):
    """Bundle of data file paths for neural and behavior data."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataPathsConfig":
        """Load from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

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
        description="Path to behavior graph YAML with zone polylines and mm_per_pixel.",
    )
    oasis: OasisConfig | None = Field(
        None,
        description=(
            "Optional OASIS parameters to override main config for this dataset. "
            "If provided, these values override the main config's neural.oasis settings."
        ),
    )


class _PlacecellConfigMixin(ConfigYAMLMixin):
    """Override config sources to include placecell bundled configs."""

    @classmethod
    def config_sources(cls) -> list[Path]:
        from mio import CONFIG_DIR as MIO_CONFIG_DIR
        from mio import Config

        return [Config().config_dir, CONFIG_DIR, MIO_CONFIG_DIR]


class AnalysisConfig(MiniscopeConfig, _PlacecellConfigMixin):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="ignore")

    neural: NeuralConfig
    behavior: BehaviorConfig | None = None

    def with_data_overrides(self, data_cfg: DataPathsConfig) -> "AnalysisConfig":
        """Create a new config with data-specific overrides applied.

        Parameters
        ----------
        data_cfg : DataPathsConfig
            Data configuration that may contain override values.

        Returns
        -------
        AnalysisConfig
            New config with overrides applied. Original config is unchanged.
        """
        if data_cfg.oasis is not None:
            new_neural = self.neural.model_copy(update={"oasis": data_cfg.oasis})
            return self.model_copy(update={"neural": new_neural})
        return self
