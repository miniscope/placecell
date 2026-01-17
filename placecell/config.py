"""Configuration models for pcell, loaded from YAML."""

from mio.models import MiniscopeConfig
from mio.models.mixins import ConfigYAMLMixin
from pydantic import Field


class OasisConfig(MiniscopeConfig, ConfigYAMLMixin):
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


class NeuralConfig(MiniscopeConfig, ConfigYAMLMixin):
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
    max_units: int | None = Field(
        None,
        ge=1,
        description=(
            "Maximum number of units to include in a visualization. "
            "If omitted/null, use all available units unless a CLI max-units "
            "override is provided."
        ),
    )


class SpatialMapConfig(MiniscopeConfig, ConfigYAMLMixin):
    """Spatial map visualization configuration."""

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
        ...,
        description="Sigma multiplier for event amplitude threshold in trajectory plot. "
        "Can be negative to include lower-amplitude events.",
    )
    p_value_threshold: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "P-value threshold for significance test pass/fail. "
            "Units with p-value < threshold pass. Default 0.05 if None."
        ),
    )
    stability_threshold: float = Field(
        0.5,
        ge=-1.0,
        le=1.0,
        description=(
            "Correlation threshold for stability test pass/fail. "
            "Units with first/second half rate map correlation >= threshold pass."
        ),
    )


class BehaviorConfig(MiniscopeConfig, ConfigYAMLMixin):
    """Behavior / place-field configuration."""

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
        description="Minimum running speed to keep events (pixels/s or cm/s).",
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
    spatial_map: SpatialMapConfig = Field(
        default_factory=SpatialMapConfig,
        description="Spatial map visualization settings.",
    )


class DataPathsConfig(MiniscopeConfig, ConfigYAMLMixin):
    """Bundle of data file paths for neural and behavior data."""

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
    curation_csv: str | None = Field(
        None,
        description=(
            "Path to curation results CSV file with columns 'unit_id' and 'keep'. "
            "Only units with keep=1 will be processed. If None, all units are used."
        ),
    )


class AppConfig(MiniscopeConfig, ConfigYAMLMixin):
    """Top-level application configuration."""

    neural: NeuralConfig
    behavior: BehaviorConfig | None = None
