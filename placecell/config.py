"""Configuration models for pcell, loaded from YAML."""

from mio.models import MiniscopeConfig
from mio.models.mixins import ConfigYAMLMixin
from pydantic import Field


class LpfConfig(MiniscopeConfig, ConfigYAMLMixin):
    """Low-pass filter configuration."""

    enabled: bool = Field(
        False,
        description="Whether to apply low-pass filtering to traces before visualization.",
    )
    cutoff_hz: float = Field(
        1.0,
        description="Cutoff frequency for the Butterworth low-pass filter (Hz).",
    )
    order: int = Field(
        4,
        ge=1,
        description="Filter order for the Butterworth low-pass filter.",
    )


class DataConfig(MiniscopeConfig, ConfigYAMLMixin):
    """Source data configuration (Minian output)."""

    fps: float = Field(
        20.0,
        description="Sampling rate in frames per second.",
    )


class OasisConfig(MiniscopeConfig, ConfigYAMLMixin):
    """OASIS deconvolution parameters."""

    g: tuple[float, float] | None = Field(
        None,
        description="AR(2) coefficients (g1, g2). If None, estimated from data.",
    )
    baseline: str = Field(
        "p10",
        description="Baseline mode: 'pXX' for percentile or numeric value.",
    )
    penalty: float = Field(
        0.0,
        ge=0.0,
        description="Sparsity penalty (L0 norm). Default 0 (no penalty).",
    )
    optimize_g: int = Field(
        0,
        ge=0,
        description=[
            "Number of events to use for optimizing AR coefficients."
            "Look into oasis documentation for more details."
        ],
    )
    lambda_: float | None = Field(
        None,
        description=["Regularization parameter for OASIS." "If None, automatically determined."],
    )
    s_min: float | None = Field(
        None,
        description=[
            "Minimum spike size threshold." "Look into oasis documentation for more details."
        ],
    )


class NeuralConfig(MiniscopeConfig, ConfigYAMLMixin):
    """Neural data configuration."""

    data: DataConfig
    lpf: LpfConfig = Field(default_factory=LpfConfig)
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
    spike_threshold_sigma: float = Field(
        ...,
        description="Sigma multiplier for spike amplitude threshold in trajectory plot. "
        "Can be negative to include lower-amplitude spikes.",
    )
    p_value_threshold: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "P-value threshold for filtering units in visualization. "
            "Only units with p-value < threshold are plotted. If None, plot all units."
        ),
    )


class BehaviorConfig(MiniscopeConfig, ConfigYAMLMixin):
    """Behavior / place-field configuration."""

    behavior_fps: float = Field(
        ...,
        gt=0.0,
        description=(
            "Behavior data sampling rate in frames per second. "
            "Required for spike-place matching."
        ),
    )
    speed_threshold: float = Field(
        50.0,
        description="Minimum running speed to keep spikes (pixels/s or cm/s).",
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


class AppConfig(MiniscopeConfig, ConfigYAMLMixin):
    """Top-level application configuration."""

    neural: NeuralConfig
    behavior: BehaviorConfig | None = None
