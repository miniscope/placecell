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
    s_min: float = Field(
        1.0,
        description="Minimum spike size.",
    )
    baseline: str = Field(
        "p10",
        description="Baseline mode: 'pXX' for percentile or numeric value.",
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
    s_threshold: float = Field(
        0.0,
        description="Minimum spike amplitude s to visualize in place browser.",
    )


class RateMapConfig(MiniscopeConfig, ConfigYAMLMixin):
    """Rate map visualization configuration."""

    bins: int = Field(
        30,
        ge=5,
        le=200,
        description="Number of spatial bins for rate map (default 30).",
    )
    min_occupancy: float = Field(
        0.1,
        ge=0.0,
        description="Minimum occupancy time (seconds) for a bin to be included (default 0.1).",
    )
    smooth_sigma: float = Field(
        1.0,
        ge=0.0,
        description="Gaussian smoothing sigma for rate map (default 1.0, 0 = no smoothing).",
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
    ratemap: RateMapConfig = Field(
        default_factory=RateMapConfig,
        description="Rate map visualization settings.",
    )


class AppConfig(MiniscopeConfig, ConfigYAMLMixin):
    """Top-level application configuration."""

    neural: NeuralConfig
    behavior: BehaviorConfig | None = None
