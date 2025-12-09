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


class NeuralConfig(MiniscopeConfig, ConfigYAMLMixin):
    """Neural data configuration."""

    data: DataConfig
    lpf: LpfConfig = Field(default_factory=LpfConfig)
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


class AppConfig(MiniscopeConfig, ConfigYAMLMixin):
    """Top-level application configuration."""

    neural: NeuralConfig
    behavior: BehaviorConfig | None = None
