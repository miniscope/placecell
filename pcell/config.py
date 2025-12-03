"""
Configuration models for pcell, loaded from YAML.

..todo::
    - Update to config system like mio package
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError


class LpfConfig(BaseModel):
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


class DataConfig(BaseModel):
    """Source data configuration (Minian output)."""

    minian_path: Path = Field(
        ...,
        description="Directory containing <trace_name>.zarr.",
    )
    trace_name: str = Field(
        "C",
        description="Base name of the zarr group (e.g. 'C' or 'C_lp').",
    )
    var_name: str = Field(
        "C",
        description="Variable name inside the Dataset, if any.",
    )
    fps: float = Field(
        20.0,
        description="Sampling rate in frames per second.",
    )


class CurationConfig(BaseModel):
    """Curation / visualization configuration."""

    data: DataConfig
    lpf: LpfConfig = LpfConfig()
    max_units: int | None = Field(
        None,
        ge=1,
        description=(
            "Maximum number of units to include in a visualization. "
            "If omitted/null, use all available units unless a CLI max-units "
            "override is provided."
        ),
    )


class AnalysisConfig(BaseModel):
    """Analysis / place-field configuration."""

    label: str = Field(
        "session",
        description="Label used for output filenames (e.g. WL25_DEC1).",
    )
    behavior_fps: float = Field(
        ...,
        gt=0.0,
        description="Behavior data sampling rate in frames per second. Required for spike-place matching.",
    )
    speed_threshold: float = Field(
        50.0,
        description="Minimum running speed to keep spikes (pixels/s or cm/s).",
    )
    cm_per_pixel: float | None = Field(
        None,
        description="Pixels-to-cm conversion; when set, thresholds are interpreted in cm/s.",
    )
    s_threshold: float = Field(
        0.0,
        description="Minimum spike amplitude s to visualize in place browser.",
    )
    speed_window_frames: int = Field(
        5,
        description="Number of frames to use for speed calculation window. "
        "Larger values give more stable speed estimates but less temporal resolution. "
        "Default 5 frames (0.25s at 20 fps).",
    )


class AppConfig(BaseModel):
    """Top-level application configuration."""

    curation: CurationConfig
    analysis: AnalysisConfig | None = None


def load_config(path: Path) -> AppConfig:
    """Load an AppConfig from a YAML file."""

    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    try:
        cfg = AppConfig.model_validate(raw)
    except ValidationError as exc:  # pragma: no cover - runtime validation
        raise ValueError(f"Invalid config file {path}:\n{exc}") from exc
    return cfg
