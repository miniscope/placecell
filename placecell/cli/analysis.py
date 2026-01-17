"""Analysis-related CLI commands."""

from datetime import datetime
from pathlib import Path

import click
from mio.logging import init_logger

from placecell.analysis import build_spike_place_dataframe
from placecell.config import AppConfig, BehaviorConfig, DataPathsConfig

logger = init_logger(__name__)


def _launch_browser(
    spike_place_csv: Path,
    behavior_position: Path,
    behavior_timestamp: Path,
    cfg: AppConfig,
    behavior_cfg: BehaviorConfig,
    neural_path: Path | None = None,
    spike_index_csv: Path | None = None,
) -> None:
    """Launch the interactive place cell browser with config settings."""
    from placecell.visualization import browse_place_cells

    browse_place_cells(
        spike_place_csv=spike_place_csv,
        behavior_position=behavior_position,
        behavior_timestamp=behavior_timestamp,
        bodypart=behavior_cfg.bodypart,
        neural_path=neural_path,
        spike_index_csv=spike_index_csv,
        trace_name=cfg.neural.trace_name,
        speed_threshold=behavior_cfg.speed_threshold,
        min_occupancy=behavior_cfg.spatial_map.min_occupancy,
        bins=behavior_cfg.spatial_map.bins,
        occupancy_sigma=behavior_cfg.spatial_map.occupancy_sigma,
        activity_sigma=behavior_cfg.spatial_map.activity_sigma,
        behavior_fps=behavior_cfg.behavior_fps,
        neural_fps=cfg.neural.fps,
        speed_window_frames=behavior_cfg.speed_window_frames,
        n_shuffles=behavior_cfg.spatial_map.n_shuffles,
        random_seed=behavior_cfg.spatial_map.random_seed,
        spike_threshold_sigma=behavior_cfg.spatial_map.spike_threshold_sigma,
        p_value_threshold=behavior_cfg.spatial_map.p_value_threshold,
    )


def _default_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_spike_place(
    spike_index: Path,
    neural_timestamp: Path,
    behavior_position: Path,
    behavior_timestamp: Path,
    bodypart: str,
    behavior_fps: float,
    speed_threshold: float,
    speed_window_frames: int,
    out_file: Path,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> None:
    """Internal function: Match spikes to behavior positions and write CSV."""

    df = build_spike_place_dataframe(
        spike_index_path=spike_index,
        neural_timestamp_path=neural_timestamp,
        behavior_position_path=behavior_position,
        behavior_timestamp_path=behavior_timestamp,
        bodypart=bodypart,
        behavior_fps=behavior_fps,
        speed_threshold=speed_threshold,
        speed_window_frames=speed_window_frames,
    )

    # Filter by unit index range
    all_unit_ids = sorted(df["unit_id"].unique())
    if end_idx is None:
        end_idx = len(all_unit_ids) - 1
    selected_units = all_unit_ids[start_idx : end_idx + 1]
    df = df[df["unit_id"].isin(selected_units)]

    out_file = out_file.resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    logger.info(f"Wrote {len(df)} rows for units {start_idx}-{end_idx} to {out_file}")


@click.command(name="spike-place")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="YAML config file with behavior settings.",
)
@click.option(
    "--spike-index",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Spike index CSV from deconvolve step.",
)
@click.option(
    "--data",
    "data_config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML file with data paths. Mutually exclusive with individual path options.",
)
@click.option(
    "--neural-timestamp",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Neural timestamp CSV file (neural_timestamp.csv).",
)
@click.option(
    "--behavior-position",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Behavior position CSV file (behavior_position.csv).",
)
@click.option(
    "--behavior-timestamp",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Behavior timestamp CSV file (behavior_timestamp.csv).",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output CSV path. Defaults to output/spike_place_<timestamp>.csv",
)
@click.option(
    "--start-idx",
    type=int,
    default=0,
    show_default=True,
    help="Start unit index.",
)
@click.option(
    "--end-idx",
    type=int,
    default=None,
    help="End unit index (inclusive). Defaults to last unit.",
)
def spike_place(
    config: Path,
    spike_index: Path,
    data_config: Path | None,
    neural_timestamp: Path | None,
    behavior_position: Path | None,
    behavior_timestamp: Path | None,
    out: Path | None,
    start_idx: int,
    end_idx: int | None,
) -> None:
    """Match spikes to behavior positions."""
    # --data and individual path options are mutually exclusive
    if data_config and (neural_timestamp or behavior_position or behavior_timestamp):
        raise click.ClickException(
            "Cannot use --data with individual path options. Use one or the other."
        )
    if data_config:
        paths = DataPathsConfig.from_yaml(data_config)
        yaml_dir = data_config.parent
        neural_timestamp = (yaml_dir / paths.neural_timestamp).resolve()
        behavior_position = (yaml_dir / paths.behavior_position).resolve()
        behavior_timestamp = (yaml_dir / paths.behavior_timestamp).resolve()

    if out is None:
        out = Path(f"output/spike_place_{_default_timestamp()}.csv")

    cfg = AppConfig.from_yaml(config)
    if cfg.behavior is None:
        raise click.ClickException("Config file must include a 'behavior' section.")

    _run_spike_place(
        spike_index=spike_index,
        neural_timestamp=neural_timestamp,
        behavior_position=behavior_position,
        behavior_timestamp=behavior_timestamp,
        bodypart=cfg.behavior.bodypart,
        behavior_fps=cfg.behavior.behavior_fps,
        speed_threshold=cfg.behavior.speed_threshold,
        speed_window_frames=cfg.behavior.speed_window_frames,
        out_file=out,
        start_idx=start_idx,
        end_idx=end_idx,
    )


@click.group(name="workflow")
def workflow() -> None:
    """Workflow commands for place cell analysis."""
    pass


@workflow.command(name="visualize")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="YAML config file defining analysis settings.",
)
@click.option(
    "--data",
    "data_config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "YAML file with data paths (neural_path, neural_timestamp, "
        "behavior_position, behavior_timestamp). Mutually exclusive with individual path options."
    ),
)
@click.option(
    "--neural-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Directory containing neural data (C.zarr, max_proj.zarr, A.zarr).",
)
@click.option(
    "--neural-timestamp",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Neural timestamp CSV file (neural_timestamp.csv).",
)
@click.option(
    "--behavior-position",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Behavior position CSV file (behavior_position.csv).",
)
@click.option(
    "--behavior-timestamp",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Behavior timestamp CSV file (behavior_timestamp.csv).",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("output"),
    show_default=True,
    help="Output directory for analysis results.",
)
@click.option(
    "--label",
    type=str,
    default=None,
    help="Label for output filenames. Defaults to timestamp.",
)
@click.option(
    "--start-idx",
    type=int,
    default=0,
    show_default=True,
    help="Start unit index.",
)
@click.option(
    "--end-idx",
    type=int,
    default=None,
    help="End unit index (inclusive). Defaults to last unit.",
)
def visualize(
    config: Path,
    data_config: Path | None,
    neural_path: Path | None,
    neural_timestamp: Path | None,
    behavior_position: Path | None,
    behavior_timestamp: Path | None,
    out_dir: Path,
    label: str | None,
    start_idx: int,
    end_idx: int | None,
) -> None:
    """Run deconvolution, spike-place matching, and launch interactive plot."""

    import subprocess

    # --data and individual path options are mutually exclusive
    if data_config and (neural_path or neural_timestamp or behavior_position or behavior_timestamp):
        raise click.ClickException(
            "Cannot use --data with individual path options. Use one or the other."
        )
    curation_csv = None
    if data_config:
        paths = DataPathsConfig.from_yaml(data_config)
        yaml_dir = data_config.parent
        neural_path = (yaml_dir / paths.neural_path).resolve()
        neural_timestamp = (yaml_dir / paths.neural_timestamp).resolve()
        behavior_position = (yaml_dir / paths.behavior_position).resolve()
        behavior_timestamp = (yaml_dir / paths.behavior_timestamp).resolve()
        if paths.curation_csv is not None:
            curation_csv = (yaml_dir / paths.curation_csv).resolve()

    cfg = AppConfig.from_yaml(config)
    if cfg.behavior is None:
        raise click.ClickException("Config file must include a 'behavior' section.")

    bodypart = cfg.behavior.bodypart

    # Auto-generate label with timestamp if not provided
    if label is None:
        label = _default_timestamp()

    # Ensure output directory exists
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Deconvolution
    cmd1 = [
        "pcell",
        "deconvolve",
        "--config",
        str(config),
        "--neural-path",
        str(neural_path),
        "--out-dir",
        str(out_dir),
        "--label",
        label,
        "--spike-index-out",
        str(out_dir / f"spike_index_{label}.csv"),
        "--start-idx",
        str(start_idx),
    ]
    if end_idx is not None:
        cmd1.extend(["--end-idx", str(end_idx)])
    if curation_csv is not None:
        cmd1.extend(["--curation-csv", str(curation_csv)])
    subprocess.run(cmd1, check=True)

    # 2) Spike-place (internal function)
    _run_spike_place(
        spike_index=out_dir / f"spike_index_{label}.csv",
        neural_timestamp=neural_timestamp,
        behavior_position=behavior_position,
        behavior_timestamp=behavior_timestamp,
        bodypart=bodypart,
        behavior_fps=cfg.behavior.behavior_fps,
        speed_threshold=cfg.behavior.speed_threshold,
        speed_window_frames=cfg.behavior.speed_window_frames,
        out_file=out_dir / f"spike_place_{label}.csv",
    )

    # 3) Interactive plot
    _launch_browser(
        spike_place_csv=out_dir / f"spike_place_{label}.csv",
        behavior_position=behavior_position,
        behavior_timestamp=behavior_timestamp,
        cfg=cfg,
        behavior_cfg=cfg.behavior,
        neural_path=neural_path,
        spike_index_csv=out_dir / f"spike_index_{label}.csv",
    )


@click.command(name="plot")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="YAML config file.",
)
@click.option(
    "--spike-place",
    "spike_place_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Spike-place CSV file.",
)
@click.option(
    "--spike-index",
    "spike_index_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Spike index CSV (optional, shows all spikes on trace plot).",
)
@click.option(
    "--data",
    "data_config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML file with data paths. Mutually exclusive with individual path options.",
)
@click.option(
    "--neural-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Directory containing neural data (for traces and max projection).",
)
@click.option(
    "--behavior-position",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Behavior position CSV file (behavior_position.csv).",
)
@click.option(
    "--behavior-timestamp",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Behavior timestamp CSV file (behavior_timestamp.csv).",
)
def plot(
    config: Path,
    spike_place_path: Path,
    spike_index_path: Path | None,
    data_config: Path | None,
    neural_path: Path | None,
    behavior_position: Path | None,
    behavior_timestamp: Path | None,
) -> None:
    """Interactive matplotlib browser for place cells."""
    # --data and individual path options are mutually exclusive
    if data_config and (neural_path or behavior_position or behavior_timestamp):
        raise click.ClickException(
            "Cannot use --data with individual path options. Use one or the other."
        )
    if data_config:
        paths = DataPathsConfig.from_yaml(data_config)
        yaml_dir = data_config.parent
        neural_path = (yaml_dir / paths.neural_path).resolve()
        behavior_position = (yaml_dir / paths.behavior_position).resolve()
        behavior_timestamp = (yaml_dir / paths.behavior_timestamp).resolve()

    cfg = AppConfig.from_yaml(config)
    if cfg.behavior is None:
        raise click.ClickException("Config file must include a 'behavior' section.")

    _launch_browser(
        spike_place_csv=spike_place_path,
        behavior_position=behavior_position,
        behavior_timestamp=behavior_timestamp,
        cfg=cfg,
        behavior_cfg=cfg.behavior,
        neural_path=neural_path,
        spike_index_csv=spike_index_path,
    )
