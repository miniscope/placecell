"""Analysis-related CLI commands."""

from datetime import datetime
from pathlib import Path

import click
from mio.logging import init_logger

from placecell.analysis import build_spike_place_dataframe
from placecell.config import AppConfig

logger = init_logger(__name__)


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
    "--neural-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing neural_timestamp.csv.",
)
@click.option(
    "--behavior-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing behavior_position.csv and behavior_timestamp.csv.",
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
    neural_path: Path,
    behavior_path: Path,
    out: Path | None,
    start_idx: int,
    end_idx: int | None,
) -> None:
    """Match spikes to behavior positions."""
    if out is None:
        out = Path(f"output/spike_place_{_default_timestamp()}.csv")

    cfg = AppConfig.from_yaml(config)
    if cfg.behavior is None:
        raise click.ClickException("Config file must include a 'behavior' section.")

    _run_spike_place(
        spike_index=spike_index,
        neural_timestamp=neural_path / "neural_timestamp.csv",
        behavior_position=behavior_path / "behavior_position.csv",
        behavior_timestamp=behavior_path / "behavior_timestamp.csv",
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
    "--neural-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory containing neural data (C.zarr, neural_timestamp.csv).",
)
@click.option(
    "--behavior-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory containing behavior data (behavior_position.csv, behavior_timestamp.csv).",
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
    neural_path: Path,
    behavior_path: Path,
    out_dir: Path,
    label: str | None,
    start_idx: int,
    end_idx: int | None,
) -> None:
    """Run deconvolution, spike-place matching, and launch interactive plot."""

    import subprocess

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

    # Construct paths from directories
    neural_timestamp = neural_path / "neural_timestamp.csv"
    behavior_position = behavior_path / "behavior_position.csv"
    behavior_timestamp = behavior_path / "behavior_timestamp.csv"

    # Verify files exist
    if not neural_timestamp.exists():
        raise click.ClickException(f"Neural timestamp file not found: {neural_timestamp}")
    if not behavior_position.exists():
        raise click.ClickException(f"Behavior position file not found: {behavior_position}")
    if not behavior_timestamp.exists():
        raise click.ClickException(f"Behavior timestamp file not found: {behavior_timestamp}")

    # 1) Deconvolution
    click.echo("=== Deconvolution ===")
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
    subprocess.run(cmd1, check=True)

    # 2) Spike-place (internal function)
    click.echo("=== Spike-place ===")
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
    click.echo("=== Interactive plot ===")
    from placecell.visualization import browse_place_cells

    browse_place_cells(
        spike_place_csv=out_dir / f"spike_place_{label}.csv",
        neural_path=neural_path,
        spike_index_csv=out_dir / f"spike_index_{label}.csv",
        trace_name=cfg.neural.trace_name,
        min_speed=cfg.behavior.speed_threshold,
        min_occupancy=cfg.behavior.ratemap.min_occupancy,
        bins=cfg.behavior.ratemap.bins,
        smooth_sigma=cfg.behavior.ratemap.smooth_sigma,
        behavior_fps=cfg.behavior.behavior_fps,
        neural_fps=cfg.neural.data.fps,
        n_shuffles=cfg.behavior.ratemap.n_shuffles,
        random_seed=cfg.behavior.ratemap.random_seed,
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
    "--neural-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory containing neural data (for traces and max projection).",
)
def plot(
    config: Path,
    spike_place_path: Path,
    spike_index_path: Path | None,
    neural_path: Path | None,
) -> None:
    """Interactive matplotlib browser for place cells."""
    from placecell.visualization import browse_place_cells

    cfg = AppConfig.from_yaml(config)
    if cfg.behavior is None:
        raise click.ClickException("Config file must include a 'behavior' section.")

    browse_place_cells(
        spike_place_csv=spike_place_path,
        neural_path=neural_path,
        spike_index_csv=spike_index_path,
        trace_name=cfg.neural.trace_name,
        min_speed=cfg.behavior.speed_threshold,
        min_occupancy=cfg.behavior.ratemap.min_occupancy,
        bins=cfg.behavior.ratemap.bins,
        smooth_sigma=cfg.behavior.ratemap.smooth_sigma,
        behavior_fps=cfg.behavior.behavior_fps,
        neural_fps=cfg.neural.data.fps,
        n_shuffles=cfg.behavior.ratemap.n_shuffles,
        random_seed=cfg.behavior.ratemap.random_seed,
    )
