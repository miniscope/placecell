"""Command-line interface for the placecell analysis pipeline."""

from pathlib import Path

import click

from placecell.log import init_logger

logger = init_logger(__name__)


@click.group()
def cli() -> None:
    """Placecell analysis pipeline."""


@cli.command()
@click.option("-c", "--config", required=True, help="Analysis config file path or config ID.")
@click.option(
    "-d",
    "--data",
    "data_paths",
    required=True,
    multiple=True,
    type=click.Path(exists=True),
    help="Per-session data YAML file(s). Repeat for batch mode.",
)
@click.option("-o", "--output", default=None, help="Output bundle path or directory.")
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    help="Skip confirmation prompt and run full analysis.",
)
@click.option(
    "--show",
    is_flag=True,
    help="Open QC figures interactively before the confirmation prompt.",
)
@click.option(
    "-w",
    "--workers",
    type=int,
    default=1,
    show_default=True,
    help="Number of parallel worker processes for analyze_units.",
)
@click.option(
    "--subset-units",
    type=int,
    default=None,
    help="Keep only the first N neural units (for generating small test data).",
)
@click.option(
    "--subset-frames",
    type=int,
    default=None,
    help="Keep only the first N frames (for generating small test data).",
)
@click.option(
    "--force-redetect",
    is_flag=True,
    help=(
        "Force re-running detect-zones even when the cached zone_tracking CSV "
        "exists. Maze datasets only; ignored for arena."
    ),
)
def analysis(
    config: str,
    data_paths: tuple[str, ...],
    output: str | None,
    yes: bool,
    show: bool,
    workers: int,
    subset_units: int | None,
    subset_frames: int | None,
    force_redetect: bool,
) -> None:
    """Run the place cell analysis pipeline.

    Single dataset: placecell analysis -c config.yaml -d data.yaml
    Batch mode:     placecell analysis -c config.yaml -d a.yaml -d b.yaml -y
    """
    for i, data_path in enumerate(data_paths):
        if len(data_paths) > 1:
            click.echo(f"\n[{i + 1}/{len(data_paths)}] {Path(data_path).stem}")
        _run_one(
            config=config,
            data_path=data_path,
            output=output,
            yes=yes,
            show=show,
            workers=workers,
            subset_units=subset_units,
            subset_frames=subset_frames,
            force_redetect=force_redetect,
        )


def _run_one(
    *,
    config: str,
    data_path: str,
    output: str | None,
    yes: bool,
    show: bool,
    workers: int,
    subset_units: int | None,
    subset_frames: int | None,
    force_redetect: bool = False,
) -> None:
    """Run the pipeline for a single dataset."""
    from tqdm.auto import tqdm

    from placecell.dataset.base import BasePlaceCellDataset
    from placecell.dataset.maze import MazeDataset

    data_p = Path(data_path)
    if output is None:
        bundle_dir = Path.cwd() / "output"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        out = str(bundle_dir / f"{data_p.stem}.pcellbundle")
    else:
        out = output

    ds = BasePlaceCellDataset.from_yaml(config, data_path)

    if isinstance(ds, MazeDataset):
        ds.load(force_redetect=force_redetect)
    else:
        if force_redetect:
            click.echo("--force-redetect ignored: only applies to maze datasets.")
        ds.load()
    if subset_units is not None or subset_frames is not None:
        ds.subset(n_units=subset_units, n_frames=subset_frames)
    ds.preprocess_behavior()
    ds.deconvolve(progress_bar=tqdm)
    ds.match_events()
    ds.compute_occupancy()

    bundle_path = ds.save_bundle(out)
    click.echo(f"QC bundle saved to {bundle_path}")
    click.echo(f"Check figures at {bundle_path / 'figures'}")

    if show:
        _show_figures(bundle_path / "figures")

    if not yes and not click.confirm("Proceed with analyze_units?"):
        click.echo("Stopped after prep.")
        return

    ds.analyze_units(progress_bar=tqdm, n_workers=workers)

    ur_dir = bundle_path / "unit_results"
    ur_dir.mkdir(exist_ok=True)
    ds._save_unit_results(ur_dir)

    figures_dir = bundle_path / "figures"
    ds._save_summary_figures(figures_dir)

    click.echo(f"Full analysis saved to {bundle_path}")


@cli.command("define-zones")
@click.option(
    "-d",
    "--data",
    "data_path",
    required=True,
    type=click.Path(exists=True),
    help="Data config YAML (reads behavior_video and behavior_graph).",
)
@click.option("--rooms", type=int, required=True, help="Number of rooms.")
@click.option("--arms", type=int, required=True, help="Number of arms.")
def define_zones_cmd(data_path: str, rooms: int, arms: int) -> None:
    """Interactive zone definition tool (requires OpenCV)."""
    from placecell.config import BaseDataConfig, MazeDataConfig
    from placecell.define_zones import define_zones

    data_p = Path(data_path)
    data_cfg = BaseDataConfig.from_yaml(data_p)
    if not isinstance(data_cfg, MazeDataConfig):
        raise click.UsageError("define-zones requires a maze-type data config (type: maze)")
    data_dir = data_p.parent

    if not data_cfg.behavior_video:
        raise click.UsageError("behavior_video is required in data config for define-zones")

    video = str(data_dir / data_cfg.behavior_video)

    if data_cfg.behavior_graph:
        output = str(data_dir / data_cfg.behavior_graph)
        if Path(output).exists() and not click.confirm(f"{output} already exists. Overwrite?"):
            return
    else:
        graph_rel = f"zone_{data_p.stem}.yaml"
        output = str(data_dir / graph_rel)
        with open(data_p, "a") as f:
            f.write(f"behavior_graph: {graph_rel}\n")
        click.echo(f"Added behavior_graph: {graph_rel} to {data_p}")

    zone_names = [f"Room_{i+1}" for i in range(rooms)] + [f"Arm_{i+1}" for i in range(arms)]
    zone_types = {name: "room" if name.startswith("Room") else "arm" for name in zone_names}

    define_zones(
        video_path=video,
        output_file=output,
        zone_names=zone_names,
        zone_types=zone_types,
    )


@cli.command("detect-zones")
@click.option(
    "-d",
    "--data",
    "data_path",
    required=True,
    type=click.Path(exists=True),
    help="Data config YAML (reads behavior_position and behavior_graph).",
)
@click.option(
    "-o", "--output", default=None, help="Output CSV path (default: zone_tracking_{stem}.csv)."
)
@click.option(
    "--interpolate",
    type=int,
    default=None,
    help="Frame subsampling factor for video export (default: from config, or 5).",
)
@click.option(
    "--playback-speed",
    type=float,
    default=None,
    help="Playback speed multiplier for exported zone video (default: from config, or 10).",
)
def detect_zones_cmd(
    data_path: str,
    output: str | None,
    interpolate: int | None,
    playback_speed: float | None,
) -> None:
    """Run zone detection on tracking CSV using the zone graph."""
    from tqdm.auto import tqdm

    from placecell.config import BaseDataConfig, MazeDataConfig, ZoneDetectionConfig
    from placecell.zone_detection import backup_file, detect_zones_from_csv

    data_p = Path(data_path)
    data_cfg = BaseDataConfig.from_yaml(data_p)
    if not isinstance(data_cfg, MazeDataConfig):
        raise click.UsageError("detect-zones requires a maze-type data config (type: maze)")
    data_dir = data_p.parent

    if not data_cfg.behavior_graph:
        raise click.UsageError("behavior_graph is required in data config for detect-zones")
    if not data_cfg.bodypart:
        raise click.UsageError("bodypart is required in data config for detect-zones")

    zd = data_cfg.zone_detection or ZoneDetectionConfig()

    input_csv = str(data_dir / data_cfg.behavior_position)
    zone_config = str(data_dir / data_cfg.behavior_graph)

    # Determine output path
    if output is not None:
        output_rel = output
    elif data_cfg.zone_tracking:
        output_rel = data_cfg.zone_tracking
    else:
        # Auto-generate and append to data config YAML
        output_rel = f"zone_tracking_{data_p.stem}.csv"
        with open(data_p, "a") as f:
            f.write(f"zone_tracking: {output_rel}\n")
        click.echo(f"Added zone_tracking: {output_rel} to {data_p}")

    output_path = str(data_dir / output_rel)

    # Auto-backup existing output file
    if Path(output_path).exists():
        bak = backup_file(output_path)
        click.echo(f"Backed up → {bak}")

    # Video export (if behavior_video is configured)
    video_path = None
    if data_cfg.behavior_video:
        video_p = data_dir / data_cfg.behavior_video
        if video_p.exists():
            video_path = str(video_p)

    detect_zones_from_csv(
        input_csv=input_csv,
        output_csv=output_path,
        zone_config_path=zone_config,
        bodypart=data_cfg.bodypart,
        arm_max_distance=zd.arm_max_distance,
        min_confidence=zd.min_confidence,
        min_confidence_forbidden=zd.min_confidence_forbidden,
        min_frames_same=zd.min_frames_same,
        min_frames_forbidden=zd.min_frames_forbidden,
        room_decay_power=zd.room_decay_power,
        arm_decay_power=zd.arm_decay_power,
        soft_boundary=zd.soft_boundary,
        hampel_window_frames=zd.hampel_window_frames,
        hampel_n_sigmas=zd.hampel_n_sigmas,
        zone_connections=data_cfg.zone_connections,
        video_path=video_path,
        interpolate=interpolate if interpolate is not None else zd.interpolate,
        playback_speed=playback_speed if playback_speed is not None else zd.playback_speed,
        progress_bar=tqdm,
    )
    click.echo(f"Zone detection saved to {output_path}")


def _show_figures(figures_dir: Path) -> None:
    """Open all PDF figures in the default viewer."""
    import platform
    import subprocess

    pdfs = sorted(figures_dir.glob("*.pdf"))
    if not pdfs:
        click.echo("No figures to show.")
        return

    click.echo(f"Opening {len(pdfs)} figure(s)...")
    system = platform.system()
    for pdf in pdfs:
        if system == "Darwin":
            subprocess.Popen(["open", str(pdf)])
        elif system == "Linux":
            subprocess.Popen(["xdg-open", str(pdf)])
        elif system == "Windows":
            subprocess.Popen(["start", "", str(pdf)], shell=True)
