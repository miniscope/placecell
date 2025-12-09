"""CLI commands for behavior analysis and visualization."""

from pathlib import Path

import click

from pcell.config import AppConfig, BehaviorConfig
from pcell.visualization import plot_trajectory

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


@click.group(name="behavior")
def behavior() -> None:
    """Behavior analysis and visualization commands."""


@behavior.command(name="trajectory")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="YAML config file (full AppConfig or BehaviorConfig).",
)
@click.option(
    "--behavior-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory containing behavior data (behavior_position.csv).",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output path for the plot. If omitted, displays interactively.",
)
@click.option(
    "--dpi",
    type=int,
    default=150,
    help="DPI for saved plot.",
)
@click.option(
    "--linewidth",
    type=float,
    default=0.5,
    help="Line width for trajectory plot.",
)
@click.option(
    "--alpha",
    type=float,
    default=0.7,
    help="Alpha (transparency) for trajectory plot.",
)
def trajectory(
    config: Path,
    behavior_path: Path,
    output: Path | None,
    dpi: int,
    linewidth: float,
    alpha: float,
) -> None:
    """Plot trajectory from behavior position data using config settings."""
    # Try to load as AppConfig first, then fall back to BehaviorConfig
    try:
        cfg = AppConfig.from_yaml(config)
        if cfg.behavior is None:
            raise click.ClickException("Config file must include a 'behavior' section.")
        bodypart = cfg.behavior.bodypart
    except Exception:
        # Try as BehaviorConfig directly
        try:
            behavior_cfg = BehaviorConfig.from_yaml(config)
            bodypart = behavior_cfg.bodypart
        except Exception as exc:
            raise click.ClickException(
                f"Failed to load config. Must be AppConfig or BehaviorConfig: {exc}"
            ) from exc

    # Construct behavior position path
    behavior_position = behavior_path / "behavior_position.csv"
    if not behavior_position.exists():
        raise click.ClickException(f"Behavior position file not found: {behavior_position}")

    # Plot trajectory
    if plt is None:
        raise click.ClickException(
            "matplotlib is required for trajectory plotting. "
            "Install it with: pip install matplotlib"
        )

    click.echo(f"Plotting trajectory for bodypart '{bodypart}' from {behavior_position}")
    plot_trajectory(behavior_position, bodypart=bodypart, linewidth=linewidth, alpha=alpha)
    plt.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=dpi)
        click.echo(f"Plot saved to {output.resolve()}")
    else:
        click.echo("Displaying plot interactively...")
        plt.show()
