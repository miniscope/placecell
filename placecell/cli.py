"""Command-line interface for the placecell analysis pipeline."""

from pathlib import Path

import click

from placecell.logging import init_logger

logger = init_logger(__name__)


@click.group()
def cli() -> None:
    """Placecell analysis pipeline."""


@cli.command()
@click.option("-c", "--config", required=True, help="Analysis config file path or config ID.")
@click.option(
    "-d", "--data", "data_path", required=True,
    type=click.Path(exists=True), help="Per-session data paths YAML file.",
)
@click.option("-o", "--output", default=None, help="Output bundle path.")
@click.option(
    "-y", "--yes", is_flag=True,
    help="Skip confirmation prompt and run full analysis.",
)
@click.option(
    "--prep-only", is_flag=True,
    help="Stop after occupancy — save QC figures only, skip analyze_units.",
)
@click.option(
    "--show", is_flag=True,
    help="Open QC figures interactively before the confirmation prompt.",
)
def analysis(
    config: str, data_path: str, output: str | None,
    yes: bool, prep_only: bool, show: bool,
) -> None:
    """Run the place cell analysis pipeline."""
    from tqdm.auto import tqdm

    from placecell.dataset import BasePlaceCellDataset

    data_p = Path(data_path)
    if output is None:
        bundle_dir = Path.cwd() / "output"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        output = str(bundle_dir / f"{data_p.stem}.pcellbundle")

    ds = BasePlaceCellDataset.from_yaml(config, data_path)

    ds.load()
    ds.preprocess_behavior()
    ds.deconvolve(progress_bar=tqdm)
    ds.match_events()
    ds.compute_occupancy()

    # Save partial bundle with QC figures
    bundle_path = ds.save_bundle(output)
    click.echo(f"QC bundle saved to {bundle_path}")
    click.echo(f"Check figures at {bundle_path / 'figures'}")

    if show:
        _show_figures(bundle_path / "figures")

    if prep_only:
        return

    if not yes:
        if not click.confirm("Proceed with analyze_units?"):
            click.echo("Stopped after prep.")
            return

    # Run the expensive analysis step
    ds.analyze_units(progress_bar=tqdm)

    # Save analysis results into the existing bundle
    ur_dir = bundle_path / "unit_results"
    ur_dir.mkdir(exist_ok=True)
    ds._save_unit_results(ur_dir)

    # Re-generate figures (now includes diagnostics + summary_scatter)
    figures_dir = bundle_path / "figures"
    ds._save_summary_figures(figures_dir)

    click.echo(f"Full analysis saved to {bundle_path}")


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
