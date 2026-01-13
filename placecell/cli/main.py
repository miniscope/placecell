"""Main CLI entry point for placecell."""

import click
from mio.logging import init_logger

from placecell.cli.analysis import analyze, generate_html, spike_place
from placecell.cli.behavior import behavior
from placecell.cli.deconvolve import deconvolve


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging.",
)
def placecell(verbose: bool) -> None:
    """placecell command-line tools."""
    init_logger(__name__, level="DEBUG" if verbose else "INFO")


# Register all command groups
placecell.add_command(deconvolve)
placecell.add_command(spike_place)
placecell.add_command(generate_html)
placecell.add_command(analyze)
placecell.add_command(behavior)
