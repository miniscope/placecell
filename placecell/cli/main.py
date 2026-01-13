"""Main CLI entry point for placecell."""

import logging

import click
from mio.logging import init_logger

from placecell.cli.analysis import analyze
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
    init_logger(level=logging.DEBUG if verbose else logging.INFO)


# Register all command groups
placecell.add_command(deconvolve)
placecell.add_command(analyze)
placecell.add_command(behavior)
