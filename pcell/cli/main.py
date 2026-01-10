"""Main CLI entry point for pcell."""

import logging

import click
from mio.logging import init_logger

from pcell.cli.analysis import analyze
from pcell.cli.behavior import behavior
from pcell.cli.deconvolve import deconvolve


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging.",
)
def pcell(verbose: bool) -> None:
    """pcell command-line tools."""
    init_logger(level=logging.DEBUG if verbose else logging.INFO)


# Register all command groups
pcell.add_command(deconvolve)
pcell.add_command(analyze)
pcell.add_command(behavior)
