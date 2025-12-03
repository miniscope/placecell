"""Main CLI entry point for pcell."""

import click

from pcell.cli.analysis import analyze
from pcell.cli.curation import curate, curate_traces
from pcell.cli.deconvolve import deconvolve


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def pcell() -> None:
    """pcell command-line tools."""


# Register all command groups
pcell.add_command(curate)
pcell.add_command(curate_traces)
pcell.add_command(deconvolve)
pcell.add_command(analyze)
