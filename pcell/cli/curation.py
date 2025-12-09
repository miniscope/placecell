"""Curation-related CLI commands."""

import webbrowser
from pathlib import Path

import click
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

from pcell.cli.utils import load_template
from pcell.config import AppConfig
from pcell.curation import load_traces
from pcell.filters import butter_lowpass_xr


@click.command(name="curate")
@click.option(
    "--mode",
    type=click.Choice(["show", "browse"], case_sensitive=False),
    default="browse",
    show_default=True,
    help=(
        "Curation mode: 'show' displays many traces with checkboxes, "
        "'browse' shows one at a time."
    ),
)
@click.option(
    "--neural-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory containing neural data (C.zarr).",
)
@click.option(
    "--trace-name",
    type=str,
    default="C",
    show_default=True,
    help="Name of the traces zarr group to load (e.g. 'C' or 'C_lp').",
)
@click.option(
    "--fps",
    type=float,
    default=20.0,
    show_default=True,
    help="Sampling rate in frames per second.",
)
@click.option(
    "--max-units",
    type=int,
    default=None,
    show_default=False,
    help="Maximum number of units to review (default: 48 for show mode, 114 for browse mode).",
)
@click.option(
    "--output-prefix",
    type=str,
    default=None,
    show_default=False,
    help=(
        "Prefix for the generated HTML files "
        "(default: 'pcell_traces' for show, 'pcell_browse' for browse)."
    ),
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    show_default=False,
    help="Optional YAML config file overriding data/LPF settings.",
)
def curate(
    mode: str,
    neural_path: Path,
    trace_name: str,
    fps: float,
    max_units: int | None,
    output_prefix: str | None,
    config: Path | None,
) -> None:
    """Interactive trace curation with HTML browser.

    Modes:
    - 'show': Display many traces with checkboxes for selection
    - 'browse': Browse one trace at a time with keep/reject buttons
    """

    # Optional YAML config overrides basic data/LFP settings
    if config is not None:
        cfg = AppConfig.from_yaml(config)
        trace_name = cfg.neural.trace_name
        fps = cfg.neural.data.fps
        if cfg.neural.max_units is not None:
            max_units = min(max_units, cfg.neural.max_units)
    else:
        cfg = None

    if neural_path is None:
        raise click.ClickException(
            "--neural-path is required. Specify the directory containing neural data (C.zarr)."
        )

    neural_path = neural_path.resolve()
    click.echo(f"Loading traces from: {neural_path}")

    C = load_traces(neural_path, trace_name=trace_name)

    # Optional low-pass filter
    if cfg is not None and cfg.neural.lpf.enabled:
        click.echo(
            f"Applying low-pass filter: cutoff={cfg.neural.lpf.cutoff_hz} Hz, "
            f"order={cfg.neural.lpf.order}"
        )
        C = butter_lowpass_xr(
            C,
            fps=fps,
            cutoff_hz=cfg.neural.lpf.cutoff_hz,
            order=cfg.neural.lpf.order,
        )
    unit_ids = list(map(int, C["unit_id"].values))
    if not unit_ids:
        raise click.ClickException("No units found.")

    unit_ids = unit_ids[:max_units]
    n = len(unit_ids)

    click.echo(f"Preparing Plotly figure for {n} units (fps={fps}).")

    t = np.arange(C.sizes["frame"]) / float(fps)

    # Vertical layout: one row per unit, shared x-axis
    # Plotly requires vertical_spacing <= 1/(rows-1)
    if n > 1:
        max_vs = 1.0 / (n - 1) - 1e-4
        vertical_spacing = min(0.02, max_vs)
    else:
        vertical_spacing = 0.0

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
    )

    for row_idx, uid in enumerate(unit_ids, start=1):
        y = np.asarray(C.sel(unit_id=uid).values, dtype=float)
        fig.add_trace(
            go.Scatter(
                x=t,
                y=y,
                mode="lines",
                name=f"unit {uid}",
                hovertemplate="t=%{x:.2f}s<br>y=%{y:.3f}<extra>unit " + str(uid) + "</extra>",
            ),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(title_text=f"{uid}", row=row_idx, col=1)

    fig.update_xaxes(title_text="Time (s)", row=n, col=1)
    fig.update_layout(
        title="pcell traces",
        height=max(400, 140 * n),
        template="plotly_white",
        showlegend=False,
    )

    # Build HTML with checkboxes + download button for curated IDs
    plot_div = pio.to_html(
        fig, include_plotlyjs="cdn", full_html=False, config={"responsive": True}
    )

    checkboxes_html = "\n".join(
        f'<label><input type="checkbox" class="unit-checkbox" value="{uid}"> unit {uid}</label><br>'
        for uid in unit_ids
    )

    html = load_template("show").format(
        n=n,
        minian_path=neural_path,  # template variable name
        plot_div=plot_div,
        checkboxes_html=checkboxes_html,
    )

    out_html = Path("export") / f"{output_prefix}.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html = out_html.resolve()
    out_html.write_text(html, encoding="utf-8")

    click.echo(f"Wrote Plotly + checkbox HTML to: {out_html}")

    # Try to open in default browser; if it fails, just print the path.
    try:
        webbrowser.open(out_html.as_uri())
    except Exception:
        click.echo("Could not open browser automatically; open the HTML file manually.")
