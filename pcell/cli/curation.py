"""Curation-related CLI commands."""

import webbrowser
from pathlib import Path

import click
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

from pcell.config import AppConfig
from pcell.curation import interactive_select_units, load_traces
from pcell.filters import butter_lowpass_xr


def _load_template(name: str) -> str:
    """Load an HTML template from the templates directory."""
    template_path = Path(__file__).parent / "templates" / f"{name}.html"
    return template_path.read_text(encoding="utf-8")


@click.command(name="curate-traces")
@click.option(
    "--minian-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    prompt="Path to Minian dataset directory (folder containing C.zarr)",
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
    "--out-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("export/curated_units.txt"),
    show_default=True,
    help="Path to write selected unit IDs (one per line).",
)
@click.option(
    "--max-units",
    type=int,
    default=None,
    show_default=False,
    help="Optional limit on number of units to review.",
)
def curate_traces(
    minian_path: Path,
    trace_name: str,
    fps: float,
    out_file: Path,
    max_units: int | None,
) -> None:
    """Interactively view traces and write kept unit IDs to a text file."""

    minian_path = minian_path.resolve()
    click.echo(f"Loading traces for curation from: {minian_path}")

    C = load_traces(minian_path, trace_name=trace_name)
    click.echo(
        f"Traces loaded: shape={tuple(C.shape)}, dims={C.dims}. " "Launching interactive viewer..."
    )

    kept = interactive_select_units(C, fps=fps, out_file=out_file, max_units=max_units)
    click.echo(f"Kept {len(kept)} units. IDs written to: {out_file.resolve()}")


@click.command(name="curate")
@click.option(
    "--mode",
    type=click.Choice(["show", "browse"], case_sensitive=False),
    default="browse",
    show_default=True,
    help="Curation mode: 'show' displays many traces with checkboxes, 'browse' shows one at a time.",
)
@click.option(
    "--minian-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    show_default=False,
    help="Minian dataset directory (folder containing C.zarr). "
    "If omitted, must be provided by --config or will be prompted.",
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
    help="Prefix for the generated HTML files (default: 'pcell_traces' for show, 'pcell_browse' for browse).",
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
    minian_path: Path | None,
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
        minian_path = cfg.curation.data.minian_path
        trace_name = cfg.curation.data.trace_name
        fps = cfg.curation.data.fps
        if cfg.curation.max_units is not None:
            max_units = min(max_units, cfg.curation.max_units)
    else:
        cfg = None

    if minian_path is None:
        # Fallback prompt only if neither CLI nor config provided a path
        minian_path_str = click.prompt(
            "Path to Minian dataset directory (folder containing C.zarr)",
            type=str,
        )
        minian_path = Path(minian_path_str)

    minian_path = minian_path.resolve()
    click.echo(f"Loading traces from: {minian_path}")

    C = load_traces(minian_path, trace_name=trace_name)

    # Optional low-pass filter
    if cfg is not None and cfg.curation.lpf.enabled:
        click.echo(
            f"Applying low-pass filter: cutoff={cfg.curation.lpf.cutoff_hz} Hz, "
            f"order={cfg.curation.lpf.order}"
        )
        C = butter_lowpass_xr(
            C,
            fps=fps,
            cutoff_hz=cfg.curation.lpf.cutoff_hz,
            order=cfg.curation.lpf.order,
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

    html = _load_template("show").format(
        n=n,
        minian_path=minian_path,
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
