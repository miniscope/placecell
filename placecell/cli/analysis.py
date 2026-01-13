"""Analysis-related CLI commands."""

import json
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import click
import numpy as np
import pandas as pd
import xarray as xr
from mio.logging import init_logger

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from placecell.analysis import build_spike_place_dataframe
from placecell.cli.utils import load_template
from placecell.config import AppConfig

logger = init_logger(__name__)


def _default_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class PlaceBrowserData(NamedTuple):
    """Data structure for place browser visualization."""

    unit_ids: list[int]
    x_full: list[float]
    y_full: list[float]
    spike_xs_above: list[list[float]]
    spike_ys_above: list[list[float]]
    spike_xs_below: list[list[float]]
    spike_ys_below: list[list[float]]
    counts_above: list[int]
    counts_below: list[int]
    trace_t: list[float]
    trace_ys_raw: list[list[float]]
    trace_ys_lp: list[list[float]]
    trace_ys_yrA: list[list[float]]
    trace_ys_S: list[list[float]]
    has_yrA: bool
    has_S: bool
    spike_ts_trace_above: list[list[float]]
    spike_y_trace_above: list[list[float]]
    spike_ts_trace_below: list[list[float]]
    spike_y_trace_below: list[list[float]]


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

    out_file = out_file.resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    logger.info(f"Wrote {len(df)} rows to {out_file}")


def prepare_place_browser_data(
    spike_place: Path,
    spike_index: Path | None,
    neural_timestamp: Path,
    behavior_position: Path,
    behavior_timestamp: Path,
    bodypart: str,
    behavior_fps: float,
    speed_threshold: float,
    speed_window_frames: int,
    s_threshold: float,
    neural_path: Path | None,
    trace_name: str,
    trace_name_lp: str,
    neural_fps: float,
    deconv_zarr: Path | None,
) -> PlaceBrowserData:
    """Prepare data for place browser visualization.

    Parameters
    ----------
    behavior_fps:
        Behavior data sampling rate (frames per second), used for spike-place matching.
    neural_fps:
        Neural data sampling rate (frames per second), used for converting neural frames to time.
    """

    from placecell.analysis import load_traces

    # Load spikes - if spike_index provided, load all spikes with speed info
    if spike_index is not None:
        # Load all spikes (not filtered by speed)
        sp_all = build_spike_place_dataframe(
            spike_index_path=spike_index,
            neural_timestamp_path=neural_timestamp,
            behavior_position_path=behavior_position,
            behavior_timestamp_path=behavior_timestamp,
            bodypart=bodypart,
            behavior_fps=behavior_fps,
            speed_threshold=0.0,  # Don't filter by speed
            speed_window_frames=speed_window_frames,
        )
        # Filter by s_threshold only
        if s_threshold > 0:
            sp_all = sp_all[sp_all["s"] >= s_threshold].reset_index(drop=True)

        # Separate by speed threshold
        sp_above = sp_all[sp_all["speed"] >= speed_threshold].reset_index(drop=True)
        sp_below = sp_all[sp_all["speed"] < speed_threshold].reset_index(drop=True)

        # Use all units that have spikes
        unit_ids = sorted(sp_all["unit_id"].unique().tolist())
    else:
        # Use existing spike_place file (already filtered)
        sp_above = pd.read_csv(spike_place)
        if s_threshold > 0:
            sp_above = sp_above[sp_above["s"] >= s_threshold].reset_index(drop=True)
        sp_below = pd.DataFrame()  # Empty - no below-threshold spikes available
        unit_ids = sorted(sp_above["unit_id"].unique().tolist())

    if sp_above.empty and sp_below.empty:
        raise click.ClickException("No spikes to plot after applying thresholds.")

    # Load full behavior trajectory
    beh_pos = pd.read_csv(behavior_position, header=[0, 1, 2])
    scorer = beh_pos.columns[1][0]
    x_full = beh_pos[(scorer, bodypart, "x")].to_numpy().tolist()
    y_full = beh_pos[(scorer, bodypart, "y")].to_numpy().tolist()

    # Precompute spike positions per unit (above and below threshold)
    spike_xs_above: list[list[float]] = []
    spike_ys_above: list[list[float]] = []
    spike_xs_below: list[list[float]] = []
    spike_ys_below: list[list[float]] = []
    counts_above: list[int] = []
    counts_below: list[int] = []
    for uid in unit_ids:
        df_u_above = sp_above[sp_above["unit_id"] == uid] if not sp_above.empty else pd.DataFrame()
        df_u_below = sp_below[sp_below["unit_id"] == uid] if not sp_below.empty else pd.DataFrame()

        spike_xs_above.append(df_u_above["x"].to_numpy().tolist() if not df_u_above.empty else [])
        spike_ys_above.append(df_u_above["y"].to_numpy().tolist() if not df_u_above.empty else [])
        spike_xs_below.append(df_u_below["x"].to_numpy().tolist() if not df_u_below.empty else [])
        spike_ys_below.append(df_u_below["y"].to_numpy().tolist() if not df_u_below.empty else [])
        counts_above.append(len(df_u_above))
        counts_below.append(len(df_u_below))

    # Load raw C, filtered C, and optionally YrA (most raw)
    trace_t: list[float] = []
    trace_ys_raw: list[list[float]] = []
    trace_ys_lp: list[list[float]] = []
    trace_ys_yrA: list[list[float]] = []
    trace_ys_S: list[list[float]] = []
    has_yrA = False
    has_S = False
    spike_ts_trace_above: list[list[float]] = []
    spike_y_trace_above: list[list[float]] = []
    spike_ts_trace_below: list[list[float]] = []
    spike_y_trace_below: list[list[float]] = []

    if neural_path is not None and Path(neural_path).is_dir():
        try:
            # Try to load YrA first (most raw - spatial components on raw video)
            try:
                YrA = load_traces(neural_path, trace_name="YrA")
                has_yrA = True
                logger.info("Loaded YrA traces (raw fluorescence)")
            except Exception:
                YrA = None
                has_yrA = False

            # Load raw C
            C_raw = load_traces(neural_path, trace_name=trace_name)
            # Try to load filtered C, fall back to applying filter if not found
            try:
                C_lp = load_traces(neural_path, trace_name=trace_name_lp)
            except Exception:
                from placecell.filters import butter_lowpass_xr

                logger.info(
                    f"Filtered trace {trace_name_lp} not found, applying low-pass filter..."
                )
                C_lp = butter_lowpass_xr(C_raw, fps=neural_fps, cutoff_hz=1.0, order=4)

            # Use YrA if available, otherwise use C_raw
            trace_source = YrA if has_yrA else C_raw
            T = int(trace_source.sizes["frame"])
            trace_t = (np.arange(T) / neural_fps).tolist()

            for uid in unit_ids:
                if has_yrA:
                    y_yrA = np.asarray(YrA.sel(unit_id=int(uid)).values, dtype=float)
                    trace_ys_yrA.append(y_yrA.tolist())
                else:
                    trace_ys_yrA.append([])

                y_raw = np.asarray(C_raw.sel(unit_id=int(uid)).values, dtype=float)
                y_lp = np.asarray(C_lp.sel(unit_id=int(uid)).values, dtype=float)
                trace_ys_raw.append(y_raw.tolist())
                trace_ys_lp.append(y_lp.tolist())

                # Spikes are detected from C (via OASIS deconvolution), so align markers with C
                y_for_spikes = y_raw

                # Spike times for above threshold
                df_u_above = (
                    sp_above[sp_above["unit_id"] == uid] if not sp_above.empty else pd.DataFrame()
                )
                if not df_u_above.empty:
                    frames = df_u_above["frame"].to_numpy(dtype=int)
                    frames = frames[(frames >= 0) & (frames < T)]
                    spike_ts_trace_above.append((frames / neural_fps).tolist())
                    spike_y_trace_above.append(y_for_spikes[frames].tolist())
                else:
                    spike_ts_trace_above.append([])
                    spike_y_trace_above.append([])

                # Spike times for below threshold
                df_u_below = (
                    sp_below[sp_below["unit_id"] == uid] if not sp_below.empty else pd.DataFrame()
                )
                if not df_u_below.empty:
                    frames = df_u_below["frame"].to_numpy(dtype=int)
                    frames = frames[(frames >= 0) & (frames < T)]
                    spike_ts_trace_below.append((frames / neural_fps).tolist())
                    spike_y_trace_below.append(y_for_spikes[frames].tolist())
                else:
                    spike_ts_trace_below.append([])
                    spike_y_trace_below.append([])
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning(f"Could not load traces from {neural_path}: {exc}")
            trace_t = []
            trace_ys_raw = []
            trace_ys_lp = []
            trace_ys_yrA = []

    # Initialize S traces if not already done
    if len(trace_ys_S) < len(unit_ids):
        trace_ys_S = [[] for _ in unit_ids]

    # Try to load S (spike train) from deconv zarr if available
    if deconv_zarr is not None and Path(deconv_zarr).is_dir():
        try:
            ds = xr.open_zarr(deconv_zarr, consolidated=False)
            if "S" in ds:
                has_S = True
                T_S = int(ds.sizes["frame"])
                fps_from_zarr = float(ds.attrs.get("fps", neural_fps))
                if not trace_t:
                    trace_t = (np.arange(T_S) / fps_from_zarr).tolist()
                for uid in unit_ids:
                    S_vec = np.asarray(ds["S"].sel(unit_id=int(uid)).values, dtype=float)
                    trace_ys_S.append(S_vec.tolist())
            else:
                for _ in unit_ids:
                    trace_ys_S.append([])
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning(f"Could not load S from {deconv_zarr}: {exc}")
            for _ in unit_ids:
                trace_ys_S.append([])

    # Fallback to deconv zarr if neural_path not provided
    if not trace_t and deconv_zarr is not None and Path(deconv_zarr).is_dir():
        try:
            ds = xr.open_zarr(deconv_zarr, consolidated=False)
            var = "C_deconv" if "C_deconv" in ds else "S" if "S" in ds else None
            if var is not None:
                T = int(ds.sizes["frame"])
                fps_from_zarr = float(ds.attrs.get("fps", neural_fps))
                trace_t = (np.arange(T) / fps_from_zarr).tolist()
                for uid in unit_ids:
                    y = np.asarray(ds[var].sel(unit_id=int(uid)).values, dtype=float)
                    trace_ys_raw.append(y.tolist())
                    trace_ys_lp.append(y.tolist())  # Use same for both if no raw available
                    trace_ys_yrA.append([])  # No YrA from deconv zarr
                    # Spike times
                    df_u = (
                        sp_above[sp_above["unit_id"] == uid]
                        if not sp_above.empty
                        else pd.DataFrame()
                    )
                    if not df_u.empty:
                        frames = df_u["frame"].to_numpy(dtype=int)
                        frames = frames[(frames >= 0) & (frames < T)]
                        spike_ts_trace_above.append((frames / fps_from_zarr).tolist())
                        spike_y_trace_above.append(y[frames].tolist())
                    else:
                        spike_ts_trace_above.append([])
                        spike_y_trace_above.append([])
                    spike_ts_trace_below.append([])
                    spike_y_trace_below.append([])
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning(f"Could not load traces from {deconv_zarr}: {exc}")

    return PlaceBrowserData(
        unit_ids=unit_ids,
        x_full=x_full,
        y_full=y_full,
        spike_xs_above=spike_xs_above,
        spike_ys_above=spike_ys_above,
        spike_xs_below=spike_xs_below,
        spike_ys_below=spike_ys_below,
        counts_above=counts_above,
        counts_below=counts_below,
        trace_t=trace_t,
        trace_ys_raw=trace_ys_raw,
        trace_ys_lp=trace_ys_lp,
        trace_ys_yrA=trace_ys_yrA,
        trace_ys_S=trace_ys_S,
        has_yrA=has_yrA,
        has_S=has_S,
        spike_ts_trace_above=spike_ts_trace_above,
        spike_y_trace_above=spike_y_trace_above,
        spike_ts_trace_below=spike_ts_trace_below,
        spike_y_trace_below=spike_y_trace_below,
    )


def generate_place_browser_html(
    data: PlaceBrowserData,
    s_threshold: float,
    speed_threshold: float,
    output_prefix: str,
    max_proj_footprints_imgs: dict[int, str] | None = None,
) -> Path:
    """Generate place browser HTML from prepared data.

    Parameters
    ----------
    max_proj_footprints_imgs:
        Dictionary mapping unit_id to base64-encoded image data URL for max projection
        and footprints plot. If None or empty, the image section will be omitted.
    """

    unit_ids_json = json.dumps(data.unit_ids)
    x_full_json = json.dumps(data.x_full)
    y_full_json = json.dumps(data.y_full)
    spike_xs_above_json = json.dumps(data.spike_xs_above)
    spike_ys_above_json = json.dumps(data.spike_ys_above)
    spike_xs_below_json = json.dumps(data.spike_xs_below)
    spike_ys_below_json = json.dumps(data.spike_ys_below)
    counts_above_json = json.dumps(data.counts_above)
    counts_below_json = json.dumps(data.counts_below)
    trace_t_json = json.dumps(data.trace_t)
    trace_ys_raw_json = json.dumps(data.trace_ys_raw)
    spike_ts_trace_above_json = json.dumps(data.spike_ts_trace_above)
    spike_y_trace_above_json = json.dumps(data.spike_y_trace_above)
    spike_ts_trace_below_json = json.dumps(data.spike_ts_trace_below)
    spike_y_trace_below_json = json.dumps(data.spike_y_trace_below)

    # Speed units are always pixels/s
    speed_units = "px/s"

    html = load_template("browse_place").format(
        s_threshold=s_threshold,
        speed_threshold=speed_threshold,
        speed_units=speed_units,
        unit_ids_json=unit_ids_json,
        x_full_json=x_full_json,
        y_full_json=y_full_json,
        spike_xs_above_json=spike_xs_above_json,
        spike_ys_above_json=spike_ys_above_json,
        spike_xs_below_json=spike_xs_below_json,
        spike_ys_below_json=spike_ys_below_json,
        counts_above_json=counts_above_json,
        counts_below_json=counts_below_json,
        trace_t_json=trace_t_json,
        trace_ys_raw_json=trace_ys_raw_json,
        spike_ts_trace_above_json=spike_ts_trace_above_json,
        spike_y_trace_above_json=spike_y_trace_above_json,
        spike_ts_trace_below_json=spike_ts_trace_below_json,
        spike_y_trace_below_json=spike_y_trace_below_json,
        max_proj_footprints_imgs_json=json.dumps(max_proj_footprints_imgs or {}),
        has_max_proj_footprints=json.dumps(
            max_proj_footprints_imgs is not None and len(max_proj_footprints_imgs) > 0
        ),
    )

    out_html = Path("export") / f"{output_prefix}.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html = out_html.resolve()
    out_html.write_text(html, encoding="utf-8")

    return out_html


def _run_browse_place(
    spike_place: Path,
    spike_index: Path | None,
    neural_timestamp: Path,
    behavior_position: Path,
    behavior_timestamp: Path,
    bodypart: str,
    behavior_fps: float,
    speed_threshold: float,
    speed_window_frames: int,
    s_threshold: float,
    neural_path: Path | None,
    trace_name: str,
    trace_name_lp: str,
    neural_fps: float,
    output_prefix: str,
    deconv_zarr: Path | None,
) -> None:
    """Internal function: Browser for place fields.

    Shows trace + trajectory + spike locations per unit.
    """

    data = prepare_place_browser_data(
        spike_place=spike_place,
        spike_index=spike_index,
        neural_timestamp=neural_timestamp,
        behavior_position=behavior_position,
        behavior_timestamp=behavior_timestamp,
        bodypart=bodypart,
        behavior_fps=behavior_fps,
        speed_threshold=speed_threshold,
        speed_window_frames=speed_window_frames,
        s_threshold=s_threshold,
        neural_path=neural_path,
        trace_name=trace_name,
        trace_name_lp=trace_name_lp,
        neural_fps=neural_fps,
        deconv_zarr=deconv_zarr,
    )

    # Generate max projection and footprints images for each unit as base64
    max_proj_footprints_imgs = {}
    if neural_path is not None:
        try:
            import base64
            import io

            from placecell.visualization import plot_max_projection_with_unit_footprint

            # Generate image for each unit in the data
            for unit_id in data.unit_ids:
                try:
                    fig = plot_max_projection_with_unit_footprint(
                        neural_path=neural_path,
                        unit_id=unit_id,
                    )
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    buf.seek(0)
                    img_data = base64.b64encode(buf.read()).decode("utf-8")
                    max_proj_footprints_imgs[unit_id] = f"data:image/png;base64,{img_data}"
                except Exception:
                    # Skip units that can't be visualized
                    continue
        except Exception as exc:
            logger.warning(f"Could not generate max projection images: {exc}")

    out_html = generate_place_browser_html(
        data=data,
        s_threshold=s_threshold,
        speed_threshold=speed_threshold,
        output_prefix=output_prefix,
        max_proj_footprints_imgs=max_proj_footprints_imgs,
    )

    logger.info(f"Wrote place browser HTML to: {out_html}")

    try:
        webbrowser.open(out_html.as_uri())
    except Exception:
        logger.info("Could not open browser automatically; open the HTML file manually.")


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
def spike_place(
    config: Path,
    spike_index: Path,
    neural_path: Path,
    behavior_path: Path,
    out: Path | None,
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
    )


@click.command(name="generate-html")
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
    help="Spike-place CSV from spike-place step.",
)
@click.option(
    "--spike-index",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Spike index CSV (optional, for trace overlay).",
)
@click.option(
    "--neural-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing neural data.",
)
@click.option(
    "--behavior-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing behavior data.",
)
@click.option(
    "--deconv-zarr",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Deconvolution zarr from deconvolve step (optional).",
)
@click.option(
    "--out-prefix",
    type=str,
    default=None,
    help="Output prefix for HTML file. Defaults to output/place_browser_<timestamp>",
)
def generate_html(
    config: Path,
    spike_place_path: Path,
    spike_index: Path | None,
    neural_path: Path,
    behavior_path: Path,
    deconv_zarr: Path | None,
    out_prefix: str | None,
) -> None:
    """Generate interactive place browser HTML."""
    if out_prefix is None:
        out_prefix = f"output/place_browser_{_default_timestamp()}"

    cfg = AppConfig.from_yaml(config)
    if cfg.behavior is None:
        raise click.ClickException("Config file must include a 'behavior' section.")

    _run_browse_place(
        spike_place=spike_place_path,
        spike_index=spike_index,
        neural_timestamp=neural_path / "neural_timestamp.csv",
        behavior_position=behavior_path / "behavior_position.csv",
        behavior_timestamp=behavior_path / "behavior_timestamp.csv",
        bodypart=cfg.behavior.bodypart,
        behavior_fps=cfg.behavior.behavior_fps,
        speed_threshold=cfg.behavior.speed_threshold,
        speed_window_frames=cfg.behavior.speed_window_frames,
        s_threshold=cfg.neural.s_threshold,
        neural_path=neural_path,
        trace_name=cfg.neural.trace_name,
        trace_name_lp="C_lp",
        neural_fps=cfg.neural.data.fps,
        output_prefix=out_prefix,
        deconv_zarr=deconv_zarr,
    )


@click.command(name="analyze")
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
def analyze(
    config: Path,
    neural_path: Path,
    behavior_path: Path,
    out_dir: Path,
    label: str | None,
) -> None:
    """Run deconvolution, spike-place matching, and place browser from one config."""

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
    ]
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

    # 3) Place browser (internal function)
    click.echo("=== Place browser ===")
    _run_browse_place(
        spike_place=out_dir / f"spike_place_{label}.csv",
        spike_index=out_dir / f"spike_index_{label}.csv",
        neural_timestamp=neural_timestamp,
        behavior_position=behavior_position,
        behavior_timestamp=behavior_timestamp,
        bodypart=bodypart,
        behavior_fps=cfg.behavior.behavior_fps,
        speed_threshold=cfg.behavior.speed_threshold,
        speed_window_frames=cfg.behavior.speed_window_frames,
        s_threshold=cfg.neural.s_threshold,
        neural_path=neural_path,
        trace_name=cfg.neural.trace_name,
        trace_name_lp="C_lp",
        neural_fps=cfg.neural.data.fps,
        output_prefix=f"{label}_place_browser",
        deconv_zarr=out_dir / f"{label}_oasis_deconv.zarr",
    )
