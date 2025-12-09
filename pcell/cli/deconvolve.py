"""Deconvolution CLI command."""

from pathlib import Path
from typing import Tuple

import click
import numpy as np
import xarray as xr
from tqdm import tqdm

from pcell.config import AppConfig
from pcell.curation import load_traces


def _parse_g(ctx, param, value: Tuple[float, float]) -> Tuple[float, float]:
    if len(value) != 2:
        raise click.BadParameter("g must be two floats: g1 g2")
    return float(value[0]), float(value[1])


@click.command(name="deconvolve")
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
    "--g",
    nargs=2,
    type=float,
    required=False,
    callback=lambda ctx, param, value: _parse_g(ctx, param, value) if value else None,
    help="Optional AR(2) coefficients g1 g2 for OASIS. "
    "If omitted, oasis-deconv will estimate AR parameters from the data.",
)
@click.option(
    "--s-min",
    type=float,
    default=1.0,
    show_default=True,
    help="Minimum spike size (OASIS s_min).",
)
@click.option(
    "--baseline",
    type=str,
    default="p10",
    show_default=True,
    help="Baseline mode: 'pXX' percentile or a numeric value.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    prompt="Output directory for deconvolution results",
)
@click.option(
    "--label",
    type=str,
    default="session",
    show_default=True,
    help="Label used in the output zarr folder name (e.g. 'WL25_DEC1').",
)
@click.option(
    "--max-units",
    type=int,
    default=None,
    show_default=False,
    help="Optional maximum number of units to deconvolve (after curation filtering).",
)
@click.option(
    "--curated-units-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("export/curated_units.txt"),
    show_default=True,
    help="Optional text file with curated unit IDs (one per line). If it exists, "
    "only those units will be deconvolved.",
)
@click.option(
    "--spike-index-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("export/spike_index.csv"),
    show_default=True,
    help="Optional CSV file to write spike indices as rows (unit_id, frame, s). "
    "Set to '' to disable.",
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    show_default=False,
    help="Optional YAML config file overriding data settings.",
)
def deconvolve(
    neural_path: Path,
    trace_name: str,
    fps: float,
    g: Tuple[float, float] | None,
    s_min: float,
    baseline: str,
    out_dir: Path,
    label: str,
    max_units: int | None,
    curated_units_file: Path,
    spike_index_out: Path,
    config: Path | None,
) -> None:
    """Run OASIS AR(2) deconvolution on all units and save results as zarr."""

    # Lazy, guarded import so that `pcell` / `pcell curate` still work
    # even if oasis-deconv is not available on this system.
    try:
        from oasis.functions import deconvolve as oasis_deconvolve  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        raise click.ClickException(
            "Could not import 'oasis-deconv' (oasis.functions.deconvolve).\n"
            "Deconvolution requires the oasis-deconv package and its C++ "
            "dependencies to be installed and loadable.\n\n"
            f"Original error:\n{exc}"
        )

    # Optional YAML config
    if config is not None:
        cfg = AppConfig.from_yaml(config)
        trace_name = cfg.curation.data.trace_name
        fps = cfg.curation.data.fps
        if cfg.curation.max_units is not None:
            max_units = (
                min(max_units, cfg.curation.max_units) if max_units else cfg.curation.max_units
            )
    else:
        cfg = None

    if neural_path is None:
        raise click.ClickException("--neural-path is required. Specify the directory containing neural data (C.zarr).")

    neural_path = neural_path.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Using neural data at: {neural_path}")

    # Use the same loader as visualization, to handle Dataset/DataArray nicely.
    click.echo(f"Loading traces from: {neural_path / (trace_name + '.zarr')}")
    C_da = load_traces(neural_path, trace_name=trace_name)

    all_unit_ids = list(map(int, C_da["unit_id"].values))

    # Optional curation filter
    selected_unit_ids = all_unit_ids
    curated_path = curated_units_file.resolve()
    if curated_path.is_file():
        try:
            curated = np.loadtxt(curated_path, dtype=int)
            if curated.ndim == 0:
                curated = curated[None]
            curated_set = set(int(x) for x in curated.tolist())
            selected_unit_ids = [uid for uid in all_unit_ids if uid in curated_set]
            if not selected_unit_ids:
                raise click.ClickException(
                    f"Curated units file {curated_path} did not match any unit_id in data."
                )
            click.echo(
                f"Using {len(selected_unit_ids)} curated units from {curated_path} "
                f"(out of {len(all_unit_ids)} available)."
            )
        except Exception as exc:
            raise click.ClickException(f"Failed to read curated units from {curated_path}: {exc}")
    else:
        # No curated file exists - prompt for range
        click.echo(
            f"Found {len(all_unit_ids)} units (IDs: {all_unit_ids[0]} to {all_unit_ids[-1]})"
        )
        click.echo("No curated units file found. Select a range of units to process.")

        start_idx = click.prompt(
            f"Start index (0 to {len(all_unit_ids) - 1})",
            type=int,
            default=0,
        )
        end_idx = click.prompt(
            f"End index ({start_idx} to {len(all_unit_ids) - 1})",
            type=int,
            default=len(all_unit_ids) - 1,
        )

        # Validate range
        if start_idx < 0 or start_idx >= len(all_unit_ids):
            raise click.ClickException(
                f"Start index {start_idx} is out of range [0, {len(all_unit_ids) - 1}]"
            )
        if end_idx < start_idx or end_idx >= len(all_unit_ids):
            raise click.ClickException(
                f"End index {end_idx} is out of range [{start_idx}, {len(all_unit_ids) - 1}]"
            )

        selected_unit_ids = all_unit_ids[start_idx : end_idx + 1]
        click.echo(
            f"Selected {len(selected_unit_ids)} units (indices {start_idx} to {end_idx}, "
            f"unit IDs: {selected_unit_ids[0]} to {selected_unit_ids[-1]})"
        )

    # Optional max_units limiter (after curation)
    if max_units is not None and len(selected_unit_ids) > max_units:
        selected_unit_ids = selected_unit_ids[:max_units]
        click.echo(f"Limiting to first {len(selected_unit_ids)} units due to --max-units.")

    unit_ids = selected_unit_ids
    if g is not None:
        g1, g2 = float(g[0]), float(g[1])
        click.echo(
            f"Running OASIS deconvolution on {len(unit_ids)} units "
            f"(user-specified initial g=({g1}, {g2}))"
        )
    else:
        click.echo(
            f"Running OASIS deconvolution on {len(unit_ids)} units "
            "(letting oasis-deconv estimate AR parameters)."
        )

    results = {}
    good_unit_ids: list[int] = []
    C_list: list[np.ndarray] = []
    S_list: list[np.ndarray] = []

    for uid in tqdm(unit_ids, desc="Deconvolving units", unit="unit"):
        y = np.ascontiguousarray(C_da.sel(unit_id=uid).values, dtype=np.float64)

        # baseline
        if isinstance(baseline, str) and baseline.startswith("p"):
            p = float(baseline[1:])
            b = float(np.percentile(y, p))
        else:
            try:
                b = float(baseline)
            except ValueError:
                raise click.ClickException(
                    f"Could not interpret baseline={baseline!r} " "as 'pXX' or numeric value."
                )

        # Use oasis-deconv high-level API.
        # If g was provided, pass it as an initial AR(2) guess; otherwise
        # let oasis-deconv estimate AR parameters from the data.
        kwargs: dict = {"penalty": 0}
        if g is not None:
            kwargs["g"] = (g1, g2)
        try:
            c, s, b_est, g_est, lam = oasis_deconvolve(y - b, **kwargs)
        except Exception as exc:
            click.echo(f"Skipping unit {uid} due to oasis-deconv error: {exc}")
            continue

        good_unit_ids.append(int(uid))
        C_list.append(np.asarray(c, dtype=float))
        S_list.append(np.asarray(s, dtype=float))
        results[uid] = dict(
            raw=y,
            c=c,
            s=s,
            b=b,
            g=np.asarray(g_est, dtype=np.float64),
            time=np.arange(len(y), dtype=np.float64) / float(fps),
        )

    if not good_unit_ids:
        raise click.ClickException("OASIS deconvolution failed for all units.")

    click.echo("OASIS finished; building xarray dataset.")

    # Build xarray.Dataset similar to the notebook's zarr output
    unit_idx = np.array(good_unit_ids, dtype=int)
    T = C_da.sizes.get("frame", len(C_list[0]))

    C_deconv = np.stack(C_list, axis=0)
    S = np.stack(S_list, axis=0)

    ds = xr.Dataset(
        {
            "C_deconv": (("unit_id", "frame"), C_deconv),
            "S": (("unit_id", "frame"), S),
        },
        coords={
            "unit_id": unit_idx,
            "frame": np.arange(T, dtype=int),
        },
    )
    ds.attrs.update(
        {
            "fps": float(fps),
            "g": "estimated" if g is None else [g1, g2],
            "s_min": float(s_min),
            "baseline": baseline,
        }
    )

    out_path = out_dir / f"{label}_oasis_deconv.zarr"
    click.echo(f"Saving deconvolution results to: {out_path}")
    ds.to_zarr(out_path, mode="w")

    # Optional spike-index CSV
    if str(spike_index_out):
        spike_index_out = spike_index_out.resolve()
        spike_index_out.parent.mkdir(parents=True, exist_ok=True)
        with spike_index_out.open("w", encoding="utf-8") as f:
            f.write("unit_id,frame,s\n")
            for i, uid in enumerate(unit_idx):
                s_vec = S[i]
                frames = np.nonzero(s_vec > 0)[0]
                for fr in frames:
                    f.write(f"{int(uid)},{int(fr)},{float(s_vec[fr])}\n")
        click.echo(f"Wrote spike index CSV to: {spike_index_out}")

    click.echo("Done.")
