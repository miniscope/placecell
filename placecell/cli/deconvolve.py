"""Deconvolution CLI command."""

from datetime import datetime
from pathlib import Path

import click
import numpy as np
import xarray as xr
from mio.logging import init_logger
from tqdm import tqdm

from placecell.analysis import load_traces
from placecell.config import AppConfig

logger = init_logger(__name__)


def _default_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@click.command(name="deconvolve")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="YAML config file with neural and OASIS settings.",
)
@click.option(
    "--neural-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory containing neural data (C.zarr).",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("output"),
    show_default=True,
    help="Output directory for deconvolution results.",
)
@click.option(
    "--label",
    type=str,
    default=None,
    help="Label used in output filenames. Defaults to timestamp.",
)
@click.option(
    "--spike-index-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="CSV file to write spike indices. Defaults to <out-dir>/spike_index_<label>.csv",
)
@click.option(
    "--start-idx",
    type=int,
    default=0,
    show_default=True,
    help="Start unit index.",
)
@click.option(
    "--end-idx",
    type=int,
    default=None,
    help="End unit index (inclusive). Defaults to last unit.",
)
def deconvolve(
    config: Path,
    neural_path: Path,
    out_dir: Path,
    label: str | None,
    spike_index_out: Path | None,
    start_idx: int,
    end_idx: int | None,
) -> None:
    """Run OASIS deconvolution using settings from config file."""

    try:
        from oasis.functions import deconvolve as oasis_deconvolve  # type: ignore
    except Exception as exc:
        raise click.ClickException(f"Could not import oasis-deconv: {exc}") from exc

    if label is None:
        label = _default_label()

    cfg = AppConfig.from_yaml(config)
    trace_name = cfg.neural.trace_name
    fps = cfg.neural.data.fps
    max_units = cfg.neural.max_units
    g = cfg.neural.oasis.g
    baseline = cfg.neural.oasis.baseline

    neural_path = neural_path.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if spike_index_out is None:
        spike_index_out = out_dir / f"spike_index_{label}.csv"

    logger.info(f"Using neural data at: {neural_path}")

    # Use the same loader as visualization, to handle Dataset/DataArray nicely.
    logger.info(f"Loading traces from: {neural_path / (trace_name + '.zarr')}")
    C_da = load_traces(neural_path, trace_name=trace_name)

    all_unit_ids = list(map(int, C_da["unit_id"].values))

    if end_idx is None:
        end_idx = len(all_unit_ids) - 1

    if (
        start_idx < 0
        or start_idx >= len(all_unit_ids)
        or end_idx < start_idx
        or end_idx >= len(all_unit_ids)
    ):
        raise click.ClickException(
            f"Invalid range: start_idx={start_idx}, end_idx={end_idx}, "
            f"valid range is 0-{len(all_unit_ids) - 1}"
        )

    unit_ids = all_unit_ids[start_idx : end_idx + 1]
    if max_units is not None and len(unit_ids) > max_units:
        unit_ids = unit_ids[:max_units]
        logger.info(f"Limiting to first {max_units} units due to max_units config.")
    if g is not None:
        logger.info(f"Running OASIS on {len(unit_ids)} units (g={g})")
    else:
        logger.info(f"Running OASIS on {len(unit_ids)} units (estimating AR params)")

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
                ) from None

        kwargs: dict = {"penalty": 0}
        if g is not None:
            kwargs["g"] = g
        try:
            c, s, b_est, g_est, lam = oasis_deconvolve(y - b, **kwargs)
        except Exception as exc:
            logger.warning(f"Skipping unit {uid} due to oasis-deconv error: {exc}")
            continue

        good_unit_ids.append(int(uid))
        C_list.append(np.asarray(c, dtype=float))
        S_list.append(np.asarray(s, dtype=float))

    if not good_unit_ids:
        raise click.ClickException("OASIS deconvolution failed for all units.")

    logger.info("OASIS finished; building xarray dataset.")

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
            "g": "estimated" if g is None else list(g),
            "baseline": baseline,
        }
    )

    out_path = out_dir / f"{label}_oasis_deconv.zarr"
    logger.info(f"Saving deconvolution results to: {out_path}")
    ds.to_zarr(out_path, mode="w")

    # Write spike-index CSV
    spike_index_out = spike_index_out.resolve()
    spike_index_out.parent.mkdir(parents=True, exist_ok=True)
    with spike_index_out.open("w", encoding="utf-8") as f:
        f.write("unit_id,frame,s\n")
        for i, uid in enumerate(unit_idx):
            s_vec = S[i]
            frames = np.nonzero(s_vec > 0)[0]
            for fr in frames:
                f.write(f"{int(uid)},{int(fr)},{float(s_vec[fr])}\n")
    logger.info(f"Wrote spike index CSV to: {spike_index_out}")

    logger.info("Done.")
