"""Generate regression test assets for the 1D maze pipeline.

Run once on a machine with access to the ProcData drive::

    python tests/generate_regression_data_1d.py

This creates ``tests/assets/regression_1d/`` containing a small data
subset (10 neural units) plus a reference ``.pcellbundle`` produced
by running the full MazeDataset pipeline on that subset.
"""

import shutil
from pathlib import Path

import pandas as pd
import xarray as xr
import yaml

# ── Source data ──────────────────────────────────────────────────────

SOURCE_CONFIG_ID = "pcell_maze_config"
SOURCE_DATA_PATH = Path(
    "/Volumes/ProcData/minizero_analysis/202512round/"
    "202512_analysis_3dmaze/20251219/WL25/WL25_20251219.yaml"
)

N_UNITS = 10  # number of neural units to keep in subset
N_FRAMES = 10000  # number of frames to keep (neural + behavior)

# ── Output directory ────────────────────────────────────────────────

OUT_DIR = Path(__file__).parent / "assets" / "regression_1d"


def main() -> None:  # noqa: D103
    from placecell.config import AnalysisConfig, DataPathsConfig

    # Load source configs
    cfg = AnalysisConfig.from_id(SOURCE_CONFIG_ID)
    data_path = Path(SOURCE_DATA_PATH)
    data_dir = data_path.parent
    data_cfg = DataPathsConfig.from_yaml(data_path)

    # ── Prepare output directory ────────────────────────────────────
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    neural_out = OUT_DIR / "neural_data"
    neural_out.mkdir()

    # ── Subset neural data (first N_UNITS) ──────────────────────────
    src_neural = data_dir / data_cfg.neural_path
    trace_name = cfg.neural.trace_name  # e.g. "C_lp"

    # Helper: open a zarr DataArray regardless of internal variable name
    def _open_zarr_da(path: Path, var_name: str) -> xr.DataArray:
        ds = xr.open_zarr(path, consolidated=False)
        if var_name in ds:
            return ds[var_name]
        # Fallback: single-variable dataset
        data_vars = list(ds.data_vars)
        if len(data_vars) == 1:
            return ds[data_vars[0]]
        raise KeyError(f"Cannot find {var_name!r} in {path}; vars={data_vars}")

    # Helper: save a DataArray back as a zarr Dataset with the same var name
    def _save_zarr_da(da: xr.DataArray, path: Path, var_name: str) -> None:
        # Drop non-essential coords (animal, session) to avoid string dtype issues
        drop = [c for c in da.coords if c not in da.dims]
        da = da.drop_vars(drop).load()
        da.to_dataset(name=var_name).to_zarr(path, mode="w", zarr_format=2)

    # C.zarr (raw calcium traces)
    c_src = _open_zarr_da(src_neural / "C.zarr", "C")
    unit_ids = c_src.coords["unit_id"].values[:N_UNITS]
    frame_ids = c_src.coords["frame"].values[:N_FRAMES]
    c_sub = c_src.sel(unit_id=unit_ids, frame=frame_ids)
    _save_zarr_da(c_sub, neural_out / "C.zarr", "C")
    print(f"C.zarr: {c_sub.shape}")

    # <trace_name>.zarr (e.g. C_lp.zarr)
    trace_path = src_neural / f"{trace_name}.zarr"
    if trace_path.exists() and trace_name != "C":
        t_src = _open_zarr_da(trace_path, trace_name)
        t_sub = t_src.sel(unit_id=unit_ids, frame=frame_ids)
        _save_zarr_da(t_sub, neural_out / f"{trace_name}.zarr", trace_name)
        print(f"{trace_name}.zarr: {t_sub.shape}")

    # A.zarr (spatial footprints)
    a_path = src_neural / "A.zarr"
    if a_path.exists():
        a_src = _open_zarr_da(a_path, "A")
        a_sub = a_src.sel(unit_id=unit_ids)
        _save_zarr_da(a_sub, neural_out / "A.zarr", "A")
        print(f"A.zarr: {a_sub.shape}")

    # max_proj.zarr — just copy the whole directory
    max_proj_path = src_neural / "max_proj.zarr"
    if max_proj_path.exists():
        shutil.copytree(max_proj_path, neural_out / "max_proj.zarr")
        print("max_proj.zarr: copied")

    # ── Subset behavior data to first N_FRAMES rows ────────────────
    beh_pos = pd.read_csv(data_dir / data_cfg.behavior_position, header=[0, 1, 2])
    beh_pos.iloc[:N_FRAMES].to_csv(OUT_DIR / "behavior_position.csv", index=False)
    print(f"behavior_position.csv: {min(N_FRAMES, len(beh_pos))} rows")

    beh_ts = pd.read_csv(data_dir / data_cfg.behavior_timestamp)
    beh_ts.iloc[:N_FRAMES].to_csv(OUT_DIR / "behavior_timestamp.csv", index=False)
    print(f"behavior_timestamp.csv: {min(N_FRAMES, len(beh_ts))} rows")

    # ── Subset neural timestamps to first N_FRAMES rows ─────────────
    neural_ts = pd.read_csv(data_dir / data_cfg.neural_timestamp)
    neural_ts.iloc[:N_FRAMES].to_csv(OUT_DIR / "neural_timestamp.csv", index=False)
    print(f"neural_timestamp.csv: {min(N_FRAMES, len(neural_ts))} rows")

    # ── Copy behavior graph YAML (if present) ───────────────────────
    if data_cfg.behavior_graph:
        src_graph = data_dir / data_cfg.behavior_graph
        if src_graph.exists():
            shutil.copy2(src_graph, OUT_DIR / "behavior_graph.yaml")
            print(f"Copied behavior_graph: {src_graph.name}")

    # ── Write analysis config (lower n_shuffles for fast test) ──────
    cfg_overridden = cfg.with_data_overrides(data_cfg)
    test_cfg = cfg_overridden.model_copy(deep=True)
    # Reduce shuffles for speed
    test_cfg.behavior.spatial_map_1d.n_shuffles = 100
    test_cfg.to_yaml(OUT_DIR / "analysis_config.yaml")

    # ── Write data paths YAML ───────────────────────────────────────
    data_paths_dict = {
        "neural_path": "neural_data",
        "neural_timestamp": "neural_timestamp.csv",
        "behavior_position": "behavior_position.csv",
        "behavior_timestamp": "behavior_timestamp.csv",
    }
    if data_cfg.behavior_graph:
        data_paths_dict["behavior_graph"] = "behavior_graph.yaml"

    with open(OUT_DIR / "data_paths.yaml", "w") as f:
        yaml.dump(data_paths_dict, f, default_flow_style=False)

    # ── Run the full pipeline and save reference bundle ─────────────
    from placecell.dataset import BasePlaceCellDataset

    ds = BasePlaceCellDataset.from_yaml(
        OUT_DIR / "analysis_config.yaml",
        OUT_DIR / "data_paths.yaml",
    )
    print(f"Dataset type: {type(ds).__name__}")

    ds.load()
    ds.preprocess_behavior()
    ds.deconvolve()
    ds.match_events()
    ds.compute_occupancy()
    ds.analyze_units()

    print(ds.summary())

    bundle_path = ds.save_bundle(OUT_DIR / "reference.pcellbundle")
    print(f"\nReference bundle saved to: {bundle_path}")
    print(f"Total assets in: {OUT_DIR}")


if __name__ == "__main__":
    main()
