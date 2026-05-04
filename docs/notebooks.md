# Notebooks

Three notebooks ship in `notebook/`. The two results viewers consume `.pcellbundle` directories produced by `placecell analysis`; the calcium trace viewer reads a Minian output directly and does not need a bundle. All three need **Jupyter Lab** for the interactive widgets:

```bash
cd notebook && jupyter lab --no-browser
```

## `view_results_arena.ipynb` — 2D arena viewer

Loads one or more arena `.pcellbundle` directories and gives you:

- **Cross-session summary table**: place-cell counts, significance, and stability fractions across loaded bundles.
- **Behavior preview**: trajectory density and speed distribution per session.
- **Trajectory + rate map gallery**: side-by-side trajectory-with-events and rate-map panels for selected units.
- **Speed + place cell traces**: animal speed over time with the top-SI place cell traces overlaid.
- **Interactive unit browser**: scroll through units, see the rate map, place field contour, SI shuffle null, stability splits, and trace view.

Edit `BUNDLE_PATHS` at the top to point at your bundle(s). For a single session, leave one entry; for cross-session comparison, list multiple.

## `view_results_maze.ipynb` — 1D maze viewer

Same shape as the arena viewer but tailored to the 1D pipeline. Adds:

- **Per-unit shuffle browser**: 1D rate map with the per-bin shuffle null overlaid.
- **Cell event overlay**: events painted on the maze graph by zone, useful for spotting arm-specific tuning.
- **Zone occupancy**: arm/room occupancy bar chart.
- **Cross-session occupancy** (mean ± SD across bundles): used to verify behavior consistency before pooling place-cell statistics.

Same `BUNDLE_PATHS` convention as the arena viewer.

## `plot_calcium_traces.ipynb` — raw trace inspection

Standalone Minian viewer — does *not* depend on `.pcellbundle` or the placecell pipeline. Reads `C.zarr`, `A.zarr`, and `max_proj.zarr` from a Minian output directory and produces:

1. Full vertically-stacked traces with auto-distributed zoom panels.
2. Max projection overlaid with filled colored spatial footprints (colors match the trace panels).

Edit `MINIAN_DIR`, `FS_HZ`, and the zoom config at the top of the notebook. Useful for sanity-checking deconvolution-input quality before kicking off `placecell analysis`.
