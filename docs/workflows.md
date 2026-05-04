# CLI Workflows

The CLI exposes three commands. Auto-generated flag reference lives in the [CLI Reference](cli.md); this page covers what each command is *for* and how they fit together. Run `placecell <command> --help` for the full flag list.

## `placecell analysis` — run the pipeline

One-shot driver for the analysis pipeline described in [Pipeline Details](pipeline.md). Loads neural + behavior, deconvolves, builds the canonical table, computes occupancy, and writes a `.pcellbundle` with results and summary figures.

```bash
placecell analysis -c config.yaml -d data_paths.yaml
```

By default it runs through `compute_occupancy()`, saves a QC bundle, then prompts `Proceed with analyze_units?` so you can inspect figures before paying for the per-unit shuffle tests. Pass `-y` to skip the prompt and run end-to-end, or `--show` to open the QC PDFs in your default viewer before answering.

**Batch mode**: repeat `-d` for multiple sessions. Each is processed independently and written to its own bundle:

```bash
placecell analysis -c config.yaml -d session_a.yaml -d session_b.yaml -y
```

**Output location**: defaults to `./output/{data_path.stem}.pcellbundle` next to where you ran the command. Override with `-o`.

**Other useful flags**:

- `-w, --workers N` — parallelize `analyze_units()` across N processes. The shuffle test is the dominant cost.
- `--subset-units N` / `--subset-frames N` — analyze only the first N units / N frames. Used to generate small fixtures (the regression bundles in `tests/assets/` are produced this way).
- `--force-redetect` — re-run zone detection even when the cached `zone_tracking` CSV exists (maze only; ignored for arena). Use after editing `zone_detection` parameters.

## `placecell define-zones` — draw the maze graph (maze only)

Interactive OpenCV tool for authoring `behavior_graph.yaml` — the zone polygons + adjacency graph that detect-zones projects trajectory onto. Loads a behavior video frame and lets you click polygon vertices for each room and arm.

```bash
placecell define-zones -d data_paths.yaml --rooms 3 --arms 4
```

Requires `behavior.video` set in the data config. If `behavior.behavior_graph` is unset, the new YAML is written next to `data_paths.yaml` and the data config is updated in place to point at it. Run once per camera setup, not per session.

## `placecell detect-zones` — project trajectory onto the graph (maze only)

Projects the raw DLC `(x, y)` trajectory onto the maze graph at the neural sample rate, runs the zone state machine, and writes `zone_tracking.csv` (one row per neural frame with `x, y, x_pinned, y_pinned, zone, arm_position, neural_time`). This is the file `MazeDataset.load()` reads.

```bash
placecell detect-zones -d data_paths.yaml
```

Output defaults to `zone_tracking_{data_path.stem}.csv` next to the data config. Also exports a validation video (subsampled by `--interpolate`, sped up by `--playback-speed`) so you can scrub through and visually verify the zone assignments.

`placecell analysis` calls this automatically when the cached CSV is missing, so you only need to invoke it directly when:

1. You want to inspect the validation video before running analysis, or
2. You're iterating on `zone_detection` parameters and want a fast loop without paying for deconvolution.

## Maze workflow order

For a brand-new session:

1. `placecell define-zones -d data_paths.yaml --rooms <n> --arms <n>` — draw polygons (one-time per camera setup).
2. `placecell detect-zones -d data_paths.yaml` — produces `zone_tracking.csv` and the validation video. Skip this if you trust the cached CSV.
3. `placecell analysis -c config.yaml -d data_paths.yaml` — full pipeline.

Arena workflow skips steps 1–2.

## Programmatic / batch use

For Python-driven batch runs, see `examples/batch_analysis.py`. It mirrors the CLI but exposes intermediate state (so you can inject custom logic between `match_events()` and `analyze_units()`, or swap out individual steps). The key entry point is:

```python
from placecell.dataset import BasePlaceCellDataset

ds = BasePlaceCellDataset.from_yaml("config.yaml", "data_paths.yaml")
ds.load()
ds.preprocess_behavior()
ds.deconvolve()
ds.match_events()
ds.compute_occupancy()
ds.analyze_units(n_workers=4)
ds.save_bundle("output/session")
```

For independent neural-only or behavior-only sessions, omit the unused block in `data_paths.yaml`; `load()` and the relevant single-side steps run, the others are no-ops, and `match_events()` raises with a targeted message if called.
