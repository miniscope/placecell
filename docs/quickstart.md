# Quickstart

`placecell` supports two workflows:

- `arena` for 2D open-field analysis
- `maze` for 1D arm/graph-based analysis

Use the notebook or Python API after preparing the right data files for your workflow.

## Required Files

**Neural data directory** (e.g., `minian/` output):
- `{trace_name}.zarr`: calcium traces (e.g., `C.zarr` or `C_lp.zarr`)
- `A.zarr`: spatial footprints for cell overlay (optional)
- `max_proj.zarr`: max projection image for visualization (optional)

**Always required:**
- `neural_timestamp.csv`: neural frame timestamps
- `behavior_timestamp.csv`: behavior frame timestamps

**Arena (2D) behavior input:**
- `behavior_position.csv`: animal position with bodypart columns (DeepLabCut format)

**Maze (1D) behavior input:**
- `behavior_position.csv`: raw DLC tracking CSV (`x`, `y`, `likelihood`)
- `behavior_graph.yaml`: zone polygons + adjacency graph
- `arm_order` in `data_paths.yaml`

**Configuration files:**
- `config.yaml`: analysis parameters
- `data_paths.yaml`: paths to your data files

## Setup

### 1. Create data paths config

Create `data_paths.yaml` with paths relative to this file:

A session config has two optional top-level blocks — `neural:` and `behavior:` — and at least one must be present. Omit either block to run that side independently (e.g. neural-only deconvolution, or behavior-only trajectory preprocessing).

:::{dropdown} arena data_paths.yaml
```yaml
neural:
  path: path/to/minian_output
  timestamp: path/to/neural_timestamp.csv
behavior:
  type: arena
  fps: 20.0
  position: path/to/behavior_position.csv
  timestamp: path/to/behavior_timestamp.csv
  bodypart: LED
```
:::

:::{dropdown} maze data_paths.yaml
```yaml
neural:
  path: path/to/minian_output
  timestamp: path/to/neural_timestamp.csv
behavior:
  type: maze
  fps: 20.0
  position: path/to/behavior_position.csv  # raw DLC output
  timestamp: path/to/behavior_timestamp.csv
  bodypart: LED
  behavior_graph: path/to/behavior_graph.yaml  # zone polygons + adjacency
  # zone_tracking: path/to/zone_tracking.csv   # optional; defaults to zone_tracking_{stem}.csv
  arm_order:
    - Arm_1
    - Arm_2
    - Arm_3
    - Arm_4
```
:::

`placecell` is scorer-agnostic for DLC-style CSVs; configure the correct `bodypart`, and the scorer name is read from the file header.

### 1b. Maze: zone detection

For maze sessions, the analysis pipeline projects the trajectory onto a zone graph at the neural sample rate. `placecell analysis` will run zone detection automatically on first use, so for a basic run you can skip ahead. See [CLI Workflows](workflows.md) for the `define-zones` → `detect-zones` flow when you want to author the zone graph or iterate on detection parameters.

### 2. Create analysis config

Create `config.yaml` with analysis parameters:

:::{dropdown} arena config.yaml
```yaml
neural:
  fps: 20.0
  trace_name: C
  oasis:
    g: [1.60, -0.63]
    baseline: p10
    penalty: 0
behavior:
  type: arena
  speed_threshold: 10.0
  speed_window_seconds: 0.25
  spatial_map_2d:
    bins: 50
    min_occupancy: 0.05
    spatial_sigma: 3
    n_shuffles: 500
    p_value_threshold: 0.05
```
:::

:::{dropdown} maze config.yaml
```yaml
neural:
  fps: 20.0
  trace_name: C_lp
  oasis:
    g: [1.60, -0.63]
    baseline: p10
    penalty: 0.8
behavior:
  type: maze
  speed_threshold: 25
  speed_window_seconds: 0.25
  spatial_map_1d:
    bin_width_mm: 10
    min_occupancy: 0.025
    spatial_sigma: 2
    n_shuffles: 500
    p_value_threshold: 0.05
    split_by_direction: true
    require_complete_traversal: true
```
:::

### 3. Run the analysis

```bash
placecell analysis -c config.yaml -d data_paths.yaml
```

Or via Python:

```python
from placecell.dataset import BasePlaceCellDataset

ds = BasePlaceCellDataset.from_yaml("config.yaml", "data_paths.yaml")
ds.load()
ds.preprocess_behavior()
ds.deconvolve()
ds.match_events()
ds.compute_occupancy()
ds.analyze_units()
ds.save_bundle("output/session_name")
```

For batch processing, see `examples/batch_analysis.py`.

## Output

The pipeline saves a `.pcellbundle` directory containing all results and summary figures. Key outputs:

- `canonical.parquet` — per-neural-frame table with position, speed, and deconvolved activity per unit
- `figures/occupancy.pdf` — trajectory density and occupancy with split-half comparison
- `figures/behavior_preview.pdf` — trajectory and speed distribution
- `figures/diagnostics.pdf` — SI and stability distributions
- `figures/summary_scatter.pdf` — SI vs stability with place cell classification
- `figures/speed_traces.pdf` — speed and place cell traces over time

To browse results interactively, see [Notebooks](notebooks.md) for the three viewers (arena, maze, raw calcium traces).

See [Pipeline Details](pipeline.md) for the full list of summary figures and how the analysis works, and [CLI Workflows](workflows.md) for the analysis command's flags, batch mode, and the maze-specific commands.
