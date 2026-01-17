# Quickstart

Run spatial neural activity analysis with a single command:

```bash
pcell workflow visualize --config config.yaml --data data_paths.yaml
```

## Required Files

**Input data:**
- `{trace_name}.zarr`: calcium traces from your neural data directory
- `neural_timestamp.csv`: neural frame timestamps
- `behavior_position.csv`: animal position with bodypart columns
- `behavior_timestamp.csv`: behavior frame timestamps

**Configuration files:**
- `config.yaml`: analysis parameters
- `data_paths.yaml`: paths to your data files

## Setup

### 1. Create data paths config

Create `data_paths.yaml` with paths relative to this file:

:::{dropdown} data_paths.yaml
```yaml
id: your_data
mio_model: placecell.config.DataPathsConfig
mio_version: 0.8.1
neural_path: directory/including/zarr_neural_files
neural_timestamp: path/to/neural_timestamp.csv
behavior_position: path/to/behavior_position.csv
behavior_timestamp: path/to/behavior_timestamp.csv
curation_csv: path/to/curation.csv  # optional
```
:::

### 2. Create analysis config

Create `config.yaml` with analysis parameters:

:::{dropdown} config.yaml
```yaml
id: your_config
mio_model: placecell.config.AppConfig
mio_version: 0.8.1
neural:
  id: neural
  fps: 20.0
  trace_name: C
  oasis:
    id: oasis
    g: [1.60, -0.63]
    baseline: p10
    penalty: 0
behavior:
  id: behavior
  behavior_fps: 20.0
  bodypart: LED
  speed_threshold: 10.0
  speed_window_frames: 5
  spatial_map:
    id: spatial_map
    bins: 50
    min_occupancy: 0.05
    occupancy_sigma: 3
    activity_sigma: 3
    n_shuffles: 500
    p_value_threshold: 0.05
```
:::

### 3. Run the workflow

```bash
pcell workflow visualize --config config.yaml --data data_paths.yaml
```

## Output

The workflow displays an occupancy preview:

![Occupancy Preview](assets/Figure_1.png)

Then launches the interactive place cell viewer:

![Place Cell Viewer](assets/Figure_2.png)

## CLI Options

```bash
pcell workflow visualize --help
```

Optional arguments:
- `--out-dir`: output directory (default: `output/`)
- `--label`: label for output files (default: timestamp)
- `--start-idx`, `--end-idx`: unit index range to process

## Next Steps
- See [Pipeline Details](pipeline.md) for how the analysis works
- See [CLI Reference](cli.md) for all available commands
