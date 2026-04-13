# Pipeline Details

This document explains how the spatial neural activity analysis pipeline works.

## Overview

Both `ArenaDataset` (2D) and `MazeDataset` (1D) implement the same abstract pipeline defined by `BasePlaceCellDataset`. Each step depends on the previous one.

1. `from_yaml(config, data_path)` — parse configs, auto-select ArenaDataset or MazeDataset
2. `load()` — load neural traces, behavior positions, visualization assets
3. `preprocess_behavior()` — geometric corrections (Hampel, perspective, clipping, unit conversion)
4. `deconvolve()` — OASIS AR(2) deconvolution → per-unit spike trains at neural fps
5. `match_events()` — interpolate behavior onto neural timestamps, compute speed, build canonical table
6. `compute_occupancy()` — spatial occupancy map (2D histogram or 1D bins)
7. `analyze_units()` — per-unit rate map, spatial information, stability, place fields
8. `save_bundle()` — write `.pcellbundle` directory with config, arrays, parquets, figures

The **canonical neural-rate table** (`ds.canonical`) is the central artifact: one row per neural frame with columns `frame_index, neural_time, x, y, speed, [pos_1d, arm_index, ...], s_unit_<id>...`. Behavior is linearly interpolated onto neural timestamps inside `match_events()` (or, for the maze pipeline, inside `detect-zones`), so every analysis downstream operates on a single common clock. The long-format `event_place` table is derived from `canonical` for the per-unit spatial analysis functions.

## Data Files

**Neural inputs** (in `neural_path` directory):
- `{trace_name}.zarr`: calcium traces (frames × units). Name set by `trace_name` in config (e.g. `C.zarr`, `C_lp.zarr`)
- `A.zarr`: spatial footprints for cell contour overlay (optional)
- `max_proj.zarr`: max projection image for visualization background (optional)

**Behavior inputs:**
- `neural_timestamp.csv`: neural frame timestamps (frame, timestamp_first, timestamp_last)
- `behavior_position.csv`: animal position per frame (DeepLabCut format with bodypart columns)
- `behavior_timestamp.csv`: behavior frame timestamps

**Intermediate DataFrames** (generated during pipeline):
- `canonical`: per-neural-frame table with `x, y, speed, [pos_1d, ...], s_unit_<id>...`. Single source of truth for all downstream analysis.
- `trajectory_filtered` / `trajectory_1d_filtered`: speed-filtered view of `canonical` (with `frame_index` aliased so existing analysis helpers consume it directly).
- `event_place`: long-format event table (`unit_id, frame_index, x, y, s, speed, ...`) derived from the speed-filtered canonical view. Each row is one above-zero spike sample from one unit.
- `event_index`: long-format event table over the **unfiltered** canonical view, used by the notebook browser.

## Processing Steps

### `ds.load()`

- Load calcium traces from `{trace_name}.zarr`
- Load behavior position and timestamps (speed is computed later at neural rate in `match_events`)
- Load visualization assets (max_proj, footprints, behavior video frame)
- For the maze pipeline, also auto-runs zone detection if the cached `zone_tracking` CSV is missing

### `ds.preprocess_behavior()`

Saves a copy of the raw trajectory in `trajectory_raw`, then applies geometric corrections (Hampel, perspective, clipping, unit conversion). Speed is computed later at the neural sample rate inside `match_events()`.

#### `ArenaDataset` (2D)

**When `arena_bounds` is configured** (full pipeline):

1. **Hampel jump removal**: per-frame outlier detection on the (x, y) trajectory using a centered rolling-median centroid (window=`behavior.hampel_window_frames`) and the MAD-scaled deviation `behavior.hampel_n_sigmas * 1.4826 * MAD`. Flagged frames are linearly interpolated.
2. **Perspective correction**: correct for camera angle using `camera_height_mm` and `tracking_height_mm`
3. **Boundary clipping**: clip positions to `arena_bounds`
4. **Recompute speed**: recalculate speed in mm/s from corrected positions, using a centered window of `behavior.speed_window_seconds`

**When `arena_bounds` is not configured** (fallback with warnings):

- Spatial corrections are skipped (Hampel jump removal, perspective correction, boundary clipping)
- Speed and position remain in pixels

#### `MazeDataset` (1D)

`MazeDataset.load()` reads a `zone_tracking` CSV that maps each **neural** frame to a zone label and an arm-relative position. If the CSV is missing, `load()` runs [Zone Detection](#zone-detection-maze-only) automatically; pass `--force-redetect` to `placecell analysis` (or `force_redetect=True` to `MazeDataset.load()`) to refresh it.

`preprocess_behavior()` then operates on the already-neural-rate trajectory and:

1. **Serialize to 1D**: convert `(zone, arm_position)` into a single concatenated `pos_1d` axis using physical arm lengths from `behavior_graph.yaml`.
2. **Optionally split by direction**: each contiguous arm traversal becomes `Arm_X_fwd` or `Arm_X_rev` based on its starting position (`split_by_direction`).
3. **Optionally drop incomplete traversals**: keep only room-to-room arm crossings (`require_complete_traversal`).
4. **Compute 1D speed**: centered-window (`behavior.speed_window_seconds`) finite-difference on `pos_1d` at the neural sample rate, with a same-arm guard so cross-arm transitions never inflate the speed estimate.

### `ds.match_events()`

Builds the canonical neural-rate table:

1. Load neural timestamps (`timestamp_first` per neural frame), validate with Hampel filter, exclude anomalous frames.
2. **Arena**: linearly interpolate `(x, y)` from the behavior-rate trajectory onto neural timestamps, then compute speed at neural rate. **Maze**: the trajectory is already at neural rate (interpolation happened inside zone detection), so this step is a direct join.
3. Stack the deconvolved per-unit spike trains (`S_list`) into `s_unit_<id>` columns.
4. Drop neural frames with no behavior coverage (outside behavior time window).
5. Derive the speed-filtered view (`trajectory_filtered` / `trajectory_1d_filtered`) and the long-format `event_place` / `event_index` tables.

The hard-error guardrail in :func:`placecell.behavior.interpolate_behavior_onto_neural` refuses to upsample when `neural_fps > 5 * behavior_fps`, since per-frame jitter would dominate.

### Zone Detection (maze only)

Zone detection projects raw DLC `(x, y)` onto the maze graph **at the neural sample rate**. It runs automatically from `MazeDataset.load()` when the `zone_tracking` CSV is missing, and can also be invoked directly via `placecell detect-zones -d data_paths.yaml` (which additionally exports a validation video).

1. Load raw DLC `(x, y)` from `behavior_position.csv`.
2. **Hampel jump removal** on the raw trajectory at behavior rate, using `zone_detection.hampel_window_frames` and `zone_detection.hampel_n_sigmas`.
3. **Linearly interpolate** the cleaned `(x, y)` onto the neural timestamp grid (`timestamp_first` from `neural_timestamp.csv`).
4. Compute per-zone soft-membership probability and run a state machine at the neural sample rate (`min_confidence`, `min_seconds_same`, `min_seconds_forbidden`, graph adjacency) to assign a zone label per neural frame.
5. For arm-zone frames, project the cleaned `(x, y)` onto the arm polyline to get `arm_position` (0–1) and pinned point `(x_pinned, y_pinned)`.
6. Write the resulting per-neural-frame table to the `zone_tracking` CSV with columns `x, y, x_pinned, y_pinned, zone, arm_position, neural_time` indexed by neural frame.

`zone_tracking` defaults to `zone_tracking_{data_path.stem}.csv` next to the data config when not set.

### `ds.deconvolve()`

- Run OASIS AR2 deconvolution on each unit's calcium trace
- Parameters: `g` (AR coefficients), `baseline`, `penalty`, `s_min`
- Output: `good_unit_ids`, per-unit spike trains (`S_list`)

### `ds.compute_occupancy()`

- Compute 2D occupancy histogram from `trajectory_filtered`
- Smooth with `spatial_sigma`, mask bins below `min_occupancy`
- Output: `occupancy_time`, `valid_mask`, bin edges

### `ds.analyze_units()` — per unit

Four independent computations from the same inputs (events, filtered trajectory, occupancy):

- **Rate map**: event weights / occupancy time, smoothed with `spatial_sigma` (for display)
- **Spatial information + shuffle test**: Skaggs SI with circular-shift shuffle → SI p-value
- **Shuffled rate percentile**: per-bin percentile of shuffled rate maps → used for place field seed detection
- **Split-half stability + shuffle test**: correlation between first/second half rate maps with circular-shift shuffle → stability p-value

**Place cell classification**: units with SI p-value < `p_value_threshold` AND stability p-value < `p_value_threshold`.

**Place field detection** (Guo et al. 2023):
1. **Seed detection**: bins where rate exceeds the shuffled rate percentile. Only contiguous seed regions with ≥ `place_field_min_bins` bins are kept
2. **Extension**: each seed region extends to contiguous bins with rate ≥ `place_field_threshold` × (seed's peak rate)

### Results

- **Coverage map**: sum of place field masks across place cells
- **Coverage curve**: cells sorted by field size (largest first), cumulative fraction of environment covered
- **Interactive browser**: max projection overlay, trajectory with events, rate map with place field contour, SI histogram, stability maps, trace view

## Data Integrity

The pipeline flags and excludes problematic data rather than silently repairing it. Every exclusion is logged with a count.

**Timestamp validation** (neural timestamps):
- Outliers from the local trend (Hampel filter, window=11, 3σ) → excluded
- Residual backward jumps after Hampel → excluded
- Large forward gaps (recording stalls, >1s or >10× median dt) → warned, NOT excluded (valid timestamps; interpolated positions within the gap may be unreliable)
- Only `timestamp_first` is used; `timestamp_last` is ignored (occasionally noisy)

**Temporal alignment** (behavior ↔ neural):
- Zero overlap between recordings → hard error
- Partial overlap (neural starts before or ends after behavior) → logged, uncovered frames dropped
- Neural fps > 5× behavior fps → hard error (upsampled jitter would dominate)

**Position filtering** (behavior trajectory):
- Hampel filter on raw (x, y) at behavior rate → outliers interpolated
- Out-of-arena positions → clipped to boundary (intentional for arena calibration)
- Non-numeric values in data columns → logged if any coerced to NaN

**Speed filtering**:
- Computed at neural rate via centered window (`speed_window_seconds`)
- Zero-dt frames (from timestamp exclusion) → NaN speed, not zero
- NaN speeds → logged and dropped by the speed threshold

**Analysis methods**:
- Rate maps: independent numerator/denominator Gaussian smoothing (Skaggs et al. 1996)
- Spatial information: Skaggs et al. 1993, with +1-corrected rank p-value (Phipson & Smyth 2010)
- Shuffle null: circular shift excluding zero-shift, independent RNG seeds for SI/stability/percentile
- Stability: interleaved split-half blocks, Fisher z-transform, separate shuffle stream
- Place field detection: Guo et al. 2023 seed-and-grow algorithm
- Place cell classification: dual criterion (SI p < threshold AND stability p < threshold)

## Key Parameters

- `speed_threshold`: minimum speed to include data (mm/s)
- `min_occupancy`: minimum time per bin to be valid (seconds)
- `bins`: spatial resolution (number of bins per axis)
- `spatial_sigma`: Gaussian smoothing sigma for occupancy and rate maps (in bins)
- `n_shuffles`: number of circular-shift shuffle iterations
- `min_shift_seconds`: minimum circular shift for shuffle test (seconds)
- `p_value_threshold`: p-value threshold for both SI and stability significance
- `si_weight_mode`: `"amplitude"` (event amplitudes) or `"binary"` (event counts)
- `place_field_threshold`: fraction of peak rate for place field extension
- `place_field_min_bins`: minimum contiguous bins for a place field seed
- `place_field_seed_percentile`: percentile of shuffled rates for seed detection

## Configuration Reference

### Data Paths Config

:::{dropdown} arena data_paths.yaml
```yaml
type: arena  # 'arena' for 2D open-field, 'maze' for 1D arm analysis
behavior_fps: 20.0  # Behavior camera sampling rate (Hz)
bodypart: LED  # DLC bodypart name for position tracking
neural_path: path/to/neural
neural_timestamp: path/to/neural_timestamp.csv
behavior_position: path/to/behavior_position.csv
behavior_timestamp: path/to/behavior_timestamp.csv
```
:::

:::{dropdown} maze data_paths.yaml
```yaml
type: maze
behavior_fps: 20.0
bodypart: LED
mm_per_pixel: 1.0
neural_path: path/to/neural
neural_timestamp: path/to/neural_timestamp.csv
behavior_position: path/to/behavior_position.csv  # raw DLC output (input to detect-zones)
behavior_timestamp: path/to/behavior_timestamp.csv
behavior_graph: path/to/behavior_graph.yaml       # zone polygons + adjacency graph
zone_tracking: path/to/zone_tracking.csv          # zone-detected output (input to MazeDataset.load)
arm_order: [Arm_1, Arm_2, Arm_3, Arm_4]
zone_column: zone
arm_position_column: arm_position
x_col: x_pinned
y_col: y_pinned
zone_detection:
  hampel_window_frames: 7  # Centered window for raw-position Hampel filter (in detect-zones)
  hampel_n_sigmas: 3.0     # MAD-scaled threshold (~99.7% Gaussian band)
  arm_max_distance: 60.0   # Max px from arm centerline for arm classification
  min_confidence: 0.5      # Min zone probability for transition
```
:::

`placecell` reads the DLC scorer name from the CSV header and does not require a fixed scorer string such as `3DMazeTrack` or `FuzzyTrack`.

### Analysis Config

:::{dropdown} arena_config.yaml
```yaml
neural:
  fps: 20.0
  oasis:
    g: [1.60, -0.63]  # AR(2) coefficients (required, usually overridden by data config)
    baseline: p10
    penalty: 0.8  # Sparsity penalty (higher = fewer events). Default 0.
    s_min: 0  # Minimum event size threshold. Default 0.
  trace_name: C_lp

behavior:
  type: arena
  speed_threshold: 10.0  # mm/s
  speed_window_seconds: 0.25
  hampel_window_frames: 7  # Centered window for raw-position Hampel filter (arena only)
  hampel_n_sigmas: 3.0  # MAD-scaled threshold (~99.7% Gaussian band)
  spatial_map_2d:
    bins: 50
    min_occupancy: 0.025  # Minimum occupancy (in seconds) to include a bin
    spatial_sigma: 3  # Gaussian smoothing (in bins) for occupancy and rate maps
    n_shuffles: 1000
    random_seed: 1
    event_threshold_sigma: 0  # Event threshold in SDs above mean (for trajectory plot only)
    p_value_threshold: 0.05  # P-value threshold for SI and stability
    min_shift_seconds: 20  # Minimum circular shift (seconds) for shuffle test
    si_weight_mode: amplitude  # 'amplitude' or 'binary'
    place_field_threshold: 0.35  # Fraction of peak rate for place field boundary
    place_field_min_bins: 5  # Minimum contiguous bins for a place field
    place_field_seed_percentile: 95  # Percentile of shuffled rates for seed detection
```
:::

:::{dropdown} maze_config.yaml
```yaml
neural:
  fps: 20.0
  oasis:
    g: [1.60, -0.63]
    baseline: p10
    penalty: 0.8
  trace_name: C_lp

behavior:
  type: maze
  speed_threshold: 25  # mm/s (note: maze Hampel lives in data config zone_detection block)
  speed_window_seconds: 0.25
  spatial_map_1d:
    bin_width_mm: 10
    min_occupancy: 0.025
    spatial_sigma: 2
    n_shuffles: 1000
    p_value_threshold: 0.05
    min_shift_seconds: 20
    si_weight_mode: amplitude
    n_split_blocks: 10
    split_by_direction: true
    require_complete_traversal: false
```
:::
