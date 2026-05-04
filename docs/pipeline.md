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
8. `save_bundle()` — write `.pcellbundle` directory with config, arrays, parquets, figures, and a `metadata.json` that records both the bundle schema version and the `placecell` package version (via `hatch-vcs`, so it includes the git SHA of the build).

Steps 3–4 are gated on which blocks the data config carries: `preprocess_behavior()` is a no-op when no `behavior:` block is present, `deconvolve()` is a no-op when no `neural:` block is present, and the place-cell steps 5–7 require both. This makes neural-only (load → deconvolve → save) and behavior-only (load → preprocess → save) workflows valid out of the box.

The **canonical neural-rate table** (`ds.canonical`) is the central artifact: one row per neural frame with columns `frame_index, neural_time, x, y, speed, [pos_1d, arm_index, ...], s_unit_<id>...`. Behavior is linearly interpolated onto neural timestamps inside `match_events()` (or, for the maze pipeline, inside `detect-zones`), so every analysis downstream operates on a single common clock. The long-format `event_place` table is derived from `canonical` for the per-unit spatial analysis functions.

## Data Files

**Neural inputs** (in `neural.path` directory):
- `{trace_name}.zarr`: calcium traces (frames × units). Name set by `trace_name` in the analysis config (e.g. `C.zarr`, `C_lp.zarr`)
- `A.zarr`: spatial footprints for cell contour overlay (optional)
- `max_proj.zarr`: max projection image for visualization background (optional)
- `neural.timestamp` CSV: neural frame timestamps (frame, timestamp_first, timestamp_last)

**Behavior inputs:**
- `behavior.position` CSV: animal position per frame (DeepLabCut format with bodypart columns)
- `behavior.timestamp` CSV: behavior frame timestamps
- `behavior.video` (optional): behavior video file used for the calibration overlay frame

**Intermediate DataFrames** (generated during pipeline):
- `canonical`: per-neural-frame table with `x, y, speed, [pos_1d, ...], s_unit_<id>...`. Single source of truth for all downstream analysis.
- `trajectory_filtered` / `trajectory_1d_filtered`: speed-filtered view of `canonical` (with `frame_index` aliased so existing analysis helpers consume it directly).
- `event_place`: long-format event table (`unit_id, frame_index, x, y, s, speed, ...`) derived from the speed-filtered canonical view. Each row is one above-zero spike sample from one unit.
- `event_index`: long-format event table over the **unfiltered** canonical view, used by the notebook browser.

## Processing Steps

### `ds.load()`

- Load calcium traces from `{trace_name}.zarr` (skipped when no `neural:` block is configured)
- Load behavior position and timestamps (skipped when no `behavior:` block is configured; speed is computed later at neural rate in `match_events`)
- Load visualization assets (max_proj, footprints, behavior video frame)
- For the maze pipeline, also auto-runs zone detection if the cached `zone_tracking` CSV is missing

### `ds.preprocess_behavior()`

Saves a copy of the raw trajectory in `trajectory_raw`, then applies geometric corrections (Hampel, perspective, clipping, unit conversion). Speed is computed later at the neural sample rate inside `match_events()`. No-op when the data config has no `behavior:` block.

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

Requires both a `neural:` and a `behavior:` block — raises with a targeted message if either side is missing. Builds the canonical neural-rate table:

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
- No-op when the data config has no `neural:` block

### `ds.compute_occupancy()`

- Compute 2D occupancy histogram from `trajectory_filtered`
- Smooth with `spatial_sigma`, mask bins below `min_occupancy`
- Output: `occupancy_time`, `valid_mask`, bin edges

### `ds.analyze_units()` — per unit

Four independent computations from the same inputs (events, filtered trajectory, occupancy):

- **Rate map**: event weights / occupancy time, smoothed with `spatial_sigma`. Stored on `UnitResult` as `rate_map_smoothed` (authoritative, firing-rate units). A `rate_map_peak_normalized` property is derived on demand for display.
- **Spatial information + shuffle test**: Skaggs SI with circular-shift shuffle → SI p-value
- **Shuffled rate percentile**: per-bin percentile of shuffled rate maps → used for place field seed detection
- **Stability + shuffle tests** (one per configured block count in `stability_splits`): correlation between odd/even block rate maps with circular-shift shuffle → one stability p-value per split. Each split runs with an independent RNG stream so the tests are statistically independent.
- **Minimum-events gate** (`min_events`): units with fewer speed-filtered events are assigned `p_val=1.0` for both SI and stability, so they cannot be classified as place cells. Their rate map is still computed for inspection.

**Place cell classification**: units with SI p-value < `p_value_threshold` AND `UnitResult.is_stable(p_value_threshold)` (every configured split's stability p-value is below the threshold).

**Place field detection** (Guo et al. 2023):
1. **Seed detection**: bins where rate exceeds the shuffled rate percentile. Only contiguous seed regions with ≥ `place_field_min_bins` bins are kept
2. **Extension**: each seed region extends to contiguous bins with rate ≥ `place_field_threshold` × (seed's peak rate)

### Results

- **Coverage map**: sum of place field masks across place cells
- **Coverage curve**: cells sorted by field size (largest first), cumulative fraction of environment covered
- **Interactive browser**: max projection overlay, trajectory with events, rate map with place field contour, SI histogram, stability maps, trace view

### Summary Figures

`save_bundle()` exports these PDFs to the `figures/` directory inside the bundle. Both arena and maze pipelines produce a consistent set.

**Shared (both pipelines):**
- `diagnostics.pdf` — SI and stability distributions across all units
- `summary_scatter.pdf` — SI vs stability scatter plot with place cell classification
- `footprints.pdf` — spatial footprints of all recorded units
- `behavior_preview.pdf` — 2D trajectory density (additive alpha), speed-filtered trajectory, and speed histogram
- `speed_traces.pdf` — animal speed over time with place cell neural traces (top 20 by SI)
- `occupancy.pdf` — trajectory and full-session occupancy on the top row, then one row of odd/even half-occupancies per configured entry in `stability_splits` (same block scheme as the stability test). Below-threshold bins (`min_occupancy`, on the smoothed half-occupancy) are outlined in cyan.

**Arena only:**
- `arena_calibration.pdf` — raw trajectory overlaid on arena bounds and video frame
- `preprocess_steps.pdf` — trajectory at each correction stage (Hampel, perspective, clipping)
- `coverage.pdf` — place field coverage map and coverage curve

**Maze only:**
- `speed_histogram.pdf` — 1D arm speed distribution with threshold line
- `graph_overlay.pdf` — maze graph polylines on behavior video frame
- `population_rate_map.pdf` — all unit 1D rate maps tiled
- `global_pvo_matrix.pdf` — population vector overlap matrix across arms. Uses `rate_map_smoothed` (firing-rate units), so the cosine similarity reflects true rate magnitudes across cells rather than being a shape-only metric.

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
- Shuffle null: circular shift excluding zero-shift, independent RNG seeds for SI, per-split stability, and rate-percentile streams (each split in `stability_splits` also gets its own seed offset)
- Stability: one test per entry in `stability_splits`. Each uses interleaved odd/even blocks at that block count and a Fisher-z-transformed correlation; `block_shift` offsets the block boundaries so "first half" isn't always frame 0.
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
- `stability_splits`: list of block counts for the stability tests (e.g. `[2, 10]`). `n=2` is a classic first/second half; `n>=4` interleaves odd/even blocks so session-long drift averages out.
- `block_shift`: fractional offset applied to the block boundaries of all stability splits (0 = no shift)
- `min_events`: minimum speed-filtered event count to run SI + stability tests. Units below the threshold keep a rate map but get `p_val=1.0`. Set to 0 to disable; typical calcium-imaging values are 20–50.
- `place_field_threshold`: fraction of peak rate for place field extension
- `place_field_min_bins`: minimum contiguous bins for a place field seed
- `place_field_seed_percentile`: percentile of shuffled rates for seed detection

## Configuration Reference

### Data Paths Config

A per-session data config has two optional top-level blocks: `neural:` and `behavior:`. At least one must be present. Omit `neural:` for behavior-only sessions or `behavior:` for neural-only sessions; the pipeline gates each step on which side is configured.

:::{dropdown} arena data_paths.yaml
```yaml
neural:
  path: path/to/neural
  timestamp: path/to/neural_timestamp.csv
behavior:
  type: arena                           # 'arena' for 2D open-field, 'maze' for 1D arm analysis
  fps: 20.0                             # Behavior camera sampling rate (Hz)
  position: path/to/behavior_position.csv
  timestamp: path/to/behavior_timestamp.csv
  bodypart: LED                         # DLC bodypart name for position tracking
```
:::

:::{dropdown} maze data_paths.yaml
```yaml
neural:
  path: path/to/neural
  timestamp: path/to/neural_timestamp.csv
behavior:
  type: maze
  fps: 20.0
  position: path/to/behavior_position.csv  # raw DLC output (input to detect-zones)
  timestamp: path/to/behavior_timestamp.csv
  bodypart: LED
  x_col: x_pinned
  y_col: y_pinned
  mm_per_pixel: 1.0
  behavior_graph: path/to/behavior_graph.yaml  # zone polygons + adjacency graph
  zone_tracking: path/to/zone_tracking.csv     # zone-detected output (input to MazeDataset.load)
  arm_order: [Arm_1, Arm_2, Arm_3, Arm_4]
  zone_column: zone
  arm_position_column: arm_position
  zone_detection:
    hampel_window_frames: 7  # Centered window for raw-position Hampel filter (in detect-zones)
    hampel_n_sigmas: 3.0     # MAD-scaled threshold (~99.7% Gaussian band)
    arm_max_distance: 60.0   # Max px from arm centerline for arm classification
    min_confidence: 0.5      # Min zone probability for transition
```
:::

:::{dropdown} neural-only data_paths.yaml
```yaml
neural:
  path: path/to/neural
  timestamp: path/to/neural_timestamp.csv
# Only load() and deconvolve() run; match_events / compute_occupancy / analyze_units
# require a behavior block and will raise if called.
```
:::

:::{dropdown} behavior-only data_paths.yaml
```yaml
behavior:
  type: arena
  fps: 20.0
  position: path/to/behavior_position.csv
  timestamp: path/to/behavior_timestamp.csv
  bodypart: LED
# Only load() and preprocess_behavior() run; deconvolve() is a no-op without neural data.
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
    p_value_threshold: 0.05  # P-value threshold for SI and every stability split
    min_shift_seconds: 20  # Minimum circular shift (seconds) for shuffle test
    si_weight_mode: amplitude  # 'amplitude' or 'binary'
    stability_splits: [2, 10]  # One stability test per entry (n=2: classic halves; n>=4: interleaved blocks)
    block_shift: 0.0           # Fractional offset of block boundaries (0 = first block starts at frame 0)
    min_events: 0              # Skip SI + stability for units with fewer events (p_val=1.0). Typical: 20-50.
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
    stability_splits: [2, 10]  # One stability test per entry (n=2: classic halves; n>=4: interleaved blocks)
    block_shift: 0.0
    min_events: 0
    split_by_direction: true
    require_complete_traversal: false
```
:::
