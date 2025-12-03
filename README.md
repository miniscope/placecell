  
### Typical workflow

1. **Prepare data**
   - Run your Minian pipeline to get `C.zarr`, `timestamp.csv` for neural data.
   - Export behavior: position CSV (DeepLabCut) and `behavior_timestamp.csv`.
   - Adjust `pcell/assets/example_pcell_config.yaml` (paths, FPS, LPF, max_units) and copy it to your own config file.

2. **Curate units (traces only)**
   - Show many traces in the browser with checkboxes:
     ```bash
     pcell show --config your_config.yaml --max-units 60 --output-prefix traces
     ```
     Open `export/traces.html`, tick units you like, click “Download `curated_units.txt`” and place it under `export/`.
   - Or browse one trace at a time and mark “keep”:
     ```bash
     pcell browse --config your_config.yaml --output-prefix browse_traces
     ```
     Open `export/browse_traces.html`, navigate units, check “keep”, download `curated_units.txt`.

3. **Deconvolve with OASIS**
   - Make sure `oasis-deconv` is installed (see above).
   - Run deconvolution, using curated units automatically if `export/curated_units.txt` exists:
     ```bash
     pcell deconvolve --config your_config.yaml --out-dir export --label SESSION
     ```
   - This writes:
     - `export/SESSION_oasis_deconv.zarr` (arrays `C_deconv`, `S`)
     - `export/spike_index.csv` (`unit_id,frame,s` for nonzero spikes)

4. **Match spikes to behavior (speed filtered, ≥50 px/s by default)**
   - Build the spike–place table:
     ```bash
     pcell spike-place
     ```
   - This reads `export/spike_index.csv` + neural/behavior timestamps, applies a running speed threshold (default 50 pixels/s; override with `--speed-threshold` and optionally `--cm-per-pixel`) and writes:
     - `export/spike_place.csv` with columns like `unit_id, frame, s, neural_time, beh_frame_index, beh_time, x, y, speed`.

5. **Browse place fields in the browser**
   - Use the place browser to see trajectory + spike dots per unit:
     ```bash
     pcell browse-place --output-prefix place_browser
     ```
   - Open `export/place_browser.html`, then use **Prev / Next** and the **Unit** dropdown to flip through units and inspect where they fire along the trajectory.

