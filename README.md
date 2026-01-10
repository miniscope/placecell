### Typical workflow

1. **Prepare data**
   - Run your Minian pipeline to get `C.zarr`, `neural_timestamp.csv` for neural data.
   - Export behavior: position CSV (DeepLabCut) and `behavior_timestamp.csv`.
   - Adjust `pcell/assets/example_pcell_config.yaml` and copy it to your own config file.

2. **Run analysis**
   ```bash
   pcell analyze --config your_config.yaml --neural-path /path/to/neural --behavior-path /path/to/behavior --out-dir export
   ```
   This runs the full pipeline: deconvolution, spike-place matching, and generates the place browser HTML.

3. **Browse results**
   - Open `export/<label>_place_browser.html` to view trajectory + spike locations per unit.

See `pcell --help` for optional commands.