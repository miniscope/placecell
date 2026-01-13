## Typical workflow

### Needed data
   - Minian: Calcium traces (`C.zarr`)
   - mio: neural timestamp and behavior timestamp CSVs
   - Deeplabcut: Export behavior: position CSV (DeepLabCut) and `behavior_timestamp.csv`.

### Analysis config
   - Copy `pcell/assets/example_pcell_config.yaml` and adjust

### Run analysis
Runs the full pipeline: deconvolution, spike-place matching, and generates the place browser HTML.

```bash
pcell analyze \
--config your_config.yaml \
--neural-path /path/to/neural \
--behavior-path /path/to/behavior \
--out-dir export
```

### Browse results
- Open `export/<label>_place_browser.html` to view trajectory + spike locations per unit.