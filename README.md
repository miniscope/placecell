# placecell

## Typical workflow

### Needed data
   - Minian: Calcium traces (`C.zarr`)
   - mio: neural timestamp and behavior timestamp CSVs
   - DeepLabCut: position CSV and `behavior_timestamp.csv`

### Analysis config
   - Copy `placecell/assets/example_config.yaml` and adjust

### Run analysis
Runs the full pipeline: deconvolution, spike-place matching, and generates the place browser HTML.

```bash
pcell workflow visualize \
  --config your_config.yaml \
  --neural-path /path/to/neural \
  --behavior-path /path/to/behavior \
  --out-dir output
```

### Browse results
- Open `output/<label>_place_browser.html` to view trajectory + spike locations per unit.

### Individual steps
Run steps separately if needed:

```bash
pcell deconvolve --help
pcell spike-place --help
pcell generate-html --help
```
