# placecell

## Typical workflow

### Needed data
   - Minian: Calcium traces (`C.zarr`)
   - mio: neural timestamp and behavior timestamp CSVs
   - DeepLabCut: position CSV and `behavior_timestamp.csv`

### Analysis config
   - Copy `placecell/assets/example_pcell_config.yaml` and adjust

### Run analysis
Runs the full pipeline: deconvolution and spike-place matching.

```bash
pcell workflow visualize \
  --config your_config.yaml \
  --neural-path /path/to/neural \
  --behavior-path /path/to/behavior \
  --out-dir output
```

### Browse results
Interactive matplotlib browser:

```bash
pcell plot \
  --config your_config.yaml \
  --spike-place output/spike_place_*.csv \
  --neural-path /path/to/neural
```

### Individual steps
Run steps separately if needed:

```bash
pcell deconvolve --help
pcell spike-place --help
```
