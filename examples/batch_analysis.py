"""Batch place cell analysis across multiple sessions.

Edit CONFIG, DATA_ROOT, and DATA_YAMLS below, then run:
    python examples/batch_analysis.py
"""

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from placecell.dataset import BasePlaceCellDataset

HERE = Path(__file__).resolve().parent

CONFIG = "example_pcell_config"  # stem name from placecell/config/ or path
DATA_ROOT = HERE / "../data"    # adjust to your data directory
OUTPUT_DIR = HERE / "../output"
WORKERS = 4

DATA_YAMLS = [
    DATA_ROOT / "data1.yaml",
    DATA_ROOT / "data2.yaml",
    # Add more YAML paths as needed
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
for i, data_path in enumerate(DATA_YAMLS):
    name = data_path.stem
    print(f"\n[{i + 1}/{len(DATA_YAMLS)}] {name}")

    ds = BasePlaceCellDataset.from_yaml(CONFIG, data_path)
    ds.load()
    ds.preprocess_behavior()
    ds.deconvolve(progress_bar=tqdm)
    ds.match_events()
    ds.compute_occupancy()
    ds.analyze_units(progress_bar=tqdm, n_workers=WORKERS)

    bundle_path = ds.save_bundle(str(OUTPUT_DIR / f"{name}.pcellbundle"))
    print(f"  Saved: {bundle_path}")

    row = {"dataset": name, "bundle": str(bundle_path), **ds.summary()}
    rows.append(row)
    print(
        f"  Significant:  {row['n_sig']}/{row['n_total']}\n"
        f"  Stable:       {row['n_stable']}/{row['n_total']}\n"
        f"  Place cells:  {row['n_place_cells']}/{row['n_total']}"
    )

df = pd.DataFrame(rows)
print(f"\n{'=' * 60}")
print(df[["dataset", "n_total", "n_sig", "n_stable", "n_place_cells"]].to_string(index=False))
