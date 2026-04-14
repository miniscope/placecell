"""Batch place cell analysis across multiple sessions.

Edit SESSIONS below, then run:
    python examples/batch_analysis.py
"""

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from placecell.dataset import BasePlaceCellDataset

# Config: stem name from placecell/config/ or path to a YAML file.
CONFIG = "example_arena_config"

# Output directory for .pcellbundle files.
OUTPUT_DIR = Path("bundle")

# Parallel workers for analyze_units.
WORKERS = 4

# List of data YAML paths. Each is one session.
SESSIONS = [
    Path("data/mouse1/mouse1_day1.yaml"),
    Path("data/mouse1/mouse1_day2.yaml"),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, data_path in enumerate(SESSIONS):
        name = data_path.stem
        print(f"\n[{i + 1}/{len(SESSIONS)}] {name}")

        ds = BasePlaceCellDataset.from_yaml(CONFIG, data_path)
        ds.load()
        ds.preprocess_behavior()
        ds.deconvolve(progress_bar=tqdm)
        ds.match_events()
        ds.compute_occupancy()
        ds.analyze_units(progress_bar=tqdm, n_workers=WORKERS)

        bundle_path = ds.save_bundle(str(OUTPUT_DIR / f"{name}.pcellbundle"))
        print(f"  Saved: {bundle_path}")

        s = ds.summary()
        rows.append({"dataset": name, **s})
        print(
            f"  Place cells: {s['n_place_cells']}/{s['n_total']} "
            f"({s['n_sig']} sig, {s['n_stable']} stable)"
        )

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 60}")
    print(df[["dataset", "n_total", "n_sig", "n_stable", "n_place_cells"]].to_string(index=False))


if __name__ == "__main__":
    main()
