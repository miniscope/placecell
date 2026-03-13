"""Batch place cell analysis across multiple sessions.

Edit SESSIONS below, then run:
    python batch_analysis.py
"""

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from placecell.dataset import BasePlaceCellDataset

ARENA_CONFIG = "example_arena_config"  # stem name from placecell/config/ or path
MAZE_CONFIG = "example_maze_config"
OUTPUT_DIR = Path("bundle")
WORKERS = 4

# List of (config, data_yaml_path) pairs.
SESSIONS = [
    (ARENA_CONFIG, Path("data/arena/mouse1/mouse1_day1.yaml")),
    (ARENA_CONFIG, Path("data/arena/mouse1/mouse1_day2.yaml")),
    (MAZE_CONFIG, Path("data/maze/mouse1/mouse1_day1.yaml")),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (config, data_path) in enumerate(SESSIONS):
        name = data_path.stem
        print(f"\n[{i + 1}/{len(SESSIONS)}] {name}")

        ds = BasePlaceCellDataset.from_yaml(config, data_path)
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


if __name__ == "__main__":
    main()
