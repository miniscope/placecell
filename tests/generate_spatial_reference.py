"""Generate reference data for spatial analysis regression tests."""

import numpy as np
import pandas as pd
from pathlib import Path

from placecell.analysis import (
    gaussian_filter_normalized,
    compute_occupancy_map,
    compute_rate_map,
    compute_spatial_information,
)

assets = Path(__file__).parent / "assets"

# 1. gaussian_filter_normalized - simple test case
data = np.zeros((10, 10))
data[0, 0] = 1.0
gaussian_result = gaussian_filter_normalized(data, sigma=1.0)

# 2. compute_occupancy_map - from real trajectory (use full trajectory)
behavior_pos = pd.read_csv(
    assets / "behavior/behavior_position.csv", header=[0, 1, 2], index_col=0
)
scorer = behavior_pos.columns[0][0]
trajectory = pd.DataFrame({
    "x": behavior_pos[(scorer, "LED_clean", "x")].values,
    "y": behavior_pos[(scorer, "LED_clean", "y")].values,
})
trajectory = trajectory.dropna()  # Use full trajectory

occupancy, valid_mask, x_edges, y_edges = compute_occupancy_map(
    trajectory, bins=20, behavior_fps=20.0, occupancy_sigma=1.0, min_occupancy=0.1
)

# 3. Get unit events for all tests (consistent filtering)
event_place = pd.read_csv(assets / "reference_event_place.csv")
unit_id = event_place["unit_id"].iloc[0]

# Create trajectory with beh_frame_index
trajectory_si = trajectory.copy()
trajectory_si["beh_frame_index"] = range(len(trajectory_si))

unit_events = event_place[event_place["unit_id"] == unit_id].copy()
unit_events = unit_events[["x", "y", "s", "beh_frame_index"]].dropna()
unit_events = unit_events[
    (unit_events["beh_frame_index"] >= 0) & 
    (unit_events["beh_frame_index"] < len(trajectory_si))
]
# Also filter by spatial bounds
unit_events = unit_events[
    (unit_events["x"] >= x_edges[0]) & (unit_events["x"] <= x_edges[-1]) &
    (unit_events["y"] >= y_edges[0]) & (unit_events["y"] <= y_edges[-1])
]
print(f"Unit {unit_id} has {len(unit_events)} events after filtering")

# 4. compute_rate_map - using consistently filtered events
rate_map = compute_rate_map(
    unit_events, occupancy, valid_mask, x_edges, y_edges, activity_sigma=1.0
)

# 5. compute_spatial_information - using same events
si, p_val, shuffled_sis = compute_spatial_information(
    unit_events, trajectory_si, occupancy, valid_mask, x_edges, y_edges,
    n_shuffles=100, random_seed=42
)

# 6. vis_threshold for compute_unit_analysis test
event_threshold_sigma = 2.0
vis_threshold = unit_events["s"].mean() + event_threshold_sigma * unit_events["s"].std()

# Save references
np.savez(
    assets / "reference_spatial.npz",
    gaussian_input=data,
    gaussian_result=gaussian_result,
    occupancy=occupancy,
    valid_mask=valid_mask,
    x_edges=x_edges,
    y_edges=y_edges,
    rate_map=rate_map,
    spatial_info=si,
    spatial_info_pval=p_val,
    vis_threshold=vis_threshold,
)
print(f"Saved {assets / 'reference_spatial.npz'}")
