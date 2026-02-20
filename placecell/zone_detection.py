"""Zone detection from position tracking data."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from placecell.geometry import (
    closest_point_on_polyline,
    get_zone_probabilities,
    is_valid_transition,
    load_zone_config,
    position_along_polyline,
)
from placecell.logging import init_logger

logger = init_logger(__name__)


def detect_zones(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    zone_polygons: dict[str, np.ndarray],
    zone_types: dict[str, str],
    zone_graph: dict[str, list[str]],
    tube_max_distance: float = 60.0,
    min_confidence: float = 0.5,
    min_confidence_forbidden: float = 0.8,
    min_frames_same: int = 1,
    min_frames_forbidden: int = 3,
) -> pd.DataFrame:
    """Detect zones from x/y position arrays.

    Parameters
    ----------
    x_coords, y_coords:
        Position arrays (one value per frame).
    zone_polygons:
        Dict mapping zone name to polygon/polyline array.
    zone_types:
        Dict mapping zone name to "room" or "tube".
    zone_graph:
        Bidirectional adjacency dict (zone -> list of connected zones).
    tube_max_distance:
        Maximum distance from tube centerline for classification.
    min_confidence:
        Minimum confidence for zone transitions.
    min_confidence_forbidden:
        Minimum confidence for forbidden (non-adjacent) transitions.
    min_frames_same:
        Minimum frames in current zone before allowing transition.
    min_frames_forbidden:
        Minimum consecutive frames for forbidden transition override.

    Returns
    -------
    DataFrame with columns: zone, x_pinned, y_pinned, tube_position.
    """
    current_zone = None
    frames_in_current_zone = 0
    forbidden_candidate_zone = None
    forbidden_consecutive_frames = 0

    zones = []
    x_pinned_list = []
    y_pinned_list = []
    tube_positions = []

    for x, y in zip(x_coords, y_coords):
        zone_probs = get_zone_probabilities(
            x,
            y,
            zone_polygons,
            zone_types,
            tube_max_distance=tube_max_distance,
            soft_boundary=True,
            normalize=True,
        )

        if zone_probs:
            new_zone = max(zone_probs, key=zone_probs.get)
            pred_confidence = zone_probs[new_zone]
        else:
            new_zone = None
            pred_confidence = 0.0

        # Zone transition logic with graph constraints
        if current_zone is not None and new_zone is not None:
            if not is_valid_transition(current_zone, new_zone, zone_graph, zone_types):
                # Check for valid alternatives
                valid_alternatives = []
                for alt_zone, alt_prob in zone_probs.items():
                    if (
                        alt_prob > 0
                        and alt_zone != new_zone
                        and is_valid_transition(current_zone, alt_zone, zone_graph, zone_types)
                    ):
                        valid_alternatives.append((alt_zone, alt_prob))

                if valid_alternatives:
                    best_alt_zone = max(valid_alternatives, key=lambda x: x[1])
                    if best_alt_zone[1] >= min_confidence and best_alt_zone[1] > pred_confidence:
                        new_zone = best_alt_zone[0]
                        pred_confidence = best_alt_zone[1]

        if current_zone is None:
            if new_zone and pred_confidence >= min_confidence:
                current_zone = new_zone
                frames_in_current_zone = 1
        elif new_zone == current_zone:
            frames_in_current_zone += 1
            forbidden_candidate_zone = None
            forbidden_consecutive_frames = 0
        else:
            is_valid = is_valid_transition(current_zone, new_zone, zone_graph, zone_types)

            if is_valid:
                if pred_confidence >= min_confidence and frames_in_current_zone >= min_frames_same:
                    current_zone = new_zone
                    frames_in_current_zone = 1
                    forbidden_candidate_zone = None
                    forbidden_consecutive_frames = 0
                else:
                    frames_in_current_zone += 1
            else:
                if pred_confidence >= min_confidence_forbidden:
                    if new_zone == forbidden_candidate_zone:
                        forbidden_consecutive_frames += 1
                    else:
                        forbidden_candidate_zone = new_zone
                        forbidden_consecutive_frames = 1

                    if (
                        forbidden_consecutive_frames >= min_frames_forbidden
                        and frames_in_current_zone >= min_frames_same
                    ):
                        current_zone = new_zone
                        frames_in_current_zone = 1
                        forbidden_candidate_zone = None
                        forbidden_consecutive_frames = 0
                    else:
                        frames_in_current_zone += 1
                else:
                    forbidden_candidate_zone = None
                    forbidden_consecutive_frames = 0
                    frames_in_current_zone += 1

        pred_label = current_zone if current_zone else "Unknown"

        # Calculate tube-pinned coordinates
        tube_position = None
        pinned_x, pinned_y = x, y
        if zone_types.get(pred_label) == "tube" and pred_label in zone_polygons:
            tube_position = position_along_polyline((x, y), zone_polygons[pred_label])
            pinned_pt = closest_point_on_polyline((x, y), zone_polygons[pred_label])
            pinned_x, pinned_y = float(pinned_pt[0]), float(pinned_pt[1])

        zones.append(pred_label)
        x_pinned_list.append(pinned_x)
        y_pinned_list.append(pinned_y)
        tube_positions.append(tube_position)

    return pd.DataFrame(
        {
            "zone": zones,
            "x_pinned": x_pinned_list,
            "y_pinned": y_pinned_list,
            "tube_position": tube_positions,
        }
    )


def detect_zones_from_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    zone_config_path: str | Path,
    bodypart: str | None = None,
    tube_max_distance: float = 60.0,
    min_confidence: float = 0.5,
    min_confidence_forbidden: float = 0.8,
    min_frames_same: int = 1,
    min_frames_forbidden: int = 3,
) -> None:
    """Detect zones from a DLC-format tracking CSV.

    Parameters
    ----------
    input_csv:
        Input CSV with x, y coordinates (DLC multi-index format).
    output_csv:
        Output CSV with zone information added.
    zone_config_path:
        Path to combined zone config YAML.
    bodypart:
        Body part name to use. If None, uses the first bodypart found.
    tube_max_distance:
        Maximum distance from tube centerline for classification.
    min_confidence:
        Minimum confidence for zone transitions.
    min_confidence_forbidden:
        Minimum confidence for forbidden transitions.
    min_frames_same:
        Minimum frames in zone before allowing transition.
    min_frames_forbidden:
        Minimum consecutive frames for forbidden transition.
    """
    logger.info("Loading position data from %s", input_csv)

    # Read DLC-format CSV
    df = pd.read_csv(input_csv, header=[0, 1, 2], index_col=0)

    scorer = df.columns.get_level_values(0)[0]
    if bodypart is None:
        bodypart = df.columns.get_level_values(1)[0]

    x_col = (scorer, bodypart, "x")
    y_col = (scorer, bodypart, "y")

    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(
            f"Could not find x, y columns for bodypart '{bodypart}'. "
            f"Available: {sorted(set(df.columns.get_level_values(1)))}"
        )

    x_coords = df[x_col].values.astype(float)
    y_coords = df[y_col].values.astype(float)
    # Load zone config
    zone_polygons, zone_types, zone_graph = load_zone_config(zone_config_path)
    logger.info("Loaded %d position samples, %d zones", len(x_coords), len(zone_polygons))
    result = detect_zones(
        x_coords,
        y_coords,
        zone_polygons,
        zone_types,
        zone_graph,
        tube_max_distance=tube_max_distance,
        min_confidence=min_confidence,
        min_confidence_forbidden=min_confidence_forbidden,
        min_frames_same=min_frames_same,
        min_frames_forbidden=min_frames_forbidden,
    )

    # Build output in DLC multi-index format
    output_data = {
        (scorer, bodypart, "x"): x_coords,
        (scorer, bodypart, "y"): y_coords,
        (scorer, bodypart, "x_pinned"): result["x_pinned"].values,
        (scorer, bodypart, "y_pinned"): result["y_pinned"].values,
        (scorer, bodypart, "zone"): result["zone"].values,
        (scorer, bodypart, "tube_position"): result["tube_position"].values,
    }

    # Include likelihood if it exists
    likelihood_col = (scorer, bodypart, "likelihood")
    if likelihood_col in df.columns:
        output_data[likelihood_col] = df[likelihood_col].values

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv)
    logger.info("Zone detection results saved to %s", output_csv)

    # Print zone statistics
    zone_counts = result["zone"].value_counts()
    logger.info("Zone distribution:")
    for zone, count in zone_counts.items():
        pct = 100 * count / len(result)
        logger.info("  %s: %d frames (%.1f%%)", zone, count, pct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect zones from tracking CSV")
    parser.add_argument("--input", "-i", required=True, help="Input tracking CSV")
    parser.add_argument("--output", "-o", required=True, help="Output CSV")
    parser.add_argument(
        "--zone-config", "-z", required=True, help="Zone config YAML"
    )
    parser.add_argument("--bodypart", "-b", default=None, help="Body part name")
    parser.add_argument(
        "--tube-max-distance", type=float, default=60.0, help="Max tube distance (px)"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.5, help="Min transition confidence"
    )
    parser.add_argument(
        "--min-confidence-forbidden",
        type=float,
        default=0.8,
        help="Min forbidden transition confidence",
    )
    parser.add_argument(
        "--min-frames-same", type=int, default=1, help="Min frames before transition"
    )
    parser.add_argument(
        "--min-frames-forbidden",
        type=int,
        default=3,
        help="Min consecutive frames for forbidden transition",
    )

    args = parser.parse_args()

    detect_zones_from_csv(
        input_csv=args.input,
        output_csv=args.output,
        zone_config_path=args.zone_config,
        bodypart=args.bodypart,
        tube_max_distance=args.tube_max_distance,
        min_confidence=args.min_confidence,
        min_confidence_forbidden=args.min_confidence_forbidden,
        min_frames_same=args.min_frames_same,
        min_frames_forbidden=args.min_frames_forbidden,
    )
