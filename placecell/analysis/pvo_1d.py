"""Population-vector overlap utilities for 1D maze analysis."""

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


@dataclass(frozen=True)
class ArmPVOResult:
    """Population-vector overlap result for one pair of arm segments."""

    arm_a: str
    arm_b: str
    pvo: np.ndarray
    n_units: int
    n_bins_arm_a: int
    n_bins_arm_b: int


def _stack_rate_maps(
    unit_results: dict[int, Any],
    unit_ids: list[int] | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Stack per-unit ``rate_map_smoothed`` arrays for PVO.

    Uses firing-rate units so cosine similarity across bins reflects the
    true rate magnitudes across cells.
    """
    selected = sorted(unit_ids) if unit_ids is not None else sorted(unit_results)
    if not selected:
        raise ValueError("No units selected for PVO.")

    rate_maps = []
    n_bins = None
    kept_ids: list[int] = []
    for uid in selected:
        res = unit_results.get(uid)
        if res is None or getattr(res, "rate_map_smoothed", None) is None:
            continue
        rm = np.asarray(res.rate_map_smoothed, dtype=float)
        if rm.size == 0:
            continue
        if rm.ndim != 1:
            raise ValueError(f"Unit {uid} rate_map_smoothed must be 1D, got shape {rm.shape}.")
        if n_bins is None:
            n_bins = rm.shape[0]
        elif rm.shape[0] != n_bins:
            raise ValueError("All unit rate maps must have the same number of bins.")
        rate_maps.append(rm)
        kept_ids.append(uid)

    if not rate_maps:
        raise ValueError("No usable rate maps found for the selected units.")
    return np.vstack(rate_maps), kept_ids


def _cosine_overlap(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() == 0:
        return np.nan
    xv = x[valid]
    yv = y[valid]
    xnorm = float(np.linalg.norm(xv))
    ynorm = float(np.linalg.norm(yv))
    if xnorm == 0.0 or ynorm == 0.0:
        return np.nan
    return float(np.dot(xv, yv) / (xnorm * ynorm))


def population_vector_overlap(
    arm_a_rates: np.ndarray,
    arm_b_rates: np.ndarray,
) -> np.ndarray:
    """Compute a position-by-position population-vector overlap matrix."""
    if arm_a_rates.ndim != 2 or arm_b_rates.ndim != 2:
        raise ValueError("arm_a_rates and arm_b_rates must both be 2D arrays.")
    if arm_a_rates.shape[0] != arm_b_rates.shape[0]:
        raise ValueError("Both arms must have the same number of units.")

    n_pos_a = arm_a_rates.shape[1]
    n_pos_b = arm_b_rates.shape[1]
    out = np.full((n_pos_a, n_pos_b), np.nan, dtype=float)

    for i in range(n_pos_a):
        vec_a = arm_a_rates[:, i]
        for j in range(n_pos_b):
            out[i, j] = _cosine_overlap(vec_a, arm_b_rates[:, j])
    return out


def compute_arm_pvo(
    unit_results: dict[int, Any],
    segment_bins: list[int],
    segment_labels: list[str],
    *,
    unit_ids: list[int] | None = None,
) -> dict[tuple[str, str], ArmPVOResult]:
    """Compute arm-by-arm population-vector overlap from 1D rate maps."""
    if segment_bins is None or len(segment_bins) < 2:
        raise ValueError("segment_bins must contain at least two boundaries.")
    if len(segment_labels) != len(segment_bins) - 1:
        raise ValueError("segment_labels length must equal len(segment_bins) - 1.")

    rate_maps, kept_ids = _stack_rate_maps(unit_results, unit_ids=unit_ids)
    segments: dict[str, np.ndarray] = {}
    for i, label in enumerate(segment_labels):
        start = segment_bins[i]
        stop = segment_bins[i + 1]
        if stop <= start:
            raise ValueError(f"Invalid segment bounds for {label}: {start}..{stop}")
        segments[label] = rate_maps[:, start:stop].astype(float, copy=True)

    results: dict[tuple[str, str], ArmPVOResult] = {}
    for arm_a, seg_a in segments.items():
        for arm_b, seg_b in segments.items():
            pvo = population_vector_overlap(seg_a, seg_b)
            results[(arm_a, arm_b)] = ArmPVOResult(
                arm_a=arm_a,
                arm_b=arm_b,
                pvo=pvo,
                n_units=len(kept_ids),
                n_bins_arm_a=seg_a.shape[1],
                n_bins_arm_b=seg_b.shape[1],
            )
    return results


def compute_dataset_arm_pvo(
    ds: Any,
    *,
    use_place_cells: bool = True,
) -> dict[tuple[str, str], ArmPVOResult]:
    """Compute arm-by-arm PVO directly from a loaded MazeDataset bundle."""
    if not hasattr(ds, "unit_results") or not hasattr(ds, "segment_bins"):
        raise ValueError("Dataset does not look like a maze bundle with analyzed units.")

    unit_ids: list[int] | None = None
    if use_place_cells:
        unit_ids = sorted(ds.place_cells())
        if not unit_ids:
            raise ValueError("No place cells available in this dataset.")

    return compute_arm_pvo(
        ds.unit_results,
        ds.segment_bins,
        ds.effective_arm_order,
        unit_ids=unit_ids,
    )


def fuse_arm_pvo(
    results: dict[tuple[str, str], ArmPVOResult],
    segment_labels: list[str],
) -> tuple[np.ndarray, list[int], list[int]]:
    """Fuse arm-pair PVO blocks into one large tiled image."""
    if not segment_labels:
        raise ValueError("segment_labels must not be empty.")
    row_sizes = [results[(arm_a, segment_labels[0])].pvo.shape[0] for arm_a in segment_labels]
    col_sizes = [results[(segment_labels[0], arm_b)].pvo.shape[1] for arm_b in segment_labels]
    row_offsets = np.cumsum([0, *row_sizes])
    col_offsets = np.cumsum([0, *col_sizes])
    fused = np.full((row_offsets[-1], col_offsets[-1]), np.nan, dtype=float)

    for i, arm_a in enumerate(segment_labels):
        for j, arm_b in enumerate(segment_labels):
            block = results[(arm_a, arm_b)].pvo
            r0 = row_offsets[i]
            r1 = row_offsets[i + 1]
            c0 = col_offsets[j]
            c1 = col_offsets[j + 1]
            fused[r0:r1, c0:c1] = block
    return fused, row_sizes, col_sizes


def plot_arm_pvo_grid(
    results: dict[tuple[str, str], ArmPVOResult],
    segment_labels: list[str],
    *,
    display_labels: list[str] | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "inferno",
    figsize_scale: float = 2.0,
    separator_color: str = "black",
    separator_linewidth: float = 1.0,
) -> Any:
    """Plot a fused arm-by-arm PVO image with separator lines and outer labels."""
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    n = len(segment_labels)
    if display_labels is None:
        display_labels = segment_labels
    if len(display_labels) != n:
        raise ValueError("display_labels length must match segment_labels length.")

    fused, row_sizes, col_sizes = fuse_arm_pvo(results, segment_labels)

    fig, ax = plt.subplots(figsize=(figsize_scale * n, figsize_scale * n))
    fig.subplots_adjust(left=0.18, right=0.90, bottom=0.14, top=0.88)
    im = ax.imshow(
        fused,
        origin="upper",
        aspect="equal",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    row_offsets = np.cumsum([0, *row_sizes])
    col_offsets = np.cumsum([0, *col_sizes])
    for b in row_offsets[1:-1] - 0.5:
        ax.axhline(b, color=separator_color, linewidth=separator_linewidth)
    for b in col_offsets[1:-1] - 0.5:
        ax.axvline(b, color=separator_color, linewidth=separator_linewidth)

    row_centers = [(row_offsets[i] + row_offsets[i + 1] - 1) / 2 for i in range(n)]
    col_centers = [(col_offsets[i] + col_offsets[i + 1] - 1) / 2 for i in range(n)]
    ax.set_xticks(col_centers)
    ax.set_yticks(row_centers)
    ax.set_xticklabels(display_labels, rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticklabels(display_labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")

    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
    cbar.set_label("Population-vector overlap")
    return fig
