"""Population-vector overlap utilities for 1D maze analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

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
    mean_diagonal: float
    n_units: int
    n_bins_per_arm: int


def _require_segment_bins(segment_bins: list[int] | None) -> list[int]:
    if segment_bins is None or len(segment_bins) < 2:
        raise ValueError("segment_bins must contain at least two boundaries.")
    return list(segment_bins)


def _stack_rate_maps(
    unit_results: dict[int, Any],
    unit_ids: list[int] | None = None,
) -> tuple[np.ndarray, list[int]]:
    selected = sorted(unit_ids) if unit_ids is not None else sorted(unit_results)
    if not selected:
        raise ValueError("No units selected for PVO.")

    rate_maps = []
    n_bins = None
    kept_ids: list[int] = []
    for uid in selected:
        res = unit_results.get(uid)
        if res is None or getattr(res, "rate_map", None) is None:
            continue
        rm = np.asarray(res.rate_map, dtype=float)
        if rm.ndim != 1:
            raise ValueError(f"Unit {uid} rate_map must be 1D, got shape {rm.shape}.")
        if n_bins is None:
            n_bins = rm.shape[0]
        elif rm.shape[0] != n_bins:
            raise ValueError("All unit rate maps must have the same number of bins.")
        rate_maps.append(rm)
        kept_ids.append(uid)

    if not rate_maps:
        raise ValueError("No usable rate maps found for the selected units.")
    return np.vstack(rate_maps), kept_ids


def _resample_segment(segment_map: np.ndarray, n_bins: int) -> np.ndarray:
    """Resample one unit-by-position segment map onto a common normalized axis."""
    if segment_map.ndim != 2:
        raise ValueError(f"segment_map must be 2D, got shape {segment_map.shape}.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    n_units, orig_bins = segment_map.shape
    if orig_bins == 0:
        return np.full((n_units, n_bins), np.nan, dtype=float)
    if orig_bins == n_bins:
        return segment_map.astype(float, copy=True)

    src = np.linspace(0.0, 1.0, orig_bins)
    dst = np.linspace(0.0, 1.0, n_bins)
    out = np.full((n_units, n_bins), np.nan, dtype=float)

    for i in range(n_units):
        row = np.asarray(segment_map[i], dtype=float)
        valid = np.isfinite(row)
        if valid.sum() == 0:
            continue
        if valid.sum() == 1:
            out[i, :] = row[valid][0]
            continue
        out[i, :] = np.interp(dst, src[valid], row[valid])
    return out


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
    n_position_bins: int = 40,
) -> dict[tuple[str, str], ArmPVOResult]:
    """Compute arm-by-arm population-vector overlap from 1D rate maps.

    Each arm segment is resampled onto a normalized 0..1 position axis so
    segments with different lengths can be compared directly.
    """
    bins = _require_segment_bins(segment_bins)
    if len(segment_labels) != len(bins) - 1:
        raise ValueError("segment_labels length must equal len(segment_bins) - 1.")

    rate_maps, kept_ids = _stack_rate_maps(unit_results, unit_ids=unit_ids)

    segments: dict[str, np.ndarray] = {}
    for i, label in enumerate(segment_labels):
        start = bins[i]
        stop = bins[i + 1]
        if stop <= start:
            raise ValueError(f"Invalid segment bounds for {label}: {start}..{stop}")
        segments[label] = _resample_segment(rate_maps[:, start:stop], n_position_bins)

    results: dict[tuple[str, str], ArmPVOResult] = {}
    for arm_a, seg_a in segments.items():
        for arm_b, seg_b in segments.items():
            pvo = population_vector_overlap(seg_a, seg_b)
            diag = np.diag(pvo)
            mean_diagonal = float(np.nanmean(diag)) if np.isfinite(diag).any() else np.nan
            results[(arm_a, arm_b)] = ArmPVOResult(
                arm_a=arm_a,
                arm_b=arm_b,
                pvo=pvo,
                mean_diagonal=mean_diagonal,
                n_units=len(kept_ids),
                n_bins_per_arm=n_position_bins,
            )
    return results


def compute_dataset_arm_pvo(
    ds: Any,
    *,
    use_place_cells: bool = True,
    n_position_bins: int = 40,
) -> dict[tuple[str, str], ArmPVOResult]:
    """Compute arm-by-arm PVO directly from a loaded MazeDataset bundle."""
    if not hasattr(ds, "unit_results") or not hasattr(ds, "segment_bins"):
        raise ValueError("Dataset does not look like a maze bundle with analyzed units.")

    unit_ids = None
    if use_place_cells:
        place_cell_results = ds.place_cells()
        unit_ids = sorted(place_cell_results)
        if not unit_ids:
            raise ValueError("No place cells available in this dataset.")

    return compute_arm_pvo(
        ds.unit_results,
        ds.segment_bins,
        ds.effective_arm_order,
        unit_ids=unit_ids,
        n_position_bins=n_position_bins,
    )


def arm_pvo_summary(results: dict[tuple[str, str], ArmPVOResult]) -> pd.DataFrame:
    """Create a flat summary table from arm-pair PVO results."""
    rows = []
    for (arm_a, arm_b), res in sorted(results.items()):
        rows.append(
            {
                "arm_a": arm_a,
                "arm_b": arm_b,
                "mean_diagonal": res.mean_diagonal,
                "n_units": res.n_units,
                "n_bins_per_arm": res.n_bins_per_arm,
            }
        )
    return pd.DataFrame(rows)


def plot_arm_pvo(
    result: ArmPVOResult,
    *,
    ax: Any = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "inferno",
) -> Any:
    """Plot one arm-pair PVO matrix."""
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.5))

    im = ax.imshow(
        result.pvo,
        origin="upper",
        aspect="equal",
        interpolation="nearest",
        extent=[0, 1, 0, 1],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_xlabel(f"{result.arm_b} position")
    ax.set_ylabel(f"{result.arm_a} position")
    ax.set_title(f"{result.arm_a} vs {result.arm_b}\nmean diag={result.mean_diagonal:.3f}")
    return im


def fuse_arm_pvo(
    results: dict[tuple[str, str], ArmPVOResult],
    segment_labels: list[str],
) -> np.ndarray:
    """Fuse arm-pair PVO blocks into one large tiled image."""
    if not segment_labels:
        raise ValueError("segment_labels must not be empty.")
    first = results[(segment_labels[0], segment_labels[0])].pvo
    block_nrows, block_ncols = first.shape
    fused = np.full(
        (len(segment_labels) * block_nrows, len(segment_labels) * block_ncols),
        np.nan,
        dtype=float,
    )

    for i, arm_a in enumerate(segment_labels):
        for j, arm_b in enumerate(segment_labels):
            block = results[(arm_a, arm_b)].pvo
            r0 = i * block_nrows
            r1 = (i + 1) * block_nrows
            c0 = j * block_ncols
            c1 = (j + 1) * block_ncols
            fused[r0:r1, c0:c1] = block
    return fused


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

    fused = fuse_arm_pvo(results, segment_labels)
    block_size = results[(segment_labels[0], segment_labels[0])].pvo.shape[0]

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

    boundaries = np.arange(1, n) * block_size - 0.5
    for b in boundaries:
        ax.axhline(b, color=separator_color, linewidth=separator_linewidth)
        ax.axvline(b, color=separator_color, linewidth=separator_linewidth)

    centers = np.arange(n) * block_size + (block_size - 1) / 2
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(display_labels, rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticklabels(display_labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")

    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
    cbar.set_label("Population-vector overlap")
    return fig
