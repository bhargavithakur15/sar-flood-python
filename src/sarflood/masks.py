"""
Masks that remove false positives from the raw change-detection result.

- Permanent water: rivers and lakes are dark in both images, but a small
  change there is not a flood.
- HAND (Height Above Nearest Drainage): water cannot pond far above the
  nearest river, so high pixels that darkened are something else
  (harvested fields, shadows, wet soil).
- Slope: optional, steep terrain rarely holds standing water.
- Minimum mapping unit: isolated specks of a few pixels are speckle residue.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def permanent_water_mask(osm_water: np.ndarray, jrc_seasonality: np.ndarray, min_months: int) -> np.ndarray:
    """OSM water polygons OR JRC Global Surface Water present >= ``min_months`` per year."""
    jrc = np.nan_to_num(jrc_seasonality, nan=0.0) >= min_months
    return osm_water.astype(bool) | jrc


def height_above_nearest_drainage(
    dem: np.ndarray,
    drainage: np.ndarray,
    pixel_size_m: float,
    max_distance_m: float | None = None,
) -> np.ndarray:
    """
    Approximate HAND: each pixel's elevation minus the elevation of its
    nearest drainage pixel (straight-line distance).

    ``scipy.ndimage.distance_transform_edt`` with ``return_indices=True``
    gives, for every pixel, the row/column of the closest drainage pixel in
    one vectorised pass, which is the same idea as the GEE focal-min
    approximation but referenced to the *nearest* channel rather than the
    lowest one within a window. A true HAND follows flow paths over the DEM,
    which needs flow-direction routing (e.g. pysheds/whitebox).

    Pixels farther than ``max_distance_m`` from any drainage get NaN.
    """
    drainage = drainage.astype(bool)
    if not drainage.any():
        raise ValueError("Drainage mask is empty: cannot compute HAND")
    distance_px, (rows, cols) = ndimage.distance_transform_edt(~drainage, return_indices=True)
    hand = dem - dem[rows, cols]
    hand = hand.astype("float32")
    if max_distance_m is not None:
        hand[distance_px * pixel_size_m > max_distance_m] = np.nan
    hand[~np.isfinite(dem)] = np.nan
    return hand


def slope_percent(dem: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """Terrain slope in percent (rise/run * 100) from a DEM on a metric grid."""
    dz_dy, dz_dx = np.gradient(dem.astype("float64"), pixel_size_m)
    return (np.hypot(dz_dx, dz_dy) * 100.0).astype("float32")


def remove_small_patches(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """
    Keep only connected patches larger than ``min_pixels`` (8-connectivity),
    matching GEE ``connectedPixelCount(..., eightConnected=true).gt(min_pixels)``.
    """
    if min_pixels <= 0:
        return mask.astype(bool)
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())       # sizes[k] = pixels in patch k
    keep = sizes > min_pixels
    keep[0] = False                           # label 0 is background
    return keep[labels]


def apply_masks(
    candidates: np.ndarray,
    permanent_water: np.ndarray,
    hand: np.ndarray,
    hand_max_m: float,
    mmu_pixels: int,
    slope: np.ndarray | None = None,
    slope_max_percent: float | None = None,
) -> np.ndarray:
    """
    Mask cascade, applied in the same order as the GEE script so variants stay
    comparable: permanent water -> slope (optional) -> HAND -> minimum mapping unit.
    NaN HAND (no drainage nearby) is treated as 'too high' and removed.
    """
    out = candidates.astype(bool) & ~permanent_water.astype(bool)
    if slope is not None and slope_max_percent is not None:
        out &= np.nan_to_num(slope, nan=np.inf) < slope_max_percent
    out &= np.nan_to_num(hand, nan=np.inf) < hand_max_m
    return remove_small_patches(out, mmu_pixels)
