"""
SAR backscatter preprocessing: unit conversion and speckle filtering.

Speckle is the grainy noise in SAR images caused by interference between many
scatterers inside one pixel. It is multiplicative, so filtering is done on
linear power values, never in decibels (averaging logs is not averaging power).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def db_to_linear(db: np.ndarray) -> np.ndarray:
    """Decibels to linear power: 10^(dB/10)."""
    return np.power(10.0, db / 10.0)


def linear_to_db(linear: np.ndarray) -> np.ndarray:
    """Linear power to decibels: 10*log10(x). Non-positive values become NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(linear)
    out[~np.isfinite(out)] = np.nan
    return out


def circular_kernel(radius_px: int) -> np.ndarray:
    """Disc-shaped boolean footprint with the given radius in pixels."""
    if radius_px < 1:
        raise ValueError("radius_px must be >= 1")
    y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    return (x * x + y * y) <= radius_px * radius_px


def nan_mean_filter(image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """
    Moving-window mean that ignores NaNs (no-data, image edges).

    Trick: convolve the values with NaNs set to 0, convolve a 0/1 validity mask
    the same way, and divide. Each output is the mean of only the valid
    neighbours. Fully vectorised: no Python loop over pixels.
    """
    valid = np.isfinite(image)
    values = np.where(valid, image, 0.0).astype("float64")
    kernel = footprint.astype("float64")
    total = ndimage.convolve(values, kernel, mode="constant", cval=0.0)
    count = ndimage.convolve(valid.astype("float64"), kernel, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    mean[~valid] = np.nan
    return mean.astype("float32")


def boxcar_filter(linear: np.ndarray, radius_m: float, pixel_size_m: float) -> np.ndarray:
    """
    Circular moving-average speckle filter, equivalent to GEE
    ``focal_mean(radius, 'circle', 'meters')`` used in the UN-SPIDER recipe.
    """
    radius_px = max(1, int(round(radius_m / pixel_size_m)))
    return nan_mean_filter(linear, circular_kernel(radius_px))


def lee_filter(linear: np.ndarray, radius_m: float, pixel_size_m: float, looks: float = 4.4) -> np.ndarray:
    """
    Lee speckle filter: an adaptive alternative to the boxcar.

    In homogeneous areas (local variance close to what speckle alone explains)
    it averages strongly; near edges (high local variance) it keeps the original
    pixel. Flood boundaries stay sharper than with a plain moving average.

    ``looks`` is the equivalent number of looks of the product (about 4.4 for
    Sentinel-1 IW GRD high resolution), which sets the expected speckle noise.
    """
    radius_px = max(1, int(round(radius_m / pixel_size_m)))
    footprint = circular_kernel(radius_px)
    mean = nan_mean_filter(linear, footprint)
    mean_sq = nan_mean_filter(linear * linear, footprint)
    var = np.maximum(mean_sq - mean * mean, 0.0)

    noise_cv2 = 1.0 / looks                      # squared coefficient of variation of speckle
    with np.errstate(invalid="ignore", divide="ignore"):
        signal_var = (var - noise_cv2 * mean * mean) / (1.0 + noise_cv2)
        weight = np.clip(signal_var / var, 0.0, 1.0)
    weight = np.nan_to_num(weight, nan=0.0)
    out = mean + weight * (linear - mean)
    out[~np.isfinite(linear)] = np.nan
    return out.astype("float32")


def despeckle(linear: np.ndarray, method: str, radius_m: float, pixel_size_m: float) -> np.ndarray:
    """Dispatch to the configured speckle filter."""
    if method == "boxcar":
        return boxcar_filter(linear, radius_m, pixel_size_m)
    if method == "lee":
        return lee_filter(linear, radius_m, pixel_size_m)
    raise ValueError(f"Unknown speckle filter: {method}")
