"""
Change detection: flag pixels whose backscatter dropped between two dates.

Calm floodwater acts like a mirror and reflects radar energy away from the
sensor (specular reflection), so flooded land turns dark. A ratio of
before/after power above a threshold means a drop. A ratio rather than a
difference is used because speckle noise is multiplicative.
"""

from __future__ import annotations

import numpy as np


def change_ratio(before_linear: np.ndarray, after_linear: np.ndarray) -> np.ndarray:
    """Before / after in linear power. Pixels where either input is invalid become NaN."""
    if before_linear.shape != after_linear.shape:
        raise ValueError(
            f"Inputs must be on the same grid, got {before_linear.shape} and {after_linear.shape}"
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = before_linear / after_linear
    ratio[~np.isfinite(ratio)] = np.nan
    return ratio.astype("float32")


def threshold_ratio(ratio: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean flood candidates: ratio strictly above ``threshold`` (NaN -> False)."""
    if threshold <= 1.0:
        raise ValueError("threshold must be > 1: a value <= 1 would flag brightening, not darkening")
    return np.nan_to_num(ratio, nan=0.0) > threshold
