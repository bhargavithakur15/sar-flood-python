"""
Pixel-wise accuracy of a flood map against a reference (GSI survey).

All counts are restricted to an evaluation region, because the reference only
covers part of the scene: outside it, "no reference flood" does not mean "dry".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ESA WorldCover class codes
WORLDCOVER_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare",
    80: "water",
    90: "wetland",
}


def confusion_counts(pred: np.ndarray, ref: np.ndarray, region: np.ndarray) -> dict[str, int]:
    """True/false positive and false negative pixel counts inside ``region``."""
    pred = pred.astype(bool) & region
    ref = ref.astype(bool) & region
    return {
        "tp": int(np.count_nonzero(pred & ref)),
        "fp": int(np.count_nonzero(pred & ~ref)),
        "fn": int(np.count_nonzero(~pred & ref)),
    }


def _safe_div(num: float, den: float) -> float:
    return float(num) / den if den else float("nan")


def scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    """
    IoU = TP / (TP + FP + FN): overlap of the two flood maps.
    Precision = TP / (TP + FP): of what we flagged, how much was really flooded.
    Recall = TP / (TP + FN): of the real flood, how much we found.
    """
    return {
        "iou": _safe_div(tp, tp + fp + fn),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
    }


def evaluate(
    pred: np.ndarray,
    ref: np.ndarray,
    region: np.ndarray,
    pixel_area_m2: float,
    landcover: np.ndarray | None = None,
) -> dict[str, float]:
    """Overall metrics plus areas in m², and recall split by land cover if given."""
    counts = confusion_counts(pred, ref, region)
    result: dict[str, float] = {**scores(**counts)}
    result.update({f"{k}_m2": v * pixel_area_m2 for k, v in counts.items()})
    result["detected_m2"] = (counts["tp"] + counts["fp"]) * pixel_area_m2
    result["reference_m2"] = (counts["tp"] + counts["fn"]) * pixel_area_m2

    if landcover is not None:
        for code, name in ((40, "cropland"), (50, "built_up")):
            in_class = region & (landcover == code)
            c = confusion_counts(pred, ref, in_class)
            result[f"recall_{name}"] = scores(**c)["recall"]
    return result


def recall_by_landcover(
    pred: np.ndarray, ref: np.ndarray, region: np.ndarray, landcover: np.ndarray, pixel_area_m2: float
) -> pd.DataFrame:
    """
    Table of reference flood area and recall per WorldCover class: the
    'why cropland detects and built-up does not' analysis, as a DataFrame.
    """
    rows = []
    for code, name in WORLDCOVER_CLASSES.items():
        in_class = region & (landcover == code)
        c = confusion_counts(pred, ref, in_class)
        ref_px = c["tp"] + c["fn"]
        if ref_px == 0:
            continue
        rows.append(
            {
                "class": name,
                "reference_flood_ha": ref_px * pixel_area_m2 / 1e4,
                "detected_ha": c["tp"] * pixel_area_m2 / 1e4,
                "recall": scores(**c)["recall"],
            }
        )
    return pd.DataFrame(rows).sort_values("reference_flood_ha", ascending=False).reset_index(drop=True)
