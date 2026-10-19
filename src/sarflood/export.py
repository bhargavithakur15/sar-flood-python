"""Outputs: raster -> vector polygons, and a quick-look map figure."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import shapes
from shapely.geometry import shape

from .grid import Grid


def polygonize(mask: np.ndarray, grid: Grid, min_area_m2: float = 0.0) -> gpd.GeoDataFrame:
    """
    Turn a boolean flood mask into polygons (raster -> vector), one row per
    connected patch, with its area in m² and hectares.

    Vector output is what downstream users actually consume: it can be
    intersected with parcels, roads or insured assets in any GIS.
    """
    mask_u8 = mask.astype("uint8")
    geoms = [
        shape(geom)
        for geom, value in shapes(mask_u8, mask=mask_u8.astype(bool), transform=grid.transform)
        if value == 1
    ]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=grid.crs)
    gdf["area_m2"] = gdf.geometry.area
    gdf = gdf[gdf["area_m2"] >= min_area_m2].copy()
    gdf["area_ha"] = gdf["area_m2"] / 1e4
    return gdf.sort_values("area_m2", ascending=False).reset_index(drop=True)


def quicklook(
    path: str | Path,
    after_db: np.ndarray,
    detected: np.ndarray,
    reference: np.ndarray,
    grid: Grid,
    title: str,
    zoom_margin_m: float | None = 1000.0,
) -> Path:
    """
    Detection (blue) vs GSI reference (red), overlap in magenta, on the
    unfiltered post-event backscatter, so the speckle texture and edges stay
    sharp. Mirrors the comparison figure on the project slides.

    ``zoom_margin_m``: crop to the GSI reference extent plus this margin, so
    the flooded area fills the figure. ``None`` shows the whole grid.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows, cols = slice(None), slice(None)
    if zoom_margin_m is not None and reference.any():
        pad = int(round(zoom_margin_m / abs(grid.transform.a)))
        r_idx, c_idx = np.nonzero(reference)
        rows = slice(max(r_idx.min() - pad, 0), min(r_idx.max() + pad + 1, grid.height))
        cols = slice(max(c_idx.min() - pad, 0), min(c_idx.max() + pad + 1, grid.width))

    after_db, detected, reference = after_db[rows, cols], detected[rows, cols], reference[rows, cols]
    res = abs(grid.transform.a)
    left = grid.transform.c + (cols.start or 0) * res
    top = grid.transform.f - (rows.start or 0) * res
    extent = (left, left + after_db.shape[1] * res, top - after_db.shape[0] * res, top)

    overlay = np.zeros((*after_db.shape, 4), dtype="float32")
    overlay[reference & ~detected] = (0.86, 0.15, 0.15, 0.70)
    overlay[detected & ~reference] = (0.15, 0.35, 0.90, 0.70)
    overlay[detected & reference] = (0.75, 0.20, 0.75, 0.80)

    height, width = after_db.shape
    fig, ax = plt.subplots(figsize=(10, 10 * height / width + 0.8), dpi=200)
    ax.imshow(after_db, cmap="gray", vmin=-25, vmax=-5, extent=extent, interpolation="nearest")
    ax.imshow(overlay, extent=extent, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(f"Easting (m, {grid.crs.to_string()})")
    ax.set_ylabel("Northing (m)")
    ax.legend(
        handles=[
            Patch(color=(0.86, 0.15, 0.15), label="GSI reference only (missed)"),
            Patch(color=(0.15, 0.35, 0.90), label="SAR only (false alarm)"),
            Patch(color=(0.75, 0.20, 0.75), label="Both (hit)"),
        ],
        loc="lower left",
        fontsize=8,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
