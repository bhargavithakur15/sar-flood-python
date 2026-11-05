"""
True-colour optical basemap for the slide figures.

Preferred: a cloud-masked median composite of Sentinel-2 L2A (B4/B3/B2) for a
pre-flood summer window, from Microsoft Planetary Computer (no account needed).
Fallback: Esri World Imagery tiles via contextily.

The result is warped onto the analysis grid once and cached in ``data/raw/``,
so later runs work offline. The basemap is only used for display; it plays no
part in detection or evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from .acquire import STAC_URL, _is_cached, wgs84_bbox
from .grid import Grid, read_to_grid

S2_FILE = "s2_truecolor.tif"
ESRI_FILE = "esri_worldimagery.tif"

S2_WINDOW = "2022-06-01/2022-07-31"   # pre-flood summer, fields green
S2_MAX_CLOUD = 30                      # scene-level cloud cover, percent
S2_MAX_SCENES = 12                     # least cloudy scenes used in the median
S2_BANDS = ("B04", "B03", "B02")       # red, green, blue
# Scene Classification Layer values treated as unusable: no data, saturated,
# cloud shadow, cloud (medium / high probability), thin cirrus, snow.
SCL_INVALID = (0, 1, 3, 8, 9, 10, 11)

S2_CREDIT = "Contains modified Copernicus Sentinel data 2022 (Sentinel-2 L2A median, Jun–Jul)"


@dataclass
class Basemap:
    rgb: np.ndarray      # (height, width, 3) floats in 0–1, ready for imshow
    credit: str


def _write_rgb(path: Path, bands: list[np.ndarray], grid: Grid, credit: str) -> None:
    profile = grid.profile("float32", nodata=float("nan"))
    profile["count"] = 3
    with rasterio.open(path, "w", **profile) as dst:
        for i, band in enumerate(bands, start=1):
            dst.write(band.astype("float32"), i)
        dst.update_tags(credit=credit)


def _read_rgb(path: Path, grid: Grid) -> tuple[np.ndarray, str]:
    rgb = np.dstack([read_to_grid(path, grid, Resampling.bilinear, band=b) for b in (1, 2, 3)])
    with rasterio.open(path) as src:
        credit = src.tags().get("credit", "")
    return rgb, credit


def download_sentinel2(grid: Grid, path: Path) -> None:
    """Median of cloud-masked Sentinel-2 L2A scenes, warped onto ``grid``."""
    import planetary_computer
    import pystac_client

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    items = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=wgs84_bbox(grid),
        datetime=S2_WINDOW,
        query={"eo:cloud_cover": {"lt": S2_MAX_CLOUD}},
    ).item_collection()
    if len(items) == 0:
        raise RuntimeError(f"No Sentinel-2 scenes below {S2_MAX_CLOUD}% cloud in {S2_WINDOW}")
    items = sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))[:S2_MAX_SCENES]

    # Per-scene validity from the Scene Classification Layer (20 m, nearest).
    valid = []
    for item in items:
        scl = read_to_grid(item.assets["SCL"].href, grid, Resampling.nearest)
        valid.append(np.isfinite(scl) & ~np.isin(scl, SCL_INVALID))

    bands = []
    for band in S2_BANDS:
        stack = np.stack(
            [np.where(ok, read_to_grid(it.assets[band].href, grid, Resampling.bilinear), np.nan)
             for it, ok in zip(items, valid)]
        )
        stack[stack <= 0] = np.nan
        bands.append(np.nanmedian(stack, axis=0))
    dates = sorted(it.datetime.date().isoformat() for it in items)
    _write_rgb(path, bands, grid, S2_CREDIT)
    print(f"  saved   {path.name}  (median of {len(items)} scenes, {dates[0]} to {dates[-1]})")


def download_esri(grid: Grid, path: Path) -> None:
    """Fallback: Esri World Imagery tiles, warped onto ``grid``."""
    import contextily as ctx

    provider = ctx.providers.Esri.WorldImagery
    west, south, east, north = wgs84_bbox(grid)
    tmp = path.with_name("esri_worldimagery_webmercator.tif")
    ctx.bounds2raster(west, south, east, north, str(tmp), source=provider, ll=True, zoom=15)
    bands = [read_to_grid(tmp, grid, Resampling.bilinear, band=b) for b in (1, 2, 3)]
    credit = f"Imagery: {provider['attribution']}"
    _write_rgb(path, bands, grid, credit)
    print(f"  saved   {path.name}  (Esri World Imagery fallback)")


def stretch(rgb: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0, gamma: float = 0.8) -> np.ndarray:
    """
    Contrast stretch to 0–1. One low/high pair is used for all three bands so
    colours keep their balance; gamma < 1 brightens the mid-tones.
    No-data becomes white so it reads as background on a slide.
    """
    finite = rgb[np.isfinite(rgb)]
    if finite.size == 0:
        raise ValueError("Basemap has no valid pixels")
    lo, hi = np.percentile(finite, [low_pct, high_pct])
    out = np.clip((rgb - lo) / (hi - lo), 0, 1) ** gamma
    out[~np.isfinite(rgb).all(axis=2)] = 1.0
    return out


def get_basemap(cfg: dict, grid: Grid, overwrite: bool = False) -> Basemap:
    """
    Return the cached basemap, downloading it first if needed:
    Sentinel-2 composite, then Esri World Imagery if that fails.
    """
    raw = Path(cfg["paths"]["raw"])
    raw.mkdir(parents=True, exist_ok=True)
    s2_path, esri_path = raw / S2_FILE, raw / ESRI_FILE

    if not overwrite:
        if _is_cached(s2_path):
            rgb, credit = _read_rgb(s2_path, grid)
            return Basemap(stretch(rgb), credit or S2_CREDIT)
        if _is_cached(esri_path):
            rgb, credit = _read_rgb(esri_path, grid)
            return Basemap(np.clip(rgb / 255.0, 0, 1), credit)

    try:
        download_sentinel2(grid, s2_path)
        rgb, credit = _read_rgb(s2_path, grid)
        return Basemap(stretch(rgb), credit)
    except Exception as err:  # network, catalogue or all-cloud problems
        print(f"  Sentinel-2 basemap failed ({err}); trying Esri World Imagery")
    download_esri(grid, esri_path)
    rgb, credit = _read_rgb(esri_path, grid)
    return Basemap(np.clip(rgb / 255.0, 0, 1), credit)
