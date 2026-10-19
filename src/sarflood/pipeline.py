"""
End-to-end run: aligned inputs -> speckle filter -> ratio threshold -> masks
-> evaluation against GSI -> GeoTIFF / GeoPackage / CSV / PNG outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.enums import Resampling

from . import acquire
from .detect import change_ratio, threshold_ratio
from .evaluate import evaluate, recall_by_landcover
from .export import polygonize, quicklook
from .grid import Grid, analysis_grid, rasterize_vector, read_to_grid, write_raster
from .masks import apply_masks, height_above_nearest_drainage, permanent_water_mask, slope_percent
from .preprocess import db_to_linear, despeckle, linear_to_db


@dataclass
class Context:
    """Layers shared by every polarization: grid, reference, region and masks."""

    grid: Grid
    reference: np.ndarray
    region: np.ndarray
    permanent_water: np.ndarray
    hand: np.ndarray
    slope: np.ndarray | None
    landcover: np.ndarray


def ensure_linear(arr: np.ndarray) -> np.ndarray:
    """
    Return backscatter in linear power with invalid pixels as NaN.

    openEO delivers linear sigma0; a GEE export is in dB. Real backscatter in
    dB is almost always negative, linear power is always positive, so the
    median sign tells the two apart.
    """
    arr = arr.astype("float32")
    if np.nanmedian(arr) < 0:
        arr = db_to_linear(arr)
    arr[~(arr > 0)] = np.nan
    return arr


def build_context(cfg: dict) -> Context:
    ref_cfg, mask_cfg = cfg["reference"], cfg["masks"]
    res = cfg["grid"]["resolution_m"]
    grid = analysis_grid(ref_cfg["gsi_flood"], cfg["grid"]["crs"], res, ref_cfg["aoi_buffer_m"])
    raw = Path(cfg["paths"]["raw"])

    gsi = gpd.read_file(ref_cfg["gsi_flood"]).to_crs(grid.crs)
    reference = rasterize_vector(gsi, grid)
    eval_area = gpd.GeoDataFrame(geometry=gsi.buffer(ref_cfg["eval_buffer_m"]), crs=grid.crs)
    region = rasterize_vector(eval_area, grid)

    osm = rasterize_vector(gpd.read_file(ref_cfg["osm_water"]), grid, all_touched=True)
    jrc = read_to_grid(raw / "jrc_seasonality.tif", grid, Resampling.nearest)
    permanent = permanent_water_mask(osm, jrc, mask_cfg["jrc_seasonality_months"])

    dem = read_to_grid(raw / "dem.tif", grid, Resampling.bilinear)
    # Drainage reference for HAND: same water layers (rivers + sea). The GEE
    # version added MERIT Hydro channels, which are not on Planetary Computer.
    hand = height_above_nearest_drainage(dem, permanent, res, mask_cfg["hand_max_distance_m"])
    slope = slope_percent(dem, res) if mask_cfg.get("slope_max_percent") is not None else None

    landcover = read_to_grid(raw / "worldcover.tif", grid, Resampling.nearest)
    return Context(grid, reference, region, permanent, hand, slope, np.nan_to_num(landcover).astype("uint8"))


def run_polarization(cfg: dict, ctx: Context, pol: str) -> dict:
    """Detect, mask, evaluate and export one polarization. Returns its metrics."""
    det, mask_cfg = cfg["detection"], cfg["masks"]
    grid, res = ctx.grid, cfg["grid"]["resolution_m"]
    raw, out = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["outputs"])

    before = ensure_linear(read_to_grid(raw / acquire.sar_filename("before", pol), grid))
    after = ensure_linear(read_to_grid(raw / acquire.sar_filename("after", pol), grid))

    before_f = despeckle(before, det["speckle_filter"], det["smoothing_radius_m"], res)
    after_f = despeckle(after, det["speckle_filter"], det["smoothing_radius_m"], res)

    ratio = change_ratio(before_f, after_f)
    candidates = threshold_ratio(ratio, det["ratio_threshold"])
    flood = apply_masks(
        candidates,
        ctx.permanent_water,
        ctx.hand,
        mask_cfg["hand_max_m"],
        mask_cfg["mmu_pixels"],
        ctx.slope,
        mask_cfg.get("slope_max_percent"),
    )

    metrics = {"polarization": pol, **evaluate(flood, ctx.reference, ctx.region, grid.pixel_area_m2, ctx.landcover)}
    metrics["raw_candidates_iou"] = evaluate(candidates, ctx.reference, ctx.region, grid.pixel_area_m2)["iou"]

    write_raster(out / f"change_ratio_{pol}.tif", ratio, grid, nodata=float("nan"))
    write_raster(out / f"flood_{pol}.tif", flood, grid, nodata=None)
    polygonize(flood, grid).to_file(out / f"flood_{pol}.gpkg", driver="GPKG")
    quicklook(
        out / f"quicklook_{pol}.png",
        linear_to_db(after),          # raw, unfiltered backscatter for a sharp background
        flood,
        ctx.reference,
        grid,
        f"{cfg['name']} | Sentinel-1 {pol} | IoU {metrics['iou']:.2f}",
    )
    recall_by_landcover(flood, ctx.reference, ctx.region, ctx.landcover, grid.pixel_area_m2).to_csv(
        out / f"recall_by_landcover_{pol}.csv", index=False
    )
    return metrics


def run(cfg: dict) -> pd.DataFrame:
    out = Path(cfg["paths"]["outputs"])
    out.mkdir(parents=True, exist_ok=True)
    ctx = build_context(cfg)
    print(f"Grid: {ctx.grid.width} x {ctx.grid.height} px at {cfg['grid']['resolution_m']} m, {ctx.grid.crs}")

    rows = [run_polarization(cfg, ctx, pol) for pol in cfg["sar"]["polarizations"]]
    table = pd.DataFrame(rows).set_index("polarization")
    table.to_csv(out / "metrics.csv")
    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in cfg.items() if k != "root"}, f, indent=2, default=str)
    return table
