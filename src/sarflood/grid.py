"""
A common analysis grid, and helpers to put rasters and vectors onto it.

Every layer (SAR, DEM, water, land cover, GSI reference) is aligned to one
grid before any pixel-wise math. Mismatched grids are the most common silent
bug in raster analysis: arrays of the same shape that cover slightly
different ground.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT


@dataclass(frozen=True)
class Grid:
    """Pixel grid: CRS, affine transform and size. Equivalent to a GeoTIFF profile."""

    crs: CRS
    transform: Affine
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def pixel_area_m2(self) -> float:
        return abs(self.transform.a * self.transform.e)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left, top = self.transform.c, self.transform.f
        right = left + self.width * self.transform.a
        bottom = top + self.height * self.transform.e
        return (left, bottom, right, top)

    def profile(self, dtype: str, nodata=None) -> dict:
        """rasterio profile for writing a cloud-optimised-friendly GeoTIFF on this grid."""
        return {
            "driver": "GTiff",
            "crs": self.crs,
            "transform": self.transform,
            "width": self.width,
            "height": self.height,
            "count": 1,
            "dtype": dtype,
            "nodata": nodata,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }


def grid_from_bounds(bounds, crs, resolution: float) -> Grid:
    """
    Build a grid covering ``bounds`` with pixel edges snapped to multiples of
    ``resolution``, so repeated runs and different inputs align exactly.
    """
    left, bottom, right, top = bounds
    left = np.floor(left / resolution) * resolution
    bottom = np.floor(bottom / resolution) * resolution
    right = np.ceil(right / resolution) * resolution
    top = np.ceil(top / resolution) * resolution
    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))
    transform = Affine(resolution, 0, left, 0, -resolution, top)
    return Grid(CRS.from_user_input(crs), transform, width, height)


def analysis_grid(reference_path: str | Path, crs, resolution: float, buffer_m: float) -> Grid:
    """
    Analysis extent = bounding box of the reference polygons buffered by
    ``buffer_m`` (same framing as the GEE script: ``gsiGeom.buffer(3000).bounds()``).
    Buffering is done in the metric CRS so the distance really is in metres.
    """
    ref = gpd.read_file(reference_path).to_crs(crs)
    bounds = ref.geometry.union_all().buffer(buffer_m).bounds
    return grid_from_bounds(bounds, crs, resolution)


def read_to_grid(
    source: str | Path,
    grid: Grid,
    resampling: Resampling = Resampling.bilinear,
    band: int = 1,
) -> np.ndarray:
    """
    Read one band of any GDAL-readable raster (local file or https COG) resampled
    onto ``grid``. Pixels outside the source, or equal to its nodata, become NaN.

    Uses a ``WarpedVRT``: GDAL reprojects on the fly while reading, so only the
    window we need is fetched, which matters for remote cloud-optimised GeoTIFFs.
    """
    with rasterio.open(source) as src:
        with WarpedVRT(
            src,
            crs=grid.crs,
            transform=grid.transform,
            width=grid.width,
            height=grid.height,
            resampling=resampling,
            src_nodata=src.nodata,
            nodata=np.nan,
            dtype="float32",
        ) as vrt:
            return vrt.read(band).astype("float32")


def mosaic_first_valid(arrays: list[np.ndarray]) -> np.ndarray:
    """Combine aligned arrays, taking the first non-NaN value at each pixel."""
    if not arrays:
        raise ValueError("Nothing to mosaic")
    out = arrays[0].copy()
    for arr in arrays[1:]:
        gaps = np.isnan(out)
        out[gaps] = arr[gaps]
    return out


def rasterize_vector(gdf: gpd.GeoDataFrame, grid: Grid, all_touched: bool = False) -> np.ndarray:
    """Burn polygons into a boolean mask on ``grid`` (True inside any polygon)."""
    gdf = gdf.to_crs(grid.crs)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros(grid.shape, dtype=bool)
    burned = rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )
    return burned.astype(bool)


def write_raster(path: str | Path, array: np.ndarray, grid: Grid, nodata=None) -> Path:
    """Write a single-band array on ``grid`` to a compressed, tiled GeoTIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.dtype == bool:
        array = array.astype("uint8")
    with rasterio.open(path, "w", **grid.profile(str(array.dtype), nodata)) as dst:
        dst.write(array, 1)
    return path
