"""
Download inputs onto the analysis grid.

- Sentinel-1 GRD sigma0 from the Copernicus Data Space Ecosystem via openEO
  (free account; the backscatter calibration runs server-side).
- DEM, JRC surface water and ESA WorldCover from Microsoft Planetary Computer
  STAC as cloud-optimised GeoTIFFs, read window-by-window with rasterio.

Everything is cached under ``data/raw``; the pipeline itself only reads local
GeoTIFFs, so processing can be rerun offline and inputs from any other source
(e.g. a GEE export) can be dropped in with the same file names.
"""

from __future__ import annotations

from pathlib import Path

from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

from .grid import Grid, mosaic_first_valid, read_to_grid, write_raster

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
OPENEO_URL = "https://openeo.dataspace.copernicus.eu"

# file name -> (STAC collection, asset key, datetime filter, resampling)
ANCILLARY = {
    "dem.tif": ("cop-dem-glo-30", "data", None, Resampling.bilinear),
    "jrc_seasonality.tif": ("jrc-gsw", "seasonality", None, Resampling.nearest),
    "worldcover.tif": ("esa-worldcover", "map", "2021", Resampling.nearest),
}


def _is_cached(path: Path) -> bool:
    """A file counts as downloaded only if it exists and is not empty (an
    interrupted download can leave a 0-byte file behind)."""
    return path.exists() and path.stat().st_size > 0


def sar_filename(epoch: str, pol: str) -> str:
    return f"s1_{epoch}_{pol}.tif"


def wgs84_bbox(grid: Grid) -> tuple[float, float, float, float]:
    """Grid bounds as (west, south, east, north) in longitude/latitude."""
    return transform_bounds(grid.crs, "EPSG:4326", *grid.bounds)


def download_ancillary(grid: Grid, out_dir: Path, overwrite: bool = False) -> None:
    """Fetch DEM, JRC seasonality and WorldCover from Planetary Computer onto ``grid``."""
    import planetary_computer
    import pystac_client

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    bbox = wgs84_bbox(grid)

    for fname, (collection, asset, when, resampling) in ANCILLARY.items():
        path = out_dir / fname
        if _is_cached(path) and not overwrite:
            print(f"  cached  {fname}")
            continue
        items = catalog.search(collections=[collection], bbox=bbox, datetime=when).item_collection()
        if len(items) == 0:
            raise RuntimeError(f"No {collection} items found for bbox {bbox}")
        arrays = [read_to_grid(item.assets[asset].href, grid, resampling) for item in items]
        write_raster(path, mosaic_first_valid(arrays), grid, nodata=float("nan"))
        print(f"  saved   {fname}  ({len(items)} tile(s) from {collection})")


def download_sar(cfg: dict, grid: Grid, out_dir: Path, overwrite: bool = False) -> None:
    """
    Fetch Sentinel-1 sigma0 (ellipsoid-corrected, linear power) for the before
    and after windows via openEO, already resampled to the analysis CRS and
    resolution. Opens a browser login the first time.
    """
    import openeo

    sar = cfg["sar"]
    west, south, east, north = wgs84_bbox(grid)
    extent = {"west": west, "south": south, "east": east, "north": north, "crs": "EPSG:4326"}
    epsg = grid.crs.to_epsg()

    properties = {"sat:orbit_state": lambda x: x == sar["orbit_state"]}
    if sar.get("relative_orbit") is not None:
        properties["sat:relative_orbit"] = lambda x: x == sar["relative_orbit"]

    conn = None
    for epoch in ("before", "after"):
        for pol in sar["polarizations"]:
            path = out_dir / sar_filename(epoch, pol)
            if _is_cached(path) and not overwrite:
                print(f"  cached  {path.name}")
                continue
            if conn is None:
                conn = openeo.connect(OPENEO_URL).authenticate_oidc()
            cube = (
                conn.load_collection(
                    "SENTINEL1_GRD",
                    spatial_extent=extent,
                    temporal_extent=sar[epoch],
                    bands=[pol],
                    properties=properties,
                )
                .sar_backscatter(coefficient="sigma0-ellipsoid")
                .resample_spatial(resolution=cfg["grid"]["resolution_m"], projection=epsg)
                .reduce_dimension(dimension="t", reducer="mean")   # mosaic slices of one pass
            )
            cube.download(str(path), format="GTiff")
            print(f"  saved   {path.name}")


def download_all(cfg: dict, grid: Grid, overwrite: bool = False) -> None:
    out_dir = Path(cfg["paths"]["raw"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Ancillary layers (Planetary Computer STAC):")
    download_ancillary(grid, out_dir, overwrite)
    print("Sentinel-1 (Copernicus Data Space, openEO):")
    download_sar(cfg, grid, out_dir, overwrite)
