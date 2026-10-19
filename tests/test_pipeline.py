"""End-to-end run on a small synthetic scene: a flooded field next to a river."""

import geopandas as gpd
import numpy as np
import pytest
import yaml
from shapely.geometry import box

from sarflood.config import load_config
from sarflood.grid import analysis_grid, write_raster
from sarflood.pipeline import run

CRS = "EPSG:32654"


@pytest.fixture
def project(tmp_path):
    (tmp_path / "config").mkdir()
    ref_dir = tmp_path / "data" / "reference"
    ref_dir.mkdir(parents=True)
    raw = tmp_path / "data" / "raw"
    raw.mkdir()

    # Flooded field 400 m x 400 m, a river along the west side
    flood_poly = box(400_000, 4_220_000, 400_400, 4_220_400)
    gpd.GeoDataFrame(geometry=[flood_poly], crs=CRS).to_file(ref_dir / "gsi.geojson")
    river = box(399_500, 4_219_500, 399_600, 4_220_900)
    gpd.GeoDataFrame(geometry=[river], crs=CRS).to_file(ref_dir / "osm.geojson")

    cfg = {
        "name": "synthetic",
        "grid": {"crs": CRS, "resolution_m": 10},
        "reference": {
            "gsi_flood": "data/reference/gsi.geojson",
            "osm_water": "data/reference/osm.geojson",
            "aoi_buffer_m": 500,
            "eval_buffer_m": 100,
        },
        "sar": {"before": ["a", "b"], "after": ["c", "d"], "orbit_state": "descending",
                "relative_orbit": None, "polarizations": ["VH"]},
        "detection": {"speckle_filter": "boxcar", "smoothing_radius_m": 50, "ratio_threshold": 1.25},
        "masks": {"jrc_seasonality_months": 10, "hand_max_m": 15, "hand_max_distance_m": 3000,
                  "slope_max_percent": None, "mmu_pixels": 20},
        "paths": {"raw": "data/raw", "outputs": "outputs"},
    }
    with open(tmp_path / "config" / "test.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    grid = analysis_grid(ref_dir / "gsi.geojson", CRS, 10, 500)
    rng = np.random.default_rng(1)
    speckle = lambda: rng.gamma(4.4, 1 / 4.4, grid.shape).astype("float32")

    rows, cols = np.indices(grid.shape)
    x = grid.transform.c + (cols + 0.5) * 10
    y = grid.transform.f - (rows + 0.5) * 10
    in_flood = (x > 400_000) & (x < 400_400) & (y > 4_220_000) & (y < 4_220_400)

    before = 0.05 * speckle()
    after = np.where(in_flood, 0.005, 0.05) * speckle()          # water goes dark
    write_raster(raw / "s1_before_VH.tif", 10 * np.log10(before), grid, nodata=float("nan"))  # dB like GEE
    write_raster(raw / "s1_after_VH.tif", after, grid, nodata=float("nan"))                   # linear like openEO
    write_raster(raw / "dem.tif", np.full(grid.shape, 5.0, "float32"), grid, nodata=float("nan"))
    write_raster(raw / "jrc_seasonality.tif", np.zeros(grid.shape, "float32"), grid, nodata=float("nan"))
    write_raster(raw / "worldcover.tif", np.full(grid.shape, 40.0, "float32"), grid, nodata=float("nan"))
    return tmp_path


def test_pipeline_end_to_end(project):
    table = run(load_config(project / "config" / "test.yaml"))
    vh = table.loc["VH"]
    assert vh["iou"] > 0.8
    assert vh["recall_cropland"] > 0.8
    out = project / "outputs"
    for name in ("flood_VH.tif", "flood_VH.gpkg", "change_ratio_VH.tif", "quicklook_VH.png",
                 "metrics.csv", "recall_by_landcover_VH.csv"):
        assert (out / name).exists(), name
    polys = gpd.read_file(out / "flood_VH.gpkg")
    # True flood is 160,000 m2. A 50 m boxcar smears the dark area outward by
    # roughly half the window on every side, so the map overestimates slightly.
    assert 160_000 * 0.9 < polys["area_m2"].sum() < 160_000 * 1.35


def test_lee_filter_tracks_flood_edge_more_tightly(project):
    cfg = load_config(project / "config" / "test.yaml")
    box_area = run(cfg).loc["VH", "detected_m2"]
    cfg["detection"]["speckle_filter"] = "lee"
    lee_area = run(cfg).loc["VH", "detected_m2"]
    assert abs(lee_area - 160_000) < abs(box_area - 160_000)
