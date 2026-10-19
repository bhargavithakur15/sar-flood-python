import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from sarflood.evaluate import evaluate, recall_by_landcover, scores
from sarflood.export import polygonize
from sarflood.grid import grid_from_bounds, rasterize_vector, read_to_grid, write_raster

CRS = "EPSG:32654"


def test_scores_known_values():
    s = scores(tp=30, fp=20, fn=50)
    assert s["iou"] == pytest.approx(0.30)
    assert s["precision"] == pytest.approx(0.60)
    assert s["recall"] == pytest.approx(0.375)


def test_scores_empty_is_nan_not_crash():
    assert np.isnan(scores(0, 0, 0)["iou"])


def test_evaluate_restricted_to_region_and_landcover():
    pred = np.array([[1, 1, 0, 0]], dtype=bool)
    ref = np.array([[1, 0, 1, 1]], dtype=bool)
    region = np.array([[1, 1, 1, 0]], dtype=bool)   # last pixel outside evaluation region
    lc = np.array([[40, 40, 50, 50]], dtype="uint8")
    m = evaluate(pred, ref, region, pixel_area_m2=100.0, landcover=lc)
    assert (m["tp_m2"], m["fp_m2"], m["fn_m2"]) == (100.0, 100.0, 100.0)
    assert m["recall_cropland"] == 1.0
    assert m["recall_built_up"] == 0.0
    table = recall_by_landcover(pred, ref, region, lc, 100.0)
    assert set(table["class"]) == {"cropland", "built_up"}


def test_grid_snaps_to_resolution():
    g = grid_from_bounds((1003, 2007, 1095, 2091), CRS, 10)
    assert g.bounds == (1000, 2000, 1100, 2100)
    assert g.shape == (10, 10)
    assert g.pixel_area_m2 == 100


def test_rasterize_polygon_area():
    g = grid_from_bounds((0, 0, 100, 100), CRS, 10)
    gdf = gpd.GeoDataFrame(geometry=[box(20, 20, 60, 60)], crs=CRS)
    mask = rasterize_vector(gdf, g)
    assert mask.sum() * g.pixel_area_m2 == pytest.approx(1600)


def test_read_to_grid_resamples_and_aligns(tmp_path):
    fine = grid_from_bounds((0, 0, 100, 100), CRS, 10)
    data = np.arange(100, dtype="float32").reshape(10, 10)
    path = write_raster(tmp_path / "x.tif", data, fine, nodata=-9999)
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 32654

    # Same grid -> identical values
    np.testing.assert_array_equal(read_to_grid(path, fine), data)

    # Larger grid -> outside the source becomes NaN
    big = grid_from_bounds((0, 0, 200, 100), CRS, 10)
    out = read_to_grid(path, big)
    assert np.isnan(out[:, 10:]).all() and np.isfinite(out[:, :10]).all()


def test_polygonize_round_trip_area():
    g = grid_from_bounds((0, 0, 100, 100), CRS, 10)
    mask = np.zeros(g.shape, dtype=bool)
    mask[1:4, 1:4] = True    # 9 px = 900 m2
    mask[7, 7] = True        # 1 px = 100 m2
    polys = polygonize(mask, g)
    assert len(polys) == 2
    assert polys["area_m2"].tolist() == pytest.approx([900, 100])
    assert polys.crs.to_epsg() == 32654
