import numpy as np
import pytest

from sarflood.detect import change_ratio, threshold_ratio
from sarflood.masks import (
    apply_masks,
    height_above_nearest_drainage,
    permanent_water_mask,
    remove_small_patches,
    slope_percent,
)


def test_ratio_flags_darkening_only():
    before = np.array([[0.10, 0.10, 0.10]], dtype="float32")
    after = np.array([[0.02, 0.10, 0.20]], dtype="float32")   # dropped, same, brightened
    flood = threshold_ratio(change_ratio(before, after), 1.25)
    assert flood.tolist() == [[True, False, False]]


def test_ratio_nan_and_zero_are_not_flood():
    before = np.array([[np.nan, 0.1]], dtype="float32")
    after = np.array([[0.01, 0.0]], dtype="float32")
    assert not threshold_ratio(change_ratio(before, after), 1.25).any()


def test_ratio_shape_mismatch_raises():
    with pytest.raises(ValueError):
        change_ratio(np.ones((2, 2)), np.ones((3, 3)))


def test_threshold_must_exceed_one():
    with pytest.raises(ValueError):
        threshold_ratio(np.ones((2, 2)), 0.8)


def test_permanent_water_combines_sources():
    osm = np.array([[True, False, False]])
    jrc = np.array([[0.0, 12.0, 5.0]])
    assert permanent_water_mask(osm, jrc, 10).tolist() == [[True, True, False]]


def test_hand_on_tilted_plane():
    # Elevation rises 1 m per pixel away from a river in column 0
    dem = np.tile(np.arange(10, dtype="float32"), (5, 1)) + 100.0
    river = np.zeros_like(dem, dtype=bool)
    river[:, 0] = True
    hand = height_above_nearest_drainage(dem, river, pixel_size_m=10)
    np.testing.assert_allclose(hand[0], np.arange(10))


def test_hand_max_distance_masks_far_pixels():
    dem = np.zeros((1, 10), dtype="float32")
    river = np.zeros_like(dem, dtype=bool)
    river[0, 0] = True
    hand = height_above_nearest_drainage(dem, river, pixel_size_m=10, max_distance_m=30)
    assert np.isfinite(hand[0, :4]).all() and np.isnan(hand[0, 4:]).all()


def test_hand_requires_drainage():
    with pytest.raises(ValueError):
        height_above_nearest_drainage(np.zeros((3, 3)), np.zeros((3, 3), bool), 10)


def test_slope_of_known_plane():
    dem = np.tile(np.arange(20, dtype="float32") * 2.0, (20, 1))  # 2 m rise per 10 m pixel = 20 %
    np.testing.assert_allclose(slope_percent(dem, 10)[5:15, 5:15], 20.0, rtol=1e-6)


def test_remove_small_patches_uses_eight_connectivity():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = mask[1, 1] = mask[2, 2] = True   # 3 pixels touching only diagonally = one patch
    mask[6:9, 6:9] = True                         # 9-pixel block
    out = remove_small_patches(mask, min_pixels=3)
    assert not out[:3, :3].any()                  # 3 is not > 3 -> removed
    assert out[6:9, 6:9].all()
    assert remove_small_patches(mask, min_pixels=2)[:3, :3].sum() == 3


def test_apply_masks_cascade():
    cand = np.ones((1, 4), dtype=bool)
    water = np.array([[True, False, False, False]])
    hand = np.array([[0.0, 2.0, 50.0, np.nan]])
    out = apply_masks(cand, water, hand, hand_max_m=15, mmu_pixels=0)
    # water removed, low HAND kept, high HAND removed, unknown HAND removed
    assert out.tolist() == [[False, True, False, False]]
