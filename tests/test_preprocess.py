import numpy as np
import pytest

from sarflood.preprocess import (
    boxcar_filter,
    circular_kernel,
    db_to_linear,
    despeckle,
    lee_filter,
    linear_to_db,
    nan_mean_filter,
)


def test_db_linear_round_trip():
    db = np.array([-25.0, -10.0, 0.0, 3.0])
    np.testing.assert_allclose(linear_to_db(db_to_linear(db)), db, atol=1e-9)


def test_linear_to_db_non_positive_is_nan():
    out = linear_to_db(np.array([0.0, -1.0, 1.0]))
    assert np.isnan(out[0]) and np.isnan(out[1]) and out[2] == 0.0


def test_circular_kernel_shape_and_corners():
    k = circular_kernel(2)
    assert k.shape == (5, 5)
    assert k[2, 2] and not k[0, 0]


def test_mean_filter_constant_image_unchanged_including_edges():
    img = np.full((20, 20), 0.05, dtype="float32")
    out = nan_mean_filter(img, circular_kernel(3))
    np.testing.assert_allclose(out, 0.05, rtol=1e-6)


def test_mean_filter_ignores_nan_neighbours():
    img = np.full((9, 9), 2.0, dtype="float32")
    img[4, 5] = np.nan
    out = nan_mean_filter(img, circular_kernel(1))
    assert out[4, 4] == pytest.approx(2.0)   # the NaN neighbour must not pull the mean down
    assert np.isnan(out[4, 5])               # invalid pixels stay invalid


def test_filters_reduce_speckle_variance():
    rng = np.random.default_rng(0)
    # Gamma-distributed multiplicative speckle around a constant backscatter
    img = (0.1 * rng.gamma(shape=4.4, scale=1 / 4.4, size=(200, 200))).astype("float32")
    for out in (boxcar_filter(img, 50, 10), lee_filter(img, 50, 10)):
        assert np.nanstd(out) < 0.5 * np.nanstd(img)
        assert np.nanmean(out) == pytest.approx(0.1, rel=0.05)


def test_lee_preserves_edge_better_than_boxcar():
    img = np.full((60, 60), 0.2, dtype="float32")
    img[:, 30:] = 0.01   # sharp land/water boundary
    box = boxcar_filter(img, 50, 10)
    lee = lee_filter(img, 50, 10)
    # pixel right next to the edge on the bright side
    assert abs(lee[30, 29] - 0.2) < abs(box[30, 29] - 0.2)


def test_despeckle_rejects_unknown_method():
    with pytest.raises(ValueError):
        despeckle(np.ones((5, 5)), "median", 50, 10)
