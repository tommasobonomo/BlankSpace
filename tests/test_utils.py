import numpy as np

from archeoview.utils import (
    interpolate,
    interpolate_bands,
    geotiff_to_numpy,
    minmax_scaling,
)


def test_interpolate():
    test_array = np.random.rand(14, 20)
    assert interpolate(test_array, (28, 40)).shape == (
        28,
        40,
    ), "Should be able to do simple scaling"

    assert interpolate(test_array, (28, 28)).shape == (
        28,
        28,
    ), "Should be able to scale differently for different dimensions"

    test_array2 = np.random.rand(25, 31)
    assert interpolate(test_array2, (28, 35)).shape == (
        28,
        35,
    ), "Should be able to scale very weird shapes"


def test_interpolate_bands():
    assert interpolate_bands([]).shape == (
        0,
    ), "If given no bands should return no bands"

    test_bands = [np.random.rand(14, 20)] * 3
    assert interpolate_bands(test_bands).shape == (
        3,
        14,
        20,
    ), "Should return np.ndarray with correct shape"

    test_bands2 = [np.random.rand(14, 20)] * 3 + [np.random.rand(12, 15)] * 2
    assert interpolate_bands(test_bands2).shape == (
        5,
        14,
        20,
    ), "Should return all bands interpolated correctly to a certain dimension"
    assert interpolate_bands(test_bands2, interpolation=False).shape == (
        3,
        14,
        20,
    ), "Should return all high-res bands and skip lower res bands"


def test_geotiff_to_numpy():
    bands_names, image = geotiff_to_numpy("data/20180807-kortgene/")

    assert len(image.shape) == 3, "Image should have shape (height, width, bands)"
    assert (
        len(bands_names) == image.shape[2]
    ), "Names should correspond to number of bands"

    _, image2 = geotiff_to_numpy(
        "data/20180926-kortgene-multires/", interpolation=False
    )
    assert image2.shape == (
        174,
        252,
        6,
    ), "Without interpolation should have only 6 bands"


def test_minmax_scaling():
    _, image = geotiff_to_numpy("data/20180807-kortgene/")
    scaled_image = minmax_scaling(image)

    assert (
        (scaled_image >= 0) & (scaled_image <= 1)
    ).all(), "All values should be in range [0, 1]"
    assert image.shape == scaled_image.shape, "Output shape should be the same as input"

    bands_first_image = np.rollaxis(image, 2, 0)
    bands_first_scaled_image = minmax_scaling(bands_first_image, bands_first=True)
    assert (
        (bands_first_scaled_image >= 0) & (bands_first_scaled_image <= 1)
    ).all(), "All values should be in range [0, 1]"
    assert (
        bands_first_image.shape == bands_first_scaled_image.shape
    ), "Output shape should be the same as input"
