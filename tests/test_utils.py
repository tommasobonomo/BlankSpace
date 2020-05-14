import numpy as np

from archeoview.utils import geotiff_to_numpy, minmax_scaling


def test_geotiff_to_numpy():
    bands_names, image = geotiff_to_numpy("data/20180807-kortgene/")

    assert len(image.shape) == 3, "Image should have shape (height, width, bands)"
    assert (
        len(bands_names) == image.shape[2]
    ), "Names should correspond to number of bands"


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
