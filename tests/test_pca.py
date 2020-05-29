import numpy as np


from archeoview.utils import geotiff_to_numpy
from archeoview.pca import pca_image_decomposition


def test_pca_image_decomposition():
    _, image = geotiff_to_numpy("data/20180807-kortgene")

    pca_image, ratio = pca_image_decomposition(image)

    assert len(pca_image.shape) == 3, "Should have 3 axes height, width and bands"
    assert (
        pca_image.shape[:2] == image.shape[:2]
    ), "Height and width axes should be the same"
    assert pca_image.shape[2] == 3, "Should be the default number of components"
    assert (
        ratio >= 0 and ratio <= 1
    ), "A ratio should always be between in the range [0,1]"

    image_2 = np.rollaxis(image, 2, 0)
    pca_image_2, _ = pca_image_decomposition(image_2, n_dimensions=4, bands_first=True)

    assert (
        pca_image_2.shape[1:] == image_2.shape[1:]
    ), "Height and width axes should be the same"
    assert pca_image_2.shape[0] == 4, "Should be the given number of components, 4"
