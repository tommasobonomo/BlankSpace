import os
import numpy as np


from archeoview.utils import geotiff_to_numpy, get_image_collection
from archeoview.pca import pca_image_decomposition, pca_series_decomposition


def test_pca_image_decomposition():
    _, image = geotiff_to_numpy("data/20180807-kortgene")

    pca_image, ratio = pca_image_decomposition(image)

    assert len(pca_image.shape) == 3, "Should have 3 axes height, width and bands"
    assert (
        pca_image.shape[:2] == image.shape[:2]
    ), "Height and width axes should be the same"
    assert pca_image.shape[2] == 3, "Should be the default number of components"
    assert 0 <= ratio < 1, "A ratio should always be between in the range [0,1["

    image_2 = np.rollaxis(image, 2, 0)
    pca_image_2, _ = pca_image_decomposition(image_2, n_dimensions=4, bands_first=True)

    assert (
        pca_image_2.shape[1:] == image_2.shape[1:]
    ), "Height and width axes should be the same"
    assert pca_image_2.shape[0] == 4, "Should be the given number of components, 4"


def test_pca_series_decomposition():
    base_path = "data/"
    paths = [
        os.path.join(base_path, path)
        for path in os.listdir(base_path)
        if path.endswith("kortgene")
    ]
    image_collection = get_image_collection(paths)

    pca_image, ratio = pca_series_decomposition(image_collection)

    assert 0 <= ratio < 1, "A ratio should always be between in the range [0,1["
    assert (
        pca_image.shape[-1] == 3
    ), "Default PCA decomposition should have 3 dimensions"

    bands_first_collection = np.rollaxis(image_collection, 3, 1)
    bands_first_pca, _ = pca_series_decomposition(
        bands_first_collection, bands_first=True
    )
    assert (
        bands_first_pca.shape[1:] == bands_first_collection.shape[2:]
    ), "Height and width axes should be the same"
