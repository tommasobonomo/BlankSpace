import numpy as np
from typing import Tuple

from sklearn.decomposition import PCA


def pca_decomposition(
    image: np.ndarray,
    n_dimensions: int = 3,
    bands_first: bool = False,
    normalise: bool = True,
) -> Tuple[np.ndarray, float]:
    """Applies Principal Component Analysis on the image and returns the number of components required

    Arguments:
        image -- np.ndarray with 3 axes: (height, width, bands).

    Keyword Arguments:
        n_dimensions -- Number of dimensions to return from PCA (default: {3})
        bands_first -- If true, axes of image are in this order: (bands, height, width) (default: {False})
        normalise -- If image output should be normalised between [0, 1] with min-max scaling (default: {True})

    Returns:
        A tuple with the output image with shape (height, width, n_dimensions) and the explained variance
        ratio of PCA, i.e. how much variance is explained by these n_dimensions
    """

    # Let's get the bands first files in bands last
    if bands_first:
        image = np.rollaxis(image, 0, 3)

    height, width, bands = image.shape

    # Must flatten each band to 1D
    flattened_image = image.reshape(-1, bands)

    # PCA
    pca = PCA(n_components=n_dimensions)
    flattened_pca = pca.fit_transform(flattened_image)

    # Back to 2D image
    pca_image = flattened_pca.reshape(height, width, n_dimensions)

    # Possibly normalise between [0, 1]
    if normalise:
        for band_idx in range(n_dimensions):
            min_band = pca_image[:, :, band_idx].min()
            max_band = pca_image[:, :, band_idx].max()
            pca_image[:, :, band_idx] -= min_band
            pca_image[:, :, band_idx] /= max_band - min_band

    return pca_image, pca.explained_variance_ratio_
