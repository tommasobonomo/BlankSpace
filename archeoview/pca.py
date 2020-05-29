import numpy as np
from typing import Tuple
from sklearn.decomposition import PCA

from archeoview.utils import minmax_scaling


def pca_image_decomposition(
    image: np.ndarray,
    n_dimensions: int = 3,
    bands_first: bool = False,
    normalise: bool = True,
) -> Tuple[np.ndarray, float]:
    """Applies Principal Component Analysis on the image and returns the number of components required

    Arguments:
        image -- np.ndarray with 3 axes: `(height, width, bands)`.

    Keyword Arguments:
        n_dimensions -- Number of dimensions to return from PCA (default: {3})
        bands_first -- If true, axes of image are in this order: `(bands, height, width)` (default: {False})
        normalise -- If image output should be normalised between [0, 1] with min-max scaling (default: {True})

    Returns:
        A tuple with the output image with shape `(height, width, n_dimensions) if bands_first = False else (n_dimensions, height, width)`
        and the explained variance ratio of PCA, i.e. how much variance is explained by these n_dimensions
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
        pca_image = minmax_scaling(pca_image)

    # Return to bands_first if that's the case
    if bands_first:
        pca_image = np.rollaxis(pca_image, 2, 0)

    return pca_image, pca.explained_variance_ratio_.sum()


def pca_series_decomposition(
    collection: np.ndarray, bands_first: bool = False, normalise: bool = True
) -> Tuple[np.ndarray, float]:

    # Idea is we have collection of RGB images (n_images, height, width, 3)
    # For each band, we treat the various images as dimensions of a point
    # We can apply PCA to reduce dimensionality to the first singular component.

    n_images = collection.shape[0]
    pca_bands = []
    for band_idx in range(3):
        band_image = np.rollaxis(collection[:, :, :, band_idx], 0, 3)
        height, width, _ = band_image.shape

        flattened_band = band_image.reshape(-1, n_images)

        pca = PCA(n_components=1)
        flattened_pca = pca.fit_transform(flattened_band)

        band_pca = flattened_pca.reshape(height, width)
        pca_bands.append(band_pca)

    pca_bands_np = np.rollaxis(np.array(pca_bands), 0, 3)
    return pca_bands_np
