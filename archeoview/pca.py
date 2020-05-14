import numpy as np

from sklearn.decomposition import PCA


def pca_decomposition(
    image: np.ndarray, n_dimensions: int = 3, bands_first: bool = False
) -> np.ndarray:
    """Applies Principal Component Analysis on the image and returns the number of components required

    Arguments:
        image -- np.ndarray with 3 axes: (height, width, bands).

    Keyword Arguments:
        n_dimensions -- Number of dimensions to return from PCA (default: {3})
        bands_first -- If true, axes of image are in this order: (bands, height, width) (default: {False})

    Returns:
        The output image with shape (height, width, n_dimensions)
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

    return pca_image
