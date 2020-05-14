import os
import numpy as np
import rasterio as rio

from typing import List, Tuple
from sklearn.decomposition import PCA


def geotiff_to_numpy(image_path: str) -> Tuple[List[str], np.ndarray]:
    """Extracts values of GeoTiff file in numpy array

    Arguments:
        image_path -- A path to a directory that contains one `.tiff` file for every band of the image

    Returns:
        A tuple of the names of the bands and the image matrix with shape (height, width, bands)
    """

    name_bands: List[str] = []
    value_bands: List[np.ndarray] = []

    for filename in os.listdir(image_path):
        if filename.endswith(".tif"):
            # Assumes that band file is in format name.bandname.tif
            name_bands.append(filename.split(".")[1])
            with rio.open(os.path.join(image_path, filename)) as tiff_file:
                # Assumes that tiff_file has only one band
                value_bands.append(tiff_file.read(1))

    # TODO: #1 add interpolation of bands with higher resolution

    # We also assume that the bands have the same resolution
    # We roll around the axes so that bands are last
    image_matrix = np.rollaxis(np.array(value_bands), 0, 3)
    return name_bands, image_matrix


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
