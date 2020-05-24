import os
import rasterio as rio
import numpy as np

from scipy.interpolate import RectBivariateSpline

from typing import List, Tuple


def interpolate(value_band: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """A function that interpolates a band to a target shape

    Arguments:
        value_band -- A matrix of values, usually of an image
        target_shape -- A tuple like `(target_height, target_width)`

    Returns:
        An array of shape `target_shape`
    """
    x_target, y_target = target_shape
    x_shape, y_shape = value_band.shape

    x_coord_range = np.linspace(0, x_target, num=x_shape, dtype="int")
    y_coord_range = np.linspace(0, y_target, num=y_shape, dtype="int")

    # Check for scaling factors smaller than 3
    scaling_factor = max(min(x_target // x_shape, y_target // y_shape, 3), 1)

    rect_bivariate_spline = RectBivariateSpline(
        x_coord_range, y_coord_range, value_band, kx=scaling_factor, ky=scaling_factor
    )

    return rect_bivariate_spline(range(x_target), range(y_target), grid=True)


def interpolate_or_filter_bands(
    name_bands: List[str], value_bands: List[np.ndarray], interpolation: bool = False
) -> Tuple[List[str], np.ndarray]:
    """Returns the input bands all in the same shape, either by filtering or interpolating

    Arguments:
        value_bands -- List of arrays, where each array represents a band and could have different shape than the others

    Keyword Arguments:
        interpolation -- A boolean that if `True` indicates interpolation should be performed to bring lower resolution bands to the max resolution found, if `False` lower resolution bands should be skipped (default: {False})

    Returns:
        A matrix of shape `(n_bands, height, width)` with all bands of same shape
    """

    # Check if interpolation is needed
    all_shapes = frozenset([band.shape for band in value_bands])

    if len(all_shapes) <= 1:
        # We don't need interpolation or filtering, all bands have same resolution
        return name_bands, np.array(value_bands)
    else:
        # There is more than one resolution
        max_shape = max(all_shapes)
        if not interpolation:
            # Should skip lower resolution bands
            high_res_name_bands = [
                name
                for name, band in zip(name_bands, value_bands)
                if band.shape == max_shape
            ]
            high_res_value_bands = [
                band for band in value_bands if band.shape == max_shape
            ]

            return high_res_name_bands, np.array(high_res_value_bands)
        else:
            # Should interpolate low resoultion to high
            interpolated_value_bands = [
                interpolate(band, max_shape) for band in value_bands
            ]
            return name_bands, np.array(interpolated_value_bands)


def upscale(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Returns an image upscaled through interpolation to a target shape

    Arguments:
        image -- An array of shape `(height, width, n_bands)`
        target_shape -- A tuple of `(target_height, target_width)`

    Returns:
        The interpolated image of shape `(target_height, target_width, n_bands)`
    """
    upscaled_bands: List[np.ndarray] = []
    for i in range(image.shape[2]):
        upscaled_bands.append(interpolate(image[:, :, i], target_shape))

    upscaled_image = np.rollaxis(np.array(upscaled_bands), 0, 3)
    return minmax_scaling(upscaled_image)


def geotiff_to_numpy(
    image_path: str, interpolation: bool = False
) -> Tuple[List[str], np.ndarray]:
    """Extracts values of GeoTiff file in numpy array

    Arguments:
        image_path -- A path to a directory that contains one `.tiff` file for every band of the image
        interpolation -- A boolean that indicates if bands of a lower resolution than the maximum should be interpolated or skipped (default: {False})

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

    # We also assume that the bands have the same resolution
    name_bands, image_matrix = interpolate_or_filter_bands(
        name_bands, value_bands, interpolation=interpolation
    )
    # We roll around the axes so that bands are last
    image_matrix = np.rollaxis(image_matrix, 0, 3)
    return name_bands, image_matrix


def minmax_scaling(image: np.ndarray, bands_first: bool = False) -> np.ndarray:
    """Performs min-max scaling of an input image, resulting with values in the range [0, 1]

    Arguments:
        image -- The input image, a np.ndarray with shape (height, width, bands), unless
        `bands_first = True`

    Keyword Arguments:
        bands_first -- If True, input image has shape (bands, height, width) (default: {False})

    Returns:
        The input image with the same format scaled in the range [0, 1]
    """
    if bands_first:
        image = np.rollaxis(image, 0, 3)

    _, _, n_bands = image.shape

    min_band = image.min()
    max_band = image.max()
    if max_band == min_band:
        image = 0
    else:
        image = (image - min_band) / (max_band - min_band)

    if bands_first:
        image = np.rollaxis(image, 2, 0)

    return image
