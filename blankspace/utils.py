import ee
import os
import rasterio as rio
import numpy as np

from datetime import date
from typing import List, Tuple, Union

from scipy.interpolate import RectBivariateSpline

Point = Tuple[int, int]
BoundingBox = Tuple[Point, Point]


def earth_engine_to_google_drive(
    point: Point,
    bounding_box: BoundingBox,
    start_date: str,
    end_date: str,
    dataset: str,
    bands: List[str],
    scale: int = 10,
    n_images: Union[int, None] = None,
    task_name: str = "ee-download",
    drive_folder: str = "EarthEngine",
    autostart: bool = True,
) -> List[ee.batch.Task]:
    """Generates Google Earth Engine tasks to upload images to user's Google Drive

    Arguments:
        point -- Two coordinates that indicate a point in our bounding box
        bounding_box -- Four coordinates that specify the bounding box, i.e. [[minX, minY], [maxX, maxY]]
        start_date -- Start date for which we want specific images in format YYYY-MM-DD
        end_date -- End date in same format. WARNING: a too big date interval might download too many images
        dataset -- The dataset to use for the retrieval, defined in the Earth Engine documentation (https://developers.google.com/earth-engine/datasets)
        bands -- Bands of the dataset to download, should correspond to Earth Engine documentation (https://developers.google.com/earth-engine/datasets)

    Keyword Arguments:
        scale -- Scale to download the images, should be univocal for all bands and must be coherent with what specified in Earth Engine documentation (default: {10})
        n_images -- Number of images to retrieve in between start_date and end_date, uniformly spaced. If None, will retrieve all images in that time period (default: {None})
        task_name -- The task name for the Google Earth Engine task (default: {"earth_engine_download"})
        drive_folder -- The folder in Drive where to put the images (default: {"EarthEngine"})
        autostart -- Specify if tasks should be started as soon as the function is called (default: {True})

    Returns:
        A list of Task objects to monitor the task progress
    """

    assert date.fromisoformat(start_date) < date.fromisoformat(
        end_date
    ), "Start date must come before end date"

    ee.Initialize()

    # Define location and bounding_box
    location = ee.Geometry.Point(coords=point)
    bbox = ee.Geometry.Rectangle(coords=bounding_box)

    # Construct relevant collection
    image_collection = (
        ee.ImageCollection(dataset)
        .filterMetadata("resolution_meters", "equals", 10)
        .filterBounds(location)
        .filterDate(start_date, end_date)
    )

    # Start batch download of all features
    list_collection = image_collection.toList(1000)
    n = list_collection.size().getInfo()

    if n_images is None:
        retrieval_idx = list(range(n))
    elif n_images > n or n_images <= 0:
        retrieval_idx = list(range(n))
    else:
        diff = n // n_images
        retrieval_idx = list(filter(lambda x: x % diff, range(n)))

    tasks = []
    for i in retrieval_idx:
        # Get image
        image = ee.Image(list_collection.get(i)).clip(bbox).select(*bands)
        # Get date from image name
        day = image.id().getInfo().split("_")[4].split("T")[0]

        task = ee.batch.Export.image.toDrive(
            image, description=f"{day}_{task_name}", folder=drive_folder, scale=scale,
        )
        tasks.append(task)
        if autostart:
            task.start()

    return tasks


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


def geotiff_file_to_numpy(
    image_path: str, interpolation: bool = False
) -> Tuple[List[str], np.ndarray]:
    with rio.open(image_path) as tiff_file:
        image_matrix = tiff_file.read()
        n_bands = tiff_file.count
    if n_bands == 3:
        name_bands = ["R", "G", "B"]
    elif n_bands == 2:
        name_bands = ["VV", "VH"]

    return name_bands, image_matrix


def geotiff_to_numpy(
    image_path: str, interpolation: bool = False
) -> Tuple[List[str], np.ndarray]:
    """Extracts values of GeoTiff file in numpy array

    Arguments:
        image_path -- Can be a path to: a directory with a `.tiff` file for each band of an image; a directory with a single `.tiff` file that contains all bands of an image; a `.tiff` file that contains all bands of an image

    Keyword Arguments:
        interpolation -- A boolean that indicates if bands of a lower resolution than the maximum should be interpolated or skipped (default: {False})

    Returns:
        A tuple of the names of the bands and the image matrix with shape (height, width, bands)
    """

    if os.path.isdir(image_path):
        if len(os.listdir(image_path)) > 1:
            # We assume that the image is made of different files, one for each band
            name_bands: List[str] = []
            value_bands: List[np.ndarray] = []
            for filename in os.listdir(image_path):
                if filename.endswith(".tif"):
                    # Assumes that band file is in format name.bandname.tif
                    name_bands.append(filename.split(".")[1])
                    with rio.open(os.path.join(image_path, filename)) as tiff_file:
                        value_bands.append(tiff_file.read(1))
            name_bands, image_matrix = interpolate_or_filter_bands(
                name_bands, value_bands, interpolation=interpolation
            )
        else:
            # We assume the only file in the folder contains all the bands
            filename = os.listdir(image_path)[0]
            with rio.open(os.path.join(image_path, filename)) as tiff_file:
                image_matrix = tiff_file.read()
            name_bands = ["R", "G", "B"]
    else:
        # If image path is just a .tiff file
        name_bands, image_matrix = geotiff_file_to_numpy(
            image_path, interpolation=interpolation
        )

    # We roll around the axes so that bands are last
    image_matrix = np.rollaxis(image_matrix, 0, 3)
    return name_bands, image_matrix


def minmax_scaling(
    image: np.ndarray, per_band_scaling: bool = False, bands_first: bool = False
) -> np.ndarray:
    """Performs min-max scaling of an input image, resulting with values in the range [0, 1]

    Arguments:
        image -- The input image, a np.ndarray with shape (height, width, bands), unless
        `bands_first = True`

    Keyword Arguments:
        per_band_scaling -- If True, will perform min-max scaling singularly for each band. Otherwise, will perform min-max scaling over the complete image (default: {False})
        bands_first -- If True, input image has shape (bands, height, width) (default: {False})

    Returns:
        The input image with the same format scaled in the range [0, 1]
    """
    out_image = image.copy()

    if bands_first:
        out_image = np.rollaxis(out_image, 0, 3)

    _, _, n_bands = out_image.shape

    if per_band_scaling:
        # Perform scaling for each singular band
        for band_idx in range(n_bands):
            band_image = out_image[:, :, band_idx]
            min_band = band_image.min()
            max_band = band_image.max()
            if max_band == min_band:
                out_image[:, :, band_idx] = 0
            else:
                out_image[:, :, band_idx] = (band_image - min_band) / (
                    max_band - min_band
                )
    else:
        # Perform scaling over the complete image
        min_image = out_image.min()
        max_image = out_image.max()
        out_image = (out_image - min_image) / (max_image - min_image)

    if bands_first:
        out_image = np.rollaxis(out_image, 2, 0)

    return out_image


def get_image_collection(
    image_paths: List[str], collate: bool = True, interpolation: bool = False
) -> Union[List[np.ndarray], np.ndarray]:
    """Helper function to read in multiple images in one collection

    Arguments:
        image_paths -- List of paths as strings to images that must be part of the collection

    Keyword Arguments:
        collate -- Wheter to join every image in one numpy array, dropping the lower resolution images. If False, will return a list of numpy arrays (default: {True})
        interpolation -- Whether to perform interpolation on bands with a lower resolution than the highest detected (default: {False})

    Returns:
        A numpy array if collate == True, else a list of arrays. Each array in both structures consists of the numbers that describe the image for the given path
    """
    images: List[np.ndarray] = []
    for image_path in image_paths:
        _, image = geotiff_to_numpy(image_path, interpolation=interpolation)
        images.append(image)

    if collate:
        all_shapes = frozenset([image.shape for image in images])
        if len(all_shapes) <= 1:
            images = np.array(images)
        else:
            max_shape = max(all_shapes)
            images = np.array([image for image in images if image.shape == max_shape])

    return images
