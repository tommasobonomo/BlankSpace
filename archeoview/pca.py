import os
import numpy as np
import rasterio as rio

from typing import List, Tuple


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
    return name_bands, np.array(value_bands)
