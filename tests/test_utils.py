import os
import numpy as np

from archeoview.utils import (
    interpolate,
    interpolate_or_filter_bands,
    upscale,
    geotiff_to_numpy,
    minmax_scaling,
    get_image_collection,
)


def test_interpolate():
    test_array = np.random.rand(14, 20)
    assert interpolate(test_array, (28, 40)).shape == (
        28,
        40,
    ), "Should be able to do simple scaling"

    assert interpolate(test_array, (28, 28)).shape == (
        28,
        28,
    ), "Should be able to scale differently for different dimensions"

    test_array2 = np.random.rand(25, 31)
    assert interpolate(test_array2, (28, 35)).shape == (
        28,
        35,
    ), "Should be able to scale very weird shapes"


def test_interpolate_or_filter_bands():
    _, empty_bands = interpolate_or_filter_bands([], [])
    assert empty_bands.shape == (0,), "If given no bands should return no bands"

    test_bands = [np.random.rand(14, 20)] * 3
    test_names = [""] * 3
    _, basic_test = interpolate_or_filter_bands(test_names, test_bands)
    assert basic_test.shape == (
        3,
        14,
        20,
    ), "Should return np.ndarray with correct shape"

    test_bands2 = [np.random.rand(14, 20)] * 3 + [np.random.rand(12, 15)] * 2
    test_names2 = [""] * 5
    _, filtered_test2 = interpolate_or_filter_bands(
        test_names2, test_bands2, interpolation=False
    )
    assert filtered_test2.shape == (
        3,
        14,
        20,
    ), "Should return all high-res bands and skip lower res bands"

    _, interpolated_test2 = interpolate_or_filter_bands(
        test_names2, test_bands2, interpolation=True
    )
    assert interpolated_test2.shape == (
        5,
        14,
        20,
    ), "Should return all bands interpolated correctly to a certain dimension"


def test_upscale():
    _, image = geotiff_to_numpy("data/20180807-kortgene/")
    target_shape = image.shape[0] * 2, image.shape[1] * 2
    upscaled_image = upscale(image, target_shape)
    assert (
        upscaled_image.shape[:2] == target_shape
    ), "Should have converted to target shape"

    target_shape2 = image.shape[0] * 8, image.shape[1] * 8
    upscaled_image2 = upscale(image, target_shape2)
    assert (
        upscaled_image2.shape[:2] == target_shape2
    ), "Should have converted to target shape"


def test_geotiff_to_numpy():
    bands_names, image = geotiff_to_numpy("data/20180807-kortgene/")

    assert len(image.shape) == 3, "Image should have shape (height, width, bands)"
    assert (
        len(bands_names) == image.shape[2]
    ), "Names should correspond to number of bands"

    bands_names2, image2 = geotiff_to_numpy("data/20180926-kortgene-multires/")
    assert image2.shape == (
        174,
        252,
        6,
    ), "With interpolation to False should have only 6 bands"
    assert (
        len(bands_names2) == image2.shape[2]
    ), "Names should correspond to number of bands"

    bands_names3, image3 = geotiff_to_numpy(
        "data/20180926-kortgene-multires/", interpolation=True
    )
    assert image3.shape == (
        174,
        252,
        12,
    ), "With interpolation should have all bands upscaled"
    assert (
        len(bands_names3) == image3.shape[2]
    ), "Names should correspond to number of bands"

    bands_names4, image4 = geotiff_to_numpy("data/20180626-kortgene-highres/")
    assert image4.shape == (
        2201,
        3163,
        3,
    ), "Correctly reading high-res picture, with only RGB bands"
    assert bands_names4 == [
        "R",
        "G",
        "B",
    ], "Band names are not given, so programmatically set to RGB"


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

    per_band_scaled_image = minmax_scaling(image, per_band_scaling=True)
    assert (
        (per_band_scaled_image >= 0) & (per_band_scaled_image <= 1)
    ).all(), "All values should be in range [0, 1]"
    assert (
        image.shape == per_band_scaled_image.shape
    ), "Output shape should be the same as input"


def test_get_image_collection():
    base_path = "data/"
    same_res_paths = [
        os.path.join(base_path, path)
        for path in os.listdir(base_path)
        if path.endswith("kortgene")
    ]
    different_res_paths = [
        os.path.join(base_path, path)
        for path in os.listdir(base_path)
        if path.startswith("201808")
    ]

    collated_collection = get_image_collection(same_res_paths, collate=True)
    collated_collection2 = get_image_collection(different_res_paths, collate=True)
    _, test_image = geotiff_to_numpy(same_res_paths[0])

    assert (
        type(collated_collection) == np.ndarray
    ), "Collated images should be of type np.ndarray"
    assert (
        collated_collection[0] == test_image
    ).all(), "Should correspond to order of given paths"
    assert (
        len(same_res_paths) == collated_collection.shape[0]
    ), "Same res images should all be put into array even with collate == True"
    assert (
        len(different_res_paths) != collated_collection2.shape[0]
    ), "Different res images should not all be put into array with collate == True"

    uncollated_collection = get_image_collection(different_res_paths, collate=False)

    assert type(uncollated_collection) == list, "Uncollated images should be list"
    assert len(uncollated_collection) == len(
        different_res_paths
    ), "Different res images should all be in list with collate == False"
    assert (
        uncollated_collection[0] == test_image
    ).all(), "Should correspond to order of given paths"
