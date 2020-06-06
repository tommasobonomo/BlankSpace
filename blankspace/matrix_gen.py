import os
import pickle
import numpy as np
import imageio
import matplotlib.pyplot as plt
from typing import List, Tuple, Union

def save_tiff_to_numpy_pkl(
    base_path: str = os.path.join('..', 'data', 'Coastal-InSAR'),
    file_name: str = 'imgs_numpy.pkl'
) -> None:
    """read tif files and convert them in a pickle file\n
    base_path: str -- directory containing tiff files\n
    file_name: str -- name of the generated pickle file"""
    imgs_path = [os.path.join(base_path, img) for img in os.listdir(base_path) if img.split('.')[-1] == 'tif']
    images = np.array([np.array(imageio.imread(path)) for path in imgs_path])
    with open(os.path.join(base_path, file_name), 'wb') as f:
        pickle.dump(images, f)

def load_numpy_pkl(
    base_path: str = os.path.join('..', 'data', 'Coastal-InSAR'),
    file_name: str = 'imgs_numpy.pkl'
) -> np.ndarray:
    """load pickle file in a numpy array\n
    base_path: str -- directory containing pickle file\n
    file_name: str -- name of the pickle file"""
    with open(os.path.join(base_path, file_name), 'rb') as f:
        numpy_array = pickle.load(f)
    return numpy_array

def mean(
    chunk: np.ndarray,
) -> float:
    """return the mean of the pixel values in the chunk"""
    return np.mean(chunk)

# add padding to images?
def generate_single_grid(
    original_img: np.ndarray,
    channel: int,
    function, # typing definition for a function(?)
    n_row: int = 20,
    n_col: int = 20
) -> np.ndarray:
    """generate numpy grid with shape (n_row, n_col). 
    values in the grid are defined by the function applied to the pixels present in a cell of the original image"""
    
    # check channel compatibility
    channel = 0 if channel >= original_img.shape[-1] else channel

    # parameters
    height, width = original_img.shape[0], original_img.shape[1]
    
    row_window_size, col_window_size = height // n_row, width // n_col
    row_window_size = 1 if row_window_size == 0 else row_window_size
    col_window_size = 1 if col_window_size == 0 else col_window_size
    
    matrix = np.empty((n_row, n_col))

    # crop image and compute chunks
    for x, row in enumerate(range(0, height - row_window_size + 1, row_window_size)):
        for y, col in enumerate(range(0, width - col_window_size + 1, col_window_size)):
            chunk = original_img[row : row + row_window_size,
                                 col : col + col_window_size,
                                 channel]
            matrix[x,y] = function(chunk)

    return matrix

x = np.arange(16).reshape((4,4,1))
print(f'original\n{x}')
m = generate_single_grid(x, 0, mean, n_col=2, n_row=2)
print(m)