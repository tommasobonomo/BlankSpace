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

def 

array = load_numpy_pkl()
plt.imshow(array[0][:,:,0])
plt.show()