import numpy as np

def mean(
    chunk: np.ndarray,
) -> float:
    """return the mean of the pixel values in the chunk"""
    return np.mean(chunk)

def maxpooling(
    chunk: np.ndarray,
) -> float:
    """return max value of the pixel values in the chunk"""
    return np.max(chunk)

def minpooling(
    chunk: np.ndarray,
) -> float:
    """return min value of the pixel values in the chunk"""
    return np.min(chunk)