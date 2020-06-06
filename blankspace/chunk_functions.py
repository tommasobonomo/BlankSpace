import numpy as np

def mean(
    chunk: np.ndarray,
) -> float:
    """return the mean of the pixel values in the chunk"""
    return np.nanmean(chunk)

def maxpooling(
    chunk: np.ndarray,
) -> float:
    """return max value of the pixel values in the chunk"""
    return np.nanmax(chunk)

def minpooling(
    chunk: np.ndarray,
) -> float:
    """return min value of the pixel values in the chunk"""
    return np.nanmin(chunk)