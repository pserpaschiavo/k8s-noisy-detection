"""
Module for similarity analysis using Dynamic Time Warping (DTW).
"""
import pandas as pd
import numpy as np
from tslearn.metrics import dtw
from sklearn.metrics.pairwise import cosine_similarity

def calculate_dtw_distance(series1: pd.Series | np.ndarray, series2: pd.Series | np.ndarray) -> float:
    """
    Calculates the Dynamic Time Warping (DTW) distance between two time series.

    Args:
        series1: The first time series (pandas Series or NumPy array).
        series2: The second time series (pandas Series or NumPy array).

    Returns:
        The DTW distance between the two series.
    """
    # Ensure inputs are numpy arrays
    if isinstance(series1, pd.Series):
        s1 = series1.to_numpy()
    else:
        s1 = series1
    
    if isinstance(series2, pd.Series):
        s2 = series2.to_numpy()
    else:
        s2 = series2

    # Reshape arrays if they are 1D, as tslearn expects (n_timestamps, n_features)
    if s1.ndim == 1:
        s1 = s1.reshape(-1, 1)
    if s2.ndim == 1:
        s2 = s2.reshape(-1, 1)
        
    distance = dtw(s1, s2)
    return distance

def calculate_cosine_similarity(series1: pd.Series | np.ndarray, series2: pd.Series | np.ndarray) -> float:
    """
    Calculates the Cosine Similarity between two time series.

    Args:
        series1: The first time series (pandas Series or NumPy array).
        series2: The second time series (pandas Series or NumPy array).

    Returns:
        The cosine similarity between the two series (a float between -1 and 1).
    """
    # Ensure inputs are numpy arrays
    if isinstance(series1, pd.Series):
        s1 = series1.to_numpy()
    else:
        s1 = np.asarray(series1) # Ensure it's a numpy array
    
    if isinstance(series2, pd.Series):
        s2 = series2.to_numpy()
    else:
        s2 = np.asarray(series2) # Ensure it's a numpy array

    # Flatten arrays to ensure they are 1D for cosine similarity of vectors
    s1 = s1.flatten()
    s2 = s2.flatten()

    # Cosine similarity expects 2D arrays (n_samples, n_features).
    # Reshape 1D arrays to (1, n_features) to represent single samples.
    s1_reshaped = s1.reshape(1, -1)
    s2_reshaped = s2.reshape(1, -1)
    
    # Calculate cosine similarity
    # The result is a 2D array (e.g., [[similarity]]), so extract the value.
    similarity = cosine_similarity(s1_reshaped, s2_reshaped)[0, 0]
    
    return similarity
