"""
Module for similarity analysis using Dynamic Time Warping (DTW).
"""
import pandas as pd
import numpy as np
from tslearn.metrics import dtw

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
