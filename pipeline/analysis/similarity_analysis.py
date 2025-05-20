"""
Module for similarity analysis using Dynamic Time Warping (DTW).
"""
import pandas as pd
import numpy as np
from tslearn.metrics import dtw
from sklearn.metrics.pairwise import cosine_similarity
from pipeline.config import COSINE_SIM_WINDOW_SIZE_S, COSINE_SIM_WINDOW_STEP_S, SCRAPE_INTERVAL_S

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

def calculate_time_varying_cosine_similarity(series1: pd.Series, series2: pd.Series) -> pd.DataFrame:
    """
    Calculates cosine similarity over sliding windows of two time series,
    using window size and step defined in config.py.

    Args:
        series1: The first time series (pandas Series with DatetimeIndex).
        series2: The second time series (pandas Series with DatetimeIndex).
                                     Assumed to be aligned with series1 (same index).

    Returns:
        A pandas DataFrame with columns 'timestamp' and 'cosine_similarity'.
        Returns an empty DataFrame if inputs are unsuitable or configs are invalid.
    """
    if not isinstance(series1, pd.Series) or not isinstance(series2, pd.Series):
        print("Error: Inputs must be pandas Series.")
        return pd.DataFrame(columns=['timestamp', 'cosine_similarity'])
    if not isinstance(series1.index, pd.DatetimeIndex) or not isinstance(series2.index, pd.DatetimeIndex):
        print("Error: Series must have a DatetimeIndex.")
        return pd.DataFrame(columns=['timestamp', 'cosine_similarity'])

    if SCRAPE_INTERVAL_S <= 0:
        print("Error: SCRAPE_INTERVAL_S must be positive.")
        return pd.DataFrame(columns=['timestamp', 'cosine_similarity'])

    window_size_points = COSINE_SIM_WINDOW_SIZE_S // SCRAPE_INTERVAL_S
    step_points = COSINE_SIM_WINDOW_STEP_S // SCRAPE_INTERVAL_S

    if window_size_points <= 0:
        print(f"Error: Window size in points ({window_size_points}), derived from COSINE_SIM_WINDOW_SIZE_S ({COSINE_SIM_WINDOW_SIZE_S}s) and SCRAPE_INTERVAL_S ({SCRAPE_INTERVAL_S}s), must be positive.")
        return pd.DataFrame(columns=['timestamp', 'cosine_similarity'])
    if step_points <= 0:
        print(f"Error: Step size in points ({step_points}), derived from COSINE_SIM_WINDOW_STEP_S ({COSINE_SIM_WINDOW_STEP_S}s) and SCRAPE_INTERVAL_S ({SCRAPE_INTERVAL_S}s), must be positive.")
        return pd.DataFrame(columns=['timestamp', 'cosine_similarity'])


    # Align series by ensuring they have the same length and index after main.py's alignment
    # For robustness, explicitly align here if they might not be perfectly aligned
    # However, main.py already does an inner join, so they should be.
    # If lengths differ after main.py's alignment, this indicates an issue there.
    if len(series1) != len(series2) or not series1.index.equals(series2.index):
        print("Warning: Series for time-varying cosine similarity are not perfectly aligned or have different lengths. Re-aligning with inner join.")
        aligned_df = pd.concat([series1.rename('s1'), series2.rename('s2')], axis=1, join='inner')
        if aligned_df.empty or len(aligned_df) < window_size_points:
            print("Error: Not enough overlapping data after alignment for time-varying cosine similarity.")
            return pd.DataFrame(columns=['timestamp', 'cosine_similarity'])
        series1 = aligned_df['s1']
        series2 = aligned_df['s2']
        
    if len(series1) < window_size_points:
        print(f"Error: Series length ({len(series1)}) is less than window size in points ({window_size_points}).")
        return pd.DataFrame(columns=['timestamp', 'cosine_similarity'])

    results = []
    for i in range(0, len(series1) - window_size_points + 1, step_points):
        window_s1 = series1.iloc[i : i + window_size_points]
        window_s2 = series2.iloc[i : i + window_size_points]
        
        if window_s1.empty or window_s2.empty or len(window_s1) < 2 or len(window_s2) < 2: # Min length for meaningful similarity
            continue

        # Timestamp for the window (e.g., end of the window)
        # Ensure index exists and is not out of bounds
        if i + window_size_points -1 < len(series1.index):
            timestamp = series1.index[i + window_size_points - 1]
            similarity_score = calculate_cosine_similarity(window_s1, window_s2)
            results.append({'timestamp': timestamp, 'cosine_similarity': similarity_score})
        else:
            # This case should ideally not be reached if loop condition is correct
            print(f"Warning: Index out of bounds at window step {i}. Skipping.")


    if not results:
        print("No results generated from time-varying cosine similarity calculation.")
        return pd.DataFrame(columns=['timestamp', 'cosine_similarity'])
        
    return pd.DataFrame(results)
