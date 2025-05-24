import pandas as pd
import numpy as np
import dcor
import seaborn as sns
import matplotlib.pyplot as plt
import os
from itertools import combinations
from tslearn.metrics import dtw  # Import tslearn.metrics.dtw for Dynamic Time Warping
from sklearn.feature_selection import mutual_info_regression  # Import for Mutual Information calculation
from ..data.io_utils import save_figure, export_to_csv

def calculate_pairwise_distance_correlation(data_df: pd.DataFrame, time_col: str, metric_col: str, group_col: str, min_observations: int = 10):
    """
    Calculates the pairwise Distance Correlation (dCor) between time series of a specific metric
    for different groups (e.g., tenants).

    Args:
        data_df (pd.DataFrame): DataFrame containing the time series data.
                                Must include time, metric, and group columns.
        time_col (str): Name of the column representing time.
        metric_col (str): Name of the column representing the metric to analyze.
        group_col (str): Name of the column representing the groups to compare (e.g., 'tenant_id').
        min_observations (int): Minimum number of observations required for a series to be included.

    Returns:
        pd.DataFrame: A DataFrame where the index and columns are the group IDs,
                      and values are the dCor between them for the specified metric.
                      Returns an empty DataFrame if not enough groups or data.
    """
    results = {}
    groups = data_df[group_col].unique()
    valid_series = {}

    for group in groups:
        group_data = data_df[data_df[group_col] == group]
        if metric_col not in group_data.columns:
            print(f"Metric '{metric_col}' not found for group '{group}'. Skipping this group.")
            continue
        series = group_data.set_index(time_col)[metric_col].dropna()
        if len(series) >= min_observations:
            valid_series[group] = series.values
        else:
            print(f"Series for group '{group}', metric '{metric_col}' has {len(series)} observations, less than minimum {min_observations}. Skipping.")

    if len(valid_series) < 2:
        print(f"Not enough valid series (found {len(valid_series)}) for metric '{metric_col}' to calculate pairwise distance correlation.")
        return pd.DataFrame()

    group_ids = list(valid_series.keys())
    dcor_matrix = pd.DataFrame(index=group_ids, columns=group_ids, dtype=float)
    for gid in group_ids:
        dcor_matrix.loc[gid, gid] = 1.0

    for group1, group2 in combinations(group_ids, 2):
        series1 = valid_series[group1]
        series2 = valid_series[group2]

        min_len = min(len(series1), len(series2))
        
        if min_len == 0:
            dcor_val = np.nan
        elif min_len < 2: # dcor requires at least 2 observations
            print(f"Skipping dCor between {group1} and {group2} for metric '{metric_col}' due to insufficient data points ({min_len}) after alignment for dcor calculation.")
            dcor_val = np.nan
        else:
            series1_trimmed = series1[:min_len]
            series2_trimmed = series2[:min_len]
            try:
                if np.all(series1_trimmed == series1_trimmed[0]) or np.all(series2_trimmed == series2_trimmed[0]):
                    dcor_val = np.nan
                else:
                    dcor_val = dcor.distance_correlation(series1_trimmed, series2_trimmed)
            except ValueError as ve:
                print(f"ValueError calculating dCor between {group1} and {group2} for metric '{metric_col}': {ve}. Series lengths: {len(series1_trimmed)}, {len(series2_trimmed)}.")
                dcor_val = np.nan
            except Exception as e:
                print(f"Error calculating dCor between {group1} and {group2} for metric '{metric_col}': {e}")
                dcor_val = np.nan
        
        dcor_matrix.loc[group1, group2] = dcor_val
        dcor_matrix.loc[group2, group1] = dcor_val

    return dcor_matrix

def plot_distance_correlation_heatmap(dcor_matrix: pd.DataFrame, title: str, output_dir: str, filename: str, cmap: str = "viridis", fmt: str = ".2f", annot: bool = True, tables_dir: str = None):
    """
    Plots a heatmap of the Distance Correlation matrix.

    Args:
        dcor_matrix (pd.DataFrame): DataFrame containing the dCor values.
        title (str): Title of the plot.
        output_dir (str): Directory to save the plot.
        filename (str): Name of the file to save the plot (e.g., 'dcor_heatmap_metric'). 
                        The .png extension will be added by save_figure.
        cmap (str): Colormap for the heatmap.
        fmt (str): String formatting for annotations.
        annot (bool): Whether to annotate the cells with dCor values.
        tables_dir (str, optional): Directory to save the CSV table. If None, no table is saved.
    """
    if dcor_matrix.empty:
        print(f"Skipping heatmap plot for '{title}' as the dCor matrix is empty.")
        return
    
    try:
        dcor_matrix_numeric = dcor_matrix.astype(float)
    except ValueError as e:
        print(f"Could not convert dCor matrix to numeric for plotting '{title}': {e}. Skipping heatmap.")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(dcor_matrix.columns) + 2), max(8, len(dcor_matrix.index) + 2)))
    sns.heatmap(dcor_matrix_numeric, annot=annot, fmt=fmt, cmap=cmap, vmin=0, vmax=1, cbar_kws={'label': 'Distance Correlation'}, square=True, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(np.arange(len(dcor_matrix.columns)) + 0.5)
    ax.set_yticks(np.arange(len(dcor_matrix.index)) + 0.5)
    ax.set_xticklabels(dcor_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(dcor_matrix.index, rotation=0)
    plt.tight_layout()

    try:
        # save_figure handles path construction and saving
        save_figure(fig, output_dir, filename) # Pass the figure object, output_dir, and base filename
        
        # Save data as CSV if tables_dir is provided
        if tables_dir is not None:
            # Extract base filename without extension
            base_filename = os.path.splitext(filename)[0] if '.' in filename else filename
            csv_filename = os.path.join(tables_dir, f"{base_filename}.csv")
            
            # Create the tables directory if it doesn't exist
            os.makedirs(tables_dir, exist_ok=True)
            
            # Export the matrix to CSV
            export_to_csv(dcor_matrix, csv_filename, float_format='.4f')
            print(f"Distance correlation matrix exported to {csv_filename}")
    except Exception as e:
        print(f"Error saving Distance Correlation heatmap '{title}' using save_figure: {e}")
    finally:
        plt.close(fig) # Close the figure object

def calculate_pairwise_cosine_similarity(data_df: pd.DataFrame, time_col: str, metric_col: str, group_col: str, min_observations: int = 10):
    """
    Calculates the pairwise Cosine Similarity between time series of a specific metric
    for different groups (e.g., tenants).
    
    Cosine similarity measures the cosine of the angle between vectors, providing a
    similarity measure that is independent of magnitude and focuses on orientation.
    
    Args:
        data_df (pd.DataFrame): DataFrame containing the time series data.
                                Must include time, metric, and group columns.
        time_col (str): Name of the column representing time.
        metric_col (str): Name of the column representing the metric to analyze.
        group_col (str): Name of the column representing the groups to compare (e.g., 'tenant').
        min_observations (int): Minimum number of observations required for a series to be included.
        
    Returns:
        pd.DataFrame: A DataFrame where the index and columns are the group IDs,
                      and values are the cosine similarity between them for the specified metric.
                      Returns an empty DataFrame if not enough groups or data.
    """
    results = {}
    groups = data_df[group_col].unique()
    valid_series = {}
    
    # Extract valid series for each group
    for group in groups:
        group_data = data_df[data_df[group_col] == group]
        if metric_col not in group_data.columns:
            print(f"Metric '{metric_col}' not found for group '{group}'. Skipping this group.")
            continue
        series = group_data.set_index(time_col)[metric_col].dropna()
        if len(series) >= min_observations:
            valid_series[group] = series.values
        else:
            print(f"Series for group '{group}', metric '{metric_col}' has {len(series)} observations, less than minimum {min_observations}. Skipping.")
    
    if len(valid_series) < 2:
        print(f"Not enough valid series (found {len(valid_series)}) for metric '{metric_col}' to calculate pairwise cosine similarity.")
        return pd.DataFrame()
    
    # Create similarity matrix
    group_ids = list(valid_series.keys())
    cosine_matrix = pd.DataFrame(index=group_ids, columns=group_ids, dtype=float)
    
    # Diagonal is always 1 (perfect similarity with itself)
    for gid in group_ids:
        cosine_matrix.loc[gid, gid] = 1.0
    
    # Calculate pairwise similarities
    for group1, group2 in combinations(group_ids, 2):
        series1_raw = valid_series[group1]
        series2_raw = valid_series[group2]
        
        # Find minimum length between the two series
        min_len = min(len(series1_raw), len(series2_raw))
        
        if min_len == 0:
            cosine_val = np.nan
        elif min_len < 2:  # Need at least 2 observations
            print(f"Skipping cosine similarity between {group1} and {group2} for metric '{metric_col}' due to insufficient data points ({min_len}) after alignment.")
            cosine_val = np.nan
        else:
            # Trim series to the same length
            s1 = series1_raw[:min_len]
            s2 = series2_raw[:min_len]

            # Mean-center the series
            s1_centered = s1 - np.mean(s1)
            s2_centered = s2 - np.mean(s2)

            # Calculate cosine similarity
            # Check for zero vectors after centering (e.g., constant series)
            norm_s1 = np.linalg.norm(s1_centered)
            norm_s2 = np.linalg.norm(s2_centered)

            if norm_s1 == 0 or norm_s2 == 0: # Happens if a series is constant after trimming
                cosine_val = np.nan # Or 0, depending on desired behavior for constant series
            else:
                cosine_val = np.dot(s1_centered, s2_centered) / (norm_s1 * norm_s2)
        
        cosine_matrix.loc[group1, group2] = cosine_val
        cosine_matrix.loc[group2, group1] = cosine_val
    
    return cosine_matrix

def plot_cosine_similarity_heatmap(cosine_matrix: pd.DataFrame, title: str, output_dir: str, filename: str, cmap: str = "viridis", fmt: str = ".2f", annot: bool = True, tables_dir: str = None):
    """
    Plots a heatmap of the Cosine Similarity matrix and optionally exports the data as CSV.
    
    Args:
        cosine_matrix (pd.DataFrame): DataFrame containing the cosine similarity values.
        title (str): Title of the plot.
        output_dir (str): Directory to save the plot.
        filename (str): Name of the file to save the plot (e.g., 'cosine_sim_heatmap_metric').
                        The .png extension will be added by save_figure.
        cmap (str): Colormap for the heatmap.
        fmt (str): String formatting for annotations.
        annot (bool): Whether to annotate the cells with cosine similarity values.
        tables_dir (str, optional): Directory to save the CSV table. If None, no table is saved.
    """
    if cosine_matrix.empty:
        print(f"Skipping heatmap plot for '{title}' as the cosine similarity matrix is empty.")
        return
    
    try:
        cosine_matrix_numeric = cosine_matrix.astype(float)
        
        fig = plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        # Create heatmap
        sns.heatmap(
            cosine_matrix_numeric,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8, "label": "Cosine Similarity"},
            ax=ax
        )
        
        plt.title(title, fontsize=14)
        plt.tight_layout()
        
        # Save figure
        save_figure(fig, output_dir, filename)
        
        print(f"Cosine similarity heatmap saved to {os.path.join(output_dir, filename)}")
        
        # Save data as CSV if tables_dir is provided
        if tables_dir is not None:
            # Extract base filename without extension
            base_filename = os.path.splitext(filename)[0] if '.' in filename else filename
            csv_filename = os.path.join(tables_dir, f"{base_filename}.csv")
            
            # Create the tables directory if it doesn't exist
            os.makedirs(tables_dir, exist_ok=True)
            
            # Export the matrix to CSV
            export_to_csv(cosine_matrix, csv_filename, float_format='.4f')
            print(f"Cosine similarity matrix exported to {csv_filename}")
        
    except Exception as e:
        print(f"Error plotting cosine similarity heatmap: {e}")
    finally:
        plt.close(fig)  # Close the figure object to prevent memory issues

def calculate_pairwise_dtw_distance(data_df: pd.DataFrame, time_col: str, metric_col: str, group_col: str, min_observations: int = 10, normalize: bool = True):
    """
    Calculates the pairwise Dynamic Time Warping (DTW) distance between time series of a specific metric
    for different groups (e.g., tenants).
    
    DTW finds the optimal alignment between two time series and measures their similarity
    independent of non-linear variations in the time dimension. This allows for comparing patterns
    that are similar but shifted, stretched or compressed in time.
    
    The function first Z-normalizes each time series to focus on pattern shape rather than
    magnitude, making the DTW measure more robust for comparing time series with different scales.
    
    Args:
        data_df (pd.DataFrame): DataFrame containing the time series data.
                                Must include time, metric, and group columns.
        time_col (str): Name of the column representing time.
        metric_col (str): Name of the column representing the metric to analyze.
        group_col (str): Name of the column representing the groups to compare (e.g., 'tenant').
        min_observations (int): Minimum number of observations required for a series to be included.
        normalize (bool): Whether to normalize the DTW distance by the average length of both series.
                         Recommended to compare series of different lengths.
        
    Returns:
        pd.DataFrame: A DataFrame where the index and columns are the group IDs,
                      and values are the DTW distances between them for the specified metric.
                      Returns an empty DataFrame if not enough groups or data.
                      
    Notes:
        - Lower DTW values indicate higher similarity between time series
        - The implementation uses tslearn's DTW function
        - Time series are reshaped to 2D arrays (n_timestamps, 1) for compatibility with tslearn
        - Z-score normalization is applied to focus on pattern shapes rather than magnitudes
    """
    results = {}
    groups = data_df[group_col].unique()
    valid_series = {}
    
    # Extract valid series for each group
    for group in groups:
        group_data = data_df[data_df[group_col] == group]
        if metric_col not in group_data.columns:
            print(f"Metric '{metric_col}' not found for group '{group}'. Skipping this group.")
            continue
        series = group_data.set_index(time_col)[metric_col].dropna()
        if len(series) >= min_observations:
            # Z-score normalization of each series to make DTW compare shapes, not magnitudes
            series_values = series.values
            if np.std(series_values) > 0:  # Avoid division by zero
                valid_series[group] = (series_values - np.mean(series_values)) / np.std(series_values)
            else:
                valid_series[group] = series_values  # If std=0, just use the original values
        else:
            print(f"Series for group '{group}', metric '{metric_col}' has {len(series)} observations, less than minimum {min_observations}. Skipping.")
    
    if len(valid_series) < 2:
        print(f"Not enough valid series (found {len(valid_series)}) for metric '{metric_col}' to calculate pairwise DTW distance.")
        return pd.DataFrame()
    
    # Create distance matrix
    group_ids = list(valid_series.keys())
    dtw_matrix = pd.DataFrame(index=group_ids, columns=group_ids, dtype=float)
    
    # Diagonal is always 0 (zero distance with itself)
    for gid in group_ids:
        dtw_matrix.loc[gid, gid] = 0.0
    
    # Calculate pairwise DTW distances
    for group1, group2 in combinations(group_ids, 2):
        series1 = valid_series[group1]
        series2 = valid_series[group2]
        
        if len(series1) < 2 or len(series2) < 2:  # DTW requires at least 2 observations
            print(f"Skipping DTW between {group1} and {group2} for metric '{metric_col}' due to insufficient data points.")
            dtw_val = np.nan
        else:
            try:
                # Ensure series are in the correct format for tslearn (2D arrays)
                # Each time series must be shaped as (n_timestamps, n_features)
                series1_2d = series1.reshape(-1, 1)
                series2_2d = series2.reshape(-1, 1)
                
                # Use tslearn's dtw implementation
                distance = dtw(series1_2d, series2_2d)
                
                # No path is returned by tslearn, so no normalization
                dtw_val = distance
                
                # Normalize by path length if requested (makes distance comparable across different length series)
                if normalize:
                    # For tslearn, we normalize by dividing by the average length of both sequences
                    avg_len = (len(series1) + len(series2)) / 2.0
                    dtw_val = distance / avg_len
                else:
                    dtw_val = distance
                    
            except Exception as e:
                print(f"Error calculating DTW distance between {group1} and {group2} for metric '{metric_col}': {e}")
                dtw_val = np.nan
        
        # Assign distance value to both positions in the matrix (it's symmetric)
        dtw_matrix.loc[group1, group2] = dtw_val
        dtw_matrix.loc[group2, group1] = dtw_val
    
    return dtw_matrix

def plot_dtw_distance_heatmap(dtw_matrix: pd.DataFrame, title: str, output_dir: str, filename: str, cmap: str = "viridis_r", fmt: str = ".2f", annot: bool = True, tables_dir: str = None):
    """
    Plots a heatmap of the DTW Distance matrix and optionally exports the data as CSV.
    
    Args:
        dtw_matrix (pd.DataFrame): DataFrame containing the DTW distance values.
        title (str): Title of the plot.
        output_dir (str): Directory to save the plot.
        filename (str): Name of the file to save the plot (e.g., 'dtw_distance_heatmap_metric').
                        The .png extension will be added by save_figure.
        cmap (str): Colormap for the heatmap. Default uses reversed viridis (smaller distance = darker color).
        fmt (str): String formatting for annotations.
        annot (bool): Whether to annotate the cells with DTW distance values.
        tables_dir (str, optional): Directory to save the CSV table. If None, no table is saved.
        
    Notes:
        - The diagonal is always 0 (zero distance with itself)
        - For visualization, a reversed colormap (viridis_r) is used by default so that
          lower values (more similar time series) are represented by darker colors
        - The function automatically sets the color scale range from 0 to the maximum non-diagonal value
        - Both the heatmap plot and the raw data (as CSV) are saved if tables_dir is provided
    """
    if dtw_matrix.empty:
        print(f"Skipping heatmap plot for '{title}' as the DTW distance matrix is empty.")
        return
    
    try:
        dtw_matrix_numeric = dtw_matrix.astype(float)
        
        # Create the mask for non-diagonal elements (diagonal is always 0)
        non_diag_mask = ~np.eye(len(dtw_matrix_numeric), dtype=bool)
        non_diag_values = dtw_matrix_numeric.values[non_diag_mask]
        
        # Check if we have any valid values (not NaN)
        if np.all(np.isnan(non_diag_values)):
            print(f"Warning: All non-diagonal values are NaN in DTW matrix for '{title}'. Using default max value.")
            max_val = 1.0  # Default max value if all are NaN
        else:
            # Get the maximum value for vmax, excluding diagonal zeros and NaN values
            max_val = np.nanmax(non_diag_values)
        
        fig, ax = plt.subplots(figsize=(max(10, len(dtw_matrix.columns) + 2), max(8, len(dtw_matrix.index) + 2)))
        
        # Create heatmap - Using viridis_r (reversed) so lower values (more similar) are darker
        sns.heatmap(
            dtw_matrix_numeric,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            vmin=0,
            vmax=max_val,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8, "label": "DTW Distance (lower = more similar)"},
            ax=ax
        )
        
        ax.set_title(title, fontsize=16)
        ax.set_xticks(np.arange(len(dtw_matrix.columns)) + 0.5)
        ax.set_yticks(np.arange(len(dtw_matrix.index)) + 0.5)
        ax.set_xticklabels(dtw_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(dtw_matrix.index, rotation=0)
        plt.tight_layout()
        
        # Save figure
        save_figure(fig, output_dir, filename)
        
        print(f"DTW distance heatmap saved to {os.path.join(output_dir, filename)}")
        
        # Save data as CSV if tables_dir is provided
        if tables_dir is not None:
            # Extract base filename without extension
            base_filename = os.path.splitext(filename)[0] if '.' in filename else filename
            csv_filename = os.path.join(tables_dir, f"{base_filename}.csv")
            
            # Create the tables directory if it doesn't exist
            os.makedirs(tables_dir, exist_ok=True)
            
            # Export the matrix to CSV
            export_to_csv(dtw_matrix, csv_filename, float_format='.4f')
            print(f"DTW distance matrix exported to {csv_filename}")
        
    except Exception as e:
        print(f"Error plotting DTW distance heatmap: {e}")
    finally:
        plt.close(fig)  # Close the figure object

def calculate_pairwise_mutual_information(data_df: pd.DataFrame, time_col: str, metric_col: str, group_col: str, min_observations: int = 10, n_neighbors: int = 3, normalize: bool = True):
    """
    Calculates the pairwise Mutual Information (MI) between time series of a specific metric
    for different groups (e.g., tenants).
    
    Mutual Information measures how much information one variable provides about another.
    It's a non-parametric measure of dependency between variables that can detect both linear
    and non-linear relationships, and is zero if and only if the variables are statistically independent.
    
    Args:
        data_df (pd.DataFrame): DataFrame containing the time series data.
                                Must include time, metric, and group columns.
        time_col (str): Name of the column representing time.
        metric_col (str): Name of the column representing the metric to analyze.
        group_col (str): Name of the column representing the groups to compare (e.g., 'tenant').
        min_observations (int): Minimum number of observations required for a series to be included.
        n_neighbors (int): Number of neighbors to use for MI estimation.
                           Higher values reduce variance but increase bias.
        normalize (bool): Whether to normalize MI values to the range [0,1] by dividing by
                          the geometric mean of the entropies of each variable.
        
    Returns:
        pd.DataFrame: A DataFrame where the index and columns are the group IDs,
                      and values are the MI values between them for the specified metric.
                      Returns an empty DataFrame if not enough groups or data.
                      
    Notes:
        - Implementation uses scikit-learn's mutual_info_regression, which estimates MI 
          using nearest neighbors approach
        - Higher values indicate stronger dependency between variables
    """
    groups = data_df[group_col].unique()
    valid_series = {}
    
    # Extract valid series for each group
    for group in groups:
        group_data = data_df[data_df[group_col] == group]
        if metric_col not in group_data.columns:
            print(f"Metric '{metric_col}' not found for group '{group}'. Skipping this group.")
            continue
        series = group_data.set_index(time_col)[metric_col].dropna()
        if len(series) >= min_observations:
            valid_series[group] = series.values
        else:
            print(f"Series for group '{group}', metric '{metric_col}' has {len(series)} observations, less than minimum {min_observations}. Skipping.")
    
    if len(valid_series) < 2:
        print(f"Not enough valid series (found {len(valid_series)}) for metric '{metric_col}' to calculate pairwise mutual information.")
        return pd.DataFrame()
    
    # Create mutual information matrix
    group_ids = list(valid_series.keys())
    mi_matrix = pd.DataFrame(index=group_ids, columns=group_ids, dtype=float)
    
    # Self-MI is equal to the entropy of the variable
    # We'll set diagonal to 1.0 (perfect information overlap)
    for gid in group_ids:
        mi_matrix.loc[gid, gid] = 1.0
    
    # Calculate pairwise MI
    for group1, group2 in combinations(group_ids, 2):
        series1 = valid_series[group1]
        series2 = valid_series[group2]
        
        # Find minimum length between the two series
        min_len = min(len(series1), len(series2))
        
        if min_len < 2:  # Need at least 2 observations
            print(f"Skipping mutual information between {group1} and {group2} for metric '{metric_col}' due to insufficient data points ({min_len}) after alignment.")
            mi_val = np.nan
        else:
            # Trim series to same length
            series1_trimmed = series1[:min_len]
            series2_trimmed = series2[:min_len]
            
            try:
                # Check if either series is constant (would result in zero MI)
                if np.all(series1_trimmed == series1_trimmed[0]) or np.all(series2_trimmed == series2_trimmed[0]):
                    mi_val = 0.0
                else:
                    # Calculate mutual information using scikit-learn
                    # Reshape series to column vectors
                    X = series1_trimmed.reshape(-1, 1)
                    y = series2_trimmed
                    
                    # Calculate mutual information from X to y
                    mi_forward = mutual_info_regression(X, y, n_neighbors=n_neighbors)[0]
                    
                    # Calculate mutual information from y to X
                    X = series2_trimmed.reshape(-1, 1)
                    y = series1_trimmed
                    mi_backward = mutual_info_regression(X, y, n_neighbors=n_neighbors)[0]
                    
                    # Average the two directions for symmetry
                    mi_val = (mi_forward + mi_backward) / 2.0
                    
                    # Normalize if requested
                    if normalize:
                        # Estimate entropy of each series
                        # For normalization, we use the same n_neighbors parameter
                        entropy1 = mutual_info_regression(X=series1_trimmed.reshape(-1, 1), 
                                                         y=series1_trimmed, 
                                                         n_neighbors=n_neighbors)[0]
                        entropy2 = mutual_info_regression(X=series2_trimmed.reshape(-1, 1), 
                                                         y=series2_trimmed, 
                                                         n_neighbors=n_neighbors)[0]
                        
                        # Normalize by geometric mean of entropies (ranges from 0 to 1)
                        if entropy1 > 0 and entropy2 > 0:
                            mi_val = mi_val / np.sqrt(entropy1 * entropy2)
                        else:
                            mi_val = 0.0  # If either entropy is zero, set MI to zero
            except Exception as e:
                print(f"Error calculating mutual information between {group1} and {group2} for metric '{metric_col}': {e}")
                mi_val = np.nan
        
        # Assign MI value to both positions in the matrix (it's symmetric)
        mi_matrix.loc[group1, group2] = mi_val
        mi_matrix.loc[group2, group1] = mi_val
    
    return mi_matrix

def plot_mutual_information_heatmap(mi_matrix: pd.DataFrame, title: str, output_dir: str, filename: str, cmap: str = "viridis", fmt: str = ".2f", annot: bool = True, tables_dir: str = None):
    """
    Plots a heatmap of the Mutual Information matrix and optionally exports the data as CSV.
    
    Args:
        mi_matrix (pd.DataFrame): DataFrame containing the mutual information values.
        title (str): Title of the plot.
        output_dir (str): Directory to save the plot.
        filename (str): Name of the file to save the plot (e.g., 'mutual_info_heatmap_metric').
                        The .png extension will be added by save_figure.
        cmap (str): Colormap for the heatmap.
        fmt (str): String formatting for annotations.
        annot (bool): Whether to annotate the cells with mutual information values.
        tables_dir (str, optional): Directory to save the CSV table. If None, no table is saved.
        
    Notes:
        - The diagonal is always 1.0 (perfect information overlap with itself)
        - MI values range from 0 (no shared information) to 1 (perfect information overlap) when normalized
        - Both the heatmap plot and the raw data (as CSV) are saved if tables_dir is provided
    """
    if mi_matrix.empty:
        print(f"Skipping heatmap plot for '{title}' as the mutual information matrix is empty.")
        return
    
    try:
        mi_matrix_numeric = mi_matrix.astype(float)
        
        fig = plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        # Create heatmap
        sns.heatmap(
            mi_matrix_numeric,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            vmin=0,
            vmax=1,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8, "label": "Mutual Information"},
            ax=ax
        )
        
        plt.title(title, fontsize=14)
        plt.tight_layout()
        
        # Save figure
        save_figure(fig, output_dir, filename)
        
        print(f"Mutual information heatmap saved to {os.path.join(output_dir, filename)}")
        
        # Save data as CSV if tables_dir is provided
        if tables_dir is not None:
            # Extract base filename without extension
            base_filename = os.path.splitext(filename)[0] if '.' in filename else filename
            csv_filename = os.path.join(tables_dir, f"{base_filename}.csv")
            
            # Create the tables directory if it doesn't exist
            os.makedirs(tables_dir, exist_ok=True)
            
            # Export the matrix to CSV
            export_to_csv(mi_matrix, csv_filename, float_format='.4f')
            print(f"Mutual information matrix exported to {csv_filename}")
        
    except Exception as e:
        print(f"Error plotting mutual information heatmap: {e}")
    finally:
        plt.close(fig)  # Close the figure object to prevent memory issues
