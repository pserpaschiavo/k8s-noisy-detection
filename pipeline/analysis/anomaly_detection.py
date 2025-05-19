"""
Module for advanced anomaly detection in data from the noisy neighbors experiment.

This module implements machine learning algorithms for anomaly detection,
change points, and abnormal behaviors in the collected metrics.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
import ruptures as rpt
from tslearn.clustering import TimeSeriesKMeans
import warnings
warnings.filterwarnings('ignore')


def detect_anomalies_isolation_forest(df, metric_column='value', contamination=0.05, group_by=None):
    """
    Detects anomalies using the Isolation Forest algorithm.
    
    Args:
        df (DataFrame): DataFrame with metric data.
        metric_column (str): Column with metric values.
        contamination (float): Expected proportion of anomalies in the data.
        group_by (list): Columns to group data by (e.g., ['tenant', 'phase']).
        
    Returns:
        DataFrame: Original DataFrame with additional columns 'is_anomaly_if' and 'anomaly_score_if'.
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Add column for anomalies
    result['is_anomaly_if'] = False
    result['anomaly_score_if'] = 0.0
    
    # If no grouping, treat the entire dataset
    if group_by is None:
        # Create and train the model
        X = result[metric_column].values.reshape(-1, 1)
        
        # Skip if there are missing values
        if np.isnan(X).any():
            X = result[metric_column].dropna().values.reshape(-1, 1)
            if len(X) == 0:
                return result
        
        model = IsolationForest(contamination=contamination, random_state=42)
        result['anomaly_score_if'] = model.fit_predict(X)
        result['is_anomaly_if'] = result['anomaly_score_if'] == -1
        
        # Convert score to positive values (higher = more anomalous)
        model_decision = model.decision_function(X) * -1
        result.loc[result.index[~np.isnan(result[metric_column])], 'anomaly_score_if'] = model_decision
    else:
        # For each group, train a separate model
        for group_name, group in result.groupby(group_by):
            # Skip if the group is too small
            if len(group) < 10:
                continue
            
            # Prepare the data
            X = group[metric_column].values.reshape(-1, 1)
            
            # Skip if there are missing values
            if np.isnan(X).any():
                continue
            
            # Create and train the model
            model = IsolationForest(contamination=contamination, random_state=42)
            predictions = model.fit_predict(X)
            
            # Update the result DataFrame
            index_in_group = group.index
            result.loc[index_in_group, 'is_anomaly_if'] = (predictions == -1)
            
            # Convert score to positive values (higher = more anomalous)
            model_decision = model.decision_function(X) * -1
            result.loc[index_in_group, 'anomaly_score_if'] = model_decision
    
    return result


def detect_anomalies_local_outlier_factor(df, metric_column='value', n_neighbors=20, contamination=0.05, group_by=None):
    """
    Detects anomalies using the Local Outlier Factor (LOF) algorithm.
    
    Args:
        df (DataFrame): DataFrame with metric data.
        metric_column (str): Column with metric values.
        n_neighbors (int): Number of neighbors for the LOF algorithm.
        contamination (float): Expected proportion of anomalies in the data.
        group_by (list): Columns to group data by (e.g., ['tenant', 'phase']).
        
    Returns:
        DataFrame: Original DataFrame with additional columns 'is_anomaly_lof' and 'anomaly_score_lof'.
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Add column for anomalies
    result['is_anomaly_lof'] = False
    result['anomaly_score_lof'] = 0.0
    
    # If no grouping, treat the entire dataset
    if group_by is None:
        # Create and train the model
        X = result[metric_column].values.reshape(-1, 1)
        
        # Skip if there are missing values
        if np.isnan(X).any():
            X = result[metric_column].dropna().values.reshape(-1, 1)
            if len(X) == 0:
                return result
        
        # Adjust n_neighbors if the dataset is small
        n_neighbors = min(n_neighbors, len(X) - 1)
        if n_neighbors < 1:
            return result
        
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        result['is_anomaly_lof'] = (model.fit_predict(X) == -1)
        
        # Get anomaly scores (negative of outreach distance)
        result.loc[result.index[~np.isnan(result[metric_column])], 'anomaly_score_lof'] = -model.negative_outlier_factor_
    else:
        # For each group, train a separate model
        for group_name, group in result.groupby(group_by):
            # Skip if the group is too small
            if len(group) <= n_neighbors:
                continue
            
            # Prepare the data
            X = group[metric_column].values.reshape(-1, 1)
            
            # Skip if there are missing values
            if np.isnan(X).any():
                continue
            
            # Adjust n_neighbors if the dataset is small
            local_n_neighbors = min(n_neighbors, len(X) - 1)
            
            # Create and train the model
            model = LocalOutlierFactor(n_neighbors=local_n_neighbors, contamination=contamination)
            predictions = model.fit_predict(X)
            
            # Update the result DataFrame
            index_in_group = group.index
            result.loc[index_in_group, 'is_anomaly_lof'] = (predictions == -1)
            result.loc[index_in_group, 'anomaly_score_lof'] = -model.negative_outlier_factor_
    
    return result


def detect_change_points(df, metric_column='value', time_column='elapsed_minutes', 
                        method='pelt', model='l2', min_size=5, penalty=3, group_by=None):
    """
    Detects change points in the time series using the specified algorithm.
    
    Args:
        df (DataFrame): DataFrame with metric data.
        metric_column (str): Column with metric values.
        time_column (str): Column with time values.
        method (str): Detection method ('pelt', 'binseg', 'window').
        model (str): Cost model ('l1', 'l2', 'rbf', etc.).
        min_size (int): Minimum segment size.
        penalty (float): Penalty for the PELT algorithm.
        group_by (list): Columns to group data by (e.g., ['tenant']).
        
    Returns:
        DataFrame: Original DataFrame with an additional column 'is_change_point'.
        dict: Information about detected change points.
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Add column for change points
    result['is_change_point'] = False
    
    changes_info = {}
    
    # If no grouping, treat the entire dataset
    if group_by is None:
        # Sort by time
        sorted_df = result.sort_values(time_column)
        
        # Prepare the data
        signal = sorted_df[metric_column].values
        
        # Skip if the signal is too small
        if len(signal) < min_size * 2:
            return result, changes_info
        
        # Configure the algorithm
        if method == 'pelt':
            algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
            change_points = algo.predict(pen=penalty)
        elif method == 'binseg':
            algo = rpt.Binseg(model=model, min_size=min_size).fit(signal)
            change_points = algo.predict(n_bkps=5)  # Detect up to 5 points
        elif method == 'window':
            algo = rpt.Window(model=model, width=40).fit(signal)
            change_points = algo.predict(n_bkps=5)  # Detect up to 5 points
        else:
            # Default method
            algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
            change_points = algo.predict(pen=penalty)
        
        # Filter change_points to ensure they are within iloc bounds
        valid_change_points_indices = [cp for cp in change_points if cp < len(sorted_df)]

        # Mark change points in the DataFrame
        for cp_idx in valid_change_points_indices:
            # cp_idx is already a valid index for iloc
            idx = sorted_df.iloc[cp_idx].name
            result.loc[idx, 'is_change_point'] = True
        
        # Record information
        # Use valid indices to fetch times
        change_point_times_list = []
        if valid_change_points_indices: # Only if there are valid change points
            change_point_times_list = sorted_df.iloc[valid_change_points_indices][time_column].tolist()

        changes_info['all'] = {
            'n_change_points': len(valid_change_points_indices), # Count only valid ones
            'change_point_indices': valid_change_points_indices, # Store valid ones
            'change_point_times': change_point_times_list
        }
    else:
        # For each group, detect change points separately
        for group_name, group in result.groupby(group_by):
            # Sort by time
            sorted_group = group.sort_values(time_column)
            
            # Skip if the group is too small
            if len(sorted_group) < min_size * 2:
                continue
            
            # Prepare the data
            signal = sorted_group[metric_column].values
            
            # Configure the algorithm
            if method == 'pelt':
                algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
                change_points = algo.predict(pen=penalty)
            elif method == 'binseg':
                algo = rpt.Binseg(model=model, min_size=min_size).fit(signal)
                change_points = algo.predict(n_bkps=5)  # Detect up to 5 points
            elif method == 'window':
                algo = rpt.Window(model=model, width=40).fit(signal)
                change_points = algo.predict(n_bkps=5)  # Detect up to 5 points
            else:
                # Default method
                algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
                change_points = algo.predict(pen=penalty)

            # Filter change_points to ensure they are within iloc bounds
            valid_change_points_indices_group = [cp for cp in change_points if cp < len(sorted_group)]
            
            # Mark change points in the DataFrame
            for cp_idx in valid_change_points_indices_group:
                idx = sorted_group.iloc[cp_idx].name
                result.loc[idx, 'is_change_point'] = True
            
            # Record information
            change_point_times_list_group = []
            if valid_change_points_indices_group:
                change_point_times_list_group = sorted_group.iloc[valid_change_points_indices_group][time_column].tolist()

            changes_info[group_name] = {
                'n_change_points': len(valid_change_points_indices_group),
                'change_point_indices': valid_change_points_indices_group,
                'change_point_times': change_point_times_list_group
            }
    
    return result, changes_info


def detect_pattern_changes(df, metrics, time_column='elapsed_minutes', 
                         window_size=10, n_clusters=3, group_by=None):
    """
    Detects pattern changes using time series clustering.
    
    Args:
        df (DataFrame): DataFrame with data.
        metrics (list): List of metric names to include in the analysis.
        time_column (str): Column with time values.
        window_size (int): Size of the sliding window for pattern extraction.
        n_clusters (int): Number of clusters to group patterns.
        group_by (list): Columns to group data by (e.g., ['tenant']).
        
    Returns:
        DataFrame: DataFrame with information about detected patterns.
    """
    # Check if we have enough metrics
    if len(metrics) == 0:
        return pd.DataFrame()
    
    results = []
    
    # If no grouping, treat the entire dataset
    if group_by is None:
        # Sort by time
        sorted_df = df.sort_values(time_column)
        
        # Feature extraction
        X = sorted_df[metrics].values
        
        # Skip if the signal is too small
        if len(X) < window_size * 2:
            return pd.DataFrame()
        
        # Normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Extract subsequences using a sliding window
        subsequences = []
        timestamps = []
        
        for i in range(len(X_scaled) - window_size + 1):
            subseq = X_scaled[i:i+window_size].flatten()
            subsequences.append(subseq)
            timestamps.append(sorted_df.iloc[i+window_size-1][time_column])
        
        # Group subsequences
        if len(subsequences) > n_clusters:
            kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
            clusters = kmeans.fit_predict(np.array(subsequences))
            
            # Detect cluster changes
            cluster_changes = np.diff(clusters, prepend=clusters[0])
            change_indices = np.where(cluster_changes != 0)[0]
            
            # Record information about pattern changes
            for i in change_indices:
                if i > 0 and i < len(timestamps):
                    results.append({
                        'time': timestamps[i],
                        'from_cluster': clusters[i-1],
                        'to_cluster': clusters[i],
                        'group': 'all'
                    })
    else:
        # For each group, detect pattern changes separately
        for group_name, group in df.groupby(group_by):
            # Sort by time
            sorted_group = group.sort_values(time_column)
            
            # Skip if the group is too small
            if len(sorted_group) < window_size * 2:
                continue
            
            # Feature extraction
            X = sorted_group[metrics].values
            
            # Normalize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Extract subsequences using a sliding window
            subsequences = []
            timestamps = []
            
            for i in range(len(X_scaled) - window_size + 1):
                subseq = X_scaled[i:i+window_size].flatten()
                subsequences.append(subseq)
                timestamps.append(sorted_group.iloc[i+window_size-1][time_column])
            
            # Group subsequences
            if len(subsequences) > n_clusters:
                kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
                clusters = kmeans.fit_predict(np.array(subsequences))
                
                # Detect cluster changes
                cluster_changes = np.diff(clusters, prepend=clusters[0])
                change_indices = np.where(cluster_changes != 0)[0]
                
                # Record information about pattern changes
                for i in change_indices:
                    if i > 0 and i < len(timestamps):
                        results.append({
                            'time': timestamps[i],
                            'from_cluster': clusters[i-1],
                            'to_cluster': clusters[i],
                            'group': group_name
                        })
    
    return pd.DataFrame(results)


def detect_anomalies_ensemble(df, metric_column='value', time_column='elapsed_minutes', 
                            contamination=0.05, group_by=None):
    """
    Detects anomalies using an ensemble of algorithms.
    
    Args:
        df (DataFrame): DataFrame with metric data.
        metric_column (str): Column with metric values.
        time_column (str): Column with time values.
        contamination (float): Expected proportion of anomalies in the data.
        group_by (list): Columns to group data by (e.g., ['tenant', 'phase']).
        
    Returns:
        DataFrame: DataFrame with consolidated anomaly detection results.
    """
    # Load configurations
    from pipeline.config import DEFAULT_NOISY_TENANT
    
    # Check and handle NaN values
    df_clean = df.copy()
    
    # Determine the noisy tenant from a DataFrame attribute or use the default
    noisy_tenant = None
    if hasattr(df, 'noisy_tenant') and df.noisy_tenant:
        noisy_tenant = df.noisy_tenant
    else:
        noisy_tenant = DEFAULT_NOISY_TENANT
    
    # Handle the case of the noisy tenant, which might not exist in certain phases
    if 'tenant' in df_clean.columns and group_by and 'tenant' in group_by:
        # Check for NaNs in the value column for the noisy tenant
        noisy_tenant_mask = df_clean['tenant'] == noisy_tenant
        if noisy_tenant_mask.any():
            # Replace NaNs with zeros only for the noisy tenant
            df_clean.loc[noisy_tenant_mask, metric_column] = df_clean.loc[noisy_tenant_mask, metric_column].fillna(0)
    
    # Fill any other remaining NaNs with the column mean
    if df_clean[metric_column].isna().any():
        df_clean[metric_column] = df_clean[metric_column].fillna(df_clean[metric_column].mean())
    
    # Apply different algorithms to the cleaned DataFrame
    result_if = detect_anomalies_isolation_forest(df_clean, metric_column, contamination, group_by)
    
    # Adjust the number of neighbors based on data size
    if group_by is not None:
        # Calculate average group size
        group_sizes = df_clean.groupby(group_by).size()
        n_neighbors = max(5, int(group_sizes.mean() * 0.1))  # 10% of average size
    else:
        n_neighbors = max(5, int(len(df_clean) * 0.1))  # 10% of total size
    
    result_lof = detect_anomalies_local_outlier_factor(df, metric_column, n_neighbors, contamination, group_by)
    
    # Combine results
    result = df.copy()
    result['anomaly_score_if'] = result_if['anomaly_score_if']
    result['is_anomaly_if'] = result_if['is_anomaly_if']
    result['anomaly_score_lof'] = result_lof['anomaly_score_lof']
    result['is_anomaly_lof'] = result_lof['is_anomaly_lof']
    
    # Normalize scores to allow combination
    if 'anomaly_score_if' in result.columns and 'anomaly_score_lof' in result.columns:
        # Function to normalize between 0 and 1
        def normalize(series):
            min_val = series.min()
            max_val = series.max()
            if max_val > min_val:
                return (series - min_val) / (max_val - min_val)
            else:
                return series * 0
        
        result['normalized_score_if'] = normalize(result['anomaly_score_if'])
        result['normalized_score_lof'] = normalize(result['anomaly_score_lof'])
        
        # Calculate combined score (average of normalized scores)
        result['anomaly_score_combined'] = (result['normalized_score_if'] + result['normalized_score_lof']) / 2
        
        # Determine combined anomalies (either algorithm classified as anomaly)
        result['is_anomaly'] = result['is_anomaly_if'] | result['is_anomaly_lof']
    
    # Add change point detection
    change_result, change_info = detect_change_points(df, metric_column, time_column, group_by=group_by)
    result['is_change_point'] = change_result['is_change_point']
    
    return result, change_info
