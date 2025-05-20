"""
Module for performing correlation and covariance analysis on telemetry data.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import numpy as np
from pipeline.visualization.plots import create_heatmap

# Functions from covariance_analysis.py will be moved here

def calculate_pearson_correlation(series1: pd.Series | np.ndarray, series2: pd.Series | np.ndarray) -> tuple[float, float]:
    """
    Calculates the Pearson correlation coefficient and p-value between two time series.

    Args:
        series1: The first time series (pandas Series or NumPy array).
        series2: The second time series (pandas Series or NumPy array).

    Returns:
        A tuple containing the Pearson correlation coefficient and the p-value.
    """
    if len(series1) != len(series2):
        raise ValueError("Series must have the same length to calculate Pearson correlation.")
    if len(series1) < 2:
        return np.nan, np.nan # Not enough data points

    correlation, p_value = pearsonr(series1, series2)
    return correlation, p_value

def calculate_spearman_correlation(series1: pd.Series | np.ndarray, series2: pd.Series | np.ndarray) -> tuple[float, float]:
    """
    Calculates the Spearman rank correlation coefficient and p-value between two time series.

    Args:
        series1: The first time series (pandas Series or NumPy array).
        series2: The second time series (pandas Series or NumPy array).

    Returns:
        A tuple containing the Spearman correlation coefficient and the p-value.
    """
    if len(series1) != len(series2):
        raise ValueError("Series must have the same length to calculate Spearman correlation.")
    if len(series1) < 2:
        return np.nan, np.nan # Not enough data points

    correlation, p_value = spearmanr(series1, series2)
    return correlation, p_value

def calculate_lagged_cross_correlation(series1: pd.Series | np.ndarray, series2: pd.Series | np.ndarray, max_lag: int) -> pd.DataFrame:
    """
    Calculates the lagged cross-correlation between two time series.

    Args:
        series1: The first time series (pandas Series or NumPy array).
        series2: The second time series (pandas Series or NumPy array).
        max_lag: The maximum lag (positive and negative) to consider.

    Returns:
        A pandas DataFrame with columns 'lag' and 'correlation'.
    """
    if len(series1) != len(series2):
        raise ValueError("Series must have the same length for lagged cross-correlation.")
    
    s1 = np.asarray(series1)
    s2 = np.asarray(series2)
    
    n = len(s1)
    if n == 0:
        return pd.DataFrame({'lag': [], 'correlation': []})

    lags = np.arange(-max_lag, max_lag + 1)
    correlations = []

    for lag in lags:
        if lag < 0:
            # series2 leads series1
            corr, _ = pearsonr(s1[:n+lag], s2[-lag:])
        elif lag == 0:
            corr, _ = pearsonr(s1, s2)
        else: # lag > 0
            # series1 leads series2
            corr, _ = pearsonr(s1[lag:], s2[:n-lag])
        correlations.append(corr)
        
    return pd.DataFrame({'lag': lags, 'correlation': correlations})

def calculate_covariance_matrix(metrics_dict, tenants=None, phase=None, round_name='round-1', time_col='datetime', value_col='value', tenant_col='tenant', phase_col='phase', round_col='round'):
    """
    Calculates a covariance matrix between metrics of different tenants.
    
    Args:
        metrics_dict (dict): Dictionary with DataFrames for each metric.
            If the format is {metric: {round: DataFrame}}, the round_name will be used to select data.
            If the format is {metric: DataFrame}, the entire DataFrame will be used.
        tenants (list): List of tenants to include (None = all).
        phase (str): Specific phase for analysis (None = all).
        round_name (str): Round to be analyzed.
        time_col (str): Name of the time/datetime column.
        value_col (str): Name of the metric value column.
        tenant_col (str): Name of the tenant column.
        phase_col (str): Name of the phase column.
        round_col (str): Name of the round column.
        
    Returns:
        DataFrame: Covariance matrix between tenant metrics.
        DataFrame: Correlation matrix (for comparison).
    """
    # Prepare data for covariance
    covariance_data = {}
    
    for metric_name, metric_data in metrics_dict.items():
        # Handle different data formats
        if isinstance(metric_data, dict):
            # Format is {metric: {round: DataFrame}}
            if round_name in metric_data:
                round_df = metric_data[round_name]
            else:
                print(f"Round '{round_name}' not found for metric '{metric_name}'. Skipping.")
                continue
        else:
            # Format is {metric: DataFrame}
            round_df = metric_data
            # If the DataFrame has a 'round' column, filter by round_name
            if round_col in round_df.columns:
                round_df = round_df[round_df[round_col] == round_name]
        
        # Filter by phase if specified
        if phase:
            round_df = round_df[round_df[phase_col] == phase]
            
        if tenants:
            round_df = round_df[round_df[tenant_col].isin(tenants)]
        
        # Pivot to have one column for each tenant
        pivot = round_df.pivot_table(
            index=time_col,
            columns=tenant_col,
            values=value_col
        )
        
        # Add to dictionary with metric prefix
        for tenant_val in pivot.columns:
            covariance_data[f"{metric_name}_{tenant_val}"] = pivot[tenant_val]
    
    # Create DataFrame with all series
    cov_df = pd.DataFrame(covariance_data)
    
    # Calculate covariance
    covariance_matrix = cov_df.cov()
    
    # Calculate correlation for comparison
    correlation_matrix = cov_df.corr()
    
    return covariance_matrix, correlation_matrix

def plot_covariance_matrix(covariance_matrix, output_path, title='Covariance Matrix', cmap='coolwarm', annot=True, fmt=".2f"):
    """
    Plots a covariance matrix as a heatmap.

    Args:
        covariance_matrix (pd.DataFrame): The covariance matrix to plot.
        output_path (str): Path to save the plot.
        title (str): Title of the plot.
        cmap (str): Colormap for the heatmap.
        annot (bool): Whether to annotate cells with values.
        fmt (str): String format for annotations.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(covariance_matrix, annot=annot, fmt=fmt, cmap=cmap, linewidths=.5)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    try:
        plt.savefig(output_path)
        print(f"Covariance matrix plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving covariance matrix plot: {e}")
    plt.close()

def calculate_inter_tenant_covariance_per_metric(
    metrics_data,
    output_dir, 
    current_round,
    phase_name=None,
    time_col='datetime',
    value_col='value',
    tenant_col='tenant',
    phase_col='phase',
    round_col='round'
):
    """
    Calculates and saves inter-tenant covariance matrices for each metric.
    """
    all_covariances = {}
    for metric_name, metric_df_or_dict in metrics_data.items():
        
        if isinstance(metric_df_or_dict, dict): # Handles {metric: {round: df}}
            if current_round not in metric_df_or_dict:
                print(f"Round {current_round} not found for metric {metric_name}. Skipping covariance calculation.")
                continue
            metric_df = metric_df_or_dict[current_round]
        elif isinstance(metric_df_or_dict, pd.DataFrame): # Handles {metric: df}
            metric_df = metric_df_or_dict
            if round_col in metric_df.columns: # Filter by round if round_col exists
                 metric_df = metric_df[metric_df[round_col] == current_round]
            # If no round_col, assume df is for the current_round or is aggregated
        else:
            print(f"Unexpected data type for metric {metric_name}. Skipping covariance calculation.")
            continue

        if metric_df.empty:
            print(f"Data for metric {metric_name}, round {current_round} is empty. Skipping covariance calculation.")
            continue
            
        df_round = metric_df # Already filtered or structured for the current round

        if phase_name:
            if phase_col not in df_round.columns:
                print(f"Phase column '{phase_col}' not found in data for metric {metric_name}. Cannot filter by phase. Skipping phase filter.")
                df_phase = df_round # Proceed without phase filtering
            else:
                df_phase = df_round[df_round[phase_col] == phase_name]
        else:
            df_phase = df_round # Use all data for the round if no phase is specified

        if df_phase.empty or tenant_col not in df_phase.columns or df_phase[tenant_col].nunique() < 2:
            # print(f"Not enough data or tenants for covariance in metric {metric_name}, phase {phase_name if phase_name else 'all'}. Skipping.")
            continue

        try:
            pivot_df = df_phase.pivot_table(index=time_col, columns=tenant_col, values=value_col)
        except Exception as e:
            print(f"Error pivoting data for metric {metric_name}, phase {phase_name if phase_name else 'all'}: {e}. Skipping.")
            continue
            
        if pivot_df.shape[1] < 2: # Need at least two tenants for covariance
            # print(f"Pivot table for metric {metric_name}, phase {phase_name if phase_name else 'all'} has less than 2 tenant columns. Skipping.")
            continue

        # Drop columns with all NaNs, which can occur if a tenant has no data for the specific phase/time
        pivot_df.dropna(axis=1, how='all', inplace=True)
        if pivot_df.shape[1] < 2: # Check again after dropping all-NaN columns
            # print(f"Pivot table for metric {metric_name}, phase {phase_name if phase_name else 'all'} has less than 2 tenant columns after dropping NaNs. Skipping.")
            continue
            
        # Fill remaining NaNs (e.g., with mean or by interpolation) or drop rows with NaNs
        # For covariance, it's often better to drop rows with any NaNs to ensure pairwise completeness
        pivot_df.dropna(axis=0, how='any', inplace=True)
        if pivot_df.shape[0] < 2: # Need at least 2 valid time points
            # print(f"Pivot table for metric {metric_name}, phase {phase_name if phase_name else 'all'} has less than 2 valid time points after dropping NaNs. Skipping.")
            continue

        covariance_matrix = pivot_df.cov()
        all_covariances[metric_name] = covariance_matrix
        
        # Optionally, save or plot each matrix here
        # Example:
        # plot_file = os.path.join(output_dir, f"{metric_name}_{phase_name if phase_name else 'all_phases'}_covariance.png")
        # plot_covariance_matrix(covariance_matrix, plot_file, title=f"Covariance - {metric_name} ({phase_name if phase_name else 'All Phases'})")

    return all_covariances
