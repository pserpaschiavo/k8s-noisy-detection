"""
Module for advanced analyses, including covariance, causality, and entropy.

This module implements advanced methods for analyzing data from the noisy neighbors
experiment, focusing on complex relationships between tenants and phases.
"""

import numpy as np
import pandas as pd
from scipy import stats
# from statsmodels.tsa.stattools import grangercausalitytests # Not currently used, consider removing if not planned
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from pipeline.config import METRICS_CONFIG, IMPACT_SCORE_WEIGHTS, PHASE_DISPLAY_NAMES


def calculate_covariance_matrix(metrics_dict, tenants=None, phase=None, round_name='round-1', time_col='datetime', value_col='value', tenant_col='tenant', phase_col='phase', round_col='round'):
    """
    Calculates a covariance matrix between metrics of different tenants.
    
    Args:
        metrics_dict (dict): Dictionary with DataFrames for each metric.
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
    
    for metric_name, metric_df in metrics_dict.items():
        # Filter by the specified round
        round_df = metric_df[metric_df[round_col] == round_name]
        
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


def calculate_cross_tenant_entropy(df, tenant1, tenant2, value_col='value', time_col='datetime', tenant_col='tenant'):
    """
    Calculates cross-entropy between two tenants for a metric.
    We use mutual information as a measure related to cross-entropy.
    
    Args:
        df (DataFrame): DataFrame with metric data.
        tenant1 (str): First tenant for analysis.
        tenant2 (str): Second tenant for analysis.
        value_col (str): Column with metric values.
        time_col (str): Name of the time/datetime column.
        tenant_col (str): Name of the tenant column.
        
    Returns:
        float: Mutual information value (related to cross-entropy).
        dict: Additional information about the relationship.
    """
    # Filter data for the two tenants
    data1 = df[df[tenant_col] == tenant1].sort_values(time_col)
    data2 = df[df[tenant_col] == tenant2].sort_values(time_col)
    
    # Check if we have enough data
    if len(data1) < 10 or len(data2) < 10:
        return None, {"error": "Insufficient data for entropy analysis (min 10 points required)."}

    # Align time series using the original logic, then refine
    common_times = pd.Series(list(set(data1[time_col]).intersection(set(data2[time_col]))))
    
    if len(common_times) < 10:
        # Attempt interpolation if few common points
        min_time = min(data1[time_col].min(), data2[time_col].min())
        max_time = max(data1[time_col].max(), data2[time_col].max())
        
        # Try to infer frequency or use a reasonable default (e.g., '1s' or '1min')
        freq = pd.infer_freq(data1[time_col]) or pd.infer_freq(data2[time_col]) or '1min'
        try:
            time_index = pd.date_range(start=min_time, end=max_time, freq=freq)
        except ValueError: # If inferred frequency is not compatible with date_range
            time_index = pd.date_range(start=min_time, end=max_time, periods=max(len(data1), len(data2)))

        series1_aligned = data1.set_index(time_col)[value_col].reindex(time_index).interpolate(method='linear')
        series2_aligned = data2.set_index(time_col)[value_col].reindex(time_index).interpolate(method='linear')
    else:
        series1_aligned = data1[data1[time_col].isin(common_times)].set_index(time_col)[value_col].sort_index()
        series2_aligned = data2[data2[time_col].isin(common_times)].set_index(time_col)[value_col].sort_index()
    
    # Remove NaNs that may have arisen from interpolation/reindexing (especially at the ends)
    series1_aligned.dropna(inplace=True)
    series2_aligned.dropna(inplace=True)

    # Ensure series have the same index after cleaning
    common_idx = series1_aligned.index.intersection(series2_aligned.index)
    
    if len(common_idx) < 10: # Check again after alignment and dropna
        return None, {"error": f"Insufficient overlapping data points after alignment for {tenant1}-{tenant2} (found {len(common_idx)})."}

    series1 = series1_aligned[common_idx]
    series2 = series2_aligned[common_idx]

    if series1.empty or series2.empty: # Additional safety check
        return None, {"error": f"Empty series after final alignment for {tenant1}-{tenant2}."}

    # Check for constant series (zero or very low standard deviation)
    epsilon = 1e-9 
    std1 = series1.std()
    std2 = series2.std()

    if std1 < epsilon or std2 < epsilon:
        info = {
            "tenant1": tenant1,
            "tenant2": tenant2,
            "mutual_information": 0.0,
            "correlation": 0.0,
            "n_samples": len(series1),
            "note": "One or both series were constant or near-constant. MI/Corr set to 0."
        }
        return 0.0, info

    # Normalize the data
    try:
        scaler1 = StandardScaler()
        X1_scaled = scaler1.fit_transform(series1.values.reshape(-1, 1))
        
        scaler2 = StandardScaler()
        X2_scaled = scaler2.fit_transform(series2.values.reshape(-1, 1))

        X1 = X1_scaled.flatten()
        X2 = X2_scaled.flatten()

        # Safety check for NaNs post-scaling (should be prevented by std check)
        if np.isnan(X1).any() or np.isnan(X2).any():
            info = {
                "tenant1": tenant1,
                "tenant2": tenant2,
                "mutual_information": 0.0,
                "correlation": 0.0,
                "n_samples": len(X1),
                "error": "NaN values after scaling (unexpected). MI/Corr set to 0."
            }
            return 0.0, info

        # Calculate mutual information
        mi = mutual_info_regression(X1.reshape(-1, 1), X2, random_state=0)[0] # random_state for reproducibility
        
        # Calculate correlation
        corr_matrix = np.corrcoef(X1, X2)
        
        if corr_matrix.shape == (2,2) and not np.isnan(corr_matrix[0, 1]):
            corr = corr_matrix[0, 1]
        else: # Fallback if correlation matrix is problematic
            corr = 0.0 
            if np.isnan(mi): # If MI is also NaN, default to 0
                mi = 0.0
                
    except Exception as e:
        info = {
            "tenant1": tenant1,
            "tenant2": tenant2,
            "mutual_information": np.nan, # Use NaN to indicate calculation error
            "correlation": np.nan,
            "n_samples": len(series1), # n_samples before the error
            "error": f"Exception in MI/corr calculation: {str(e)}"
        }
        return np.nan, info

    info = {
        "tenant1": tenant1,
        "tenant2": tenant2,
        "mutual_information": mi,
        "correlation": corr,
        "n_samples": len(X1)
    }
    
    return mi, info


def calculate_entropy_metrics(df, tenants=None, phase=None, value_col='value', time_col='datetime', tenant_col='tenant', phase_col='phase'):
    """
    Calculates entropy metrics for different tenants and phases.
    
    Args:
        df (DataFrame): DataFrame with metric data.
        tenants (list): List of tenants to include (None = all).
        phase (str): Specific phase for analysis (None = all).
        value_col (str): Column with metric values.
        time_col (str): Name of the time/datetime column.
        tenant_col (str): Name of the tenant column.
        phase_col (str): Name of the phase column.
        
    Returns:
        DataFrame: Results of entropy metrics.
    """
    if tenants is None:
        tenants = sorted(df[tenant_col].unique())
    
    # Filter by phase if specified
    if phase:
        df = df[df[phase_col] == phase]
    
    results = []
    
    # Calculate entropy for each pair of tenants
    for i, tenant1 in enumerate(tenants):
        for tenant2 in tenants[i+1:]:
            mi, info = calculate_cross_tenant_entropy(df, tenant1, tenant2, value_col=value_col, time_col=time_col, tenant_col=tenant_col)
            
            if mi is not None:
                info["phase"] = phase if phase else "all"
                results.append(info)
    
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame()


def calculate_inter_tenant_correlation_per_metric(
    metrics_data,
    output_dir, # Not used in current implementation, consider removing or using
    current_round,
    phase_name=None,
    time_col='datetime',
    value_col='value',
    tenant_col='tenant',
    phase_col='phase',
    round_col='round'
):
    """
    Calculates and saves inter-tenant correlation matrices for each metric.
    """
    all_correlations = {}
    for metric_name, metric_df in metrics_data.items():
        df_round = metric_df[metric_df[round_col] == current_round]
        
        if phase_name:
            df_phase = df_round[df_round[phase_col] == phase_name]
        else:
            df_phase = df_round # Use all data for the round if no phase is specified

        if df_phase.empty or df_phase[tenant_col].nunique() < 2:
            continue

        pivot_df = df_phase.pivot_table(index=time_col, columns=tenant_col, values=value_col)
        
        if pivot_df.shape[1] < 2: # Need at least two tenants to correlate
            continue
            
        correlation_matrix = pivot_df.corr()
        all_correlations[metric_name] = correlation_matrix

    return all_correlations


def calculate_inter_tenant_covariance_per_metric(
    metrics_data,
    output_dir, # Not used in current implementation, consider removing or using
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
    for metric_name, metric_df in metrics_data.items():
        df_round = metric_df[metric_df[round_col] == current_round]
        
        if phase_name:
            df_phase = df_round[df_round[phase_col] == phase_name]
        else:
            df_phase = df_round # Use all data for the round if no phase is specified

        if df_phase.empty or df_phase[tenant_col].nunique() < 2:
            continue

        pivot_df = df_phase.pivot_table(index=time_col, columns=tenant_col, values=value_col)

        if pivot_df.shape[1] < 2: # Need at least two tenants for covariance
            continue

        covariance_matrix = pivot_df.cov()
        all_covariances[metric_name] = covariance_matrix

    return all_covariances


def calculate_normalized_impact_score(metrics_data, noisy_tenant, impact_phases, baseline_phases, weights, metrics_config, tenant_col='tenant', phase_col='phase', value_col='value', round_col='round'):
    """
    Calculates a normalized impact score for each tenant.

    Args:
        metrics_data (dict): Dictionary of DataFrames, one for each metric.
                             If using aggregated data, these should already be in the expected format
                             (e.g., 'mean' as value_col, 'time_bin' as time_col, 'tenant', 'phase').
        noisy_tenant (str): The tenant identified as noisy (for reference, not used directly in calculating others' scores).
        impact_phases (list): List of impact phases (e.g., ["2 - Attack"]).
        baseline_phases (list): List of baseline phases (e.g., ["1 - Baseline"]).
        weights (dict): Weights for each metric in the score calculation (from config.IMPACT_SCORE_WEIGHTS).
        metrics_config (dict): Configuration of metrics, including 'higher_is_worse' (from config.METRICS_CONFIG).
        tenant_col (str): Name of the tenant column.
        phase_col (str): Name of the phase column.
        value_col (str): Name of the value column (e.g., 'value' or 'mean' for aggregated data).
        round_col (str): Name of the round column (used to distinguish data if not aggregated).

    Returns:
        pd.DataFrame: DataFrame with 'tenant' and 'normalized_impact_score'.
    """
    all_tenants_impact = {}

    # Ensure impact and baseline phases are lists
    if isinstance(impact_phases, str):
        impact_phases = [impact_phases]
    if isinstance(baseline_phases, str):
        baseline_phases = [baseline_phases]

    # Collect all unique tenants present in the data
    all_tenants = set()
    for metric_df in metrics_data.values():
        if tenant_col in metric_df.columns:
            all_tenants.update(metric_df[tenant_col].unique())
    
    if not all_tenants:
        print("Warning: No tenants found in metrics_data for impact score calculation.")
        return pd.DataFrame(columns=[tenant_col, 'normalized_impact_score'])

    for tenant in all_tenants:
        tenant_scores = []
        for metric_name, metric_df in metrics_data.items():
            if metric_name not in weights or metric_name not in metrics_config:
                continue

            # Filter data for the current tenant
            tenant_metric_df = metric_df[metric_df[tenant_col] == tenant]
            if tenant_metric_df.empty:
                tenant_scores.append({'metric': metric_name, 'normalized_change': 0, 'weight': weights[metric_name]})
                continue

            # Calculate mean value in baseline and impact phases
            baseline_values = tenant_metric_df[tenant_metric_df[phase_col].isin(baseline_phases)][value_col]
            impact_values = tenant_metric_df[tenant_metric_df[phase_col].isin(impact_phases)][value_col]

            if baseline_values.empty or impact_values.empty:
                percent_change = 0.0
            else:
                mean_baseline = baseline_values.mean()
                mean_impact = impact_values.mean()

                if mean_baseline == 0: 
                    if mean_impact == 0:
                        percent_change = 0.0
                    else:
                        # If baseline is zero and impact is non-zero, assign a large change
                        percent_change = np.sign(mean_impact) * 100.0 
                else:
                    percent_change = ((mean_impact - mean_baseline) / abs(mean_baseline)) * 100

            # Invert change if lower is worse for the metric
            if not metrics_config[metric_name].get('higher_is_worse', True):
                percent_change *= -1 # Lower values are better, so a decrease is a positive impact (or less negative)
            
            tenant_scores.append({'metric': metric_name, 'raw_change': percent_change, 'weight': weights[metric_name]})

        if not tenant_scores: # Should not happen if all_tenants is populated and metrics_data is not empty
            all_tenants_impact[tenant] = 0 
            continue

        current_tenant_total_score = 0
        total_weight = 0
        for score_info in tenant_scores:
            # Cap the change to avoid extreme values dominating the score
            capped_change = np.clip(score_info['raw_change'], -200, 200) # Cap at +/- 200%
            current_tenant_total_score += capped_change * score_info['weight']
            total_weight += score_info['weight']
        
        if total_weight > 0:
            all_tenants_impact[tenant] = current_tenant_total_score / total_weight
        else:
            all_tenants_impact[tenant] = 0 # Avoid division by zero if no weighted metrics found

    impact_df = pd.DataFrame(list(all_tenants_impact.items()), columns=[tenant_col, 'normalized_impact_score'])
    
    # Normalize scores to a 0-100 range for easier interpretation
    if not impact_df.empty and 'normalized_impact_score' in impact_df.columns:
        min_score = impact_df['normalized_impact_score'].min()
        max_score = impact_df['normalized_impact_score'].max()
        if max_score > min_score: # Standard min-max normalization
            impact_df['normalized_impact_score'] = 100 * (impact_df['normalized_impact_score'] - min_score) / (max_score - min_score)
        elif max_score == min_score and min_score != 0: # All scores are the same but non-zero
            impact_df['normalized_impact_score'] = 50 # Assign a middle value
        else: # All scores are zero or df is empty
            impact_df['normalized_impact_score'] = 0 
            
    return impact_df.sort_values(by='normalized_impact_score', ascending=False)

# Potential future addition: Granger Causality (requires careful consideration of stationarity and lag selection)
# def calculate_granger_causality(df, tenant1, tenant2, metric_name, max_lag=5, test='ssr_chi2test', verbose=False, time_col='datetime', value_col='value', tenant_col='tenant'):
#     """
#     Performs Granger causality test between two tenants for a specific metric.
#     NOTE: This is a placeholder and requires careful implementation and data preprocessing.
#     """
#     data1 = df[(df[tenant_col] == tenant1)][[time_col, value_col]].set_index(time_col).sort_index()
#     data2 = df[(df[tenant_col] == tenant2)][[time_col, value_col]].set_index(time_col).sort_index()

#     # Ensure data is stationary (e.g., by differencing) - CRITICAL STEP
#     # data1_diff = data1.diff().dropna()
#     # data2_diff = data2.diff().dropna()
    
#     # Align data - ensure same length and time points
#     # merged_data = pd.merge(data1_diff, data2_diff, on=time_col, suffixes=('_t1', '_t2'))
#     # if merged_data.empty or len(merged_data) < max_lag + 5: # Need enough data points
#     #     return None, {"error": "Not enough overlapping data after differencing and merging."}

#     # temp_df = merged_data[[f'{value_col}_t2', f'{value_col}_t1']] # Target variable first (does t1 granger-cause t2?)
    
#     # try:
#     #     gc_results = grangercausalitytests(temp_df, maxlag=max_lag, verbose=verbose)
#     #     # Process results to extract p-values or F-statistics
#     #     # ...
#     #     return gc_results, {} # Placeholder
#     # except Exception as e:
#     #     return None, {"error": f"Granger causality test failed: {str(e)}"}
#     pass # Functionality not fully implemented
