"""
Module for comparative analysis between tenants for the noisy neighbors experiment.

This module provides functions to compare metrics between different tenants
and analyze the impact of the noisy tenant on others.
"""

import pandas as pd
import numpy as np
from scipy import stats


def calculate_correlation_matrix(metrics_dict, tenants=None, round_name='round-1', noisy_tenant=None):
    """
    Calculates a correlation matrix between metrics of different tenants.
    
    Args:
        metrics_dict (dict): Dictionary with DataFrames for each metric.
        tenants (list): List of tenants to include (None = all).
        round_name (str): Round to be analyzed.
        noisy_tenant (str): Specific tenant generating noise (default: DEFAULT_NOISY_TENANT from configuration).
        
    Returns:
        DataFrame: Correlation matrix between tenant metrics.
    """
    from pipeline.config import DEFAULT_NOISY_TENANT
    
    # Determine which tenant is generating noise
    noisy_tenant_id = noisy_tenant if noisy_tenant else DEFAULT_NOISY_TENANT
    
    # Prepare data for correlation
    correlation_data = {}
    
    for metric_name, metric_df in metrics_dict.items():
        # Filter by the specified round
        round_df = metric_df[metric_df['round'] == round_name]
        
        if tenants:
            round_df = round_df[round_df['tenant'].isin(tenants)]
        
        # Check if the noise-generating tenant is present
        has_noisy_tenant = noisy_tenant_id in round_df['tenant'].unique()
        
        # Pivot to have one column for each tenant
        pivot = round_df.pivot_table(
            index='datetime',
            columns='tenant',
            values='value'
        )
        
        # Fill NaN values with 0
        pivot.fillna(0, inplace=True)
        
        # Ensure noise-generating tenant is present for consistency in analyses
        if not has_noisy_tenant and noisy_tenant_id not in pivot.columns:
            pivot[noisy_tenant_id] = 0  # Add noise-generating tenant with zero values
        
        # Add to dictionary with metric prefix
        for tenant_col_name in pivot.columns:
            correlation_data[f"{metric_name}_{tenant_col_name}"] = pivot[tenant_col_name]
    
    # Create DataFrame with all series
    corr_df = pd.DataFrame(correlation_data)
    
    # Calculate correlation
    correlation_matrix = corr_df.corr()
    
    return correlation_matrix


def compare_tenant_metrics(df, baseline_tenant='tenant-a', metric_column='value'):
    """
    Compares metrics of different tenants with a reference tenant.
    
    Args:
        df (DataFrame): DataFrame with metrics data.
        baseline_tenant (str): Reference tenant for comparison.
        metric_column (str): Column with metric values.
        
    Returns:
        DataFrame: DataFrame with comparisons between tenants.
    """
    # Filter only the reference tenant
    baseline_df = df[df['tenant'] == baseline_tenant].copy()
    
    # Prepare a DataFrame to store the results
    results = []
    
    # For each combination of round and phase
    for (round_name, phase_name), group in df.groupby(['round', 'phase']):
        # Reference tenant data for this combination
        base = baseline_df[(baseline_df['round'] == round_name) & 
                          (baseline_df['phase'] == phase_name)]
        
        if len(base) == 0:
            continue  # Skip if no baseline data
            
        base_mean = base[metric_column].mean()
        base_std = base[metric_column].std()
        
        # For each tenant
        for tenant, tenant_group in group.groupby('tenant'):
            if tenant == baseline_tenant:
                continue  # Skip baseline tenant
                
            tenant_mean = tenant_group[metric_column].mean()
            tenant_std = tenant_group[metric_column].std()
            
            # Calculate percentage difference
            percent_diff = ((tenant_mean - base_mean) / base_mean) * 100 if base_mean != 0 else np.nan
            
            # Statistical test (t-test for two independent samples)
            t_stat, p_value = stats.ttest_ind(
                base[metric_column].dropna(),
                tenant_group[metric_column].dropna(),
                equal_var=False  # Assume different variances (Welch's t-test)
            )
            
            # Add to results
            results.append({
                'round': round_name,
                'phase': phase_name,
                'baseline_tenant': baseline_tenant,
                'compared_tenant': tenant,
                'baseline_mean': base_mean,
                'baseline_std': base_std,
                'tenant_mean': tenant_mean,
                'tenant_std': tenant_std,
                'percent_difference': percent_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_difference': p_value < 0.05  # Statistical significance
            })
    
    return pd.DataFrame(results)


def calculate_inter_tenant_correlation_per_metric(metric_df_single_round: pd.DataFrame, value_col='value', time_col='datetime', tenant_col='tenant') -> pd.DataFrame:
    """
    Calculates the inter-tenant correlation matrix for a single metric and a single round.

    Args:
        metric_df_single_round (pd.DataFrame): DataFrame containing data for one metric
                                                 for multiple tenants in a single round.
        value_col (str): Name of the column with metric values.
        time_col (str): Name of the column with timestamps.
        tenant_col (str): Name of the column with tenant identifiers.
                                                 
    Returns:
        pd.DataFrame: Inter-tenant correlation matrix.
    """
    pivot_df = metric_df_single_round.pivot_table(
        index=time_col,
        columns=tenant_col,
        values=value_col
    )
    # Fill NaNs that may arise if tenants do not have data at all timestamps
    pivot_df.ffill(inplace=True) # Forward fill
    pivot_df.bfill(inplace=True) # Backward fill
    pivot_df.fillna(0, inplace=True) # For tenants with no data at all
    return pivot_df.corr()


def calculate_inter_tenant_covariance_per_metric(metric_df_single_round: pd.DataFrame, value_col='value', time_col='datetime', tenant_col='tenant') -> pd.DataFrame:
    """
    Calculates the inter-tenant covariance matrix for a single metric and a single round.

    Args:
        metric_df_single_round (pd.DataFrame): DataFrame containing data for one metric
                                                 for multiple tenants in a single round.
        value_col (str): Name of the column with metric values.
        time_col (str): Name of the column with timestamps.
        tenant_col (str): Name of the column with tenant identifiers.

    Returns:
        pd.DataFrame: Inter-tenant covariance matrix.
    """
    pivot_df = metric_df_single_round.pivot_table(
        index=time_col,
        columns=tenant_col,
        values=value_col
    )
    # Fill NaNs that may arise if tenants do not have data at all timestamps
    pivot_df.ffill(inplace=True) # Forward fill
    pivot_df.bfill(inplace=True) # Backward fill
    pivot_df.fillna(0, inplace=True) # For tenants with no data at all
    return pivot_df.cov()
