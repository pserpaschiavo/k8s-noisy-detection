"""
Data aggregation module for the noisy neighbors experiment.

This module provides functions to aggregate and summarize metric data
from the experiment by tenant, phase, round, etc.
"""

import pandas as pd
import numpy as np


def calculate_tenant_stats(df, value_column='value', group_columns=['tenant', 'round', 'phase']):
    """
    Calculates summary statistics for metric values, grouped by tenant, round, and phase.
    
    Args:
        df (DataFrame): DataFrame with metric data
        value_column (str): Column containing the metric values
        group_columns (list): Columns to group by (by default, tenant, round, and phase)
        
    Returns:
        DataFrame: DataFrame with calculated statistics
    """
    stats = df.groupby(group_columns)[value_column].agg([
        'mean', 'median', 'std', 'min', 'max', 'count',
        lambda x: np.percentile(x, 25),
        lambda x: np.percentile(x, 75),
        lambda x: np.percentile(x, 95)
    ]).reset_index()
    
    # Rename the calculated columns
    stats = stats.rename(columns={
        '<lambda_0>': 'percentile_25',
        '<lambda_1>': 'percentile_75',
        '<lambda_2>': 'percentile_95'
    })
    
    return stats


def calculate_inter_tenant_impact(df, noisy_tenant='tenant-b', attack_phase='2 - Attack', 
                                  value_column='mean', baseline_phase='1 - Baseline'):
    """
    Calculates the impact of the noisy tenant on other tenants.
    
    Args:
        df (DataFrame): DataFrame with summary statistics by tenant and phase
        noisy_tenant (str): Name of the noisy tenant
        attack_phase (str): Name of the attack phase
        value_column (str): Name of the column with metric values (usually 'mean')
        baseline_phase (str): Name of the baseline phase
        
    Returns:
        DataFrame: DataFrame with the calculated impact
    """
    # Filter only the baseline and attack phases
    filtered_df = df[df['phase'].isin([baseline_phase, attack_phase])].copy()
    
    # Pivot to have columns for each phase
    pivot = filtered_df.pivot_table(
        index=['tenant', 'round'],
        columns='phase',
        values=value_column
    ).reset_index()
    
    # Calculate the impact (percentage change)
    pivot['impact_percent'] = ((pivot[attack_phase] - pivot[baseline_phase]) / pivot[baseline_phase]) * 100
    
    # Remove the noisy tenant if necessary
    if noisy_tenant is not None:
        impact_summary = pivot[pivot['tenant'] != noisy_tenant]
    else:
        impact_summary = pivot
    
    return impact_summary


def calculate_recovery_effectiveness(df, value_column='mean', 
                                   baseline_phase='1 - Baseline',
                                   attack_phase='2 - Attack',
                                   recovery_phase='3 - Recovery'):
    """
    Calculates the effectiveness of recovery after the attack phase.
    
    Args:
        df (DataFrame): DataFrame with summary statistics by tenant and phase
        value_column (str): Name of the column with metric values (usually 'mean')
        baseline_phase (str): Name of the baseline phase
        attack_phase (str): Name of the attack phase
        recovery_phase (str): Name of the recovery phase
        
    Returns:
        DataFrame: DataFrame with the calculated recovery effectiveness
    """
    # Filter only the relevant phases
    filtered_df = df[df['phase'].isin([baseline_phase, attack_phase, recovery_phase])].copy()
    
    # Pivot to have columns for each phase
    pivot = filtered_df.pivot_table(
        index=['tenant', 'round'],
        columns='phase',
        values=value_column
    ).reset_index()
    
    # Calculate the degradation during the attack (how much it worsened)
    pivot['attack_degradation'] = ((pivot[attack_phase] - pivot[baseline_phase]) / pivot[baseline_phase]) * 100
    
    # Calculate how much of the degradation was recovered
    pivot['recovery_percent'] = ((pivot[recovery_phase] - pivot[attack_phase]) / (pivot[baseline_phase] - pivot[attack_phase])) * 100
    
    # Calculate how much is still missing to return to baseline (in percentage)
    pivot['baseline_diff_percent'] = ((pivot[recovery_phase] - pivot[baseline_phase]) / pivot[baseline_phase]) * 100
    
    return pivot


def aggregate_data_by_custom_elements(df, aggregation_keys=None, elements_to_aggregate=None, value_column='value', agg_functions=None):
    """
    Aggregates data based on aggregation keys and specific user-defined elements.

    Args:
        df (DataFrame): DataFrame with metric data.
        aggregation_keys (list): List of columns to group by (e.g., ["tenant", "phase", "custom_group"]).
        elements_to_aggregate (list or dict): List of specific values to filter in the first aggregation key,
                                             or a dictionary where keys are column names from `aggregation_keys`
                                             and values are lists of elements to filter in those columns.
                                             If None, all elements are considered.
        value_column (str): Column containing the metric values.
        agg_functions (list or dict): List of aggregation functions (e.g., ['mean', 'std']) or a dictionary
                                      mapping columns to aggregation functions.
                                      Default is ['mean', 'std', 'count'].

    Returns:
        DataFrame: Aggregated DataFrame.
    """
    if aggregation_keys is None:
        aggregation_keys = ['tenant', 'phase'] # Default aggregation

    df_filtered = df.copy()

    if elements_to_aggregate:
        if isinstance(elements_to_aggregate, dict):
            for filter_column, filter_values in elements_to_aggregate.items():
                if filter_column in df_filtered.columns and filter_values:
                    df_filtered = df_filtered[df_filtered[filter_column].isin(filter_values)]
        elif isinstance(elements_to_aggregate, list) and aggregation_keys:
            # Keeps the original behavior if it's a list: filters in the first aggregation key
            filter_column = aggregation_keys[0]
            if filter_column in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[filter_column].isin(elements_to_aggregate)]
        # If elements_to_aggregate is neither dict nor list, or if it's an empty list, do nothing here.
    
    # If the DataFrame is empty after filtering, return an empty DataFrame.
    if df_filtered.empty:
        return pd.DataFrame(columns=df.columns.tolist() + ['mean', 'std', 'count', 'percentile_25', 'percentile_75', 'percentile_95'])

    # Calculate statistics
    agg_stats = df_filtered.groupby(aggregation_keys)[value_column].agg([
        'mean', 'std', 'count',
        lambda x: np.percentile(x, 25),
        lambda x: np.percentile(x, 75),
        lambda x: np.percentile(x, 95)
    ]).reset_index()

    # Rename the calculated columns
    agg_stats = agg_stats.rename(columns={
        '<lambda_0>': 'percentile_25',
        '<lambda_1>': 'percentile_75',
        '<lambda_2>': 'percentile_95'
    })

    return agg_stats


def aggregate_by_time(df, time_column='elapsed_minutes', value_column='value', agg_interval='5T', agg_funcs=None):
    """
    Aggregates data by time intervals.

    Args:
        df (DataFrame): DataFrame with time column (datetime or timedelta) and value.
        time_column (str): Name of the time column to be used for resampling.
                           Must be datetime or convertible to timedelta if numeric (minutes, seconds).
        value_column (str): Name of the value column to be aggregated.
        agg_interval (str): String representing the aggregation interval (e.g., '1T' for 1 minute, '5S' for 5 seconds).
                            Used with pd.Grouper or resample.
        agg_funcs (list or dict, optional): Aggregation functions to apply (e.g., ['mean', 'std']).
                                            Defaults to ['mean'].

    Returns:
        DataFrame: DataFrame aggregated by time.
    """
    if agg_funcs is None:
        agg_funcs = ['mean']

    if df.empty or time_column not in df.columns or value_column not in df.columns:
        # Return an empty DataFrame with expected columns if the input data is invalid
        return pd.DataFrame()

    df_copy = df.copy()

    # Ensure the time column is datetime to use resample or pd.Grouper
    if pd.api.types.is_numeric_dtype(df_copy[time_column]):
        df_copy[time_column] = pd.to_timedelta(df_copy[time_column], unit='m')
        base_timestamp = df_copy['datetime'].min() if 'datetime' in df_copy.columns else pd.Timestamp("2000-01-01")
        df_copy['resample_time'] = base_timestamp + df_copy[time_column]
        time_col_for_resample = 'resample_time'
    elif pd.api.types.is_datetime64_any_dtype(df_copy[time_column]):
        time_col_for_resample = time_column
    else:
        raise ValueError(f"The time column '{time_column}' must be numeric (minutes/seconds) or datetime.")

    grouping_cols = [col for col in ['tenant', 'phase', 'round'] if col in df_copy.columns]

    if grouping_cols:
        try:
            aggregated_df = df_copy.groupby(
                [pd.Grouper(key=time_col_for_resample, freq=agg_interval)] + grouping_cols
            )[value_column].agg(agg_funcs).reset_index()
        except Exception as e:
            print(f"Error aggregating with pd.Grouper and additional groups: {e}. Trying global resample.")
            df_copy = df_copy.set_index(time_col_for_resample)
            aggregated_df = df_copy[value_column].resample(agg_interval).agg(agg_funcs).reset_index()
    else:
        df_copy = df_copy.set_index(time_col_for_resample)
        aggregated_df = df_copy[value_column].resample(agg_interval).agg(agg_funcs).reset_index()

    return aggregated_df
