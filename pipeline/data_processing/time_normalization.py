"""
Time normalization module for the noisy neighbors experiment.

This module provides functions to normalize timestamps and add
elapsed time fields to the experiment DataFrames.
"""

import pandas as pd


def add_elapsed_time(df, group_by=['round', 'phase']):
    """
    Adds elapsed time columns, calculated from the start of each group.
    
    Args:
        df (DataFrame): DataFrame with experiment data
        group_by (list): Columns to group by when calculating initial time
        
    Returns:
        DataFrame: DataFrame with additional elapsed time columns
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Find the initial timestamp for each group
    start_times = df.groupby(group_by)['datetime'].min().reset_index()
    
    # Rename the column to facilitate merging
    start_times = start_times.rename(columns={'datetime': 'start_time'})
    
    # Merge with the original DataFrame
    result = pd.merge(result, start_times, on=group_by)
    
    # Calculate elapsed time in seconds since the start of each group
    result['elapsed_seconds'] = (result['datetime'] - result['start_time']).dt.total_seconds()
    
    # Calculate elapsed time in minutes (for easier visualization)
    result['elapsed_minutes'] = result['elapsed_seconds'] / 60.0
    
    return result


def add_experiment_elapsed_time(df, experiment_start_time=None, group_by=None):
    """
    Adds an elapsed time column since the start of the experiment or round.
    If experiment_start_time is provided, it is used as the global reference.
    Otherwise, groups by `group_by` to find the initial time for each group.

    Args:
        df (DataFrame): DataFrame with experiment data.
        experiment_start_time (datetime, optional): Timestamp of the global start of the experiment.
        group_by (list, optional): Columns to group by when calculating initial time if 
                                   experiment_start_time is not provided (usually ['round']).
                                   Defaults to ['round'] if experiment_start_time is None.
        
    Returns:
        DataFrame: DataFrame with an additional experiment elapsed time column.
    """
    result = df.copy()

    if experiment_start_time is not None:
        # Use the provided global start time
        result['experiment_start_time'] = experiment_start_time
    else:
        # Determine the start time per group
        if group_by is None:
            group_by = ['round'] # Default if not specified and experiment_start_time is None
        
        if not all(col in df.columns for col in group_by):
            # If grouping columns do not exist, calculate global time for the entire DataFrame
            # This can happen if, for example, data is already filtered for a single round/group
            global_start_time = df['datetime'].min()
            result['experiment_start_time'] = global_start_time
        else:
            start_times = df.groupby(group_by)['datetime'].min().reset_index()
            start_times = start_times.rename(columns={'datetime': 'experiment_start_time'})
            result = pd.merge(result, start_times, on=group_by, how='left')

    # Calculate elapsed time in seconds since the start of each group/global
    result['experiment_elapsed_seconds'] = (result['datetime'] - result['experiment_start_time']).dt.total_seconds()
    result['experiment_elapsed_minutes'] = result['experiment_elapsed_seconds'] / 60.0
    
    return result


def add_phase_markers(df, phase_column='phase', phase_display_names=None):
    """
    Adds phase start and end markers for easier visualization.
    
    Args:
        df (DataFrame): DataFrame with experiment data
        phase_column (str): Name of the column containing the phase name
        phase_display_names (dict, optional): Dictionary to map raw phase names to display names.
        
    Returns:
        DataFrame: The same DataFrame with an additional simplified phase name column
        dict: Dictionary with phase start markers for use in plots
    """
    # Extract simplified phase names
    if phase_display_names:
        df['phase_name'] = df[phase_column].apply(lambda x: phase_display_names.get(x, x.split('-')[-1].strip()) if isinstance(x, str) else x)
    else:
        df['phase_name'] = df[phase_column].apply(lambda x: x.split('-')[-1].strip() if isinstance(x, str) else x)
    
    # Group by round and calculate the start of each phase
    phase_markers = {}
    
    for round_name, group in df.groupby('round'):
        round_markers = {}
        for phase, phase_df in group.groupby('phase'):
            if len(phase_df) > 0:
                start_time = phase_df['experiment_elapsed_minutes'].min()
                round_markers[phase] = start_time
        
        phase_markers[round_name] = round_markers
    
    return df, phase_markers


def merge_phase_info(df, phase_info=None):
    """
    Adds phase duration information to the DataFrame.
    
    Args:
        df (DataFrame): DataFrame with experiment data
        phase_info (dict): Dictionary with phase information
                           (if None, it will be inferred from the data)
        
    Returns:
        DataFrame: DataFrame with additional phase information
    """
    result = df.copy()
    
    if phase_info is None:
        # Infer phase information from the data
        phase_durations = {}
        
        for (round_name, phase_name), group in df.groupby(['round', 'phase']):
            duration = group['elapsed_minutes'].max() - group['elapsed_minutes'].min()
            if round_name not in phase_durations:
                phase_durations[round_name] = {}
            phase_durations[round_name][phase_name] = duration
    else:
        phase_durations = phase_info
    
    # Add duration information
    phase_duration_list = []
    
    for idx, row in df.iterrows():
        round_name = row['round']
        phase_name = row['phase']
        
        if round_name in phase_durations and phase_name in phase_durations[round_name]:
            duration = phase_durations[round_name][phase_name]
            phase_duration_list.append(duration)
        else:
            phase_duration_list.append(None)
    
    result['phase_duration_minutes'] = phase_duration_list
    
    return result


def normalize_time(df, time_column='datetime', group_by=None, method='elapsed_seconds'):
    """
    Normalizes the time column according to the specified method.

    Args:
        df (DataFrame): Input DataFrame.
        time_column (str): Name of the timestamp column (datetime objects).
        group_by (list, optional): Columns to group by before normalizing.
                                   If None, normalizes globally.
        method (str): Normalization method:
                      'elapsed_seconds': Seconds since the first timestamp (in group or globally).
                      'elapsed_minutes': Minutes since the first timestamp.
                      'experiment_elapsed_seconds': Seconds since the start of the experiment (requires 'round' in group_by).
                      'experiment_elapsed_minutes': Minutes since the start of the experiment.

    Returns:
        DataFrame: DataFrame with the new normalized time column.
    """
    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in the DataFrame.")
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        raise ValueError(f"Time column '{time_column}' must be of datetime type.")

    df_copy = df.copy()

    if method in ['elapsed_seconds', 'elapsed_minutes']:
        if group_by:
            df_copy['start_time_group'] = df_copy.groupby(group_by)[time_column].transform('min')
            elapsed = (df_copy[time_column] - df_copy['start_time_group']).dt.total_seconds()
            df_copy.drop(columns=['start_time_group'], inplace=True)
        else:
            elapsed = (df_copy[time_column] - df_copy[time_column].min()).dt.total_seconds()
        
        if method == 'elapsed_seconds':
            df_copy['normalized_time'] = elapsed
        else: # elapsed_minutes
            df_copy['normalized_time'] = elapsed / 60.0

    elif method in ['experiment_elapsed_seconds', 'experiment_elapsed_minutes']:
        # This method is more specific and usually uses add_experiment_elapsed_time
        # Here, we replicate similar logic for the sake of a generic 'normalize_time' function
        # We assume that 'round' (or a similar high-level grouping) is in group_by
        if not group_by or not any(g_col in df_copy.columns for g_col in group_by):
            # If there is no group_by, calculate from the global start of the dataset
             exp_start_col_name = 'experiment_start_time_calc'
             df_copy[exp_start_col_name] = df_copy[time_column].min()
        else:
            exp_start_col_name = 'experiment_start_time_calc'
            df_copy[exp_start_col_name] = df_copy.groupby(group_by)[time_column].transform('min')

        elapsed_exp = (df_copy[time_column] - df_copy[exp_start_col_name]).dt.total_seconds()
        df_copy.drop(columns=[exp_start_col_name], inplace=True)

        if method == 'experiment_elapsed_seconds':
            df_copy['normalized_time'] = elapsed_exp
        else: # experiment_elapsed_minutes
            df_copy['normalized_time'] = elapsed_exp / 60.0
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return df_copy
