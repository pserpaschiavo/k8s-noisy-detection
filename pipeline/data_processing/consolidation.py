"""
Data consolidation module for the noisy neighbors experiment.

This module provides functions to load and consolidate metric data
from different tenants, phases, and rounds of the experiment.
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import re


def list_available_tenants(experiment_dir):
    """
    Lists all available tenants in the experiment directory.
    
    Args:
        experiment_dir (str): Path to the experiment directory
        
    Returns:
        list: List of tenant names
    """
    tenants = set()
    
    # Search for tenant directories in all phases and rounds
    # Corrected to iterate correctly over the directory structure
    for round_folder_name in os.listdir(experiment_dir):
        round_dir_path = os.path.join(experiment_dir, round_folder_name)
        if os.path.isdir(round_dir_path) and round_folder_name.startswith("round-"):
            for phase_folder_name in os.listdir(round_dir_path):
                phase_dir_path = os.path.join(round_dir_path, phase_folder_name)
                if os.path.isdir(phase_dir_path): # Assume any subdirectory here is a phase
                    for tenant_folder_name in os.listdir(phase_dir_path):
                        tenant_dir_path = os.path.join(phase_dir_path, tenant_folder_name)
                        if os.path.isdir(tenant_dir_path): # Assume any subdirectory here is a tenant
                            tenants.add(tenant_folder_name)
    
    return sorted(list(tenants))


def list_available_metrics(experiment_dir, tenant="tenant-a"):
    """
    Lists all available metrics for a specific tenant.
    
    Args:
        experiment_dir (str): Path to the experiment directory
        tenant (str): Name of the tenant to list metrics for (can be None to search in all)
        
    Returns:
        list: List of metric names
    """
    metrics = set()
    
    # Search for CSV files within the specified tenant's directories
    # If tenant is None, search in any tenant directory
    # Corrected to iterate correctly over the directory structure
    for round_folder_name in os.listdir(experiment_dir):
        round_dir_path = os.path.join(experiment_dir, round_folder_name)
        if os.path.isdir(round_dir_path) and round_folder_name.startswith("round-"):
            for phase_folder_name in os.listdir(round_dir_path):
                phase_dir_path = os.path.join(round_dir_path, phase_folder_name)
                if os.path.isdir(phase_dir_path):
                    # If a specific tenant is provided, search only in it
                    search_tenants_in_phase = [tenant] if tenant else os.listdir(phase_dir_path)
                    for tenant_folder_name in search_tenants_in_phase:
                        tenant_dir_path = os.path.join(phase_dir_path, tenant_folder_name)
                        if os.path.isdir(tenant_dir_path):
                            for file_name in os.listdir(tenant_dir_path):
                                if file_name.endswith(".csv"):
                                    metric_name = file_name.replace(".csv", "")
                                    metrics.add(metric_name)
    
    return sorted(list(metrics))


def parse_timestamp(timestamp_str):
    """
    Converts a timestamp string in the format used in CSV files to a datetime object.
    
    Args:
        timestamp_str (str): Timestamp string in "YYYYMMDD_HHMMSS" format
        
    Returns:
        datetime: Corresponding datetime object
    """
    return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")


def load_metric_data(experiment_dir, metric_name, tenants=None, phases=None, rounds=None):
    """
    Loads data for a specific metric for the selected tenants, phases, and rounds.
    
    Args:
        experiment_dir (str): Path to the experiment directory
        metric_name (str): Name of the metric to load
        tenants (list): List of tenants to include (None = all)
        phases (list): List of phases to include (None = all)
        rounds (list): List of rounds to include (None = all)
        
    Returns:
        DataFrame: Consolidated DataFrame with the metric data
    """
    all_data = []
    
    # If no specific list is provided, include everything
    # Corrected to correctly list rounds and phases if they are None
    if rounds is None:
        rounds_to_scan = [r for r in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, r)) and r.startswith("round-")]
    else:
        rounds_to_scan = rounds

    if tenants is None:
        tenants_to_scan = list_available_tenants(experiment_dir)
    else:
        tenants_to_scan = tenants
    
    for round_name in rounds_to_scan:
        round_path = os.path.join(experiment_dir, round_name)
        if not os.path.isdir(round_path):
            continue

        phases_to_scan_for_round = []
        if phases is None: # If phases is None, list all subdirectories of round_path
            phases_to_scan_for_round = [p for p in os.listdir(round_path) if os.path.isdir(os.path.join(round_path, p))]
        else: # If phases is a list, use that list
            phases_to_scan_for_round = phases

        for phase_name_pattern in phases_to_scan_for_round: # phase_name_pattern can be an exact name or a glob
            phase_dir_actual_path = os.path.join(round_path, phase_name_pattern)
            
            if not os.path.isdir(phase_dir_actual_path):
                continue

            phase_name_actual = os.path.basename(phase_dir_actual_path) # Actual phase name
            phase_number_match = re.search(r'^(\d+)', phase_name_actual)
            phase_number = int(phase_number_match.group(1)) if phase_number_match else 0
                
            for tenant_name in tenants_to_scan:
                csv_path = os.path.join(phase_dir_actual_path, tenant_name, f"{metric_name}.csv")
                    
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        df['tenant'] = tenant_name
                        df['round'] = round_name
                        df['phase'] = phase_name_actual # Use the actual phase name
                        df['phase_number'] = phase_number
                        df['datetime'] = df['timestamp'].apply(parse_timestamp)
                        all_data.append(df)
                    except Exception as e:
                        print(f"Error loading {csv_path}: {e}")
    
    if not all_data:
        print(f"No data found for metric '{metric_name}' with the applied filters.")
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)


def load_multiple_metrics(experiment_dir, metric_names, tenants=None, phases=None, rounds=None):
    """
    Loads data for multiple metrics for the selected tenants, phases, and rounds.
    
    Args:
        experiment_dir (str): Path to the experiment directory
        metric_names (list): List of metric names to load
        tenants (list): List of tenants to include (None = all)
        phases (list): List of phases to include (None = all)
        rounds (list): List of rounds to include (None = all)
        
    Returns:
        dict: Dictionary where keys are metric names. 
              Each value is another dictionary where keys are round names 
              and values are DataFrames with the metric data for that round.
    """
    metrics_data_final_structure = {}
    
    for metric in metric_names:
        df_all_rounds_for_metric = load_metric_data(experiment_dir, metric, tenants, phases, rounds)
        
        if not df_all_rounds_for_metric.empty:
            rounds_data_for_metric = {}
            if 'round' in df_all_rounds_for_metric.columns:
                for round_name, group_df in df_all_rounds_for_metric.groupby('round'):
                    rounds_data_for_metric[round_name] = group_df.copy()
            else:
                # This case should ideally not happen if load_metric_data correctly adds the 'round' column.
                print(f"Warning: 'round' column not found for metric {metric} after loading. Check the load_metric_data function.")
                # Fallback: Store the entire DataFrame under a generic key if 'round' column is missing.
                rounds_data_for_metric['unknown_round_data'] = df_all_rounds_for_metric
            
            if rounds_data_for_metric: 
                metrics_data_final_structure[metric] = rounds_data_for_metric
            elif not df_all_rounds_for_metric.empty:
                 print(f"Warning: No round data was grouped for metric {metric}, although the DataFrame was not empty.")
    
    return metrics_data_final_structure


def load_experiment_data(experiment_dir, tenants=None, metrics=None, phases=None, rounds=None):
    """
    Loads all relevant data from an experiment, potentially all available metrics.

    Args:
        experiment_dir (str): Base directory of the experiment.
        tenants (list, optional): List of tenants to include. Defaults to all available.
        metrics (list, optional): List of metrics to load. Defaults to all available for the first tenant.
        phases (list, optional): List of phases to include. Defaults to all.
        rounds (list, optional): List of rounds to include. Defaults to all.

    Returns:
        dict: Dictionary with DataFrames for each loaded metric.
    """
    print(f"Loading experiment data from: {experiment_dir}")

    if tenants is None:
        tenants = list_available_tenants(experiment_dir)
        if not tenants:
            print("No tenants found.")
            return {}
        print(f"Tenants to be loaded: {tenants}")

    if metrics is None:
        # Try to list metrics from the first found tenant as representative
        if tenants:
            metrics = list_available_metrics(experiment_dir, tenant=tenants[0])
        if not metrics:
            print("No metrics found for the specified tenants.")
            return {}
        print(f"Metrics to be loaded: {metrics}")

    # Use load_multiple_metrics to load the data
    experiment_data = load_multiple_metrics(
        experiment_dir,
        metric_names=metrics,
        tenants=tenants,
        phases=phases,
        rounds=rounds
    )

    if not experiment_data:
        print("No data was loaded for the experiment.")
    
    return experiment_data


def select_tenants(metrics_data, selected_tenants):
    """
    Filters metric data to include only the selected tenants.

    Args:
        metrics_data (dict): Dictionary of DataFrames by metric.
        selected_tenants (list): List of tenants to keep.

    Returns:
        dict: Filtered dictionary of DataFrames.
    """
    if not selected_tenants:
        return metrics_data # Return original if no selection is made

    filtered_data = {}
    for metric_name, df in metrics_data.items():
        if 'tenant' in df.columns:
            filtered_df = df[df['tenant'].isin(selected_tenants)]
            if not filtered_df.empty:
                filtered_data[metric_name] = filtered_df
        else:
            # If the tenant column does not exist, keep the DataFrame (or decide other logic)
            filtered_data[metric_name] = df 
    return filtered_data
