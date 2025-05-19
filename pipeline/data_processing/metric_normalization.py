"""
Module for resource metric normalization.

This module provides functions to convert raw metrics into percentages
relative to total cluster resources or limits defined in quota manifests.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple

# Import quota parser
try:
    from .quota_parser import (
        get_tenant_quotas, create_node_config_from_quotas, 
        get_best_unit_for_value, format_value_with_unit,
        convert_to_best_unit, get_formatted_quota_values,
        get_quota_summary
    )
except ImportError:
    # Alternative import path when run as a script
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.data_processing.quota_parser import (
        get_tenant_quotas, create_node_config_from_quotas,
        get_best_unit_for_value, format_value_with_unit,
        convert_to_best_unit, get_formatted_quota_values,
        get_quota_summary
    )


def normalize_metrics_by_node_capacity(metrics_dict: Dict[str, pd.DataFrame],
                                     node_config: Dict[str, float],
                                     use_tenant_quotas: bool = True,
                                     quota_file: str = None,
                                     add_relative_values: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Normalizes raw metrics to percentages of total available resources or tenant limits.
    
    Args:
        metrics_dict: Dictionary with metric DataFrames
        node_config: Node configuration with total resource capacities
        use_tenant_quotas: If True, normalizes against specific tenant limits
        quota_file: Path to quota file (optional)
        add_relative_values: If True, adds columns with relative values (% of limits)
        
    Returns:
        Dictionary with normalized metrics
    """
    normalized_metrics = {}
    
    # Load tenant quotas if requested
    tenant_quotas = {}
    if use_tenant_quotas:
        tenant_quotas = get_tenant_quotas(quota_file)
    
    # Normalize CPU (assuming raw values are in cores)
    if 'cpu_usage' in metrics_dict and 'CPUS' in node_config:
        df = metrics_dict['cpu_usage'].copy()
        
        # Apply per-tenant normalization when use_tenant_quotas=True
        if use_tenant_quotas:
            # Apply per-tenant normalization
            for tenant, group in df.groupby('tenant'):
                namespace = tenant  # Assuming tenant name is the same as namespace
                # Get specific tenant limit or use global value
                tenant_cpu_limit = tenant_quotas.get(namespace, {}).get('cpu_limit', 0)
                
                if tenant_cpu_limit > 0:
                    # Normalize against the specific tenant limit (values in cores)
                    tenant_mask = df['tenant'] == tenant
                    
                    # Convert to percentage of tenant limit
                    if add_relative_values:
                        df.loc[tenant_mask, 'normalized_value'] = df.loc[tenant_mask, 'value'] * 100 / tenant_cpu_limit
                    
                    # Format the tenant limit legibly for the description
                    cpu_value, cpu_unit = convert_to_best_unit(tenant_cpu_limit, 'cpu')
                    formatted_limit = f"{cpu_value:.2f} {cpu_unit}"
                    
                    # Add detailed information about the normalization
                    if add_relative_values:
                        df.loc[tenant_mask, 'normalized_description'] = f"% of {formatted_limit} (quota)"
                    
                    # Preserve quota values in different formats for later use
                    df.loc[tenant_mask, 'quota_limit'] = tenant_cpu_limit
                    df.loc[tenant_mask, 'quota_limit_formatted'] = formatted_limit
                    df.loc[tenant_mask, 'quota_limit_cores'] = tenant_cpu_limit
                    df.loc[tenant_mask, 'quota_limit_millicores'] = tenant_cpu_limit * 1000
                    
                    # Add percentages relative to total node capacity
                    df.loc[tenant_mask, 'quota_percent_of_node'] = tenant_cpu_limit * 100 / node_config['CPUS']
                else:
                    # Fallback to global normalization when tenant has no defined quota
                    tenant_mask = df['tenant'] == tenant
                    
                    if add_relative_values:
                        df.loc[tenant_mask, 'normalized_value'] = df.loc[tenant_mask, 'value'] * 100 / node_config['CPUS']
                    
                    # Format the global limit legibly
                    cpu_value, cpu_unit = convert_to_best_unit(node_config['CPUS'], 'cpu')
                    formatted_limit = f"{cpu_value:.2f} {cpu_unit}"
                    
                    if add_relative_values:
                        df.loc[tenant_mask, 'normalized_description'] = f"% of {formatted_limit} (node)"
                    
                    # Add the limit value for use in other parts of the code
                    df.loc[tenant_mask, 'quota_limit'] = node_config['CPUS']
                    df.loc[tenant_mask, 'quota_limit_formatted'] = formatted_limit
                    df.loc[tenant_mask, 'quota_limit_cores'] = node_config['CPUS']
                    df.loc[tenant_mask, 'quota_limit_millicores'] = node_config['CPUS'] * 1000
                    df.loc[tenant_mask, 'quota_percent_of_node'] = 100  # 100% of the node
        else:
            # Global normalization for all tenants
            total_cpu_cores = node_config['CPUS']
            
            if add_relative_values:
                df['normalized_value'] = df['value'] * 100 / total_cpu_cores
            
            # Format the global limit legibly
            cpu_value, cpu_unit = convert_to_best_unit(total_cpu_cores, 'cpu')
            formatted_limit = f"{cpu_value:.2f} {cpu_unit}"
            
            if add_relative_values:
                df['normalized_description'] = f"% of {formatted_limit}"
            
            # Add the limit value for use in other parts of the code
            df['quota_limit'] = total_cpu_cores
            df['quota_limit_formatted'] = formatted_limit
            df['quota_limit_cores'] = total_cpu_cores
            df['quota_limit_millicores'] = total_cpu_cores * 1000
            df['quota_percent_of_node'] = 100  # 100% of the node
        
        # Add additional information
        df['value_cores'] = df['value']  # Preserve value in cores
        
        # Add other useful representations
        df['value_millicores'] = df['value'] * 1000  # Convert to millicores
        
        # Format values in different units for later use
        df['value_formatted'] = df.apply(
            lambda row: format_value_with_unit(row['value'], 'cpu'), axis=1
        )
        
        if add_relative_values:
            df['unit'] = '%'
            
        df['metric_type'] = 'cpu'
        normalized_metrics['cpu_usage'] = df
    
    # Normalize memory (assuming raw values are in bytes)
    if 'memory_usage' in metrics_dict and 'MEMORY_BYTES' in node_config:
        df = metrics_dict['memory_usage'].copy()
        
        # Apply per-tenant normalization when use_tenant_quotas=True
        if use_tenant_quotas:
            # Apply per-tenant normalization
            for tenant, group in df.groupby('tenant'):
                namespace = tenant  # Assuming tenant name is the same as namespace
                # Get specific tenant limit or use global value
                tenant_memory_limit = tenant_quotas.get(namespace, {}).get('memory_limit', 0)
                
                if tenant_memory_limit > 0:
                    # Normalize against the specific tenant limit
                    tenant_mask = df['tenant'] == tenant
                    
                    # Convert to percentage of tenant limit
                    if add_relative_values:
                        df.loc[tenant_mask, 'normalized_value'] = df.loc[tenant_mask, 'value'] * 100 / tenant_memory_limit
                    
                    # Format the tenant limit legibly
                    mem_value, mem_unit = convert_to_best_unit(tenant_memory_limit, 'memory')
                    formatted_limit = f"{mem_value:.2f} {mem_unit}"
                    
                    # Add detailed information about the normalization
                    if add_relative_values:
                        df.loc[tenant_mask, 'normalized_description'] = f"% of {formatted_limit} (quota)"
                    
                    # Preserve original values in different units for later use
                    df.loc[tenant_mask, 'quota_limit'] = tenant_memory_limit
                    df.loc[tenant_mask, 'quota_limit_formatted'] = formatted_limit
                    df.loc[tenant_mask, 'quota_limit_bytes'] = tenant_memory_limit
                    df.loc[tenant_mask, 'quota_limit_kib'] = tenant_memory_limit / (2**10)
                    df.loc[tenant_mask, 'quota_limit_mib'] = tenant_memory_limit / (2**20)
                    df.loc[tenant_mask, 'quota_limit_gib'] = tenant_memory_limit / (2**30)
                    
                    # Add percentages relative to total node capacity
                    df.loc[tenant_mask, 'quota_percent_of_node'] = tenant_memory_limit * 100 / node_config['MEMORY_BYTES']
                else:
                    # Fallback to global normalization when tenant has no defined quota
                    tenant_mask = df['tenant'] == tenant
                    
                    if add_relative_values:
                        df.loc[tenant_mask, 'normalized_value'] = df.loc[tenant_mask, 'value'] * 100 / node_config['MEMORY_BYTES']
                    
                    # Format the global limit legibly
                    mem_value, mem_unit = convert_to_best_unit(node_config['MEMORY_BYTES'], 'memory')
                    formatted_limit = f"{mem_value:.2f} {mem_unit}"
                    
                    if add_relative_values:
                        df.loc[tenant_mask, 'normalized_description'] = f"% of {formatted_limit} (node)"
                    
                    # Add the limit value for use in other parts of the code
                    df.loc[tenant_mask, 'quota_limit'] = node_config['MEMORY_BYTES']
                    df.loc[tenant_mask, 'quota_limit_formatted'] = formatted_limit
                    df.loc[tenant_mask, 'quota_limit_bytes'] = node_config['MEMORY_BYTES']
                    df.loc[tenant_mask, 'quota_limit_kib'] = node_config['MEMORY_BYTES'] / (2**10)
                    df.loc[tenant_mask, 'quota_limit_mib'] = node_config['MEMORY_BYTES'] / (2**20)
                    df.loc[tenant_mask, 'quota_limit_gib'] = node_config['MEMORY_BYTES'] / (2**30)
                    df.loc[tenant_mask, 'quota_percent_of_node'] = 100  # 100% of the node
        else:
            # Global normalization for all tenants
            total_memory = node_config['MEMORY_BYTES']
            
            if add_relative_values:
                df['normalized_value'] = df['value'] * 100 / total_memory
            
            # Format the global limit legibly
            mem_value, mem_unit = convert_to_best_unit(total_memory, 'memory')
            formatted_limit = f"{mem_value:.2f} {mem_unit}"
            
            if add_relative_values:
                df['normalized_description'] = f"% of {formatted_limit}"
            
            # Add the limit value for use in other parts of the code
            df['quota_limit'] = total_memory
            df['quota_limit_formatted'] = formatted_limit
            df['quota_limit_bytes'] = total_memory
            df['quota_limit_kib'] = total_memory / (2**10)
            df['quota_limit_mib'] = total_memory / (2**20)
            df['quota_limit_gib'] = total_memory / (2**30)
            df['quota_percent_of_node'] = 100  # 100% of the node
        
        # Add additional information in different units for later use
        df['value_bytes'] = df['value']  # Preserve original value in bytes
        df['value_kib'] = df['value'] / (2**10)
        df['value_mib'] = df['value'] / (2**20)
        df['value_gib'] = df['value'] / (2**30)
        
        # Format values in readable units for later use
        df['value_formatted'] = df.apply(
            lambda row: format_value_with_unit(row['value'], 'memory'), axis=1
        )
        
        if add_relative_values:
            df['unit'] = '%'
            
        df['metric_type'] = 'memory'
        normalized_metrics['memory_usage'] = df
    
    return normalized_metrics


def apply_normalization_to_all_metrics(metrics_dict: Dict[str, pd.DataFrame], 
                                     node_config: Dict[str, float], 
                                     replace_original: bool = False,
                                     use_tenant_quotas: bool = True,
                                     show_as_percentage: bool = False,
                                     use_readable_units: bool = True,
                                     add_relative_values: bool = True,
                                     quota_file: str = None) -> Dict[str, pd.DataFrame]:
    """
    Applies normalization to all metrics and optionally replaces original values.
    
    Args:
        metrics_dict: Dictionary with metric DataFrames
        node_config: Node configuration with total resource capacities
        replace_original: If True, replaces the 'value' column with normalized values
        use_tenant_quotas: If True, normalizes against specific tenant limits
        show_as_percentage: If True, displays values as percentages
        use_readable_units: If True, converts values to more readable units
        add_relative_values: If True, adds columns with relative values (% of limits)
        quota_file: Path to quota file (optional)
        
    Returns:
        Dictionary with processed metrics
    """
    processed_metrics = {}
    
    # First step: Normalize against capacities or quotas
    normalized = normalize_metrics_by_node_capacity(
        metrics_dict, 
        node_config, 
        use_tenant_quotas=use_tenant_quotas,
        quota_file=quota_file,
        add_relative_values=add_relative_values
    )
    
    # Second step: Apply readable units if requested
    if use_readable_units:
        for metric_name, df in normalized.items():
            processed_df = df.copy()
            
            if replace_original and 'normalized_value' in df.columns:
                # Replace original values with normalized values
                processed_df['original_value'] = processed_df['value'].copy()
                processed_df['value'] = processed_df['normalized_value']
                
                if show_as_percentage:
                    processed_df['unit'] = '%'
            
            # Preserve in output dictionary
            processed_metrics[metric_name] = processed_df
    else:
        # If not using readable units, just copy normalized DataFrames
        processed_metrics = {k: v.copy() for k, v in normalized.items()}
        
        if replace_original:
            # Replace original values with normalized values when requested
            for metric_name, df in processed_metrics.items():
                if 'normalized_value' in df.columns:
                    df['original_value'] = df['value'].copy()
                    df['value'] = df['normalized_value']
                    
                    if show_as_percentage:
                        df['unit'] = '%'
    
    # Process metrics not handled in normalization (like disk and network)
    for metric_name, df in metrics_dict.items():
        if metric_name not in processed_metrics:
            if use_readable_units:
                # Determine metric type based on name
                if 'disk' in metric_name.lower():
                    metric_type = 'disk'
                elif 'network' in metric_name.lower() or 'bandwidth' in metric_name.lower():
                    metric_type = 'network'
                else:
                    metric_type = None
                
                # If type determined, format with appropriate units
                if metric_type:
                    processed_df = df.copy()
                    
                    # Add readable formatting
                    processed_df['value_formatted'] = processed_df['value'].apply(
                        lambda x: format_value_with_unit(x, metric_type)
                    )
                    
                    # Add values in common units for later use
                    if metric_type == 'disk':
                        processed_df['value_bytes'] = processed_df['value']
                        processed_df['value_kb'] = processed_df['value'] / (2**10)
                        processed_df['value_mb'] = processed_df['value'] / (2**20)
                        processed_df['value_gb'] = processed_df['value'] / (2**30)
                        processed_df['metric_type'] = 'disk'
                        
                    elif metric_type == 'network':
                        processed_df['value_bytes'] = processed_df['value']
                        processed_df['value_kb'] = processed_df['value'] / (2**10)
                        processed_df['value_mb'] = processed_df['value'] / (2**20)
                        processed_df['value_gb'] = processed_df['value'] / (2**30)
                        processed_df['metric_type'] = 'network'
                    
                    processed_metrics[metric_name] = processed_df
                else:
                    # If type not determined, just copy data
                    processed_metrics[metric_name] = df.copy()
            else:
                # If not using readable units, just copy data
                processed_metrics[metric_name] = df.copy()
    
    return processed_metrics


def auto_format_metrics(metrics_dict: Dict[str, Dict[str, pd.DataFrame]],
                        metric_type_map: Dict[str, str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Automatically formats metrics for display with the best units.
    This function does not normalize, it only converts to more readable units.
    
    Args:
        metrics_dict: Nested dictionary (metric_name -> round_name -> DataFrame) of metrics.
                      The structure is metric_name (str) -> round_name (str) -> pd.DataFrame.
        metric_type_map: Optional mapping of metric names to their types.
        
    Returns:
        Nested dictionary with formatted metrics, preserving the input structure.
    """
    formatted_outer = {}
    
    # Default mapping of suffixes to types
    default_type_map = {
        'cpu': ['cpu', 'cores', 'processor'],
        'memory': ['memory', 'mem', 'ram'],
        'disk': ['disk', 'storage', 'volume', 'pv', 'io_', 'iops'],
        'network': ['network', 'bandwidth', 'throughput', 'rx', 'tx']
    }
    
    for metric_name, rounds_dict in metrics_dict.items():
        formatted_inner = {}
        for round_name, df in rounds_dict.items():
            # Ensure df is a DataFrame before processing
            if not isinstance(df, pd.DataFrame):
                # If not a DataFrame, carry it over as is (e.g. if it's None or already processed)
                # Consider logging a warning here if this case is unexpected.
                formatted_inner[round_name] = df 
                continue

            if df.empty:
                formatted_inner[round_name] = df.copy()
                continue
            
            formatted_df = df.copy()
            
            # Determine metric type
            metric_type = None
            
            # Use user-provided map if available
            if metric_type_map and metric_name in metric_type_map:
                metric_type = metric_type_map[metric_name]
            else:
                # Automatic detection based on metric name
                metric_name_lower = metric_name.lower()
                for type_key, keywords in default_type_map.items():
                    if any(keyword in metric_name_lower for keyword in keywords):
                        metric_type = type_key
                        break
                
                # Special cases for subtypes
                if metric_type == 'disk' and any(kw in metric_name_lower for kw in ['iops', 'io_']):
                    metric_type = 'disk_iops'
            
            # If type cannot be detected, keep original value
            if not metric_type:
                formatted_inner[round_name] = formatted_df
                continue
            
            # Check if 'unit' column already exists - in that case, preserve existing values
            if 'unit' in formatted_df.columns and not formatted_df['unit'].isna().all():
                formatted_inner[round_name] = formatted_df
                continue
            
            # Calculate statistics to determine the best unit
            if 'value' in formatted_df:
                # Use 75th percentile instead of mean to better represent typical values
                typical_value = formatted_df['value'].quantile(0.75)
                if pd.isna(typical_value) or typical_value == 0:
                    # Fallback to mean if percentile is null or zero
                    typical_value = formatted_df['value'].mean()
                
                # Process only if typical_value is a valid number
                if pd.notna(typical_value):
                    converted_value, unit = convert_to_best_unit(typical_value, metric_type)
                    
                    # Define conversion_factor, ensuring converted_value is not zero and is a number
                    if pd.notna(converted_value) and converted_value != 0:
                        conversion_factor = typical_value / converted_value
                    else:
                        conversion_factor = 1.0 # Default to 1.0 if no valid conversion
                    
                    formatted_df['original_value'] = formatted_df['value'].copy()
                    
                    # Apply conversion if factor is not zero
                    if conversion_factor != 0:
                        formatted_df['value'] = formatted_df['value'] / conversion_factor
                    # Else: values remain as original_value (scaled by 1.0 effectively)
                    
                    formatted_df['unit'] = unit
                    formatted_df['metric_type'] = metric_type
                    
                    # Add readable format for each individual value
                    formatted_df['value_formatted'] = formatted_df['original_value'].apply(
                        lambda x: format_value_with_unit(x, metric_type)
                    )
                    
                    # Add conversion value for easier later manipulations
                    formatted_df['conversion_factor'] = conversion_factor
                # else: if typical_value is NaN, no formatting changes are applied to 'value', 'unit', etc.
                # formatted_df remains a copy of the original df in terms of these columns.
            
            formatted_inner[round_name] = formatted_df
        
        formatted_outer[metric_name] = formatted_inner
    
    return formatted_outer
