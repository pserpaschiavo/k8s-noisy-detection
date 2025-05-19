"""
Module to parse Kubernetes Resource Quota manifests.

This module provides functions to extract resource information from quota
manifests and use it for proper metric normalization.
"""

import os
import yaml
import re
from typing import Dict, Optional, Any, Tuple, Union


def parse_quantity(quantity_str: str) -> float:
    """
    Converts a Kubernetes-style quantity string to a numerical value.
    
    Args:
        quantity_str: Quantity string (e.g., "100m", "2Gi", "500Mi")
        
    Returns:
        float: Numerical value converted to the base unit
    """
    if not quantity_str or not isinstance(quantity_str, str):
        return 0.0
    
    # CPU can be specified in millicores with "m" suffix
    if quantity_str.endswith('m'):
        # Convert millicores to cores (divide by 1000)
        return float(quantity_str[:-1]) / 1000

    # Memory and storage with unit suffixes
    suffixes = {
        'Ei': 2**60, 'Pi': 2**50, 'Ti': 2**40, 'Gi': 2**30, 'Mi': 2**20, 'Ki': 2**10,  # Binary
        'E': 10**18, 'P': 10**15, 'T': 10**12, 'G': 10**9, 'M': 10**6, 'K': 10**3      # Decimal
    }
    
    for suffix, multiplier in suffixes.items():
        if quantity_str.endswith(suffix):
            return float(quantity_str[:-len(suffix)]) * multiplier
            
    # No suffix, try to convert directly
    try:
        return float(quantity_str)
    except ValueError:
        return 0.0


def get_best_unit_for_value(value: float, metric_type: str) -> Tuple[float, str]:
    """
    Determines the best unit to represent a value based on the metric type.
    
    Args:
        value: Numerical value to convert
        metric_type: Type of metric ('cpu', 'memory', 'disk', 'network')
        
    Returns:
        Tuple (converted_value, unit)
    """
    if value is None or not isinstance(value, (int, float)):
        return 0.0, ''
    
    if metric_type == 'cpu':
        if value < 0.01:
            # Very small values in millicores
            return value * 1000, 'millicores'
        elif value < 0.1:
            # Less than 0.1 cores = show in millicores
            return value * 1000, 'm'
        else:
            # Show in cores
            return value, 'cores'
    
    elif metric_type == 'memory':
        if value < 2**10:  # < 1 KiB
            return value, 'B'
        elif value < 2**20:  # < 1 MiB
            return value / (2**10), 'KiB'
        elif value < 2**30:  # < 1 GiB
            return value / (2**20), 'MiB'
        else:
            return value / (2**30), 'GiB'
    
    elif metric_type in ['disk', 'storage']:
        # For storage, use binary units (KiB, MiB, GiB)
        if value < 2**10:  # < 1 KiB
            return value, 'B'
        elif value < 2**20:  # < 1 MiB
            return value / (2**10), 'KiB'
        elif value < 2**30:  # < 1 GiB
            return value / (2**20), 'MiB'
        else:
            return value / (2**30), 'GiB'
    
    elif metric_type in ['disk_iops', 'io']:
        # For IOPS, no specific unit beyond operations/s
        if value < 1000:
            return value, 'IOPS'
        else:
            return value / 1000, 'kIOPS'
    
    elif metric_type in ['network', 'bandwidth']:
        # For transfer rates, use decimal units (KB/s, MB/s, GB/s)
        if value < 1024:  # < 1 KB/s
            return value, 'B/s'
        elif value < 1024**2:  # < 1 MB/s
            return value / 1024, 'KB/s'
        elif value < 1024**3:  # < 1 GB/s
            return value / (1024**2), 'MB/s'
        else:
            return value / (1024**3), 'GB/s'
    
    # Generic case, just return the value without a unit
    return value, ''


def format_value_with_unit(value: float, metric_type: str = None, 
                          custom_format: str = '{:.2f} {}',
                          auto_detect_unit: bool = True,
                          preserve_small_values: bool = True) -> str:
    """
    Formats a numerical value with the appropriate unit.
    
    Args:
        value: Numerical value to format
        metric_type: Type of metric ('cpu', 'memory', 'disk', 'network')
        custom_format: Custom format string
        auto_detect_unit: If True, automatically detects the best unit
        preserve_small_values: If True, small values are not rounded to zero
        
    Returns:
        Formatted string with value and unit
    """
    if value is None:
        return "N/A"
    
    if metric_type is None:
        return custom_format.format(value, '')
    
    if auto_detect_unit:
        converted_value, unit = get_best_unit_for_value(value, metric_type)
        
        # For very small values, use scientific notation to avoid rounding to zero
        if preserve_small_values and converted_value < 0.01 and converted_value > 0:
            return f"{converted_value:.2e} {unit}"
        
        return custom_format.format(converted_value, unit)
    
    # Specific formats for metric types without auto-detection
    if metric_type == 'cpu':
        # Raw value in cores, direct formatting
        if value < 0.01:
            return f"{value * 1000:.2f} millicores"
        return custom_format.format(value, 'cores')
    
    elif metric_type == 'memory':
        # Raw value in bytes, direct formatting
        return custom_format.format(value / (2**30), 'GiB')
    
    elif metric_type in ['disk', 'network']:
        # Raw value, direct formatting
        return custom_format.format(value / (2**20), 'MiB')
    
    # Default for other types
    return custom_format.format(value, metric_type)


def convert_to_best_unit(value: float, metric_type: str) -> Tuple[float, str]:
    """
    Converts a value to the most readable unit based on the metric type.
    
    Args:
        value: Numerical value to convert
        metric_type: Type of metric ('cpu', 'memory', 'disk', 'network')
        
    Returns:
        Tuple (converted_value, unit)
    """
    return get_best_unit_for_value(value, metric_type)


def load_resource_quotas(yaml_file: str) -> Dict[str, Dict[str, float]]:
    """
    Loads resource quotas from a YAML file.
    
    Args:
        yaml_file: Path to the YAML file containing ResourceQuotas
        
    Returns:
        Dict: Dictionary mapping namespaces to their resource limits
    """
    quotas = {}
    
    try:
        with open(yaml_file, 'r') as f:
            # The YAML file may contain multiple documents separated by "---"
            documents = yaml.safe_load_all(f)
            
            for doc in documents:
                if not doc:
                    continue
                    
                # Check if it is a ResourceQuota
                if doc.get('kind') == 'ResourceQuota':
                    namespace = doc.get('metadata', {}).get('namespace', 'default')
                    
                    # Extract limits
                    hard_limits = doc.get('spec', {}).get('hard', {})
                    
                    if namespace not in quotas:
                        quotas[namespace] = {}
                    
                    # Process CPU
                    if 'limits.cpu' in hard_limits:
                        quotas[namespace]['cpu_limit'] = parse_quantity(hard_limits['limits.cpu'])
                    if 'requests.cpu' in hard_limits:
                        quotas[namespace]['cpu_request'] = parse_quantity(hard_limits['requests.cpu'])
                        
                    # Process Memory
                    if 'limits.memory' in hard_limits:
                        quotas[namespace]['memory_limit'] = parse_quantity(hard_limits['limits.memory'])
                    if 'requests.memory' in hard_limits:
                        quotas[namespace]['memory_request'] = parse_quantity(hard_limits['requests.memory'])
                    
        return quotas
    except Exception as e:
        print(f"Error loading quotas: {e}")
        return {}


def get_tenant_quotas(quota_file: str = None) -> Dict[str, Dict[str, float]]:
    """
    Retrieves resource quotas for each tenant.
    
    Args:
        quota_file: Path to the quota file (optional)
        
    Returns:
        Dict: Dictionary mapping tenants to their resource limits
    """
    if not quota_file:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        quota_file = os.path.join(base_dir, 'resource-quotas.yaml')
    
    if not os.path.exists(quota_file):
        print(f"Quota file not found: {quota_file}")
        return {}
    
    namespace_quotas = load_resource_quotas(quota_file)
    tenant_quotas = {}
    
    # Convert from namespaces to tenant-X
    for namespace, quotas in namespace_quotas.items():
        tenant_quotas[namespace] = quotas
    
    return tenant_quotas


def get_formatted_quota_values(namespace: str, quota_file: str = None, 
                              include_requests: bool = False) -> Dict[str, str]:
    """
    Retrieves formatted quota values for a specific namespace.
    
    Args:
        namespace: The namespace to retrieve quotas for
        quota_file: Path to the quota file (optional)
        include_requests: If True, also includes 'requests' values
        
    Returns:
        Dict: Dictionary with values formatted in readable units
    """
    quotas = get_tenant_quotas(quota_file)
    if namespace not in quotas:
        return {}
    
    tenant_quota = quotas[namespace]
    formatted = {}
    
    # Format CPU
    if 'cpu_limit' in tenant_quota:
        cpu_value, cpu_unit = convert_to_best_unit(tenant_quota['cpu_limit'], 'cpu')
        formatted['cpu_limit'] = f"{cpu_value:.2f} {cpu_unit}"
    
    # Format Memory
    if 'memory_limit' in tenant_quota:
        mem_value, mem_unit = convert_to_best_unit(tenant_quota['memory_limit'], 'memory')
        formatted['memory_limit'] = f"{mem_value:.2f} {mem_unit}"
    
    # Include requests if requested
    if include_requests:
        if 'cpu_request' in tenant_quota:
            req_cpu_value, req_cpu_unit = convert_to_best_unit(tenant_quota['cpu_request'], 'cpu')
            formatted['cpu_request'] = f"{req_cpu_value:.2f} {req_cpu_unit}"
        
        if 'memory_request' in tenant_quota:
            req_mem_value, req_mem_unit = convert_to_best_unit(tenant_quota['memory_request'], 'memory')
            formatted['memory_request'] = f"{req_mem_value:.2f} {req_mem_unit}"
    
    return formatted


def create_node_config_from_quotas(quota_file: str = None) -> Dict[str, float]:
    """
    Creates a node configuration based on total quotas.
    This function sums all resource quotas and adds margins to estimate
    the total node capacity.
    
    Args:
        quota_file: Path to the quota file (optional)
        
    Returns:
        Dict: Node configuration with total resource capacities
    """
    tenant_quotas = get_tenant_quotas(quota_file)
    
    # Calculate totals by summing all resource limits
    total_cpu_limit = sum(quota.get('cpu_limit', 0) for quota in tenant_quotas.values())
    total_memory_limit = sum(quota.get('memory_limit', 0) for quota in tenant_quotas.values())
    
    # Add margin for system resources (usually 10-20%)
    # This reflects that a node reserves resources for the system and other components
    system_margin = 0.2  # 20% margin
    estimated_node_cpu = total_cpu_limit / (1 - system_margin)
    estimated_node_memory = total_memory_limit / (1 - system_margin)
    
    # Convert memory to different units
    memory_bytes = estimated_node_memory
    memory_kb = memory_bytes / (2**10)
    memory_mb = memory_bytes / (2**20)
    memory_gb = memory_bytes / (2**30)
    
    # Estimate storage metrics
    # General rule: ~10x total memory for permanent storage, ~2-4x for scratch
    disk_size_bytes = memory_bytes * 10
    disk_size_gb = disk_size_bytes / (2**30)
    
    # For I/O, estimate based on the number of cores
    # General rule: ~100 IOPS per core for normal workloads
    disk_iops = max(500, total_cpu_limit * 100)
    disk_bandwidth_mbps = max(100, disk_iops * 0.256)  # ~256 KB per I/O operation
    
    # Estimate network bandwidth based on CPU and memory
    # General rule: ~125-250 Mbps per core for normal workloads
    bandwidth_factor = 250  # Mbps per core
    network_bandwidth_mbps = max(1000, total_cpu_limit * bandwidth_factor)
    
    # Build the node configuration with all values
    node_config = {
        # CPU resources
        'CPUS': estimated_node_cpu,
        'TOTAL_CPU_CORES': estimated_node_cpu,
        'CPU_CORES_PER_TENANT': total_cpu_limit / len(tenant_quotas) if tenant_quotas else 0,
        
        # Memory resources
        'MEMORY_BYTES': memory_bytes,
        'MEMORY_KB': memory_kb,
        'MEMORY_MB': memory_mb,
        'MEMORY_GB': memory_gb,
        
        # Storage resources
        'DISK_SIZE_BYTES': disk_size_bytes,
        'DISK_SIZE_GB': disk_size_gb,
        'DISK_IOPS': disk_iops,
        'DISK_BANDWIDTH_MBPS': disk_bandwidth_mbps,
        
        # Network resources
        'NETWORK_BANDWIDTH_MBPS': network_bandwidth_mbps,
        'NETWORK_BANDWIDTH_GBPS': network_bandwidth_mbps / 1000,
        
        # Metadata
        'TENANT_COUNT': len(tenant_quotas),
        'GENERATED_FROM_QUOTAS': True,
        'SYSTEM_RESOURCES_MARGIN': system_margin * 100  # in percentage
    }
    
    return node_config


def get_quota_summary(quota_file: str = None, include_requests: bool = False,
                   calculate_percentages: bool = True,
                   use_markdown: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Generates a summary of all quotas with unit formatting for readability.
    
    Args:
        quota_file: Path to the quota file (optional)
        include_requests: If True, also includes 'requests' values
        calculate_percentages: If True, calculates the proportion of resources for each tenant
        use_markdown: If True, formats the summary for display in markdown
        
    Returns:
        Dict: Formatted summary of quotas
    """
    quotas = get_tenant_quotas(quota_file)
    summary = {}
    
    # Calculate totals for percentages
    total_cpu_limit = sum(quota.get('cpu_limit', 0) for quota in quotas.values())
    total_memory_limit = sum(quota.get('memory_limit', 0) for quota in quotas.values())
    
    for namespace, quota in quotas.items():
        namespace_summary = get_formatted_quota_values(namespace, quota_file, include_requests)
        
        # Add raw values for calculations
        raw_values = {}
        if 'cpu_limit' in quota:
            raw_values['cpu_limit_raw'] = quota['cpu_limit']
        if 'memory_limit' in quota:
            raw_values['memory_limit_raw'] = quota['memory_limit']
        
        if include_requests:
            if 'cpu_request' in quota:
                raw_values['cpu_request_raw'] = quota['cpu_request']
            if 'memory_request' in quota:
                raw_values['memory_request_raw'] = quota['memory_request']
        
        # Add percentages relative to the total when requested
        if calculate_percentages:
            if 'cpu_limit' in quota and total_cpu_limit > 0:
                cpu_percent = (quota['cpu_limit'] / total_cpu_limit) * 100
                namespace_summary['cpu_percent'] = f"{cpu_percent:.1f}%"
                raw_values['cpu_percent_raw'] = cpu_percent
            
            if 'memory_limit' in quota and total_memory_limit > 0:
                memory_percent = (quota['memory_limit'] / total_memory_limit) * 100
                namespace_summary['memory_percent'] = f"{memory_percent:.1f}%"
                raw_values['memory_percent_raw'] = memory_percent
        
        # Add proportion between request and limit
        if include_requests:
            if 'cpu_request' in quota and 'cpu_limit' in quota and quota['cpu_limit'] > 0:
                req_pct = (quota['cpu_request'] / quota['cpu_limit']) * 100
                namespace_summary['cpu_req_vs_limit'] = f"{req_pct:.0f}%"
                
            if 'memory_request' in quota and 'memory_limit' in quota and quota['memory_limit'] > 0:
                req_pct = (quota['memory_request'] / quota['memory_limit']) * 100
                namespace_summary['memory_req_vs_limit'] = f"{req_pct:.0f}%"
        
        # Format for markdown if requested
        if use_markdown:
            format_dict = {}
            for key, value in namespace_summary.items():
                if key.endswith('_percent'):
                    # Highlight percentages of the total
                    format_dict[key] = f"**{value}**"
                elif key.endswith('_req_vs_limit'):
                    # Highlight request/limit proportions
                    format_dict[key] = f"*{value}*"
                else:
                    format_dict[key] = value
            namespace_summary.update(format_dict)
            
        # Merge raw values into the summary
        namespace_summary.update(raw_values)
        summary[namespace] = namespace_summary
        
    # Add totals to the summary
    if total_cpu_limit > 0 or total_memory_limit > 0:
        totals = {}
        
        if total_cpu_limit > 0:
            cpu_value, cpu_unit = convert_to_best_unit(total_cpu_limit, 'cpu')
            totals['cpu_limit'] = f"{cpu_value:.2f} {cpu_unit}"
            totals['cpu_limit_raw'] = total_cpu_limit
        
        if total_memory_limit > 0:
            mem_value, mem_unit = convert_to_best_unit(total_memory_limit, 'memory')
            totals['memory_limit'] = f"{mem_value:.2f} {mem_unit}"
            totals['memory_limit_raw'] = total_memory_limit
        
        # Format for markdown if requested
        if use_markdown:
            for key, value in totals.items():
                if isinstance(value, str) and not key.endswith('_raw'):
                    totals[key] = f"**{value}**"
        
        summary['__total__'] = totals
    
    return summary
