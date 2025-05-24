"""
Simplified metric formatter for testing and validation.

This module provides intelligent unit detection and conversion functionality
to replace hard-coded byte conversions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logger = logging.getLogger(__name__)


class MetricFormatter:
    """Simplified metric formatter with intelligent unit detection."""
    
    def __init__(self):
        """Initialize the formatter with unit definitions."""
        # Binary units (1024-based) for memory
        self.memory_units = [
            ('B', 1),
            ('KiB', 1024),
            ('MiB', 1024**2),
            ('GiB', 1024**3),
            ('TiB', 1024**4),
        ]
        
        # Decimal units (1000-based) for throughput
        self.throughput_units = [
            ('B/s', 1),
            ('KB/s', 1000),
            ('MB/s', 1000**2),
            ('GB/s', 1000**3),
        ]
        
        # Metric type patterns
        self.metric_patterns = {
            'memory': ['memory', 'mem', 'ram'],
            'disk': ['disk', 'throughput', 'io'],
            'network': ['network', 'bandwidth', 'rx', 'tx'],
            'cpu': ['cpu', 'processor'],
        }
    
    def detect_metric_type(self, metric_name: str) -> str:
        """Detect metric type based on name."""
        name_lower = metric_name.lower()
        
        for metric_type, patterns in self.metric_patterns.items():
            if any(pattern in name_lower for pattern in patterns):
                return metric_type
        
        return 'generic'
    
    def detect_best_unit(self, values: pd.Series, metric_type: str) -> Tuple[str, float]:
        """Detect the best unit for display based on data magnitude."""
        # Remove nulls and zeros
        clean_values = values.dropna()
        clean_values = clean_values[clean_values > 0]
        
        if len(clean_values) == 0:
            return 'unknown', 1.0
        
        # Use 75th percentile as reference value
        ref_value = clean_values.quantile(0.75)
        
        if metric_type == 'memory':
            units = self.memory_units
        elif metric_type in ['disk', 'network']:
            units = self.throughput_units
        else:
            # Generic decimal units
            units = [('', 1), ('K', 1000), ('M', 1000000), ('G', 1000000000)]
        
        # Find best unit (largest unit where ref_value >= 1)
        for unit, factor in reversed(units):
            if ref_value >= factor:
                return unit, factor
        
        return units[0][0], units[0][1]
    
    def format_dataframe(self, df: pd.DataFrame, metric_name: str, 
                        value_col: str = 'value') -> pd.DataFrame:
        """Format DataFrame with intelligent unit detection."""
        if df.empty or value_col not in df.columns:
            return df.copy()
        
        result_df = df.copy()
        
        # Detect metric type
        metric_type = self.detect_metric_type(metric_name)
        
        # Detect best unit
        best_unit, conversion_factor = self.detect_best_unit(result_df[value_col], metric_type)
        
        # Preserve original values
        result_df['original_value'] = result_df[value_col].copy()
        result_df['original_unit'] = 'detected_bytes'
        
        # Convert to best unit
        result_df[value_col] = result_df[value_col] / conversion_factor
        result_df['display_unit'] = best_unit
        
        # Add formatted strings
        result_df['formatted_value'] = result_df.apply(
            lambda row: self._format_value(row[value_col], row['display_unit']),
            axis=1
        )
        
        # Add metadata
        result_df['metric_type'] = metric_type
        result_df['conversion_factor'] = conversion_factor
        
        logger.info(f"Formatted metric '{metric_name}' as {metric_type}, unit: {best_unit}")
        
        return result_df
    
    def _format_value(self, value: float, unit: str) -> str:
        """Format value with appropriate precision."""
        if pd.isna(value):
            return 'N/A'
        
        if abs(value) >= 100:
            precision = 1
        elif abs(value) >= 10:
            precision = 2
        else:
            precision = 3
        
        return f"{value:.{precision}f} {unit}"


def detect_and_convert_units(df: pd.DataFrame, metric_name: str, 
                           value_col: str = 'value') -> pd.DataFrame:
    """Utility function for unit detection and conversion."""
    formatter = MetricFormatter()
    return formatter.format_dataframe(df, metric_name, value_col)


def format_memory_metric(df: pd.DataFrame, metric_name: str = "memory_usage") -> pd.DataFrame:
    """Format memory metrics specifically."""
    return detect_and_convert_units(df, metric_name)


def format_throughput_metric(df: pd.DataFrame, metric_name: str = "throughput") -> pd.DataFrame:
    """Format throughput metrics specifically."""
    return detect_and_convert_units(df, metric_name)


def format_network_metric(df: pd.DataFrame, metric_name: str = "network_bandwidth") -> pd.DataFrame:
    """Format network metrics specifically."""
    return detect_and_convert_units(df, metric_name)
