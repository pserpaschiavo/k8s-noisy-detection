"""
Tests for the metric formatter module.

This module tests the intelligent unit detection and conversion functionality
implemented to fix hard-coded byte conversion issues.
"""

import pytest
import pandas as pd
import numpy as np
from src.utils.metric_formatter import (
    MetricFormatter,
    detect_and_convert_units,
    remove_hardcoded_conversions,
    format_memory_metric,
    format_throughput_metric,
    format_network_metric
)


class TestMetricFormatter:
    """Test the MetricFormatter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = MetricFormatter()
    
    def test_detect_metric_type(self):
        """Test metric type detection."""
        assert self.formatter.detect_metric_type('memory_usage') == 'memory'
        assert self.formatter.detect_metric_type('Memory_Usage') == 'memory'
        assert self.formatter.detect_metric_type('disk_throughput_total') == 'disk'
        assert self.formatter.detect_metric_type('network_total_bandwidth') == 'network'
        assert self.formatter.detect_metric_type('cpu_usage') == 'cpu'
        assert self.formatter.detect_metric_type('unknown_metric') == 'generic'
    
    def test_detect_memory_unit(self):
        """Test memory unit detection."""
        # Test with byte values
        values_bytes = pd.Series([1024, 2048, 4096])  # ~1-4 KiB
        unit, factor = self.formatter._detect_memory_unit(values_bytes.quantile(0.75))
        assert unit == 'KiB'
        assert factor == 1024
        
        # Test with megabyte values
        values_mb = pd.Series([1048576, 2097152, 4194304])  # ~1-4 MiB
        unit, factor = self.formatter._detect_memory_unit(values_mb.quantile(0.75))
        assert unit == 'MiB'
        assert factor == 1024**2
        
        # Test with gigabyte values
        values_gb = pd.Series([1073741824, 2147483648])  # ~1-2 GiB
        unit, factor = self.formatter._detect_memory_unit(values_gb.quantile(0.75))
        assert unit == 'GiB'
        assert factor == 1024**3
    
    def test_detect_throughput_unit(self):
        """Test throughput unit detection."""
        # Test with byte/s values
        values_bytes = pd.Series([1000, 2000, 4000])  # ~1-4 KB/s
        unit, factor = self.formatter._detect_throughput_unit(values_bytes.quantile(0.75))
        assert unit == 'KB/s'
        assert factor == 1000
        
        # Test with MB/s values
        values_mb = pd.Series([1000000, 2000000, 4000000])  # ~1-4 MB/s
        unit, factor = self.formatter._detect_throughput_unit(values_mb.quantile(0.75))
        assert unit == 'MB/s'
        assert factor == 1000**2
    
    def test_detect_network_unit(self):
        """Test network unit detection."""
        # Test with small values (likely bytes)
        values_small = pd.Series([1000, 2000, 4000])
        unit, factor = self.formatter._detect_network_unit(values_small.quantile(0.75))
        assert unit == 'KB/s'
        assert factor == 1000
        
        # Test with large values (likely bits)
        values_large = pd.Series([8000000, 16000000, 32000000])  # ~8-32 Mbps
        unit, factor = self.formatter._detect_network_unit(values_large.quantile(0.75))
        assert unit == 'Mbps'
        assert factor == 1000**2
    
    def test_format_dataframe_memory(self):
        """Test DataFrame formatting for memory metrics."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'value': [1048576, 2097152, 4194304, 8388608, 16777216],  # 1-16 MiB in bytes
            'pod': ['pod1', 'pod2', 'pod3', 'pod4', 'pod5']
        })
        
        result = self.formatter.format_dataframe(df, 'memory_usage')
        
        # Check that formatting was applied
        assert 'original_value' in result.columns
        assert 'original_unit' in result.columns
        assert 'display_unit' in result.columns
        assert 'formatted_value' in result.columns
        
        # Check that original values were preserved
        pd.testing.assert_series_equal(result['original_value'], df['value'])
        
        # Check that units were detected
        assert result['original_unit'].iloc[0] == 'MiB'
        assert result['display_unit'].iloc[0] == 'MiB'
        
        # Check that values were converted appropriately
        assert abs(result['value'].iloc[0] - 1.0) < 0.01  # Should be ~1 MiB
    
    def test_format_dataframe_throughput(self):
        """Test DataFrame formatting for throughput metrics."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1min'),
            'value': [1000000, 2000000, 3000000],  # 1-3 MB/s in bytes/s
            'node': ['node1', 'node2', 'node3']
        })
        
        result = self.formatter.format_dataframe(df, 'disk_throughput_total')
        
        # Check formatting
        assert 'original_value' in result.columns
        assert 'display_unit' in result.columns
        
        # Check unit detection
        assert result['original_unit'].iloc[0] == 'MB/s'
        
        # Check conversion
        assert abs(result['value'].iloc[0] - 1.0) < 0.01  # Should be ~1 MB/s
    
    def test_format_dataframe_network(self):
        """Test DataFrame formatting for network metrics."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1min'),
            'value': [8000000, 16000000, 24000000],  # 8-24 Mbps
            'interface': ['eth0', 'eth1', 'eth2']
        })
        
        result = self.formatter.format_dataframe(df, 'network_total_bandwidth')
        
        # Check formatting
        assert 'original_value' in result.columns
        assert 'display_unit' in result.columns
        
        # Check that large values are treated as bits
        assert 'bps' in result['original_unit'].iloc[0] or 'Mbps' in result['original_unit'].iloc[0]


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_detect_and_convert_units(self):
        """Test the detect_and_convert_units utility function."""
        df = pd.DataFrame({
            'value': [1024, 2048, 4096],
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1min')
        })
        
        result = detect_and_convert_units(df, 'memory_usage')
        
        assert 'original_value' in result.columns
        assert 'display_unit' in result.columns
        assert len(result) == len(df)
    
    def test_remove_hardcoded_conversions(self):
        """Test removal of hard-coded conversions."""
        df = pd.DataFrame({
            'value': [1.0, 2.0, 3.0],  # Values that might have been incorrectly converted
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1min')
        })
        
        result = remove_hardcoded_conversions(df, 'memory_usage')
        
        # Should add warning flags for problematic metrics
        assert 'potential_conversion_issue' in result.columns
        assert 'conversion_warning' in result.columns
        assert result['potential_conversion_issue'].all()
    
    def test_format_memory_metric(self):
        """Test memory-specific formatting function."""
        df = pd.DataFrame({
            'value': [1073741824, 2147483648],  # 1-2 GiB
            'timestamp': pd.date_range('2024-01-01', periods=2, freq='1min')
        })
        
        result = format_memory_metric(df)
        
        assert 'display_unit' in result.columns
        assert 'GiB' in result['display_unit'].iloc[0]
    
    def test_format_throughput_metric(self):
        """Test throughput-specific formatting function."""
        df = pd.DataFrame({
            'value': [1000000, 2000000],  # 1-2 MB/s
            'timestamp': pd.date_range('2024-01-01', periods=2, freq='1min')
        })
        
        result = format_throughput_metric(df)
        
        assert 'display_unit' in result.columns
        assert 'MB/s' in result['display_unit'].iloc[0]
    
    def test_format_network_metric(self):
        """Test network-specific formatting function."""
        df = pd.DataFrame({
            'value': [8000000, 16000000],  # 8-16 Mbps
            'timestamp': pd.date_range('2024-01-01', periods=2, freq='1min')
        })
        
        result = format_network_metric(df)
        
        assert 'display_unit' in result.columns


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = MetricFormatter()
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        result = self.formatter.format_dataframe(df, 'test_metric')
        assert result.empty
    
    def test_dataframe_without_value_column(self):
        """Test handling of DataFrame without value column."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1min'),
            'data': [1, 2, 3]
        })
        
        result = self.formatter.format_dataframe(df, 'test_metric')
        # Should return original DataFrame unchanged
        pd.testing.assert_frame_equal(result, df)
    
    def test_values_with_nulls_and_zeros(self):
        """Test handling of null and zero values."""
        df = pd.DataFrame({
            'value': [0, np.nan, 1024, 2048, 0, np.nan, 4096],
            'timestamp': pd.date_range('2024-01-01', periods=7, freq='1min')
        })
        
        result = self.formatter.format_dataframe(df, 'memory_usage')
        
        # Should still work with non-null, non-zero values
        assert 'display_unit' in result.columns
        assert len(result) == len(df)
    
    def test_all_zero_values(self):
        """Test handling of all zero values."""
        df = pd.DataFrame({
            'value': [0, 0, 0],
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1min')
        })
        
        result = self.formatter.format_dataframe(df, 'test_metric')
        
        # Should handle gracefully
        assert len(result) == len(df)
        assert 'unknown' in result['original_unit'].iloc[0]
    
    def test_format_multiple_metrics(self):
        """Test formatting multiple metrics at once."""
        metrics_dict = {
            'memory_usage': pd.DataFrame({
                'value': [1048576, 2097152],
                'timestamp': pd.date_range('2024-01-01', periods=2, freq='1min')
            }),
            'disk_throughput': pd.DataFrame({
                'value': [1000000, 2000000],
                'timestamp': pd.date_range('2024-01-01', periods=2, freq='1min')
            })
        }
        
        result = self.formatter.format_multiple_metrics(metrics_dict)
        
        assert len(result) == 2
        assert 'memory_usage' in result
        assert 'disk_throughput' in result
        assert 'display_unit' in result['memory_usage'].columns
        assert 'display_unit' in result['disk_throughput'].columns


if __name__ == '__main__':
    pytest.main([__file__])
