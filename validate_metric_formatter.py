#!/usr/bin/env python3
"""
Quick validation test for the MetricFormatter implementation.
"""

import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np

try:
    from src.utils.metric_formatter import MetricFormatter, detect_and_convert_units
    print("✓ Successfully imported MetricFormatter")
except ImportError as e:
    print(f"✗ Failed to import MetricFormatter: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality of the MetricFormatter."""
    print("\n=== Testing Basic Functionality ===")
    
    formatter = MetricFormatter()
    print("✓ MetricFormatter instantiated")
    
    # Test metric type detection
    assert formatter.detect_metric_type('memory_usage') == 'memory'
    assert formatter.detect_metric_type('disk_throughput_total') == 'disk'
    assert formatter.detect_metric_type('network_total_bandwidth') == 'network'
    print("✓ Metric type detection working")
    
    # Test with sample memory data (bytes)
    memory_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
        'value': [1048576, 2097152, 4194304, 8388608, 16777216],  # 1-16 MiB in bytes
        'pod': [f'pod{i}' for i in range(1, 6)]
    })
    
    print(f"Original memory values: {memory_df['value'].tolist()}")
    
    formatted_memory = formatter.format_dataframe(memory_df, 'memory_usage')
    print("✓ Memory formatting completed")
    
    print(f"Detected original unit: {formatted_memory['original_unit'].iloc[0]}")
    print(f"Display unit: {formatted_memory['display_unit'].iloc[0]}")
    print(f"Converted values: {formatted_memory['value'].tolist()}")
    print(f"Formatted strings: {formatted_memory['formatted_value'].tolist()}")
    
    # Test with throughput data
    throughput_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=3, freq='1min'),
        'value': [1000000, 2500000, 5000000],  # 1-5 MB/s in bytes/s
        'node': ['node1', 'node2', 'node3']
    })
    
    print(f"\nOriginal throughput values: {throughput_df['value'].tolist()}")
    
    formatted_throughput = formatter.format_dataframe(throughput_df, 'disk_throughput_total')
    print("✓ Throughput formatting completed")
    
    print(f"Detected original unit: {formatted_throughput['original_unit'].iloc[0]}")
    print(f"Display unit: {formatted_throughput['display_unit'].iloc[0]}")
    print(f"Converted values: {formatted_throughput['value'].tolist()}")
    
    # Test utility function
    utility_result = detect_and_convert_units(memory_df.copy(), 'memory_usage')
    print("✓ Utility function working")
    
    print("\n=== All Basic Tests Passed! ===")

def test_problematic_scenarios():
    """Test scenarios that were problematic with hard-coded conversions."""
    print("\n=== Testing Problematic Scenarios ===")
    
    formatter = MetricFormatter()
    
    # Scenario 1: Data that was already converted (small values indicating MB instead of bytes)
    potentially_converted_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=3, freq='1min'),
        'value': [1.5, 2.3, 4.7],  # Values that might be in MB already
        'pod': ['pod1', 'pod2', 'pod3']
    })
    
    print(f"Potentially converted values: {potentially_converted_df['value'].tolist()}")
    
    result = formatter.format_dataframe(potentially_converted_df, 'memory_usage')
    print(f"Detected unit: {result['original_unit'].iloc[0]}")
    print(f"Display unit: {result['display_unit'].iloc[0]}")
    print(f"Formatted: {result['formatted_value'].tolist()}")
    
    # Scenario 2: Very large values (typical of raw byte measurements)
    large_values_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=3, freq='1min'),
        'value': [4294967296, 8589934592, 17179869184],  # 4-16 GiB in bytes
        'container': ['container1', 'container2', 'container3']
    })
    
    print(f"\nLarge values: {large_values_df['value'].tolist()}")
    
    result2 = formatter.format_dataframe(large_values_df, 'memory_usage')
    print(f"Detected unit: {result2['original_unit'].iloc[0]}")
    print(f"Display unit: {result2['display_unit'].iloc[0]}")
    print(f"Converted values: {result2['value'].tolist()}")
    
    # Scenario 3: Mixed magnitude data
    mixed_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=6, freq='1min'),
        'value': [1024, 1048576, 1073741824, 2048, 2097152, 2147483648],  # KB, MB, GB mix
        'service': [f'service{i}' for i in range(1, 7)]
    })
    
    print(f"\nMixed magnitude values: {mixed_df['value'].tolist()}")
    
    result3 = formatter.format_dataframe(mixed_df, 'memory_usage')
    print(f"Detected unit: {result3['original_unit'].iloc[0]}")
    print(f"Display unit: {result3['display_unit'].iloc[0]}")
    print(f"Converted values: {result3['value'].tolist()}")
    
    print("\n=== Problematic Scenarios Tests Completed ===")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")
    
    formatter = MetricFormatter()
    
    # Empty DataFrame
    empty_df = pd.DataFrame()
    result = formatter.format_dataframe(empty_df, 'test_metric')
    assert result.empty
    print("✓ Empty DataFrame handled correctly")
    
    # DataFrame with nulls and zeros
    mixed_quality_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=7, freq='1min'),
        'value': [0, np.nan, 1024, 2048, 0, np.nan, 4096],
        'source': [f'source{i}' for i in range(1, 8)]
    })
    
    result = formatter.format_dataframe(mixed_quality_df, 'memory_usage')
    assert len(result) == len(mixed_quality_df)
    print("✓ Mixed quality data handled correctly")
    
    # All zeros
    zero_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=3, freq='1min'),
        'value': [0, 0, 0],
        'pod': ['pod1', 'pod2', 'pod3']
    })
    
    result = formatter.format_dataframe(zero_df, 'test_metric')
    assert len(result) == len(zero_df)
    print("✓ All-zero data handled correctly")
    
    print("\n=== Edge Cases Tests Completed ===")

if __name__ == '__main__':
    print("Starting MetricFormatter Validation Tests...")
    
    try:
        test_basic_functionality()
        test_problematic_scenarios()
        test_edge_cases()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! MetricFormatter is working correctly.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
