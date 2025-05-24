"""
Tests for data handling modules.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.normalization import (
    normalize_cpu_usage, 
    normalize_memory_usage,
    add_elapsed_time,
    clean_metrics_data
)


class TestNormalization:
    """Test data normalization functions."""
    
    def test_normalize_cpu_usage(self):
        """Test CPU usage normalization."""
        data = pd.DataFrame({
            'cpu_usage': [1.0, 2.0, 3.0],
            'cpu_requests': [0.5, 1.0, 1.5]
        })
        
        result = normalize_cpu_usage(data, total_cpu_cores=4)
        
        assert 'cpu_usage_percent' in result.columns
        assert 'cpu_requests_percent' in result.columns
        assert result['cpu_usage_percent'].iloc[0] == 25.0  # 1/4 * 100
    
    def test_normalize_memory_usage(self):
        """Test memory usage normalization."""
        data = pd.DataFrame({
            'memory_usage': [2.0, 4.0, 6.0],
            'memory_requests': [1.0, 2.0, 3.0]
        })
        
        result = normalize_memory_usage(data, total_memory_gb=8.0)
        
        assert 'memory_usage_percent' in result.columns
        assert 'memory_requests_percent' in result.columns
        assert result['memory_usage_percent'].iloc[0] == 25.0  # 2/8 * 100
    
    def test_add_elapsed_time(self):
        """Test elapsed time calculation."""
        data = pd.DataFrame({
            'round': [1, 1, 2, 2],
            'phase': ['baseline', 'baseline', 'baseline', 'baseline'],
            'datetime': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-01 10:01:00', 
                '2024-01-01 11:00:00',
                '2024-01-01 11:02:00'
            ])
        })
        
        result = add_elapsed_time(data)
        
        assert 'elapsed_seconds' in result.columns
        assert 'elapsed_minutes' in result.columns
        assert result['elapsed_seconds'].iloc[1] == 60  # 1 minute
    
    def test_clean_metrics_data(self):
        """Test data cleaning."""
        data = pd.DataFrame({
            'cpu_usage': [1.0, -1.0, 2.0],  # Negative value should be removed
            'memory_usage': [2.0, 3.0, 4.0],
            'tenant': ['A', 'B', None]  # None should be removed
        })
        
        result = clean_metrics_data(data)
        
        # Should remove row with negative CPU and row with None tenant
        assert len(result) == 1
        assert result['cpu_usage'].iloc[0] == 1.0
