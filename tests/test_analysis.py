"""
Tests for analysis modules.
"""

import pytest
import pandas as pd
import numpy as np
from src.analysis.descriptive_statistics import calculate_descriptive_statistics
from src.analysis.correlation_covariance import calculate_inter_tenant_correlation_per_metric


class TestDescriptiveStatistics:
    """Test descriptive statistics calculations."""
    
    def test_calculate_descriptive_statistics(self):
        """Test basic descriptive statistics calculation."""
        # Create sample data
        data = pd.DataFrame({
            'tenant': ['A', 'A', 'B', 'B'] * 10,
            'cpu_usage': np.random.rand(40),
            'memory_usage': np.random.rand(40)
        })
        
        result = calculate_descriptive_statistics(data, ['cpu_usage', 'memory_usage'])
        
        assert isinstance(result, dict)
        assert 'cpu_usage' in result
        assert 'memory_usage' in result


class TestCorrelationCovariance:
    """Test correlation and covariance calculations."""
    
    def test_inter_tenant_correlation(self):
        """Test inter-tenant correlation calculation."""
        # Create sample data with known correlation
        data = pd.DataFrame({
            'tenant': ['A'] * 20 + ['B'] * 20,
            'cpu_usage': list(range(20)) + list(range(20, 40)),
            'timestamp': pd.date_range('2024-01-01', periods=40, freq='1min')
        })
        
        result = calculate_inter_tenant_correlation_per_metric(data, ['cpu_usage'])
        
        assert isinstance(result, dict)
        assert 'cpu_usage' in result
