"""
Tests for visualization modules.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.plots import (
    plot_correlation_heatmap,
    plot_descriptive_stats_boxplot
)


class TestVisualization:
    """Test visualization functions."""
    
    def test_plot_correlation_heatmap(self):
        """Test correlation heatmap plotting."""
        # Create sample correlation matrix
        correlation_data = pd.DataFrame(
            np.random.rand(5, 5),
            columns=['A', 'B', 'C', 'D', 'E'],
            index=['A', 'B', 'C', 'D', 'E']
        )
        
        # Should not raise any exceptions
        fig, ax = plot_correlation_heatmap(
            correlation_data, 
            title="Test Correlation",
            output_dir=None  # Don't save to file
        )
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_plot_descriptive_stats_boxplot(self):
        """Test descriptive statistics boxplot."""
        # Create sample data
        data = pd.DataFrame({
            'tenant': ['A', 'A', 'B', 'B'] * 10,
            'cpu_usage': np.random.rand(40),
            'phase': ['baseline'] * 40
        })
        
        # Should not raise any exceptions
        fig, ax = plot_descriptive_stats_boxplot(
            data,
            metric='cpu_usage',
            title="Test Boxplot",
            output_dir=None  # Don't save to file
        )
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
