#!/usr/bin/env python3
import sys, os
# Ensure project root is on sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
Test script to verify that the fixes in new_main.py work correctly.
This script creates sample data and tests the PCA/ICA analysis functions.
"""

import numpy as np
import pandas as pd
import os
import tempfile

from analysis_modules.multivariate_exploration import perform_pca, perform_ica
from visualization.new_plots import (
    plot_pca_explained_variance, plot_pca_biplot, plot_pca_loadings_heatmap,
    plot_ica_components_heatmap, plot_ica_scatter
)

def test_pca_ica_workflow():
    """Test the fixed PCA/ICA workflow."""
    print("Creating sample data...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 8
    
    # Generate some correlated features
    base_signal = np.random.randn(n_samples)
    noise = np.random.randn(n_samples, n_features) * 0.3
    
    data = {
        'cpu_usage': base_signal + noise[:, 0],
        'memory_usage': base_signal * 0.8 + noise[:, 1],
        'network_io': base_signal * -0.5 + noise[:, 2],
        'disk_io': np.random.randn(n_samples) + noise[:, 3],
        'response_time': base_signal * 1.2 + noise[:, 4],
        'error_rate': np.random.randn(n_samples) * 0.5 + noise[:, 5],
        'throughput': base_signal * -0.9 + noise[:, 6],
        'latency': base_signal * 0.7 + noise[:, 7]
    }
    
    sample_df = pd.DataFrame(data)
    print(f"Sample data shape: {sample_df.shape}")
    print("Sample data:")
    print(sample_df.head())
    print()
    
    # Test PCA
    print("Testing PCA...")
    try:
        pca_results_df, pca_components_df, pca_explained_variance = perform_pca(
            sample_df.copy(), 
            variance_threshold=0.85
        )
        # Ensure explained variance is available for plotting
        assert pca_explained_variance is not None, "Expected explained variance array, got None"
        print(f"PCA Results shape: {pca_results_df.shape}")
        print(f"PCA Components shape: {pca_components_df.shape}")
        print(f"PCA Explained variance: {pca_explained_variance}")
        print("PCA test passed!")
        print()
    except Exception as e:
        print(f"PCA test failed: {e}")
        return False
    
    # Test ICA
    print("Testing ICA...")
    try:
        ica_results_df, ica_components_df, ica_mixing_df = perform_ica(
            sample_df.copy(), 
            n_components=4
        )
        print(f"ICA Results shape: {ica_results_df.shape}")
        print(f"ICA Components shape: {ica_components_df.shape}")
        print(f"ICA Mixing shape: {ica_mixing_df.shape}")
        print("ICA test passed!")
        print()
    except Exception as e:
        print(f"ICA test failed: {e}")
        return False
    
    # Test visualization functions
    print("Testing visualization functions...")
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Test PCA visualizations
            print("  Testing PCA visualizations...")
            plot_pca_explained_variance(
                pca_explained_variance,
                title="Test PCA Explained Variance",
                output_dir=temp_dir,
                filename="test_pca_explained_variance.png",
                metric_name="test_metric"
            )
            
            plot_pca_biplot(
                pca_results_df, pca_components_df,
                title="Test PCA Biplot",
                output_dir=temp_dir,
                filename="test_pca_biplot.png",
                metric_name="test_metric"
            )
            
            plot_pca_loadings_heatmap(
                pca_components_df,
                title="Test PCA Loadings Heatmap",
                output_dir=temp_dir,
                filename="test_pca_loadings.png",
                metric_name="test_metric"
            )
            print("    PCA visualizations passed!")
            
            # Test ICA visualizations
            print("  Testing ICA visualizations...")
            plot_ica_scatter(
                ica_results_df,
                title="Test ICA Scatter Plot",
                output_dir=temp_dir,
                filename="test_ica_scatter.png",
                metric_name="test_metric"
            )
            
            plot_ica_components_heatmap(
                ica_components_df,
                title="Test ICA Components Heatmap",
                output_dir=temp_dir,
                filename="test_ica_components.png",
                metric_name="test_metric"
            )
            print("    ICA visualizations passed!")
            
            # List created files
            files = os.listdir(temp_dir)
            print(f"  Created visualization files: {files}")
            print("All visualization tests passed!")
            
        except Exception as e:
            print(f"Visualization test failed: {e}")
            return False
    
    print()
    print("All tests passed successfully! ✅")
    print("The fixes in new_main.py should work correctly.")
    return True

if __name__ == "__main__":
    test_pca_ica_workflow()
