#!/usr/bin/env python3
"""
Sanity tests for correlation and covariance analysis module and heatmap plots.
This script generates sample data for two tenants and tests correlation/covariance functions,
CSV export, and visualization via heatmaps.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import tempfile

from analysis_modules.correlation_covariance import (
    calculate_inter_tenant_correlation_per_metric,
    calculate_inter_tenant_covariance_per_metric
)
from visualization.new_plots import plot_correlation_heatmap, plot_covariance_heatmap
from data_handling.save_results import export_to_csv


def test_correlation_covariance_and_plots():
    # generate sample data with datetime index
    date_range = pd.date_range(start='2021-01-01', periods=10, freq='T')
    # create values for two tenants with some correlation
    x = np.arange(10)
    noise = np.random.randn(10) * 0.1
    df = pd.DataFrame({
        'datetime': list(date_range) * 2,
        'tenant': ['tA'] * 10 + ['tB'] * 10,
        'value': list(x + noise) + list(x * 0.5 + noise)
    })

    # test correlation
    corr = calculate_inter_tenant_correlation_per_metric(df, method='pearson', time_col='datetime')
    assert isinstance(corr, pd.DataFrame), "Correlation output must be a DataFrame"
    # should have two tenants
    assert corr.shape == (2, 2), f"Expected 2x2 correlation matrix, got {corr.shape}"

    # test covariance
    cov = calculate_inter_tenant_covariance_per_metric(df, time_col='datetime')
    assert isinstance(cov, pd.DataFrame), "Covariance output must be a DataFrame"
    assert cov.shape == (2, 2), f"Expected 2x2 covariance matrix, got {cov.shape}"

    # test CSV export and plots
    with tempfile.TemporaryDirectory() as td:
        # export CSVs
        corr_csv = os.path.join(td, 'corr.csv')
        cov_csv = os.path.join(td, 'cov.csv')
        export_to_csv(corr, corr_csv)
        export_to_csv(cov, cov_csv)
        assert os.path.exists(corr_csv), "Correlation CSV should exist"
        assert os.path.exists(cov_csv), "Covariance CSV should exist"
        # heatmap plots
        fig1 = plot_correlation_heatmap(corr, title='Corr Heatmap', output_dir=td, filename='corr.png')
        fig2 = plot_covariance_heatmap(cov, title='Cov Heatmap', output_dir=td, filename='cov.png')
        assert os.path.exists(os.path.join(td, 'corr.png'))
        assert os.path.exists(os.path.join(td, 'cov.png'))

    print("Correlation and covariance tests passed.")

if __name__ == '__main__':
    test_correlation_covariance_and_plots()
