#!/usr/bin/env python3
"""
Sanity tests for similarity analysis module and heatmap plots.
This script generates sample data for two tenants and tests distance correlation and cosine similarity functions,
CSV export, and visualization via heatmaps.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import tempfile

from analysis_modules.similarity import (
    calculate_pairwise_distance_correlation,
    calculate_pairwise_cosine_similarity,
    plot_distance_correlation_heatmap,
    plot_distance_correlation_heatmap as plot_cosine_similarity_heatmap
)
from data_handling.save_results import export_to_csv


def test_similarity_and_plots():
    # generate sample data with datetime index
    date_range = pd.date_range(start='2021-01-01', periods=15, freq='T')
    # sample values for two tenants
    x = np.sin(np.linspace(0, 3.14, 15))
    noise = np.random.randn(15) * 0.05
    df = pd.DataFrame({
        'datetime': list(date_range) * 2,
        'tenant': ['tenant1'] * 15 + ['tenant2'] * 15,
        'value': list(x + noise) + list(x[::-1] + noise)
    })

    # test distance correlation
    dist_df = calculate_pairwise_distance_correlation(
        df, time_col='datetime', metric_col='value', group_col='tenant'
    )
    assert isinstance(dist_df, pd.DataFrame), "Distance correlation output must be a DataFrame"
    assert dist_df.shape == (2, 2), f"Expected 2x2 matrix, got {dist_df.shape}"

    # test cosine similarity
    cos_df = calculate_pairwise_cosine_similarity(
        df, time_col='datetime', metric_col='value', group_col='tenant'
    )
    assert isinstance(cos_df, pd.DataFrame), "Cosine similarity output must be a DataFrame"
    assert cos_df.shape == (2, 2), f"Expected 2x2 matrix, got {cos_df.shape}"

    # test CSV export and plots
    with tempfile.TemporaryDirectory() as td:
        dist_csv = os.path.join(td, 'distance_corr.csv')
        cos_csv = os.path.join(td, 'cosine_sim.csv')
        export_to_csv(dist_df, dist_csv)
        export_to_csv(cos_df, cos_csv)
        assert os.path.exists(dist_csv), "Distance correlation CSV should exist"
        assert os.path.exists(cos_csv), "Cosine similarity CSV should exist"

        # heatmap plots
        plot_distance_correlation_heatmap(
            dist_df,
            title='Distance Correlation Heatmap',
            output_dir=td,
            filename='distance_heatmap.png'
        )
        plot_cosine_similarity_heatmap(
            cos_df,
            title='Cosine Similarity Heatmap',
            output_dir=td,
            filename='cosine_heatmap.png'
        )
        assert os.path.exists(os.path.join(td, 'distance_heatmap.png'))
        assert os.path.exists(os.path.join(td, 'cosine_heatmap.png'))

    print("Similarity analysis tests passed.")


if __name__ == '__main__':
    test_similarity_and_plots()
