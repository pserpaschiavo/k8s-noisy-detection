#!/usr/bin/env python3
"""
Sanity tests for descriptive statistics module and plots.
This script generates sample data and tests calculate_descriptive_statistics, CSV export,
and visualization via boxplot and lineplot.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import tempfile

from analysis_modules.descritive_statistics import calculate_descriptive_statistics
from visualization.new_plots import plot_descriptive_stats_boxplot, plot_descriptive_stats_lineplot
from data_handling.save_results import export_to_csv


def test_descriptive_statistics_and_plots():
    # create sample data
    np.random.seed(0)
    n = 50
    df = pd.DataFrame({
        'round': ['r1'] * n + ['r2'] * n,
        'phase': ['p1'] * (n//2) + ['p2'] * (n//2) + ['p1'] * (n//2) + ['p2'] * (n//2),
        'tenant': ['tA', 'tB'] * n,
        'value': np.random.randn(2*n)
    })

    # test calculate_descriptive_statistics overall
    stats_overall = calculate_descriptive_statistics(df, metric_column='value')
    assert not stats_overall.empty, "Overall descriptive stats should not be empty"

    # test CSV export
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, 'stats_overall.csv')
        export_to_csv(stats_overall, csv_path)
        assert os.path.exists(csv_path), "CSV file should be created"

        # test boxplot
        fig1 = plot_descriptive_stats_boxplot(
            df, 'test_metric', 'value',
            title='Boxplot Test', output_dir=td, filename='boxplot.png'
        )
        assert os.path.exists(os.path.join(td, 'boxplot.png'))

        # test lineplot (requires time axis: use index)
        df_line = df.copy()
        df_line.index.name = 'sample'
        fig2 = plot_descriptive_stats_lineplot(
            df_line, 'test_metric', 'value',
            title='Lineplot Test', output_dir=td, filename='lineplot.png'
        )
        assert os.path.exists(os.path.join(td, 'lineplot.png'))

    print("Descriptive statistics tests passed.")


if __name__ == '__main__':
    test_descriptive_statistics_and_plots()
