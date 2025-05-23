#!/usr/bin/env python3
"""
Sanity tests for causal (SEM) analysis module and visualizations.
This script generates synthetic data with a known causal relationship,
tests perform_sem_analysis, CSV export, path diagram, and fit indices plots.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import tempfile

from analysis_modules.causality import perform_sem_analysis, plot_sem_path_diagram, plot_sem_fit_indices
from data_handling.save_results import export_to_csv


def test_sem_analysis_and_plots():
    # generate synthetic data
    np.random.seed(42)
    n = 100
    X = np.random.normal(size=n)
    Z = np.random.normal(size=n)
    # Y depends on X and Z
    Y = 0.6 * X - 0.4 * Z + np.random.normal(scale=0.1, size=n)
    df = pd.DataFrame({'X': X, 'Z': Z, 'Y': Y})

    # SEM model spec: Y ~ X + Z; error covariances
    model_spec = 'Y ~ X + Z'
    sem_results = perform_sem_analysis(df, model_spec, exog_vars=['X', 'Z'])
    # check results structure
    assert isinstance(sem_results, dict), "SEM results should be a dict"
    estimates = sem_results.get('estimates')
    stats = sem_results.get('stats')
    assert isinstance(estimates, pd.DataFrame), "Estimates should be a DataFrame"
    assert 'Estimate' in estimates.columns, "Estimates DataFrame should contain 'Estimate' column"
    assert isinstance(stats, dict), "Stats should be a dict"

    # test CSV export and plots
    with tempfile.TemporaryDirectory() as td:
        # export estimates
        est_csv = os.path.join(td, 'estimates.csv')
        stats_csv = os.path.join(td, 'stats.csv')
        export_to_csv(estimates, est_csv)
        # stats is dict, convert to DataFrame for export
        stats_df = pd.DataFrame([stats])
        export_to_csv(stats_df, stats_csv)
        assert os.path.exists(est_csv), "Estimates CSV should exist"
        assert os.path.exists(stats_csv), "Stats CSV should exist"

        # plots
        path_png = 'sem_path.png'
        fit_png = 'sem_fit.png'
        plot_sem_path_diagram(
            sem_results,
            title='Test SEM Path Diagram',
            output_dir=td,
            filename=path_png
        )
        plot_sem_fit_indices(
            sem_results,
            title='Test SEM Fit Indices',
            output_dir=td,
            filename=fit_png
        )
        assert os.path.exists(os.path.join(td, path_png)), "SEM path diagram PNG should exist"
        assert os.path.exists(os.path.join(td, fit_png)), "SEM fit indices PNG should exist"

    print("SEM analysis tests passed.")


if __name__ == '__main__':
    test_sem_analysis_and_plots()
