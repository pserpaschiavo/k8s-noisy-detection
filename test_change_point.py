#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script for the CausalAnalyzer.change_point_impact_analysis method.

IMPORTANTE: Este script requer Python 3.10 para funcionar corretamente.
A biblioteca ruptures e outras dependências científicas podem apresentar
problemas de compatibilidade em versões mais recentes do Python.

Para executar corretamente:
1. Use o ambiente Python 3.10: source .venv310/bin/activate
2. Execute: python test_change_point.py
"""

import numpy as np
import pandas as pd
import logging
import os
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Import after logging config
from analysis_pipeline.causal_analysis import CausalAnalyzer, RUPTURES_AVAILABLE

# Print status of optional libraries
print(f"RUPTURES_AVAILABLE: {RUPTURES_AVAILABLE}")

def run_test():
    # Create synthetic test data
    np.random.seed(42)

    # Source metric with a clear change point
    x = np.concatenate([np.random.normal(0, 1, 50), np.random.normal(5, 1, 50)])
    source_metric = pd.Series(x)

    # Target metric with a delayed change point
    y1 = np.concatenate([np.random.normal(0, 1, 55), np.random.normal(3, 1, 45)])
    y2 = np.concatenate([np.random.normal(0, 1, 60), np.random.normal(4, 1, 40)])

    target_metrics = {
        'metric1': pd.Series(y1),
        'metric2': pd.Series(y2)
    }

    # Ensure output directory exists
    output_dir = '/tmp/causal_test'
    os.makedirs(output_dir, exist_ok=True)

    # Run the analysis
    logging.info('Creating analyzer')
    analyzer = CausalAnalyzer(output_dir)
    logging.info('Running change point impact analysis')
    try:
        result = analyzer.change_point_impact_analysis(source_metric, target_metrics)
        logging.info('Analysis completed successfully')
        if isinstance(result, dict) and 'detailed_results' in result:
            logging.info(f'Found {len(result["detailed_results"])} impact points')
            print(result['detailed_results'].head())
        else:
            logging.info(f'Result type: {type(result)}')
            logging.info(f'No impacts found or error: {result}')
    except Exception as e:
        logging.exception('Error in analysis')

if __name__ == "__main__":
    run_test()
