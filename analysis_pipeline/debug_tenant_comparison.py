#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script for tenant comparison module
"""
import sys
import logging
import traceback
from tenant_comparison_module import run_standalone_comparison

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logging.info("Starting debug test of tenant comparison module")

# Try to import required modules
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    from data_loader import DataLoader
    from visualizations import VisualizationGenerator
    logging.info("All required modules imported successfully")
except ImportError as e:
    logging.error(f"Failed to import required module: {e}")
    traceback.print_exc()
    sys.exit(1)

# Run the comparison
success = run_standalone_comparison(
    experiment_path="2025-05-11/16-58-00/default-experiment-1",
    round_number="round-1",
    output_path="/home/phil/Projects/k8s-noisy-lab-data-pipe/analysis_pipeline/results/tenant_debug_test"
)

logging.info(f"Test completed with success={success}")
sys.exit(0 if success else 1)
