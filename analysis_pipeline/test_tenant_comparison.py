#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path
from data_loader import DataLoader
from visualizations import VisualizationGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def test_tenant_comparison():
    # Set up paths
    base_path = "/home/phil/Projects/k8s-noisy-lab-data-pipe"
    experiment_name = "2025-05-11/16-58-00/default-experiment-1"
    round_number = "round-1"
    output_dir = Path(base_path) / "analysis_pipeline/results/tenant_comparison_test"
    
    # Initialize data loader and load data
    logging.info(f"Loading data for experiment {experiment_name}, round {round_number}")
    data_loader = DataLoader(base_path, experiment_name, round_number)
    logging.info("Loading all phases...")
    data = data_loader.load_all_phases()
    
    # Create visualizations
    vis_generator = VisualizationGenerator(output_dir)
    
    # List of tenants to include
    tenants = ["tenant-a", "tenant-b", "tenant-c", "tenant-d"]
    
    # Define metrics to plot
    metrics_to_plot = [
        ("cpu_usage", "CPU Usage (%)"),
        ("memory_usage", "Memory Usage (%)"),
        ("disk_io_total", "Disk I/O Operations"),
        ("network_total_bandwidth", "Network Bandwidth (bytes)")
    ]
    
    # Generate plots
    plots = vis_generator.generate_tenant_comparison_plots(
        data=data,
        tenants=tenants,
        metrics_list=metrics_to_plot
    )
    
    logging.info(f"Generated plots: {plots}")

if __name__ == "__main__":
    test_tenant_comparison()
