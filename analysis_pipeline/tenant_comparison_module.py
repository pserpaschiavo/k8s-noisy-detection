#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tenant Comparison Module for Kubernetes Noisy Neighbors Lab
This module provides functionality for generating tenant comparison visualizations.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
from data_loader import DataLoader
from visualizations import VisualizationGenerator

def generate_tenant_comparisons(data_loader, phases, tenants, metrics_list=None, output_dir=None, colorblind_friendly=True):
    """
    Generate comparison plots for tenants across experiment phases.
    
    Args:
        data_loader (DataLoader): DataLoader object with loaded data
        phases (list): List of phases to include 
        tenants (list): List of tenant names to include in comparison
        metrics_list (list, optional): List of metrics to plot as (name, display_title) tuples
        output_dir (Path or str, optional): Output directory for plots
        colorblind_friendly (bool): Use colorblind friendly palette (default: True)
        
    Returns:
        list: Paths to the generated plot files
    """
    logging.info(f"Generating tenant comparison plots for {len(tenants)} tenants across {len(phases)} phases")
    logging.info(f"Colorblind friendly mode: {'Enabled' if colorblind_friendly else 'Disabled'}")
    
    # If no metrics list provided, use default metrics
    if not metrics_list:
        metrics_list = [
            ("cpu_usage", "CPU Usage (%)"),
            ("memory_usage", "Memory Usage (%)"),
            ("disk_io_total", "Disk I/O Operations"),
            ("network_total_bandwidth", "Network Bandwidth (bytes)")
        ]
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("./results/tenant_comparison")
    else:
        output_dir = Path(output_dir)
        
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualization generator with colorblind friendly setting
    vis_generator = VisualizationGenerator(output_dir=output_dir, colorblind_friendly=colorblind_friendly)
    
    # Get all data from the data_loader
    if hasattr(data_loader, 'data'):
        # Data is already loaded in the data_loader
        data = {}
        for phase in phases:
            if phase in data_loader.data:
                data[phase] = data_loader.data[phase]
            else:
                logging.warning(f"No data found for phase {phase}")
    else:
        # Need to load data
        logging.info("Loading all phases from data_loader...")
        data = data_loader.load_all_phases()
        # Filter to include only requested phases
        data = {phase: data[phase] for phase in phases if phase in data}
    
    if not data:
        logging.error("No phase data was loaded, cannot generate tenant comparison plots")
        return []
        
    # Visualization generator was already created above, we should not create a new one
    # This is our fixed version that uses the previously created vis_generator with colorblind settings
    
    # Generate plots
    try:
        plot_paths = vis_generator.generate_tenant_comparison_plots(
            data=data,
            tenants=tenants,
            metrics_list=metrics_list,
            output_subdir='tenant_comparison'
        )
        return plot_paths
    except Exception as e:
        logging.error(f"Error generating tenant comparison plots: {e}", exc_info=True)
        return []

def run_standalone_comparison(experiment_path, round_number="round-1", output_path=None, colorblind_friendly=True):
    """
    Run tenant comparison as a standalone process.
    
    Args:
        experiment_path (str): Path to experiment (e.g., "2025-05-11/16-58-00/default-experiment-1")
        round_number (str): Round to analyze (default: "round-1")
        output_path (str, optional): Output directory for results
        colorblind_friendly (bool): Use colorblind friendly palette (default: True)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Set up paths
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if not output_path:
            output_path = os.path.join(base_path, "analysis_pipeline", "results", "tenant_comparison")
        
        # Initialize data loader and load data
        logging.info(f"Loading data for experiment {experiment_path}, round {round_number}")
        data_loader = DataLoader(base_path, experiment_path, round_number)
        
        # Load all phases
        data = data_loader.load_all_phases()
        if not data:
            logging.error("Failed to load experiment data")
            return False
        
        phases = list(data.keys())
        logging.info(f"Loaded {len(phases)} phases: {phases}")
        
        # Find tenants in the data
        tenants = []
        for phase_name, phase_data in data.items():
            for component in phase_data.keys():
                if component.startswith("tenant-"):
                    if component not in tenants:
                        tenants.append(component)
        
        if not tenants:
            logging.error("No tenant data found in the experiment")
            return False
        
        logging.info(f"Found {len(tenants)} tenants: {tenants}")
        
        # Generate plots using the DataLoader directly
        plot_paths = generate_tenant_comparisons(
            data_loader=data_loader,
            phases=phases,
            tenants=tenants,
            output_dir=output_path,
            colorblind_friendly=colorblind_friendly
        )
        
        if plot_paths:
            logging.info(f"Successfully generated {len(plot_paths)} tenant comparison plots")
            for path in plot_paths:
                logging.info(f"  - {path}")
            return True
        else:
            logging.error("Failed to generate tenant comparison plots")
            return False
            
    except Exception as e:
        logging.error(f"Error in tenant comparison standalone process: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # When run as a script, load default experiment
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate tenant comparison plots')
    parser.add_argument('--experiment', type=str, default="2025-05-11/16-58-00/default-experiment-1",
                      help='Path to experiment relative to base path')
    parser.add_argument('--round', type=str, default="round-1",
                      help='Round number to analyze')
    parser.add_argument('--output', type=str, default=None,
                      help='Output directory for plots')
    parser.add_argument('--colorblind-friendly', type=bool, default=True,
                      help='Use colorblind friendly palette for plots')
                      
    args = parser.parse_args()
    
    # Agora passamos também o parâmetro de visualização amigável para daltônicos
    success = run_standalone_comparison(
        experiment_path=args.experiment,
        round_number=args.round,
        output_path=args.output,
        colorblind_friendly=args.colorblind_friendly
    )
    
    sys.exit(0 if success else 1)