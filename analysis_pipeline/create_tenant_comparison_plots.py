#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot comparisons of tenant metrics across experiment phases
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
import logging
from data_loader import DataLoader
from visualizations import VisualizationGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def merge_phase_data(data, tenants, metric_name):
    """
    Merge data from different phases for comparison and convert to elapsed time
    """
    merged_data = {}
    all_timestamps = []
    
    # First pass: collect all timestamps to determine experiment start time
    for tenant in tenants:
        for phase_name, phase_data in data.items():
            if tenant in phase_data and metric_name in phase_data[tenant]:
                phase_df = phase_data[tenant][metric_name].copy()
                if not phase_df.empty and isinstance(phase_df.index, pd.DatetimeIndex):
                    all_timestamps.extend(phase_df.index.tolist())
    
    # Determine experiment start time
    if all_timestamps:
        experiment_start = min(all_timestamps)
        logging.info(f"Experiment start time: {experiment_start}")
    else:
        experiment_start = None
        logging.warning("Could not determine experiment start time")
    
    # Second pass: process data with normalized time
    for tenant in tenants:
        tenant_data = pd.DataFrame()
        
        # Process each phase
        for phase_name, phase_data in data.items():
            if tenant in phase_data and metric_name in phase_data[tenant]:
                # Get phase data for this tenant and metric
                phase_df = phase_data[tenant][metric_name].copy()
                
                # Convert index to datetime if it's not already
                if not isinstance(phase_df.index, pd.DatetimeIndex):
                    try:
                        phase_df.index = pd.to_datetime(phase_df.index)
                    except:
                        # Keep numeric index if conversion fails
                        pass
                
                # If phase dataframe is empty or has no numeric columns, skip
                if phase_df.empty or not any(pd.api.types.is_numeric_dtype(phase_df[col]) for col in phase_df.columns):
                    logging.warning(f"No usable data for {tenant}/{metric_name} in phase {phase_name}")
                    continue
                
                # Convert datetime index to elapsed time in seconds if we have experiment start
                if isinstance(phase_df.index, pd.DatetimeIndex) and experiment_start is not None:
                    # Calculate seconds since start
                    elapsed_seconds = [(ts - experiment_start).total_seconds() for ts in phase_df.index]
                    
                    # Create a new DataFrame with elapsed seconds as index
                    new_df = phase_df.copy()
                    new_df['elapsed_seconds'] = elapsed_seconds
                    new_df = new_df.reset_index()
                    new_df = new_df.set_index('elapsed_seconds')
                    phase_df = new_df
                
                # Add phase label column
                phase_df['phase'] = phase_name
                
                # Identify the value column (first numeric column)
                value_col = next((col for col in phase_df.columns 
                                if pd.api.types.is_numeric_dtype(phase_df[col]) and 
                                col not in ['elapsed_seconds', 'index']), None)
                
                if value_col:
                    # Rename to 'value' for consistency
                    if value_col != 'value':
                        phase_df['value'] = phase_df[value_col]
                
                # Append to tenant data
                tenant_data = pd.concat([tenant_data, phase_df])
            else:
                logging.warning(f"No data for {tenant}/{metric_name} in phase {phase_name}")
        
        # Store the merged data for this tenant
        merged_data[tenant] = tenant_data
    
    return merged_data

def get_phase_boundaries(data):
    """
    Get elapsed time points for phase boundaries and deduplicate them
    """
    # Format: (timestamp, source_phase, target_phase)
    raw_boundaries = []
    phase_transitions = set()  # To track unique phase transitions
    
    # Extract all phase changes
    for tenant, tenant_data in data.items():
        if tenant_data.empty:
            continue
        
        rows = list(tenant_data.iterrows())
        current_phase = None
        
        for i, (timestamp, row) in enumerate(rows):
            if 'phase' in row:
                phase = row['phase']
                if phase != current_phase and current_phase is not None:
                    # Record phase transition
                    raw_boundaries.append((timestamp, current_phase, phase))
                current_phase = phase
    
    # Sort boundaries by timestamp and deduplicate based on transitions
    # (This ensures we only have one boundary per phase transition)
    seen_transitions = set()
    boundaries = []
    
    for timestamp, source, target in sorted(raw_boundaries, key=lambda x: x[0]):
        transition = (source, target)
        if transition not in seen_transitions:
            seen_transitions.add(transition)
            boundaries.append((timestamp, source, target))
    
    return boundaries

def plot_tenant_comparison(data, tenants, metric_name, output_path, title=None, ylim=None):
    """
    Create a plot comparing metrics across tenants with elapsed time on x-axis
    """
    plt.figure(figsize=(14, 8))
    
    # Prepare merged data with elapsed time
    merged_data = merge_phase_data(data, tenants, metric_name)
    
    # Color map for tenants
    tenant_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    
    # Plot data for each tenant
    for i, tenant in enumerate(tenants):
        if tenant not in merged_data or merged_data[tenant].empty:
            logging.warning(f"No data for tenant {tenant}")
            continue
            
        tenant_data = merged_data[tenant]
        
        # Plot the metric value
        if 'value' in tenant_data.columns:
            plt.plot(tenant_data.index, tenant_data['value'], 
                    color=tenant_colors[i % len(tenant_colors)], 
                    label=f"{tenant}", 
                    linewidth=2, 
                    alpha=0.8)
    
    # Get unique phase boundaries (deduplicated)
    data_boundaries = get_phase_boundaries(merged_data)
    
    # Add background colors for phases
    phase_colors = {
        '1 - Baseline': 'lightblue',
        '2 - Attack': 'lightcoral',
        '3 - Recovery': 'lightgreen'
    }
    
    # If we have phase data, add background shading and boundary lines
    if data_boundaries:
        # First, determine the start and end of experiment
        all_indices = []
        for tenant_data in merged_data.values():
            if not tenant_data.empty:
                all_indices.extend(tenant_data.index.tolist())
        
        if all_indices:
            experiment_start = min(all_indices)
            experiment_end = max(all_indices)
            
            # Get first phase
            first_phase = data_boundaries[0][1]  # Source phase of first boundary
            
            # Add shading for first phase from start to first boundary
            plt.axvspan(experiment_start, data_boundaries[0][0], 
                      alpha=0.2, color=phase_colors.get(first_phase, 'white'))
            
            # Add boundary lines and shading for the rest
            for i, (timestamp, source_phase, target_phase) in enumerate(data_boundaries):
                # Add boundary line - only one per transition after deduplication
                plt.axvline(x=timestamp, color='black', linestyle='--', alpha=0.7)
                
                # Add label for phase transition
                y_pos = plt.ylim()[1] * 0.95
                plt.text(timestamp, y_pos, f"{source_phase} → {target_phase}",
                        rotation=90, verticalalignment='top', horizontalalignment='right')
                
                # Add shading for next phase
                if i < len(data_boundaries) - 1:  # If not the last boundary
                    next_timestamp = data_boundaries[i+1][0]
                    plt.axvspan(timestamp, next_timestamp, 
                              alpha=0.2, color=phase_colors.get(target_phase, 'white'))
                else:  # Last boundary - shade to the end
                    plt.axvspan(timestamp, experiment_end, 
                              alpha=0.2, color=phase_colors.get(target_phase, 'white'))
    
    # Set plot title and labels
    plt.title(title or f"Comparison of {metric_name} across Tenants")
    plt.xlabel("Time Elapsed (seconds)")
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Set y-axis limits if provided
    if ylim:
        plt.ylim(ylim)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved plot to {output_path}")
    plt.close()

def main():
    # Set up paths
    base_path = "/home/phil/Projects/k8s-noisy-lab-data-pipe"
    experiment_name = "2025-05-11/16-58-00/default-experiment-1"
    round_number = "round-1"
    output_dir = f"{base_path}/analysis_pipeline/results/tenant_comparison"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")
    
    # Initialize data loader and load data
    logging.info(f"Loading data for experiment {experiment_name}, round {round_number}")
    data_loader = DataLoader(base_path, experiment_name, round_number)
    logging.info("Loading all phases...")
    data = data_loader.load_all_phases()
    
    # Log the phases and tenants that were found
    phases = list(data.keys())
    logging.info(f"Loaded {len(phases)} phases: {phases}")
    
    for phase, phase_data in data.items():
        components = list(phase_data.keys())
        logging.info(f"Phase {phase} has {len(components)} components: {components}")
    
    # List of metrics to plot
    metrics = [
        ("cpu_usage", "CPU Usage (%)"),
        ("memory_usage", "Memory Usage (%)"),
        ("disk_io_total", "Disk I/O Operations"),
        ("network_total_bandwidth", "Network Bandwidth (bytes)")
    ]
    
    # List of tenants to include
    tenants = ["tenant-a", "tenant-b", "tenant-c", "tenant-d"]
    
    # Generate plots for each metric
    for metric_name, metric_title in metrics:
        logging.info(f"Generating plot for {metric_name}")
        plot_tenant_comparison(
            data=data,
            tenants=tenants,
            metric_name=metric_name,
            output_path=f"{output_dir}/{metric_name}_comparison.png",
            title=f"{metric_title} Comparison Across Tenants and Phases"
        )
    
    logging.info(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    main()
