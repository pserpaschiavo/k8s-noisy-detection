#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo runner for testing the causal analysis module
"""

import sys
import os
from pathlib import Path
import logging
import pandas as pd
import networkx as nx
from causal_analysis_integrated import CausalAnalysisIntegrated

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_demo_data():
    """Load demo data directly"""
    data = {}
    base_path = Path('/home/phil/Projects/k8s-noisy-detection')
    demo_path = base_path / 'demo-data/default-experiment-1-round/round-1'
    
    print(f"Loading demo data from: {demo_path}")
    
    # Check if path exists
    if not demo_path.exists():
        print(f"Error: Demo data path does not exist: {demo_path}")
        return None
    
    # For each phase directory
    for phase_dir in demo_path.glob("*"):
        if phase_dir.is_dir():
            phase_name = phase_dir.name
            data[phase_name] = {}
            print(f"Found phase: {phase_name}")
            
            # For each component directory
            for component_dir in phase_dir.glob("*"):
                if component_dir.is_dir():
                    component_name = component_dir.name
                    data[phase_name][component_name] = {}
                    print(f"Found component: {component_name}")
                    
                    # Load each CSV file as a metric
                    for csv_file in component_dir.glob("*.csv"):
                        metric_name = csv_file.stem
                        try:
                            df = pd.read_csv(csv_file)
                            data[phase_name][component_name][metric_name] = df
                            print(f"Loaded {metric_name} for {component_name}")
                        except Exception as e:
                            print(f"Error loading {csv_file}: {str(e)}")
    
    return data

def main():
    # Load data
    data = load_demo_data()
    
    if not data:
        print("Failed to load demo data")
        return 1
    
    # Setup output directory
    output_dir = Path('/home/phil/Projects/k8s-noisy-detection/analysis_pipeline/test_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoaded data structure:")
    for phase_name, phase_data in data.items():
        print(f"Phase: {phase_name}")
        for component_name, component_data in phase_data.items():
            print(f"  Component: {component_name}")
            for metric_name in component_data.keys():
                print(f"    Metric: {metric_name}")
    
    # Initialize causal analyzer
    causal_analyzer = CausalAnalysisIntegrated(output_dir)
    
    # Run causal analysis with toda-yamamoto method
    print("\nRunning causal analysis with Toda-Yamamoto method...")
    results = causal_analyzer.run_causal_analysis(
        phase_data=data,
        method='toda-yamamoto',
        metrics_of_interest=['cpu_usage', 'memory_usage', 'network_total_bandwidth'],
        components=['tenant-a', 'ingress-nginx'],
        save_results=True
    )
    
    if not results.empty:
        print(f"\nCausal analysis results:")
        print(results.head())
        print(f"\nTotal results: {len(results)}")
        print(f"Results saved to: {output_dir}")
    else:
        print("No causal relationships found or error in analysis")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
