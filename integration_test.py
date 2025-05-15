#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de integração completa para K8s Noisy Neighbors Lab

Este arquivo implementa a integração das análises recomendadas para
detecção de vizinhos barulhentos no Kubernetes.

Autor: Equipe de Análise de Dados
Data: 14/05/2025
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def print_separator():
    """Print a separator line for better output readability"""
    print("\n" + "="*70 + "\n")

def print_section(title):
    """Print a section title"""
    print_separator()
    print(f"{title}")
    print("-" * len(title))

def main():
    """Main function for the integration"""
    print_section("K8s Noisy Neighbors Integrated Analysis Framework")

    # Set up directories
    project_root = Path('/home/phil/Projects/k8s-noisy-detection')
    output_dir = project_root / 'analysis_pipeline/integration_test_output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Project root: {project_root}")
    print(f"Output directory: {output_dir}")

    # Import modules with checks
    print_section("Importing Required Modules")
    
    try:
        # Pipeline Manager
        from analysis_pipeline.pipeline_manager import PipelineManager
        print("✅ Successfully imported Pipeline Manager")
        
        # Fixed Causal Analysis
        from analysis_pipeline.causal_fixed import CausalAnalysisFixed
        print("✅ Successfully imported fixed Causal Analysis")

    except ImportError as e:
        print(f"❌ Error importing modules: {str(e)}")
        return 1

    # Test data generation
    print_section("Generating Test Data")
    
    # Generate synthetic time series data
    n = 100
    time_index = pd.date_range(start='2025-01-01', periods=n, freq='H')
    
    np.random.seed(42)  # For reproducibility
    
    # Generate causal relationships
    # x -> y with lag 2
    x = np.sin(np.linspace(0, 8*np.pi, n)) + np.random.normal(0, 0.2, n)
    y = np.roll(x, 2) + np.random.normal(0, 0.5, n)
    
    # z is independent
    z = np.random.normal(0, 1, n)
    
    # Create DataFrames
    df_x = pd.DataFrame({'value': x}, index=time_index)
    df_y = pd.DataFrame({'value': y}, index=time_index)
    df_z = pd.DataFrame({'value': z}, index=time_index)
    
    print(f"Generated {n} data points for each series")
    
    # Create a simple data structure that mimics the pipeline format
    data = {
        'Phase1': {
            'ComponentA': {
                'metric_x': df_x,
                'metric_y': df_y
            },
            'ComponentB': {
                'metric_z': df_z,
                'metric_w': pd.DataFrame({'value': df_x['value'] * 1.2 + np.random.normal(0, 0.3, n)}, index=time_index)
            }
        },
        'Phase2': {
            'ComponentA': {
                'metric_x': pd.DataFrame({'value': df_x['value'] * 2}, index=time_index),
                'metric_y': pd.DataFrame({'value': df_y['value'] * 2}, index=time_index)
            },
            'ComponentB': {
                'metric_z': pd.DataFrame({'value': df_z['value'] * 1.5}, index=time_index),
                'metric_w': pd.DataFrame({'value': df_x['value'] * 2.5 + np.random.normal(0, 0.3, n)}, index=time_index)
            }
        }
    }
    
    print("Created test data structure with 2 phases, 2 components, and 4 metrics")
    
    # Test causal analysis
    print_section("Testing Causal Analysis")
    
    try:
        # Initialize the causal analyzer
        print("Initializing CausalAnalysisFixed...")
        causal_analyzer = CausalAnalysisFixed(output_dir)
        print("✅ Successfully initialized causal analyzer")
        
        # Run causal analysis
        print("\nRunning causal analysis...")
        results = causal_analyzer.run_causal_analysis(
            phase_data=data,
            method='toda-yamamoto',
            save_results=True
        )
        
        # Print results
        if not results.empty:
            print("\n✅ Causal analysis successful!")
            print("\nResults summary:")
            print(f"- Found {len(results)} causal relationships")
            print(f"- Detected in phases: {results['phase'].unique()}")
            
            # Show significant relationships
            sig_results = results[results['causality'] == True]
            if len(sig_results) > 0:
                print(f"\nSignificant causal relationships ({len(sig_results)}):")
                for _, row in sig_results.iterrows():
                    print(f"  {row['source_metric']} → {row['target_metric']} (p={row['p_value']:.4f})")
            else:
                print("\nNo significant causal relationships found.")
                
        else:
            print("❌ No results returned from causal analysis")
    
    except Exception as e:
        print(f"❌ Error in causal analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print_section("Integration Test Complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
