#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Main Pipeline for Kubernetes Noisy Neighbours Lab
This module serves as the entry point for the integrated analysis pipeline, 
coordinating metrics, phase, and tenant analyses.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
import traceback

# Import pipeline modules
from analysis_pipeline.data_loader import DataLoader
from analysis_pipeline.metrics_analysis import MetricsAnalyzer
from analysis_pipeline.phase_analysis import PhaseAnalyzer
from analysis_pipeline.tenant_analysis import TenantAnalyzer
from analysis_pipeline.suggestion_engine import SuggestionEngine
from analysis_pipeline.causal_fixed import CausalAnalysisFixed
from analysis_pipeline.pipeline_manager import PipelineManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("analysis_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='K8s Noisy Neighbours Lab - Integrated Analysis Pipeline')
    
    parser.add_argument('--experiment', type=str,
                        help='Path to the experiment results directory relative to results/ (e.g., "2025-05-11/16-58-00/default-experiment-1")')
    
    parser.add_argument('--round', type=str, nargs='+', default=['round-1'],
                        help='Round(s) to analyze (e.g., "round-1" or multiple rounds "round-1 round-2 round-3")')
    
    parser.add_argument('--combine-rounds', action='store_true',
                        help='Combine data from multiple rounds when analyzing')
    
    parser.add_argument('--combine-method', type=str, default='mean',
                        choices=['mean', 'median', 'min', 'max'],
                        help='Method to combine data from multiple rounds (default: mean)')
    
    parser.add_argument('--phases', type=str, nargs='+', 
                        default=["1 - Baseline", "2 - Attack", "3 - Recovery"],
                        help='Phases to analyze (default: "1 - Baseline" "2 - Attack" "3 - Recovery")')
    
    parser.add_argument('--output', type=str,
                        help='Output directory (default: analysis/YYYY-MM-DD_HH-MM-SS/)')
    
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip generating visualizations (for faster analysis)')
    
    parser.add_argument('--skip-advanced', action='store_true',
                        help='Skip advanced time series analysis (faster)')
    
    parser.add_argument('--metrics-of-interest', type=str, nargs='+',
                        default=['cpu_usage', 'memory_usage', 'network_total_bandwidth', 'disk_io_total'],
                        help='List of key metrics to focus on for cross-phase comparison')
    
    parser.add_argument('--components', type=str, nargs='+',
                        default=['tenant-a', 'tenant-b', 'tenant-c', 'tenant-d', 'ingress-nginx'],
                        help='List of components to analyze')
    
    # Analysis types
    parser.add_argument('--metrics-analysis', action='store_true',
                        help='Run metrics-focused analysis')
    
    parser.add_argument('--phase-analysis', action='store_true',
                        help='Run phase comparison analysis')
    
    parser.add_argument('--tenant-analysis', action='store_true',
                        help='Run tenant-focused analysis')
    
    parser.add_argument('--suggest-visualizations', action='store_true',
                        help='Generate suggestions for most relevant visualizations')
    
    # Advanced analysis options
    parser.add_argument('--advanced-analysis', action='store_true', default=False,
                        help='Perform advanced statistical analysis (change points, clustering, etc.)')
    
    parser.add_argument('--distribution-analysis', action='store_true', default=False,
                        help='Perform distribution analysis and fitting')
    
    parser.add_argument('--anomaly-detection', type=str, choices=['iforest', 'zscore', 'iqr'], default=None,
                        help='Perform anomaly detection using the specified method')
    
    parser.add_argument('--tenant-comparison', action='store_true', default=False,
                        help='Generate tenant comparison plots across phases')
    
    parser.add_argument('--colorblind-friendly', action='store_true', default=True,
                        help='Use colorblind friendly palette for visualizations (default: enabled)')
    
    parser.add_argument('--change-point-detection', action='store_true', default=False,
                        help='Perform change point detection in time series data')
    
    parser.add_argument('--clustering', action='store_true', default=False,
                        help='Perform K-means clustering on multivariate data')
    
    parser.add_argument('--recovery-analysis', action='store_true', default=False,
                        help='Analyze system recovery metrics after attack')
    
    parser.add_argument('--tenant-degradation', action='store_true', default=False,
                        help='Analyze cross-tenant relationships to identify sources of service degradation')
    
    # Causal analysis options
    parser.add_argument('--causal-analysis', action='store_true', default=False,
                        help='Run causal analysis between metrics')
                        
    parser.add_argument('--causal-method', type=str, default='toda-yamamoto',
                        choices=['toda-yamamoto', 'transfer-entropy', 'change-point-impact'],
                        help='Method to use for causal analysis (default: toda-yamamoto)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no analysis type specified, run all
    if not (args.metrics_analysis or args.phase_analysis or args.tenant_analysis):
        args.metrics_analysis = True
        args.phase_analysis = True
        args.tenant_analysis = True
        args.suggest_visualizations = True
    
    # Setup output directory
    if not args.output:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.output = f"analysis/{timestamp}/"
        
    return args

def setup_output_directory(output_path):
    """
    Setup output directory structure.
    
    Args:
        output_path (str): Path to create output directory
        
    Returns:
        Path: Path object for the base output directory
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "plots" / "correlations").mkdir(exist_ok=True)
    (output_dir / "plots" / "time_series").mkdir(exist_ok=True)
    (output_dir / "plots" / "distributions").mkdir(exist_ok=True)
    (output_dir / "plots" / "tenant_analysis").mkdir(exist_ok=True)
    
    (output_dir / "stats_results").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "metrics_analysis").mkdir(exist_ok=True)
    (output_dir / "phase_analysis").mkdir(exist_ok=True)
    (output_dir / "tenant_analysis").mkdir(exist_ok=True)
    (output_dir / "advanced_analysis").mkdir(exist_ok=True)
    
    logging.info(f"Created output directory structure at {output_dir}")
    return output_dir

def save_analysis_config(args, output_dir):
    """
    Save analysis configuration to file.
    
    Args:
        args: Command line arguments
        output_dir: Output directory path
    """
    config = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": args.experiment,
        "rounds": args.round,
        "phases": args.phases,
        "metrics_analysis": args.metrics_analysis,
        "phase_analysis": args.phase_analysis,
        "tenant_analysis": args.tenant_analysis,
        "skip_plots": args.skip_plots,
        "skip_advanced": args.skip_advanced
    }
    
    # Add optional parameters if present
    if args.metrics_of_interest:
        config["metrics_of_interest"] = args.metrics_of_interest
    
    if args.components:
        config["components"] = args.components
    
    # Save as JSON
    with open(output_dir / "analysis_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Also save as readable text
    with open(output_dir / "analysis_config.txt", 'w') as f:
        f.write("K8s Noisy Detection Analysis Configuration\n")
        f.write("=========================================\n\n")
        
        for key, value in config.items():
            if isinstance(value, list):
                f.write(f"{key}: {', '.join(value)}\n")
            else:
                f.write(f"{key}: {value}\n")

def main():
    """Main entry point for the pipeline."""
    print("K8s Noisy Neighbours Lab - Integrated Analysis Pipeline")
    print("======================================================")
    
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup base path
        base_path = Path('/home/phil/Projects/k8s-noisy-detection')
        
        # Special handling for demo data
        demo_data = False
        if args.experiment.startswith('demo-data'):
            demo_data = True
            print(f"Using demo data: {args.experiment}")
        
        logging.info(f"Using base path: {base_path}")
        logging.info(f"Experiment path: {args.experiment}")
        
        try:
            # Initialize data loader with special handling for demo data
            data_loader = DataLoader(base_path, args.experiment, args.round)
        except ValueError as e:
            if demo_data:
                print("Creating simple data loader for demo data...")
                # Create a more direct approach for demo data
                from data_loader import DataLoader
                # Monkey patch the DataLoader.__init__ method temporarily
                original_init = DataLoader.__init__
                
                def demo_init(self, base_path, experiment_name, round_number=None):
                    self.base_path = Path(base_path)
                    self.experiment_name = experiment_name
                    self.experiment_path = self.base_path / experiment_name
                    
                    # Handle round number options
                    self.rounds = []
                    if isinstance(round_number, list):
                        self.rounds = round_number
                    else:
                        self.rounds = [round_number]
                    
                    self.current_round = self.rounds[0] if self.rounds else None
                    self.results_path = self.experiment_path / self.current_round if self.current_round else None
                    self.data = {}
                    self.combined_data = {}
                    
                    logging.info(f"Demo data loader initialized for {experiment_name}, rounds: {self.rounds}")
                
                # Replace __init__ temporarily
                DataLoader.__init__ = demo_init
                
                # Try again with the patched version
                data_loader = DataLoader(base_path, args.experiment, args.round)
                
                # Restore original init
                DataLoader.__init__ = original_init
            else:
                # Re-raise if not demo data
                raise e
        
        # Set up output directory
        output_dir = setup_output_directory(args.output)
        save_analysis_config(args, output_dir)
        
        # Load data
        success = False
        
        # Check if we're using demo data
        if demo_data:
            try:
                # Try using the special demo data method
                phases_data = data_loader.load_demo_data()
                if phases_data:
                    success = True
                    print(f"Successfully loaded demo data with {len(phases_data)} phases")
                else:
                    print("Failed to load demo data")
            except Exception as e:
                print(f"Error loading demo data: {str(e)}")
                logging.error(f"Error loading demo data: {str(e)}")
        else:
            # Standard data loading
            if args.combine_rounds and len(args.round) > 1:
                phases_data = data_loader.load_all_rounds(args.phases)
                if not phases_data:
                    logging.error("Failed to load data from all rounds. Exiting.")
                    sys.exit(1)
                success = True
            else:
                phases_data = data_loader.load_all_phases(args.phases)
                if not phases_data:
                    logging.error("Failed to load phase data. Exiting.")
                    sys.exit(1)
                success = True
                
        if not success:
            logging.error("Failed to load any data. Exiting.")
            sys.exit(1)
        
        # Initialize analyzers
        metrics_analyzer = MetricsAnalyzer(output_dir)
        phase_analyzer = PhaseAnalyzer(output_dir)
        tenant_analyzer = TenantAnalyzer(output_dir)
        suggestion_engine = SuggestionEngine(output_dir)
        causal_analyzer = CausalAnalysisFixed(output_dir)
        
        # Package analyzers for pipeline manager
        analyzers = {
            "metrics": metrics_analyzer,
            "phase": phase_analyzer,
            "tenant": tenant_analyzer,
            "suggestion": suggestion_engine,
            "causal": causal_analyzer
        }
        
        # Initialize and run pipeline manager
        pipeline_manager = PipelineManager(data_loader, output_dir)
        pipeline_results = pipeline_manager.run_pipeline(analyzers, args)
        
        # Generate report summary
        generate_report_summary(pipeline_results, output_dir)
        
        print("\nAnalysis complete!")
        print(f"Results saved to: {output_dir}")
        print(f"Total execution time: {pipeline_results['execution_time']:.2f} seconds")
        print(f"Components executed: {', '.join(pipeline_results['components_executed'])}")
        
        return 0
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"\nERROR: {str(e)}")
        print("Check analysis_pipeline.log for details.")
        return 1

def generate_report_summary(pipeline_results, output_dir):
    """
    Generate a summary report of the analysis.
    
    Args:
        pipeline_results: Results from pipeline execution
        output_dir: Output directory path
    """
    # Create summary HTML
    html_path = output_dir / "analysis_summary.html"
    
    with open(html_path, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>K8s Noisy Neighbours Analysis Summary</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 1200px; margin: 0 auto; }
                h1 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
                h2 { color: #444; margin-top: 30px; }
                .section { margin-bottom: 30px; background: #f9f9f9; padding: 20px; border-radius: 5px; }
                .component { margin-bottom: 15px; }
                .results-list { list-style-type: none; padding-left: 20px; }
                .results-list li { margin-bottom: 8px; }
                .timestamp { color: #888; font-style: italic; }
                .summary { font-weight: bold; }
                .plot-gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }
                .plot-item { border: 1px solid #ddd; border-radius: 4px; overflow: hidden; }
                .plot-item img { width: 100%; height: auto; display: block; }
                .plot-caption { padding: 10px; background: #f5f5f5; }
            </style>
        </head>
        <body>
            <h1>K8s Noisy Neighbours Analysis Summary</h1>
            <p class="timestamp">Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            
            <div class="section">
                <h2>Execution Summary</h2>
                <p><span class="summary">Status:</span> """ + pipeline_results.get("status", "Unknown") + """</p>
                <p><span class="summary">Total execution time:</span> """ + f"{pipeline_results.get('execution_time', 0):.2f}" + """ seconds</p>
                <p><span class="summary">Analysis components:</span> """ + ", ".join(pipeline_results.get("components_executed", [])) + """</p>
            </div>
            
            <div class="section">
                <h2>Output Directory Structure</h2>
                <ul class="results-list">
                    <li><strong>plots/</strong> - All visualizations</li>
                    <li><strong>stats_results/</strong> - Statistical analysis results</li>
                    <li><strong>tables/</strong> - Data tables in various formats</li>
                    <li><strong>metrics_analysis/</strong> - Metrics-focused analysis results</li>
                    <li><strong>phase_analysis/</strong> - Phase comparison analysis results</li>
                    <li><strong>tenant_analysis/</strong> - Tenant-focused analysis results</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Key Visualizations</h2>
                <p>Selected visualizations from the analysis:</p>
                
                <div class="plot-gallery">
                    <!-- Plot items will be dynamically added when available -->
                    <!-- This is a placeholder that would be populated in a more advanced implementation -->
                    <div class="plot-item">
                        <div class="plot-caption">Key plots will appear here after the first analysis run</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Next Steps</h2>
                <ul class="results-list">
                    <li>Review the detailed analysis results in the output directories</li>
                    <li>Check the suggestions for further analysis opportunities</li>
                    <li>Compare results across different experiment runs</li>
                    <li>Use advanced analysis flags for deeper insights</li>
                </ul>
            </div>
        </body>
        </html>
        """)
    
    logging.info(f"Analysis summary generated at {html_path}")

if __name__ == "__main__":
    sys.exit(main())
