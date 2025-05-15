#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Pipeline for Kubernetes Noisy Neighbours Lab
Este módulo implementa o pipeline principal para análise de dados do experimento.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import json

# Importar módulos do pipeline
from data_loader import DataLoader
from metrics_analysis import MetricsAnalyzer
from phase_analysis import PhaseAnalyzer
from tenant_analysis import TenantAnalyzer
from visualizations import VisualizationGenerator
from suggestion_engine import SuggestionEngine

# Configurar logging
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
    parser = argparse.ArgumentParser(description='K8s Noisy Neighbours Lab Analysis Pipeline')
    
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
    
    # Tipos de análises
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
    
    parser.add_argument('--causality-method', type=str, default=None,
                        choices=['toda-yamamoto', 'transfer-entropy', 'change-point-impact'],
                        help='Method to use for causal analysis (if not specified, causal analysis is skipped)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Se não especificou tipo de análise, executar todas
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
    
    logging.info(f"Saved analysis configuration to {output_dir / 'analysis_config.txt'}")

def run_metrics_analysis(data_loader, metrics_analyzer, args):
    """
    Execute metrics-focused analysis.
    
    Args:
        data_loader: DataLoader instance with loaded data
        metrics_analyzer: MetricsAnalyzer instance
        args: Command line arguments
    """
    logging.info("Starting metrics-focused analysis...")
    
    # Get metrics for each phase
    all_phase_metrics = {}
    for phase in args.phases:
        logging.info(f"Analyzing phase: {phase}")
        
        # Get all metrics for this phase
        all_phase_metrics[phase] = data_loader.get_all_metrics_for_phase(phase)
        
        # Skip empty phases
        if not all_phase_metrics[phase]:
            logging.warning(f"No metrics found for phase {phase}, skipping")
            continue
            
        # Analyze each component in this phase
        for component in args.components:
            logging.info(f"Analyzing component: {component} in phase {phase}")
            
            # Get metrics for this component
            component_metrics = data_loader.get_all_metrics_for_component(phase, component)
            
            if not component_metrics:
                logging.warning(f"No metrics found for component {component} in phase {phase}, skipping")
                continue
                
            # Analyze each metric
            for metric_name, metric_data in component_metrics.items():
                if metric_data is None or (isinstance(metric_data, pd.DataFrame) and metric_data.empty):
                    continue
                    
                # Full name for identification
                full_name = f"{phase}_{component}_{metric_name}"
                
                # Descriptive statistics
                metrics_analyzer.get_descriptive_stats(metric_data, full_name)
                
                # Skip additional analyses if specified
                if args.skip_advanced:
                    continue
                    
                # Time series analysis
                if isinstance(metric_data, pd.DataFrame) and len(metric_data) > 10:
                    # Check stationarity
                    metrics_analyzer.check_stationarity(metric_data, full_name)
                    
                    # Select first numeric column for time series analysis
                    value_col = None
                    for col in metric_data.columns:
                        if pd.api.types.is_numeric_dtype(metric_data[col]):
                            value_col = col
                            break
                            
                    if value_col:
                        # Time series decomposition
                        metrics_analyzer.decompose_time_series(metric_data[value_col], full_name)
                        
                        # Anomaly detection if requested
                        if args.anomaly_detection:
                            metrics_analyzer.detect_anomalies(
                                metric_data[value_col], 
                                method=args.anomaly_detection,
                                threshold=3.0, 
                                metric_name=full_name
                            )
    
    # Get key metrics across phases for comparison
    if len(args.phases) >= 2 and args.metrics_of_interest:
        logging.info("Comparing key metrics across phases...")
        
        for component in args.components:
            for metric in args.metrics_of_interest:
                # Get this metric across all phases
                metrics_across_phases = {}
                
                for phase in args.phases:
                    comp_metrics = data_loader.get_all_metrics_for_component(phase, component)
                    if comp_metrics and metric in comp_metrics:
                        metrics_across_phases[phase] = comp_metrics[metric]
                
                # Only analyze if we have this metric in multiple phases
                if len(metrics_across_phases) >= 2:
                    # Create a dict of series for correlation analysis
                    series_dict = {}
                    for phase, data in metrics_across_phases.items():
                        if isinstance(data, pd.DataFrame) and not data.empty:
                            # Get first numeric column
                            value_col = None
                            for col in data.columns:
                                if pd.api.types.is_numeric_dtype(data[col]):
                                    value_col = col
                                    break
                                    
                            if value_col:
                                series_dict[phase] = data[value_col]
                    
                    # Correlation analysis between phases
                    if len(series_dict) >= 2:
                        metrics_analyzer.analyze_correlations(
                            series_dict,
                            title=f"{component}_{metric}_across_phases",
                            corr_method='pearson'
                        )
    
    # Generate suggestions if requested
    if args.suggest_visualizations:
        # Use first 10 metrics from the first phase for suggestions
        if all_phase_metrics and args.phases:
            first_phase = args.phases[0]
            if first_phase in all_phase_metrics:
                # Get all component metrics from this phase
                all_components = {}
                for component in args.components:
                    comp_metrics = data_loader.get_all_metrics_for_component(first_phase, component)
                    if comp_metrics:
                        all_components[component] = comp_metrics
                
                # Get suggestions for each component's metrics
                for component, metrics in all_components.items():
                    # Limit to first 5 metrics
                    for i, (metric_name, metric_data) in enumerate(metrics.items()):
                        if i >= 5:
                            break
                            
                        # Get plot suggestions
                        suggestions = metrics_analyzer.suggest_plots(metric_data, f"{component}_{metric_name}")
                        
                        if suggestions and suggestions != {"error": "Dados insuficientes para sugestões"}:
                            # Save suggestions to file
                            with open(metrics_analyzer.output_dir / f"{component}_{metric_name}_suggestions.txt", 'w') as f:
                                f.write(f"Visualization Suggestions for {component} - {metric_name}:\n\n")
                                
                                for viz_key, viz_data in suggestions.items():
                                    if viz_key != "error":
                                        f.write(f"- {viz_data['type']}: {viz_data['justification']}\n")
                                
                            # Get table suggestions
                            table_suggestions = metrics_analyzer.suggest_tables(metric_data, f"{component}_{metric_name}")
                            
                            if table_suggestions and table_suggestions != {"error": "Dados insuficientes para sugestões"}:
                                with open(metrics_analyzer.output_dir / f"{component}_{metric_name}_table_suggestions.txt", 'w') as f:
                                    f.write(f"Table Suggestions for {component} - {metric_name}:\n\n")
                                    
                                    for table_key, table_data in table_suggestions.items():
                                        if table_key != "error":
                                            formats = ", ".join(table_data['format']) if 'format' in table_data else "CSV"
                                            f.write(f"- {table_data['type']} ({formats}): {table_data['justification']}\n")
    
    logging.info("Metrics-focused analysis complete")

def run_phase_analysis(data_loader, phase_analyzer, args):
    """
    Execute phase comparison analysis.
    
    Args:
        data_loader: DataLoader instance with loaded data
        phase_analyzer: PhaseAnalyzer instance
        args: Command line arguments
    """
    logging.info("Starting phase comparison analysis...")
    
    # Get all metrics for each phase
    all_phase_metrics = {}
    for phase in args.phases:
        all_phase_metrics[phase] = data_loader.get_all_metrics_for_phase(phase)
    
    # Get specified metrics of interest or select key metrics
    metrics_to_analyze = args.metrics_of_interest if args.metrics_of_interest else [
        'cpu_usage', 'memory_usage', 'network_total_bandwidth', 'disk_io_total'
    ]
    
    # Compare phases for each component and metric
    for component in args.components:
        logging.info(f"Analyzing phase differences for component: {component}")
        
        for metric in metrics_to_analyze:
            # Get this metric across all phases
            metric_by_phase = {}
            
            for phase in args.phases:
                comp_metrics = data_loader.get_all_metrics_for_component(phase, component)
                if comp_metrics and metric in comp_metrics:
                    metric_by_phase[phase] = comp_metrics[metric]
            
            # Skip if we don't have enough phases to compare
            if len(metric_by_phase) < 2:
                continue
                
            # Statistical comparison
            phase_analyzer.compare_phases(metric_by_phase, f"{component}_{metric}")
            
            # Compare distributions
            if not args.skip_plots:
                phase_analyzer.compare_distributions(metric_by_phase, f"{component}_{metric}")
            
            # If we have baseline, attack, and recovery phases, analyze recovery patterns
            if len(args.phases) >= 3:
                # Try to identify phases by name
                baseline_phase = next((p for p in args.phases if "baseline" in p.lower()), None)
                attack_phase = next((p for p in args.phases if "attack" in p.lower()), None)
                recovery_phase = next((p for p in args.phases if "recovery" in p.lower()), None)
                
                # Only proceed if we have all three phases
                if baseline_phase and attack_phase and recovery_phase:
                    if (baseline_phase in metric_by_phase and 
                        attack_phase in metric_by_phase and
                        recovery_phase in metric_by_phase):
                        
                        phase_analyzer.analyze_recovery(metric_by_phase, 
                                                      baseline_phase, 
                                                      attack_phase, 
                                                      recovery_phase, 
                                                      metric)
    
    # Generate suggestions if requested
    if args.suggest_visualizations:
        # Get suggestions for each metric across phases
        for metric in metrics_to_analyze[:10]:  # Limit to first 10 metrics
            metric_by_phase = {}
            for phase in args.phases:
                all_components_with_metric = {}
                for component in args.components:
                    comp_metrics = data_loader.get_all_metrics_for_component(phase, component)
                    if comp_metrics and metric in comp_metrics:
                        all_components_with_metric[component] = comp_metrics[metric]
                
                if all_components_with_metric:
                    metric_by_phase[phase] = all_components_with_metric
                    
            if len(metric_by_phase) < 2:
                continue
                
            suggestions = phase_analyzer.suggest_analyses(metric_by_phase, metric)
            
            if suggestions and suggestions != {"error": "Dados insuficientes para sugestões de análise de fase"}:
                # Save suggestions to file
                with open(phase_analyzer.output_dir / f"{metric}_phase_suggestions.txt", 'w') as f:
                    f.write(f"Phase Analysis Suggestions for {metric}:\n\n")
                    
                    for analysis_key, analysis_data in suggestions.items():
                        if analysis_key != "error":
                            f.write(f"- {analysis_data['type']}: {analysis_data['justification']}\n")
                            if 'methods' in analysis_data:
                                f.write(f"  Methods: {', '.join(analysis_data['methods'])}\n")
                            f.write("\n")
    
    logging.info("Phase comparison analysis complete")

def run_tenant_analysis(data_loader, tenant_analyzer, args):
    """
    Execute tenant-focused analysis.
    
    Args:
        data_loader: DataLoader instance with loaded data
        tenant_analyzer: TenantAnalyzer instance
        args: Command line arguments
    """
    logging.info("Starting tenant-focused analysis...")
    
    # Get tenant data by phase
    tenant_data_by_phase = {}
    for phase in args.phases:
        tenant_data_by_phase[phase] = data_loader.get_tenant_metrics(phase)
    
    # For each phase, analyze tenant metrics
    for phase, tenant_data in tenant_data_by_phase.items():
        if not tenant_data or not tenant_data.get('metrics'):
            logging.warning(f"No tenant metrics found for phase {phase}")
            continue
            
        tenant_metrics = tenant_data['metrics']
        
        # Find common metrics across tenants
        common_metrics = set()
        for tenant, metrics in tenant_metrics.items():
            if not common_metrics:
                common_metrics = set(metrics.keys())
            else:
                common_metrics &= set(metrics.keys())
                
        logging.info(f"Found {len(common_metrics)} common metrics across tenants in phase {phase}")
        
        # Analyze key metrics across tenants
        for metric in list(common_metrics)[:10]:  # Limit to 10 metrics
            tenant_metric_data = {tenant: metrics[metric] for tenant, metrics in tenant_metrics.items()}
            
            # Compare tenants for this metric
            tenant_analyzer.compare_tenants(tenant_metric_data, metric, phase)
            
            # Analyze time series
            tenant_analyzer.analyze_tenant_time_series(tenant_metric_data, metric, phase)
            
            # Correlations between tenants
            if len(tenant_metric_data) >= 2:
                tenant_analyzer.correlation_between_tenants(tenant_metric_data, metric, phase)
    
    # If we have baseline and attack phases, run degradation analysis
    if (any("baseline" in p.lower() for p in args.phases) and
        any("attack" in p.lower() for p in args.phases)):
        
        # Find the actual phase names
        baseline_phase = next(p for p in args.phases if "baseline" in p.lower())
        attack_phase = next(p for p in args.phases if "attack" in p.lower())
        
        # Prepare nested data structure for degradation analysis
        # {tenant: {metric: {phase: data}}}
        tenant_metrics_for_degradation = {}
        
        # Extract tenant metrics from each phase
        for phase in [baseline_phase, attack_phase]:
            if phase in tenant_data_by_phase and tenant_data_by_phase[phase]:
                phase_tenant_metrics = tenant_data_by_phase[phase].get('metrics', {})
                
                for tenant, metrics in phase_tenant_metrics.items():
                    if tenant not in tenant_metrics_for_degradation:
                        tenant_metrics_for_degradation[tenant] = {}
                        
                    for metric_name, metric_data in metrics.items():
                        if metric_name not in tenant_metrics_for_degradation[tenant]:
                            tenant_metrics_for_degradation[tenant][metric_name] = {}
                            
                        tenant_metrics_for_degradation[tenant][metric_name][phase] = metric_data
        
        # Run degradation analysis
        if tenant_metrics_for_degradation:
            degradation_results = tenant_analyzer.analyze_tenant_degradation(
                tenant_metrics_for_degradation, baseline_phase, attack_phase, threshold=0.10)
    
    # Generate visualization suggestions if requested
    if args.suggest_visualizations:
        suggestions = tenant_analyzer.suggest_visualizations(tenant_metrics_for_degradation)
        
        if suggestions and suggestions != {"error": "Dados insuficientes para sugestões"}:
            # Save suggestions to file
            with open(tenant_analyzer.output_dir / "tenant_visualization_suggestions.txt", 'w') as f:
                f.write("Tenant Visualization Suggestions:\n\n")
                
                # Sort by priority if present
                items = list(suggestions.items())
                items.sort(key=lambda x: x[1].get('priority', 999))
                
                for viz_key, viz_data in items:
                    f.write(f"- {viz_data['type']}: {viz_data['description']}\n")
                    f.write(f"  Justification: {viz_data['justification']}\n")
                    f.write(f"  Function: {viz_data['function']}\n\n")
    
    logging.info("Tenant-focused analysis complete")

def main():
    """Main entry point for the pipeline."""
    args = parse_args()
    
    # Setup output directory
    output_dir = setup_output_directory(args.output)
    save_analysis_config(args, output_dir)
    
    # Initialize data loader
    base_path = Path.cwd().parent if Path.cwd().name == 'analysis_pipeline' else Path.cwd()
    data_loader = DataLoader(base_path, args.experiment, args.round)
    
    # Load data
    if args.combine_rounds and len(args.round) > 1:
        success = data_loader.load_data(combine=True, method=args.combine_method)
    else:
        success = data_loader.load_data()
        
    if not success:
        logging.error("Failed to load data. Exiting.")
        sys.exit(1)
        
    # Initialize analyzers
    metrics_analyzer = MetricsAnalyzer(output_dir)
    phase_analyzer = PhaseAnalyzer(output_dir)
    tenant_analyzer = TenantAnalyzer(output_dir)
    
    # Execute analyses based on arguments
    if args.metrics_analysis:
        run_metrics_analysis(data_loader, metrics_analyzer, args)
        
    if args.phase_analysis:
        run_phase_analysis(data_loader, phase_analyzer, args)
        
    if args.tenant_analysis:
        run_tenant_analysis(data_loader, tenant_analyzer, args)
        
    logging.info(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
