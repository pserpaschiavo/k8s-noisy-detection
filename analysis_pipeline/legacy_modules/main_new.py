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

# Importar módulos do pipeline
from data_loader import DataLoader
from metrics_analysis import MetricsAnalyzer
from phase_analysis import PhaseAnalyzer
from tenant_analysis import TenantAnalyzer
from visualizations import VisualizationGenerator

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
    
    parser.add_argument('--experiment', type=str, required=True,
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
                        help='List of key metrics to compare between phases')
    
    parser.add_argument('--components', type=str, nargs='+',
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
        output_path (str): Path to output directory
    
    Returns:
        Path: Path object for output directory
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Subdirectories
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "stats_results").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    
    return output_dir

def save_analysis_config(args, output_dir):
    """
    Save analysis configuration for reproducibility.
    
    Args:
        args: Parsed command line arguments
        output_dir (Path): Output directory path
    """
    config_path = output_dir / "analysis_config.txt"
    with open(config_path, 'w') as f:
        f.write(f"Experiment: {args.experiment}\n")
        f.write(f"Rounds: {' '.join(args.round)}\n")
        f.write(f"Combine Rounds: {args.combine_rounds}\n")
        
        if args.combine_rounds:
            f.write(f"Combine Method: {args.combine_method}\n")
            
        f.write(f"Phases: {' '.join(args.phases)}\n")
        f.write(f"Skip Plots: {args.skip_plots}\n")
        f.write(f"Skip Advanced: {args.skip_advanced}\n")
        
        if args.metrics_of_interest:
            f.write(f"Metrics of Interest: {' '.join(args.metrics_of_interest)}\n")
            
        if args.components:
            f.write(f"Components: {' '.join(args.components)}\n")
            
        f.write(f"Analysis Types:\n")
        f.write(f"  Metrics Analysis: {args.metrics_analysis}\n")
        f.write(f"  Phase Analysis: {args.phase_analysis}\n")
        f.write(f"  Tenant Analysis: {args.tenant_analysis}\n")
        f.write(f"  Visualizations Suggestions: {args.suggest_visualizations}\n")
        
    logging.info(f"Analysis configuration saved to {config_path}")

def run_metrics_analysis(data_loader, metrics_analyzer, args):
    """
    Execute metrics-focused analysis.
    
    Args:
        data_loader: DataLoader instance with loaded data
        metrics_analyzer: MetricsAnalyzer instance
        args: Command line arguments
    """
    logging.info("Starting metrics-focused analysis...")
    
    # Analisar métricas interessantes em todas as fases
    for phase in args.phases:
        phase_metrics = data_loader.get_phase_metrics(phase)
        
        if not phase_metrics:
            logging.warning(f"No metrics found for phase {phase}")
            continue
            
        # Process each metric in this phase
        for metric_name, metric_data in phase_metrics.items():
            # Skip if we have a filter and this metric isn't in it
            if args.metrics_of_interest and metric_name not in args.metrics_of_interest:
                continue
                
            logging.info(f"Analyzing metric: {metric_name} in phase {phase}")
            
            # Get descriptive statistics
            stats_df = metrics_analyzer.get_descriptive_stats(metric_data, f"{phase}_{metric_name}")
            
            # Check stationarity
            stationarity = metrics_analyzer.check_stationarity(metric_data, f"{phase}_{metric_name}")
            
            if not args.skip_advanced:
                # Decompose time series if enough data
                if len(metric_data) > 30:  # Minimum length for sensible decomposition
                    decomposition = metrics_analyzer.decompose_time_series(metric_data, f"{phase}_{metric_name}")
                
                # Detect anomalies
                anomalies = metrics_analyzer.detect_anomalies(metric_data, 'zscore', 3.0, f"{phase}_{metric_name}")
                
                # Calculate entropy
                entropy = metrics_analyzer.calculate_entropy(metric_data, 'sample', f"{phase}_{metric_name}")
    
    # Analyze correlations between metrics within each phase
    for phase in args.phases:
        phase_metrics = data_loader.get_phase_metrics(phase)
        
        if not phase_metrics or len(phase_metrics) < 2:
            continue
            
        # Filter metrics if specified
        if args.metrics_of_interest:
            filtered_metrics = {k: v for k, v in phase_metrics.items() if k in args.metrics_of_interest}
            if len(filtered_metrics) >= 2:
                corr_matrix = metrics_analyzer.analyze_correlations(filtered_metrics, f"{phase}_correlation", 'pearson')
        else:
            # If too many metrics, select a reasonable subset
            if len(phase_metrics) > 20:
                # Take the first 20 metrics as a sample
                sample_metrics = dict(list(phase_metrics.items())[:20])
                corr_matrix = metrics_analyzer.analyze_correlations(sample_metrics, f"{phase}_correlation_sample", 'pearson')
            else:
                corr_matrix = metrics_analyzer.analyze_correlations(phase_metrics, f"{phase}_correlation", 'pearson')
    
    # Check for cross-correlation between specific metrics if requested
    if args.metrics_of_interest and len(args.metrics_of_interest) >= 2:
        for phase in args.phases:
            phase_metrics = data_loader.get_phase_metrics(phase)
            
            # Get the first two metrics of interest that are available
            available_metrics = [m for m in args.metrics_of_interest if m in phase_metrics]
            
            if len(available_metrics) >= 2:
                for i in range(len(available_metrics)):
                    for j in range(i+1, len(available_metrics)):
                        m1, m2 = available_metrics[i], available_metrics[j]
                        metrics_analyzer.cross_correlation(
                            phase_metrics[m1], 
                            phase_metrics[m2], 
                            20, 
                            f"{phase} {m1} vs {m2}", 
                            m1, 
                            m2
                        )
    
    # Generate suggestions if requested
    if args.suggest_visualizations:
        for phase in args.phases:
            phase_metrics = data_loader.get_phase_metrics(phase)
            
            if not phase_metrics:
                continue
                
            # Create suggestions for key metrics
            key_metrics = list(phase_metrics.keys())[:10]  # Limit to first 10 metrics
            suggestions = {}
            
            for metric in key_metrics:
                metric_suggestions = metrics_analyzer.suggest_plots(phase_metrics[metric], metric)
                if metric_suggestions and metric_suggestions != {"error": "Dados insuficientes para sugestões"}:
                    suggestions[metric] = metric_suggestions
            
            # Save suggestions to file
            if suggestions:
                with open(metrics_analyzer.output_dir / f"{phase}_plot_suggestions.txt", 'w') as f:
                    f.write(f"Plot Suggestions for {phase}:\n\n")
                    
                    for metric, metric_suggestions in suggestions.items():
                        f.write(f"== {metric} ==\n")
                        for viz_key, viz_data in metric_suggestions.items():
                            f.write(f"- {viz_data['type']}: {viz_data['justification']}\n")
                        f.write("\n")
    
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
    
    # Get all metrics across phases
    all_phase_metrics = {}
    for phase in args.phases:
        all_phase_metrics[phase] = data_loader.get_phase_metrics(phase)
    
    # If metrics of interest is specified, use those, otherwise find common metrics across phases
    if args.metrics_of_interest:
        metrics_to_analyze = args.metrics_of_interest
    else:
        # Find metrics common to all phases
        common_metrics = set()
        for phase, metrics in all_phase_metrics.items():
            if not common_metrics:
                common_metrics = set(metrics.keys())
            else:
                common_metrics &= set(metrics.keys())
                
        metrics_to_analyze = list(common_metrics)
        
    logging.info(f"Analyzing {len(metrics_to_analyze)} metrics across phases")
    
    # Compare each metric across phases
    for metric in metrics_to_analyze:
        # Check if the metric exists in all phases
        metric_by_phase = {}
        for phase in args.phases:
            if metric in all_phase_metrics[phase]:
                metric_by_phase[phase] = all_phase_metrics[phase][metric]
                
        if len(metric_by_phase) < 2:
            logging.warning(f"Metric {metric} not found in enough phases for comparison")
            continue
            
        # Run comparison
        logging.info(f"Comparing phases for metric {metric}")
        comparison_methods = ['boxplot', 'violin', 'stats_test']
        
        if not args.skip_plots:
            comparison_methods.append('time_series')
            
        phase_analyzer.compare_phases(metric_by_phase, metric, args.phases, comparison_methods)
        
        # Detect change points if we have enough data and not skipping advanced analysis
        if not args.skip_advanced:
            for phase, data in metric_by_phase.items():
                if len(data) > 30:  # Minimum length for change point detection
                    phase_analyzer.detect_change_points(data, f"{metric}_{phase}")
    
    # Run recovery analysis if we have baseline, attack and recovery phases
    if (not args.skip_advanced and len(args.phases) >= 3 and
        any("baseline" in p.lower() for p in args.phases) and
        any("attack" in p.lower() for p in args.phases) and
        any("recovery" in p.lower() for p in args.phases)):
        
        # Find the actual phase names
        baseline_phase = next(p for p in args.phases if "baseline" in p.lower())
        attack_phase = next(p for p in args.phases if "attack" in p.lower())
        recovery_phase = next(p for p in args.phases if "recovery" in p.lower())
        
        # Analyze recovery for each metric
        for metric in metrics_to_analyze:
            # Check if the metric exists in all required phases
            if (metric in all_phase_metrics[baseline_phase] and
                metric in all_phase_metrics[attack_phase] and 
                metric in all_phase_metrics[recovery_phase]):
                
                metric_by_phase = {
                    baseline_phase: all_phase_metrics[baseline_phase][metric],
                    attack_phase: all_phase_metrics[attack_phase][metric],
                    recovery_phase: all_phase_metrics[recovery_phase][metric]
                }
                
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
                if metric in all_phase_metrics[phase]:
                    metric_by_phase[phase] = all_phase_metrics[phase][metric]
                    
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
        
        # First, examine the baseline phase
        if baseline_phase in tenant_data_by_phase and tenant_data_by_phase[baseline_phase]:
            baseline_tenant_metrics = tenant_data_by_phase[baseline_phase].get('metrics', {})
            
            for tenant, metrics in baseline_tenant_metrics.items():
                tenant_metrics_for_degradation[tenant] = {}
                
                for metric, data in metrics.items():
                    tenant_metrics_for_degradation[tenant][metric] = {baseline_phase: data}
        
        # Now add the attack phase data
        if attack_phase in tenant_data_by_phase and tenant_data_by_phase[attack_phase]:
            attack_tenant_metrics = tenant_data_by_phase[attack_phase].get('metrics', {})
            
            for tenant, metrics in attack_tenant_metrics.items():
                if tenant not in tenant_metrics_for_degradation:
                    tenant_metrics_for_degradation[tenant] = {}
                    
                for metric, data in metrics.items():
                    if metric not in tenant_metrics_for_degradation[tenant]:
                        tenant_metrics_for_degradation[tenant][metric] = {}
                        
                    tenant_metrics_for_degradation[tenant][metric][attack_phase] = data
        
        # Run the degradation analysis
        if len(tenant_metrics_for_degradation) >= 2:
            logging.info("Running tenant degradation analysis...")
            tenant_analyzer.analyze_tenant_degradation(
                tenant_metrics_for_degradation, 
                baseline_phase, 
                attack_phase, 
                threshold=0.10
            )
            
    # Generate suggestions if requested
    if args.suggest_visualizations:
        # Get tenant metrics for all phases to use in suggestions
        all_tenant_metrics = {}
        
        for phase, tenant_data in tenant_data_by_phase.items():
            if tenant_data and 'metrics' in tenant_data:
                for tenant, metrics in tenant_data['metrics'].items():
                    if tenant not in all_tenant_metrics:
                        all_tenant_metrics[tenant] = {}
                        
                    for metric, data in metrics.items():
                        if metric not in all_tenant_metrics[tenant]:
                            all_tenant_metrics[tenant][metric] = {}
                            
                        all_tenant_metrics[tenant][metric][phase] = data
        
        if all_tenant_metrics:
            suggestions = tenant_analyzer.suggest_visualizations(all_tenant_metrics)
            
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
