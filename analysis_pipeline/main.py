#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Pipeline for Kubernetes Noisy Neighbours Lab
This is the main entry point for the data analysis pipeline.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

# Import pipeline modules
from data_loader import DataLoader
from stats_summary import StatsAnalyzer
from time_series_analysis import TimeSeriesAnalyzer
from correlation_analysis import CorrelationAnalyzer
from visualizations import VisualizationGenerator
from advanced_analysis import AdvancedAnalyzer
from tenant_degradation_analysis import TenantDegradationAnalyzer, analyze_tenant_degradation
from causal_analysis import CausalAnalyzer

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
                        default=['1 - Baseline', '2 - Attack', '3 - Recovery'],
                        help='Phases to analyze (e.g., "1 - Baseline" "2 - Attack" "3 - Recovery")')
    
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results (default: analysis/YYYY-MM-DD_HH-MM-SS/)')
    
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip generating plots (useful for quick analysis)')
    
    parser.add_argument('--skip-advanced', action='store_true',
                        help='Skip advanced time series analysis (faster)')
    
    parser.add_argument('--metrics-of-interest', type=str, nargs='+',
                        default=['cpu_usage', 'memory_usage', 'network_total_bandwidth', 'disk_io_total'],
                        help='List of key metrics to focus on for cross-phase comparison')
    
    parser.add_argument('--components', type=str, nargs='+',
                        default=['tenant-a', 'tenant-b', 'tenant-c', 'tenant-d', 'ingress-nginx'],
                        help='List of components to analyze')
    
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
    
    return parser.parse_args()

def setup_output_dirs(base_dir=None):
    """Set up output directories for results."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Use provided output directory or create a timestamp-based one
    if base_dir:
        output_dir = Path(base_dir)
    else:
        # Salvar na pasta 'analysis' na raiz do repositório em vez de 'results/analysis'
        output_dir = Path('..') / 'analysis' / timestamp
    
    # Create main directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    stats_dir = output_dir / 'stats_results'
    stats_dir.mkdir(exist_ok=True)
    
    advanced_dir = output_dir / 'advanced_analysis'
    advanced_dir.mkdir(exist_ok=True)
    
    causal_dir = output_dir / 'causal_analysis'
    causal_dir.mkdir(exist_ok=True)
    
    return {
        'base': output_dir,
        'plots': plots_dir,
        'stats': stats_dir,
        'advanced': advanced_dir,
        'causal': causal_dir
    }

def analyze_phase(phase_data, phase_name, output_dirs, skip_plots=False):
    """Analyze a single phase."""
    logging.info(f"Analyzing phase: {phase_name}")
    
    # Initialize analyzers
    stats_analyzer = StatsAnalyzer(output_dirs['stats'])
    time_series_analyzer = TimeSeriesAnalyzer(output_dirs['stats'])
    corr_analyzer = CorrelationAnalyzer(output_dirs['plots'])
    vis_generator = VisualizationGenerator(output_dirs['plots'])
    
    # Skip visualizations if requested
    if not skip_plots:
        # Generate visualizations for each component in the phase
        vis_count = vis_generator.generate_phase_visualizations(phase_data, phase_name)
        logging.info(f"Generated {vis_count} visualizations for phase {phase_name}")
    
    # Statistical analysis
    stats_results = stats_analyzer.analyze_phase_metrics(phase_data, phase_name)
    stationarity_results = stats_analyzer.analyze_stationarity(phase_data)
    
    # Save statistical results
    for component, metrics_stats in stats_results.items():
        for metric, stats in metrics_stats.items():
            filename = f"{phase_name}_{component}_{metric}_stats.csv".replace(' ', '_')
            stats_analyzer.save_results_to_csv(stats, filename)
            
            # Also save as LaTeX for papers
            latex_filename = f"{phase_name}_{component}_{metric}_stats.tex".replace(' ', '_')
            stats_analyzer.save_results_to_latex(stats, latex_filename)
    
    # Correlation analysis for each component
    for component, component_data in phase_data.items():
        corr_matrix = corr_analyzer.analyze_component_correlations(
            component_data, component_name=component, method='pearson'
        )
        
        # Find and visualize strong correlations
        if not skip_plots and corr_matrix is not None and not corr_matrix.empty:
            # Extract series from component data for visualization
            series_dict = {}
            for metric, data in component_data.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    value_col = data.columns[0] if len(data.columns) > 0 else None
                    if value_col:
                        series_dict[metric] = data[value_col]
                        
            corr_analyzer.visualize_strong_correlations(corr_matrix, series_dict, threshold=0.7)
    
    # Advanced time series analysis for selected metrics
    # (limited to keep processing time reasonable)
    for component, component_data in phase_data.items():
        # Select a few key metrics for advanced analysis
        key_metrics = []
        if 'cpu_usage' in component_data:
            key_metrics.append(('cpu_usage', component_data['cpu_usage']))
        if 'memory_usage' in component_data:
            key_metrics.append(('memory_usage', component_data['memory_usage']))
        if 'network_total_bandwidth' in component_data:
            key_metrics.append(('network_total_bandwidth', component_data['network_total_bandwidth']))
        
        if len(key_metrics) >= 2:
            # Create series dict for analysis
            series_dict = {}
            for metric_name, metric_data in key_metrics:
                if isinstance(metric_data, pd.DataFrame) and not metric_data.empty:
                    value_col = metric_data.columns[0] if len(metric_data.columns) > 0 else None
                    if value_col:
                        series_dict[metric_name] = metric_data[value_col]
            
            # Analyze time series group if we have at least 2 metrics
            if len(series_dict) >= 2:
                time_series_analyzer.analyze_time_series_group(series_dict, f"{component}_{phase_name}")
    
    return {
        'stats': stats_results,
        'stationarity': stationarity_results
    }

def compare_phases(data_loader, phases, components_of_interest, metrics_of_interest, 
                  output_dirs, skip_plots=False, skip_advanced=False):
    """Compare metrics across different phases."""
    logging.info(f"Comparing phases: {phases}")
    
    # Initialize analyzers
    stats_analyzer = StatsAnalyzer(output_dirs['stats'])
    time_series_analyzer = TimeSeriesAnalyzer(output_dirs['stats'])
    vis_generator = VisualizationGenerator(output_dirs['plots'])
    
    phase_comparison_results = {}
    
    # Compare each metric of interest across phases
    for component in components_of_interest:
        component_results = {}
        
        for metric in metrics_of_interest:
            metrics_across_phases = data_loader.get_same_metric_across_phases(component, metric, phases)
            
            if len(metrics_across_phases) >= 2:
                # Statistical comparison
                comparison_result = stats_analyzer.compare_phases(metrics_across_phases, metric)
                component_results[metric] = comparison_result
                
                # Save comparison results
                filename = f"phase_comparison_{component}_{metric}.csv".replace(' ', '_')
                if comparison_result and 'comparisons' in comparison_result:
                    comparison_df = pd.DataFrame.from_dict(
                        {k: v for k, v in comparison_result['comparisons'].items()}, 
                        orient='index'
                    )
                    stats_analyzer.save_results_to_csv(comparison_df, filename)
                
                # Visualizations
                if not skip_plots:
                    # Line, box, and violin plots
                    vis_generator.plot_phase_comparison(
                        metrics_across_phases, metric_name=metric, component_name=component,
                        plot_type='line'
                    )
                    vis_generator.plot_phase_comparison(
                        metrics_across_phases, metric_name=metric, component_name=component,
                        plot_type='box'
                    )
                
                # Advanced time series comparison (entropy)
                if not skip_advanced:
                    time_series_analyzer.compare_entropy_across_phases(
                        metrics_across_phases, method='sample',
                        series_name=f"{component}_{metric}"
                    )
        
        phase_comparison_results[component] = component_results
    
    return phase_comparison_results

def perform_advanced_analysis(data_loader, phases, components_of_interest, metrics_of_interest, 
                             output_dirs, args):
    """Perform advanced statistical analyses."""
    logging.info("Performing advanced statistical analyses")
    
    # Initialize advanced analyzer
    adv_analyzer = AdvancedAnalyzer(output_dirs['advanced'])
    
    # Storage for results
    advanced_results = {
        'decomposition': {},
        'change_points': {},
        'anomalies': {},
        'distributions': {},
        'clustering': {},
        'recovery': {}
    }
    
    # Process each component and metric
    for component in components_of_interest:
        advanced_results['decomposition'][component] = {}
        advanced_results['change_points'][component] = {}
        advanced_results['anomalies'][component] = {}
        advanced_results['distributions'][component] = {}
        
        for phase in phases:
            # Get all metrics for this component in this phase
            all_metrics = data_loader.get_all_metrics_for_component(phase, component)
            
            # Filter to metrics of interest if specified
            metrics_to_process = metrics_of_interest if metrics_of_interest else list(all_metrics.keys())
            
            for metric in metrics_to_process:
                if metric in all_metrics:
                    metric_data = all_metrics[metric]
                    
                    # Skip empty dataframes
                    if metric_data is None or metric_data.empty:
                        continue
                    
                    logging.info(f"Analyzing {component}/{metric} in phase {phase}")
                    
                    # Get primary value column
                    value_col = metric_data.columns[0] if len(metric_data.columns) > 0 else None
                    
                    if value_col:
                        # Time series decomposition
                        if args.advanced_analysis:
                            decomp_result = adv_analyzer.time_series_decomposition(
                                metric_data, 
                                column=value_col,
                                component_name=component,
                                metric_name=metric
                            )
                            if decomp_result:
                                key = f"{phase}_{metric}"
                                advanced_results['decomposition'][component][key] = decomp_result
                        
                        # Change point detection
                        if args.change_point_detection:
                            cp_result = adv_analyzer.detect_change_points(
                                metric_data, 
                                column=value_col,
                                component_name=component,
                                metric_name=metric
                            )
                            if cp_result:
                                key = f"{phase}_{metric}"
                                advanced_results['change_points'][component][key] = cp_result
                        
                        # Anomaly detection
                        if args.anomaly_detection:
                            anomaly_result = adv_analyzer.detect_anomalies(
                                metric_data, 
                                column=value_col,
                                method=args.anomaly_detection,
                                component_name=component,
                                metric_name=metric
                            )
                            if anomaly_result:
                                key = f"{phase}_{metric}"
                                advanced_results['anomalies'][component][key] = anomaly_result
                        
                        # Distribution analysis
                        if args.distribution_analysis:
                            dist_result = adv_analyzer.distribution_analysis(
                                metric_data, 
                                column=value_col,
                                component_name=component,
                                metric_name=metric
                            )
                            if dist_result:
                                key = f"{phase}_{metric}"
                                advanced_results['distributions'][component][key] = dist_result
        
        # Clustering - group metrics together for each component/phase
        if args.clustering:
            logging.info(f"Performing clustering for component {component}")
            for phase in phases:
                # Get all metrics for this component in this phase
                all_metrics = data_loader.get_all_metrics_for_component(phase, component)
                
                # Create a dictionary of key metrics for clustering
                metrics_for_clustering = {}
                for metric in metrics_of_interest:
                    if metric in all_metrics:
                        metric_data = all_metrics[metric]
                        if metric_data is not None and not metric_data.empty and len(metric_data.columns) > 0:
                            value_col = metric_data.columns[0]
                            metrics_for_clustering[metric] = metric_data[value_col]
                
                # Perform clustering if we have enough metrics
                if len(metrics_for_clustering) >= 2:
                    cluster_result = adv_analyzer.perform_kmeans_clustering(
                        metrics_for_clustering,
                        component_name=f"{component}_{phase}",
                        metrics=list(metrics_for_clustering.keys())
                    )
                    if cluster_result:
                        advanced_results['clustering'][f"{component}_{phase}"] = cluster_result
    
    # Recovery analysis - compare same metric across phases
    if args.recovery_analysis and len(phases) >= 3:
        logging.info("Performing recovery analysis")
        advanced_results['recovery'] = {}
        
        for component in components_of_interest:
            advanced_results['recovery'][component] = {}
            
            for metric in metrics_of_interest:
                metrics_across_phases = data_loader.get_same_metric_across_phases(component, metric, phases)
                
                # Need all three phases for recovery analysis
                if len(metrics_across_phases) >= 3:
                    phase_keys = list(metrics_across_phases.keys())
                    phase_keys.sort()  # Ensure order: baseline, attack, recovery
                    
                    if len(phase_keys) >= 3:
                        before_attack = metrics_across_phases[phase_keys[0]]
                        during_attack = metrics_across_phases[phase_keys[1]]
                        after_attack = metrics_across_phases[phase_keys[2]]
                        
                        # Get primary value columns
                        before_col = before_attack.columns[0] if len(before_attack.columns) > 0 else None
                        during_col = during_attack.columns[0] if len(during_attack.columns) > 0 else None
                        after_col = after_attack.columns[0] if len(after_attack.columns) > 0 else None
                        
                        # Only proceed if we have valid data
                        if before_col and during_col and after_col:
                            recovery_result = adv_analyzer.analyze_recovery_metrics(
                                before_attack,
                                during_attack,
                                after_attack,
                                column=before_col,  # Assuming same column name in all phases
                                component_name=component,
                                metric_name=metric
                            )
                            if recovery_result:
                                advanced_results['recovery'][component][metric] = recovery_result
    
    return advanced_results

def perform_causal_analysis(data_loader, phases, components_of_interest, metrics_of_interest, 
                         output_dirs, causality_method):
    """Perform causal analysis between metrics."""
    logging.info(f"Performing causal analysis using method: {causality_method}")
    
    # Initialize causal analyzer with output directory
    causal_analyzer = CausalAnalyzer(output_dirs['stats'])
    
    # Prepare data structure for causal analysis
    phase_data = {}
    for phase in phases:
        phase_data[phase] = {}
        for component in components_of_interest:
            # Get metrics data for this component in this phase
            component_data = data_loader.get_all_metrics_for_component(phase, component)
            
            if component_data:
                # Extract series from component data
                series_dict = {}
                for metric, data in component_data.items():
                    if metric in metrics_of_interest and isinstance(data, pd.DataFrame) and not data.empty:
                        value_col = data.columns[0] if len(data.columns) > 0 else None
                        if value_col:
                            series_dict[metric] = data[value_col]
                
                if series_dict:
                    phase_data[phase][component] = series_dict
    
    # Run causal analysis
    results_df = causal_analyzer.run_causal_analysis(
        phase_data, 
        method=causality_method,
        metrics_of_interest=metrics_of_interest,
        components=components_of_interest,
        save_results=True
    )
    
    # Log summary of results
    if not results_df.empty and 'causality' in results_df.columns:
        causal_count = results_df['causality'].sum()
        logging.info(f"Found {causal_count} significant causal relationships using {causality_method} method")
        
        if causal_count > 0:
            # Print top causal relationships
            top_results = results_df[results_df['causality'] == True].head(10)
            print(f"\nTop causal relationships detected ({causality_method}):")
            
            # Format for display based on method
            if causality_method == 'toda-yamamoto':
                for _, row in top_results.iterrows():
                    print(f"- {row['source_metric']} → {row['target_metric']} "
                          f"(p-value: {row['p_value']:.4f}, lag: {row['lag_order']})")
                          
            elif causality_method == 'transfer-entropy':
                for _, row in top_results.iterrows():
                    print(f"- {row['source_metric']} → {row['target_metric']} "
                          f"(TE: {row['transfer_entropy']:.4f}, p-value: {row['p_value']:.4f})")
                          
            elif causality_method == 'change-point-impact':
                for _, row in top_results.iterrows():
                    print(f"- {row['source_metric']} → {row['target_metric']} "
                          f"(impact: {row['impact_strength']:.4f}, lag: {row['lag']})")
    else:
        logging.warning(f"No significant causal relationships found using {causality_method} method")
    
    return results_df

def generate_tenant_comparison(data_loader, phases, tenants, metrics_of_interest, output_dir, colorblind_friendly=True):
    """
    Generate tenant comparison plots across different phases.
    
    Args:
        data_loader: DataLoader object with loaded data
        phases: List of phases to include
        tenants: List of tenants to include
        metrics_of_interest: List of metrics to analyze
        output_dir: Output directory for plots
        colorblind_friendly: Use colorblind friendly palette for plots (default: True)
        
    Returns:
        dict: Dictionary with plot paths
    """
    logging.info(f"Generating tenant comparison plots for {len(tenants)} tenants across {len(phases)} phases")
    
    # Load data from all phases
    data = {}
    if hasattr(data_loader, 'data'):
        # Data is already loaded in the data_loader
        for phase in phases:
            if phase in data_loader.data:
                data[phase] = data_loader.data[phase]
            else:
                logging.warning(f"Phase {phase} not found in data_loader")
    else:
        # Need to load data
        logging.info("Loading all phases from data_loader...")
        data = data_loader.load_all_phases(phases)
    
    # Create visualizations with colorblind friendly option
    vis_generator = VisualizationGenerator(output_dir, colorblind_friendly=colorblind_friendly)
    logging.info(f"Visualization generator created with colorblind friendly mode: {colorblind_friendly}")
    
    # Define metrics to plot as (name, display_title) tuples
    metrics_to_plot = [
        ("cpu_usage", "CPU Usage (%)"),
        ("memory_usage", "Memory Usage (%)"),
        ("disk_io_total", "Disk I/O Operations"),
        ("network_total_bandwidth", "Network Bandwidth (bytes)")
    ]
    
    # Filter metrics to include only those in metrics_of_interest
    filtered_metrics = [(name, title) for name, title in metrics_to_plot 
                       if name in metrics_of_interest]
    
    # If no metrics match, use default metrics
    if not filtered_metrics:
        logging.warning(f"No metrics matched in metrics_of_interest. Using default metrics.")
        filtered_metrics = metrics_to_plot
    
    # Generate plots
    plot_paths = vis_generator.generate_tenant_comparison_plots(
        data=data,
        tenants=tenants,
        metrics_list=filtered_metrics,
        output_subdir='tenant_comparison'
    )
    
    return {"tenant_comparison_plots": plot_paths}

def detect_significant_changes(phase_comparison_results, p_value_threshold=0.05, effect_size_threshold=0.5):
    """Detect metrics with significant changes between phases."""
    significant_changes = []
    
    for component, metrics in phase_comparison_results.items():
        for metric, comparison in metrics.items():
            if 'comparisons' not in comparison:
                continue
                
            for comparison_key, results in comparison['comparisons'].items():
                if results.get('is_significant', False) and results.get('effect_size', 0) >= effect_size_threshold:
                    significant_changes.append({
                        'component': component,
                        'metric': metric,
                        'comparison': comparison_key,
                        'p_value': results.get('p_value'),
                        'effect_size': results.get('effect_size'),
                        'effect_magnitude': results.get('effect_magnitude'),
                        'percent_change': results.get('percent_change')
                    })
    
    # Convert to DataFrame and sort by effect size
    if significant_changes:
        significant_df = pd.DataFrame(significant_changes)
        significant_df = significant_df.sort_values('effect_size', ascending=False)
        return significant_df
    else:
        return pd.DataFrame()

def main():
    """Main function to run the analysis pipeline."""
    args = parse_args()
    
    # Determine the experiment path if not provided
    if not args.experiment:
        # Find the most recent experiment directory
        results_dir = Path('results')
        if not results_dir.exists():
            logging.error("Results directory not found. Please specify the experiment path.")
            return
        
        # Find latest date directory
        date_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
        if not date_dirs:
            logging.error("No experiment directories found in results/.")
            return
            
        latest_date = max(date_dirs, key=lambda d: d.name)
        
        # Find latest time directory
        time_dirs = [d for d in latest_date.iterdir() if d.is_dir() and d.name[0].isdigit()]
        if not time_dirs:
            logging.error(f"No time directories found in {latest_date}/.")
            return
            
        latest_time = max(time_dirs, key=lambda d: d.name)
        
        # Find experiment directory
        exp_dirs = [d for d in latest_time.iterdir() if d.is_dir()]
        if not exp_dirs:
            logging.error(f"No experiment directories found in {latest_time}/.")
            return
            
        latest_exp = exp_dirs[0]  # Just use the first one if multiple
        
        args.experiment = f"{latest_date.name}/{latest_time.name}/{latest_exp.name}"
        logging.info(f"Using latest experiment: {args.experiment}")
    
    # Set up output directories
    suffix = "_combined" if args.combine_rounds else ""
    if args.combine_rounds and args.combine_method != 'mean':
        suffix += f"_{args.combine_method}"
    
    if args.output:
        output_base = args.output + suffix
    else:
        output_base = None  # Will use default timestamped directory
    
    output_dirs = setup_output_dirs(output_base)
    
    # Initialize data loader
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    logging.info(f"Base path: {base_path}")
    
    # Check if multiple rounds are specified
    if args.combine_rounds and len(args.round) <= 1:
        logging.warning("--combine-rounds specified, but only one round provided. Will load single round without combining.")
        args.combine_rounds = False
    
    # Log the rounds being analyzed
    round_str = ", ".join(args.round)
    logging.info(f"Loading data from rounds: {round_str}")
    
    try:
        # Initialize data loader with possibly multiple rounds
        data_loader = DataLoader(base_path, args.experiment, args.round)
        
        # Combine rounds if requested
        if args.combine_rounds:
            logging.info(f"Combining {len(args.round)} rounds using method: {args.combine_method}")
            data_loader.combine_rounds(args.phases, args.combine_method)
            data_loader.use_combined_data(True)
            logging.info("Using combined data for analysis")
        else:
            # Load data from first specified round only
            data_loader.set_current_round(args.round[0])
            data = data_loader.load_all_phases(args.phases)
        
        logging.info(f"Loaded data for {len(data_loader.data)} phases: {list(data_loader.data.keys())}")
        
        # Log available components in each phase
        for phase in data_loader.data:
            components = list(data_loader.data[phase].keys())
            logging.info(f"Phase {phase} has {len(components)} components: {components}")
        
        # Analyze each phase
        phase_results = {}
        for phase in data_loader.data:
            phase_results[phase] = analyze_phase(
                data_loader.data[phase], phase, output_dirs, args.skip_plots
            )
        
        # Compare phases
        # Filter to only include components that exist in the data
        available_components = set()
        for phase in data_loader.data:
            available_components.update(data_loader.data[phase].keys())
        
        components_of_interest = [comp for comp in args.components if comp in available_components]
        logging.info(f"Comparing metrics across phases for components: {components_of_interest}")
        
        phase_comparison_results = compare_phases(
            data_loader, list(data_loader.data.keys()), components_of_interest, 
            args.metrics_of_interest, output_dirs, 
            args.skip_plots, args.skip_advanced
        )
        
        # Detect significant changes
        significant_changes = detect_significant_changes(phase_comparison_results)
        if not significant_changes.empty:
            logging.info(f"Detected {len(significant_changes)} metrics with significant changes between phases")
            
            # Save to CSV
            significant_file = output_dirs['stats'] / "significant_changes.csv"
            significant_changes.to_csv(significant_file)
            
            # Save to LaTeX for papers
            latex_file = output_dirs['stats'] / "significant_changes.tex"
            with open(latex_file, 'w') as f:
                f.write(significant_changes.to_latex())
            
            # Print top changes
            print("\nTop metrics with significant changes between phases:")
            pd.set_option('display.max_columns', None)
            print(significant_changes.head(10))
        else:
            logging.info("No significant changes detected between phases")
        
        # Perform advanced analysis if requested
        if (args.advanced_analysis or args.distribution_analysis or args.anomaly_detection or
            args.change_point_detection or args.clustering or args.recovery_analysis):
            
            logging.info("Starting advanced statistical analyses...")
            
            advanced_results = perform_advanced_analysis(
                data_loader, list(data_loader.data.keys()), 
                components_of_interest, args.metrics_of_interest,
                output_dirs, args
            )
            
            logging.info("Advanced analyses complete")
            
            # Summarize advanced analysis results
            summary_items = []
            
            for analysis_type, results in advanced_results.items():
                if isinstance(results, dict) and results:
                    count = sum(1 for comp in results.values() for metric in comp.values())
                    summary_items.append(f"{analysis_type}: {count} results")
            
            if summary_items:
                summary = "Advanced Analysis Summary:\n" + "\n".join(f"- {item}" for item in summary_items)
                
                # Print summary
                print("\n" + summary)
                
                # Save summary to file
                with open(output_dirs['advanced'] / "advanced_analysis_summary.txt", "w") as f:
                    f.write(summary)
        
        # Generate tenant comparison plots if requested
        if args.tenant_comparison:
            logging.info("Generating tenant comparison plots...")
            logging.info(f"Using colorblind friendly mode: {args.colorblind_friendly}")
            tenant_comparison_results = generate_tenant_comparison(
                data_loader, list(data_loader.data.keys()), 
                components_of_interest, args.metrics_of_interest, 
                output_dirs['plots'],
                colorblind_friendly=args.colorblind_friendly
            )
            logging.info("Tenant comparison plots generated")
        
        # Run causal analysis if requested
        if args.causality_method:
            logging.info(f"Running causal analysis with method: {args.causality_method}...")
            
            # Create a dedicated directory for causal analysis results
            causal_dir = output_dirs['base'] / "causal_analysis"
            causal_dir.mkdir(exist_ok=True)
            
            # Update output_dirs dictionary with new directory
            output_dirs['causal'] = causal_dir
            
            causal_results = perform_causal_analysis(
                data_loader, 
                list(data_loader.data.keys()),
                components_of_interest, 
                args.metrics_of_interest,
                output_dirs,
                args.causality_method
            )
            
            if not causal_results.empty and 'causality' in causal_results.columns:
                # Save a summary of causal relationships
                with open(causal_dir / "causal_analysis_summary.txt", "w") as f:
                    f.write(f"Causal Analysis Summary ({args.causality_method}):\n")
                    f.write(f"- Total relationships tested: {len(causal_results)}\n")
                    f.write(f"- Significant causal relationships found: {causal_results['causality'].sum()}\n\n")
                    
                    # Add details about top causal relationships
                    significant = causal_results[causal_results['causality'] == True]
                    if not significant.empty:
                        f.write("Top significant causal relationships:\n")
                        for i, (_, row) in enumerate(significant.head(10).iterrows(), 1):
                            f.write(f"{i}. {row['source_metric']} → {row['target_metric']}\n")
                
                logging.info("Causal analysis completed successfully")
            else:
                logging.warning("Causal analysis did not produce significant results")
        
        # Run tenant degradation analysis if requested
        if args.tenant_degradation:
            logging.info("Running tenant degradation analysis...")
            
            # Create a dedicated directory for tenant degradation analysis
            degradation_dir = output_dirs['base'] / "tenant_degradation_analysis"
            degradation_dir.mkdir(exist_ok=True)
            
            degradation_results = analyze_tenant_degradation(
                data_loader, 
                output_dir=degradation_dir
            )
            
            if degradation_results:
                logging.info("Tenant degradation analysis completed successfully")
                
                # Create a summary of degradation analysis
                summary = ["Tenant Degradation Analysis Summary:"]
                for metric, results in degradation_results.items():
                    if 'likely_degradation_sources' in results and results['likely_degradation_sources']:
                        sources = [s['tenant'] for s in results['likely_degradation_sources']]
                        summary.append(f"- {metric}: Likely degradation sources: {', '.join(sources)}")
                    else:
                        summary.append(f"- {metric}: No clear degradation sources identified")
                
                # Print summary
                print("\n" + "\n".join(summary))
                
                # Save summary to file
                with open(degradation_dir / "degradation_analysis_summary.txt", "w") as f:
                    f.write("\n".join(summary))
            else:
                logging.warning("Tenant degradation analysis did not produce results")
        
        # Add a summary of the analysis configuration
        with open(output_dirs['base'] / "analysis_config.txt", "w") as f:
            f.write(f"Experiment: {args.experiment}\n")
            if args.combine_rounds:
                f.write(f"Rounds: {round_str} (combined using {args.combine_method})\n")
            else:
                f.write(f"Round: {args.round[0]}\n")
            f.write(f"Phases: {', '.join(args.phases)}\n")
            f.write(f"Components analyzed: {', '.join(components_of_interest)}\n")
            f.write(f"Metrics of interest: {', '.join(args.metrics_of_interest)}\n")
            f.write(f"Advanced analyses: {args.advanced_analysis or args.distribution_analysis or args.anomaly_detection or args.change_point_detection or args.clustering or args.recovery_analysis or args.tenant_degradation}\n")
            if args.anomaly_detection:
                f.write(f"Anomaly detection method: {args.anomaly_detection}\n")
            if args.tenant_degradation:
                f.write(f"Tenant degradation analysis: Enabled\n")
            if args.causality_method:
                f.write(f"Causal analysis method: {args.causality_method}\n")
            f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logging.info(f"Analysis complete. Results saved to {output_dirs['base']}")
        
    except Exception as e:
        logging.error(f"Error in analysis pipeline: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
