"""
Main script for the data analysis pipeline of the noisy neighbors experiment.

This script orchestrates the entire pipeline, from data loading to
the generation of visualizations and reports.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import scipy.stats as stats

# Import pipeline modules
from pipeline.data_processing.consolidation import list_available_tenants, list_available_metrics
from pipeline.data_processing.consolidation import load_experiment_data, select_tenants, load_multiple_metrics
from pipeline.data_processing.time_normalization import add_elapsed_time, add_experiment_elapsed_time, add_phase_markers
from pipeline.data_processing.time_normalization import normalize_time
from pipeline.data_processing.aggregation import calculate_tenant_stats, calculate_inter_tenant_impact, calculate_recovery_effectiveness
from pipeline.data_processing.aggregation import aggregate_by_time, aggregate_data_by_custom_elements
from pipeline.data_processing.metric_normalization import (
    normalize_metrics_by_node_capacity, apply_normalization_to_all_metrics,
    auto_format_metrics
)

from pipeline.data_processing.quota_parser import (
    get_tenant_quotas, create_node_config_from_quotas,
    get_quota_summary, format_value_with_unit, convert_to_best_unit
)

from pipeline.analysis.tenant_analysis import calculate_correlation_matrix, compare_tenant_metrics
from pipeline.analysis.phase_analysis import compare_phases_ttest, analyze_recovery_effectiveness
from pipeline.analysis.advanced_analysis import calculate_covariance_matrix, calculate_entropy_metrics, calculate_normalized_impact_score
# from pipeline.analysis.advanced_analysis import granger_causality_test, analyze_causal_relationships # Commented out due to ImportError
from pipeline.analysis.tenant_analysis import calculate_inter_tenant_correlation_per_metric, calculate_inter_tenant_covariance_per_metric
from pipeline.analysis.inter_tenant_causality import identify_causal_chains, visualize_causal_graph
from pipeline.analysis.noisy_tenant_detection import identify_noisy_tenant
from pipeline.analysis.experiment_comparison import (
    load_multiple_experiments,
    preprocess_experiments,
    calculate_statistics_summary,
    compare_distributions,
    summarize_anomalies,
    compare_experiment_phases
)

from pipeline.visualization.plots import (plot_metric_by_phase, plot_phase_comparison,
                                plot_tenant_impact_heatmap, plot_recovery_effectiveness,
                                plot_impact_score_barplot, plot_impact_score_trend,
                                create_heatmap, plot_multivariate_anomalies, plot_correlation_heatmap,
                                plot_entropy_heatmap, plot_entropy_top_pairs_barplot)

from pipeline.visualization.table_generator import (export_to_latex, export_to_csv,
                                         create_phase_comparison_table, create_impact_summary_table,
                                         convert_df_to_markdown, create_causality_results_table)

from pipeline.config import (DEFAULT_DATA_DIR, DEFAULT_METRICS, AGGREGATION_CONFIG,
                    IMPACT_CALCULATION_DEFAULTS, VISUALIZATION_CONFIG,
                    NODE_RESOURCE_CONFIGS, DEFAULT_NODE_CONFIG_NAME,
                    DEFAULT_CAUSALITY_MAX_LAG, DEFAULT_CAUSALITY_THRESHOLD_P_VALUE,
                    DEFAULT_METRICS_FOR_CAUSALITY, CAUSALITY_METRIC_COLORS, PHASE_DISPLAY_NAMES,
                    METRIC_DISPLAY_NAMES, DEFAULT_NOISY_TENANT)
from pipeline.analysis.application_metrics_analysis import (
    analyze_latency_impact, analyze_error_rate_correlation, calculate_application_slo_violations
)
from pipeline.analysis.technology_comparison import (
    normalize_metrics_between_experiments, calculate_relative_efficiency,
    plot_experiment_comparison, compare_technologies
)
from pipeline.analysis.rounds_aggregation import (
    aggregate_metrics_across_rounds, 
    plot_aggregated_metrics,
    plot_aggregated_metrics_boxplot, # Add import
    test_for_significant_differences
)
from dotenv import load_dotenv


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Data analysis pipeline for the noisy neighbors experiment.')
    parser.add_argument('--data-dir', type=str, default='/home/phil/Projects/k8s-noisy-detection/demo-data/demo-experiment-3-rounds',
                        help='Directory with the experiment data')
    parser.add_argument('--data-dir-comparison', type=str, nargs='+',
                        help='Additional directory(s) to compare multiple experiments. Used with --compare-experiments.')
    parser.add_argument('--comparison-names', type=str, nargs='+',
                        help='Names for the comparison experiments. Must match the number of directories in --data-dir-comparison.')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save the results')
    parser.add_argument('--tenants', type=str, nargs='+',
                        help='Specific tenant(s) to analyze')
    parser.add_argument('--noisy-tenant', type=str, 
                        help='Specific tenant generating noise (default: tenant-b)')
    parser.add_argument('--auto-detect-noisy', action='store_true',
                        help='Automatically detect which tenant is the noise generator')
    parser.add_argument('--metrics', type=str, nargs='+',
                        help='Specific metric(s) to analyze')
    parser.add_argument('--phases', type=str, nargs='+',
                        help='Specific phase(s) to analyze')
    parser.add_argument('--rounds', type=str, nargs='+',
                        help='Specific round(s) to analyze')
    parser.add_argument('--show-as-percentage', action='store_true',
                        help='Display metrics as a percentage of total cluster capacity')
    parser.add_argument('--advanced', action='store_true',
                        help='Run advanced analyses (covariance, entropy, causality)')
    parser.add_argument('--compare-experiments', action='store_true',
                        help='Compare multiple experiments')
    parser.add_argument('--generate-reports', action='store_true',
                        help='Generate reports in Markdown, LaTeX, and HTML')
    parser.add_argument('--elements-to-aggregate', type=str, nargs='+',
                        help='Specific element(s) to aggregate (e.g., tenant-a, tenant-b, ingress-nginx)')
    parser.add_argument('--node-config', type=str, default=None,
                        help='Name of the node configuration to use (e.g., Default, Limited). Overrides .env.')
    parser.add_argument('--use-quotas-for-normalization', action='store_true', default=False,
                        help='Use quota-based configuration for metric normalization.')
    parser.add_argument('--app-metrics-analysis', action='store_true',
                        help='Run application-level metrics analysis')
    parser.add_argument('--slo-thresholds', type=str, 
                        help='JSON file with SLO threshold definitions per metric')
    parser.add_argument('--compare-technologies', action='store_true',
                        help='Compare experiments with different technologies')
    parser.add_argument('--technology-names', type=str, nargs='+',
                        help='Names of the technologies being compared (e.g., Docker Vanilla, Kata Containers)')
    parser.add_argument('--inter-tenant-causality', action='store_true',
                        help='Run inter-tenant causality analysis.')
    parser.add_argument('--compare-rounds-intra', action='store_true',
                        help='Run formal comparison between rounds of the same experiment.')
    parser.add_argument('--compare-tenants-directly', action='store_true',
                        help='Run direct statistical comparison between tenants.')
    parser.add_argument('--entropy-plot-type', type=str, default='all',
                        choices=['heatmap', 'barplot', 'all'],
                        help='Plot type for entropy results: heatmap, barplot (top N), or all.')
    parser.add_argument('--compare-experiments-rounds', type=str, nargs='+',
                        help='Specific round(s) to use for --compare-experiments')
    parser.add_argument('--compare-experiments-tenants', type=str, nargs='+',
                        help='Specific tenant(s) to use for --compare-experiments')
    parser.add_argument('--use-aggregated-rounds-for-advanced', action='store_true',
                        help='Use data aggregated across rounds for advanced analyses.')
    parser.add_argument('--granger-max-lag', type=int, default=5,
                        help='Maximum lag for Granger causality tests.')
    
    return parser.parse_args()


def setup_output_directories(output_dir):
    """Set up output directories."""
    plots_dir = os.path.join(output_dir, 'plots')
    tables_dir = os.path.join(output_dir, 'tables')
    advanced_dir = os.path.join(output_dir, 'advanced')
    comparison_dir = os.path.join(output_dir, 'comparisons')
    reports_dir = os.path.join(output_dir, 'reports')
    causality_dir = os.path.join(output_dir, 'causality')
    rounds_comparison_intra_dir = os.path.join(output_dir, 'rounds_comparison_intra')
    tenant_comparison_dir = os.path.join(output_dir, 'tenant_comparison') # New directory
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(advanced_dir, exist_ok=True)
    os.makedirs(os.path.join(advanced_dir, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(advanced_dir, 'plots'), exist_ok=True) # Ensure advanced plots directory is created
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(causality_dir, exist_ok=True)
    os.makedirs(os.path.join(causality_dir, 'tables'), exist_ok=True)  # Ensure causality tables directory is created
    os.makedirs(os.path.join(causality_dir, 'plots'), exist_ok=True)  # Ensure causality plots directory is created
    os.makedirs(rounds_comparison_intra_dir, exist_ok=True)
    os.makedirs(os.path.join(rounds_comparison_intra_dir, 'plots'), exist_ok=True)
    os.makedirs(tenant_comparison_dir, exist_ok=True) # Create new directory
    os.makedirs(os.path.join(tenant_comparison_dir, 'tables'), exist_ok=True) # Create tables subdirectory
    
    return plots_dir, tables_dir, advanced_dir, comparison_dir, reports_dir, causality_dir, rounds_comparison_intra_dir, tenant_comparison_dir


def compare_rounds_within_experiment(experiment_results, output_dir_main, metrics_to_compare=None, phases_to_compare=None, 
                            show_as_percentage=False, node_config=None):
    """
    Formally compares different rounds within the same experiment for specified metrics and phases.
    Performs ANOVA tests, generates bar plots, and saves aggregated results (e.g., mean per round) to CSV.
    Returns a dictionary with paths to CSVs, plots, and ANOVA statistics for the report.
    
    Args:
        experiment_results: Experiment results
        output_dir_main: Main output directory
        metrics_to_compare: List of metrics to compare
        phases_to_compare: List of phases to compare
        show_as_percentage: If True, displays values as percentages of total capacity
        node_config: Node configuration with total resource capacities
    """
    print("\nStarting Comparison Between Rounds of the Same Experiment...")
    
    rounds_comparison_output_dir = os.path.join(output_dir_main, "rounds_comparison_intra")
    plots_subdir = os.path.join(rounds_comparison_output_dir, "plots")

    processed_data = experiment_results.get('processed_data')
    all_comparison_outputs = {}

    if not processed_data:
        print("No processed data available for comparison between rounds.")
        return all_comparison_outputs

    # Default metrics and phases if not provided
    if metrics_to_compare is None:
        metrics_to_compare = DEFAULT_METRICS # Use default metrics from config if none are specified
    if phases_to_compare is None:
        # This default should align with how phases_to_compare_rounds is set in main()
        # It expects raw phase names like "2 - Attack"
        phases_to_compare = ["2 - Attack"] 

    for metric_name in metrics_to_compare:
        metric_display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
        if metric_name not in processed_data:
            print(f"Metric {metric_name} not found in processed data. Skipping round comparison for this metric.")
            continue
        
        metric_df = processed_data[metric_name]
        if not all(col in metric_df.columns for col in ['phase', 'round', 'value']):
            print(f"Columns 'phase', 'round', or 'value' not found in the DataFrame for metric {metric_name}. Skipping.")
            continue

        for raw_phase_name in phases_to_compare: # Iterate over raw phase names
            # Get the display name for reporting and filenames
            current_phase_display_for_report = PHASE_DISPLAY_NAMES.get(raw_phase_name, raw_phase_name)
            print(f"  Comparing rounds for Metric: {metric_display_name}, Phase: {current_phase_display_for_report} (Raw: {raw_phase_name})")
            
            # Filter using the raw phase name
            phase_specific_df = metric_df[metric_df['phase'] == raw_phase_name]
            
            output_key = f"{metric_name}_{current_phase_display_for_report.replace(' ', '_')}"
            current_output = {"csv_path": None, "plot_path": None, "anova_f_stat": None, "anova_p_value": None}

            if not phase_specific_df.empty:
                # Check if there is enough data for each round
                rounds_count = phase_specific_df['round'].value_counts()
                valid_rounds = rounds_count[rounds_count >= 5].index.tolist()
                
                if not valid_rounds:
                    print(f"    Not enough rounds with sufficient data for analysis. Skipping.")
                    continue
                    
                # Filter only for rounds with sufficient data
                phase_specific_df_filtered = phase_specific_df[phase_specific_df['round'].isin(valid_rounds)]
                
                # Aggregate data for CSV and plotting
                if 'tenant' in phase_specific_df_filtered.columns and len(phase_specific_df_filtered['tenant'].unique()) > 1:
                    comparison_data_agg = phase_specific_df_filtered.groupby(['round', 'tenant'])['value'].mean().reset_index()
                    comparison_data = comparison_data_agg.groupby('round')['value'].mean().reset_index()
                    comparison_data.rename(columns={'value': f'mean_value_across_tenants'}, inplace=True)
                else:
                    comparison_data = phase_specific_df_filtered.groupby('round')['value'].mean().reset_index()
                    comparison_data.rename(columns={'value': 'mean_value'}, inplace=True)
                
                print(f"    Mean values of {metric_display_name} in phase {current_phase_display_for_report} per round:")
                print(comparison_data)
                
                # Use current_phase_display_for_report for filenames
                csv_filename = f"{metric_name}_{current_phase_display_for_report.replace(' ', '_')}_round_comparison.csv"
                csv_path = os.path.join(rounds_comparison_output_dir, csv_filename)
                try:
                    comparison_data.to_csv(csv_path, index=False)
                    print(f"    Comparison (CSV) saved to: {csv_path}")
                    current_output["csv_path"] = csv_path
                except Exception as e:
                    print(f"    Error saving round comparison CSV: {e}")

                # Perform ANOVA
                rounds_with_data = phase_specific_df['round'].unique()
                if len(rounds_with_data) >= 2:
                    grouped_values = [
                        group['value'].dropna() for name, group in phase_specific_df.groupby('round')
                        if not group['value'].dropna().empty
                    ]
                    if len(grouped_values) >= 2:
                        try:
                            pass # Added pass to fix empty try block
                        except Exception as e:
                            print(f"    Error performing ANOVA (exception during attempt): {e}")
                            pass # Added pass to fix empty except block
                    else:
                        print("    Not enough groups with data for ANOVA after filtering.")
                else:
                    print("    Not enough rounds with data to perform ANOVA.")

                # Generate and save bar plot
                if not comparison_data.empty:
                    plt.figure(figsize=VISUALIZATION_CONFIG.get('figure_size', (10, 6)))
                    value_col_for_plot = 'mean_value_across_tenants' if 'mean_value_across_tenants' in comparison_data.columns else 'mean_value'
                    
                    plt.bar(comparison_data['round'].astype(str), comparison_data[value_col_for_plot])
                    # Use current_phase_display_for_report for plot title
                    plt.title(f'Mean {metric_display_name} per Round during {current_phase_display_for_report}')
                    plt.xlabel('Round')
                    
                    # Configure Y-axis label depending on percentage display
                    if show_as_percentage and node_config:
                        if metric_name == 'cpu_usage' and 'CPUS' in node_config:
                            unit_info = f"% of {node_config['CPUS']} CPU cores"
                        elif metric_name == 'memory_usage' and 'MEMORY_GB' in node_config:
                            unit_info = f"% of {node_config['MEMORY_GB']} GB memory"
                        elif metric_name == 'disk_throughput_total' and 'DISK_SIZE_GB' in node_config:
                            unit_info = f"% of max throughput"
                        elif metric_name == 'network_total_bandwidth' and 'NETWORK_BANDWIDTH_MBPS' in node_config:
                            unit_info = f"% of {node_config['NETWORK_BANDWIDTH_MBPS']} Mbps"
                        else:
                            unit_info = "%"
                        plt.ylabel(f'Mean {metric_display_name} ({unit_info})')
                    else:
                        plt.ylabel(f'Mean {metric_display_name}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Use current_phase_display_for_report for plot filename
                    plot_filename = f"{metric_name}_{current_phase_display_for_report.replace(' ', '_')}_round_comparison_plot.png"
                    plot_path = os.path.join(plots_subdir, plot_filename)
                    try:
                        plt.savefig(plot_path)
                        print(f"    Round comparison plot saved to: {plot_path}")
                        current_output["plot_path"] = plot_path
                    except Exception as e:
                        print(f"    Error saving round comparison plot: {e}")
                    plt.close()
                all_comparison_outputs[output_key] = current_output
            else:
                print(f"    No data for metric {metric_display_name} in phase {current_phase_display_for_report} for round comparison.")
    
    print("Comparison Between Rounds of the Same Experiment completed.")
    return all_comparison_outputs


def run_application_metrics_analysis(metrics_dict, app_metrics_dict=None, noisy_tenant="tenant-b", slo_thresholds=None, output_dir=None):
    """
    Runs application metrics analysis.
    
    Args:
        metrics_dict: Dictionary with infrastructure metrics
        app_metrics_dict: Dictionary with application metrics (optional)
        noisy_tenant: The tenant identified as noisy
        slo_thresholds: Thresholds for SLOs per metric
        output_dir: Directory to save results
        
    Returns:
        Dict: Results of application metrics analysis
    """
    results = {}
    
    # If no application metrics, use infrastructure metrics
    if app_metrics_dict is None:
        app_metrics_dict = metrics_dict
    
    print("\n=== Running Application Metrics Analysis ===")
    
    # 1. Latency Impact Analysis
    if any(metric.startswith('latency') or metric == 'response_time' for metric in app_metrics_dict.keys()):
        print("Analyzing latency impact...")
        latency_impact = analyze_latency_impact(app_metrics_dict, metrics_dict, noisy_tenant)
        results['latency_impact'] = latency_impact
        
        # Display results
        print(f"\nLatency impact per tenant caused by {noisy_tenant}:")
        for tenant, impact in latency_impact.items():
            sig = "✓" if impact['significant_impact'] else "✗"
            print(f"  {tenant}: {impact['increase_percentage']:.2f}% ({sig} p={impact['p_value']:.4f})")
    
    # 2. Error Rate Correlation
    if any(metric.startswith('error') for metric in app_metrics_dict.keys()):
        print("\nAnalyzing correlation between CPU usage and error rates...")
        error_correlations = analyze_error_rate_correlation(app_metrics_dict, metrics_dict, noisy_tenant)
        results['error_correlations'] = error_correlations
        
        # Display results
        print(f"\nCorrelation between CPU usage of {noisy_tenant} and error rates:")
        for tenant, corr in error_correlations.items():
            print(f"  {tenant}: {corr:.4f}")
    
    # 3. SLO Violation Analysis
    if slo_thresholds:
        print("\nAnalyzing SLO violations...")
        slo_violations = calculate_application_slo_violations(app_metrics_dict, slo_thresholds, noisy_tenant)
        results['slo_violations'] = slo_violations
        
        # Display results
        print(f"\nSLO violations per tenant:")
        for tenant, violations in slo_violations.items():
            print(f"  {tenant}:")
            for metric, stats in violations.items():
                increase = stats['violation_increase'] * 100
                print(f"    {metric}: +{increase:.2f}% of violations during attack")
    
    return results


def run_technology_comparison(exp1_data, exp2_data, metrics_list=None, tenants_list=None, 
                           exp1_name="Technology 1", exp2_name="Technology 2",
                           output_dir=None, skip_plots=False):
    """
    Runs comparison between experiments with different technologies.
    
    Args:
        exp1_data: Data from the first experiment
        exp2_data: Data from the second experiment  
        metrics_list: List of metrics for analysis
        tenants_list: List of tenants to filter
        exp1_name: Name of the first technology
        exp2_name: Name of the second technology
        output_dir: Directory to save results
        skip_plots: If True, do not generate visualizations
        
    Returns:
        Dict: Comparison results
    """
    print(f"\n=== Comparing Technologies: {exp1_name} vs {exp2_name} ===")
    
    # Ensure we have common metrics to compare
    if metrics_list is None:
        metrics_list = [m for m in exp1_data.keys() if m in exp2_data.keys()]
        print(f"Common metrics for comparison: {metrics_list}")
    
    # Perform comparison
    comparison_results = compare_technologies(
        exp1_data, exp2_data, 
        metrics_list=metrics_list,
        tenants_list=tenants_list,
        exp1_name=exp1_name, 
        exp2_name=exp2_name,
        output_dir=output_dir,
        generate_plots=True
    )
    
    # Show summary of results
    efficiency_metrics = comparison_results['efficiency_metrics']['all_phases']
    print("\nSummary of significant differences:")
    
    significant_differences = efficiency_metrics[efficiency_metrics['statistically_significant']]
    if not significant_differences.empty:
        for _, row in significant_differences.iterrows():
            better = row['better_experiment']
            worse = exp2_name if better == exp1_name else exp2_name
            print(f"  {row['metric']} ({row['tenant']}): {better} is better than {worse} by {abs(row['percent_difference']):.2f}%")
    else:
        print("  No statistically significant differences found")
    
    return comparison_results


def main():
    """Main function that runs the analysis pipeline."""
    args = parse_arguments()
    
    plots_dir, tables_dir, advanced_dir, comparison_dir, reports_dir, causality_dir, rounds_comparison_intra_dir, tenant_comparison_dir = setup_output_directories(args.output_dir)
    
    experiment_data_dir_input = args.data_dir
    experiment_data_dir = ""

    if os.path.isabs(experiment_data_dir_input):
        experiment_data_dir = experiment_data_dir_input
    else:
        cwd = os.getcwd()
        path_from_cwd = os.path.join(cwd, experiment_data_dir_input)
        path_from_cwd_with_demodata_prefix = os.path.join(cwd, "demo-data", experiment_data_dir_input)

        if os.path.isdir(path_from_cwd):
            experiment_data_dir = path_from_cwd
        elif os.path.isdir(path_from_cwd_with_demodata_prefix) and not experiment_data_dir_input.startswith("demo-data"):
            experiment_data_dir = path_from_cwd_with_demodata_prefix
        else:
            experiment_data_dir = path_from_cwd

    experiment_data_dir = os.path.normpath(experiment_data_dir)

    if not os.path.isdir(experiment_data_dir):
        print(f"Error: Data directory not found: {experiment_data_dir}")
        print(f"  Input --data-dir: '{args.data_dir}'")
        print(f"  Absolute path checked: '{os.path.abspath(experiment_data_dir)}'")
        if not os.path.isabs(args.data_dir):
            cwd_for_hint = os.getcwd()
            print(f"  Check if '{args.data_dir}' exists relative to '{cwd_for_hint}'")
            if not args.data_dir.startswith("demo-data"):
                 print(f"  or if '{os.path.join('demo-data', args.data_dir)}' exists relative to '{cwd_for_hint}'")
        return

    print(f"Using data directory: {experiment_data_dir}")

    # Try to load node configuration from quotas
    quota_file_path = os.path.join(os.path.dirname(os.path.dirname(experiment_data_dir)), 'resource-quotas.yaml')
    
    # Default node configuration
    node_config_to_use = NODE_RESOURCE_CONFIGS.get(DEFAULT_NODE_CONFIG_NAME, NODE_RESOURCE_CONFIGS.get("Default"))
    
    # Flag to indicate if we are using quotas - checking command line argument
    using_quotas = args.use_quotas_for_normalization
    
    # Create an adapted copy of NODE_RESOURCE_CONFIGS for the format expected by normalization modules
    if 'MEMORY_GB' in node_config_to_use and 'MEMORY_BYTES' not in node_config_to_use:
        node_config_to_use = node_config_to_use.copy()  # Avoid altering the original
        node_config_to_use['MEMORY_BYTES'] = node_config_to_use['MEMORY_GB'] * (2**30)
        node_config_to_use['DISK_SIZE_BYTES'] = node_config_to_use['DISK_SIZE_GB'] * (2**30)
        # Estimate bandwidth based on CPU 
        node_config_to_use['NETWORK_BANDWIDTH_MBPS'] = max(1000, node_config_to_use['CPUS'] * 250)
    
    if os.path.exists(quota_file_path):
        print(f"Quota file found: {quota_file_path}")
        quota_based_config = create_node_config_from_quotas(quota_file_path)
        
        if using_quotas and quota_based_config and quota_based_config['CPUS'] > 0:
            print(f"Using quota-based configuration for metric normalization")
            print(f"  Total CPU: {format_value_with_unit(quota_based_config['CPUS'], 'cpu')}")
            print(f"  Total Memory: {format_value_with_unit(quota_based_config['MEMORY_BYTES'], 'memory')}")
            print(f"  Estimated Storage: {format_value_with_unit(quota_based_config['DISK_SIZE_BYTES'], 'disk')}")
            print(f"  Estimated Bandwidth: {format_value_with_unit(quota_based_config['NETWORK_BANDWIDTH_MBPS']*1e6/8, 'network')}")
            print(f"  Margin for system resources: {quota_based_config['SYSTEM_RESOURCES_MARGIN']:.1f}%")
            
            node_config_to_use = quota_based_config
        else:
            print(f"Using fixed NODE_RESOURCE_CONFIGS configuration for metric normalization")
            print(f"  Total CPU: {format_value_with_unit(node_config_to_use['CPUS'], 'cpu')}")
            print(f"  Total Memory: {format_value_with_unit(node_config_to_use.get('MEMORY_BYTES', node_config_to_use.get('MEMORY_GB', 0) * 2**30), 'memory')}")
            print(f"  Storage: {format_value_with_unit(node_config_to_use.get('DISK_SIZE_BYTES', node_config_to_use.get('DISK_SIZE_GB', 0) * 2**30), 'disk')}")
    else:
        print(f"Quota file not found. Using default node configuration.")
        # Show default configuration information
        print(f"  Total CPU: {format_value_with_unit(node_config_to_use['CPUS'], 'cpu')}")
        print(f"  Total Memory: {format_value_with_unit(node_config_to_use.get('MEMORY_BYTES', node_config_to_use.get('MEMORY_GB', 0) * 2**30), 'memory')}")

    # Add formatted summary of quotas
    if quota_file_path and os.path.exists(quota_file_path):
        print("\nSummary of resource quotas per tenant:")
        quota_summary = get_quota_summary(quota_file_path, include_requests=True, calculate_percentages=True)
        total_entry = quota_summary.get('__total__', {})
        for tenant, quota_info in quota_summary.items():
            if tenant == '__total__':
                continue
            print(f"  {tenant.upper()}:")
            if 'cpu_limit' in quota_info:
                cpu_text = f"    CPU Limit: {quota_info['cpu_limit']}"
                if 'cpu_percent' in quota_info:
                    cpu_text += f" ({quota_info['cpu_percent']} of cluster)"
                print(cpu_text)
            if 'memory_limit' in quota_info:
                mem_text = f"    Memory Limit: {quota_info['memory_limit']}"
                if 'memory_percent' in quota_info:
                    mem_text += f" ({quota_info['memory_percent']} of cluster)"
                print(mem_text)
            if 'cpu_request' in quota_info:
                cpu_req_text = f"    CPU Request: {quota_info['cpu_request']}"
                if 'cpu_req_vs_limit' in quota_info:
                    cpu_req_text += f" ({quota_info['cpu_req_vs_limit']} of limit)"
                print(cpu_req_text)
            if 'memory_request' in quota_info:
                mem_req_text = f"    Memory Request: {quota_info['memory_request']}"
                if 'memory_req_vs_limit' in quota_info:
                    mem_req_text += f" ({quota_info['memory_req_vs_limit']} of limit)"
                print(mem_req_text)
        if total_entry:
            print(f"\n  CLUSTER TOTAL:")
            if 'cpu_limit' in total_entry:
                print(f"    Total CPU: {total_entry['cpu_limit']}")
            if 'memory_limit' in total_entry:
                print(f"    Total Memory: {total_entry['memory_limit']}")

    # UNIFIED DATA LOADING AND INITIAL PROCESSING
    print("\nLoading and Processing Experiment Data...")
    all_metrics_data = load_experiment_data(
        experiment_data_dir,
        metrics=args.metrics if args.metrics else DEFAULT_METRICS,
        rounds=args.rounds
    )

    if not all_metrics_data:
        print("CRITICAL ERROR: No data was loaded. Check the specified data directory, metrics, and rounds.")
        return 

    print("Experiment data loaded successfully.")
            
    metric_type_map = {
        'cpu_usage': 'cpu',
        'memory_usage': 'memory',
        'disk_read_bytes': 'disk',
        'disk_write_bytes': 'disk',
        'network_rx_bytes': 'network',
        'network_tx_bytes': 'network',
        'disk_io_time': 'disk_iops',
        'disk_iops': 'disk_iops'
    }
            
    print("\nFormatting metrics with appropriate units...")
    all_metrics_data = auto_format_metrics(all_metrics_data, metric_type_map)
    print("Metrics formatted with readable units.")
            
    # Print summary of formatted units
    for metric_name, rounds_data in all_metrics_data.items():
        if isinstance(rounds_data, dict):
            for round_name, df in rounds_data.items():
                if isinstance(df, pd.DataFrame) and 'unit' in df.columns and not df['unit'].isna().all():
                    unit = df['unit'].iloc[0]
                    print(f"  {metric_name} (Round: {round_name}): {unit}")
                    break 
                elif isinstance(df, pd.DataFrame) and 'value_formatted' in df.columns:
                    print(f"  {metric_name} (Round: {round_name}): custom formatting applied")
                    break
            else: 
                if not rounds_data:
                     print(f"  {metric_name}: No rounds data found to determine unit.")
        elif isinstance(rounds_data, pd.DataFrame): 
            df = rounds_data 
            if 'unit' in df.columns and not df['unit'].isna().all():
                unit = df['unit'].iloc[0]
                print(f"  {metric_name}: {unit}")
            elif 'value_formatted' in df.columns:
                print(f"  {metric_name}: custom formatting applied")
        else:
            print(f"  {metric_name}: Data is not in expected format (DataFrame or Dict[str, DataFrame]).")

    # Normalize metrics as percentages if requested
    if args.show_as_percentage:
        print("\nNormalizing metrics as percentages of resource capacity...")
        
        # Path to the quota file to use for normalization
        quota_file_for_norm = quota_file_path if os.path.exists(quota_file_path) else None
        
        # Apply normalization using our improved configuration
        print("Applying normalization...")
        normalized_metrics = apply_normalization_to_all_metrics(
            all_metrics_data, 
            node_config_to_use, 
            replace_original=False,  # Keep original values
            use_tenant_quotas=(quota_file_for_norm is not None),
            add_relative_values=True,
            show_as_percentage=args.show_as_percentage,
            use_formatted_values=True,
            quota_file=quota_file_for_norm
        )
        
        # Merge with existing data or replace
        all_metrics_data = normalized_metrics
        
        print("Normalization completed successfully.")
        # Show example of formatting for the first metric as a reference
        for metric_name, df in all_metrics_data.items():
            if isinstance(df, dict):
                for round_name, round_df in df.items():
                    if isinstance(round_df, pd.DataFrame) and 'normalized_value' in round_df.columns and not round_df.empty:
                        print(f"  Example for {metric_name} (Round: {round_name}):")
                        sample_row = round_df.iloc[0]
                        print(f"    Original value: {sample_row.get('original_value', sample_row.get('value', 'N/A'))}")
                        print(f"    Normalized value: {sample_row.get('normalized_value', 'N/A')}%")
                        print(f"    Description: {sample_row.get('normalized_description', 'N/A')}")
                        break 
            elif isinstance(df, pd.DataFrame): 
                if 'normalized_value' in df.columns and not df.empty:
                    print(f"  Example for {metric_name}:")
                    sample_row = df.iloc[0]
                    print(f"    Original value: {sample_row.get('original_value', sample_row.get('value', 'N/A'))}")
                    print(f"    Normalized value: {sample_row.get('normalized_value', 'N/A')}%")
                    print(f"    Description: {sample_row.get('normalized_description', 'N/A')}")
            else:
                print(f"  {metric_name}: Data is not in expected DataFrame or Dict[str, DataFrame] format for normalization example.")

    # Continue with data processing

    experiment_results = {
        'processed_data': all_metrics_data
    }

    metrics_data = experiment_results.get('processed_data', {})
    aggregated_data = experiment_results.get("aggregated_data", {})
    impact_score_results = experiment_results.get("impact_score_results", {})
    total_experiment_duration_seconds = experiment_results.get("total_experiment_duration_seconds", None)

    if not metrics_data:
        print("No processed data (metrics_data) available. Many analyses and plots will be skipped.")
        # Not returning, as some parts of the script might still be useful (e.g. empty report generation)

    print("\nNormalizing global experiment time for all metric DataFrames...")
    # Iterate over each metric
    all_phase_markers = {} # Initialize all_phase_markers here
    for metric_name, rounds_data in all_metrics_data.items():
        all_phase_markers[metric_name] = {} # Initialize for each metric
        # Iterate over each round DataFrame within the metric
        for round_name, df_round in rounds_data.items():
            if df_round is not None and not df_round.empty:
                # Apply time normalization
                df_round = add_experiment_elapsed_time(df_round) # Reassign the result
                # Add phase markers
                if not df_round.empty:
                    # Pass the phase column name explicitly and the display names dictionary
                    df_round, phase_markers_round = add_phase_markers(df_round, phase_column='phase', phase_display_names=PHASE_DISPLAY_NAMES)
                    all_phase_markers[metric_name][round_name] = phase_markers_round
                    all_metrics_data[metric_name][round_name] = df_round # Update the DataFrame with the new 'phase_name' column
            else:
                print(f"DataFrame for metric '{metric_name}', round '{round_name}' is empty or None. Skipping time normalization.")
    print("Global time normalization completed.")

    # >>> BEGIN MODIFICATION TO USE AGGREGATED DATA FOR ADVANCED ANALYSIS <<<
    # Option to use aggregated data if multiple rounds exist and flag is set
    multiple_rounds_available = False
    if metrics_data:
        # Check if any metric has data from more than one round
        for metric_name_check, rounds_data_check in metrics_data.items():
            if isinstance(rounds_data_check, dict) and len(rounds_data_check) > 1:
                multiple_rounds_available = True
                break
    
    if args.use_aggregated_rounds_for_advanced and multiple_rounds_available:
        print("\nPreparing data for aggregation across multiple rounds...")
        metrics_for_aggregation = {}
        for metric_name, rounds_data_map in all_metrics_data.items():
            if isinstance(rounds_data_map, dict):
                metric_dfs = []
                for round_name, df_round in rounds_data_map.items():
                    df_copy = df_round.copy()
                    df_copy['round'] = round_name # Ensure 'round' column is present
                    metric_dfs.append(df_copy)
                if metric_dfs:
                    metrics_for_aggregation[metric_name] = pd.concat(metric_dfs, ignore_index=True)
            else:
                # If it's already a DataFrame (e.g. single round loaded, or already concatenated)
                # Ensure 'round' column exists or add a default one
                if 'round' not in rounds_data_map.columns:
                    df_copy = rounds_data_map.copy()
                    df_copy['round'] = 'default_round' # Placeholder
                    metrics_for_aggregation[metric_name] = df_copy
                else:
                    metrics_for_aggregation[metric_name] = rounds_data_map

        if metrics_for_aggregation:
            aggregated_data_for_advanced = aggregate_metrics_across_rounds(metrics_for_aggregation, value_column='value')
            print("Aggregation complete. Using aggregated data for subsequent advanced/causality analyses.")
            
            # Generate aggregated plots and tables
            for metric_name_agg, agg_df in aggregated_data_for_advanced.items():
                if not agg_df.empty:
                    # Pass all_metrics_data to plot_aggregated_metrics
                    fig_agg = plot_aggregated_metrics(aggregated_data_for_advanced, metric_name_agg, all_metrics_data_for_phases=all_metrics_data)
                    if fig_agg:
                        agg_plot_path = os.path.join(advanced_dir, 'plots', f'aggregated_{metric_name_agg}_across_rounds.png')
                        fig_agg.savefig(agg_plot_path)
                        plt.close(fig_agg)
                        print(f"    Aggregated plot for {metric_name_agg} saved to {agg_plot_path}")
                    
                    # Generate boxplots for aggregated data
                    plot_aggregated_metrics_boxplot(aggregated_data_for_advanced, metric_name_agg, os.path.join(advanced_dir, 'plots'))

                    agg_table_path = os.path.join(advanced_dir, 'tables', f'aggregated_{metric_name_agg}_across_rounds.csv')
                    export_to_csv(agg_df, agg_table_path)
                    print(f"    Aggregated table for {metric_name_agg} saved to {agg_table_path}")
        else:
            print("No metrics suitable for aggregation were found.")
    # >>> END MODIFICATION <<<

    # Main Processing and Analysis (if data and active flags exist)
    if args.advanced and metrics_data:
        print("\nRunning Advanced Analyses and Generating Plots...")
        advanced_plots_dir = os.path.join(advanced_dir, 'plots')
        os.makedirs(advanced_plots_dir, exist_ok=True)
        advanced_tables_dir = os.path.join(advanced_dir, 'tables')
        os.makedirs(advanced_tables_dir, exist_ok=True)
        
        # Ensure advanced_analysis_results is initialized correctly
        # Results will be stored in experiment_results['advanced_analysis']
        # If using aggregated, new sub-keys like 'correlation_matrices_aggregated' might be used,
        # or we can overwrite existing keys if the structure is compatible.
        # For clarity, let's use distinct keys for aggregated results within advanced_analysis.
        if 'advanced_analysis' not in experiment_results:
            experiment_results['advanced_analysis'] = {}
        # This variable will hold the results for the current run (either aggregated or not)
        current_advanced_results_dict = experiment_results['advanced_analysis']


        if args.use_aggregated_rounds_for_advanced and 'aggregated_data_for_advanced' in locals() and aggregated_data_for_advanced:
            print("\nUsing AGGREGATED data for Advanced Analyses (correlation, covariance, entropy)...")
            
            available_phases_in_agg = set()
            if aggregated_data_for_advanced: # Check if dict is not empty
                for metric_df_agg_check in aggregated_data_for_advanced.values():
                    if 'phase' in metric_df_agg_check.columns and not metric_df_agg_check.empty:
                        available_phases_in_agg.update(metric_df_agg_check['phase'].unique())
            
            if not available_phases_in_agg and aggregated_data_for_advanced: # If no phases, but data exists (e.g. not phase-specific aggregation)
                available_phases_in_agg.add("all_aggregated") # Use a placeholder for non-phased aggregated data
            elif not aggregated_data_for_advanced:
                 print("    No aggregated data available to process for advanced analysis.")

            for phase_to_analyze_agg in sorted(list(p for p in available_phases_in_agg if p is not None)):
                phase_str = f"phase_{phase_to_analyze_agg}" if phase_to_analyze_agg != "all_aggregated" else "all_aggregated"
                print(f"  Analyzing AGGREGATED data for: {phase_str}")

                metrics_dict_for_agg_phase = {}
                for metric_name, agg_df in aggregated_data_for_advanced.items():
                    df_for_corr_cov = agg_df.copy()
                    if phase_to_analyze_agg != "all_aggregated" and 'phase' in df_for_corr_cov.columns:
                        df_for_corr_cov = df_for_corr_cov[df_for_corr_cov['phase'] == phase_to_analyze_agg]
                    
                    if not df_for_corr_cov.empty:
                        # Corrected renaming: 'experiment_elapsed_seconds' is the time column from aggregation
                        df_for_corr_cov = df_for_corr_cov.rename(columns={'experiment_elapsed_seconds': 'datetime', 'mean': 'value'})
                        df_for_corr_cov['round'] = 'aggregated_data' # Dummy round for compatibility
                        # Ensure 'phase' column exists if phase_to_analyze_agg is not "all_aggregated"
                        if phase_to_analyze_agg != "all_aggregated" and 'phase' not in df_for_corr_cov.columns:
                             df_for_corr_cov['phase'] = phase_to_analyze_agg
                        elif phase_to_analyze_agg == "all_aggregated" and 'phase' not in df_for_corr_cov.columns:
                             df_for_corr_cov['phase'] = "all_aggregated" # ensure phase column

                        # Ensure 'datetime' column exists after renaming
                        if 'datetime' not in df_for_corr_cov.columns:
                            print(f"    WARNING: 'datetime' column missing for {metric_name}, {phase_str} after rename. Columns: {df_for_corr_cov.columns.tolist()}")
                            # Attempt to use 'time_bin_global' if 'experiment_elapsed_seconds' was not found and it exists
                            if 'time_bin_global' in agg_df.columns:
                                print(f"    Attempting rename from 'time_bin_global' instead for {metric_name}")
                                df_for_corr_cov = agg_df.copy().rename(columns={'time_bin_global': 'datetime', 'mean': 'value'})
                                df_for_corr_cov['round'] = 'aggregated_data' 
                                if phase_to_analyze_agg != "all_aggregated" and 'phase' not in df_for_corr_cov.columns:
                                    df_for_corr_cov['phase'] = phase_to_analyze_agg
                                elif phase_to_analyze_agg == "all_aggregated" and 'phase' not in df_for_corr_cov.columns:
                                    df_for_corr_cov['phase'] = "all_aggregated"
                            else:
                                print(f"    ERROR: Neither 'experiment_elapsed_seconds' nor 'time_bin_global' found in columns for {metric_name}. Skipping.")
                                continue # Skip this metric for this phase if time column is missing

                        metrics_dict_for_agg_phase[metric_name] = df_for_corr_cov[['datetime', 'tenant', 'value', 'round', 'phase']]

                if not metrics_dict_for_agg_phase:
                    print(f"    No aggregated data for {phase_str} to analyze for inter-metric correlation/covariance.")
                    continue

                # 1. AGGREGATED Inter-Metric Correlation Analysis
                try:
                    print(f"    Calculating AGGREGATED inter-metric correlation matrix for {phase_str}")
                    correlation_matrix_agg = calculate_correlation_matrix(
                        metrics_dict=metrics_dict_for_agg_phase,
                        round_name='aggregated_data',
                        tenants=args.tenants,
                        noisy_tenant=args.noisy_tenant
                    )
                    if correlation_matrix_agg is not None and not correlation_matrix_agg.empty:
                        plot_path_corr_agg = os.path.join(advanced_plots_dir, f'correlation_heatmap_aggregated_{phase_str}.png')
                        fig_corr_agg = plot_correlation_heatmap(
                            correlation_matrix_agg,
                            title=f'Aggregated Inter-Metric Correlation ({PHASE_DISPLAY_NAMES.get(phase_to_analyze_agg, phase_str)})'
                        )
                        fig_corr_agg.savefig(plot_path_corr_agg)
                        plt.close(fig_corr_agg)
                        print(f"      Aggregated correlation heatmap saved to: {plot_path_corr_agg}")
                        current_advanced_results_dict.setdefault('correlation_matrices_aggregated', {})[phase_str] = correlation_matrix_agg.to_dict()
                    else:
                        print(f"      Aggregated correlation matrix is empty for {phase_str}.")
                except Exception as e:
                    print(f"    Error in AGGREGATED inter-metric correlation analysis for {phase_str}: {e}")

                # 2. AGGREGATED Inter-Metric Covariance Analysis
                try:
                    print(f"    Calculating AGGREGATED inter-metric covariance matrix for {phase_str}")
                    covariance_matrix_agg, _ = calculate_covariance_matrix(
                        metrics_dict=metrics_dict_for_agg_phase,
                        round_name='aggregated_data',
                        tenants=args.tenants
                    )
                    if covariance_matrix_agg is not None and not covariance_matrix_agg.empty:
                        plot_path_cov_agg = os.path.join(advanced_plots_dir, f'covariance_heatmap_aggregated_{phase_str}.png')
                        fig_cov_agg = plot_correlation_heatmap(
                            covariance_matrix_agg,
                            title=f'Aggregated Inter-Metric Covariance ({PHASE_DISPLAY_NAMES.get(phase_to_analyze_agg, phase_str)})',
                            cmap='coolwarm'
                        )
                        fig_cov_agg.savefig(plot_path_cov_agg)
                        plt.close(fig_cov_agg)
                        print(f"      Aggregated covariance heatmap saved to: {plot_path_cov_agg}")
                        current_advanced_results_dict.setdefault('covariance_matrices_aggregated', {})[phase_str] = covariance_matrix_agg.to_dict()
                    else:
                        print(f"      Aggregated covariance matrix is empty for {phase_str}.")
                except Exception as e:
                    print(f"    Error in AGGREGATED inter-metric covariance analysis for {phase_str}: {e}")

            # AGGREGATED Inter-Tenant Correlation and Covariance per Metric
            print("\n  Calculating AGGREGATED Inter-Tenant Correlation and Covariance per Metric...")
            for metric_name_agg_inter, agg_df_inter_original in aggregated_data_for_advanced.items():
                agg_df_inter = agg_df_inter_original.copy() # Work on a copy to avoid modifying the original dict entry
                if not agg_df_inter.empty and 'tenant' in agg_df_inter.columns and agg_df_inter['tenant'].nunique() > 1:
                    # Corrected renaming for inter-tenant analysis from the correct source DataFrame
                    if 'experiment_elapsed_seconds' in agg_df_inter.columns:
                        df_for_inter_tenant_analysis = agg_df_inter.rename(columns={'experiment_elapsed_seconds': 'datetime', 'mean': 'value'})
                    elif 'time_bin_global' in agg_df_inter.columns: # Fallback if 'experiment_elapsed_seconds' was somehow not the primary name
                        df_for_inter_tenant_analysis = agg_df_inter.rename(columns={'time_bin_global': 'datetime', 'mean': 'value'})
                    else:
                        print(f"    ERROR: Time column ('experiment_elapsed_seconds' or 'time_bin_global') not found for {metric_name_agg_inter} in inter-tenant analysis. Columns: {agg_df_inter.columns.tolist()}. Skipping.")
                        continue
                    
                    # Verify 'datetime' column exists after renaming before proceeding
                    if 'datetime' not in df_for_inter_tenant_analysis.columns:
                        print(f"    ERROR: 'datetime' column still missing after rename for {metric_name_agg_inter} in inter-tenant. Cols: {df_for_inter_tenant_analysis.columns.tolist()}. Skipping.")
                        continue

                    if args.tenants:
                        df_for_inter_tenant_analysis = df_for_inter_tenant_analysis[df_for_inter_tenant_analysis['tenant'].isin(args.tenants)]
                    
                    if df_for_inter_tenant_analysis['tenant'].nunique() < 2:
                        print(f"    Skipping AGGREGATED inter-tenant analysis for {metric_name_agg_inter}: < 2 tenants after filtering.")
                        continue

                    try: # AGGREGATED Inter-Tenant Correlation
                        print(f"    Calculating AGGREGATED inter-tenant correlation for metric: {metric_name_agg_inter}")
                        inter_tenant_corr_matrix_agg = calculate_inter_tenant_correlation_per_metric(df_for_inter_tenant_analysis, value_col='value', time_col='datetime', tenant_col='tenant')
                        if inter_tenant_corr_matrix_agg is not None and not inter_tenant_corr_matrix_agg.empty:
                            csv_path = os.path.join(advanced_tables_dir, f'inter_tenant_correlation_aggregated_{metric_name_agg_inter}.csv')
                            export_to_csv(inter_tenant_corr_matrix_agg, csv_path)
                            plot_path = os.path.join(advanced_plots_dir, f'inter_tenant_correlation_heatmap_aggregated_{metric_name_agg_inter}.png')
                            fig = plot_correlation_heatmap(
                                inter_tenant_corr_matrix_agg,
                                title=f'Aggregated Inter-Tenant Correlation - {METRIC_DISPLAY_NAMES.get(metric_name_agg_inter, metric_name_agg_inter)}',
                                cbar_label='Correlation Coefficient'
                            )
                            fig.savefig(plot_path)
                            plt.close(fig)
                            print(f"      Aggregated inter-tenant correlation for {metric_name_agg_inter} saved (CSV/Plot).")
                            current_advanced_results_dict.setdefault('inter_tenant_correlation_matrices_aggregated', {}).setdefault(metric_name_agg_inter, {})['aggregated'] = inter_tenant_corr_matrix_agg.to_dict()
                        else:
                            print(f"      Aggregated inter-tenant correlation matrix empty for {metric_name_agg_inter}.")
                    except Exception as e:
                        print(f"      Error in AGGREGATED inter-tenant correlation for {metric_name_agg_inter}: {e}")

                    try: # AGGREGATED Inter-Tenant Covariance
                        print(f"    Calculating AGGREGATED inter-tenant covariance for metric: {metric_name_agg_inter}")
                        inter_tenant_cov_matrix_agg = calculate_inter_tenant_covariance_per_metric(df_for_inter_tenant_analysis, value_col='value', time_col='datetime', tenant_col='tenant')
                        if inter_tenant_cov_matrix_agg is not None and not inter_tenant_cov_matrix_agg.empty:
                            csv_path = os.path.join(advanced_tables_dir, f'inter_tenant_covariance_aggregated_{metric_name_agg_inter}.csv')
                            export_to_csv(inter_tenant_cov_matrix_agg, csv_path)
                            plot_path = os.path.join(advanced_plots_dir, f'inter_tenant_covariance_heatmap_aggregated_{metric_name_agg_inter}.png')
                            fig = plot_correlation_heatmap(
                                inter_tenant_cov_matrix_agg,
                                title=f'Aggregated Inter-Tenant Covariance - {METRIC_DISPLAY_NAMES.get(metric_name_agg_inter, metric_name_agg_inter)}',
                                cbar_label='Covariance'
                            )
                            fig.savefig(plot_path)
                            plt.close(fig)
                            print(f"      Aggregated inter-tenant covariance for {metric_name_agg_inter} saved (CSV/Plot).")
                            current_advanced_results_dict.setdefault('inter_tenant_covariance_matrices_aggregated', {}).setdefault(metric_name_agg_inter, {})['aggregated'] = inter_tenant_cov_matrix_agg.to_dict()
                        else:
                            print(f"      Aggregated inter-tenant covariance matrix empty for {metric_name_agg_inter}.")
                    except Exception as e:
                        print(f"      Error in AGGREGATED inter-tenant covariance for {metric_name_agg_inter}: {e}")
                else:
                    print(f"    Skipping AGGREGATED inter-tenant analysis for {metric_name_agg_inter}: Not enough unique tenants or data.")
            
            # 3. AGGREGATED Entropy Analysis
            print(f"\n  Calculating AGGREGATED entropy metrics...")
            all_entropy_results_agg_list = []
            for metric_name_entropy_agg, agg_df_entropy_original in aggregated_data_for_advanced.items():
                agg_df_entropy = agg_df_entropy_original.copy() # Work on a copy
                if not agg_df_entropy.empty and 'tenant' in agg_df_entropy.columns and agg_df_entropy['tenant'].nunique() >=2 :
                    # Corrected renaming for entropy analysis from the correct source DataFrame
                    if 'experiment_elapsed_seconds' in agg_df_entropy.columns:
                        df_for_entropy = agg_df_entropy.rename(columns={'experiment_elapsed_seconds': 'datetime', 'mean': 'value'})
                    elif 'time_bin_global' in agg_df_entropy.columns: # Fallback
                        df_for_entropy = agg_df_entropy.rename(columns={'time_bin_global': 'datetime', 'mean': 'value'})
                    else:
                        print(f"    ERROR: Time column ('experiment_elapsed_seconds' or 'time_bin_global') not found for {metric_name_entropy_agg} in entropy analysis. Columns: {agg_df_entropy.columns.tolist()}. Skipping.")
                        continue

                    # Verify 'datetime' column exists after renaming before proceeding
                    if 'datetime' not in df_for_entropy.columns:
                        print(f"    ERROR: 'datetime' column still missing after rename for {metric_name_entropy_agg} in entropy. Cols: {df_for_entropy.columns.tolist()}. Skipping.")
                        continue

                    if args.tenants:
                        df_for_entropy = df_for_entropy[df_for_entropy['tenant'].isin(args.tenants)]
                    
                    if df_for_entropy['tenant'].nunique() < 2:
                        print(f"    Skipping AGGREGATED entropy for {metric_name_entropy_agg}: < 2 tenants after filtering.")
                        continue

                    current_metric_phases_entropy = df_for_entropy['phase'].unique() if 'phase' in df_for_entropy.columns else ["all_aggregated"]
                    for phase_entropy in current_metric_phases_entropy:
                        phase_entropy_str = f"phase_{phase_entropy}" if phase_entropy != "all_aggregated" else "all_aggregated"
                        df_phase_entropy = df_for_entropy.copy()
                        if phase_entropy != "all_aggregated" and 'phase' in df_phase_entropy.columns:
                            df_phase_entropy = df_phase_entropy[df_phase_entropy['phase'] == phase_entropy]
                        elif 'phase' not in df_phase_entropy.columns : # If no phase column, use all data for "all_aggregated"
                            pass


                        if df_phase_entropy.empty or df_phase_entropy['tenant'].nunique() < 2:
                            print(f"      Skipping AGGREGATED entropy for {metric_name_entropy_agg}, {phase_entropy_str}: not enough data or tenants.")
                            continue
                        
                        print(f"    Calculating AGGREGATED entropy for metric: {metric_name_entropy_agg}, {phase_entropy_str}")
                        try:
                            entropy_results_df_agg = calculate_entropy_metrics(
                                df_phase_entropy,
                                tenants=None, # Uses all tenants in the provided df_phase_entropy
                                phase=phase_entropy if phase_entropy != "all_aggregated" else None, # Pass phase if specific, else None
                                value_col='value', # CORRECTED from metric_column and made explicit
                                time_col='datetime', # Explicitly pass
                                tenant_col='tenant',   # Explicitly pass
                                phase_col='phase'     # Explicitly pass
                            )
                            if entropy_results_df_agg is not None and not entropy_results_df_agg.empty:
                                entropy_results_df_agg['metric'] = metric_name_entropy_agg
                                entropy_results_df_agg['aggregation_phase_analyzed'] = phase_entropy_str
                                all_entropy_results_agg_list.append(entropy_results_df_agg)
                                table_path_entropy_agg = os.path.join(advanced_tables_dir, f'entropy_analysis_aggregated_{metric_name_entropy_agg}_{phase_entropy_str}.csv')
                                export_to_csv(entropy_results_df_agg, table_path_entropy_agg)
                                print(f"      Aggregated entropy results for {metric_name_entropy_agg}, {phase_entropy_str} saved to {table_path_entropy_agg}")
                            else:
                                print(f"      No AGGREGATED entropy results for {metric_name_entropy_agg}, {phase_entropy_str}.")
                        except Exception as e_entropy_agg:
                            print(f"    Error in AGGREGATED entropy analysis for {metric_name_entropy_agg}, {phase_entropy_str}: {e_entropy_agg}")
                else:
                    print(f"    Skipping AGGREGATED entropy for {metric_name_entropy_agg}: Not enough unique tenants or data.")

            if all_entropy_results_agg_list:
                final_entropy_df_agg = pd.concat(all_entropy_results_agg_list, ignore_index=True)
                current_advanced_results_dict['entropy_analysis_aggregated'] = final_entropy_df_agg.to_dict(orient='records')
                export_to_csv(final_entropy_df_agg, os.path.join(advanced_tables_dir, 'entropy_analysis_aggregated_combined.csv'))
                print("  Combined aggregated entropy analysis saved.")

                # Generate plots for aggregated entropy results
                if not final_entropy_df_agg.empty:
                    print("  Generating plots for aggregated entropy analysis...")
                    # Ensure 'metric' and 'aggregation_phase_analyzed' columns exist
                    if 'metric' in final_entropy_df_agg.columns and 'aggregation_phase_analyzed' in final_entropy_df_agg.columns:
                        for (metric_name_plot, phase_analyzed_plot), group_df in final_entropy_df_agg.groupby(['metric', 'aggregation_phase_analyzed']):
                            # Sanitize filenames by replacing potentially problematic characters
                            safe_metric_name = str(metric_name_plot).replace('/', '_').replace('\\', '_') # Corrected backslash replacement
                            safe_phase_analyzed = str(phase_analyzed_plot).replace('/', '_').replace('\\', '_') # Corrected backslash replacement
                            
                            plot_title_suffix = f"Aggregated - {METRIC_DISPLAY_NAMES.get(metric_name_plot, metric_name_plot)} - {PHASE_DISPLAY_NAMES.get(phase_analyzed_plot.replace('phase_', ''), phase_analyzed_plot)}"
                            
                            if args.entropy_plot_type in ['heatmap', 'all']:
                                try:
                                    # Construct output_path for plot_entropy_heatmap
                                    heatmap_filename = f'entropy_heatmap_aggregated_{safe_metric_name}_{safe_phase_analyzed}.png'
                                    heatmap_output_path = os.path.join(advanced_plots_dir, heatmap_filename)
                                    
                                    plot_entropy_heatmap(
                                        group_df, # This is the entropy_results_df for this group
                                        metric_name=metric_name_plot, # Pass the metric name
                                        round_name=phase_analyzed_plot, # Pass the phase/aggregation identifier
                                        output_path=heatmap_output_path # Pass the full output path
                                    )
                                    print(f"    Entropy heatmap for {metric_name_plot}, {phase_analyzed_plot} generated and saved to {heatmap_output_path}.")
                                except Exception as e_plot_heatmap:
                                    print(f"    Error generating entropy heatmap for {metric_name_plot}, {phase_analyzed_plot}: {e_plot_heatmap}")

                            if args.entropy_plot_type in ['barplot', 'all']:
                                try:
                                    # Construct output_path for plot_entropy_top_pairs_barplot
                                    barplot_filename = f'entropy_top_pairs_aggregated_{safe_metric_name}_{safe_phase_analyzed}.png'
                                    barplot_output_path = os.path.join(advanced_plots_dir, barplot_filename)
                                    
                                    plot_entropy_top_pairs_barplot(
                                        group_df, # This is the entropy_results_df for this group
                                        metric_name=metric_name_plot, # Pass the metric name
                                        round_name=phase_analyzed_plot, # Pass the phase/aggregation identifier
                                        output_path=barplot_output_path, # Pass the full output path
                                        top_n=10
                                    )
                                    print(f"    Entropy barplot for {metric_name_plot}, {phase_analyzed_plot} generated and saved to {barplot_output_path}.")
                                except Exception as e_plot_barplot:
                                    print(f"    Error generating entropy barplot for {metric_name_plot}, {phase_analyzed_plot}: {e_plot_barplot}")
                    else:
                        print("    Skipping aggregated entropy plot generation: 'metric' or 'aggregation_phase_analyzed' column missing in final_entropy_df_agg.")

            experiment_results['advanced_analysis'] = current_advanced_results_dict

        elif not args.use_aggregated_rounds_for_advanced and metrics_data:
            print("\nUsing NON-AGGREGATED (per-round or concatenated) data for Advanced Analyses (correlation, covariance, entropy)...")
            # Prepare data for correlation/covariance: concatenate rounds for each metric
            all_metrics_data_concatenated = {}
            if metrics_data: # Ensure metrics_data is not None
                for metric_name_loop, rounds_data_for_metric_loop in metrics_data.items():
                    if isinstance(rounds_data_for_metric_loop, dict) and rounds_data_for_metric_loop:
                        all_rounds_dfs_loop = []
                        for round_df_loop in rounds_data_for_metric_loop.values():
                            if isinstance(round_df_loop, pd.DataFrame) and not round_df_loop.empty:
                                # Ensure 'round' column exists
                                if 'round' not in round_df_loop.columns:
                                    # This case should ideally not happen if data loader is consistent
                                    # For safety, skip if round info is missing for concatenated analysis
                                    print(f"Warning: 'round' column missing in data for {metric_name_loop}. Skipping for concatenation.")
                                    continue
                                all_rounds_dfs_loop.append(round_df_loop)
                        if all_rounds_dfs_loop:
                            concatenated_df_loop = pd.concat(all_rounds_dfs_loop, ignore_index=True)
                            all_metrics_data_concatenated[metric_name_loop] = concatenated_df_loop
            
            all_round_names = set()
            if isinstance(metrics_data, dict):
                for metric_name_iter, rounds_data_iter in metrics_data.items():
                    if isinstance(rounds_data_iter, dict):
                        all_round_names.update(rounds_data_iter.keys())
            
            if not all_metrics_data_concatenated and not (args.round and len(args.round) == 1 and metrics_data):
                 print("No concatenated data available and not single round analysis. Skipping advanced inter-metric correlation/covariance analyses.")
            else:
                # Determine rounds to analyze: either specified single round or all found rounds
                rounds_to_iterate = []
                if args.round and len(args.round) == 1:
                    single_round_name = args.round[0]
                    # Prepare metrics_dict for the single round
                    metrics_dict_single_round = {}
                    for metric_name, round_data_dict in metrics_data.items():
                        if single_round_name in round_data_dict:
                             metrics_dict_single_round[metric_name] = round_data_dict[single_round_name]
                    
                    if metrics_dict_single_round:
                         # For single round analysis, calculate_correlation_matrix expects dict of DFs for that round
                         # The function itself will handle the filtering by round_name if it's still in the DFs.
                         # Or, ensure DFs passed only contain that round's data.
                         # The current calculate_correlation_matrix filters by round_name from the concatenated DFs.
                         # So, we still need all_metrics_data_concatenated for it to work as designed.
                         # If only one round is loaded, all_metrics_data_concatenated will contain only that round.
                        rounds_to_iterate.append(single_round_name)
                    else:
                        print(f"Data for the specified round '{single_round_name}' not found. Skipping inter-metric correlation/covariance.")
                
                elif all_metrics_data_concatenated : # Multiple rounds, use concatenated data
                    rounds_to_iterate = sorted(list(all_round_names))

                if not rounds_to_iterate and all_metrics_data_concatenated: # Fallback if specific round logic failed but concatenated exists
                    rounds_to_iterate = sorted(list(all_round_names))


                for round_name_to_analyze in rounds_to_iterate:
                    print(f"  Analyzing Round: {round_name_to_analyze} for correlation and covariance inter-metric")
                    
                    # Data for this specific round for calculate_correlation_matrix
                    # It expects a dict of DFs, where each DF might contain multiple rounds,
                    # and it filters by 'round_name_to_analyze'. So all_metrics_data_concatenated is correct.
                    current_metrics_for_round_analysis = all_metrics_data_concatenated
                    if not current_metrics_for_round_analysis:
                        print(f"    Concatenated data not available for round {round_name_to_analyze}. Skipping.")
                        continue

                    # 1. Correlation Analysis (Non-Aggregated)
                    try:
                        print(f"    Calculating correlation matrix for round: {round_name_to_analyze}")
                        correlation_matrix = calculate_correlation_matrix(
                            metrics_dict=current_metrics_for_round_analysis, # This contains all rounds, func filters
                            round_name=round_name_to_analyze,
                            tenants=args.tenants,
                            noisy_tenant=args.noisy_tenant
                        )

                        if correlation_matrix is not None and not correlation_matrix.empty:
                            plot_path_corr = os.path.join(advanced_plots_dir, f'correlation_heatmap_round_{round_name_to_analyze}.png')
                            fig_corr = plot_correlation_heatmap(
                                correlation_matrix,
                                title=f'Inter-Metric Correlation Heatmap (Round: {round_name_to_analyze})'
                            )
                            fig_corr.savefig(plot_path_corr)
                            plt.close(fig_corr)
                            print(f"      Correlation heatmap saved to: {plot_path_corr}")
                            current_advanced_results_dict.setdefault('correlation_matrices', {}).setdefault(round_name_to_analyze, {}).update(correlation_matrix.to_dict())
                        else:
                            print(f"      Correlation matrix empty or None for round: {round_name_to_analyze}. Skipping plot.")
                    except Exception as e:
                        print(f"    Error calculating or plotting correlation matrix for round {round_name_to_analyze}: {e}")

                    # 2. Covariance Analysis (Non-Aggregated)
                    try:
                        print(f"    Calculating covariance matrix for round: {round_name_to_analyze}")
                        covariance_matrix, _ = calculate_covariance_matrix(
                            metrics_dict=current_metrics_for_round_analysis, # This contains all rounds, func filters
                            round_name=round_name_to_analyze,
                            tenants=args.tenants
                        )

                        if covariance_matrix is not None and not covariance_matrix.empty:
                            plot_path_cov = os.path.join(advanced_plots_dir, f'covariance_heatmap_round_{round_name_to_analyze}.png')
                            fig_cov = plot_correlation_heatmap(
                                covariance_matrix,
                                title=f'Inter-Metric Covariance Heatmap (Round: {round_name_to_analyze})',
                                cmap='coolwarm'
                            )
                            fig_cov.savefig(plot_path_cov)
                            plt.close(fig_cov)
                            print(f"      Covariance heatmap saved to: {plot_path_cov}")
                            current_advanced_results_dict.setdefault('covariance_matrices', {}).setdefault(round_name_to_analyze, {}).update(covariance_matrix.to_dict())
                        else:
                            print(f"      Covariance matrix empty or None for round: {round_name_to_analyze}. Skipping plot.")
                    except Exception as e:
                        print(f"    Error calculating or plotting covariance matrix for round {round_name_to_analyze}: {e}")

            # Inter-Tenant Correlation and Covariance per Metric (Non-Aggregated)
            print("\n  Calculating Inter-Tenant Correlation and Covariance per Metric (NON-AGGREGATED)...")
            if metrics_data: # Check if metrics_data is not None
                for metric_name_inter_tenant, rounds_data_inter_tenant in metrics_data.items():
                    if isinstance(rounds_data_inter_tenant, dict):
                        # Determine rounds to iterate: specified single round or all available for this metric
                        rounds_to_analyze_inter_tenant = []
                        if args.round and len(args.round) == 1:
                            if args.round[0] in rounds_data_inter_tenant:
                                rounds_to_analyze_inter_tenant.append(args.round[0])
                            else:
                                print(f"    Data for the specified round '{args.round[0]}' not found for metric {metric_name_inter_tenant}. Skipping inter-tenant analysis for this round.")
                        else:
                            rounds_to_analyze_inter_tenant = rounds_data_inter_tenant.keys()

                        for round_name_inter_tenant in rounds_to_analyze_inter_tenant:
                            df_round_inter_tenant = rounds_data_inter_tenant.get(round_name_inter_tenant)
                            if isinstance(df_round_inter_tenant, pd.DataFrame) and not df_round_inter_tenant.empty and 'tenant' in df_round_inter_tenant.columns and df_round_inter_tenant['tenant'].nunique() > 1:
                                df_to_analyze = df_round_inter_tenant
                                if args.tenants:
                                    df_to_analyze = df_round_inter_tenant[df_round_inter_tenant['tenant'].isin(args.tenants)]
                                
                                if df_to_analyze['tenant'].nunique() < 2:
                                    print(f"    Skipping inter-tenant correlation/covariance for {metric_name_inter_tenant}, round {round_name_inter_tenant}: Less than 2 tenants after filtering.")
                                    continue

                                # Inter-Tenant Correlation (Non-Aggregated)
                                try:
                                    print(f"    Calculating inter-tenant correlation for metric: {metric_name_inter_tenant}, round: {round_name_inter_tenant}")
                                    inter_tenant_corr_matrix = calculate_inter_tenant_correlation_per_metric(df_to_analyze) # Uses default 'value', 'datetime', 'tenant'
                                    if inter_tenant_corr_matrix is not None and not inter_tenant_corr_matrix.empty:
                                        csv_path_inter_tenant_corr = os.path.join(advanced_tables_dir, f'inter_tenant_correlation_{metric_name_inter_tenant}_round_{round_name_inter_tenant}.csv')
                                        export_to_csv(inter_tenant_corr_matrix, csv_path_inter_tenant_corr)
                                        plot_path_inter_tenant_corr = os.path.join(advanced_plots_dir, f'inter_tenant_correlation_heatmap_{metric_name_inter_tenant}_round_{round_name_inter_tenant}.png')
                                        fig_inter_tenant_corr = plot_correlation_heatmap(
                                            inter_tenant_corr_matrix,
                                            title=f'Inter-Tenant Correlation - {METRIC_DISPLAY_NAMES.get(metric_name_inter_tenant, metric_name_inter_tenant)} (Round: {round_name_inter_tenant})',
                                            cbar_label='Correlation Coefficient'
                                        )
                                        fig_inter_tenant_corr.savefig(plot_path_inter_tenant_corr)
                                        plt.close(fig_inter_tenant_corr)
                                        print(f"      Inter-tenant correlation matrix saved for {metric_name_inter_tenant}, round {round_name_inter_tenant}.")
                                        current_advanced_results_dict.setdefault('inter_tenant_correlation_matrices', {}).setdefault(metric_name_inter_tenant, {}).setdefault(round_name_inter_tenant, {}).update(inter_tenant_corr_matrix.to_dict())
                                    else:
                                        print(f"      Inter-tenant correlation matrix empty for {metric_name_inter_tenant}, round {round_name_inter_tenant}.")
                                except Exception as e_inter_corr:
                                    print(f"      Error calculating/plotting inter-tenant correlation for {metric_name_inter_tenant}, round {round_name_inter_tenant}: {e_inter_corr}")

                                # Inter-Tenant Covariance (Non-Aggregated)
                                try:
                                    print(f"    Calculating inter-tenant covariance for metric: {metric_name_inter_tenant}, round: {round_name_inter_tenant}")
                                    inter_tenant_cov_matrix = calculate_inter_tenant_covariance_per_metric(df_to_analyze) # Uses default 'value', 'datetime', 'tenant'
                                    if inter_tenant_cov_matrix is not None and not inter_tenant_cov_matrix.empty:
                                        csv_path_inter_tenant_cov = os.path.join(advanced_tables_dir, f'inter_tenant_covariance_{metric_name_inter_tenant}_round_{round_name_inter_tenant}.csv')
                                        export_to_csv(inter_tenant_cov_matrix, csv_path_inter_tenant_cov)
                                        plot_path_inter_tenant_cov = os.path.join(advanced_plots_dir, f'inter_tenant_covariance_heatmap_{metric_name_inter_tenant}_round_{round_name_inter_tenant}.png')
                                        fig_inter_tenant_cov = plot_correlation_heatmap(
                                            inter_tenant_cov_matrix,
                                            title=f'Inter-Tenant Covariance - {METRIC_DISPLAY_NAMES.get(metric_name_inter_tenant, metric_name_inter_tenant)} (Round: {round_name_inter_tenant})',
                                            cbar_label='Covariance'
                                        )
                                        fig_inter_tenant_cov.savefig(plot_path_inter_tenant_cov)
                                        plt.close(fig_inter_tenant_cov)
                                        print(f"      Inter-tenant covariance matrix saved for {metric_name_inter_tenant}, round {round_name_inter_tenant}.")
                                        current_advanced_results_dict.setdefault('inter_tenant_covariance_matrices', {}).setdefault(metric_name_inter_tenant, {}).setdefault(round_name_inter_tenant, {}).update(inter_tenant_cov_matrix.to_dict())
                                    else:
                                        print(f"      Inter-tenant covariance matrix empty for {metric_name_inter_tenant}, round {round_name_inter_tenant}.")
                                except Exception as e_inter_cov:
                                    print(f"      Error calculating/plotting inter-tenant covariance for {metric_name_inter_tenant}, round {round_name_inter_tenant}: {e_inter_cov}")
                            elif isinstance(df_round_inter_tenant, pd.DataFrame) and df_round_inter_tenant['tenant'].nunique() <= 1:
                                print(f"    Skipping inter-tenant correlation/covariance for {metric_name_inter_tenant}, round {round_name_inter_tenant}: Less than 2 tenants in original data.")
            
            # 3. Entropy Analysis (Non-Aggregated)
            print(f"\n  Calculating entropy metrics (NON-AGGREGATED)...")
            all_entropy_results = []
            if metrics_data: # Check if metrics_data is not None
                for metric_name_entropy, rounds_data_entropy in metrics_data.items():
                    if isinstance(rounds_data_entropy, dict):
                        rounds_to_analyze_entropy = []
                        if args.round and len(args.round) == 1:
                            if args.round[0] in rounds_data_entropy:
                                rounds_to_analyze_entropy.append(args.round[0])
                        else:
                            rounds_to_analyze_entropy = rounds_data_entropy.keys()
                        
                        for round_name_entropy in rounds_to_analyze_entropy:
                            df_round_entropy = rounds_data_entropy.get(round_name_entropy)
                            if isinstance(df_round_entropy, pd.DataFrame) and not df_round_entropy.empty and 'tenant' in df_round_entropy.columns:
                                df_to_analyze_entropy = df_round_entropy
                                if args.tenants: # Filter by tenants if specified
                                    df_to_analyze_entropy = df_round_entropy[df_round_entropy['tenant'].isin(args.tenants)]
                                
                                if df_to_analyze_entropy['tenant'].nunique() < 2:
                                    print(f"    Skipping entropy for {metric_name_entropy}, round {round_name_entropy}: Less than 2 tenants after filtering.")
                                    continue

                                print(f"    Calculating entropy for metric: {metric_name_entropy}, round: {round_name_entropy}")
                                try:
                                    # calculate_entropy_metrics can iterate through phases if 'phase' column exists and phase arg is None
                                    # Or analyze a specific phase if phase arg is provided.
                                    # For non-aggregated, we typically analyze per round, possibly all phases within that round or specific phases.
                                    # Let's assume for now it processes all phases within the round's data or as per its internal logic.
                                    entropy_results_df = calculate_entropy_metrics(
                                        df_to_analyze_entropy, 
                                        tenants=None, # Uses tenants from df_to_analyze_entropy
                                        phase=None, # Analyze all phases within this round's data, or specific if df is pre-filtered
                                        metric_column='value'
                                    )
                                    if entropy_results_df is not None and not entropy_results_df.empty:
                                        entropy_results_df['metric'] = metric_name_entropy
                                        entropy_results_df['round'] = round_name_entropy
                                        all_entropy_results.append(entropy_results_df)
                                        table_path_entropy = os.path.join(advanced_tables_dir, f'entropy_analysis_{metric_name_entropy}_round_{round_name_entropy}.csv')
                                        export_to_csv(entropy_results_df, table_path_entropy)
                                        print(f"      Entropy analysis results saved to: {table_path_entropy}")
                                    else:
                                        print(f"      No entropy results for {metric_name_entropy}, round {round_name_entropy}.")
                                except Exception as e_entropy:
                                    print(f"    Error calculating entropy for {metric_name_entropy}, round {round_name_entropy}: {e_entropy}")
            if all_entropy_results:
                final_entropy_df = pd.concat(all_entropy_results, ignore_index=True)
                current_advanced_results_dict['entropy_analysis'] = final_entropy_df.to_dict(orient='records')
                export_to_csv(final_entropy_df, os.path.join(advanced_tables_dir, 'entropy_analysis_combined.csv'))
                print("  Combined entropy analysis results (non-aggregated) saved.")
            
            experiment_results['advanced_analysis'] = current_advanced_results_dict

    # Análise de Impacto Normalizado (após todas as outras análises avançadas)
    if args.advanced and metrics_data: # Pode ser executado com dados agregados ou não
        print("\nCalculating Normalized Impact Score...")
        
        # Determinar quais dados usar: agregados ou não
        data_for_impact_score = {}
        value_col_for_impact = 'value' # Default para dados não agregados

        if args.use_aggregated_rounds_for_advanced and aggregated_data_for_advanced:
            print("  Using AGGREGATED data for Normalized Impact Score.")
            data_for_impact_score = aggregated_data_for_advanced
            value_col_for_impact = 'mean'
        elif metrics_data:
            print("  Using NON-AGGREGATED data for Normalized Impact Score.")
            if not all_metrics_data_concatenated:
                for metric_name_loop, rounds_data_for_metric_loop in metrics_data.items():
                    if isinstance(rounds_data_for_metric_loop, dict) and rounds_data_for_metric_loop:
                        all_rounds_dfs_loop = []
                        for round_name_loop, round_df_loop in rounds_data_for_metric_loop.items():
                            if isinstance(round_df_loop, pd.DataFrame) and not round_df_loop.empty:
                                df_copy = round_df_loop.copy()
                                if 'round' not in df_copy.columns:
                                    df_copy['round'] = round_name_loop
                                all_rounds_dfs_loop.append(df_copy)
                        if all_rounds_dfs_loop:
                            all_metrics_data_concatenated[metric_name_loop] = pd.concat(all_rounds_dfs_loop, ignore_index=True)
            data_for_impact_score = all_metrics_data_concatenated
            value_col_for_impact = 'value'
        else:
            print("  No suitable data (aggregated or non-aggregated) found for Normalized Impact Score calculation.")

        if data_for_impact_score:
            impact_phases_for_score = [p for p in PHASE_DISPLAY_NAMES.keys() if "attack" in p.lower() or "stress" in p.lower()]
            baseline_phases_for_score = [p for p in PHASE_DISPLAY_NAMES.keys() if "baseline" in p.lower()]

            if not impact_phases_for_score:
                print("Warning: No impact phases (e.g., '2 - Attack') found in PHASE_DISPLAY_NAMES. Impact score may be incorrect.")
                impact_phases_for_score = ["2 - Attack"]
            if not baseline_phases_for_score:
                print("Warning: No baseline phases (e.g., '1 - Baseline') found in PHASE_DISPLAY_NAMES. Impact score may be incorrect.")
                baseline_phases_for_score = ["1 - Baseline"]

            impact_score_df = calculate_normalized_impact_score(
                metrics_data=data_for_impact_score,
                noisy_tenant=args.noisy_tenant if args.noisy_tenant else DEFAULT_NOISY_TENANT,
                impact_phases=impact_phases_for_score,
                baseline_phases=baseline_phases_for_score,
                weights=IMPACT_CALCULATION_DEFAULTS.get('weights', {}),
                metrics_config=IMPACT_CALCULATION_DEFAULTS.get('metrics_config', {}),
                value_col=value_col_for_impact,
                tenant_col='tenant',
                phase_col='phase',
                round_col='round'
            )

            if impact_score_df is not None and not impact_score_df.empty:
                print("\nNormalized Impact Scores:")
                print(impact_score_df)
                impact_score_table_path = os.path.join(advanced_tables_dir, 'normalized_impact_scores.csv')
                export_to_csv(impact_score_df, impact_score_table_path)
                print(f"  Normalized impact scores saved to: {impact_score_table_path}")

                if 'advanced_analysis' not in experiment_results:
                    experiment_results['advanced_analysis'] = {}
                experiment_results['advanced_analysis']['normalized_impact_score'] = impact_score_df.to_dict(orient='records')
                
                fig_impact_bar = plot_impact_score_barplot(impact_score_df, score_col='normalized_impact_score', tenant_col='tenant')
                if fig_impact_bar:
                    plot_path_bar = os.path.join(advanced_plots_dir, 'normalized_impact_score_barplot.png')
                    fig_impact_bar.savefig(plot_path_bar)
                    plt.close(fig_impact_bar)
                    print(f"  Impact score barplot saved to: {plot_path_bar}")
            else:
                print("  Normalized impact score calculation did not return results.")
        else:
            print("  Skipping normalized impact score calculation as no data was prepared.")

    # COMPARAÇÃO ENTRE RODADAS (INTRA-EXPERIMENTO)
    if args.compare_rounds_intra and metrics_data:
        compare_rounds_within_experiment(
            experiment_results=experiment_results,
            output_dir_main=rounds_comparison_intra_dir,
            metrics_to_compare=args.metrics,
            phases_to_compare=args.phases,
            show_as_percentage=args.show_as_percentage,
            node_config=node_config_to_use
        )

    print("\nPipeline execution finished successfully!")
    
    return experiment_results


if __name__ == "__main__":
    main()
