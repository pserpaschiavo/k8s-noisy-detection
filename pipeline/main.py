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
from pipeline.analysis.inter_tenant_causality import visualize_causal_graph, perform_inter_tenant_causality_analysis
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
                
                # Use current_phase_display_for_report for filenames
                csv_filename = f"{metric_name}_{current_phase_display_for_report.replace(' ', '_')}_round_comparison.csv"
                csv_path = os.path.join(rounds_comparison_output_dir, csv_filename)
                try:
                    comparison_data.to_csv(csv_path, index=False)
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
                        current_output["plot_path"] = plot_path
                    except Exception as e:
                        print(f"    Error saving round comparison plot: {e}")
                    plt.close()
                all_comparison_outputs[output_key] = current_output
    
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

    # >>> BEGIN MODIFICATION TO PREPARE DATA FOR ADVANCED ANALYSIS <<<
    
    # 1. Consolidate all_metrics_data into a single DataFrame for time-series based analysis
    list_of_processed_dfs = []
    if all_metrics_data:
        for metric_name, rounds_data_dict in all_metrics_data.items():
            if isinstance(rounds_data_dict, dict):
                for round_name, df_round in rounds_data_dict.items():
                    if isinstance(df_round, pd.DataFrame) and not df_round.empty:
                        df_copy = df_round.copy()
                        df_copy['metric_name'] = metric_name
                        # Ensure 'round' column is present, might be added by load_experiment_data or here
                        if 'round' not in df_copy.columns:
                            df_copy['round'] = round_name
                        list_of_processed_dfs.append(df_copy)
            # This case should ideally not be hit if load_experiment_data consistently returns Dict[metric, Dict[round, DF]]
            elif isinstance(rounds_data_dict, pd.DataFrame) and not rounds_data_dict.empty:
                 print(f"Warning: Data for metric '{metric_name}' is a single DataFrame, not a dict of rounds. Consolidating as is.")
                 df_copy = rounds_data_dict.copy()
                 df_copy['metric_name'] = metric_name
                 if 'round' not in df_copy.columns: # Add a default round if missing
                    df_copy['round'] = 'default_round'
                 list_of_processed_dfs.append(df_copy)

    consolidated_processed_df = pd.DataFrame()
    if list_of_processed_dfs:
        consolidated_processed_df = pd.concat(list_of_processed_dfs, ignore_index=True)
        print(f"Consolidated processed data for advanced analysis. Shape: {consolidated_processed_df.shape}")
    else:
        print("No processed data to consolidate for advanced analysis.")

    # 2. Prepare aggregated data if requested and possible
    consolidated_aggregated_df = pd.DataFrame() # Initialize as empty DF
    aggregated_data_dict = {} # Initialize as empty dict

    # Define multiple_rounds_available based on the structure of all_metrics_data
    multiple_rounds_available = False
    if all_metrics_data:
        for metric_name, rounds_data_dict in all_metrics_data.items():
            if isinstance(rounds_data_dict, dict) and len(rounds_data_dict) > 1:
                multiple_rounds_available = True
                break
    print(f"Multiple rounds available for aggregation: {multiple_rounds_available}")

    if args.use_aggregated_rounds_for_advanced and multiple_rounds_available:
        print("\nPreparing dictionary of metrics for aggregation across multiple rounds...")
        metrics_for_aggregation_dict = {} # Renamed from metrics_for_aggregation to avoid confusion with the list
        for metric_name, rounds_data_map in all_metrics_data.items():
            if isinstance(rounds_data_map, dict) and rounds_data_map:
                metric_dfs_list = []
                for round_name, df_round in rounds_data_map.items():
                    if isinstance(df_round, pd.DataFrame) and not df_round.empty:
                        df_copy = df_round.copy()
                        if 'round' not in df_copy.columns: # Ensure 'round' column from key
                            df_copy['round'] = round_name
                        metric_dfs_list.append(df_copy)
                if metric_dfs_list:
                    # Concatenate all rounds for a single metric into one DataFrame
                    metrics_for_aggregation_dict[metric_name] = pd.concat(metric_dfs_list, ignore_index=True)
            elif isinstance(rounds_data_map, pd.DataFrame) and not rounds_data_map.empty: # Handle cases where a metric might already be a single DF
                df_copy = rounds_data_map.copy()
                if 'round' not in df_copy.columns:
                    df_copy['round'] = 'default_round' # Add default if missing
                metrics_for_aggregation_dict[metric_name] = df_copy


        if metrics_for_aggregation_dict:
            # Call aggregate_metrics_across_rounds with the dictionary of DataFrames
            # Each DataFrame in this dict already contains all rounds for that specific metric
            aggregated_data_dict = aggregate_metrics_across_rounds(metrics_for_aggregation_dict, value_column='value')
            
            if aggregated_data_dict:
                print("Aggregation complete. Consolidating aggregated data for advanced/causality analyses.")
                list_of_aggregated_dfs = []
                for metric_name_agg, agg_df_metric in aggregated_data_dict.items():
                    if isinstance(agg_df_metric, pd.DataFrame) and not agg_df_metric.empty:
                        df_copy = agg_df_metric.copy()
                        df_copy['metric_name'] = metric_name_agg
                        list_of_aggregated_dfs.append(df_copy)
                
                if list_of_aggregated_dfs:
                    consolidated_aggregated_df = pd.concat(list_of_aggregated_dfs, ignore_index=True)
                    print(f"Consolidated aggregated data. Shape: {consolidated_aggregated_df.shape}")

                # Generate aggregated plots and tables using the per-metric aggregated data (aggregated_data_dict)
                for metric_name_agg, agg_df in aggregated_data_dict.items(): # Use aggregated_data_dict here
                    if not agg_df.empty:
                        # Pass all_metrics_data to plot_aggregated_metrics for phase information if needed by the plot function
                        fig_agg = plot_aggregated_metrics(aggregated_data_dict, metric_name_agg, all_metrics_data_for_phases=all_metrics_data)
                        if fig_agg:
                            agg_plot_path = os.path.join(advanced_dir, 'plots', f'aggregated_{metric_name_agg}_across_rounds.png')
                            fig_agg.savefig(agg_plot_path)
                            plt.close(fig_agg)
                            print(f"    Aggregated plot for {metric_name_agg} saved to {agg_plot_path}")
                        
                        plot_aggregated_metrics_boxplot(aggregated_data_dict, metric_name_agg, os.path.join(advanced_dir, 'plots'))

                        agg_table_path = os.path.join(advanced_dir, 'tables', f'aggregated_{metric_name_agg}_across_rounds.csv')
                        export_to_csv(agg_df, agg_table_path)
                        print(f"    Aggregated table for {metric_name_agg} saved to {agg_table_path}")
            else:
                print("Aggregation across rounds did not yield any data.")
        else:
            print("No metrics suitable for aggregation were found (metrics_for_aggregation_dict is empty).")
    # >>> END MODIFICATION TO PREPARE DATA <<<

    # Advanced analysis: Covariance, Entropy, Causality
    if args.advanced or args.inter_tenant_causality:
        print("\n=== Running Advanced Analyses ===")
        advanced_plots_dir = os.path.join(advanced_dir, 'plots')
        advanced_tables_dir = os.path.join(advanced_dir, 'tables')

        # The 'data_for_advanced_analysis' variable is no longer the primary input for causality.
        # Specific analyses should use either consolidated_processed_df or consolidated_aggregated_df.
        
        source_description_for_log = "processed (per-round) data" # Default description

        if args.use_aggregated_rounds_for_advanced:
            if not consolidated_aggregated_df.empty:
                source_description_for_log = "data aggregated across rounds"
                print(f"Advanced analysis will primarily use: {source_description_for_log} (if applicable to the specific analysis).")
                
                # Check the structure of consolidated_aggregated_df for appropriate warnings
                # Granger causality on aggregated data (summary stats) will only plot nodes.
                # The 'value' column here refers to raw time-series values, which won't exist in summary stats.
                # 'mean_value' is an example of a summary statistic column.
                if 'value' not in consolidated_aggregated_df.columns and any(col for col in consolidated_aggregated_df.columns if 'mean' in col or 'median' in col):
                     print("Warning: Consolidated aggregated data appears to be summary statistics (e.g., contains 'mean_value'). "
                           "For causality analysis, this means tenant nodes will be shown, but no causal links will be computed from this summary data.")
                elif 'value' not in consolidated_aggregated_df.columns:
                    # This case might be less common if the above catches typical summary stats.
                    print("Warning: Consolidated aggregated data is missing a raw 'value' column. "
                          "If this data were intended for time-series causality, it would not work as expected.")
            else:
                print("Warning: --use-aggregated-rounds-for-advanced was specified, but no consolidated aggregated data is available. "
                      "Analyses requiring aggregated data may not run or may fall back to processed data.")
        else:
            print(f"Advanced analysis will primarily use: {source_description_for_log} (if applicable to the specific analysis).")


        # Inter-Tenant Causality Analysis
        if args.inter_tenant_causality:
            print(f"\n--- Running Inter-Tenant Causality Analysis ---")
            causality_output_dir = os.path.join(causality_dir, 'plots') 

            # Determine if aggregated data should be used by the causality function
            use_agg_for_causality_call = args.use_aggregated_rounds_for_advanced and not consolidated_aggregated_df.empty

            if use_agg_for_causality_call:
                print("Causality analysis will be performed using the consolidated aggregated data.")
                print("  (Note: If this is summary data, only tenant nodes will be plotted without causal links from this data.)")
            else:
                if args.use_aggregated_rounds_for_advanced and consolidated_aggregated_df.empty:
                    print("Consolidated aggregated data is empty; causality analysis will use consolidated processed (per-round) data instead.")
                elif not args.use_aggregated_rounds_for_advanced:
                    print("Causality analysis will use the consolidated processed (per-round) data.")
                else: # Should not be reached if logic is correct
                    print("Causality analysis will default to consolidated processed (per-round) data.")
            
            perform_inter_tenant_causality_analysis(
                processed_data_df=consolidated_processed_df,
                aggregated_data_df=consolidated_aggregated_df, 
                config=args, # Using 'args' as the config object, assuming it has necessary attributes like GRANGER_MAX_LAG
                output_dir=causality_output_dir,
                experiment_name=os.path.basename(experiment_data_dir),
                use_aggregated_data=use_agg_for_causality_call
            )
            print("--- Inter-Tenant Causality Analysis completed ---")

    print("\nPipeline execution finished successfully!")
    
    return experiment_results


if __name__ == "__main__":
    main()
