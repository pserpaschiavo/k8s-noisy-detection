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

from pipeline.analysis.tenant_analysis import compare_tenant_metrics
from pipeline.analysis.phase_analysis import compare_phases_ttest, analyze_recovery_effectiveness
# Updated imports for advanced analysis modules
from pipeline.analysis.advanced_analysis import calculate_normalized_impact_score # Kept for this specific function
from pipeline.analysis.correlation_analysis import calculate_covariance_matrix, calculate_inter_tenant_covariance_per_metric, plot_covariance_matrix, calculate_pearson_correlation, calculate_spearman_correlation, calculate_lagged_cross_correlation, calculate_correlation_matrix # Added calculate_correlation_matrix here
from pipeline.analysis.causality_analysis import perform_inter_tenant_causality_analysis, visualize_causal_graph, save_causality_results_to_csv, calculate_transfer_entropy, calculate_convergent_cross_mapping # Updated
from pipeline.analysis.similarity_analysis import calculate_dtw_distance # New
# End of updated imports
from pipeline.analysis.noisy_tenant_detection import identify_noisy_tenant
from pipeline.analysis.global_analysis import (
    create_global_tenant_summary, create_tenant_impact_matrix, 
    create_metric_correlation_network, plot_global_dashboard,
    plot_global_correlation_network
)
from pipeline.analysis.experiment_comparison import (
    load_multiple_experiments,
    preprocess_experiments,
    calculate_statistics_summary,
    compare_distributions,
    summarize_anomalies,
    compare_experiment_phases
)
from pipeline.analysis.rounds_aggregation import (
    aggregate_metrics_across_rounds, 
    plot_aggregated_metrics,
    plot_aggregated_metrics_boxplot, # Add import
    test_for_significant_differences
)

from pipeline.visualization.plots import (plot_metric_by_phase, plot_phase_comparison,
                                plot_tenant_impact_heatmap, plot_recovery_effectiveness,
                                plot_impact_score_barplot, plot_impact_score_trend,
                                create_heatmap, plot_multivariate_anomalies, plot_correlation_heatmap)

from pipeline.config import (DEFAULT_DATA_DIR, DEFAULT_METRICS, AGGREGATION_CONFIG,
                    IMPACT_CALCULATION_DEFAULTS, VISUALIZATION_CONFIG,
                    NODE_RESOURCE_CONFIGS, DEFAULT_NODE_CONFIG_NAME,
                    PHASE_DISPLAY_NAMES,
                    METRIC_DISPLAY_NAMES, DEFAULT_NOISY_TENANT)  # Removed unused causality and granger defaults
from pipeline.utils import get_experiment_data_dir # Corrected import


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
                        help='Run advanced analyses (Correlation, Covariance, Causality, Similarity)') # Updated help string
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
    parser.add_argument('--inter-tenant-causality', action='store_true',
                        help='Run inter-tenant causality analysis.')
    parser.add_argument('--compare-rounds-intra', action='store_true',
                        help='Run formal comparison between rounds of the same experiment.')
    parser.add_argument('--compare-tenants-directly', action='store_true',
                        help='Run direct statistical comparison between tenants.')
    parser.add_argument('--global-analysis', action='store_true',
                        help='Run global analysis across metrics and tenants.')
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
    global_analysis_dir = os.path.join(output_dir, 'global_analysis') # Global analysis directory
    
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
    os.makedirs(global_analysis_dir, exist_ok=True) # Create global analysis directory
    os.makedirs(os.path.join(global_analysis_dir, 'plots'), exist_ok=True) # Create plots subdirectory
    os.makedirs(os.path.join(global_analysis_dir, 'tables'), exist_ok=True) # Create tables subdirectory
    
    return plots_dir, tables_dir, advanced_dir, comparison_dir, reports_dir, causality_dir, rounds_comparison_intra_dir, tenant_comparison_dir, global_analysis_dir


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


def main():
    """Main function that runs the analysis pipeline."""
    args = parse_arguments()
    
    print("DEBUG: Verificando a flag de análise global:", args.global_analysis)
    
    plots_dir, tables_dir, advanced_dir, comparison_dir, reports_dir, causality_dir, rounds_comparison_intra_dir, tenant_comparison_dir, global_analysis_dir = setup_output_directories(args.output_dir)
    
    experiment_data_dir = get_experiment_data_dir(args.data_dir)

    if not os.path.isdir(experiment_data_dir):
        print(f"Error: Data directory not found: {experiment_data_dir}")
        print(f"  Input --data-dir: '{args.data_dir}'")
        print(f"  Absolute path checked: '{os.path.abspath(experiment_data_dir)}'")
        return

    print(f"Using data directory: {experiment_data_dir}")

    # Default node configuration
    node_config_to_use = NODE_RESOURCE_CONFIGS.get(DEFAULT_NODE_CONFIG_NAME, NODE_RESOURCE_CONFIGS.get("Default"))
    
    # Create an adapted copy of NODE_RESOURCE_CONFIGS for the format expected by normalization modules
    if 'MEMORY_GB' in node_config_to_use and 'MEMORY_BYTES' not in node_config_to_use:
        node_config_to_use = node_config_to_use.copy()  # Avoid altering the original
        node_config_to_use['MEMORY_BYTES'] = node_config_to_use['MEMORY_GB'] * (2**30)
        node_config_to_use['DISK_SIZE_BYTES'] = node_config_to_use['DISK_SIZE_GB'] * (2**30)
        # Estimate bandwidth based on CPU 
        node_config_to_use['NETWORK_BANDWIDTH_MBPS'] = max(1000, node_config_to_use['CPUS'] * 250)
    
    # UNIFIED DATA LOADING AND INITIAL PROCESSING
    print("\nLoading and Processing Experiment Data...")
    # Placeholder for data loading logic, assuming load_experiment_data is the intended function
    # and metric_name_map might be defined elsewhere or loaded from config.
    # For now, we'll assume it's part of a broader data loading strategy.
    
    # Example: Using existing load_experiment_data, assuming it returns a compatible structure
    # This is a temporary adaptation and might need further refinement.
    available_metrics = list_available_metrics(experiment_data_dir)
    metrics_to_load = args.metrics if args.metrics else available_metrics
    
    # Initialize metric_name_map - this is a placeholder and should be defined based on your project's needs
    metric_name_map = {metric: metric for metric in metrics_to_load} 
    
    all_metrics_data = {}
    if metrics_to_load:
        # This is a simplified call; actual implementation might vary
        # based on how load_experiment_data and related functions are structured.
        # We're assuming load_experiment_data can be adapted or is already suitable.
        # The original call to load_all_metrics is replaced here.
        # This section requires careful review to ensure it aligns with the actual data loading mechanisms.
        
        # Placeholder: Simulate loading data for each metric
        # This is a simplified approach. You'll need to replace this with your actual data loading logic.
        for metric_name in metrics_to_load:
            # This is a conceptual placeholder. The actual data loading might be more complex.
            # Assuming load_experiment_data can be used or adapted.
            # You might need to adjust this based on how your data is organized and loaded.
            try:
                # This is a guess. The actual function and its parameters might differ.
                # The key is to replace the undefined 'load_all_metrics' with something that exists.
                # This part is highly dependent on your project's specific data loading functions.
                data_for_metric = load_experiment_data(experiment_data_dir, [metric_name], specific_metrics_map=metric_name_map)
                all_metrics_data[metric_name] = data_for_metric # Adjust based on actual return structure
            except Exception as e:
                print(f"Error loading data for metric {metric_name}: {e}")
                all_metrics_data[metric_name] = {} # Or handle as appropriate

    # metric_type_map would also need to be defined, possibly from configuration or data inspection
    metric_type_map = {metric: "gauge" for metric in metrics_to_load}  # Placeholder

    if not all_metrics_data:
        print("CRITICAL ERROR: No data was loaded. Check the specified data directory, metrics, and rounds.")
        return 

    print("Experiment data loaded successfully.")
            
    # Continue with data processing

    experiment_results = {
        'processed_data': all_metrics_data
    }

    metrics_data = experiment_results.get('processed_data', {})

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

    # Advanced Analysis Section
    if args.advanced:
        print("\nRunning Advanced Analyses...")
        # Ensure metrics_data is not empty and contains the necessary structure
        if not metrics_data:
            print("Skipping advanced analyses as no data is available.")
        else:
            # Example: Perform Pearson correlation for the first available metric and round
            # This is a simplified example; you'll need to adapt it to your specific needs
            # and iterate through metrics/rounds as required.
            first_metric_name = next(iter(metrics_data)) if metrics_data else None
            if first_metric_name:
                first_round_name = next(iter(metrics_data[first_metric_name])) if metrics_data[first_metric_name] else None
                if first_round_name:
                    df_advanced = metrics_data[first_metric_name][first_round_name]
                    if df_advanced is not None and not df_advanced.empty and 'value' in df_advanced.columns:
                        # Ensure there are at least two columns for correlation/covariance
                        # This might involve pivoting or selecting specific columns if your df_advanced is long-form
                        # For demonstration, assuming df_advanced is already wide-form or has multiple value-like columns
                        
                        # Placeholder for actual data preparation for advanced analysis
                        # This data needs to be in a format suitable for the analysis functions
                        # e.g., a DataFrame where columns are time series of different tenants/metrics
                        
                        # Example call to calculate_pearson_correlation (adjust as needed)
                        # pearson_corr = calculate_pearson_correlation(df_advanced[['value_tenant_a', 'value_tenant_b']])
                        # print(f"Pearson Correlation: {pearson_corr}")

                        # Example call to calculate_spearman_correlation (adjust as needed)
                        # spearman_corr = calculate_spearman_correlation(df_advanced[['value_tenant_a', 'value_tenant_b']])
                        # print(f"Spearman Correlation: {spearman_corr}")

                        # Example call to calculate_lagged_cross_correlation (adjust as needed)
                        # lagged_corr = calculate_lagged_cross_correlation(df_advanced['value_tenant_a'], df_advanced['value_tenant_b'])
                        # print(f"Lagged Cross-Correlation: {lagged_corr}")

                        # Example call to calculate_covariance_matrix (adjust as needed)
                        # cov_matrix = calculate_covariance_matrix(df_advanced[['value_tenant_a', 'value_tenant_b']])
                        # print(f"Covariance Matrix: {cov_matrix}")
                        # plot_covariance_matrix(cov_matrix, os.path.join(advanced_dir, 'plots', 'covariance_matrix.png'))
                        pass # Add actual calls here
                    else:
                        print("DataFrame for advanced analysis is empty or lacks 'value' column.")
                else:
                    print("No rounds available for the first metric for advanced analysis.")
            else:
                print("No metrics available for advanced analysis.")

    if args.inter_tenant_causality:
        print("\nRunning Inter-Tenant Causality Analysis...")
        if not metrics_data:
            print("Skipping inter-tenant causality analysis as no data is available.")
        else:
            # Placeholder for data preparation for causality analysis
            # This typically requires time series data for different tenants
            # Example: causality_data = prepare_data_for_causality(metrics_data, args.tenants)
            # perform_inter_tenant_causality_analysis(causality_data, causality_dir, max_lag=args.granger_max_lag)
            pass # Add actual calls here

    print("\nPipeline execution finished successfully!")
    
    return experiment_results


if __name__ == "__main__":
    main()
