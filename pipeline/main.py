"""
Main script for the data analysis pipeline of the noisy neighbors experiment.

This script orchestrates the entire pipeline, from data loading to
the generation of visualizations and reports.
"""

import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pipeline.analysis.correlation_analysis import (
    calculate_pearson_correlation,
    calculate_spearman_correlation
)
from pipeline.analysis.causality_analysis import (
    perform_granger_causality_test,
    calculate_transfer_entropy,
    calculate_convergent_cross_mapping
)
from pipeline.analysis.similarity_analysis import (
    calculate_dtw_distance,
    calculate_cosine_similarity,
    calculate_time_varying_cosine_similarity
)
from pipeline.visualization.plots import (
    plot_metric_by_phase, plot_phase_comparison,
    plot_tenant_impact_heatmap, plot_recovery_effectiveness,
    plot_impact_score_barplot, plot_impact_score_trend,
    plot_multivariate_anomalies, 
    plot_correlation_heatmap,
    visualize_causal_graph,
    plot_time_series_with_cosine_similarity,
    plot_cosine_similarity_heatmap,
    plot_time_varying_cosine_similarity
)
from pipeline.config import (
    DEFAULT_DATA_DIR, DEFAULT_METRICS, AGGREGATION_CONFIG,
    IMPACT_CALCULATION_DEFAULTS, VISUALIZATION_CONFIG,
    NODE_RESOURCE_CONFIGS, DEFAULT_NODE_CONFIG_NAME,
    PHASE_DISPLAY_NAMES, METRIC_DISPLAY_NAMES, DEFAULT_NOISY_TENANT,
    DEFAULT_METRICS_FOR_CAUSALITY, DEFAULT_CAUSALITY_MAX_LAG,
    STATISTICAL_CONFIG, DEFAULT_GRANGER_MIN_OBSERVATIONS,
    TENANT_COLORS, DEFAULT_OUTPUT_DIR, TABLE_EXPORT_CONFIG,
    METRICS_CONFIG, IMPACT_SCORE_WEIGHTS, DEFAULT_CAUSALITY_THRESHOLD_P_VALUE,
    GRANGER_CAUSALITY_DEFAULTS
)
from pipeline.data_processing.data_loader import load_experiment_data, list_available_metrics
from pipeline.utils import (get_experiment_data_dir, add_experiment_elapsed_time, add_phase_markers)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Data analysis pipeline for the noisy neighbors experiment.')
    parser.add_argument('--data-dir', type=str, default='/home/phil/Projects/k8s-noisy-detection/demo-data/demo-experiment-3-rounds',
                        help='Directory with the experiment data')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save the results')
    parser.add_argument('--tenants', type=str, nargs='+',
                        help='Specific tenant(s) to analyze')
    parser.add_argument('--metrics', type=str, nargs='+',
                        help='Specific metric(s) to analyze')
    parser.add_argument('--phases', type=str, nargs='+',
                        help='Specific phase(s) to analyze')
    parser.add_argument('--rounds', type=str, nargs='+',
                        help='Specific round(s) to analyze')
    parser.add_argument('--node-config', type=str, default=None)
    parser.add_argument('--advanced', action='store_true',
                        help='Run advanced analyses')
    parser.add_argument('--inter-tenant-causality', action='store_true',
                        help='Run inter-tenant causality analysis')

    return parser.parse_args()


def setup_output_directories(output_dir):
    """Set up output directories."""
    plots_dir = os.path.join(output_dir, 'plots')
    tables_dir = os.path.join(output_dir, 'tables')
    advanced_dir = os.path.join(output_dir, 'advanced_analysis')
    causality_dir = os.path.join(advanced_dir, 'causality')
    correlation_dir = os.path.join(advanced_dir, 'correlation')
    similarity_dir = os.path.join(advanced_dir, 'similarity')
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(advanced_dir, exist_ok=True)
    os.makedirs(causality_dir, exist_ok=True)
    os.makedirs(correlation_dir, exist_ok=True)
    os.makedirs(similarity_dir, exist_ok=True)
    
    return plots_dir, tables_dir, advanced_dir, causality_dir, correlation_dir, similarity_dir


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
        metrics_to_compare = DEFAULT_METRICS
    if phases_to_compare is None:
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

        for raw_phase_name in phases_to_compare:
            current_phase_display_for_report = PHASE_DISPLAY_NAMES.get(raw_phase_name, raw_phase_name)
            print(f"  Comparing rounds for Metric: {metric_display_name}, Phase: {current_phase_display_for_report} (Raw: {raw_phase_name})")
            
            phase_specific_df = metric_df[metric_df['phase'] == raw_phase_name]
            
            output_key = f"{metric_name}_{current_phase_display_for_report.replace(' ', '_')}"
            current_output = {"csv_path": None, "plot_path": None, "anova_f_stat": None, "anova_p_value": None}

            if not phase_specific_df.empty:
                rounds_count = phase_specific_df['round'].value_counts()
                valid_rounds = rounds_count[rounds_count >= 5].index.tolist()
                
                if not valid_rounds:
                    print(f"    Not enough rounds with sufficient data for analysis. Skipping.")
                    continue
                    
                phase_specific_df_filtered = phase_specific_df[phase_specific_df['round'].isin(valid_rounds)]
                
                if 'tenant' in phase_specific_df_filtered.columns and len(phase_specific_df_filtered['tenant'].unique()) > 1:
                    comparison_data_agg = phase_specific_df_filtered.groupby(['round', 'tenant'])['value'].mean().reset_index()
                    comparison_data = comparison_data_agg.groupby('round')['value'].mean().reset_index()
                    comparison_data.rename(columns={'value': f'mean_value_across_tenants'}, inplace=True)
                else:
                    comparison_data = phase_specific_df_filtered.groupby('round')['value'].mean().reset_index()
                    comparison_data.rename(columns={'value': 'mean_value'}, inplace=True)
                
                csv_filename = f"{metric_name}_{current_phase_display_for_report.replace(' ', '_')}_round_comparison.csv"
                csv_path = os.path.join(rounds_comparison_output_dir, csv_filename)
                try:
                    comparison_data.to_csv(csv_path, index=False)
                    current_output["csv_path"] = csv_path
                except Exception as e:
                    print(f"    Error saving round comparison CSV: {e}")

                rounds_with_data = phase_specific_df['round'].unique()
                if len(rounds_with_data) >= 2:
                    grouped_values = [
                        group['value'].dropna() for name, group in phase_specific_df.groupby('round')
                        if not group['value'].dropna().empty
                    ]
                    if len(grouped_values) >= 2:
                        try:
                            pass
                        except Exception as e:
                            print(f"    Error performing ANOVA (exception during attempt): {e}")
                            pass
                    else:
                        print("    Not enough groups with data for ANOVA after filtering.")
                else:
                    print("    Not enough rounds with data to perform ANOVA.")

                if not comparison_data.empty:
                    plt.figure(figsize=VISUALIZATION_CONFIG.get('figure_size', (10, 6)))
                    value_col_for_plot = 'mean_value_across_tenants' if 'mean_value_across_tenants' in comparison_data.columns else 'mean_value'
                    
                    plt.bar(comparison_data['round'].astype(str), comparison_data[value_col_for_plot])
                    plt.title(f'Mean {metric_display_name} per Round during {current_phase_display_for_report}')
                    plt.xlabel('Round')
                    
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
    
    plots_dir, tables_dir, advanced_dir, causality_dir, correlation_dir, similarity_dir = setup_output_directories(args.output_dir)
    
    experiment_data_dir = get_experiment_data_dir(args.data_dir)

    if not os.path.isdir(experiment_data_dir):
        print(f"Error: Data directory not found: {experiment_data_dir}")
        print(f"  Input --data-dir: '{args.data_dir}'")
        print(f"  Absolute path checked: '{os.path.abspath(experiment_data_dir)}'")
        return

    print(f"Using data directory: {experiment_data_dir}")

    node_config_to_use = NODE_RESOURCE_CONFIGS.get(DEFAULT_NODE_CONFIG_NAME, NODE_RESOURCE_CONFIGS.get("Default"))
    
    if 'MEMORY_GB' in node_config_to_use and 'MEMORY_BYTES' not in node_config_to_use:
        node_config_to_use = node_config_to_use.copy()
        node_config_to_use['MEMORY_BYTES'] = node_config_to_use['MEMORY_GB'] * (2**30)
        node_config_to_use['DISK_SIZE_BYTES'] = node_config_to_use['DISK_SIZE_GB'] * (2**30)
        node_config_to_use['NETWORK_BANDWIDTH_MBPS'] = max(1000, node_config_to_use['CPUS'] * 250)
    
    print("\nLoading and Processing Experiment Data...")
    
    available_metrics = list_available_metrics(experiment_data_dir)
    metrics_to_load = args.metrics if args.metrics else available_metrics
    
    metric_name_map = {metric: metric for metric in metrics_to_load} 
    
    all_metrics_data = {}
    if metrics_to_load:
        for metric_name in metrics_to_load:
            try:
                data_for_metric = load_experiment_data(experiment_data_dir, [metric_name], specific_metrics_map=metric_name_map)
                all_metrics_data[metric_name] = data_for_metric
            except Exception as e:
                print(f"Error loading data for metric {metric_name}: {e}")
                all_metrics_data[metric_name] = {}

    metric_type_map = {metric: "gauge" for metric in metrics_to_load}

    if not all_metrics_data:
        print("CRITICAL ERROR: No data was loaded. Check the specified data directory, metrics, and rounds.")
        return 

    print("Experiment data loaded successfully.")
            
    experiment_results = {
        'processed_data': all_metrics_data
    }

    metrics_data = experiment_results.get('processed_data', {})

    if not metrics_data:
        print("No processed data (metrics_data) available. Many analyses and plots will be skipped.")

    print("\nNormalizing global experiment time for all metric DataFrames...")
    all_phase_markers = {}
    for metric_name, rounds_data in all_metrics_data.items():
        all_phase_markers[metric_name] = {}
        for round_name, df_round in rounds_data.items():
            if df_round is not None and not df_round.empty:
                df_round = add_experiment_elapsed_time(df_round)
                if not df_round.empty:
                    df_round, phase_markers_round = add_phase_markers(df_round, phase_column='phase', phase_display_names=PHASE_DISPLAY_NAMES)
                    all_phase_markers[metric_name][round_name] = phase_markers_round
                    all_metrics_data[metric_name][round_name] = df_round
            else:
                print(f"DataFrame for metric '{metric_name}', round '{round_name}' is empty or None. Skipping time normalization.")
    print("Global time normalization completed.")

    # Helper function to extract time series data
    def extract_series_from_metrics_data(metrics_data_local, target_metric, target_round, target_tenant, series_name_prefix="Series", min_len=10):
        series = pd.Series(dtype=float) # Default empty series
        series_name_default = f"Empty_{series_name_prefix}_{target_metric}_{target_tenant}_{target_round}"
        series.name = series_name_default

        if not metrics_data_local:
            print(f"Metrics data is empty. Cannot extract {target_metric} for tenant {target_tenant} in round {target_round}.")
            return series

        if target_metric in metrics_data_local and target_round in metrics_data_local[target_metric]:
            df_round = metrics_data_local[target_metric][target_round]
            if df_round is not None and not df_round.empty and 'timestamp' in df_round.columns and 'value' in df_round.columns:
                
                # Handle tenant-specific vs node-level
                if 'tenant' in df_round.columns:
                    tenant_data = df_round[df_round['tenant'] == target_tenant]
                    series_label_name = f"{series_name_prefix}_{target_metric}_{target_tenant}"
                elif target_tenant is None or target_tenant == "NODE_LEVEL": # Node-level metric
                    tenant_data = df_round # Use the whole DataFrame
                    series_label_name = f"{series_name_prefix}_{target_metric}_NODE"
                else: # Tenant column exists, but target_tenant doesn't match, or tenant column missing when specific tenant requested
                    print(f"Tenant '{target_tenant}' not found for metric '{target_metric}' in round '{target_round}', or 'tenant' column missing.")
                    return series

                if not tenant_data.empty:
                    if not pd.api.types.is_datetime64_any_dtype(tenant_data['timestamp']):
                        try:
                            # Create a copy to avoid SettingWithCopyWarning if tenant_data is a slice
                            tenant_data = tenant_data.copy()
                            tenant_data['timestamp'] = pd.to_datetime(tenant_data['timestamp'])
                        except Exception as e:
                            print(f"Warning: Could not convert 'timestamp' to datetime for {target_metric}, {target_tenant or 'NODE'}: {e}")
                            return series

                    processed_series = tenant_data.set_index('timestamp')['value'].sort_index()
                    processed_series = processed_series[~processed_series.index.duplicated(keep='first')]
                    
                    if len(processed_series) >= min_len:
                        series = processed_series
                        series.name = series_label_name
                        print(f"Successfully extracted series: {series.name} with {len(series)} points for round {target_round}.")
                    else:
                        print(f"Extracted series for {series_label_name} in round {target_round} is too short ({len(processed_series)} points, min_len={min_len}).")
                        series.name = f"Short_{series_label_name}" # Keep name for empty/short series
            else:
                print(f"No data or missing required columns for metric '{target_metric}' in round '{target_round}'.")
        else:
            print(f"Metric '{target_metric}' or round '{target_round}' not found in metrics_data.")
        
        if series.empty:
             series.name = series_name_default # Ensure name is set if it remained empty
        return series

    print("\nRunning Advanced Analyses...")
    if not metrics_data:
        print("Skipping advanced analyses as no data (metrics_data) is available.")
    else:
        print("\n--- Preparing Series for Advanced Analysis from Real Data ---")
        
        s_a, s_b, s_c = pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
        # Assign default names for placeholders
        s_a.name, s_b.name, s_c.name = "Placeholder_S_A", "Placeholder_S_B", "Placeholder_S_C"

        first_round_name = None
        available_rounds = set()
        if metrics_data:
            for metric_key in metrics_data:
                if metrics_data[metric_key]:
                    available_rounds.update(metrics_data[metric_key].keys())
        if available_rounds:
            first_round_name = sorted(list(available_rounds))[0] # Pick the first round alphanumerically
            print(f"Using data from round: {first_round_name} for extracting example series.")

        if first_round_name:
            # Try to get diverse metrics and tenants
            potential_metrics = DEFAULT_METRICS_FOR_CAUSALITY[:2] or DEFAULT_METRICS[:2] or list(metrics_data.keys())[:2]
            
            all_tenants_in_round = set()
            for m_key in metrics_data: # Check all metrics for tenants in the chosen round
                if first_round_name in metrics_data[m_key]:
                    df_r = metrics_data[m_key][first_round_name]
                    if df_r is not None and 'tenant' in df_r.columns:
                        all_tenants_in_round.update(df_r['tenant'].unique())
            
            potential_tenants = sorted(list(all_tenants_in_round))
            
            metric1, metric2 = (potential_metrics[0] if len(potential_metrics) > 0 else None), (potential_metrics[1] if len(potential_metrics) > 1 else None)
            tenant1, tenant2 = (potential_tenants[0] if len(potential_tenants) > 0 else None), (potential_tenants[1] if len(potential_tenants) > 1 else None)

            print(f"Attempting to extract series with: metric1={metric1}, metric2={metric2}, tenant1={tenant1}, tenant2={tenant2}")

            if metric1 and tenant1:
                s_a = extract_series_from_metrics_data(metrics_data, metric1, first_round_name, tenant1, "S_A")
            
            if metric2 and tenant1 and metric1 != metric2: # Different metric, same tenant
                s_b = extract_series_from_metrics_data(metrics_data, metric2, first_round_name, tenant1, "S_B")
            elif metric1 and tenant2 and tenant1 != tenant2: # Same metric, different tenant (if s_b not set)
                 if s_b.empty: s_b = extract_series_from_metrics_data(metrics_data, metric1, first_round_name, tenant2, "S_B")

            if metric1 and tenant2 and tenant1 != tenant2: # Metric 1, Tenant 2 for s_c
                s_c = extract_series_from_metrics_data(metrics_data, metric1, first_round_name, tenant2, "S_C")
            elif metric2 and tenant2 and tenant1 != tenant2 and metric1 != metric2: # Metric 2, Tenant 2 for s_c (if s_c not set and metric2 exists)
                 if s_c.empty: s_c = extract_series_from_metrics_data(metrics_data, metric2, first_round_name, tenant2, "S_C_fallback")
            
            # Fallback for s_b, s_c if they are still empty and s_a is not
            if s_b.empty and not s_a.empty and metric2 and tenant1: # Try metric2, tenant1 again if previous logic missed
                 s_b = extract_series_from_metrics_data(metrics_data, metric2, first_round_name, tenant1, "S_B_fallback")
            if s_c.empty and not s_a.empty and metric1 and tenant2: # Try metric1, tenant2 again
                 s_c = extract_series_from_metrics_data(metrics_data, metric1, first_round_name, tenant2, "S_C_fallback")

        # Ensure s_a, s_b, s_c are pd.Series, even if empty, and have names.
        # If any series is still empty, use a placeholder.
        # Min length for general display/simple calcs. Granger etc. have own higher limits.
        min_display_len = 5 
        if s_a.empty or len(s_a) < min_display_len:
            print(f"Series S_A ({s_a.name}) is empty or too short after extraction attempt. Using placeholder.")
            s_a = pd.Series(np.random.randn(50), index=pd.date_range(start='2023-01-01', periods=50, freq='H'), name='Placeholder_S_A')
        if s_b.empty or len(s_b) < min_display_len:
            print(f"Series S_B ({s_b.name}) is empty or too short after extraction attempt. Using placeholder.")
            s_b = pd.Series(np.random.randn(50), index=pd.date_range(start='2023-01-01', periods=50, freq='H'), name='Placeholder_S_B')
        if s_c.empty or len(s_c) < min_display_len:
            print(f"Series S_C ({s_c.name}) is empty or too short after extraction attempt. Using placeholder.")
            s_c = pd.Series(np.random.randn(50), index=pd.date_range(start='2023-01-01', periods=50, freq='H'), name='Placeholder_S_C')
        
        # Align valid series if they have DatetimeIndex
        series_to_align = [s for s in [s_a, s_b, s_c] if not s.empty and isinstance(s.index, pd.DatetimeIndex)]
        if len(series_to_align) >= 2:
            print("Aligning extracted series using inner join on timestamps...")
            aligned_df = None
            if len(series_to_align) == 2:
                aligned_df = pd.merge(series_to_align[0].rename('s1'), series_to_align[1].rename('s2'), left_index=True, right_index=True, how='inner')
                if not aligned_df.empty:
                    series_to_align[0] = aligned_df['s1']; series_to_align[0].name = s_a.name # Keep original intended name
                    series_to_align[1] = aligned_df['s2']; series_to_align[1].name = s_b.name
            elif len(series_to_align) == 3:
                df1 = series_to_align[0].rename('s1'); df2 = series_to_align[1].rename('s2'); df3 = series_to_align[2].rename('s3')
                aligned_df = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
                aligned_df = pd.merge(aligned_df, df3, left_index=True, right_index=True, how='inner')
                if not aligned_df.empty:
                    series_to_align[0] = aligned_df['s1']; series_to_align[0].name = s_a.name
                    series_to_align[1] = aligned_df['s2']; series_to_align[1].name = s_b.name
                    series_to_align[2] = aligned_df['s3']; series_to_align[2].name = s_c.name
            
            # Re-assign s_a, s_b, s_c if they were part of alignment and df is not empty
            if aligned_df is not None and not aligned_df.empty:
                if len(series_to_align) > 0 and not series_to_align[0].empty: s_a = series_to_align[0]
                if len(series_to_align) > 1 and not series_to_align[1].empty: s_b = series_to_align[1]
                if len(series_to_align) > 2 and not series_to_align[2].empty: s_c = series_to_align[2]
                print("Series aligned. Re-checking lengths.")
                if s_a.empty or len(s_a) < min_display_len: s_a = pd.Series(np.random.randn(50), index=pd.date_range(start='2023-01-01', periods=50, freq='H'), name='Placeholder_S_A_after_align')
                if s_b.empty or len(s_b) < min_display_len: s_b = pd.Series(np.random.randn(50), index=pd.date_range(start='2023-01-01', periods=50, freq='H'), name='Placeholder_S_B_after_align')
                if s_c.empty or len(s_c) < min_display_len: s_c = pd.Series(np.random.randn(50), index=pd.date_range(start='2023-01-01', periods=50, freq='H'), name='Placeholder_S_C_after_align')

            elif aligned_df is not None and aligned_df.empty:
                print("Inner join for alignment resulted in empty series. Using pre-alignment or placeholders.")

        series_list_for_analysis = [s for s in [s_a, s_b, s_c] if not s.empty]
        print(f"Final series for analysis: {[s.name for s in series_list_for_analysis if hasattr(s, 'name')]}")
        if not series_list_for_analysis:
             print("CRITICAL: No valid series (real or placeholder) available for advanced analysis. Most analyses will be skipped.")
             s_a = pd.Series(np.random.randn(50), index=pd.date_range(start='2023-01-01', periods=50, freq='H'), name='Emergency_Placeholder_S_A')
             s_b = pd.Series(np.random.randn(50), index=pd.date_range(start='2023-01-01', periods=50, freq='H'), name='Emergency_Placeholder_S_B')
             s_c = pd.Series(np.random.randn(50), index=pd.date_range(start='2023-01-01', periods=50, freq='H'), name='Emergency_Placeholder_S_C')
             series_list_for_analysis = [s_a, s_b, s_c]

        print("\n--- Running Correlation Analysis ---")
        if not s_a.empty and not s_b.empty:
            pearson_corr, pearson_p_value = calculate_pearson_correlation(s_a, s_b)
            print(f"Pearson Correlation between {s_a.name} and {s_b.name}: {pearson_corr:.4f} (p-value: {pearson_p_value:.4f})")
        else:
            print(f"Skipping Pearson correlation between {s_a.name} and {s_b.name} due to empty series.")

        if not s_a.empty and not s_c.empty:
            spearman_corr, spearman_p_value = calculate_spearman_correlation(s_a, s_c)
            print(f"Spearman Correlation between {s_a.name} and {s_c.name}: {spearman_corr:.4f} (p-value: {spearman_p_value:.4f})")
        else:
            print(f"Skipping Spearman correlation between {s_a.name} and {s_c.name} due to empty series.")
        
        valid_series_for_heatmap = [s for s in [s_a, s_b, s_c] if not s.empty and len(s) > 1]
        if len(valid_series_for_heatmap) >= 2:
            df_for_heatmap = pd.concat(valid_series_for_heatmap, axis=1).dropna()
            df_for_heatmap.columns = [s.name for s in valid_series_for_heatmap][:len(df_for_heatmap.columns)]

            if not df_for_heatmap.empty and len(df_for_heatmap) > 1 and len(df_for_heatmap.columns) > 1:
                actual_corr_df = df_for_heatmap.corr(method='pearson')
                plot_correlation_heatmap(actual_corr_df, title=f"Correlation Heatmap ({', '.join(df_for_heatmap.columns)})",
                                         output_filename=os.path.join(correlation_dir, "actual_correlation_heatmap.png"))
                print(f"Actual correlation heatmap saved to {os.path.join(correlation_dir, 'actual_correlation_heatmap.png')}")
            else:
                print("Not enough data/series for actual correlation heatmap after processing. Using dummy heatmap.")
                dummy_corr_data = {s_a.name: [1.0, 0.5, 0.2], s_b.name: [0.5, 1.0, 0.3], s_c.name: [0.2, 0.3, 1.0]}
                dummy_corr_df = pd.DataFrame(dummy_corr_data, index=[s_a.name, s_b.name, s_c.name])
                plot_correlation_heatmap(dummy_corr_df, title="Dummy Correlation Heatmap", 
                                         output_filename=os.path.join(correlation_dir, "dummy_correlation_heatmap.png"))
        else:
            print("Not enough valid series for correlation heatmap. Using dummy heatmap.")
            dummy_corr_data = {s_a.name: [1.0, 0.5, 0.2], s_b.name: [0.5, 1.0, 0.3], s_c.name: [0.2, 0.3, 1.0]}
            dummy_corr_df = pd.DataFrame(dummy_corr_data, index=[s_a.name, s_b.name, s_c.name])
            plot_correlation_heatmap(dummy_corr_df, title="Dummy Correlation Heatmap", 
                                     output_filename=os.path.join(correlation_dir, "dummy_correlation_heatmap.png"))

        print("\n--- Running Similarity Analysis ---")
        if not s_a.empty and not s_b.empty:
            dtw_dist = calculate_dtw_distance(s_a, s_b)
            print(f"DTW Distance between {s_a.name} and {s_b.name}: {dtw_dist:.4f}")
        else:
            print(f"Skipping DTW distance between {s_a.name} and {s_b.name} due to empty series.")
        
        if not s_a.empty and not s_c.empty:
            cosine_sim = calculate_cosine_similarity(s_a, s_c)
            print(f"Cosine Similarity between {s_a.name} and {s_c.name}: {cosine_sim:.4f}")

            plot_time_series_with_cosine_similarity(
                s_a, s_c, cosine_sim,
                title=f"Time Series Comparison: {s_a.name} vs {s_c.name}",
                output_filename=os.path.join(similarity_dir, "time_series_cosine_sim.png")
            )
            print(f"Time series with cosine similarity plot saved to {os.path.join(similarity_dir, 'time_series_cosine_sim.png')}")
        else:
            print(f"Skipping Cosine similarity between {s_a.name} and {s_c.name} and related plot due to empty series.")
            cosine_sim = np.nan

        if len(valid_series_for_heatmap) >= 2:
            df_for_sim_heatmap = pd.concat(valid_series_for_heatmap, axis=1).dropna()
            df_for_sim_heatmap.columns = [s.name for s in valid_series_for_heatmap][:len(df_for_sim_heatmap.columns)]

            if not df_for_sim_heatmap.empty and len(df_for_sim_heatmap.columns) >=2:
                num_series_sim = len(df_for_sim_heatmap.columns)
                sim_matrix = np.ones((num_series_sim, num_series_sim))
                series_values_list = [df_for_sim_heatmap[col].values for col in df_for_sim_heatmap.columns]

                for i in range(num_series_sim):
                    for j in range(i + 1, num_series_sim):
                        sim = calculate_cosine_similarity(pd.Series(series_values_list[i]), pd.Series(series_values_list[j]))
                        sim_matrix[i, j] = sim
                        sim_matrix[j, i] = sim
                
                actual_cosine_sim_df = pd.DataFrame(sim_matrix, index=df_for_sim_heatmap.columns, columns=df_for_sim_heatmap.columns)
                plot_cosine_similarity_heatmap(
                    actual_cosine_sim_df, 
                    title=f"Cosine Similarity Heatmap ({', '.join(df_for_sim_heatmap.columns)})",
                    output_filename=os.path.join(similarity_dir, "actual_cosine_similarity_heatmap.png")
                )
                print(f"Actual cosine similarity heatmap saved to {os.path.join(similarity_dir, 'actual_cosine_similarity_heatmap.png')}")
            else:
                print("Not enough data/columns for actual cosine similarity heatmap. Using dummy heatmap.")
                dummy_cosine_sim_data = {s_a.name: [1.0, cosine_sim if not np.isnan(cosine_sim) else 0.5, 0.2], s_c.name: [cosine_sim if not np.isnan(cosine_sim) else 0.5, 1.0, 0.4], s_b.name: [0.2, 0.4, 1.0]}
                dummy_cosine_sim_df = pd.DataFrame(dummy_cosine_sim_data, index=[s_a.name, s_c.name, s_b.name])
                plot_cosine_similarity_heatmap(dummy_cosine_sim_df, title="Dummy Cosine Similarity Heatmap", output_filename=os.path.join(similarity_dir, "dummy_cosine_similarity_heatmap.png"))
        else:
            print("Not enough valid series for cosine similarity heatmap. Using dummy heatmap.")
            dummy_cosine_sim_data = {s_a.name: [1.0, cosine_sim if not np.isnan(cosine_sim) else 0.5, 0.2], s_c.name: [cosine_sim if not np.isnan(cosine_sim) else 0.5, 1.0, 0.4], s_b.name: [0.2, 0.4, 1.0]}
            dummy_cosine_sim_df = pd.DataFrame(dummy_cosine_sim_data, index=[s_a.name, s_c.name, s_b.name])
            plot_cosine_similarity_heatmap(dummy_cosine_sim_df, title="Dummy Cosine Similarity Heatmap", output_filename=os.path.join(similarity_dir, "dummy_cosine_similarity_heatmap.png"))

        print("\n--- Time-Varying Cosine Similarity ---")
        tv_cosine_sim_df = pd.DataFrame(columns=['timestamp', 'cosine_similarity'])
        if not s_a.empty and not s_c.empty and \
           isinstance(s_a.index, pd.DatetimeIndex) and isinstance(s_c.index, pd.DatetimeIndex) and \
           not s_a.index.empty and not s_c.index.empty:
            
            aligned_df_tv = pd.concat([s_a.rename('s_a_tv'), s_c.rename('s_c_tv')], axis=1, join='inner')

            if not aligned_df_tv.empty and len(aligned_df_tv) >= 10:
                s_a_aligned_tv = aligned_df_tv['s_a_tv']
                s_c_aligned_tv = aligned_df_tv['s_c_tv']
                
                print(f"Calculating real time-varying cosine similarity between {s_a.name} and {s_c.name} using config-defined window/step.")
                tv_cosine_sim_df = calculate_time_varying_cosine_similarity(
                    s_a_aligned_tv, 
                    s_c_aligned_tv
                )
            else:
                print(f"Not enough overlapping data between {s_a.name} and {s_c.name} for time-varying cosine similarity (need at least 10 points after alignment). Using dummy data for plot.")
                timestamps_tv = pd.date_range(start='2023-01-01', periods=10, freq='D')
                tv_cosine_sim_df = pd.DataFrame({
                    'timestamp': timestamps_tv,
                    'cosine_similarity': np.random.rand(10) * 2 - 1
                })
        else:
            print(f"Series {s_a.name} or {s_c.name} are not suitable for real time-varying cosine similarity (empty, no DatetimeIndex, or not enough data). Using dummy data for plot.")
            timestamps_tv = pd.date_range(start='2023-01-01', periods=10, freq='D')
            tv_cosine_sim_df = pd.DataFrame({
                'timestamp': timestamps_tv,
                'cosine_similarity': np.random.rand(10) * 2 - 1
            })

        if tv_cosine_sim_df.empty:
            print("Real time-varying cosine similarity calculation resulted in empty DataFrame. Using dummy data for plot.")
            timestamps_tv = pd.date_range(start='2023-01-01', periods=10, freq='D')
            tv_cosine_sim_df = pd.DataFrame({
                'timestamp': timestamps_tv,
                'cosine_similarity': np.random.rand(10) * 2 - 1
            })

        phase_start_for_plot = None
        if not tv_cosine_sim_df.empty and 'timestamp' in tv_cosine_sim_df.columns and not tv_cosine_sim_df['timestamp'].empty:
            phase_start_for_plot = tv_cosine_sim_df['timestamp'].min()
        elif not timestamps_tv.empty:
             phase_start_for_plot = timestamps_tv.min()

        plot_time_varying_cosine_similarity(
            tv_cosine_sim_df, 
            series_a_name=s_a.name,
            series_b_name=s_c.name,
            output_filename=os.path.join(similarity_dir, "time_varying_cosine_sim.png"),
            phase_start_time=phase_start_for_plot
        )
        print(f"Time-varying cosine similarity plot saved to {os.path.join(similarity_dir, 'time_varying_cosine_sim.png')}")

        print("\n--- Running Causality Analysis ---")
        print("\n--- Preparing Data for Granger Causality Test from Real Data ---")
        actual_granger_input_long_df = pd.DataFrame()
        granger_input_dfs_list = []
        
        metrics_for_granger = DEFAULT_METRICS_FOR_CAUSALITY[:] 
        if not metrics_for_granger: metrics_for_granger = DEFAULT_METRICS[:2]
        if not metrics_for_granger and metrics_data: metrics_for_granger = list(metrics_data.keys())[:2]

        noisy_tenant_for_granger_default = DEFAULT_NOISY_TENANT
        noisy_tenant_for_granger = None
        other_tenants_for_granger = []

        if first_round_name and metrics_data and metrics_for_granger:
            all_tenants_in_round_for_granger = set()
            for metric_name_granger in metrics_for_granger:
                if metric_name_granger in metrics_data and first_round_name in metrics_data[metric_name_granger]:
                    df_round_granger = metrics_data[metric_name_granger][first_round_name]
                    if df_round_granger is not None and 'tenant' in df_round_granger.columns:
                        all_tenants_in_round_for_granger.update(df_round_granger['tenant'].unique())
            
            sorted_tenants_for_granger = sorted(list(all_tenants_in_round_for_granger))

            if noisy_tenant_for_granger_default and noisy_tenant_for_granger_default in sorted_tenants_for_granger:
                noisy_tenant_for_granger = noisy_tenant_for_granger_default
            elif sorted_tenants_for_granger:
                noisy_tenant_for_granger = sorted_tenants_for_granger[0]
                print(f"Default noisy tenant '{noisy_tenant_for_granger_default}' not found or not specified for Granger; using first available: {noisy_tenant_for_granger}")
            
            if noisy_tenant_for_granger:
                other_tenants_for_granger = [t for t in sorted_tenants_for_granger if t != noisy_tenant_for_granger]
            
            if not other_tenants_for_granger and len(sorted_tenants_for_granger) > 1 and noisy_tenant_for_granger:
                 other_tenants_for_granger = [t for t in sorted_tenants_for_granger if t != noisy_tenant_for_granger][:1]

            if noisy_tenant_for_granger and other_tenants_for_granger:
                tenants_for_granger_analysis = [noisy_tenant_for_granger] + other_tenants_for_granger
                print(f"Granger analysis tenants: Noisy={noisy_tenant_for_granger}, Others={other_tenants_for_granger}, Metrics: {metrics_for_granger}")

                for metric_name_granger in metrics_for_granger:
                    if metric_name_granger in metrics_data and first_round_name in metrics_data[metric_name_granger]:
                        df_round_metric_granger = metrics_data[metric_name_granger][first_round_name]
                        if df_round_metric_granger is not None and not df_round_metric_granger.empty:
                            for tenant_name_granger in tenants_for_granger_analysis:
                                if 'tenant' not in df_round_metric_granger.columns:
                                    if tenant_name_granger is None or tenant_name_granger == "NODE_LEVEL":
                                        tenant_df_granger = df_round_metric_granger
                                    else: continue
                                else:
                                    tenant_df_granger = df_round_metric_granger[df_round_metric_granger['tenant'] == tenant_name_granger]

                                if not tenant_df_granger.empty and 'timestamp' in tenant_df_granger.columns and 'value' in tenant_df_granger.columns:
                                    current_phase_granger = tenant_df_granger['phase'].iloc[0] if 'phase' in tenant_df_granger.columns and not tenant_df_granger.empty else 'unknown_phase'
                                    
                                    granger_series_df = tenant_df_granger[['timestamp', 'value']].copy()
                                    granger_series_df.rename(columns={'timestamp': 'datetime'}, inplace=True)
                                    granger_series_df['tenant'] = tenant_name_granger if 'tenant' in df_round_metric_granger.columns else "NODE_LEVEL"
                                    granger_series_df['metric_name'] = metric_name_granger
                                    granger_series_df['phase'] = current_phase_granger
                                    granger_series_df['round'] = first_round_name
                                    granger_input_dfs_list.append(granger_series_df)
            else:
                print("Not enough distinct tenants or metrics found for Granger causality with real data based on config and available data.")
        
        if granger_input_dfs_list:
            actual_granger_input_long_df = pd.concat(granger_input_dfs_list).reset_index(drop=True)
            actual_granger_input_long_df.dropna(subset=['value'], inplace=True)
            print(f"Prepared actual_granger_input_long_df with {len(actual_granger_input_long_df)} rows for Granger causality.")
        
        if actual_granger_input_long_df.empty or \
           len(actual_granger_input_long_df['metric_name'].unique()) == 0 or \
           len(actual_granger_input_long_df['tenant'].unique()) < 2:
            
            print("Failed to prepare sufficient real data for Granger causality. Constructing fallback Granger input from s_a, s_b, s_c.")
            
            mock_data_granger_list_fallback = []
            series_for_fallback_granger = [s for s in [s_a, s_b, s_c] if s is not None and not s.empty and len(s) >= DEFAULT_GRANGER_MIN_OBSERVATIONS]

            if len(series_for_fallback_granger) >= 2:
                temp_tenants_fallback = []
                temp_metrics_fallback = []
                for i, s_item_fallback in enumerate(series_for_fallback_granger):
                    temp_df_fallback = s_item_fallback.reset_index()
                    temp_df_fallback.columns = ['datetime', 'value']
                    
                    name_parts_fallback = s_item_fallback.name.split('_')
                    tenant_name_fallback = name_parts_fallback[-1] if "tenant" in name_parts_fallback[-1] else f"fallbacktenant{i+1}"
                    metric_name_fallback = "_".join(name_parts_fallback[2:-1]) if len(name_parts_fallback) > 3 else (name_parts_fallback[1] if len(name_parts_fallback) > 1 else f"fallbackmetric{i+1}")
                    if not metric_name_fallback : metric_name_fallback = f"fallbackmetric{i+1}"

                    temp_df_fallback['tenant'] = tenant_name_fallback
                    temp_df_fallback['metric_name'] = metric_name_fallback
                    temp_df_fallback['phase'] = 'fallback_phase'
                    temp_df_fallback['round'] = first_round_name or 'fallback_round'
                    mock_data_granger_list_fallback.append(temp_df_fallback)
                    temp_tenants_fallback.append(tenant_name_fallback)
                    temp_metrics_fallback.append(metric_name_fallback)

                if mock_data_granger_list_fallback:
                    actual_granger_input_long_df = pd.concat(mock_data_granger_list_fallback).reset_index(drop=True)
                    actual_granger_input_long_df.dropna(subset=['value'], inplace=True)
                    print(f"Using fallback Granger input data from s_a,s_b,s_c: {len(actual_granger_input_long_df)} rows.")
                    unique_fallback_tenants = sorted(list(set(temp_tenants_fallback)))
                    if unique_fallback_tenants:
                        noisy_tenant_for_granger = unique_fallback_tenants[0]
                        other_tenants_for_granger = unique_fallback_tenants[1:]
                    metrics_for_granger = sorted(list(set(temp_metrics_fallback)))
            else:
                print("Not enough valid series (s_a, s_b, s_c) for fallback Granger input.")
                actual_granger_input_long_df = pd.DataFrame()

        granger_results_df = pd.DataFrame()
        if not actual_granger_input_long_df.empty and \
           metrics_for_granger and \
           noisy_tenant_for_granger and \
           len(actual_granger_input_long_df['tenant'].unique()) >= (1 if not other_tenants_for_granger else 2):
            
            print(f"Running Granger Causality Test with: Metrics={metrics_for_granger}, NoisyTenant={noisy_tenant_for_granger}, OtherTenants={other_tenants_for_granger}")
            granger_results_df = perform_granger_causality_test(
                data=actual_granger_input_long_df, 
                metrics_for_causality=metrics_for_granger, 
                noisy_tenant=noisy_tenant_for_granger, 
                other_tenants=other_tenants_for_granger,
                max_lag=GRANGER_CAUSALITY_DEFAULTS.get('max_lag', 5),
                test=GRANGER_CAUSALITY_DEFAULTS.get('test', 'ssr_chi2test'),
                significance_level=DEFAULT_CAUSALITY_THRESHOLD_P_VALUE,
                min_observations=DEFAULT_GRANGER_MIN_OBSERVATIONS,
                verbose=True
            )
        else:
            print("Skipping Granger causality test as not enough valid data or configuration could be prepared.")

        if not granger_results_df.empty:
            print("Granger Causality Results:")
            print(granger_results_df.head())
            visualize_causal_graph(granger_results_df, 
                                   output_path=os.path.join(causality_dir, "granger_causal_graph.png"),
                                   title="Example Granger Causal Graph")
            print(f"Granger causal graph saved to {os.path.join(causality_dir, 'granger_causal_graph.png')}")
        else:
            print("Granger causality analysis did not produce results or encountered an issue.")

        te_s1_to_s2 = calculate_transfer_entropy(s_a.values, s_b.values, k=1)
        te_s2_to_s1 = calculate_transfer_entropy(s_b.values, s_a.values, k=1)
        print(f"Transfer Entropy ({s_a.name} -> {s_b.name}): {te_s1_to_s2:.4f}")
        print(f"Transfer Entropy ({s_b.name} -> {s_a.name}): {te_s2_to_s1:.4f}")

        ccm_scores = calculate_convergent_cross_mapping(s_a, s_c, embed_dim=2, tau=1)
        if ccm_scores:
            print(f"CCM Score ({s_a.name} xmaps {s_c.name}): {ccm_scores['ccm_s1_xmaps_s2']:.4f}")
            print(f"CCM Score ({s_c.name} xmaps {s_a.name}): {ccm_scores['ccm_s2_xmaps_s1']:.4f}")
        else:
            print(f"CCM analysis for {s_a.name} and {s_c.name} did not produce results.")

    print("\nPipeline execution finished successfully!")
    
    return experiment_results


if __name__ == "__main__":
    main()
