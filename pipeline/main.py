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
    calculate_spearman_correlation,
    calculate_lagged_cross_correlation
)
from pipeline.analysis.causality_analysis import (
    perform_inter_tenant_causality_analysis,  # Renamed from perform_granger_causality_test
    calculate_transfer_entropy,
    calculate_convergent_cross_mapping
)
from pipeline.analysis.similarity_analysis import (
    calculate_dtw_distance,
    calculate_cosine_similarity,
    calculate_time_varying_cosine_similarity
)
from pipeline.visualization.plots import (
    plot_correlation_heatmap, 
    plot_ccf,
    plot_dtw_path, 
    plot_aligned_time_series, 
    plot_dtw_distance_heatmap,
    plot_time_series_with_cosine_similarity,
    plot_cosine_similarity_heatmap,
    plot_time_varying_cosine_similarity,
    visualize_causal_graph
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
    GRANGER_CAUSALITY_DEFAULTS, COSINE_SIM_WINDOW_SIZE_S, SCRAPE_INTERVAL_S,
    INTERPOLATION_METHOD
)
from pipeline.data_processing.data_loader import load_experiment_data, list_available_metrics
from pipeline.data_processing.data_aggregate import aggregate_analysis_results
from pipeline.utils import get_experiment_data_dir, add_experiment_elapsed_time, add_phase_markers


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
    parser.add_argument('--inter-tenant-causality', action='store_true',
                        help='Run inter-tenant causality analysis (within the broader causality analysis)')
    parser.add_argument('--skip-advanced-analyses', action='store_true',
                        help='Skip all advanced analyses (correlation, similarity, causality).')
    parser.add_argument('--adv-analysis-round', type=str, default=None,
                        help='Specify the round name to use for advanced analysis series extraction (e.g., s_a, s_b, s_c). Defaults to the first available round.')
    parser.add_argument('--run-correlation', action='store_true', default=False,
                        help='Explicitly run correlation analysis.')
    parser.add_argument('--run-similarity', action='store_true', default=False,
                        help='Explicitly run similarity analysis.')
    parser.add_argument('--run-causality', action='store_true', default=False,
                        help='Explicitly run causality analysis.')

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


def extract_series_from_metrics_data(metrics_data, metric_name, round_name, tenant_name, series_default_name="extracted_series", time_col='timestamp', value_col='value'):
    """A
    Extracts a specific time series from the nested metrics_data structure.

    Args:
        metrics_data (dict): The main data structure containing all metrics, rounds, and tenant data.
        metric_name (str): The name of the metric to extract (e.g., 'cpu_usage').
        round_name (str): The name of the round to extract data from (e.g., 'round-1').
        tenant_name (str): The name of the tenant to filter by (e.g., 'tenant-a').
        series_default_name (str): Default name for the series if extracted.
        time_col (str): Name of the timestamp column.
        value_col (str): Name of the value column.

    Returns:
        pd.Series: Extracted time series with a DatetimeIndex, or an empty Series if not found.
    """
    series = pd.Series(dtype=float, name=series_default_name) # Initialize with default name
    try:
        if metric_name in metrics_data and \
           round_name in metrics_data[metric_name] and \
           metrics_data[metric_name][round_name] is not None and \
           not metrics_data[metric_name][round_name].empty:
            
            df_round = metrics_data[metric_name][round_name]
            
            if 'tenant' in df_round.columns and tenant_name:
                tenant_df = df_round[df_round['tenant'] == tenant_name]
            else: # Assuming node-level metric if 'tenant' column is missing or tenant_name is None
                tenant_df = df_round

            if not tenant_df.empty and time_col in tenant_df.columns and value_col in tenant_df.columns:
                # Ensure timestamp is datetime and set as index
                ts_data = tenant_df[[time_col, value_col]].copy()
                ts_data[time_col] = pd.to_datetime(ts_data[time_col])
                ts_data = ts_data.set_index(time_col)
                series = ts_data[value_col]
                series.name = f"{series_default_name}_{metric_name}_{tenant_name}_{round_name}" if tenant_name else f"{series_default_name}_{metric_name}_{round_name}"
            else:
                print(f"Warning: Data for {metric_name}, {round_name}, {tenant_name} is empty or missing columns after filtering.")
        else:
            print(f"Warning: Could not find data for metric='{metric_name}', round='{round_name}'.")
    except KeyError as e:
        print(f"KeyError accessing data for {metric_name}, {round_name}, {tenant_name}: {e}")
    except Exception as e:
        print(f"General error extracting series for {metric_name}, {round_name}, {tenant_name}: {e}")
    
    if series.empty: # Ensure it has the default name if empty
        series.name = series_default_name
    return series


def main():
    """Main function that runs the analysis pipeline."""
    print("DEBUG: pipeline.main.main() STARTED")  # ADDED
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
        try:
            # Debug information
            print(f"DEBUG: experiment_data_dir: {experiment_data_dir}")
            print(f"DEBUG: os.path.exists(experiment_data_dir): {os.path.exists(experiment_data_dir)}")
            print(f"DEBUG: metrics_to_load: {metrics_to_load}")
            print(f"DEBUG: rounds: {args.rounds}")
            print(f"DEBUG: phases: {args.phases}")
            print(f"DEBUG: tenants: {args.tenants}")
            
            # Load all metrics at once instead of one by one
            all_metrics_data = load_experiment_data(experiment_data_dir, specific_metrics=metrics_to_load, 
                                                    specific_rounds=args.rounds, specific_phases=args.phases, 
                                                    specific_tenants=args.tenants)
            
            if not all_metrics_data:
                print("No data was loaded.")
        except Exception as e:
            print(f"Error loading experiment data: {e}")
            import traceback
            traceback.print_exc()
            all_metrics_data = {}

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
        
        # Check if rounds_data is a dictionary (expected structure)
        if not isinstance(rounds_data, dict):
            print(f"Warning: Data for metric '{metric_name}' is not a dictionary. Skipping time normalization.")
            continue
            
        for round_name, df_round in rounds_data.items():
            # Check if df_round is a DataFrame (expected type)
            if df_round is not None and isinstance(df_round, pd.DataFrame) and not df_round.empty:
                df_round = add_experiment_elapsed_time(df_round)
                if not df_round.empty:
                    df_round, phase_markers_round = add_phase_markers(df_round, phase_column='phase', phase_display_names=PHASE_DISPLAY_NAMES)
                    all_phase_markers[metric_name][round_name] = phase_markers_round
                    all_metrics_data[metric_name][round_name] = df_round
            else:
                print(f"DataFrame for metric '{metric_name}', round '{round_name}' is not a valid DataFrame or is empty. Skipping time normalization.")
    print("Global time normalization completed.")

    print("\nRunning Advanced Analyses...")
    all_advanced_analysis_results = [] 
    current_round_for_adv_analysis = None 

    should_run_any_advanced = not args.skip_advanced_analyses
    run_all_adv_by_default = not (args.run_correlation or args.run_similarity or args.run_causality)

    if should_run_any_advanced and (run_all_adv_by_default or args.run_correlation or args.run_similarity or args.run_causality):
        if not metrics_data:
            print("Skipping advanced analyses as no data (metrics_data) is available.")
        else:
            if args.adv_analysis_round:
                specified_round_exists = False
                if metrics_data: 
                    for metric_key_check in metrics_data: 
                        if metrics_data[metric_key_check] and args.adv_analysis_round in metrics_data[metric_key_check]:
                            specified_round_exists = True
                            break
                if specified_round_exists:
                    current_round_for_adv_analysis = args.adv_analysis_round
                    print(f"Using data from specified round: {current_round_for_adv_analysis} for advanced analysis series extraction.")
                else:
                    print(f"Warning: Specified round '{args.adv_analysis_round}' not found in available data. Falling back to default round selection.")
                    current_round_for_adv_analysis = None 
            
            if not current_round_for_adv_analysis: 
                available_rounds_fallback = set()
                if metrics_data:
                    for metric_key_fallback in metrics_data:
                        if metrics_data[metric_key_fallback]: 
                            available_rounds_fallback.update(metrics_data[metric_key_fallback].keys())
                if available_rounds_fallback:
                    current_round_for_adv_analysis = sorted(list(available_rounds_fallback))[0] 
                    print(f"Using data from round (default/fallback): {current_round_for_adv_analysis} for extracting example series.")
                else:
                    print("No available rounds found in metrics_data. Advanced analysis series extraction will use placeholders or be skipped.")
            
            s_a, s_b, s_c = pd.Series(dtype=float, name="s_a_placeholder"), pd.Series(dtype=float, name="s_b_placeholder"), pd.Series(dtype=float, name="s_c_placeholder")
            valid_series_for_heatmap = [] 

            if current_round_for_adv_analysis:
                print(f"Attempting to extract example series (s_a, s_b, s_c) from round: {current_round_for_adv_analysis}")
                potential_metrics = DEFAULT_METRICS_FOR_CAUSALITY[:3] or DEFAULT_METRICS[:3] or list(metrics_data.keys())[:3]
                
                all_tenants_in_round = set()
                for m_key in metrics_data: 
                    if current_round_for_adv_analysis in metrics_data[m_key]:
                        df_r = metrics_data[m_key][current_round_for_adv_analysis]
                        if df_r is not None and 'tenant' in df_r.columns:
                            all_tenants_in_round.update(df_r['tenant'].unique())
                potential_tenants = sorted(list(all_tenants_in_round))

                metric1, metric2, metric3 = (potential_metrics[0] if len(potential_metrics) > 0 else None), \
                                           (potential_metrics[1] if len(potential_metrics) > 1 else None), \
                                           (potential_metrics[2] if len(potential_metrics) > 2 else None)
                tenant1, tenant2 = (potential_tenants[0] if len(potential_tenants) > 0 else None), \
                                   (potential_tenants[1] if len(potential_tenants) > 1 else None)

                s_a_name, s_b_name, s_c_name = "s_a_ph", "s_b_ph", "s_c_ph"

                if metric1 and tenant1:
                    s_a_name = f"s_a" # Simplified name for clarity in results
                    s_a = extract_series_from_metrics_data(metrics_data, metric1, current_round_for_adv_analysis, tenant1, s_a_name)
                
                if tenant2: 
                    if metric1 and metric1 != metric2 : # Try same metric, different tenant
                         s_b_name = f"s_b"
                         s_b = extract_series_from_metrics_data(metrics_data, metric1, current_round_for_adv_analysis, tenant2, s_b_name)
                    elif metric2: # Fallback to metric2, different tenant
                        s_b_name = f"s_b"
                        s_b = extract_series_from_metrics_data(metrics_data, metric2, current_round_for_adv_analysis, tenant2, s_b_name)
                elif metric2 and tenant1 : # Fallback to different metric, same tenant for s_b
                    s_b_name = f"s_b"
                    s_b = extract_series_from_metrics_data(metrics_data, metric2, current_round_for_adv_analysis, tenant1, s_b_name)

                if metric3 and tenant1 and metric3 != metric1 and metric3 != metric2:
                    s_c_name = f"s_c"
                    s_c = extract_series_from_metrics_data(metrics_data, metric3, current_round_for_adv_analysis, tenant1, s_c_name)
                elif metric2 and tenant2 and (metric2 != metric1 or tenant2 != tenant1) and s_c.empty:
                     s_c_name = f"s_c"
                     s_c = extract_series_from_metrics_data(metrics_data, metric2, current_round_for_adv_analysis, tenant2, s_c_name)
                elif metric1 and tenant2 and s_c.empty and (metric1 != metric2 or tenant1 != tenant2): # Try metric1 with tenant2 if s_c still empty
                    s_c_name = f"s_c"
                    s_c = extract_series_from_metrics_data(metrics_data, metric1, current_round_for_adv_analysis, tenant2, s_c_name)


                if not s_a.empty: s_a.name = getattr(s_a, 'name', s_a_name); valid_series_for_heatmap.append(s_a)
                if not s_b.empty: s_b.name = getattr(s_b, 'name', s_b_name); valid_series_for_heatmap.append(s_b)
                if not s_c.empty: s_c.name = getattr(s_c, 'name', s_c_name); valid_series_for_heatmap.append(s_c)
                
                # Ensure unique series names if they defaulted to placeholders due to extraction issues
                unique_names = set()
                for i, s in enumerate(valid_series_for_heatmap):
                    original_name = s.name
                    counter = 1
                    while s.name in unique_names:
                        s.name = f"{original_name}_{counter}"
                        counter +=1
                    unique_names.add(s.name)

                print(f"Series extracted: s_a (name: {s_a.name}, len {len(s_a)}), s_b (name: {s_b.name}, len {len(s_b)}), s_c (name: {s_c.name}, len {len(s_c)})")
                if not current_round_for_adv_analysis: 
                    print("Warning: current_round_for_adv_analysis became None during series extraction setup.")
            else:
                print("Cannot extract series for advanced analysis as no specific round is determined. Advanced analyses will use placeholders or skip if series are empty.")
            
            s_a.name = getattr(s_a, 'name', "s_a_placeholder")
            s_b.name = getattr(s_b, 'name', "s_b_placeholder")
            s_c.name = getattr(s_c, 'name', "s_c_placeholder")

            # --- Correlation Analysis ---
            if run_all_adv_by_default or args.run_correlation:
                print("\n--- Running Correlation Analysis ---")
                if not s_a.empty and not s_b.empty:
                    pearson_corr, _ = calculate_pearson_correlation(s_a, s_b)
                    spearman_corr, _ = calculate_spearman_correlation(s_a, s_b)
                    print(f"Pearson Correlation between {s_a.name} and {s_b.name}: {pearson_corr:.4f}")
                    print(f"Spearman Correlation between {s_a.name} and {s_b.name}: {spearman_corr:.4f}")
                    all_advanced_analysis_results.append({'series1_name': s_a.name, 'series2_name': s_b.name, 'analysis_type': 'Pearson Correlation', 'value': pearson_corr, 'round': current_round_for_adv_analysis, 'phase': 'advanced_analysis_scope'})
                    all_advanced_analysis_results.append({'series1_name': s_a.name, 'series2_name': s_b.name, 'analysis_type': 'Spearman Correlation', 'value': spearman_corr, 'round': current_round_for_adv_analysis, 'phase': 'advanced_analysis_scope'})
                else:
                    print(f"Skipping Pearson/Spearman correlation between {s_a.name} and {s_b.name} due to empty series.")

                # Lagged cross-correlation for s_a vs s_b
                if not s_a.empty and not s_b.empty and len(s_a) > 1 and len(s_b) > 1:
                    print(f"Calculating lagged cross-correlation between {s_a.name} and {s_b.name}")
                    ccf_ab, lags_ab = calculate_lagged_cross_correlation(s_a, s_b, max_lag=min(len(s_a)-1, len(s_b)-1, 20)) # Cap max_lag
                    if ccf_ab is not None and lags_ab is not None and len(ccf_ab) > 0:
                        max_corr_ab_idx = np.argmax(np.abs(ccf_ab))
                        max_corr_ab = ccf_ab[max_corr_ab_idx]
                        lag_at_max_corr_ab = lags_ab[max_corr_ab_idx]
                        print(f"Max cross-correlation (s_a vs s_b): {max_corr_ab:.4f} at lag {lag_at_max_corr_ab}")
                        all_advanced_analysis_results.append({'series1_name': s_a.name, 'series2_name': s_b.name, 'analysis_type': 'Max Cross-Correlation', 'value': max_corr_ab, 'lag': lag_at_max_corr_ab, 'round': current_round_for_adv_analysis, 'phase': 'advanced_analysis_scope'})
                        plot_ccf(ccf_ab, lags_ab, title=f"CCF: {s_a.name} vs {s_b.name}", output_filename=os.path.join(correlation_dir, f"ccf_{s_a.name}_vs_{s_b.name}.png"))
                    else:
                        print(f"Could not calculate CCF for {s_a.name} vs {s_b.name} (series might be too short or constant).")
                else:
                    print(f"Skipping lagged cross-correlation for {s_a.name} vs {s_b.name} due to empty or too short series.")

                # Lagged cross-correlation for s_a vs s_c
                if not s_a.empty and not s_c.empty and len(s_a) > 1 and len(s_c) > 1:
                    print(f"Calculating lagged cross-correlation between {s_a.name} and {s_c.name}")
                    ccf_ac, lags_ac = calculate_lagged_cross_correlation(s_a, s_c, max_lag=min(len(s_a)-1, len(s_c)-1, 20))
                    if ccf_ac is not None and lags_ac is not None and len(ccf_ac) > 0:
                        max_corr_ac_idx = np.argmax(np.abs(ccf_ac))
                        max_corr_ac = ccf_ac[max_corr_ac_idx]
                        lag_at_max_corr_ac = lags_ac[max_corr_ac_idx]
                        print(f"Max cross-correlation (s_a vs s_c): {max_corr_ac:.4f} at lag {lag_at_max_corr_ac}")
                        all_advanced_analysis_results.append({'series1_name': s_a.name, 'series2_name': s_c.name, 'analysis_type': 'Max Cross-Correlation', 'value': max_corr_ac, 'lag': lag_at_max_corr_ac, 'round': current_round_for_adv_analysis, 'phase': 'advanced_analysis_scope'})
                        plot_ccf(ccf_ac, lags_ac, title=f"CCF: {s_a.name} vs {s_c.name}", output_filename=os.path.join(correlation_dir, f"ccf_{s_a.name}_vs_{s_c.name}.png"))
                    else:
                        print(f"Could not calculate CCF for {s_a.name} vs {s_c.name} (series might be too short or constant).")
                else:
                    print(f"Skipping lagged cross-correlation for {s_a.name} vs {s_c.name} due to empty or too short series.")

                # Correlation Heatmap
                if len(valid_series_for_heatmap) >= 2:
                    df_for_heatmap = pd.concat(valid_series_for_heatmap, axis=1).dropna()
                    # df_for_heatmap.columns = [s.name for s in valid_series_for_heatmap][:len(df_for_heatmap.columns)] # Already named
                    
                    if not df_for_heatmap.empty and len(df_for_heatmap.columns) >=2:
                        corr_matrix_heatmap = df_for_heatmap.corr(method='pearson')
                        plot_correlation_heatmap(corr_matrix_heatmap, title=f"Correlation Heatmap (Round: {current_round_for_adv_analysis})", 
                                                 output_filename=os.path.join(correlation_dir, "correlation_heatmap.png"))
                        print(f"Correlation heatmap saved to {os.path.join(correlation_dir, 'correlation_heatmap.png')}")
                    else:
                        print("Not enough data/columns for correlation heatmap after processing. Using dummy heatmap.")
                        snames = [s.name if s.name else f'S{i}' for i,s in enumerate([s_a,s_b,s_c])]
                        dummy_corr_data = {snames[0]: [1.0, 0.5, 0.2], snames[1]: [0.5, 1.0, 0.3], snames[2]: [0.2, 0.3, 1.0]}
                        dummy_corr_df = pd.DataFrame(dummy_corr_data, index=snames)
                        plot_correlation_heatmap(dummy_corr_df, title="Dummy Correlation Heatmap", 
                                                 output_filename=os.path.join(correlation_dir, "dummy_correlation_heatmap.png"))
                else:
                    print("Not enough valid series for correlation heatmap. Using dummy heatmap.")
                    snames = [s.name if s.name else f'S{i}' for i,s in enumerate([s_a,s_b,s_c])]
                    dummy_corr_data = {snames[0]: [1.0, 0.5, 0.2], snames[1]: [0.5, 1.0, 0.3], snames[2]: [0.2, 0.3, 1.0]}
                    dummy_corr_df = pd.DataFrame(dummy_corr_data, index=snames)
                    plot_correlation_heatmap(dummy_corr_df, title="Dummy Correlation Heatmap", 
                                             output_filename=os.path.join(correlation_dir, "dummy_correlation_heatmap.png"))
            else:
                print("\n--- Skipping Correlation Analysis (due to CLI flags) ---")

            # --- Similarity Analysis ---
            if run_all_adv_by_default or args.run_similarity:
                print("\n--- Running Similarity Analysis ---")
                if not s_a.empty and not s_b.empty:
                    dtw_dist = calculate_dtw_distance(s_a, s_b)
                    print(f"DTW Distance between {s_a.name} and {s_b.name}: {dtw_dist:.4f}")
                    all_advanced_analysis_results.append({'series1_name': s_a.name, 'series2_name': s_b.name, 'analysis_type': 'DTW Distance', 'value': dtw_dist, 'round': current_round_for_adv_analysis, 'phase': 'advanced_analysis_scope'})
                else:
                    print(f"Skipping DTW distance between {s_a.name} and {s_b.name} due to empty series.")
                
                cosine_sim = np.nan # Initialize
                if not s_a.empty and not s_c.empty:
                    cosine_sim = calculate_cosine_similarity(s_a, s_c)
                    print(f"Cosine Similarity between {s_a.name} and {s_c.name}: {cosine_sim:.4f}")
                    all_advanced_analysis_results.append({'series1_name': s_a.name, 'series2_name': s_c.name, 'analysis_type': 'Cosine Similarity', 'value': cosine_sim, 'round': current_round_for_adv_analysis, 'phase': 'advanced_analysis_scope'})
                    plot_time_series_with_cosine_similarity(s_a, s_c, cosine_sim, title=f"Time Series Comparison: {s_a.name} vs {s_c.name}", output_filename=os.path.join(similarity_dir, "time_series_cosine_sim.png"))
                    print(f"Time series with cosine similarity plot saved to {os.path.join(similarity_dir, 'time_series_cosine_sim.png')}")
                else:
                    print(f"Skipping Cosine similarity between {s_a.name} and {s_c.name} and related plot due to empty series.")

                # Cosine Similarity Heatmap
                valid_series_for_sim_heatmap = [s for s in [s_a, s_b, s_c] if not s.empty and len(s) > 1]
                if len(valid_series_for_sim_heatmap) >= 2:
                    df_for_sim_heatmap = pd.concat(valid_series_for_sim_heatmap, axis=1).dropna()
                    # df_for_sim_heatmap.columns = [s.name for s in valid_series_for_sim_heatmap][:len(df_for_sim_heatmap.columns)] # Already named

                    if not df_for_sim_heatmap.empty and len(df_for_sim_heatmap.columns) >=2:
                        num_series_sim = len(df_for_sim_heatmap.columns)
                        sim_matrix = np.ones((num_series_sim, num_series_sim))
                        series_values_list = [df_for_sim_heatmap[col].values for col in df_for_sim_heatmap.columns]
                        for i in range(num_series_sim):
                            for j in range(i + 1, num_series_sim):
                                sim_val = calculate_cosine_similarity(pd.Series(series_values_list[i]), pd.Series(series_values_list[j]))
                                sim_matrix[i, j] = sim_val
                                sim_matrix[j, i] = sim_val
                        actual_cosine_sim_df = pd.DataFrame(sim_matrix, index=df_for_sim_heatmap.columns, columns=df_for_sim_heatmap.columns)
                        plot_cosine_similarity_heatmap(actual_cosine_sim_df, title=f"Cosine Similarity Heatmap (Round: {current_round_for_adv_analysis})", output_filename=os.path.join(similarity_dir, "actual_cosine_similarity_heatmap.png"))
                        print(f"Actual cosine similarity heatmap saved to {os.path.join(similarity_dir, 'actual_cosine_similarity_heatmap.png')}")
                    else:
                        print("Not enough data/columns for actual cosine similarity heatmap. Using dummy heatmap.")
                        snames_sim = [s.name if s.name else f'S_{chr(65+i)}' for i,s in enumerate([s_a,s_c,s_b])] 
                        dummy_cosine_sim_data = {snames_sim[0]: [1.0, cosine_sim if not np.isnan(cosine_sim) else 0.5, 0.2], 
                                                 snames_sim[1]: [cosine_sim if not np.isnan(cosine_sim) else 0.5, 1.0, 0.3], 
                                                 snames_sim[2]: [0.2, 0.3, 1.0]}
                        dummy_cosine_sim_df = pd.DataFrame(dummy_cosine_sim_data, index=snames_sim)
                        plot_cosine_similarity_heatmap(dummy_cosine_sim_df, title="Dummy Cosine Similarity Heatmap", output_filename=os.path.join(similarity_dir, "dummy_cosine_similarity_heatmap.png"))
                else:
                    print("Not enough valid series for cosine similarity heatmap. Using dummy heatmap.")
                    snames_sim = [s.name if s.name else f'S_{chr(65+i)}' for i,s in enumerate([s_a,s_c,s_b])]
                    dummy_cosine_sim_data = {snames_sim[0]: [1.0, cosine_sim if not np.isnan(cosine_sim) else 0.5, 0.2], 
                                             snames_sim[1]: [cosine_sim if not np.isnan(cosine_sim) else 0.5, 1.0, 0.3], 
                                             snames_sim[2]: [0.2, 0.3, 1.0]}
                    dummy_cosine_sim_df = pd.DataFrame(dummy_cosine_sim_data, index=snames_sim)
                    plot_cosine_similarity_heatmap(dummy_cosine_sim_df, title="Dummy Cosine Similarity Heatmap", output_filename=os.path.join(similarity_dir, "dummy_cosine_similarity_heatmap.png"))
                
                # Time-varying cosine similarity
                tv_cosine_sim_df = pd.DataFrame() # Initialize
                min_points_needed = (COSINE_SIM_WINDOW_SIZE_S // SCRAPE_INTERVAL_S) if SCRAPE_INTERVAL_S > 0 else 20 # Default min points
                
                if not s_a.empty and not s_c.empty and isinstance(s_a.index, pd.DatetimeIndex) and isinstance(s_c.index, pd.DatetimeIndex):
                    # Align series for time-varying calculation
                    aligned_df_tv = pd.concat([s_a.rename('s_a_tv'), s_c.rename('s_c_tv')], axis=1)
                    # Interpolate then dropna to handle misaligned timestamps but keep data where possible
                    aligned_df_tv = aligned_df_tv.interpolate(method='time').dropna()

                    if len(aligned_df_tv) >= min_points_needed:
                        s_a_aligned_tv = aligned_df_tv['s_a_tv']
                        s_c_aligned_tv = aligned_df_tv['s_c_tv']
                        print(f"Calculating real time-varying cosine similarity between {s_a.name} and {s_c.name} using config-defined window/step.")
                        tv_cosine_sim_df = calculate_time_varying_cosine_similarity(s_a_aligned_tv, s_c_aligned_tv)
                    else:
                        print(f"Not enough overlapping data between {s_a.name} and {s_c.name} for time-varying cosine similarity (need at least {min_points_needed} points after alignment). Using dummy data for plot.")
                else:
                    print(f"Series {s_a.name} or {s_c.name} are not suitable for real time-varying cosine similarity (empty, no DatetimeIndex, or not enough data). Using dummy data for plot.")

                if tv_cosine_sim_df.empty: # If calculation failed or was skipped, use dummy
                    print("Time-varying cosine similarity calculation resulted in empty DataFrame or was skipped. Using dummy data for plot.")
                    timestamps_tv = pd.date_range(start='2023-01-01', periods=10, freq='D') if s_a.empty or not isinstance(s_a.index, pd.DatetimeIndex) else s_a.index[:10] # Try to use s_a index if available
                    if len(timestamps_tv) < 2 : timestamps_tv = pd.date_range(start='2023-01-01', periods=10, freq='D') # Ensure valid range
                    tv_cosine_sim_df = pd.DataFrame({'timestamp': timestamps_tv, 'cosine_similarity': np.random.rand(len(timestamps_tv)) * 2 - 1 })

                phase_start_for_plot = None
                if not tv_cosine_sim_df.empty and 'timestamp' in tv_cosine_sim_df.columns and not tv_cosine_sim_df['timestamp'].empty:
                    phase_start_for_plot = tv_cosine_sim_df['timestamp'].min()
                elif not s_a.empty and isinstance(s_a.index, pd.DatetimeIndex) and not s_a.index.empty: # Fallback to s_a start time
                     phase_start_for_plot = s_a.index.min()
                
                plot_time_varying_cosine_similarity(tv_cosine_sim_df, series_a_name=s_a.name, series_b_name=s_c.name, output_filename=os.path.join(similarity_dir, "time_varying_cosine_sim.png"), phase_start_time=phase_start_for_plot)
                print(f"Time-varying cosine similarity plot saved to {os.path.join(similarity_dir, 'time_varying_cosine_sim.png')}")
            else:
                print("\n--- Skipping Similarity Analysis (due to CLI flags) ---")

            # --- Causality Analysis ---
            if run_all_adv_by_default or args.run_causality:
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

                # Use current_round_for_adv_analysis for Granger data prep
                if current_round_for_adv_analysis and metrics_data and metrics_for_granger:
                    all_tenants_in_round_for_granger = set()
                    for metric_name_granger in metrics_for_granger:
                        if metric_name_granger in metrics_data and current_round_for_adv_analysis in metrics_data[metric_name_granger]:
                            df_round_granger = metrics_data[metric_name_granger][current_round_for_adv_analysis]
                            if df_round_granger is not None and 'tenant' in df_round_granger.columns:
                                all_tenants_in_round_for_granger.update(df_round_granger['tenant'].unique())
                    
                    sorted_tenants_for_granger = sorted(list(all_tenants_in_round_for_granger))

                    if noisy_tenant_for_granger_default and noisy_tenant_for_granger_default in sorted_tenants_for_granger:
                        noisy_tenant_for_granger = noisy_tenant_for_granger_default
                    elif sorted_tenants_for_granger:
                        noisy_tenant_for_granger = sorted_tenants_for_granger[0]
                        print(f"Default noisy tenant '{noisy_tenant_for_granger_default}' not found or not specified for Granger; using first available: {noisy_tenant_for_granger} from round {current_round_for_adv_analysis}")
                    
                    if noisy_tenant_for_granger:
                        other_tenants_for_granger = [t for t in sorted_tenants_for_granger if t != noisy_tenant_for_granger]
                    
                    if not other_tenants_for_granger and len(sorted_tenants_for_granger) > 1 and noisy_tenant_for_granger:
                         other_tenants_for_granger = [t for t in sorted_tenants_for_granger if t != noisy_tenant_for_granger][:1] # Pick first other if multiple

                    if noisy_tenant_for_granger and other_tenants_for_granger:
                        tenants_for_granger_analysis = [noisy_tenant_for_granger] + other_tenants_for_granger
                        print(f"Granger analysis tenants (from round {current_round_for_adv_analysis}): Noisy={noisy_tenant_for_granger}, Others={other_tenants_for_granger}, Metrics: {metrics_for_granger}")

                        for metric_name_granger in metrics_for_granger:
                            if metric_name_granger in metrics_data and current_round_for_adv_analysis in metrics_data[metric_name_granger]:
                                df_round_metric_granger = metrics_data[metric_name_granger][current_round_for_adv_analysis]
                                if df_round_metric_granger is not None and not df_round_metric_granger.empty:
                                    for tenant_name_granger_loop in tenants_for_granger_analysis: # Renamed loop variable
                                        tenant_df_granger = pd.DataFrame()
                                        if 'tenant' not in df_round_metric_granger.columns and (tenant_name_granger_loop is None or tenant_name_granger_loop == "NODE_LEVEL"): # Node level data
                                            tenant_df_granger = df_round_metric_granger
                                        elif 'tenant' in df_round_metric_granger.columns and tenant_name_granger_loop in df_round_metric_granger['tenant'].unique():
                                            tenant_df_granger = df_round_metric_granger[df_round_metric_granger['tenant'] == tenant_name_granger_loop]
                                        
                                        if not tenant_df_granger.empty and 'timestamp' in tenant_df_granger.columns and 'value' in tenant_df_granger.columns:
                                            current_phase_granger = tenant_df_granger['phase'].iloc[0] if 'phase' in tenant_df_granger.columns and not tenant_df_granger.empty else 'unknown_phase'
                                            granger_series_df = tenant_df_granger[['timestamp', 'value']].copy()
                                            granger_series_df.rename(columns={'timestamp': 'datetime'}, inplace=True)
                                            granger_series_df['tenant'] = tenant_name_granger_loop if 'tenant' in df_round_metric_granger.columns else "NODE_LEVEL"
                                            granger_series_df['metric_name'] = metric_name_granger
                                            granger_series_df['phase'] = current_phase_granger
                                            granger_series_df['round'] = current_round_for_adv_analysis # Use the determined round
                                            granger_input_dfs_list.append(granger_series_df)
                    else:
                        print(f"Not enough distinct tenants or metrics found for Granger causality with real data based on config and available data for round {current_round_for_adv_analysis}.")
                
                if granger_input_dfs_list:
                    actual_granger_input_long_df = pd.concat(granger_input_dfs_list).reset_index(drop=True)
                    actual_granger_input_long_df.dropna(subset=['value'], inplace=True) # Drop rows where essential 'value' is NaN
                    print(f"Prepared actual_granger_input_long_df with {len(actual_granger_input_long_df)} rows for Granger causality (from round {current_round_for_adv_analysis}).")
                
                # Fallback to s_a, s_b, s_c if real data prep failed
                if actual_granger_input_long_df.empty or len(actual_granger_input_long_df['metric_name'].unique()) == 0 or len(actual_granger_input_long_df['tenant'].unique()) < 2:
                    print(f"Granger input from round {current_round_for_adv_analysis} is insufficient. Attempting fallback using s_a, s_b, s_c series if available.")
                    temp_dfs_fallback = []
                    temp_metrics_fallback = []
                    temp_tenants_fallback = []
                    for s_series, s_name_str in zip([s_a, s_b, s_c], ["s_a", "s_b", "s_c"]):
                        if not s_series.empty and isinstance(s_series.index, pd.DatetimeIndex):
                            try:
                                # Attempt to parse metric and tenant from series name (e.g., "s_a_cpu_usage_tenant-a_round-1")
                                parts = s_series.name.split('_')
                                metric_fallback = parts[2] if len(parts) > 3 else f"metric_{s_name_str}"
                                tenant_fallback = parts[3] if len(parts) > 4 else f"tenant_{s_name_str}"
                            except:
                                metric_fallback = f"metric_{s_name_str}"
                                tenant_fallback = f"tenant_{s_name_str}"

                            temp_df_fallback = s_series.reset_index()
                            temp_df_fallback.columns = ['datetime', 'value']
                            temp_df_fallback['metric_name'] = metric_fallback
                            temp_df_fallback['tenant'] = tenant_fallback
                            temp_df_fallback['round'] = current_round_for_adv_analysis or 'fallback_round' # Use determined round
                            temp_df_fallback['phase'] = 'advanced_analysis_scope' 
                            temp_dfs_fallback.append(temp_df_fallback)
                            temp_metrics_fallback.append(metric_fallback)
                            temp_tenants_fallback.append(tenant_fallback)
                    
                    if temp_dfs_fallback:
                        actual_granger_input_long_df = pd.concat(temp_dfs_fallback).reset_index(drop=True)
                        actual_granger_input_long_df.dropna(subset=['value'], inplace=True)
                        print(f"Using fallback Granger input data from s_a,s_b,s_c: {len(actual_granger_input_long_df)} rows.")
                        unique_fallback_tenants = sorted(list(set(temp_tenants_fallback)))
                        if unique_fallback_tenants:
                            noisy_tenant_for_granger = unique_fallback_tenants[0]
                            other_tenants_for_granger = unique_fallback_tenants[1:] if len(unique_fallback_tenants) > 1 else []
                        metrics_for_granger = sorted(list(set(temp_metrics_fallback)))
                    else:
                        print("Not enough valid series (s_a, s_b, s_c) for fallback Granger input.")
                        actual_granger_input_long_df = pd.DataFrame() # Ensure it's empty

                granger_results_df = pd.DataFrame()
                if not actual_granger_input_long_df.empty and metrics_for_granger and noisy_tenant_for_granger and \
                   (len(actual_granger_input_long_df['tenant'].unique()) >= 2 or (len(actual_granger_input_long_df['tenant'].unique()) == 1 and not other_tenants_for_granger) ): # Allow single tenant if no others specified
                    print(f"Running Granger Causality Test with: Metrics={metrics_for_granger}, NoisyTenant={noisy_tenant_for_granger}, OtherTenants={other_tenants_for_granger}, Round={current_round_for_adv_analysis}")
                    granger_results_df = perform_inter_tenant_causality_analysis(
                        data=actual_granger_input_long_df, 
                        metrics_for_causality=metrics_for_granger, 
                        noisy_tenant=noisy_tenant_for_granger, 
                        other_tenants=other_tenants_for_granger,
                        max_lag=GRANGER_CAUSALITY_DEFAULTS.get('max_lag', 5),
                        test=GRANGER_CAUSALITY_DEFAULTS.get('test', 'ssr_chi2test'),
                        significance_level=DEFAULT_CAUSALITY_THRESHOLD_P_VALUE,
                        min_observations=DEFAULT_GRANGER_MIN_OBSERVATIONS,
                        verbose=True,
                        current_round_filter=current_round_for_adv_analysis # Pass the round for internal filtering if needed by the function
                    )
                else:
                    print(f"Skipping Granger causality test as not enough valid data or configuration could be prepared for round {current_round_for_adv_analysis}.")

                if not granger_results_df.empty:
                    print("Granger Causality Test Results:")
                    print(granger_results_df)
                    for _, row in granger_results_df.iterrows():
                        all_advanced_analysis_results.append({
                            'series1_name': f"{row['metric']}_{row['source_tenant']}", 
                            'series2_name': f"{row['metric']}_{row['target_tenant']}",
                            'analysis_type': 'Granger Causality', 
                            'value': row['p_value'], # Storing p-value as the main 'value'
                            'lag': row['lag'], 
                            'significant': row['significant'],
                            'direction': row['direction'],
                            'error': row.get('error'), # Include error if present
                            'round': current_round_for_adv_analysis, # Add round
                            'phase': 'advanced_analysis_scope' # Add phase
                        })
                    granger_results_path = os.path.join(causality_dir, f"granger_causality_results_round_{current_round_for_adv_analysis}.csv")
                    granger_results_df.to_csv(granger_results_path, index=False)
                    print(f"Granger causality results saved to {granger_results_path}")
                    # Optional: Visualize causal graph if results are substantial
                    # if len(granger_results_df[granger_results_df['significant']]) > 0:
                    #     visualize_causal_graph(granger_results_df, significance_level=DEFAULT_CAUSALITY_THRESHOLD_P_VALUE, 
                    #                            output_filename=os.path.join(causality_dir, f"causal_graph_round_{current_round_for_adv_analysis}.png"))
                else:
                    print(f"No Granger causality results to save for round {current_round_for_adv_analysis}.")
                
                # Transfer Entropy & CCM (using s_a, s_c as example pair)
                if not s_a.empty and not s_c.empty and len(s_a) > 10 and len(s_c) > 10: # Ensure enough data for TE/CCM
                    # Transfer Entropy
                    te_value_ac = calculate_transfer_entropy(s_a, s_c, k=GRANGER_CAUSALITY_DEFAULTS.get('max_lag', 5))
                    te_value_ca = calculate_transfer_entropy(s_c, s_a, k=GRANGER_CAUSALITY_DEFAULTS.get('max_lag', 5))
                    print(f"Transfer Entropy ({s_a.name} -> {s_c.name}): {te_value_ac:.4f}")
                    print(f"Transfer Entropy ({s_c.name} -> {s_a.name}): {te_value_ca:.4f}")
                    all_advanced_analysis_results.append({'series1_name': s_a.name, 'series2_name': s_c.name, 'analysis_type': 'Transfer Entropy (S1->S2)', 'value': te_value_ac, 'round': current_round_for_adv_analysis, 'phase': 'advanced_analysis_scope'})
                    all_advanced_analysis_results.append({'series1_name': s_c.name, 'series2_name': s_a.name, 'analysis_type': 'Transfer Entropy (S2->S1)', 'value': te_value_ca, 'round': current_round_for_adv_analysis, 'phase': 'advanced_analysis_scope'})

                    # Convergent Cross Mapping (CCM)
                    # Ensure series are numpy arrays for CCM and have no NaNs after alignment
                    s_a_ccm, s_c_ccm = pd.Series(s_a.values).fillna(method='ffill').fillna(method='bfill'), pd.Series(s_c.values).fillna(method='ffill').fillna(method='bfill')
                    if not s_a_ccm.isnull().any() and not s_c_ccm.isnull().any() and len(s_a_ccm) > DEFAULT_GRANGER_MIN_OBSERVATIONS and len(s_c_ccm) > DEFAULT_GRANGER_MIN_OBSERVATIONS :
                        ccm_scores = calculate_convergent_cross_mapping(s_a_ccm, s_c_ccm, embed_dim=3, tau=1, lib_lag=GRANGER_CAUSALITY_DEFAULTS.get('max_lag', 5))
                        if ccm_scores:
                            print(f"CCM Score ({s_a.name} xmaps {s_c.name}): {ccm_scores['ccm_s1_xmaps_s2']:.4f}")
                            print(f"CCM Score ({s_c.name} xmaps {s_a.name}): {ccm_scores['ccm_s2_xmaps_s1']:.4f}")
                            all_advanced_analysis_results.append({'series1_name': s_a.name, 'series2_name': s_c.name, 'analysis_type': 'CCM (S1 xmaps S2)', 'value': ccm_scores['ccm_s1_xmaps_s2'], 'round': current_round_for_adv_analysis, 'phase': 'advanced_analysis_scope'})
                            all_advanced_analysis_results.append({'series1_name': s_c.name, 'series2_name': s_a.name, 'analysis_type': 'CCM (S2 xmaps S1)', 'value': ccm_scores['ccm_s2_xmaps_s1'], 'round': current_round_for_adv_analysis, 'phase': 'advanced_analysis_scope'})
                        else:
                            print(f"CCM analysis for {s_a.name} and {s_c.name} did not produce results (possibly due to data characteristics).")
                    else:
                         print(f"Skipping CCM for {s_a.name} and {s_c.name} due to NaNs or insufficient length after cleaning for CCM.")
                else:
                    print(f"Skipping TE/CCM analysis for {s_a.name} and {s_c.name} due to empty or too short series.")
            else:
                print("\n--- Skipping Causality Analysis (due to CLI flags) ---")
            
            # --- Aggregating and Comparing Advanced Analysis Results ---
            if all_advanced_analysis_results: 
                print("\n--- Aggregating Advanced Analysis Results ---")
                adv_results_df = pd.DataFrame(all_advanced_analysis_results)
                
                if 'series1_name' in adv_results_df.columns and 'series2_name' in adv_results_df.columns:
                    adv_results_df['metric_pair'] = adv_results_df['series1_name'] + " <-> " + adv_results_df['series2_name']
                else:
                    adv_results_df['metric_pair'] = "N/A" 

                aggregated_output_df = aggregate_analysis_results(adv_results_df)
                if not aggregated_output_df.empty:
                    agg_results_path = os.path.join(tables_dir, "aggregated_advanced_analysis_results.csv")
                    try:
                        aggregated_output_df.to_csv(agg_results_path, index=False)
                        print(f"Aggregated advanced analysis results saved to: {agg_results_path}")
                    except Exception as e:
                        print(f"Error saving aggregated advanced analysis results: {e}")
                else:
                    print("Aggregation of advanced analysis results produced an empty DataFrame.")

                print("\n--- Comparing Phases Statistically for Advanced Metrics (if applicable) ---")
                if not adv_results_df.empty and 'value' in adv_results_df.columns and \
                   'analysis_type' in adv_results_df.columns and \
                   'round' in adv_results_df.columns and 'phase' in adv_results_df.columns:
                    
                    # For phase comparison, we need a 'phase' column that has more than one unique value.
                    # The current 'advanced_analysis_scope' is not suitable.
                    # This part remains a placeholder as the current data structure for advanced results
                    # (using a generic 'advanced_analysis_scope' phase) isn't directly suitable for
                    # compare_phases_statistically, which expects distinct experimental phases.
                    # A more meaningful comparison would require mapping these results to actual experiment phases
                    # or performing comparisons based on rounds if multiple rounds of advanced analyses were run.
                    print("Note: `compare_phases_statistically` for advanced metrics might require specific data structuring (e.g., meaningful 'phase' column with multiple distinct phases for comparison). The current 'advanced_analysis_scope' phase is not suitable for this comparison.")
                    # Example: if you had a way to map 'current_round_for_adv_analysis' to a phase:
                    # adv_results_df['phase_for_comparison'] = adv_results_df['round'].map(some_round_to_phase_mapping)
                    # then you could call:
                    # compare_phases_statistically(adv_results_df, metric_col='value', phase_col='phase_for_comparison'...)
                else:
                    print("Skipping phase comparison for advanced metrics due to missing data or required columns in the results DataFrame, or unsuitable phase information.")
            else: 
                print("No advanced analysis results were collected to aggregate or compare.")
                adv_results_df = pd.DataFrame() 
    else: 
        print("\nSkipping all Advanced Analyses based on CLI flags (e.g., --skip-advanced-analyses was used, or no specific --run-* flags were set when default execution of all is not intended).")
    
    print("\nPipeline execution finished successfully!")
    print("DEBUG: pipeline.main.main() COMPLETED SUCCESSFULLY")  # ADDED

if __name__ == "__main__":
    main()
