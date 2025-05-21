import argparse
import os
import sys
import traceback
import pandas as pd
import numpy as np

# Ensure traceback is imported at the top
# Added for itertools.combinations
# Added for plt.close

from new_config import TENANT_COLORS, METRICS_CONFIG
# from data_handling.loader import load_data_from_all_rounds_and_phases, load_data_from_specific_round_consolidated # Corrected import names
from data_handling.loader import load_experiment_data # Changed to import load_experiment_data
from data_handling.new_time_normalization import normalize_time_across_phases_and_rounds
from data_handling.new_metric_normalization import scale_features_across_phases_and_rounds
from data_handling.save_results import export_to_csv # Assuming setup_output_directories is in new_main.py
from analysis_modules.descritive_statistics import (
    calculate_descriptive_statistics_for_metric, 
    plot_descriptive_stats_boxplots_for_metric, 
    plot_descriptive_stats_histograms_for_metric
)
from analysis_modules.multivariate_exploration import perform_pca, perform_ica, get_top_features_per_component
from analysis_modules.correlation_covariance import (
    calculate_correlations_for_metric, plot_correlation_heatmap, plot_correlation_scatter_matrix,
    calculate_covariances_for_metric, plot_covariance_heatmap
)
from visualization.new_plots import (
    plot_pca_scree_plot, plot_pca_score_plot, plot_pca_loadings_heatmap,
    plot_ica_score_plot, plot_ica_loadings_heatmap
)

# Function definitions (parse_arguments, setup_output_directories, load_and_preprocess_data, etc.)
# are assumed to be defined in this file or correctly imported if they were meant to be elsewhere.

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run data analysis pipeline for k8s noisy detection.")
    parser.add_argument("--data-dir", required=True, help="Root directory containing the experimental data.")
    parser.add_argument("--output-dir", required=True, help="Directory to save results and plots.")
    parser.add_argument("--metrics-config", default="refactor/new_config.py", help="Path to the metrics configuration file.")
    parser.add_argument("--selected-metrics", nargs='*', help="List of specific metrics to process (e.g., cpu_usage memory_usage). Processes all if not specified.")
    parser.add_argument("--start-time", help="Global start time for analysis (YYYY-MM-DDTHH:MM:SS or relative like -60s, -10m, -1h from first event).")
    parser.add_argument("--end-time", help="Global end time for analysis (YYYY-MM-DDTHH:MM:SS or relative like +60s, +10m, +1h from last event).")
    parser.add_argument("--normalization-scope", choices=['phase', 'round', 'experiment'], default='phase', help="Scope for time normalization.")
    parser.add_argument("--time-normalization-target", nargs='*', default=['AttackStart', 'AttackEnd'], help="Event names to normalize time against.")
    parser.add_argument("--feature-scaling-method", choices=['minmax', 'standard', 'none'], default='minmax', help="Method for feature scaling.")
    parser.add_argument("--run-per-phase", action='store_true', help="Run analysis per phase instead of consolidated per round.")
    
    parser.add_argument("--run-descriptive-stats", action="store_true", help="Run descriptive statistics module.")
    parser.add_argument("--run-correlation-covariance", action="store_true", help="Run correlation and covariance analysis module.")
    parser.add_argument("--correlation-methods", nargs='*', default=['pearson', 'spearman'], help="Methods for correlation (pearson, kendall, spearman).")
    parser.add_argument("--run-pca", action="store_true", help="Run PCA module.")
    parser.add_argument("--pca-n-components", default=None, help="Number of principal components (int) or variance threshold (float, e.g., 0.95).")
    parser.add_argument("--pca-variance-threshold", type=float, default=0.95, help="Variance threshold for PCA if n_components is not an int.")
    parser.add_argument("--run-ica", action="store_true", help="Run ICA module.")
    parser.add_argument("--ica-n-components", type=int, default=None, help="Number of independent components.")
    parser.add_argument("--compare-pca-ica", action="store_true", help="Compare top features from PCA and ICA.")
    parser.add_argument("--n-top-features-comparison", type=int, default=5, help="Number of top features to compare between PCA and ICA.")

    return parser.parse_args()

def setup_output_directories(output_dir_base):
    plots_dir = os.path.join(output_dir_base, 'plots')
    tables_dir = os.path.join(output_dir_base, 'tables')

    desc_stats_plots_dir = os.path.join(plots_dir, "descriptive_stats")
    desc_stats_tables_dir = os.path.join(tables_dir, "descriptive_stats")

    correlation_plots_dir = os.path.join(plots_dir, "correlation")
    correlation_tables_dir = os.path.join(tables_dir, "correlation")

    covariance_plots_dir = os.path.join(plots_dir, "covariance")
    covariance_tables_dir = os.path.join(tables_dir, "covariance")

    multivariate_plots_dir = os.path.join(plots_dir, "multivariate")
    pca_plots_output_dir = os.path.join(multivariate_plots_dir, "pca")
    ica_plots_output_dir = os.path.join(multivariate_plots_dir, "ica")

    multivariate_tables_dir = os.path.join(tables_dir, "multivariate")
    pca_tables_output_dir = os.path.join(multivariate_tables_dir, "pca")
    ica_tables_output_dir = os.path.join(multivariate_tables_dir, "ica")
    comparison_tables_output_dir = os.path.join(multivariate_tables_dir, "comparison")

    # Create all directories
    for path in [plots_dir, tables_dir, desc_stats_plots_dir, desc_stats_tables_dir,
                 correlation_plots_dir, correlation_tables_dir, covariance_plots_dir, covariance_tables_dir,
                 multivariate_plots_dir, pca_plots_output_dir, ica_plots_output_dir,
                 multivariate_tables_dir, pca_tables_output_dir, ica_tables_output_dir, comparison_tables_output_dir]:
        os.makedirs(path, exist_ok=True)

    # Return the 7 specific paths expected by the previous version of main's unpacking
    # Plus the general plots and tables dirs for other modules if they need them directly
    return (plots_dir, tables_dir, 
            pca_tables_output_dir, ica_tables_output_dir, comparison_tables_output_dir,
            pca_plots_output_dir, ica_plots_output_dir,
            desc_stats_plots_dir, desc_stats_tables_dir, # Added for descriptive stats
            correlation_plots_dir, correlation_tables_dir, # Added for correlation
            covariance_plots_dir, covariance_tables_dir # Added for covariance
           ) 

def load_and_preprocess_data(data_dir, metrics_config_path, selected_metrics_list, 
                               start_time_str, end_time_str, normalization_scope, 
                               time_normalization_target_list, feature_scaling_method_name, 
                               run_per_phase_flag):
    print(f"DEBUG: Starting load_and_preprocess_data.")
    sys.stdout.flush()
    metrics_processing_config = METRICS_CONFIG 
    print(f"DEBUG: Metrics config loaded. {len(metrics_processing_config)} metrics defined.")
    sys.stdout.flush()

    if not selected_metrics_list: # If None or empty, process all configured metrics
        selected_metrics_list = list(metrics_processing_config.keys())
    print(f"DEBUG: Selected metrics for processing: {selected_metrics_list}")
    sys.stdout.flush()

    all_metrics_data_dict = {}
    print(f"DEBUG: Loading data. Data directory: {data_dir}, Per phase: {run_per_phase_flag}, Metrics: {selected_metrics_list}")
    sys.stdout.flush()
    
    # Call load_experiment_data with the correct arguments
    # tenants, phases, and rounds arguments are omitted to use their defaults (load all).
    all_metrics_data_dict = load_experiment_data(
        experiment_dir=data_dir,
        metrics=selected_metrics_list,
        group_by_phase=run_per_phase_flag
    )
    
    print(f"DEBUG: Raw data loaded for {len(all_metrics_data_dict)} metrics.")
    sys.stdout.flush()

    # Time Normalization
    print(f"DEBUG: Starting time normalization. Scope: {normalization_scope}, Target: {time_normalization_target_list}")
    sys.stdout.flush()
    # all_metrics_data_dict = normalize_time_across_phases_and_rounds(
    #     all_metrics_data_dict, 
    #     target_events=time_normalization_target_list,
    #     scope=normalization_scope,
    #     start_time_str=start_time_str, 
    #     end_time_str=end_time_str
    # )
    # print(f"DEBUG: Time normalization complete.")
    # sys.stdout.flush()

    # Feature Scaling
    # if feature_scaling_method_name != 'none':
    #     print(f"DEBUG: Starting feature scaling. Method: {feature_scaling_method_name}")
    #     sys.stdout.flush()
    #     all_metrics_data_dict = scale_features_across_phases_and_rounds(
    #         all_metrics_data_dict, 
    #         method=feature_scaling_method_name,
    #         scope=normalization_scope # Assuming scaling scope is same as normalization scope for now
    #     )
    #     print(f"DEBUG: Feature scaling complete.")
    #     sys.stdout.flush()
    # else:
    #     print(f"DEBUG: Feature scaling skipped (method is 'none').")
    #     sys.stdout.flush()

    print(f"DEBUG: load_and_preprocess_data finished.")
    sys.stdout.flush()
    return all_metrics_data_dict, selected_metrics_list, run_per_phase_flag


def main():
    print(f"DEBUG: Script new_main.py starting...")
    sys.stdout.flush()
    args = parse_arguments()
    print(f"DEBUG: Parsed arguments: {args}")
    sys.stdout.flush()

    try:
        # Unpack all 13 values returned by the corrected setup_output_directories
        plots_dir, tables_dir, \
        pca_tables_output_dir, ica_tables_output_dir, comparison_tables_output_dir, \
        pca_plots_output_dir, ica_plots_output_dir, \
        desc_stats_plots_dir, desc_stats_tables_dir, \
        correlation_plots_dir, correlation_tables_dir, \
        covariance_plots_dir, covariance_tables_dir = setup_output_directories(args.output_dir)

        print(f"DEBUG: Output directories setup complete.")
        print(f"DEBUG: General plots_dir: {plots_dir}")
        print(f"DEBUG: General tables_dir: {tables_dir}")
        print(f"DEBUG: PCA plots dir: {pca_plots_output_dir}, PCA tables dir: {pca_tables_output_dir}")
        print(f"DEBUG: ICA plots dir: {ica_plots_output_dir}, ICA tables dir: {ica_tables_output_dir}")
        print(f"DEBUG: Comparison tables dir: {comparison_tables_output_dir}")
        print(f"DEBUG: Descriptive Stats plots dir: {desc_stats_plots_dir}, tables dir: {desc_stats_tables_dir}")
        print(f"DEBUG: Correlation plots dir: {correlation_plots_dir}, tables dir: {correlation_tables_dir}")
        print(f"DEBUG: Covariance plots dir: {covariance_plots_dir}, tables dir: {covariance_tables_dir}")
        sys.stdout.flush()
    except Exception as e:
        print(f"CRITICAL ERROR during setup_output_directories: {e}", file=sys.stderr)
        print(f"Full traceback:\\n{traceback.format_exc()}", file=sys.stderr)
        sys.stderr.flush()
        return 

    all_metrics_data = None
    run_per_phase_analysis = False 
    selected_metrics = []

    try:
        all_metrics_data, selected_metrics, run_per_phase_analysis = load_and_preprocess_data(
            args.data_dir, 
            args.metrics_config, # This should be the path to the config file
            args.selected_metrics, 
            args.start_time, 
            args.end_time,
            args.normalization_scope,
            args.time_normalization_target,
            args.feature_scaling_method,
            args.run_per_phase 
        )
        print(f"DEBUG: Data loading and preprocessing complete. run_per_phase_analysis: {run_per_phase_analysis}")
        sys.stdout.flush()
    except Exception as e:
        print(f"CRITICAL ERROR during load_and_preprocess_data: {e}", file=sys.stderr)
        print(f"Full traceback:\\n{traceback.format_exc()}", file=sys.stderr)
        sys.stderr.flush()
        return 

    if not all_metrics_data:
        print(f"DEBUG: all_metrics_data is empty or None after load_and_preprocess_data. Exiting main processing logic.")
        sys.stdout.flush()
        return # Exit if no data
    else:
        print(f"DEBUG: all_metrics_data keys: {list(all_metrics_data.keys()) if all_metrics_data else 'None'}")
        sys.stdout.flush()
    
    print(f"DEBUG: run_per_phase_analysis is set to: {run_per_phase_analysis}")
    sys.stdout.flush()

    # if args.run_descriptive_stats:
    #     print("DEBUG: Running descriptive statistics...")
    #     sys.stdout.flush()
    #     for metric_name in selected_metrics:
    #         if metric_name in all_metrics_data:
    #             print(f"  Calculating descriptive stats for metric: {metric_name}")
    #             sys.stdout.flush()
    #             stats_results, phase_stats_figs, round_stats_figs = calculate_descriptive_statistics_for_metric(
    #                 all_metrics_data[metric_name], 
    #                 metric_name,
    #                 desc_stats_tables_dir, # Use the specific tables dir
    #                 run_per_phase_analysis
    #             )
    #             # Plotting calls would go here, using desc_stats_plots_dir
    #             plot_descriptive_stats_boxplots_for_metric(all_metrics_data[metric_name], metric_name, desc_stats_plots_dir, run_per_phase_analysis, TENANT_COLORS)
    #             plot_descriptive_stats_histograms_for_metric(all_metrics_data[metric_name], metric_name, desc_stats_plots_dir, run_per_phase_analysis)
    #             print(f"  Descriptive stats for {metric_name} completed and saved.")
    #             sys.stdout.flush()
    #         else:
    #             print(f"  Skipping descriptive stats for metric {metric_name} as it's not in loaded data.")
    #             sys.stdout.flush()
    
    # if args.run_correlation_covariance:
    #     print("DEBUG: Running correlation/covariance analysis...")
    #     sys.stdout.flush()
    #     for metric_name in selected_metrics:
    #         if metric_name in all_metrics_data:
    #             print(f"  Calculating correlation/covariance for metric: {metric_name}")
    #             sys.stdout.flush()
    #             # Correlation
    #             correlation_results = calculate_correlations_for_metric(
    #                 all_metrics_data[metric_name],
    #                 metric_name,
    #                 args.correlation_methods,
    #                 correlation_tables_dir, # Use specific tables dir
    #                 run_per_phase_analysis
    #             )
    #             plot_correlation_heatmap(all_metrics_data[metric_name], metric_name, correlation_plots_dir, run_per_phase_analysis, args.correlation_methods)
    #             # plot_correlation_scatter_matrix(all_metrics_data[metric_name], metric_name, correlation_plots_dir, run_per_phase_analysis)
                
    #             # Covariance
    #             covariance_results = calculate_covariances_for_metric(
    #                 all_metrics_data[metric_name],
    #                 metric_name,
    #                 covariance_tables_dir, # Use specific tables dir
    #                 run_per_phase_analysis
    #             )
    #             plot_covariance_heatmap(all_metrics_data[metric_name], metric_name, covariance_plots_dir, run_per_phase_analysis)
    #             print(f"  Correlation/Covariance for {metric_name} completed and saved.")
    #             sys.stdout.flush()
    #         else:
    #             print(f"  Skipping correlation/covariance for metric {metric_name} as it's not in loaded data.")
    #             sys.stdout.flush()

    print(f"DEBUG: Starting main processing loop for PCA/ICA. run_per_phase_analysis: {run_per_phase_analysis}")
    sys.stdout.flush()

    if args.run_pca or args.run_ica:
        if not all_metrics_data:
            print("DEBUG: Skipping PCA/ICA because all_metrics_data is empty.")
            sys.stdout.flush()
        else:
            for metric_name, rounds_or_phases_data in all_metrics_data.items():
                print(f"DEBUG: Processing metric: {metric_name} for PCA/ICA")
                sys.stdout.flush()
                if not rounds_or_phases_data:
                    print(f"DEBUG: No data for metric {metric_name} in rounds_or_phases_data for PCA/ICA, skipping.")
                    sys.stdout.flush()
                    continue
                for round_name, phases_or_metric_df_data in rounds_or_phases_data.items(): # Renamed to avoid conflict
                    print(f"DEBUG: Processing round: {round_name} for metric: {metric_name} for PCA/ICA. run_per_phase_analysis: {run_per_phase_analysis}")
                    sys.stdout.flush()
                    
                    current_data_for_analysis = None # This will hold the DataFrame to be analyzed
                    analysis_label_suffix = ""

                    if run_per_phase_analysis:
                        if not isinstance(phases_or_metric_df_data, dict):
                            print(f"  Skipping round {round_name} for metric {metric_name}: Expected a dictionary of phase data for per-phase PCA/ICA, got {type(phases_or_metric_df_data)}.")
                            sys.stdout.flush()
                            continue # to next round
                        
                        # Loop through phases within the round for per-phase analysis
                        for phase_name, phase_df in phases_or_metric_df_data.items():
                            print(f"DEBUG: Per-phase PCA/ICA: Processing phase: {phase_name} for metric: {metric_name}, round: {round_name}")
                            sys.stdout.flush()
                            if not isinstance(phase_df, pd.DataFrame) or phase_df.empty:
                                print(f"    Skipping phase {phase_name} for metric {metric_name}, round {round_name} due to empty or invalid DataFrame.")
                                sys.stdout.flush()
                                continue # to next phase
                            
                            current_data_for_analysis = phase_df
                            analysis_label_suffix = f"{phase_name}"
                            # Proceed with PCA/ICA for this specific phase_df
                            perform_multivariate_analysis_on_df(current_data_for_analysis, metric_name, analysis_label_suffix, round_name, args, pca_plots_output_dir, pca_tables_output_dir, ica_plots_output_dir, ica_tables_output_dir, comparison_tables_output_dir)

                    else: # Consolidated Analysis (per round)
                        print(f"DEBUG: Consolidated PCA/ICA for metric: {metric_name}, round: {round_name}")
                        sys.stdout.flush()
                        if not isinstance(phases_or_metric_df_data, pd.DataFrame) or phases_or_metric_df_data.empty:
                            print(f"    Skipping consolidated analysis for metric {metric_name}, round {round_name} due to empty or invalid DataFrame. Type: {type(phases_or_metric_df_data)}")
                            sys.stdout.flush()
                            continue # to next round or metric
                        
                        current_data_for_analysis = phases_or_metric_df_data
                        analysis_label_suffix = "consolidated"
                        # Proceed with PCA/ICA for this consolidated round_df
                        perform_multivariate_analysis_on_df(current_data_for_analysis, metric_name, analysis_label_suffix, round_name, args, pca_plots_output_dir, pca_tables_output_dir, ica_plots_output_dir, ica_tables_output_dir, comparison_tables_output_dir)
        
        print("DEBUG: Finished processing all metrics and rounds for PCA/ICA.")
        sys.stdout.flush()

    if not (args.run_pca or args.run_ica or args.run_descriptive_stats or args.run_correlation_covariance or args.compare_pca_ica):
        print("No analysis modules selected to run. Please specify at least one analysis type.")
        sys.stdout.flush()

    print("DEBUG: Script new_main.py finished.")
    sys.stdout.flush()


def perform_multivariate_analysis_on_df(data_df, metric_name, analysis_label, round_name, args, pca_plots_dir, pca_tables_dir, ica_plots_dir, ica_tables_dir, comparison_tables_dir):
    """Helper function to perform PCA and/or ICA on a given DataFrame."""
    print(f"DEBUG: perform_multivariate_analysis_on_df for {metric_name} - {analysis_label} (Round: {round_name}). Shape: {data_df.shape}")
    sys.stdout.flush()

    numeric_df = data_df.select_dtypes(include=np.number)
    numeric_df_cleaned = numeric_df.dropna(axis=0, how='any').dropna(axis=1, how='all')

    if numeric_df_cleaned.empty or numeric_df_cleaned.shape[0] < 2 or numeric_df_cleaned.shape[1] < 1:
        print(f"Skipping PCA/ICA for {metric_name} - {analysis_label} (Round: {round_name}) due to insufficient data after cleaning (Shape: {numeric_df_cleaned.shape}).")
        sys.stdout.flush()
        return

    print(f"Preparing for PCA/ICA for {metric_name} - {analysis_label} (Round: {round_name}). Cleaned Shape: {numeric_df_cleaned.shape}")
    sys.stdout.flush()

    pca_components_df_for_comparison = None
    ica_components_df_for_comparison = None

    # PCA Analysis
    if args.run_pca:
        print(f"Running PCA for {metric_name} ({analysis_label} - Round: {round_name})...")
        sys.stdout.flush()
        pca_results_df, pca_components_df, pca_explained_variance = None, None, None
        try:
            pca_n_components_arg = args.pca_n_components
            if pca_n_components_arg and not isinstance(pca_n_components_arg, int):
                try: pca_n_components_arg = float(pca_n_components_arg) 
                except ValueError: pass # Keep as string if not float for int conversion attempt
            if isinstance(pca_n_components_arg, str): # If still string, try int
                try: pca_n_components_arg = int(pca_n_components_arg)
                except ValueError:
                    print(f"Invalid pca_n_components value: {pca_n_components_arg}. Using default variance threshold.")
                    pca_n_components_arg = None # Fallback to variance threshold if invalid string
            
            pca_results_df, pca_components_df, pca_explained_variance = perform_pca(
                numeric_df_cleaned.copy(), 
                n_components=pca_n_components_arg if isinstance(pca_n_components_arg, int) else None,
                variance_threshold=pca_n_components_arg if isinstance(pca_n_components_arg, float) else args.pca_variance_threshold
            )
            pca_components_df_for_comparison = pca_components_df
            
            base_filename_pca = f"{metric_name}_{analysis_label}_{round_name}_pca"
            if pca_results_df is not None: export_to_csv(pca_results_df, os.path.join(pca_tables_dir, f"{base_filename_pca}_principal_components.csv"))
            if pca_components_df is not None: export_to_csv(pca_components_df, os.path.join(pca_tables_dir, f"{base_filename_pca}_loadings.csv"))
            
            if pca_explained_variance is not None:
                explained_variance_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(len(pca_explained_variance))],
                    'ExplainedVarianceRatio': pca_explained_variance,
                    'CumulativeVarianceRatio': np.cumsum(pca_explained_variance)
                })
                export_to_csv(explained_variance_df, os.path.join(pca_tables_dir, f"{base_filename_pca}_explained_variance.csv"))
            print(f"PCA results saved for {metric_name} ({analysis_label} - Round: {round_name}).")
            sys.stdout.flush()

            if pca_explained_variance is not None and len(pca_explained_variance) > 0:
                plot_pca_scree_plot(pca_explained_variance, np.cumsum(pca_explained_variance), pca_plots_dir, f"{base_filename_pca}_scree_plot.png", f"PCA Scree Plot - {metric_name} ({analysis_label} - {round_name})")
            if pca_results_df is not None and not pca_results_df.empty and pca_results_df.shape[1] >= 2:
                plot_pca_score_plot(pca_results_df, 0, 1, pca_plots_dir, f"{base_filename_pca}_score_plot_pc1_pc2.png", f"PCA Score Plot - {metric_name} ({analysis_label} - {round_name})", None, TENANT_COLORS)
            if pca_components_df is not None and not pca_components_df.empty:
                plot_pca_loadings_heatmap(pca_components_df, pca_plots_dir, f"{base_filename_pca}_loadings_heatmap.png", f"PCA Loadings Heatmap - {metric_name} ({analysis_label} - {round_name})")

        except Exception as e:
            print(f"ERROR during PCA for {metric_name} ({analysis_label} - {round_name}): {e}", file=sys.stderr)
            print(f"Full traceback for PCA error:\\n{traceback.format_exc()}", file=sys.stderr)
            sys.stderr.flush()

    # ICA Analysis
    if args.run_ica:
        print(f"Running ICA for {metric_name} ({analysis_label} - Round: {round_name})...")
        sys.stdout.flush()
        ica_results_df, ica_components_df = None, None
        try:
            ica_results_df, ica_components_df = perform_ica(
                numeric_df_cleaned.copy(), 
                n_components=args.ica_n_components
            )
            ica_components_df_for_comparison = ica_components_df
            
            base_filename_ica = f"{metric_name}_{analysis_label}_{round_name}_ica"
            if ica_results_df is not None: export_to_csv(ica_results_df, os.path.join(ica_tables_dir, f"{base_filename_ica}_independent_components.csv"))
            if ica_components_df is not None: export_to_csv(ica_components_df, os.path.join(ica_tables_dir, f"{base_filename_ica}_unmixing_matrix.csv"))
            print(f"ICA results saved for {metric_name} ({analysis_label} - Round: {round_name}).")
            sys.stdout.flush()

            if ica_results_df is not None and not ica_results_df.empty and ica_results_df.shape[1] >= 2:
                plot_ica_score_plot(ica_results_df, 0, 1, ica_plots_dir, f"{base_filename_ica}_score_plot_ic1_ic2.png", f"ICA Score Plot - {metric_name} ({analysis_label} - {round_name})", None, TENANT_COLORS)
            if ica_components_df is not None and not ica_components_df.empty:
                plot_ica_loadings_heatmap(ica_components_df, ica_plots_dir, f"{base_filename_ica}_loadings_heatmap.png", f"ICA Loadings Heatmap - {metric_name} ({analysis_label} - {round_name})")

        except Exception as e:
            print(f"ERROR during ICA for {metric_name} ({analysis_label} - {round_name}): {e}", file=sys.stderr)
            print(f"Full traceback for ICA error:\\n{traceback.format_exc()}", file=sys.stderr)
            sys.stderr.flush()

    # PCA vs ICA Comparison
    if args.compare_pca_ica and pca_components_df_for_comparison is not None and ica_components_df_for_comparison is not None:
        print(f"Generating PCA vs ICA top features comparison for {metric_name} ({analysis_label} - Round: {round_name})...")
        sys.stdout.flush()
        try:
            top_pca_features = get_top_features_per_component(pca_components_df_for_comparison, args.n_top_features_comparison)
            top_ica_features = get_top_features_per_component(ica_components_df_for_comparison, args.n_top_features_comparison)
            top_pca_features['Method'] = 'PCA'
            top_ica_features['Method'] = 'ICA'
            comparison_df = pd.concat([top_pca_features, top_ica_features], ignore_index=True)
            comparison_df = comparison_df[['Method', 'Component', 'Rank', 'Feature', 'Coefficient']]
            base_filename_comparison = f"{metric_name}_{analysis_label}_{round_name}_pca_ica_top_features_comparison.csv"
            export_to_csv(comparison_df, os.path.join(comparison_tables_dir, base_filename_comparison))
            print(f"PCA vs ICA top features comparison table saved for {metric_name} ({analysis_label} - Round: {round_name}).")
            sys.stdout.flush()
        except Exception as e:
            print(f"Error generating PCA vs ICA comparison for {metric_name} ({analysis_label} - {round_name}): {e}", file=sys.stderr)
            print(f"Full traceback:\\n{traceback.format_exc()}", file=sys.stderr)
            sys.stderr.flush()
    elif args.compare_pca_ica:
        print(f"Skipping PCA vs ICA comparison for {metric_name} ({analysis_label} - {round_name}) as PCA or ICA components are missing.")
        sys.stdout.flush()
