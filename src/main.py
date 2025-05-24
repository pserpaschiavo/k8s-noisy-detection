"""
Main entry point for the k8s-noisy-detection analysis pipeline.

This refactored version uses modular analysis runners to improve maintainability
and reduce the complexity of the main function.
"""

import argparse
import logging
import os
import sys
import traceback

# Import centralized utilities and exceptions
from .utils.common import plt, pd, np
try:
    from .utils.exceptions import (
        handle_critical_error, handle_recoverable_error,
        ConfigurationError, DataLoadingError
    )
except ImportError:
    # Fallback error handling if exceptions module fails
    import logging
    def handle_critical_error(error, operation):
        logging.critical(f"Critical error during {operation}: {str(error)}")
    def handle_recoverable_error(error, operation, default_return=None):
        logging.warning(f"Recoverable error during {operation}: {str(error)}")
        return default_return
    class ConfigurationError(Exception): pass
    class DataLoadingError(Exception): pass

# Import configuration and core modules
from .config import TENANT_COLORS, METRICS_CONFIG
from .data.loader import load_experiment_data

# Import modular analysis runners
from .analysis.analysis_runners import (
    run_descriptive_statistics_analysis,
    run_correlation_covariance_analysis,
    run_causality_analysis,
    run_similarity_analysis,
    run_multivariate_analysis,
    run_root_cause_analysis
)

# Import individual analysis functions for backward compatibility
from .analysis.descriptive_statistics import calculate_descriptive_statistics
from .analysis.correlation_covariance import (
    calculate_inter_tenant_correlation_per_metric,
    calculate_inter_tenant_covariance_per_metric
)
from .analysis.causality import perform_sem_analysis, plot_sem_path_diagram, plot_sem_fit_indices
from .analysis.similarity import (
    calculate_pairwise_distance_correlation,
    calculate_pairwise_cosine_similarity,
    calculate_pairwise_mutual_information,
    plot_distance_correlation_heatmap,
    plot_cosine_similarity_heatmap,
    plot_mutual_information_heatmap
)
from .analysis.multivariate import (
    perform_pca, perform_ica, get_top_features_per_component,
    perform_kpca, perform_tsne
)
from .analysis.root_cause import perform_complete_root_cause_analysis
from .visualization.plots import (
    plot_correlation_heatmap,
    plot_covariance_heatmap,
    plot_descriptive_stats_boxplot,
    plot_descriptive_stats_lineplot,
    plot_pca_explained_variance,
    plot_pca_biplot,
    plot_pca_loadings_heatmap,
    plot_ica_scatter,
    plot_ica_components_heatmap
)
from .data.io_utils import export_to_csv

# Function definitions (parse_arguments, setup_output_directories, load_and_preprocess_data, etc.)
# are assumed to be defined in this file or correctly imported if they were meant to be elsewhere.

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run data analysis pipeline for k8s noisy detection.")
    parser.add_argument("--data-dir", required=True, help="Root directory containing the experimental data.")
    parser.add_argument("--output-dir", required=True, help="Directory to save results and plots.")
    parser.add_argument("--metrics-config", default="src/config.py", help="Path to the metrics configuration file.")
    parser.add_argument("--selected-metrics", nargs='*', help="List of specific metrics to process (e.g., cpu_usage memory_usage). Processes all if not specified.")
    parser.add_argument("--start-time", help="Global start time for analysis (YYYY-MM-DDTHH:MM:SS or relative like -60s, -10m, -1h from first event).")
    parser.add_argument("--end-time", help="Global end time for analysis (YYYY-MM-DDTHH:MM:SS or relative like +60s, +10m, +1h from last event).")
    parser.add_argument("--normalization-scope", choices=['phase', 'round', 'experiment'], default='phase', help="Scope for time normalization.")
    parser.add_argument("--time-normalization-target", nargs='*', default=['AttackStart', 'AttackEnd'], help="Event names to normalize time against.")
    parser.add_argument("--feature-scaling-method", choices=['minmax', 'standard', 'none'], default='minmax', help="Method for feature scaling.")
    parser.add_argument("--run-per-phase", action='store_true', help="Run analysis per phase instead of consolidated per round.")
    
    parser.add_argument("--run-descriptive-stats", action="store_true", help="Run descriptive statistics module.")
    parser.add_argument("--run-correlation-covariance", action="store_true", help="Run correlation and covariance analysis module.")
    parser.add_argument("--run-causality", action="store_true", help="Run causal analysis module.")
    parser.add_argument("--run-similarity", action="store_true", help="Run similarity analysis module.")
    parser.add_argument("--run-root-cause", action="store_true", help="Run root cause analysis module.")
    parser.add_argument("--correlation-methods", nargs='*', default=['pearson', 'spearman'], help="Methods for correlation (pearson, kendall, spearman).")
    parser.add_argument("--run-pca", action="store_true", help="Run PCA module.")
    parser.add_argument("--pca-n-components", default=None, help="Number of principal components (int) or variance threshold (float, e.g., 0.95).")
    parser.add_argument("--pca-variance-threshold", type=float, default=0.95, help="Variance threshold for PCA if n_components is not an int.")
    parser.add_argument("--run-ica", action="store_true", help="Run ICA module.")
    parser.add_argument("--ica-n-components", type=int, default=None, help="Number of independent components.")
    # Non-linear multivariate analysis
    parser.add_argument("--run-kpca", action="store_true", help="Run KernelPCA module.")
    parser.add_argument("--kpca-kernel", choices=['linear','poly','rbf','sigmoid'], default='rbf', help="Kernel to use for KernelPCA.")
    parser.add_argument("--kpca-gamma", type=float, default=None, help="Gamma parameter for KernelPCA.")
    parser.add_argument("--run-tsne", action="store_true", help="Run t-SNE module.")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0, help="Perplexity for t-SNE.")
    parser.add_argument("--tsne-random-state", type=int, default=42, help="Random state for t-SNE.")

    # SEM Analysis parameters
    sem_group = parser.add_argument_group('SEM Analysis')
    sem_group.add_argument("--sem-model-spec", type=str,
                           help="SEM model specification string in semopy syntax (e.g., 'Y ~ X1 + X2; X1 ~~ X2')")
    sem_group.add_argument("--sem-exog-vars", nargs='*',
                           help="List of exogenous variable names for SEM")
                           
    # Root Cause Analysis parameters
    rca_group = parser.add_argument_group('Root Cause Analysis')
    rca_group.add_argument("--rca-max-order", type=int, default=3,
                           help="Maximum order for cascading impact analysis (default: 3)")
    rca_group.add_argument("--rca-impact-threshold", type=float, default=0.5,
                           help="Threshold for determining significant impact (default: 0.5)")

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

    # Root Cause Analysis directories
    root_cause_plots_dir = os.path.join(plots_dir, "root_cause")
    root_cause_tables_dir = os.path.join(tables_dir, "root_cause")

    multivariate_tables_dir = os.path.join(tables_dir, "multivariate")
    pca_tables_output_dir = os.path.join(multivariate_tables_dir, "pca")
    ica_tables_output_dir = os.path.join(multivariate_tables_dir, "ica")
    comparison_tables_output_dir = os.path.join(multivariate_tables_dir, "comparison")

    # Create all directories
    for path in [plots_dir, tables_dir, desc_stats_plots_dir, desc_stats_tables_dir,
                 correlation_plots_dir, correlation_tables_dir, covariance_plots_dir, covariance_tables_dir,
                 multivariate_plots_dir, pca_plots_output_dir, ica_plots_output_dir,
                 multivariate_tables_dir, pca_tables_output_dir, ica_tables_output_dir, comparison_tables_output_dir,
                 root_cause_plots_dir, root_cause_tables_dir]:
        os.makedirs(path, exist_ok=True)

    # Return the paths needed by the main function
    return (plots_dir, tables_dir, 
            pca_tables_output_dir, ica_tables_output_dir, comparison_tables_output_dir,
            pca_plots_output_dir, ica_plots_output_dir,
            desc_stats_plots_dir, desc_stats_tables_dir, # Added for descriptive stats
            correlation_plots_dir, correlation_tables_dir, # Added for correlation
            covariance_plots_dir, covariance_tables_dir, # Added for covariance
            root_cause_plots_dir, root_cause_tables_dir # Added for root cause
           ) 

def load_and_preprocess_data(data_dir, metrics_config_path, selected_metrics_list, 
                               start_time_str, end_time_str, normalization_scope, 
                               time_normalization_target_list, feature_scaling_method_name, 
                               run_per_phase_flag):
    logging.debug("Starting load_and_preprocess_data.")

    metrics_processing_config = METRICS_CONFIG 
    logging.debug(f"Metrics config loaded. {len(metrics_processing_config)} metrics defined.")

    if not selected_metrics_list: # If None or empty, process all configured metrics
        selected_metrics_list = list(metrics_processing_config.keys())
    logging.debug(f"Selected metrics for processing: {selected_metrics_list}")

    all_metrics_data_dict = {}
    logging.debug(f"Loading data. Data directory: {data_dir}, Per phase: {run_per_phase_flag}, Metrics: {selected_metrics_list}")
    
    # Call load_experiment_data with the correct arguments
    # tenants, phases, and rounds arguments are omitted to use their defaults (load all).
    all_metrics_data_dict = load_experiment_data(
        experiment_dir=data_dir,
        metrics=selected_metrics_list,
        group_by_phase=run_per_phase_flag
    )
    
    logging.debug(f"Raw data loaded for {len(all_metrics_data_dict)} metrics.")

    logging.debug("load_and_preprocess_data finished.")
    return all_metrics_data_dict, selected_metrics_list, run_per_phase_flag


def main():
    logging.debug("Script new_main.py starting...")
    args = parse_arguments()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.debug(f"Parsed arguments: {args}")

    # Validate SEM parameters (single validation)
    if args.run_causality and not args.sem_model_spec:
        logging.error("SEM model spec (--sem-model-spec) is required when --run-causality is set.")
        sys.exit(1)

    try:
        # Unpack all 15 values returned by the corrected setup_output_directories
        plots_dir, tables_dir, \
        pca_tables_output_dir, ica_tables_output_dir, comparison_tables_output_dir, \
        pca_plots_output_dir, ica_plots_output_dir, \
        desc_stats_plots_dir, desc_stats_tables_dir, \
        correlation_plots_dir, correlation_tables_dir, \
        covariance_plots_dir, covariance_tables_dir, \
        root_cause_plots_dir, root_cause_tables_dir = setup_output_directories(args.output_dir)

        logging.info("Output directories setup complete.")
        logging.debug(f"General plots_dir: {plots_dir}")
        logging.debug(f"General tables_dir: {tables_dir}")
        logging.debug(f"PCA plots dir: {pca_plots_output_dir}, PCA tables dir: {pca_tables_output_dir}")
        logging.debug(f"ICA plots dir: {ica_plots_output_dir}, ICA tables dir: {ica_tables_output_dir}")
        logging.debug(f"Comparison tables dir: {comparison_tables_output_dir}")
        logging.debug(f"Descriptive Stats plots dir: {desc_stats_plots_dir}, tables dir: {desc_stats_tables_dir}")
        logging.debug(f"Correlation plots dir: {correlation_plots_dir}, tables dir: {correlation_tables_dir}")
        logging.debug(f"Covariance plots dir: {covariance_plots_dir}, tables dir: {covariance_tables_dir}")
        logging.debug(f"Root Cause plots dir: {root_cause_plots_dir}, tables dir: {root_cause_tables_dir}")
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
            args.metrics_config,
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

    # Descriptive Statistics
    if args.run_descriptive_stats:
        print("DEBUG: Running descriptive statistics module...")
        for metric_name, rounds_data in all_metrics_data.items():
            # Combine data across rounds and phases
            dfs = []
            if run_per_phase_analysis:
                for rd in rounds_data.values():
                    if isinstance(rd, dict):
                        dfs.extend([df for df in rd.values() if isinstance(df, pd.DataFrame)])
                    elif isinstance(rd, pd.DataFrame):
                        dfs.append(rd)
            else:
                dfs = [df for df in rounds_data.values() if isinstance(df, pd.DataFrame)]
            if not dfs:
                print(f"DEBUG: No data for descriptive stats of {metric_name}, skipping.")
                continue
            metric_df = pd.concat(dfs, ignore_index=True)
            # Calculate descriptive statistics
            stats_df = calculate_descriptive_statistics(metric_df, metric_column='value')
            # Export CSV
            csv_out = os.path.join(desc_stats_tables_dir, f"{metric_name}_descriptive_stats.csv")
            export_to_csv(stats_df, csv_out)
            # Plots
            # Boxplot
            plot_descriptive_stats_boxplot(
                metric_df, metric_name, 'value',
                title=f"{metric_name} Distribution",
                output_dir=desc_stats_plots_dir,
                filename=f"{metric_name}_boxplot.png"
            )
            # Lineplot (use sample index if no time column)
            metric_df_line = metric_df.copy()
            metric_df_line.index.name = 'sample'
            plot_descriptive_stats_lineplot(
                metric_df_line, metric_name, 'value',
                title=f"{metric_name} Over Samples",
                output_dir=desc_stats_plots_dir,
                filename=f"{metric_name}_lineplot.png"
            )
            print(f"DEBUG: Descriptive stats completed for {metric_name}.")
            sys.stdout.flush()
    
    # Correlation and Covariance Analysis
    if args.run_correlation_covariance:
        print("DEBUG: Running correlation and covariance analysis module...")
        for metric_name, rounds_data in all_metrics_data.items():
            # Iterate through rounds
            for round_key, rd in rounds_data.items():
                # Determine dataframes based on per-phase flag
                phase_dicts = {}
                if run_per_phase_analysis and isinstance(rd, dict):
                    phase_dicts = rd
                elif not run_per_phase_analysis and isinstance(rd, pd.DataFrame):
                    phase_dicts = {None: rd}
                else:
                    continue
                for phase_key, df_metric in phase_dicts.items():
                    if df_metric.empty:
                        continue
                    # Calculate inter-tenant correlation and covariance
                    corr_df = calculate_inter_tenant_correlation_per_metric(df_metric, method=args.correlation_methods[0], time_col='datetime')
                    cov_df = calculate_inter_tenant_covariance_per_metric(df_metric, time_col='datetime')
                    # Export and plot if not empty
                    label = f"{round_key}" + (f"_{phase_key}" if phase_key else "")
                    if not corr_df.empty:
                        # CSV
                        corr_csv = os.path.join(correlation_tables_dir, f"{metric_name}_{label}_correlation.csv")
                        export_to_csv(corr_df, corr_csv)
                        # Heatmap
                        plot_correlation_heatmap(corr_df,
                                               title=f"{metric_name} Inter-Tenant Correlation",
                                               output_dir=correlation_plots_dir,
                                               filename=f"{metric_name}_{label}_correlation_heatmap.png")
                    if not cov_df.empty:
                        cov_csv = os.path.join(covariance_tables_dir, f"{metric_name}_{label}_covariance.csv")
                        export_to_csv(cov_df, cov_csv)
                        plot_covariance_heatmap(cov_df,
                                               title=f"{metric_name} Inter-Tenant Covariance",
                                               output_dir=covariance_plots_dir,
                                               filename=f"{metric_name}_{label}_covariance_heatmap.png")
        print("DEBUG: Correlation and covariance analysis completed.")
        sys.stdout.flush()

    # Causality Analysis
    if args.run_causality:
        logging.info("Running causal analysis module...")
        # Setup SEM-specific directories
        sem_plots_dir = os.path.join(plots_dir, 'causality')
        sem_tables_dir = os.path.join(tables_dir, 'causality')
        os.makedirs(sem_plots_dir, exist_ok=True)
        os.makedirs(sem_tables_dir, exist_ok=True)

        if not args.sem_model_spec:
            logging.error("SEM model spec (--sem-model-spec) is required to run causal analysis. Skipping SEM block.")
        else:
            for metric_name, rounds_data in all_metrics_data.items():
                for round_key, rd in rounds_data.items():
                    phase_dicts = {}
                    if run_per_phase_analysis and isinstance(rd, dict):
                        phase_dicts = rd
                    elif not run_per_phase_analysis and isinstance(rd, pd.DataFrame):
                        phase_dicts = {None: rd}
                    else:
                        continue
                    for phase_key, df_metric in phase_dicts.items():
                        if df_metric.empty:
                            continue
                        # Prepare data for SEM: select numeric columns including exogenous/endogenous
                        sem_data = df_metric.select_dtypes(include=[np.number]).dropna()
                        try:
                            sem_results = perform_sem_analysis(
                                sem_data,
                                args.sem_model_spec,
                                exog_vars=args.sem_exog_vars or []
                            )
                        except Exception as e:
                            logging.error(f"Error during SEM fit for {metric_name} {round_key} {phase_key}: {e}")
                            continue
                        # Export estimates and stats
                        estimates_df = sem_results.get('estimates')
                        stats_dict = sem_results.get('stats')
                        stats_df = pd.DataFrame([stats_dict]) if stats_dict else pd.DataFrame()

                        est_csv = os.path.join(sem_tables_dir, f"{metric_name}_{round_key}_{phase_key}_sem_estimates.csv")
                        stats_csv = os.path.join(sem_tables_dir, f"{metric_name}_{round_key}_{phase_key}_sem_stats.csv")
                        export_to_csv(estimates_df, est_csv)
                        export_to_csv(stats_df, stats_csv)

                        # Plot path diagram and fit indices
                        path_file = f"{metric_name}_{round_key}_{phase_key}_sem_path.png"
                        fit_file = f"{metric_name}_{round_key}_{phase_key}_sem_fit.png"
                        plot_sem_path_diagram(
                            sem_results,
                            title=f"SEM Path Diagram for {metric_name} {round_key} {phase_key}",
                            output_dir=sem_plots_dir,
                            filename=path_file,
                            metric_name=metric_name,
                            round_name=round_key,
                            phase_name=phase_key
                        )
                        plot_sem_fit_indices(
                            sem_results,
                            title=f"SEM Fit Indices for {metric_name} {round_key} {phase_key}",
                            output_dir=sem_plots_dir,
                            filename=fit_file,
                            metric_name=metric_name,
                            round_name=round_key,
                            phase_name=phase_key
                        )
            logging.info("Causal analysis completed.")
            sys.stdout.flush()

    # Similarity Analysis
    if args.run_similarity:
        logging.info("Running similarity analysis module...")
        sim_plots_dir = os.path.join(plots_dir, 'similarity')
        sim_tables_dir = os.path.join(tables_dir, 'similarity')
        os.makedirs(sim_plots_dir, exist_ok=True)
        os.makedirs(sim_tables_dir, exist_ok=True)

        time_col_default = 'datetime' 
        metric_col_default = 'value'
        group_col_default = 'tenant'

        for metric_name, rounds_data in all_metrics_data.items():
            for round_key, rd_data in rounds_data.items(): # Renamed rd to rd_data to avoid conflict with phase_data
                phase_data_map = {}
                if run_per_phase_analysis and isinstance(rd_data, dict):
                    phase_data_map = rd_data
                elif not run_per_phase_analysis and isinstance(rd_data, pd.DataFrame):
                    phase_data_map = {None: rd_data}
                else:
                    logging.debug(f"Skipping similarity for {metric_name} {round_key} due to unexpected data structure: {type(rd_data)}")
                    continue
                
                for phase_key, df_metric_current in phase_data_map.items(): # Renamed df_metric to df_metric_current
                    if not isinstance(df_metric_current, pd.DataFrame) or df_metric_current.empty:
                        logging.debug(f"Skipping similarity for {metric_name} {round_key} {phase_key if phase_key else 'consolidated'} due to empty/invalid DataFrame.")
                        continue

                    # Determine actual column names to use
                    current_time_col = time_col_default
                    if time_col_default not in df_metric_current.columns and 'timestamp' in df_metric_current.columns:
                        current_time_col = 'timestamp'
                        logging.info(f"Using 'timestamp' as time_col for {metric_name} {round_key} {phase_key if phase_key else 'consolidated'}")
                    
                    current_metric_col = metric_col_default
                    current_group_col = group_col_default

                    if not all(col in df_metric_current.columns for col in [current_time_col, current_metric_col, current_group_col]):
                        logging.warning(f"Missing one or more required columns ({current_time_col}, {current_metric_col}, {current_group_col}) in DataFrame for {metric_name} {round_key} {phase_key if phase_key else 'consolidated'}. Skipping similarity for this entry.")
                        continue

                    logging.debug(f"Processing similarity for {metric_name}, Round: {round_key}, Phase: {phase_key if phase_key else 'consolidated'}")
                    
                    distance_corr_df = calculate_pairwise_distance_correlation(
                        df_metric_current, time_col=current_time_col, metric_col=current_metric_col, group_col=current_group_col
                    )
                    cosine_sim_df = calculate_pairwise_cosine_similarity(
                        df_metric_current, time_col=current_time_col, metric_col=current_metric_col, group_col=current_group_col
                    )
                    mutual_info_df = calculate_pairwise_mutual_information(
                        df_metric_current, time_col=current_time_col, metric_col=current_metric_col, group_col=current_group_col
                    )
                    
                    label_elements = [metric_name, round_key]
                    if phase_key:
                        label_elements.append(phase_key)
                    label = "_".join(str(elem) for elem in label_elements if elem is not None)

                    if not distance_corr_df.empty:
                        dist_csv = os.path.join(sim_tables_dir, f"{label}_distance_correlation.csv")
                        export_to_csv(distance_corr_df, dist_csv)
                        plot_distance_correlation_heatmap(
                            distance_corr_df,
                            title=f"{metric_name} Distance Correlation ({round_key}{f'-{phase_key}' if phase_key else ''})",
                            output_dir=sim_plots_dir,
                            filename=f"{label}_distance_correlation.png",
                            tables_dir=sim_tables_dir
                        )
                    if not cosine_sim_df.empty:
                        cos_csv = os.path.join(sim_tables_dir, f"{label}_cosine_similarity.csv")
                        export_to_csv(cosine_sim_df, cos_csv)
                        plot_cosine_similarity_heatmap(
                            cosine_sim_df,
                            title=f"{metric_name} Cosine Similarity ({round_key}{f'-{phase_key}' if phase_key else ''})",
                            output_dir=sim_plots_dir,
                            filename=f"{label}_cosine_similarity.png",
                            tables_dir=sim_tables_dir
                        )
                    if not mutual_info_df.empty:
                        mi_csv = os.path.join(sim_tables_dir, f"{label}_mutual_information.csv")
                        export_to_csv(mutual_info_df, mi_csv)
                        plot_mutual_information_heatmap(
                            mutual_info_df,
                            title=f"{metric_name} Mutual Information ({round_key}{f'-{phase_key}' if phase_key else ''})",
                            output_dir=sim_plots_dir,
                            filename=f"{label}_mutual_information.png",
                            tables_dir=sim_tables_dir
                        )
        logging.info("Similarity analysis completed.")
        sys.stdout.flush()

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

    # Non-linear Multivariate Analysis
    if args.run_kpca:
        logging.info("Running KernelPCA module...")
        # Combine dataframes as in PCA
        data_list = []
        for m, rounds_data in all_metrics_data.items():
            for rd in rounds_data.values():
                if isinstance(rd, dict):
                    data_list.extend([df for df in rd.values() if isinstance(df, pd.DataFrame)])
                elif isinstance(rd, pd.DataFrame):
                    data_list.append(rd)
        if data_list:
            df_all = pd.concat(data_list, ignore_index=True).select_dtypes(include=[np.number]).dropna()
            kpca_df = perform_kpca(df_all, n_components=None if args.kpca_kernel!='precomputed' else None,
                                   kernel=args.kpca_kernel, gamma=args.kpca_gamma)
            # Export
            kpca_csv = os.path.join(comparison_tables_output_dir, 'kpca_components.csv')
            export_to_csv(kpca_df, kpca_csv)
            # Scatter first two components
            fig, ax = plt.subplots()
            ax.scatter(kpca_df.iloc[:,0], kpca_df.iloc[:,1], alpha=0.7)
            ax.set_xlabel(kpca_df.columns[0]); ax.set_ylabel(kpca_df.columns[1])
            ax.set_title('KernelPCA Scatter')
            os.makedirs(pca_plots_output_dir, exist_ok=True)
            fig_path = os.path.join(pca_plots_output_dir, 'kpca_scatter.png')
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
    if args.run_tsne:
        logging.info("Running t-SNE module...")
        data_list = []
        for m, rounds_data in all_metrics_data.items():
            for rd in rounds_data.values():
                if isinstance(rd, dict):
                    data_list.extend([df for df in rd.values() if isinstance(df, pd.DataFrame)])
                elif isinstance(rd, pd.DataFrame):
                    data_list.append(rd)
        if data_list:
            df_all = pd.concat(data_list, ignore_index=True).select_dtypes(include=[np.number]).dropna()
            tsne_df = perform_tsne(df_all, n_components=2, perplexity=args.tsne_perplexity, random_state=args.tsne_random_state)
            tsne_csv = os.path.join(comparison_tables_output_dir, 'tsne_components.csv')
            export_to_csv(tsne_df, tsne_csv)
            fig, ax = plt.subplots()
            ax.scatter(tsne_df.iloc[:,0], tsne_df.iloc[:,1], alpha=0.7)
            ax.set_xlabel(tsne_df.columns[0]); ax.set_ylabel(tsne_df.columns[1])
            ax.set_title('t-SNE Scatter')
            os.makedirs(pca_plots_output_dir, exist_ok=True)
            fig_path = os.path.join(pca_plots_output_dir, 'tsne_scatter.png')
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

    # Root Cause Analysis Module
    if args.run_root_cause:
        logging.info("Running root cause analysis module...")
        # For RCA, we need the tenant names and impact matrix from correlation analysis
        for metric_name, rounds_data in all_metrics_data.items():
            for round_key, rd in rounds_data.items():
                phase_dicts = {}
                if run_per_phase_analysis and isinstance(rd, dict):
                    phase_dicts = rd
                elif not run_per_phase_analysis and isinstance(rd, pd.DataFrame):
                    phase_dicts = {None: rd}
                else:
                    continue
                    
                for phase_key, df_metric in phase_dicts.items():
                    if df_metric.empty:
                        continue
                        
                    # Extract tenant names from the dataframe
                    if 'tenant' in df_metric.columns:
                        tenant_names = df_metric['tenant'].unique().tolist()
                    else:
                        # Try to infer tenant names from column names if there's no tenant column
                        try:
                            tenant_columns = [col for col in df_metric.columns if col not in ['datetime', 'timestamp', 'value', 'metric']]
                            if tenant_columns:
                                tenant_names = tenant_columns
                            else:
                                logging.warning(f"Could not determine tenant names for {metric_name} {round_key} {phase_key}")
                                continue
                        except Exception as e:
                            logging.error(f"Error determining tenant names: {e}")
                            continue
                    
                    # Calculate impact matrix from correlation matrix or create one if needed
                    try:
                        corr_df = calculate_inter_tenant_correlation_per_metric(df_metric, method='pearson', time_col='datetime')
                        impact_matrix = np.abs(corr_df.values)  # Use absolute correlation values for impact
                        
                        # Apply the threshold from the command line args
                        impact_matrix = np.where(impact_matrix < args.rca_impact_threshold, 0, impact_matrix)
                        
                        # Create result directory name
                        result_dir = f"{round_key}" + (f"_{phase_key}" if phase_key else "")
                        output_dir = os.path.join(root_cause_plots_dir, metric_name, result_dir)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Perform root cause analysis
                        logging.info(f"Performing root cause analysis for {metric_name} {round_key} {phase_key if phase_key else 'consolidated'}")
                        rca_results = perform_complete_root_cause_analysis(
                            impact_matrix, 
                            df_metric, 
                            tenant_names, 
                            output_dir
                        )
                        
                        # Save primary results to CSV
                        if rca_results['confidence_ranking']:
                            conf_df = pd.DataFrame({
                                'tenant': list(rca_results['confidence_ranking'].keys()),
                                'confidence': [data['overall_confidence'] for data in rca_results['confidence_ranking'].values()],
                                'normalized_impact': [data['normalized_impact'] for data in rca_results['confidence_ranking'].values()],
                                'orders_involved': [str(data['orders_involved']) for data in rca_results['confidence_ranking'].values()]
                            })
                            
                            conf_csv = os.path.join(root_cause_tables_dir, f"{metric_name}_{result_dir}_root_cause_confidence.csv")
                            export_to_csv(conf_df, conf_csv)
                            logging.info(f"Root cause analysis results saved to {conf_csv}")
                            
                    except Exception as e:
                        logging.error(f"Error in root cause analysis for {metric_name} {round_key} {phase_key if phase_key else 'consolidated'}: {e}")
                        logging.debug(f"Full traceback: {traceback.format_exc()}")
                        
        logging.info("Root cause analysis completed.")
        sys.stdout.flush()

    if not (args.run_pca or args.run_ica or args.run_descriptive_stats or args.run_correlation_covariance or 
            args.compare_pca_ica or args.run_kpca or args.run_tsne or args.run_root_cause):
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
                plot_pca_explained_variance(pca_explained_variance, np.cumsum(pca_explained_variance), 
                                          title="PCA Explained Variance", 
                                          output_dir=pca_plots_dir, 
                                          filename=f"{base_filename_pca}_explained_variance.png", 
                                          metric_name=metric_name,
                                          round_name=round_name,
                                          phase_name=analysis_label)
            if pca_results_df is not None and not pca_results_df.empty and pca_results_df.shape[1] >= 2:
                plot_pca_biplot(pca_results_df, pca_components_df, 
                              x_component=1, y_component=2,
                              title="PCA Biplot", 
                              output_dir=pca_plots_dir, 
                              filename=f"{base_filename_pca}_biplot_pc1_pc2.png",
                              metric_name=metric_name,
                              round_name=round_name,
                              phase_name=analysis_label)
            if pca_components_df is not None and not pca_components_df.empty:
                plot_pca_loadings_heatmap(pca_components_df, 
                                        title="PCA Loadings Heatmap", 
                                        output_dir=pca_plots_dir, 
                                        filename=f"{base_filename_pca}_loadings_heatmap.png",
                                        metric_name=metric_name,
                                        round_name=round_name,
                                        phase_name=analysis_label)

        except Exception as e:
            print(f"ERROR during PCA for {metric_name} ({analysis_label} - {round_name}): {e}", file=sys.stderr)
            print(f"Full traceback for PCA error:\\n{traceback.format_exc()}", file=sys.stderr)
            sys.stderr.flush()

    # ICA Analysis
    if args.run_ica:
        print(f"Running ICA for {metric_name} ({analysis_label} - Round: {round_name})...")
        sys.stdout.flush()
        ica_results_df, ica_components_df, ica_mixing_df = None, None, None
        try:
            ica_results_df, ica_components_df, ica_mixing_df = perform_ica(
                numeric_df_cleaned.copy(), 
                n_components=args.ica_n_components
            )
            ica_components_df_for_comparison = ica_components_df
            
            base_filename_ica = f"{metric_name}_{analysis_label}_{round_name}_ica"
            if ica_results_df is not None: export_to_csv(ica_results_df, os.path.join(ica_tables_dir, f"{base_filename_ica}_independent_components.csv"))
            if ica_components_df is not None: export_to_csv(ica_components_df, os.path.join(ica_tables_dir, f"{base_filename_ica}_unmixing_matrix.csv"))
            if ica_mixing_df is not None: export_to_csv(ica_mixing_df, os.path.join(ica_tables_dir, f"{base_filename_ica}_mixing_matrix.csv"))
            print(f"ICA results saved for {metric_name} ({analysis_label} - Round: {round_name}).")
            sys.stdout.flush()

            if ica_results_df is not None and not ica_results_df.empty and ica_results_df.shape[1] >= 2:
                plot_ica_scatter(ica_results_df, 
                               x_component=1, y_component=2,
                               title="ICA Scatter Plot", 
                               output_dir=ica_plots_dir, 
                               filename=f"{base_filename_ica}_scatter_plot_ic1_ic2.png",
                               metric_name=metric_name,
                               round_name=round_name,
                               phase_name=analysis_label)
            if ica_components_df is not None and not ica_components_df.empty:
                plot_ica_components_heatmap(ica_components_df, 
                                          title="ICA Components Heatmap", 
                                          output_dir=ica_plots_dir, 
                                          filename=f"{base_filename_ica}_components_heatmap.png",
                                          metric_name=metric_name,
                                          round_name=round_name,
                                          phase_name=analysis_label)

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
