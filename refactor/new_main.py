import os
import sys

# Add project root to sys.path to allow direct execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools  # Added import
import gc  # For garbage collection

"""
New main script for the refactored experiment analysis pipeline.
"""

# Refactored modules
from refactor.data_handling.loader import load_experiment_data
from refactor.data_handling.save_results import export_to_csv, save_figure  # Corrected import
from refactor.analysis_modules.correlation_covariance import (
    calculate_correlation_matrix,  # Corrected import name
    calculate_covariance_matrix,  # Not used in this basic setup yet
    calculate_inter_tenant_correlation_per_metric,  # Used for correlation analysis
    calculate_inter_tenant_covariance_per_metric  # Not used in this basic setup yet
)
from refactor.analysis_modules.descritive_statistics import calculate_descriptive_statistics
from refactor.analysis_modules.multivariate_exploration import (
    perform_pca, perform_ica, get_top_features_per_component
)
from refactor.analysis_modules.similarity import (
    calculate_pairwise_distance_correlation,
    plot_distance_correlation_heatmap,
    calculate_pairwise_cosine_similarity,
    plot_cosine_similarity_heatmap,
    calculate_pairwise_dtw_distance,
    plot_dtw_distance_heatmap,
    calculate_pairwise_mutual_information,
    plot_mutual_information_heatmap
)
# Added import for causal analysis modules
from refactor.analysis_modules.causality import (
    perform_sem_analysis, create_sem_model_from_correlation, 
    plot_sem_path_diagram, plot_sem_fit_indices, plot_sem_coefficient_heatmap,
    calculate_transfer_entropy, calculate_pairwise_transfer_entropy,
    plot_transfer_entropy_heatmap, plot_transfer_entropy_network,
    calculate_pairwise_ccm, plot_ccm_convergence, summarize_ccm_results,
    plot_ccm_causality_heatmap, calculate_pairwise_granger_causality,
    plot_granger_causality_heatmap, plot_granger_causality_network
)
# Corrected import for plot_correlation_heatmap
from refactor.visualization.new_plots import (
    plot_correlation_heatmap, plot_covariance_heatmap,
    plot_descriptive_stats_lineplot, plot_descriptive_stats_boxplot, plot_descriptive_stats_catplot_mean,
    # PCA plots
    plot_pca_explained_variance,
    plot_pca_biplot,
    plot_pca_loadings_heatmap,
    # ICA plots
    plot_ica_components_heatmap,
    plot_ica_time_series,
    plot_ica_scatter
)
from refactor.data_handling.new_time_normalization import add_experiment_elapsed_time  # Added import

# Existing pipeline modules (will be gradually replaced or integrated)
from refactor.new_config import (
    DEFAULT_DATA_DIR, DEFAULT_METRICS, METRIC_DISPLAY_NAMES,
    VISUALIZATION_CONFIG, TENANT_COLORS, PHASE_DISPLAY_NAMES  # Adicionado PHASE_DISPLAY_NAMES
)
# Add other necessary imports from pipeline.config or other modules as needed
from refactor.utils.figure_management import close_all_figures


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Refactored pipeline for experiment analysis.')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Directory with the experiment data')
    parser.add_argument('--output-dir', type=str, default='output_refactored',
                        help='Directory to save the results')
    parser.add_argument('--tenants', type=str, nargs='+',
                        help='Specific tenant(s) to analyze')
    parser.add_argument('--metrics', type=str, nargs='+', default=DEFAULT_METRICS,
                        help='Specific metric(s) to analyze')
    parser.add_argument('--rounds', type=str, nargs='+',
                        help='Specific round(s) to analyze')
    # Add more arguments as functionality is integrated
    parser.add_argument('--run-correlation', action='store_true', help='Run correlation analysis')
    parser.add_argument('--correlation-methods', type=str, nargs='+', default=['pearson'],
                        choices=['pearson', 'spearman', 'kendall'],
                        help='One or more correlation methods to run. E.g., pearson spearman. Defaults to [\'pearson\'] if --run-correlation is set and this is not provided. Overridden by --run-all-correlation-methods.')
    parser.add_argument('--run-all-correlation-methods', action='store_true',
                        help='Run correlation analysis for all methods (pearson, spearman, kendall). Overrides --correlation-methods.')

    # Arguments for Covariance Analysis
    parser.add_argument('--run-covariance', action='store_true', help='Run covariance analysis')
    # Covariance typically doesn't have multiple "methods" like correlation,
    # but adding for consistency in argument structure if variations arise.
    # For now, it's a simple flag to run it or not.

    # Argument for Descriptive Statistics Analysis
    parser.add_argument('--run-descriptive-stats', action='store_true', help='Run descriptive statistics analysis.')  # Added argument

    # Arguments for PCA
    parser.add_argument('--run-pca', action='store_true', help="Run Principal Component Analysis.")
    parser.add_argument('--pca-n-components', type=str, default=None, help="Number of PCA components (int) or variance explained (float, e.g., 0.95).")
    parser.add_argument('--pca-variance-threshold', type=float, default=None, help="PCA variance threshold to select number of components (e.g., 0.95 for 95%%).")

    # Arguments for ICA
    parser.add_argument('--run-ica', action='store_true', help="Run Independent Component Analysis.")
    parser.add_argument('--ica-n-components', type=int, default=None, help="Number of ICA components.")
    parser.add_argument('--ica-max-iter', type=int, default=200, help="Maximum number of iterations for ICA.")

    # Argument for PCA vs ICA comparison
    parser.add_argument('--compare-pca-ica', action='store_true', help="Generate a comparison table of top features for PCA and ICA.")
    parser.add_argument('--n-top-features-comparison', type=int, default=5, help="Number of top features to show in PCA/ICA comparison.")
    
    # Arguments for Similarity Analysis
    parser.add_argument('--dcor', action='store_true', help="Perform Distance Correlation analysis.")
    parser.add_argument('--min-obs-dcor', type=int, default=10, help="Minimum number of observations for dCor calculation.")
    parser.add_argument('--cosine-sim', action='store_true', help="Perform Cosine Similarity analysis.")
    parser.add_argument('--min-obs-cosine', type=int, default=10, help="Minimum number of observations for Cosine Similarity calculation.")
    parser.add_argument('--dtw', action='store_true', help="Perform Dynamic Time Warping (DTW) analysis.")
    parser.add_argument('--min-obs-dtw', type=int, default=10, help="Minimum number of observations for DTW calculation.")
    parser.add_argument('--normalize-dtw', action='store_true', help="Normalize DTW distance by path length.", default=True)
    parser.add_argument('--mutual-info', action='store_true', help="Perform Mutual Information analysis.")
    parser.add_argument('--min-obs-mi', type=int, default=10, help="Minimum number of observations for Mutual Information calculation.")
    parser.add_argument('--mi-n-neighbors', type=int, default=3, help="Number of neighbors for MI estimation.")
    parser.add_argument('--normalize-mi', action='store_true', help="Normalize MI values to range [0,1].", default=True)

    # Arguments for Causal Analysis
    parser.add_argument('--run-sem', action='store_true', help="Run Structural Equation Modeling for causal analysis.")
    parser.add_argument('--sem-correlation-threshold', type=float, default=0.3, 
                       help="Minimum absolute correlation to include in the SEM model (e.g., 0.3 means include relationships with |r| >= 0.3).")
    parser.add_argument('--sem-standardize', action='store_true', default=True,
                       help="Standardize variables before SEM analysis.")
                       
    # Transfer Entropy arguments
    parser.add_argument('--run-transfer-entropy', action='store_true', 
                       help="Run Transfer Entropy analysis for information-theoretic causal analysis.")
    parser.add_argument('--te-lag', type=int, default=1, 
                       help="Lag parameter for Transfer Entropy calculation.")
    parser.add_argument('--te-threshold', type=float, default=0.05, 
                       help="Threshold for Transfer Entropy visualization.")
    
    # Convergent Cross Mapping arguments
    parser.add_argument('--run-ccm', action='store_true', 
                       help="Run Convergent Cross Mapping for nonlinear causal analysis.")
    parser.add_argument('--ccm-embed-dimensions', type=int, nargs='+', default=[2, 3, 4], 
                       help="Embedding dimensions to try for CCM.")
    parser.add_argument('--ccm-tau', type=int, default=1, 
                       help="Time delay parameter for CCM embedding.")
    parser.add_argument('--ccm-threshold', type=float, default=0.3, 
                       help="Significance threshold for CCM results.")
    
    # Granger Causality arguments
    parser.add_argument('--run-granger', action='store_true', 
                       help="Run Granger Causality tests for time series causal analysis.")
    parser.add_argument('--granger-max-lag', type=int, default=5, 
                       help="Maximum lag for Granger Causality test.")
    parser.add_argument('--granger-criterion', type=str, default='aic', choices=['aic', 'bic', 'fpe'], 
                       help="Information criterion for Granger model selection (aic, bic, or fpe).")
    parser.add_argument('--granger-alpha', type=float, default=0.05, 
                       help="Significance level for Granger causality tests.")
    parser.add_argument('--granger-fstat-threshold', type=float, default=0.0, 
                       help="Threshold for F-statistics in Granger causality network visualization.")

    # Argument for analysis scope
    parser.add_argument('--consolidated-analysis', action='store_true',
                        help='Run analysis consolidated across all phases (old behavior). If not set, analysis will be per-phase.')

    return parser.parse_args()


def setup_output_directories(output_dir):
    """Configures output directories."""
    plots_dir = os.path.join(output_dir, 'plots')
    tables_dir = os.path.join(output_dir, 'tables')

    # Descriptive Statistics
    descriptive_stats_plots_dir = os.path.join(plots_dir, "descriptive_stats")
    descriptive_stats_tables_dir = os.path.join(tables_dir, "descriptive_stats")
    os.makedirs(descriptive_stats_plots_dir, exist_ok=True)
    os.makedirs(descriptive_stats_tables_dir, exist_ok=True)

    # Correlation
    correlation_plots_dir = os.path.join(plots_dir, "correlation")
    correlation_tables_dir = os.path.join(tables_dir, "correlation")
    os.makedirs(correlation_plots_dir, exist_ok=True)
    os.makedirs(correlation_tables_dir, exist_ok=True)

    # Covariance
    covariance_plots_dir = os.path.join(plots_dir, "covariance")
    covariance_tables_dir = os.path.join(tables_dir, "covariance")
    os.makedirs(covariance_plots_dir, exist_ok=True)
    os.makedirs(covariance_tables_dir, exist_ok=True)

    # Multivariate (PCA, ICA, Comparison)
    multivariate_plots_dir = os.path.join(plots_dir, "multivariate")
    multivariate_tables_dir = os.path.join(tables_dir, "multivariate")
    os.makedirs(multivariate_plots_dir, exist_ok=True)
    os.makedirs(multivariate_tables_dir, exist_ok=True)

    pca_plots_dir = os.path.join(multivariate_plots_dir, "pca")
    pca_tables_dir = os.path.join(multivariate_tables_dir, "pca")
    os.makedirs(pca_plots_dir, exist_ok=True)
    os.makedirs(pca_tables_dir, exist_ok=True)

    ica_plots_dir = os.path.join(multivariate_plots_dir, "ica")
    ica_tables_dir = os.path.join(multivariate_tables_dir, "ica")
    os.makedirs(ica_plots_dir, exist_ok=True)
    os.makedirs(ica_tables_dir, exist_ok=True)

    comparison_plots_dir = os.path.join(multivariate_plots_dir, "comparison")
    comparison_tables_dir = os.path.join(multivariate_tables_dir, "comparison")
    os.makedirs(comparison_plots_dir, exist_ok=True)
    os.makedirs(comparison_tables_dir, exist_ok=True)
    
    # Causality (SEM, TE, CCM, Granger)
    causality_plots_dir = os.path.join(plots_dir, "causality")
    causality_tables_dir = os.path.join(tables_dir, "causality")
    os.makedirs(causality_plots_dir, exist_ok=True)
    os.makedirs(causality_tables_dir, exist_ok=True)

    sem_plots_dir = os.path.join(causality_plots_dir, "sem")
    sem_tables_dir = os.path.join(causality_tables_dir, "sem")
    os.makedirs(sem_plots_dir, exist_ok=True)
    os.makedirs(sem_tables_dir, exist_ok=True)
    
    te_plots_dir = os.path.join(causality_plots_dir, "transfer_entropy")
    te_tables_dir = os.path.join(causality_tables_dir, "transfer_entropy")
    os.makedirs(te_plots_dir, exist_ok=True)
    os.makedirs(te_tables_dir, exist_ok=True)
    
    ccm_plots_dir = os.path.join(causality_plots_dir, "ccm")
    ccm_tables_dir = os.path.join(causality_tables_dir, "ccm")
    os.makedirs(ccm_plots_dir, exist_ok=True)
    os.makedirs(ccm_tables_dir, exist_ok=True)
    
    granger_plots_dir = os.path.join(causality_plots_dir, "granger")
    granger_tables_dir = os.path.join(causality_tables_dir, "granger")
    os.makedirs(granger_plots_dir, exist_ok=True)
    os.makedirs(granger_tables_dir, exist_ok=True)
    
    # Similarity (dCor, Cosine, DTW, MI) - Already well-defined
    similarity_plots_dir = os.path.join(plots_dir, "similarity")
    similarity_tables_dir = os.path.join(tables_dir, "similarity")
    os.makedirs(similarity_plots_dir, exist_ok=True)
    os.makedirs(similarity_tables_dir, exist_ok=True)
    
    dcor_plots_dir = os.path.join(similarity_plots_dir, "distance_correlation")
    dcor_tables_dir = os.path.join(similarity_tables_dir, "distance_correlation")
    os.makedirs(dcor_plots_dir, exist_ok=True)
    os.makedirs(dcor_tables_dir, exist_ok=True)
    
    cosine_plots_dir = os.path.join(similarity_plots_dir, "cosine_similarity")
    cosine_tables_dir = os.path.join(similarity_tables_dir, "cosine_similarity")
    os.makedirs(cosine_plots_dir, exist_ok=True)
    os.makedirs(cosine_tables_dir, exist_ok=True)
    
    dtw_plots_dir = os.path.join(similarity_plots_dir, "dtw")
    dtw_tables_dir = os.path.join(similarity_tables_dir, "dtw")
    os.makedirs(dtw_plots_dir, exist_ok=True)
    os.makedirs(dtw_tables_dir, exist_ok=True)
    
    mi_plots_dir = os.path.join(similarity_plots_dir, "mutual_information")
    mi_tables_dir = os.path.join(similarity_tables_dir, "mutual_information")
    os.makedirs(mi_plots_dir, exist_ok=True)
    os.makedirs(mi_tables_dir, exist_ok=True)

    # General directories (already created by specific ones, but good for completeness)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    return (
        plots_dir, tables_dir,
        descriptive_stats_plots_dir, descriptive_stats_tables_dir,
        correlation_plots_dir, correlation_tables_dir,
        covariance_plots_dir, covariance_tables_dir,
        pca_plots_dir, pca_tables_dir,
        ica_plots_dir, ica_tables_dir,
        comparison_plots_dir, comparison_tables_dir,
        sem_plots_dir, sem_tables_dir,
        te_plots_dir, te_tables_dir,
        ccm_plots_dir, ccm_tables_dir,
        granger_plots_dir, granger_tables_dir,
        dcor_plots_dir, dcor_tables_dir,
        cosine_plots_dir, cosine_tables_dir,
        dtw_plots_dir, dtw_tables_dir,
        mi_plots_dir, mi_tables_dir
    )


def main():
    """Main function to run the refactored analysis pipeline."""
    args = parse_arguments()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    (
        plots_dir, tables_dir,
        descriptive_stats_plots_dir, descriptive_stats_tables_dir,
        correlation_plots_dir, correlation_tables_dir,
        covariance_plots_dir, covariance_tables_dir,
        pca_plots_dir, pca_tables_dir,
        ica_plots_dir, ica_tables_dir,
        comparison_plots_dir, comparison_tables_dir,
        sem_plots_dir, sem_tables_dir,
        te_plots_dir, te_tables_dir,
        ccm_plots_dir, ccm_tables_dir,
        granger_plots_dir, granger_tables_dir,
        dcor_plots_dir, dcor_tables_dir,
        cosine_plots_dir, cosine_tables_dir,
        dtw_plots_dir, dtw_tables_dir,
        mi_plots_dir, mi_tables_dir
    ) = setup_output_directories(args.output_dir)

    # Resolve data directory path
    experiment_data_dir_input = args.data_dir
    experiment_data_dir = ""

    if os.path.isabs(experiment_data_dir_input):
        experiment_data_dir = experiment_data_dir_input
    else:
        cwd = os.getcwd()
        path_from_cwd = os.path.join(cwd, experiment_data_dir_input)
        path_from_project_root_perspective = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), experiment_data_dir_input)

        if os.path.isdir(path_from_cwd):
            experiment_data_dir = path_from_cwd
        elif os.path.isdir(path_from_project_root_perspective) and not os.path.isdir(path_from_cwd):
            experiment_data_dir = os.path.normpath(path_from_project_root_perspective)
        else:
            experiment_data_dir = path_from_cwd

    experiment_data_dir = os.path.normpath(experiment_data_dir)

    if not os.path.isdir(experiment_data_dir):
        print(f"Error: Data directory not found: {experiment_data_dir}")
        print(f"  Input --data-dir: '{args.data_dir}'")
        print(f"  Resolved absolute path: '{os.path.abspath(experiment_data_dir)}'")
        return

    print(f"Using data directory: {experiment_data_dir}")
    print(f"Saving results to: {args.output_dir}")

    # 1. Load data
    run_per_phase_analysis = not args.consolidated_analysis
    if run_per_phase_analysis:
        print("Data loading mode: Per-Phase")
    else:
        print("Data loading mode: Consolidated")

    all_metrics_data = load_experiment_data(
        experiment_dir=experiment_data_dir,
        metrics=args.metrics,
        tenants=args.tenants,
        rounds=args.rounds,
        group_by_phase=run_per_phase_analysis
    )

    if not all_metrics_data:
        print("No data loaded. Exiting.")
        return

    print(f"Loaded metrics: {list(all_metrics_data.keys())}")

    # --- Apply Time Normalization ---
    print("\nApplying Time Normalization...")
    for metric_name, rounds_or_phases_data in all_metrics_data.items():
        print(f"Processing time normalization for metric: {metric_name}")
        if run_per_phase_analysis:
            for round_name, phases_data in rounds_or_phases_data.items():
                for phase_name, metric_df in phases_data.items():
                    print(f"  Processing {metric_name}, {round_name}, {phase_name}")
                    if isinstance(metric_df, pd.DataFrame):
                        print(f"    DataFrame shape: {metric_df.shape}")
                        print(f"    DataFrame columns before: {metric_df.columns.tolist()}")
                        if 'datetime' in metric_df.columns:
                            print(f"    'datetime' column found, calling add_experiment_elapsed_time...")
                            # Group by round for experiment_elapsed_time context, even in per-phase
                            result_df = add_experiment_elapsed_time(
                                metric_df, group_by=['round']
                            )
                            all_metrics_data[metric_name][round_name][phase_name] = result_df
                            print(f"    DataFrame columns after: {result_df.columns.tolist()}")
                            if 'experiment_elapsed_time' in result_df.columns:
                                print(f"    ✓ experiment_elapsed_time column successfully added")
                            else:
                                print(f"    ✗ experiment_elapsed_time column NOT added")
                        else:
                            print(f"    'datetime' column missing from columns: {metric_df.columns.tolist()}")
                    else:
                        print(f"    Skipping time normalization for {metric_name}, {round_name}, {phase_name}: DataFrame not found, got {type(metric_df)}.")
        else:  # Consolidated analysis
            for round_name, metric_df in rounds_or_phases_data.items():
                print(f"  Processing {metric_name}, {round_name} (Consolidated)")
                if isinstance(metric_df, pd.DataFrame):
                    print(f"    DataFrame shape: {metric_df.shape}")
                    print(f"    DataFrame columns before: {metric_df.columns.tolist()}")
                    if 'datetime' in metric_df.columns:
                        print(f"    'datetime' column found, calling add_experiment_elapsed_time...")
                        result_df = add_experiment_elapsed_time(
                            metric_df, group_by=['round']
                        )
                        all_metrics_data[metric_name][round_name] = result_df
                        print(f"    DataFrame columns after: {result_df.columns.tolist()}")
                        if 'experiment_elapsed_time' in result_df.columns:
                            print(f"    ✓ experiment_elapsed_time column successfully added")
                        else:
                            print(f"    ✗ experiment_elapsed_time column NOT added")
                    else:
                        print(f"    'datetime' column missing from columns: {metric_df.columns.tolist()}")
                else:
                    print(f"    Skipping time normalization for {metric_name}, {round_name} (Consolidated): DataFrame not found, got {type(metric_df)}.")

    # --- Distance Correlation Analysis ---
    if args.dcor:
        print("\nRunning Distance Correlation Analysis...")
        for metric_name, rounds_or_phases_data in all_metrics_data.items():
            print(f"Processing metric: {metric_name}")
            if not isinstance(rounds_or_phases_data, dict):
                print(f"  Skipping metric {metric_name}: Expected a dictionary of round/phase data, got {type(rounds_or_phases_data)}.")
                continue

            for round_name, phases_or_metric_df in rounds_or_phases_data.items():
                if run_per_phase_analysis:
                    if not isinstance(phases_or_metric_df, dict):
                        print(f"  Skipping round {round_name} for metric {metric_name}: Expected a dictionary of phase data, got {type(phases_or_metric_df)}.")
                        continue
                    for phase_name, original_phase_df in phases_or_metric_df.items():
                        print(f"  Processing round: {round_name}, phase: {phase_name} for metric: {metric_name}")
                        if not isinstance(original_phase_df, pd.DataFrame):
                            print(f"    Skipping round {round_name}, phase {phase_name} for metric {metric_name}: Expected a DataFrame, got {type(original_phase_df)}.")
                            continue
                        if original_phase_df.empty:
                            print(f"    Skipping Distance Correlation for metric {metric_name}, round {round_name}, phase {phase_name} due to empty DataFrame.")
                            continue

                        if 'experiment_elapsed_time' not in original_phase_df.columns:
                            print(f"    Skipping dCor for {metric_name}, {round_name}, {phase_name}: 'experiment_elapsed_time' column missing.")
                            continue

                        try:
                            print(f"    Calculating Distance Correlation for {metric_name}, round {round_name}, phase {phase_name}...")
                            # Ensure value column is numeric
                            original_phase_df['value'] = pd.to_numeric(original_phase_df['value'], errors='coerce')
                            
                            dcor_matrix = calculate_pairwise_distance_correlation(
                                data_df=original_phase_df,
                                time_col='experiment_elapsed_time',
                                metric_col='value',  # 'value' is the actual metric value column
                                group_col='tenant',  # 'tenant' is the column containing tenant IDs
                                min_observations=args.min_obs_dcor
                            )
                            
                            if not dcor_matrix.empty:
                                plot_title = f"Distance Correlation - {metric_name}"
                                plot_filename = f"{metric_name}_{round_name}_{phase_name}_distance_correlation_heatmap.png"
                                plot_distance_correlation_heatmap(
                                    dcor_matrix,
                                    title=plot_title,
                                    output_dir=dcor_plots_dir,
                                    filename=plot_filename,
                                    tables_dir=dcor_tables_dir
                                )
                                print(f"    Distance Correlation heatmap created for {metric_name}, round {round_name}, phase {phase_name}.")
                            else:
                                print(f"    No Distance Correlation results for {metric_name}, round {round_name}, phase {phase_name}.")
                                
                        except Exception as e:
                            print(f"    Error during Distance Correlation analysis for {metric_name}, round {round_name}, phase {phase_name}: {e}")
                            import traceback
                            traceback.print_exc()
                else:
                    original_metric_df = phases_or_metric_df
                    print(f"  Processing round: {round_name} for metric: {metric_name} (Consolidated)")
                    if not isinstance(original_metric_df, pd.DataFrame):
                        print(f"    Skipping round {round_name} for metric {metric_name} (Consolidated): Expected a DataFrame, got {type(original_metric_df)}.")
                        continue
                    if original_metric_df.empty:
                        print(f"    Skipping Distance Correlation for metric {metric_name}, round {round_name} (Consolidated) due to empty DataFrame.")
                        continue

                    if 'experiment_elapsed_time' not in original_metric_df.columns:
                        print(f"    Skipping dCor for {metric_name}, {round_name} (Consolidated): 'experiment_elapsed_time' column missing.")
                        continue

                    try:
                        print(f"    Calculating Distance Correlation for {metric_name}, round {round_name} (Consolidated)...")
                        # Ensure value column is numeric
                        original_metric_df['value'] = pd.to_numeric(original_metric_df['value'], errors='coerce')
                        
                        dcor_matrix = calculate_pairwise_distance_correlation(
                            data_df=original_metric_df,
                            time_col='experiment_elapsed_time',
                            metric_col='value',  # 'value' is the actual metric value column
                            group_col='tenant',  # 'tenant' is the column containing tenant IDs
                            min_observations=args.min_obs_dcor
                        )
                        
                        if not dcor_matrix.empty:
                            plot_title = f"Distance Correlation - {metric_name}"
                            plot_filename = f"{metric_name}_{round_name}_distance_correlation_heatmap.png"
                            plot_distance_correlation_heatmap(
                                dcor_matrix,
                                title=plot_title,
                                output_dir=dcor_plots_dir,
                                filename=plot_filename,
                                tables_dir=dcor_tables_dir
                            )
                            print(f"    Distance Correlation heatmap created for {metric_name}, round {round_name} (Consolidated).")
                        else:
                            print(f"    No Distance Correlation results for {metric_name}, round {round_name} (Consolidated).")
                            
                    except Exception as e:
                        print(f"    Error during Distance Correlation analysis for {metric_name}, round {round_name} (Consolidated): {e}")
                        import traceback
                        traceback.print_exc()

    # --- Cosine Similarity Analysis ---
    if args.cosine_sim:
        print("\nRunning Cosine Similarity Analysis...")
        for metric_name, rounds_or_phases_data in all_metrics_data.items():
            print(f"Processing metric: {metric_name}")
            if not isinstance(rounds_or_phases_data, dict):
                print(f"  Skipping metric {metric_name}: Expected a dictionary of round/phase data, got {type(rounds_or_phases_data)}.")
                continue

            for round_name, phases_or_metric_df in rounds_or_phases_data.items():
                if run_per_phase_analysis:
                    if not isinstance(phases_or_metric_df, dict):
                        print(f"  Skipping round {round_name} for metric {metric_name}: Expected a dictionary of phase data, got {type(phases_or_metric_df)}.")
                        continue
                    for phase_name, original_phase_df in phases_or_metric_df.items():
                        print(f"  Processing round: {round_name}, phase: {phase_name} for metric: {metric_name}")
                        if not isinstance(original_phase_df, pd.DataFrame):
                            print(f"    Skipping round {round_name}, phase {phase_name} for metric {metric_name}: Expected a DataFrame, got {type(original_phase_df)}.")
                            continue
                        if original_phase_df.empty:
                            print(f"    Skipping Cosine Similarity for metric {metric_name}, round {round_name}, phase {phase_name} due to empty DataFrame.")
                            continue

                        if 'experiment_elapsed_time' not in original_phase_df.columns:
                            print(f"    Skipping Cosine Similarity for {metric_name}, {round_name}, {phase_name}: 'experiment_elapsed_time' column missing.")
                            continue

                        try:
                            print(f"    Calculating Cosine Similarity for {metric_name}, round {round_name}, phase {phase_name}...")
                            # Ensure value column is numeric
                            original_phase_df['value'] = pd.to_numeric(original_phase_df['value'], errors='coerce')
                            
                            cosine_matrix = calculate_pairwise_cosine_similarity(
                                data_df=original_phase_df,
                                time_col='experiment_elapsed_time',
                                metric_col='value',  # 'value' is the actual metric value column
                                group_col='tenant',  # 'tenant' is the column containing tenant IDs
                                min_observations=args.min_obs_cosine
                            )
                            
                            if not cosine_matrix.empty:
                                plot_title = f"Cosine Similarity - {metric_name}"
                                plot_filename = f"{metric_name}_{round_name}_{phase_name}_cosine_similarity_heatmap.png"
                                plot_cosine_similarity_heatmap(
                                    cosine_matrix,
                                    title=plot_title,
                                    output_dir=cosine_plots_dir,
                                    filename=plot_filename,
                                    tables_dir=cosine_tables_dir
                                )
                                print(f"    Cosine Similarity heatmap created for {metric_name}, round {round_name}, phase {phase_name}.")
                            else:
                                print(f"    No Cosine Similarity results for {metric_name}, round {round_name}, phase {phase_name}.")
                                
                        except Exception as e:
                            print(f"    Error during Cosine Similarity analysis for {metric_name}, round {round_name}, phase {phase_name}: {e}")
                            import traceback
                            traceback.print_exc()
                else:
                    original_metric_df = phases_or_metric_df
                    print(f"  Processing round: {round_name} for metric: {metric_name} (Consolidated)")
                    if not isinstance(original_metric_df, pd.DataFrame):
                        print(f"    Skipping round {round_name} for metric {metric_name} (Consolidated): Expected a DataFrame, got {type(original_metric_df)}.")
                        continue
                    if original_metric_df.empty:
                        print(f"    Skipping Cosine Similarity for metric {metric_name}, round {round_name} (Consolidated) due to empty DataFrame.")
                        continue

                    if 'experiment_elapsed_time' not in original_metric_df.columns:
                        print(f"    Skipping Cosine Similarity for {metric_name}, {round_name} (Consolidated): 'experiment_elapsed_time' column missing.")
                        continue

                    try:
                        print(f"    Calculating Cosine Similarity for {metric_name}, round {round_name} (Consolidated)...")
                        # Ensure value column is numeric
                        original_metric_df['value'] = pd.to_numeric(original_metric_df['value'], errors='coerce')
                        
                        cosine_matrix = calculate_pairwise_cosine_similarity(
                            data_df=original_metric_df,
                            time_col='experiment_elapsed_time',
                            metric_col='value',  # 'value' is the actual metric value column
                            group_col='tenant',  # 'tenant' is the column containing tenant IDs
                            min_observations=args.min_obs_cosine
                        )
                        
                        if not cosine_matrix.empty:
                            plot_title = f"Cosine Similarity - {metric_name}"
                            plot_filename = f"{metric_name}_{round_name}_cosine_similarity_heatmap.png"
                            plot_cosine_similarity_heatmap(
                                cosine_matrix,
                                title=plot_title,
                                output_dir=cosine_plots_dir,
                                filename=plot_filename,
                                tables_dir=cosine_tables_dir
                            )
                            print(f"    Cosine Similarity heatmap created for {metric_name}, round {round_name} (Consolidated).")
                        else:
                            print(f"    No Cosine Similarity results for {metric_name}, round {round_name} (Consolidated).")
                            
                    except Exception as e:
                        print(f"    Error during Cosine Similarity analysis for {metric_name}, round {round_name} (Consolidated): {e}")
                        import traceback
                        traceback.print_exc()

    # --- DTW Distance Analysis ---
    if args.dtw:
        print("\nRunning Dynamic Time Warping (DTW) Analysis...")
        for metric_name, rounds_or_phases_data in all_metrics_data.items():
            print(f"Processing metric: {metric_name}")
            if not isinstance(rounds_or_phases_data, dict):
                print(f"  Skipping metric {metric_name}: Expected a dictionary of round/phase data, got {type(rounds_or_phases_data)}.")
                continue

            for round_name, phases_or_metric_df in rounds_or_phases_data.items():
                if run_per_phase_analysis:
                    if not isinstance(phases_or_metric_df, dict):
                        print(f"  Skipping round {round_name} for metric {metric_name}: Expected a dictionary of phase data, got {type(phases_or_metric_df)}.")
                        continue
                    for phase_name, original_phase_df in phases_or_metric_df.items():
                        print(f"  Processing round: {round_name}, phase: {phase_name} for metric: {metric_name}")
                        if not isinstance(original_phase_df, pd.DataFrame):
                            print(f"    Skipping round {round_name}, phase {phase_name} for metric {metric_name}: Expected a DataFrame, got {type(original_phase_df)}.")
                            continue
                        if original_phase_df.empty:
                            print(f"    Skipping DTW for metric {metric_name}, round {round_name}, phase {phase_name} due to empty DataFrame.")
                            continue

                        if 'experiment_elapsed_time' not in original_phase_df.columns:
                            print(f"    Skipping DTW for {metric_name}, {round_name}, phase {phase_name}: 'experiment_elapsed_time' column missing.")
                            continue

                        try:
                            print(f"    Calculating DTW for {metric_name}, round {round_name}, phase {phase_name}...")
                            # Ensure value column is numeric
                            original_phase_df['value'] = pd.to_numeric(original_phase_df['value'], errors='coerce')
                            
                            dtw_matrix = calculate_pairwise_dtw_distance(
                                data_df=original_phase_df,
                                time_col='experiment_elapsed_time',
                                metric_col='value',  # 'value' is the actual metric value column
                                group_col='tenant',  # 'tenant' is the column containing tenant IDs
                                min_observations=args.min_obs_dtw,
                                normalize=args.normalize_dtw
                            )
                            
                            if not dtw_matrix.empty:
                                plot_title = f"DTW Distance - {metric_name} ({phase_name})"
                                plot_filename = f"{metric_name}_{round_name}_{phase_name}_dtw_distance_heatmap.png"
                                plot_dtw_distance_heatmap(
                                    dtw_matrix,
                                    title=plot_title,
                                    output_dir=dtw_plots_dir,
                                    filename=plot_filename,
                                    tables_dir=dtw_tables_dir
                                )
                                print(f"    DTW Distance heatmap created for {metric_name}, round {round_name}, phase {phase_name}.")
                            else:
                                print(f"    No DTW Distance results for {metric_name}, round {round_name}, phase {phase_name}.")
                                
                        except Exception as e:
                            print(f"    Error during DTW Distance analysis for {metric_name}, round {round_name}, phase {phase_name}: {e}")
                            import traceback
                            traceback.print_exc()
                else:
                    original_metric_df = phases_or_metric_df
                    print(f"  Processing round: {round_name} for metric: {metric_name} (Consolidated)")
                    if not isinstance(original_metric_df, pd.DataFrame):
                        print(f"    Skipping round {round_name} for metric {metric_name} (Consolidated): Expected a DataFrame, got {type(original_metric_df)}.")
                        continue
                    if original_metric_df.empty:
                        print(f"    Skipping DTW Distance for metric {metric_name}, round {round_name} (Consolidated) due to empty DataFrame.")
                        continue

                    if 'experiment_elapsed_time' not in original_metric_df.columns:
                        print(f"    Skipping DTW Distance for {metric_name}, {round_name} (Consolidated): 'experiment_elapsed_time' column missing.")
                        continue

                    try:
                        print(f"    Calculating DTW Distance for {metric_name}, round {round_name} (Consolidated)...")
                        # Ensure value column is numeric
                        original_metric_df['value'] = pd.to_numeric(original_metric_df['value'], errors='coerce')
                        
                        dtw_matrix = calculate_pairwise_dtw_distance(
                            data_df=original_metric_df,
                            time_col='experiment_elapsed_time',
                            metric_col='value',  # 'value' is the actual metric value column
                            group_col='tenant',  # 'tenant' is the column containing tenant IDs
                            min_observations=args.min_obs_dtw,
                            normalize=args.normalize_dtw
                        )
                        
                        if not dtw_matrix.empty:
                            plot_title = f"DTW Distance - {metric_name}"
                            plot_filename = f"{metric_name}_{round_name}_dtw_distance_heatmap.png"
                            plot_dtw_distance_heatmap(
                                dtw_matrix,
                                title=plot_title,
                                output_dir=dtw_plots_dir,
                                filename=plot_filename,
                                tables_dir=dtw_tables_dir
                            )
                            print(f"    DTW Distance heatmap created for {metric_name}, round {round_name} (Consolidated).")
                        else:
                            print(f"    No DTW Distance results for {metric_name}, round {round_name} (Consolidated).")
                            
                    except Exception as e:
                        print(f"    Error during DTW Distance analysis for {metric_name}, round {round_name} (Consolidated): {e}")
                        import traceback
                        traceback.print_exc()

    # --- Mutual Information Analysis ---
    if args.mutual_info:
        print("\nRunning Mutual Information Analysis...")
        for metric_name, rounds_or_phases_data in all_metrics_data.items():
            print(f"Processing metric: {metric_name}")
            if not isinstance(rounds_or_phases_data, dict):
                print(f"  Skipping metric {metric_name}: Expected a dictionary of round/phase data, got {type(rounds_or_phases_data)}.")
                continue

            for round_name, phases_or_metric_df in rounds_or_phases_data.items():
                if run_per_phase_analysis:
                    if not isinstance(phases_or_metric_df, dict):
                        print(f"  Skipping round {round_name} for metric {metric_name}: Expected a dictionary of phase data, got {type(phases_or_metric_df)}.")
                        continue
                    for phase_name, original_phase_df in phases_or_metric_df.items():
                        phase_display_name = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
                        print(f"  Processing round: {round_name}, phase: {phase_display_name} for metric: {metric_name}")
                        if not isinstance(original_phase_df, pd.DataFrame):
                            print(f"    Skipping round {round_name}, phase {phase_display_name} for metric {metric_name}: Expected a DataFrame, got {type(original_phase_df)}.")
                            continue
                        if original_phase_df.empty:
                            print(f"    Skipping Mutual Information for metric {metric_name}, round {round_name}, phase {phase_display_name} due to empty DataFrame.")
                            continue

                        if 'experiment_elapsed_time' not in original_phase_df.columns:
                            print(f"    Skipping Mutual Information for {metric_name}, {round_name}, phase {phase_display_name}: 'experiment_elapsed_time' column missing.")
                            continue

                        try:
                            print(f"    Calculating Mutual Information for {metric_name}, round {round_name}, phase {phase_display_name}...")
                            # Ensure value column is numeric
                            current_df_processed = original_phase_df.copy()
                            current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                            
                            mi_matrix = calculate_pairwise_mutual_information(
                                data_df=current_df_processed,
                                time_col='experiment_elapsed_time',
                                metric_col='value',
                                group_col='tenant',
                                min_observations=args.min_obs_mi,
                                n_neighbors=args.mi_n_neighbors,
                                normalize=args.normalize_mi
                            )
                            
                            if not mi_matrix.empty:
                                plot_title = f"Mutual Information - {METRIC_DISPLAY_NAMES.get(metric_name, metric_name)} ({phase_display_name})"
                                plot_filename = f"{metric_name}_{round_name}_{phase_name}_mutual_information_heatmap.png"
                                table_filename_base = f"{metric_name}_{round_name}_{phase_name}_mutual_information_heatmap"
                                plot_mutual_information_heatmap(
                                    mi_matrix,
                                    title=plot_title,
                                    output_dir=mi_plots_dir,
                                    filename=plot_filename,
                                    tables_dir=mi_tables_dir,
                                    table_filename_base=table_filename_base,
                                    metric_name=METRIC_DISPLAY_NAMES.get(metric_name, metric_name),
                                    round_name=round_name,
                                    phase_name=phase_display_name
                                )
                                print(f"    Mutual Information heatmap created for {metric_name}, round {round_name}, phase {phase_display_name}.")
                            else:
                                print(f"    No Mutual Information results for {metric_name}, round {round_name}, phase {phase_display_name}.")
                                
                        except Exception as e:
                            print(f"    Error during Mutual Information analysis for {metric_name}, round {round_name}, phase {phase_display_name}: {e}")
                            import traceback
                            traceback.print_exc()
                else: # Consolidated analysis
                    original_metric_df = phases_or_metric_df
                    print(f"  Processing round: {round_name} for metric: {metric_name} (Consolidated)")
                    if not isinstance(original_metric_df, pd.DataFrame):
                        print(f"    Skipping round {round_name} for metric {metric_name} (Consolidated): Expected a DataFrame, got {type(original_metric_df)}.")
                        continue
                    if original_metric_df.empty:
                        print(f"    Skipping Mutual Information for metric {metric_name}, round {round_name} (Consolidated) due to empty DataFrame.")
                        continue

                    if 'experiment_elapsed_time' not in original_metric_df.columns:
                        print(f"    Skipping Mutual Information for {metric_name}, {round_name} (Consolidated): 'experiment_elapsed_time' column missing.")
                        continue

                    try:
                        print(f"    Calculating Mutual Information for {metric_name}, round {round_name} (Consolidated)...")
                        current_df_processed = original_metric_df.copy()
                        current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                        
                        mi_matrix = calculate_pairwise_mutual_information(
                            data_df=current_df_processed,
                            time_col='experiment_elapsed_time',
                            metric_col='value',
                            group_col='tenant',
                            min_observations=args.min_obs_mi,
                            n_neighbors=args.mi_n_neighbors,
                            normalize=args.normalize_mi
                        )
                        
                        if not mi_matrix.empty:
                            plot_title = f"Mutual Information - {METRIC_DISPLAY_NAMES.get(metric_name, metric_name)}"
                            plot_filename = f"{metric_name}_{round_name}_mutual_information_heatmap.png"
                            table_filename_base = f"{metric_name}_{round_name}_mutual_information_heatmap"
                            plot_mutual_information_heatmap(
                                mi_matrix,
                                title=plot_title,
                                output_dir=mi_plots_dir,
                                filename=plot_filename,
                                tables_dir=mi_tables_dir,
                                table_filename_base=table_filename_base,
                                metric_name=METRIC_DISPLAY_NAMES.get(metric_name, metric_name),
                                round_name=round_name
                            )
                            print(f"    Mutual Information heatmap created for {metric_name}, round {round_name} (Consolidated).")
                        else:
                            print(f"    No Mutual Information results for {metric_name}, round {round_name} (Consolidated).")
                            
                    except Exception as e:
                        print(f"    Error during Mutual Information analysis for {metric_name}, round {round_name} (Consolidated): {e}")
                        import traceback
                        traceback.print_exc()
        gc.collect() # Garbage collect after processing each metric

    # --- Descriptive Statistics Analysis ---
    if args.run_descriptive_stats:
        print("\nRunning Descriptive Statistics Analysis...")
        for metric_name, rounds_or_phases_data in all_metrics_data.items():
            print(f"Processing metric: {metric_name}")
            if not isinstance(rounds_or_phases_data, dict):
                print(f"  Skipping metric {metric_name}: Expected dict, got {type(rounds_or_phases_data)}.")
                continue

            for round_name, phases_or_metric_df in rounds_or_phases_data.items():
                if run_per_phase_analysis:
                    if not isinstance(phases_or_metric_df, dict):
                        print(f"  Skipping round {round_name} for {metric_name}: Expected dict, got {type(phases_or_metric_df)}.")
                        continue
                    for phase_name, original_phase_df in phases_or_metric_df.items():
                        phase_display_name = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
                        print(f"  Processing round: {round_name}, phase: {phase_display_name} for metric: {metric_name}")
                        if not isinstance(original_phase_df, pd.DataFrame) or original_phase_df.empty:
                            print(f"    Skipping: DataFrame is not valid or empty.")
                            continue
                        try:
                            current_df_processed = original_phase_df.copy()
                            current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                            stats_df = calculate_descriptive_statistics(current_df_processed, group_by_cols=['tenant']) # Simplified grouping for now
                            
                            if not stats_df.empty:
                                table_filename = f"{metric_name}_{round_name}_{phase_name}_descriptive_stats.csv"
                                export_to_csv(stats_df, descriptive_stats_tables_dir, table_filename)
                                print(f"    Descriptive statistics table saved to {table_filename}")

                                plot_title_base = f"Descriptive Stats - {METRIC_DISPLAY_NAMES.get(metric_name, metric_name)} ({phase_display_name})"
                                # Example plots (can be expanded)
                                plot_descriptive_stats_boxplot(current_df_processed, 'value', 'tenant', title=f"{plot_title_base} - Boxplot",
                                                               output_dir=descriptive_stats_plots_dir, filename=f"{metric_name}_{round_name}_{phase_name}_boxplot.png",
                                                               metric_name=METRIC_DISPLAY_NAMES.get(metric_name, metric_name), round_name=round_name, phase_name=phase_display_name)
                                print(f"    Descriptive statistics boxplot created.")
                        except Exception as e:
                            print(f"    Error during Descriptive Statistics for {metric_name}, {round_name}, {phase_display_name}: {e}")
                            traceback.print_exc()
                else: # Consolidated
                    original_metric_df = phases_or_metric_df
                    print(f"  Processing round: {round_name} for metric: {metric_name} (Consolidated)")
                    if not isinstance(original_metric_df, pd.DataFrame) or original_metric_df.empty:
                        print(f"    Skipping: DataFrame is not valid or empty.")
                        continue
                    try:
                        current_df_processed = original_metric_df.copy()
                        current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                        stats_df = calculate_descriptive_statistics(current_df_processed, group_by_cols=['tenant'])
                        
                        if not stats_df.empty:
                            table_filename = f"{metric_name}_{round_name}_descriptive_stats.csv"
                            export_to_csv(stats_df, descriptive_stats_tables_dir, table_filename)
                            print(f"    Descriptive statistics table saved to {table_filename}")
                            
                            plot_title_base = f"Descriptive Stats - {METRIC_DISPLAY_NAMES.get(metric_name, metric_name)}"
                            plot_descriptive_stats_boxplot(current_df_processed, 'value', 'tenant', title=f"{plot_title_base} - Boxplot",
                                                           output_dir=descriptive_stats_plots_dir, filename=f"{metric_name}_{round_name}_boxplot.png",
                                                           metric_name=METRIC_DISPLAY_NAMES.get(metric_name, metric_name), round_name=round_name)
                            print(f"    Descriptive statistics boxplot created.")
                    except Exception as e:
                        print(f"    Error during Descriptive Statistics for {metric_name}, {round_name} (Consolidated): {e}")
                        traceback.print_exc()
            gc.collect()

    # --- Correlation Analysis ---
    if args.run_correlation or args.run_all_correlation_methods:
        print("\nRunning Correlation Analysis...")
        correlation_methods_to_run = args.correlation_methods
        if args.run_all_correlation_methods:
            correlation_methods_to_run = ['pearson', 'spearman', 'kendall']
        
        for method in correlation_methods_to_run:
            print(f"  Using correlation method: {method}")
            for metric_name, rounds_or_phases_data in all_metrics_data.items():
                print(f"Processing metric: {metric_name}")
                # ... (similar iteration structure as above) ...
                for round_name, phases_or_metric_df in rounds_or_phases_data.items():
                    if run_per_phase_analysis:
                        for phase_name, original_phase_df in phases_or_metric_df.items():
                            phase_display_name = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
                            print(f"  Processing round: {round_name}, phase: {phase_display_name} for metric: {metric_name}, method: {method}")
                            if not isinstance(original_phase_df, pd.DataFrame) or original_phase_df.empty:
                                print(f"    Skipping: DataFrame is not valid or empty.")
                                continue
                            try:
                                current_df_processed = original_phase_df.copy()
                                current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                                # Pivot data: time as index, tenants as columns, metric value as values
                                pivoted_df = current_df_processed.pivot_table(index='experiment_elapsed_time', columns='tenant', values='value')
                                pivoted_df.dropna(how='all', axis=1, inplace=True) # Drop tenant columns if all NaN
                                pivoted_df.dropna(how='any', axis=0, inplace=True)   # Drop rows with any NaN to ensure pair-wise completeness for correlation

                                if pivoted_df.shape[1] < 2: # Need at least 2 tenants to correlate
                                    print(f"    Skipping correlation: Not enough tenants with data (need at least 2, found {pivoted_df.shape[1]}).")
                                    continue

                                correlation_matrix = calculate_inter_tenant_correlation_per_metric(pivoted_df, method=method, time_col=None) # Already pivoted
                                
                                if not correlation_matrix.empty:
                                    plot_title = f"Inter-Tenant Correlation ({method.capitalize()}) - {METRIC_DISPLAY_NAMES.get(metric_name, metric_name)} ({phase_display_name})"
                                    plot_filename = f"{metric_name}_{round_name}_{phase_name}_{method}_correlation_heatmap.png"
                                    table_filename_base = f"{metric_name}_{round_name}_{phase_name}_{method}_correlation"
                                    
                                    plot_correlation_heatmap(
                                        correlation_matrix, title=plot_title, output_dir=correlation_plots_dir, filename=plot_filename,
                                        tables_dir=correlation_tables_dir, table_filename_base=table_filename_base,
                                        metric_name=METRIC_DISPLAY_NAMES.get(metric_name, metric_name), round_name=round_name, phase_name=phase_display_name
                                    )
                                    print(f"    Correlation ({method}) heatmap created.")
                            except Exception as e:
                                print(f"    Error during Correlation ({method}) for {metric_name}, {round_name}, {phase_display_name}: {e}")
                                traceback.print_exc()
                    else: # Consolidated
                        original_metric_df = phases_or_metric_df
                        print(f"  Processing round: {round_name} for metric: {metric_name} (Consolidated), method: {method}")
                        if not isinstance(original_metric_df, pd.DataFrame) or original_metric_df.empty:
                             print(f"    Skipping: DataFrame is not valid or empty.")
                             continue
                        try:
                            current_df_processed = original_metric_df.copy()
                            current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                            pivoted_df = current_df_processed.pivot_table(index='experiment_elapsed_time', columns='tenant', values='value')
                            pivoted_df.dropna(how='all', axis=1, inplace=True)
                            pivoted_df.dropna(how='any', axis=0, inplace=True)

                            if pivoted_df.shape[1] < 2:
                                print(f"    Skipping correlation: Not enough tenants with data (need at least 2, found {pivoted_df.shape[1]}).")
                                continue
                                
                            correlation_matrix = calculate_inter_tenant_correlation_per_metric(pivoted_df, method=method, time_col=None)
                            if not correlation_matrix.empty:
                                plot_title = f"Inter-Tenant Correlation ({method.capitalize()}) - {METRIC_DISPLAY_NAMES.get(metric_name, metric_name)}"
                                plot_filename = f"{metric_name}_{round_name}_{method}_correlation_heatmap.png"
                                table_filename_base = f"{metric_name}_{round_name}_{method}_correlation"
                                plot_correlation_heatmap(
                                    correlation_matrix, title=plot_title, output_dir=correlation_plots_dir, filename=plot_filename,
                                    tables_dir=correlation_tables_dir, table_filename_base=table_filename_base,
                                    metric_name=METRIC_DISPLAY_NAMES.get(metric_name, metric_name), round_name=round_name
                                )
                                print(f"    Correlation ({method}) heatmap created.")
                        except Exception as e:
                            print(f"    Error during Correlation ({method}) for {metric_name}, {round_name} (Consolidated): {e}")
                            traceback.print_exc()
                gc.collect()

    # --- Covariance Analysis ---
    if args.run_covariance:
        print("\nRunning Covariance Analysis...")
        for metric_name, rounds_or_phases_data in all_metrics_data.items():
            print(f"Processing metric: {metric_name}")
            # ... (similar iteration structure) ...
            for round_name, phases_or_metric_df in rounds_or_phases_data.items():
                if run_per_phase_analysis:
                    for phase_name, original_phase_df in phases_or_metric_df.items():
                        phase_display_name = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
                        print(f"  Processing round: {round_name}, phase: {phase_display_name} for metric: {metric_name}")
                        if not isinstance(original_phase_df, pd.DataFrame) or original_phase_df.empty:
                            print(f"    Skipping: DataFrame is not valid or empty.")
                            continue
                        try:
                            current_df_processed = original_phase_df.copy()
                            current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                            pivoted_df = current_df_processed.pivot_table(index='experiment_elapsed_time', columns='tenant', values='value')
                            pivoted_df.dropna(how='all', axis=1, inplace=True)
                            pivoted_df.dropna(how='any', axis=0, inplace=True)

                            if pivoted_df.shape[1] < 2:
                                print(f"    Skipping covariance: Not enough tenants with data (need at least 2, found {pivoted_df.shape[1]}).")
                                continue

                            covariance_matrix = calculate_inter_tenant_covariance_per_metric(pivoted_df, time_col=None) # Already pivoted
                            
                            if not covariance_matrix.empty:
                                plot_title = f"Inter-Tenant Covariance - {METRIC_DISPLAY_NAMES.get(metric_name, metric_name)} ({phase_display_name})"
                                plot_filename = f"{metric_name}_{round_name}_{phase_name}_covariance_heatmap.png"
                                table_filename_base = f"{metric_name}_{round_name}_{phase_name}_covariance"
                                plot_covariance_heatmap(
                                    covariance_matrix, title=plot_title, output_dir=covariance_plots_dir, filename=plot_filename,
                                    tables_dir=covariance_tables_dir, table_filename_base=table_filename_base,
                                    metric_name=METRIC_DISPLAY_NAMES.get(metric_name, metric_name), round_name=round_name, phase_name=phase_display_name
                                )
                                print(f"    Covariance heatmap created.")
                        except Exception as e:
                            print(f"    Error during Covariance for {metric_name}, {round_name}, {phase_display_name}: {e}")
                            traceback.print_exc()
                else: # Consolidated
                    original_metric_df = phases_or_metric_df
                    # ... (similar logic for consolidated covariance) ...
                    print(f"  Processing round: {round_name} for metric: {metric_name} (Consolidated)")
                    if not isinstance(original_metric_df, pd.DataFrame) or original_metric_df.empty:
                        print(f"    Skipping: DataFrame is not valid or empty.")
                        continue
                    try:
                        current_df_processed = original_metric_df.copy()
                        current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                        pivoted_df = current_df_processed.pivot_table(index='experiment_elapsed_time', columns='tenant', values='value')
                        pivoted_df.dropna(how='all', axis=1, inplace=True)
                        pivoted_df.dropna(how='any', axis=0, inplace=True)
                        
                        if pivoted_df.shape[1] < 2:
                            print(f"    Skipping covariance: Not enough tenants with data (need at least 2, found {pivoted_df.shape[1]}).")
                            continue

                        covariance_matrix = calculate_inter_tenant_covariance_per_metric(pivoted_df, time_col=None)
                        if not covariance_matrix.empty:
                            plot_title = f"Inter-Tenant Covariance - {METRIC_DISPLAY_NAMES.get(metric_name, metric_name)}"
                            plot_filename = f"{metric_name}_{round_name}_covariance_heatmap.png"
                            table_filename_base = f"{metric_name}_{round_name}_covariance"
                            plot_covariance_heatmap(
                                covariance_matrix, title=plot_title, output_dir=covariance_plots_dir, filename=plot_filename,
                                tables_dir=covariance_tables_dir, table_filename_base=table_filename_base,
                                metric_name=METRIC_DISPLAY_NAMES.get(metric_name, metric_name), round_name=round_name
                            )
                            print(f"    Covariance heatmap created.")
                    except Exception as e:
                        print(f"    Error during Covariance for {metric_name}, {round_name} (Consolidated): {e}")
                        traceback.print_exc()
            gc.collect()

    # --- PCA Analysis ---
    if args.run_pca:
        print("\nRunning PCA Analysis...")
        pca_n_components_parsed = args.pca_n_components
        if pca_n_components_parsed is not None:
            try: pca_n_components_parsed = int(pca_n_components_parsed)
            except ValueError:
                try: 
                    pca_n_components_parsed = float(pca_n_components_parsed)
                    if not (0 < pca_n_components_parsed <= 1.0): pca_n_components_parsed = None
                except ValueError: pca_n_components_parsed = None
        
        for metric_name, rounds_or_phases_data in all_metrics_data.items():
            print(f"Processing metric: {metric_name}")
            # ... (similar iteration structure) ...
            for round_name, phases_or_metric_df in rounds_or_phases_data.items():
                data_for_pca = None
                context_suffix = ""
                current_phase_name_for_plot = None

                if run_per_phase_analysis:
                    # PCA per phase
                    for phase_name, original_phase_df in phases_or_metric_df.items():
                        phase_display_name = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
                        print(f"  Processing round: {round_name}, phase: {phase_display_name} for PCA on metric: {metric_name}")
                        if not isinstance(original_phase_df, pd.DataFrame) or original_phase_df.empty:
                            print(f"    Skipping PCA: DataFrame is not valid or empty.")
                            continue
                        
                        current_df_processed = original_phase_df.copy()
                        current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                        pivoted_df = current_df_processed.pivot_table(index='experiment_elapsed_time', columns='tenant', values='value').dropna()
                        
                        if pivoted_df.shape[0] < 2 or pivoted_df.shape[1] < 2 : # Need enough samples and features
                            print(f"    Skipping PCA: Not enough data after pivoting and NaN removal (shape: {pivoted_df.shape}).")
                            continue
                        data_for_pca = pivoted_df
                        context_suffix = f"_{phase_name}"
                        current_phase_name_for_plot = phase_display_name
                        
                        try:
                            pca_results, pca_components, explained_variance_ratio = perform_pca(
                                data_for_pca, n_components=pca_n_components_parsed, variance_threshold=args.pca_variance_threshold
                            )
                            if pca_results is not None and not pca_results.empty:
                                pca_results_filename = f"{metric_name}_{round_name}{context_suffix}_pca_results.csv"
                                export_to_csv(pca_results, pca_tables_dir, pca_results_filename)
                                pca_components_filename = f"{metric_name}_{round_name}{context_suffix}_pca_components.csv"
                                export_to_csv(pca_components, pca_tables_dir, pca_components_filename)
                                print(f"    PCA results and components saved.")

                                # Plotting
                                plot_pca_explained_variance(explained_variance_ratio, title=f"PCA Explained Variance - {METRIC_DISPLAY_NAMES.get(metric_name, metric_name)} ({current_phase_name_for_plot})",
                                                            output_dir=pca_plots_dir, filename=f"{metric_name}_{round_name}{context_suffix}_pca_explained_variance.png",
                                                            metric_name=METRIC_DISPLAY_NAMES.get(metric_name, metric_name), round_name=round_name, phase_name=current_phase_name_for_plot)
                                plot_pca_loadings_heatmap(pca_components, title=f"PCA Loadings - {METRIC_DISPLAY_NAMES.get(metric_name, metric_name)} ({current_phase_name_for_plot})",
                                                          output_dir=pca_plots_dir, filename=f"{metric_name}_{round_name}{context_suffix}_pca_loadings_heatmap.png",
                                                          metric_name=METRIC_DISPLAY_NAMES.get(metric_name, metric_name), round_name=round_name, phase_name=current_phase_name_for_plot)
                                if pca_results.shape[1] >= 2:
                                     plot_pca_biplot(pca_results, pca_components, title=f"PCA Biplot - {METRIC_DISPLAY_NAMES.get(metric_name, metric_name)} ({current_phase_name_for_plot})",
                                                    output_dir=pca_plots_dir, filename=f"{metric_name}_{round_name}{context_suffix}_pca_biplot.png",
                                                    metric_name=METRIC_DISPLAY_NAMES.get(metric_name, metric_name), round_name=round_name, phase_name=current_phase_name_for_plot)
                                print(f"    PCA plots created.")
                        except Exception as e:
                            print(f"    Error during PCA for {metric_name}, {round_name}, {current_phase_name_for_plot}: {e}")
                            traceback.print_exc()
                else: # Consolidated PCA
                    original_metric_df = phases_or_metric_df
                    print(f"  Processing round: {round_name} for PCA on metric: {metric_name} (Consolidated)")
                    if not isinstance(original_metric_df, pd.DataFrame) or original_metric_df.empty:
                        print(f"    Skipping PCA: DataFrame is not valid or empty.")
                        continue
                    current_df_processed = original_metric_df.copy()
                    current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                    pivoted_df = current_df_processed.pivot_table(index='experiment_elapsed_time', columns='tenant', values='value').dropna()

                    if pivoted_df.shape[0] < 2 or pivoted_df.shape[1] < 2 :
                        print(f"    Skipping PCA: Not enough data after pivoting and NaN removal (shape: {pivoted_df.shape}).")
                        continue
                    data_for_pca = pivoted_df
                    context_suffix = "_consolidated"
                    try:
                        # ... (PCA logic similar to per-phase, but with consolidated context_suffix and no phase_name)
                        pca_results, pca_components, explained_variance_ratio = perform_pca(
                            data_for_pca, n_components=pca_n_components_parsed, variance_threshold=args.pca_variance_threshold
                        )
                        if pca_results is not None and not pca_results.empty:
                            # ... (save and plot logic for consolidated)
                            print(f"    PCA (Consolidated) analysis complete.")
                    except Exception as e:
                        print(f"    Error during PCA (Consolidated) for {metric_name}, {round_name}: {e}")
                        traceback.print_exc()
            gc.collect()

    # --- ICA Analysis ---
    if args.run_ica:
        print("\nRunning ICA Analysis...")
        for metric_name, rounds_or_phases_data in all_metrics_data.items():
            print(f"Processing metric: {metric_name} for ICA")
            if not isinstance(rounds_or_phases_data, dict):
                print(f"  Skipping metric {metric_name}: Expected dict, got {type(rounds_or_phases_data)}.")
                continue

            for round_name, phases_or_metric_df in rounds_or_phases_data.items():
                if run_per_phase_analysis:
                    if not isinstance(phases_or_metric_df, dict):
                        print(f"  Skipping round {round_name} for {metric_name}: Expected dict, got {type(phases_or_metric_df)}.")
                        continue
                    for phase_name, original_phase_df in phases_or_metric_df.items():
                        phase_display_name = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
                        print(f"  Processing round: {round_name}, phase: {phase_display_name} for ICA on metric: {metric_name}")
                        
                        if not isinstance(original_phase_df, pd.DataFrame) or original_phase_df.empty:
                            print(f"    Skipping ICA: DataFrame is not valid or empty for {metric_name}, {round_name}, {phase_display_name}.")
                            continue
                        
                        if 'experiment_elapsed_time' not in original_phase_df.columns:
                            print(f"    Skipping ICA for {metric_name}, {round_name}, {phase_display_name}: 'experiment_elapsed_time' column missing.")
                            continue

                        try:
                            current_df_processed = original_phase_df.copy()
                            current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                            
                            pivoted_df = current_df_processed.pivot_table(
                                index='experiment_elapsed_time', 
                                columns='tenant', 
                                values='value'
                            )
                            pivoted_df.dropna(axis=1, how='all', inplace=True) # Drop tenant columns if all values are NaN
                            pivoted_df.dropna(axis=0, how='any', inplace=True)   # Drop rows with any NaN

                            if pivoted_df.shape[0] < 2 or pivoted_df.shape[1] < 2: 
                                print(f"    Skipping ICA: Not enough data after pivoting and NaN removal (shape: {pivoted_df.shape}) for {metric_name}, {round_name}, {phase_display_name}.")
                                continue

                            ica_transformed_data, ica_mixing_matrix, ica_source_signals = perform_ica(
                                pivoted_df, 
                                n_components=args.ica_n_components, 
                                random_state=42, 
                                max_iter=args.ica_max_iter
                            )

                            if ica_source_signals is not None and not ica_source_signals.empty and \
                               ica_mixing_matrix is not None and not ica_mixing_matrix.empty:
                                
                                context_suffix = f"_{phase_name}"
                                metric_display = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)

                                # Save ICA source signals
                                ica_sources_filename = f"{metric_name}_{round_name}{context_suffix}_ica_sources.csv"
                                export_to_csv(ica_source_signals, ica_tables_dir, ica_sources_filename)
                                print(f"    ICA source signals saved to {ica_sources_filename}")

                                # Save ICA mixing matrix (components)
                                ica_components_filename = f"{metric_name}_{round_name}{context_suffix}_ica_components.csv"
                                export_to_csv(ica_mixing_matrix, ica_tables_dir, ica_components_filename)
                                print(f"    ICA mixing matrix saved to {ica_components_filename}")

                                # Plot ICA components heatmap (mixing matrix)
                                plot_title_components = f"ICA Components (Mixing Matrix) - {metric_display} ({phase_display_name})"
                                plot_filename_components = f"{metric_name}_{round_name}{context_suffix}_ica_components_heatmap.png"
                                plot_ica_components_heatmap(
                                    ica_mixing_matrix,
                                    title=plot_title_components,
                                    output_dir=ica_plots_dir,
                                    filename=plot_filename_components,
                                    metric_name=metric_display,
                                    round_name=round_name,
                                    phase_name=phase_display_name
                                )
                                print(f"    ICA components heatmap created for {metric_name}, {round_name}, {phase_display_name}.")

                                # Plot ICA source signals (time series)
                                plot_title_sources_ts = f"ICA Source Signals - {metric_display} ({phase_display_name})"
                                plot_filename_sources_ts = f"{metric_name}_{round_name}{context_suffix}_ica_sources_timeseries.png"
                                plot_ica_time_series(
                                    ica_source_signals, 
                                    title=plot_title_sources_ts,
                                    output_dir=ica_plots_dir,
                                    filename=plot_filename_sources_ts,
                                    metric_name=metric_display,
                                    round_name=round_name,
                                    phase_name=phase_display_name
                                )
                                print(f"    ICA source signals time series plot created for {metric_name}, {round_name}, {phase_display_name}.")
                                
                                # Scatter plot of first two ICA source signals
                                if ica_source_signals.shape[1] >= 2:
                                    plot_title_scatter = f"ICA Source Scatter (IC1 vs IC2) - {metric_display} ({phase_display_name})"
                                    plot_filename_scatter = f"{metric_name}_{round_name}{context_suffix}_ica_scatter_ic1_ic2.png"
                                    plot_ica_scatter(
                                        ica_source_signals, 
                                        component_1=0, 
                                        component_2=1, 
                                        title=plot_title_scatter,
                                        output_dir=ica_plots_dir,
                                        filename=plot_filename_scatter,
                                        metric_name=metric_display,
                                        round_name=round_name,
                                        phase_name=phase_display_name
                                    )
                                    print(f"    ICA scatter plot (IC1 vs IC2) created for {metric_name}, {round_name}, {phase_display_name}.")
                            else:
                                print(f"    ICA did not produce valid results for {metric_name}, {round_name}, {phase_display_name}.")
                        except Exception as e:
                            print(f"    Error during ICA for {metric_name}, {round_name}, {phase_display_name}: {e}")
                            import traceback
                            traceback.print_exc()
                else: # Consolidated ICA
                    original_metric_df = phases_or_metric_df
                    print(f"  Processing round: {round_name} for ICA on metric: {metric_name} (Consolidated)")

                    if not isinstance(original_metric_df, pd.DataFrame) or original_metric_df.empty:
                        print(f"    Skipping ICA: DataFrame is not valid or empty for {metric_name}, {round_name} (Consolidated).")
                        continue
                    
                    if 'experiment_elapsed_time' not in original_metric_df.columns:
                        print(f"    Skipping ICA for {metric_name}, {round_name} (Consolidated): 'experiment_elapsed_time' column missing.")
                        continue

                    try:
                        current_df_processed = original_metric_df.copy()
                        current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                        
                        pivoted_df = current_df_processed.pivot_table(
                            index='experiment_elapsed_time', 
                            columns='tenant', 
                            values='value'
                        )
                        pivoted_df.dropna(axis=1, how='all', inplace=True)
                        pivoted_df.dropna(axis=0, how='any', inplace=True)

                        if pivoted_df.shape[0] < 2 or pivoted_df.shape[1] < 2:
                            print(f"    Skipping ICA: Not enough data after pivoting and NaN removal (shape: {pivoted_df.shape}) for {metric_name}, {round_name} (Consolidated).")
                            continue
                        
                        ica_transformed_data, ica_mixing_matrix, ica_source_signals = perform_ica(
                            pivoted_df, 
                            n_components=args.ica_n_components, 
                            random_state=42,
                            max_iter=args.ica_max_iter
                        )

                        if ica_source_signals is not None and not ica_source_signals.empty and \
                           ica_mixing_matrix is not None and not ica_mixing_matrix.empty:
                            
                            context_suffix = "_consolidated"
                            metric_display = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)

                            ica_sources_filename = f"{metric_name}_{round_name}{context_suffix}_ica_sources.csv"
                            export_to_csv(ica_source_signals, ica_tables_dir, ica_sources_filename)
                            print(f"    ICA source signals saved to {ica_sources_filename}")

                            ica_components_filename = f"{metric_name}_{round_name}{context_suffix}_ica_components.csv"
                            export_to_csv(ica_mixing_matrix, ica_tables_dir, ica_components_filename)
                            print(f"    ICA mixing matrix saved to {ica_components_filename}")
                            
                            plot_title_components = f"ICA Components (Mixing Matrix) - {metric_display} (Consolidated)"
                            plot_filename_components = f"{metric_name}_{round_name}{context_suffix}_ica_components_heatmap.png"
                            plot_ica_components_heatmap(
                                ica_mixing_matrix,
                                title=plot_title_components,
                                output_dir=ica_plots_dir,
                                filename=plot_filename_components,
                                metric_name=metric_display,
                                round_name=round_name
                            )
                            print(f"    ICA components heatmap created for {metric_name}, {round_name} (Consolidated).")

                            plot_title_sources_ts = f"ICA Source Signals - {metric_display} (Consolidated)"
                            plot_filename_sources_ts = f"{metric_name}_{round_name}{context_suffix}_ica_sources_timeseries.png"
                            plot_ica_time_series(
                                ica_source_signals,
                                title=plot_title_sources_ts,
                                output_dir=ica_plots_dir,
                                filename=plot_filename_sources_ts,
                                metric_name=metric_display,
                                round_name=round_name
                            )
                            print(f"    ICA source signals time series plot created for {metric_name}, {round_name} (Consolidated).")

                            if ica_source_signals.shape[1] >= 2:
                                plot_title_scatter = f"ICA Source Scatter (IC1 vs IC2) - {metric_display} (Consolidated)"
                                plot_filename_scatter = f"{metric_name}_{round_name}{context_suffix}_ica_scatter_ic1_ic2.png"
                                plot_ica_scatter(
                                    ica_source_signals,
                                    component_1=0,
                                    component_2=1,
                                    title=plot_title_scatter,
                                    output_dir=ica_plots_dir,
                                    filename=plot_filename_scatter,
                                    metric_name=metric_display,
                                    round_name=round_name
                                )
                                print(f"    ICA scatter plot (IC1 vs IC2) created for {metric_name}, {round_name} (Consolidated).")
                        else:
                            print(f"    ICA did not produce valid results for {metric_name}, {round_name} (Consolidated).")
                    except Exception as e:
                        print(f"    Error during ICA for {metric_name}, {round_name} (Consolidated): {e}")
                        import traceback
                        traceback.print_exc()
            gc.collect() # After all rounds/phases for a metric

    # --- Compare PCA and ICA ---
    if args.compare_pca_ica:
        print("\nComparing PCA and ICA top features...")
        # This section requires that PCA and ICA have been run and their results (components/loadings)
        # are stored or accessible. The current script saves them to CSVs.
        # We will iterate through metrics, rounds, and phases, load the respective results,
        # compare them, and save comparison tables/plots.

        for metric_name, rounds_or_phases_data in all_metrics_data.items():
            metric_display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
            print(f"Processing metric: {metric_display_name} for PCA/ICA Comparison")

            if not isinstance(rounds_or_phases_data, dict):
                print(f"  Skipping metric {metric_name} for comparison: Expected dict, got {type(rounds_or_phases_data)}.")
                continue

            for round_name, phases_or_metric_df in rounds_or_phases_data.items():
                if run_per_phase_analysis:
                    if not isinstance(phases_or_metric_df, dict):
                        print(f"  Skipping round {round_name} for {metric_name} (comparison): Expected dict, got {type(phases_or_metric_df)}.")
                        continue
                    for phase_name, _ in phases_or_metric_df.items(): # We don't need original_phase_df here, will load from files
                        phase_display_name = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
                        print(f"  Comparing PCA/ICA for round: {round_name}, phase: {phase_display_name}, metric: {metric_display_name}")
                        
                        context_suffix = f"_{phase_name}"
                        pca_components_file = os.path.join(pca_tables_dir, f"{metric_name}_{round_name}{context_suffix}_pca_components.csv")
                        ica_components_file = os.path.join(ica_tables_dir, f"{metric_name}_{round_name}{context_suffix}_ica_components.csv") # Mixing matrix

                        if os.path.exists(pca_components_file) and os.path.exists(ica_components_file):
                            try:
                                pca_components_df = pd.read_csv(pca_components_file, index_col=0)
                                ica_mixing_matrix_df = pd.read_csv(ica_components_file, index_col=0)

                                if pca_components_df.empty or ica_mixing_matrix_df.empty:
                                    print(f"    Skipping comparison for {metric_name}, {round_name}, {phase_name}: PCA or ICA components are empty.")
                                    continue

                                # Ensure components are features x n_components for PCA, and features x n_components for ICA mixing matrix
                                # PCA components from perform_pca are already in this format (features are index, PCs are columns)
                                # ICA mixing matrix from perform_ica is features x n_components (features are index, ICs are columns)

                                # Get top features for PCA
                                pca_top_features = get_top_features_per_component(pca_components_df, top_n=args.n_top_features_comparison)
                                pca_top_features_df = pd.DataFrame.from_dict(pca_top_features, orient='index')
                                pca_top_features_filename = f"{metric_name}_{round_name}{context_suffix}_pca_top_features_comparison.csv"
                                export_to_csv(pca_top_features_df, comparison_tables_dir, pca_top_features_filename)
                                print(f"    PCA top features saved to {pca_top_features_filename}")

                                # Get top features for ICA (based on magnitude of mixing matrix coefficients)
                                ica_top_features = get_top_features_per_component(ica_mixing_matrix_df, top_n=args.n_top_features_comparison)
                                ica_top_features_df = pd.DataFrame.from_dict(ica_top_features, orient='index')
                                ica_top_features_filename = f"{metric_name}_{round_name}{context_suffix}_ica_top_features_comparison.csv"
                                export_to_csv(ica_top_features_df, comparison_tables_dir, ica_top_features_filename)
                                print(f"    ICA top features saved to {ica_top_features_filename}")
                                
                                # Basic comparison: Overlap in top features
                                comparison_summary = {}
                                for pc_idx, pc_features in pca_top_features.items():
                                    for ic_idx, ic_features in ica_top_features.items():
                                        overlap = len(set(pc_features).intersection(set(ic_features)))
                                        comparison_summary[f"{pc_idx}_vs_{ic_idx}"] = {
                                            'pca_component': pc_idx,
                                            'ica_component': ic_idx,
                                            'overlapping_features_count': overlap,
                                            'pca_top_features': ", ".join(pc_features),
                                            'ica_top_features': ", ".join(ic_features)
                                        }
                                
                                comparison_summary_df = pd.DataFrame.from_dict(comparison_summary, orient='index')
                                comparison_filename = f"{metric_name}_{round_name}{context_suffix}_pca_ica_feature_overlap.csv"
                                export_to_csv(comparison_summary_df, comparison_tables_dir, comparison_filename)
                                print(f"    PCA-ICA feature overlap summary saved to {comparison_filename}")

                                # TODO: Add a plot for comparison if meaningful (e.g., upset plot for feature overlap, or comparative heatmaps)
                                # For now, focusing on table-based comparison.

                            except Exception as e:
                                print(f"    Error comparing PCA/ICA for {metric_name}, {round_name}, {phase_name}: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print(f"    Skipping comparison for {metric_name}, {round_name}, {phase_name}: PCA or ICA component files not found.")
                else: # Consolidated PCA/ICA Comparison
                    print(f"  Comparing PCA/ICA for round: {round_name}, metric: {metric_display_name} (Consolidated)")
                    context_suffix = "_consolidated"
                    pca_components_file = os.path.join(pca_tables_dir, f"{metric_name}_{round_name}{context_suffix}_pca_components.csv")
                    ica_components_file = os.path.join(ica_tables_dir, f"{metric_name}_{round_name}{context_suffix}_ica_components.csv") # Mixing matrix

                    if os.path.exists(pca_components_file) and os.path.exists(ica_components_file):
                        try:
                            pca_components_df = pd.read_csv(pca_components_file, index_col=0)
                            ica_mixing_matrix_df = pd.read_csv(ica_components_file, index_col=0)

                            if pca_components_df.empty or ica_mixing_matrix_df.empty:
                                print(f"    Skipping comparison for {metric_name}, {round_name} (Consolidated): PCA or ICA components are empty.")
                                continue

                            pca_top_features = get_top_features_per_component(pca_components_df, top_n=args.n_top_features_comparison)
                            pca_top_features_df = pd.DataFrame.from_dict(pca_top_features, orient='index')
                            pca_top_features_filename = f"{metric_name}_{round_name}{context_suffix}_pca_top_features_comparison.csv"
                            export_to_csv(pca_top_features_df, comparison_tables_dir, pca_top_features_filename)
                            print(f"    PCA top features saved to {pca_top_features_filename}")

                            ica_top_features = get_top_features_per_component(ica_mixing_matrix_df, top_n=args.n_top_features_comparison)
                            ica_top_features_df = pd.DataFrame.from_dict(ica_top_features, orient='index')
                            ica_top_features_filename = f"{metric_name}_{round_name}{context_suffix}_ica_top_features_comparison.csv"
                            export_to_csv(ica_top_features_df, comparison_tables_dir, ica_top_features_filename)
                            print(f"    ICA top features saved to {ica_top_features_filename}")

                            comparison_summary = {}
                            for pc_idx, pc_features in pca_top_features.items():
                                for ic_idx, ic_features in ica_top_features.items():
                                    overlap = len(set(pc_features).intersection(set(ic_features)))
                                    comparison_summary[f"{pc_idx}_vs_{ic_idx}"] = {
                                        'pca_component': pc_idx,
                                        'ica_component': ic_idx,
                                        'overlapping_features_count': overlap,
                                        'pca_top_features': ", ".join(pc_features),
                                        'ica_top_features': ", ".join(ic_features)
                                    }
                            
                            comparison_summary_df = pd.DataFrame.from_dict(comparison_summary, orient='index')
                            comparison_filename = f"{metric_name}_{round_name}{context_suffix}_pca_ica_feature_overlap.csv"
                            export_to_csv(comparison_summary_df, comparison_tables_dir, comparison_filename)
                            print(f"    PCA-ICA feature overlap summary saved to {comparison_filename}")

                        except Exception as e:
                            print(f"    Error comparing PCA/ICA for {metric_name}, {round_name} (Consolidated): {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"    Skipping comparison for {metric_name}, {round_name} (Consolidated): PCA or ICA component files not found.")
            gc.collect()

    # --- Causality Analysis (SEM, TE, CCM, Granger) ---
    # Each causality method will have its own block similar to PCA/ICA
    # SEM
    if args.run_sem:
        print("\nRunning Structural Equation Modeling (SEM)...")
        for metric_name, rounds_or_phases_data in all_metrics_data.items():
            print(f"Processing metric: {metric_name} for SEM")
            if not isinstance(rounds_or_phases_data, dict):
                print(f"  Skipping metric {metric_name}: Expected dict, got {type(rounds_or_phases_data)}.")
                continue

            for round_name, phases_or_metric_df in rounds_or_phases_data.items():
                if run_per_phase_analysis:
                    if not isinstance(phases_or_metric_df, dict):
                        print(f"  Skipping round {round_name} for {metric_name}: Expected dict, got {type(phases_or_metric_df)}.")
                        continue
                    for phase_name, original_phase_df in phases_or_metric_df.items():
                        phase_display_name = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
                        metric_display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
                        print(f"  Processing round: {round_name}, phase: {phase_display_name} for SEM on metric: {metric_display_name}")
                        
                        if not isinstance(original_phase_df, pd.DataFrame) or original_phase_df.empty:
                            print(f"    Skipping SEM: DataFrame is not valid or empty for {metric_name}, {round_name}, {phase_name}.")
                            continue
                        
                        if 'experiment_elapsed_time' not in original_phase_df.columns:
                            print(f"    Skipping SEM for {metric_name}, {round_name}, {phase_name}: 'experiment_elapsed_time' column missing.")
                            continue

                        try:
                            current_df_processed = original_phase_df.copy()
                            current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                            
                            pivoted_df = current_df_processed.pivot_table(
                                index='experiment_elapsed_time', 
                                columns='tenant', 
                                values='value'
                            )
                            pivoted_df.dropna(axis=1, how='all', inplace=True) # Drop tenant columns if all values are NaN
                            pivoted_df.dropna(axis=0, how='any', inplace=True)   # Drop rows with any NaN

                            if pivoted_df.shape[0] < 2 or pivoted_df.shape[1] < 2: 
                                print(f"    Skipping SEM: Not enough data after pivoting and NaN removal (shape: {pivoted_df.shape}) for {metric_name}, {round_name}, {phase_name}.")
                                continue
                            
                            # SEM often starts from a correlation matrix to define the model
                            correlation_matrix_for_sem = pivoted_df.corr(method='pearson') # Using pearson as a common default

                            model_spec_df = create_sem_model_from_correlation(
                                correlation_matrix_for_sem, 
                                threshold=args.sem_correlation_threshold
                            )

                            if model_spec_df is None or model_spec_df.empty:
                                print(f"    Skipping SEM: No model specification generated based on correlation threshold for {metric_name}, {round_name}, {phase_name}.")
                                continue
                            
                            print(f"    Generated SEM model spec with {len(model_spec_df)} paths.")
                            # Save model spec
                            model_spec_filename = f"{metric_name}_{round_name}_{phase_name}_sem_model_spec.csv"
                            export_to_csv(model_spec_df, sem_tables_dir, model_spec_filename)
                            print(f"    SEM model specification saved to {model_spec_filename}")

                            sem_results, fit_summary = perform_sem_analysis(
                                data=pivoted_df, 
                                model_spec_df=model_spec_df, 
                                standardize=args.sem_standardize
                            )

                            context_suffix = f"_{phase_name}"
                            title_base = f"SEM - {metric_display_name} ({phase_display_name})"

                            if sem_results is not None and not sem_results.empty:
                                sem_results_filename = f"{metric_name}_{round_name}{context_suffix}_sem_results.csv"
                                export_to_csv(sem_results, sem_tables_dir, sem_results_filename)
                                print(f"    SEM results saved to {sem_results_filename}")

                                # Plot SEM path diagram
                                plot_sem_path_diagram(
                                    sem_results, 
                                    title=f"{title_base} - Path Diagram",
                                    output_dir=sem_plots_dir, 
                                    filename=f"{metric_name}_{round_name}{context_suffix}_sem_path_diagram.png",
                                    metric_name=metric_display_name, round_name=round_name, phase_name=phase_display_name
                                )
                                print(f"    SEM path diagram created.")

                                # Plot SEM coefficient heatmap
                                plot_sem_coefficient_heatmap(
                                    sem_results, 
                                    title=f"{title_base} - Coefficients",
                                    output_dir=sem_plots_dir, 
                                    filename=f"{metric_name}_{round_name}{context_suffix}_sem_coefficients_heatmap.png",
                                    metric_name=metric_display_name, round_name=round_name, phase_name=phase_display_name
                                )
                                print(f"    SEM coefficient heatmap created.")
                            else:
                                print(f"    SEM analysis did not produce significant results for {metric_name}, {round_name}, {phase_name}.")

                            if fit_summary is not None and not fit_summary.empty:
                                sem_fit_filename = f"{metric_name}_{round_name}{context_suffix}_sem_fit_indices.csv"
                                export_to_csv(fit_summary, sem_tables_dir, sem_fit_filename)
                                print(f"    SEM fit indices saved to {sem_fit_filename}")
                                
                                # Plot SEM fit indices (if the plotting function exists and is relevant)
                                # Assuming plot_sem_fit_indices takes the fit_summary DataFrame
                                plot_sem_fit_indices(
                                    fit_summary, 
                                    title=f"{title_base} - Fit Indices",
                                    output_dir=sem_plots_dir, 
                                    filename=f"{metric_name}_{round_name}{context_suffix}_sem_fit_indices.png",
                                    metric_name=metric_display_name, round_name=round_name, phase_name=phase_display_name
                                )
                                print(f"    SEM fit indices plot created.")
                            else:
                                print(f"    SEM analysis did not produce fit summary for {metric_name}, {round_name}, {phase_name}.")

                        except Exception as e:
                            print(f"    Error during SEM for {metric_name}, {round_name}, {phase_name}: {e}")
                            import traceback
                            traceback.print_exc()
                else: # Consolidated SEM
                    original_metric_df = phases_or_metric_df
                    metric_display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
                    print(f"  Processing round: {round_name} for SEM on metric: {metric_display_name} (Consolidated)")

                    if not isinstance(original_metric_df, pd.DataFrame) or original_metric_df.empty:
                        print(f"    Skipping SEM: DataFrame is not valid or empty for {metric_name}, {round_name} (Consolidated).")
                        continue
                    
                    if 'experiment_elapsed_time' not in original_metric_df.columns:
                        print(f"    Skipping SEM for {metric_name}, {round_name} (Consolidated): 'experiment_elapsed_time' column missing.")
                        continue

                    try:
                        current_df_processed = original_metric_df.copy()
                        current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                        
                        pivoted_df = current_df_processed.pivot_table(
                            index='experiment_elapsed_time', 
                            columns='tenant', 
                            values='value'
                        )
                        pivoted_df.dropna(axis=1, how='all', inplace=True)
                        pivoted_df.dropna(axis=0, how='any', inplace=True)

                        if pivoted_df.shape[0] < 2 or pivoted_df.shape[1] < 2:
                            print(f"    Skipping SEM: Not enough data after pivoting and NaN removal (shape: {pivoted_df.shape}) for {metric_name}, {round_name} (Consolidated).")
                            continue

                        correlation_matrix_for_sem = pivoted_df.corr(method='pearson')
                        model_spec_df = create_sem_model_from_correlation(
                            correlation_matrix_for_sem, 
                            threshold=args.sem_correlation_threshold
                        )

                        if model_spec_df is None or model_spec_df.empty:
                            print(f"    Skipping SEM: No model specification generated for {metric_name}, {round_name} (Consolidated).")
                            continue
                        
                        print(f"    Generated SEM model spec with {len(model_spec_df)} paths (Consolidated).")
                        model_spec_filename = f"{metric_name}_{round_name}_consolidated_sem_model_spec.csv"
                        export_to_csv(model_spec_df, sem_tables_dir, model_spec_filename)
                        print(f"    SEM model specification saved to {model_spec_filename}")

                        sem_results, fit_summary = perform_sem_analysis(
                            data=pivoted_df, 
                            model_spec_df=model_spec_df, 
                            standardize=args.sem_standardize
                        )
                        
                        context_suffix = "_consolidated"
                        title_base = f"SEM - {metric_display_name} (Consolidated)"

                        if sem_results is not None and not sem_results.empty:
                            sem_results_filename = f"{metric_name}_{round_name}{context_suffix}_sem_results.csv"
                            export_to_csv(sem_results, sem_tables_dir, sem_results_filename)
                            print(f"    SEM results saved to {sem_results_filename}")

                            plot_sem_path_diagram(
                                sem_results, 
                                title=f"{title_base} - Path Diagram",
                                output_dir=sem_plots_dir, 
                                filename=f"{metric_name}_{round_name}{context_suffix}_sem_path_diagram.png",
                                metric_name=metric_display_name, round_name=round_name
                            )
                            print(f"    SEM path diagram created.")

                            plot_sem_coefficient_heatmap(
                                sem_results, 
                                title=f"{title_base} - Coefficients",
                                output_dir=sem_plots_dir, 
                                filename=f"{metric_name}_{round_name}{context_suffix}_sem_coefficients_heatmap.png",
                                metric_name=metric_display_name, round_name=round_name
                            )
                            print(f"    SEM coefficient heatmap created.")
                        else:
                            print(f"    SEM analysis did not produce significant results for {metric_name}, {round_name} (Consolidated).")

                        if fit_summary is not None and not fit_summary.empty:
                            sem_fit_filename = f"{metric_name}_{round_name}{context_suffix}_sem_fit_indices.csv"
                            export_to_csv(fit_summary, sem_tables_dir, sem_fit_filename)
                            print(f"    SEM fit indices saved to {sem_fit_filename}")
                            
                            plot_sem_fit_indices(
                                fit_summary, 
                                title=f"{title_base} - Fit Indices",
                                output_dir=sem_plots_dir, 
                                filename=f"{metric_name}_{round_name}{context_suffix}_sem_fit_indices.png",
                                metric_name=metric_display_name, round_name=round_name
                            )
                            print(f"    SEM fit indices plot created.")
                        else:
                            print(f"    SEM analysis did not produce fit summary for {metric_name}, {round_name} (Consolidated).")

                    except Exception as e:
                        print(f"    Error during SEM for {metric_name}, {round_name} (Consolidated): {e}")
                        import traceback
                        traceback.print_exc()
            gc.collect() # After all rounds/phases for a metric

    # Transfer Entropy
    if args.run_transfer_entropy:
        print("\nRunning Transfer Entropy Analysis...")
        for metric_name, rounds_or_phases_data in all_metrics_data.items():
            print(f"Processing metric: {metric_name} for Transfer Entropy")
            if not isinstance(rounds_or_phases_data, dict):
                print(f"  Skipping metric {metric_name}: Expected dict, got {type(rounds_or_phases_data)}.")
                continue

            for round_name, phases_or_metric_df in rounds_or_phases_data.items():
                if run_per_phase_analysis:
                    if not isinstance(phases_or_metric_df, dict):
                        print(f"  Skipping round {round_name} for {metric_name}: Expected dict, got {type(phases_or_metric_df)}.")
                        continue
                    for phase_name, original_phase_df in phases_or_metric_df.items():
                        phase_display_name = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
                        metric_display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
                        print(f"  Processing round: {round_name}, phase: {phase_display_name} for TE on metric: {metric_display_name}")
                        
                        if not isinstance(original_phase_df, pd.DataFrame) or original_phase_df.empty:
                            print(f"    Skipping TE: DataFrame is not valid or empty for {metric_name}, {round_name}, {phase_name}.")
                            continue
                        
                        if 'experiment_elapsed_time' not in original_phase_df.columns:
                            print(f"    Skipping TE for {metric_name}, {round_name}, {phase_name}: 'experiment_elapsed_time' column missing.")
                            continue

                        try:
                            current_df_processed = original_phase_df.copy()
                            current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                            
                            pivoted_df = current_df_processed.pivot_table(
                                index='experiment_elapsed_time', 
                                columns='tenant', 
                                values='value'
                            )
                            pivoted_df.dropna(axis=1, how='all', inplace=True) 
                            pivoted_df.dropna(axis=0, how='any', inplace=True)   

                            if pivoted_df.shape[0] < args.te_lag + 1 or pivoted_df.shape[1] < 2: 
                                print(f"    Skipping TE: Not enough data after pivoting (shape: {pivoted_df.shape}, lag: {args.te_lag}) for {metric_name}, {round_name}, {phase_name}.")
                                continue
                            
                            te_matrix = calculate_pairwise_transfer_entropy(
                                data=pivoted_df,
                                k=args.te_lag 
                            )

                            context_suffix = f"_{phase_name}"
                            title_base = f"Transfer Entropy (Lag {args.te_lag}) - {metric_display_name} ({phase_display_name})"

                            if te_matrix is not None and not te_matrix.empty:
                                te_results_filename = f"{metric_name}_{round_name}{context_suffix}_transfer_entropy_k{args.te_lag}.csv"
                                export_to_csv(te_matrix, te_tables_dir, te_results_filename)
                                print(f"    Transfer Entropy results saved to {te_results_filename}")

                                # Plot TE heatmap
                                plot_transfer_entropy_heatmap(
                                    te_matrix,
                                    title=title_base,
                                    output_dir=te_plots_dir,
                                    filename=f"{metric_name}_{round_name}{context_suffix}_te_heatmap_k{args.te_lag}.png",
                                    metric_name=metric_display_name, round_name=round_name, phase_name=phase_display_name
                                )
                                print(f"    Transfer Entropy heatmap created.")

                                # Plot TE network
                                plot_transfer_entropy_network(
                                    te_matrix,
                                    title=f"{title_base} - Network (Threshold {args.te_threshold})",
                                    output_dir=te_plots_dir,
                                    filename=f"{metric_name}_{round_name}{context_suffix}_te_network_k{args.te_lag}_thresh{args.te_threshold}.png",
                                    threshold=args.te_threshold,
                                    metric_name=metric_display_name, round_name=round_name, phase_name=phase_display_name
                                )
                                print(f"    Transfer Entropy network plot created.")
                            else:
                                print(f"    Transfer Entropy analysis did not produce results for {metric_name}, {round_name}, {phase_name}.")

                        except Exception as e:
                            print(f"    Error during Transfer Entropy for {metric_name}, {round_name}, {phase_name}: {e}")
                            import traceback
                            traceback.print_exc()
                else: # Consolidated TE
                    original_metric_df = phases_or_metric_df
                    metric_display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
                    print(f"  Processing round: {round_name} for TE on metric: {metric_display_name} (Consolidated)")

                    if not isinstance(original_metric_df, pd.DataFrame) or original_metric_df.empty:
                        print(f"    Skipping TE: DataFrame is not valid or empty for {metric_name}, {round_name} (Consolidated).")
                        continue
                    
                    if 'experiment_elapsed_time' not in original_metric_df.columns:
                        print(f"    Skipping TE for {metric_name}, {round_name} (Consolidated): 'experiment_elapsed_time' column missing.")
                        continue
                    
                    try:
                        current_df_processed = original_metric_df.copy()
                        current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                        
                        pivoted_df = current_df_processed.pivot_table(
                            index='experiment_elapsed_time', 
                            columns='tenant', 
                            values='value'
                        )
                        pivoted_df.dropna(axis=1, how='all', inplace=True)
                        pivoted_df.dropna(axis=0, how='any', inplace=True)

                        if pivoted_df.shape[0] < args.te_lag + 1 or pivoted_df.shape[1] < 2:
                            print(f"    Skipping TE: Not enough data after pivoting (shape: {pivoted_df.shape}, lag: {args.te_lag}) for {metric_name}, {round_name} (Consolidated).")
                            continue
                        
                        te_matrix = calculate_pairwise_transfer_entropy(
                            data=pivoted_df,
                            k=args.te_lag
                        )

                        context_suffix = "_consolidated"
                        title_base = f"Transfer Entropy (Lag {args.te_lag}) - {metric_display_name} (Consolidated)"

                        if te_matrix is not None and not te_matrix.empty:
                            te_results_filename = f"{metric_name}_{round_name}{context_suffix}_transfer_entropy_k{args.te_lag}.csv"
                            export_to_csv(te_matrix, te_tables_dir, te_results_filename)
                            print(f"    Transfer Entropy results saved to {te_results_filename}")

                            plot_transfer_entropy_heatmap(
                                te_matrix,
                                title=title_base,
                                output_dir=te_plots_dir,
                                filename=f"{metric_name}_{round_name}{context_suffix}_te_heatmap_k{args.te_lag}.png",
                                metric_name=metric_display_name, round_name=round_name
                            )
                            print(f"    Transfer Entropy heatmap created.")

                            plot_transfer_entropy_network(
                                te_matrix,
                                title=f"{title_base} - Network (Threshold {args.te_threshold})",
                                output_dir=te_plots_dir,
                                filename=f"{metric_name}_{round_name}{context_suffix}_te_network_k{args.te_lag}_thresh{args.te_threshold}.png",
                                threshold=args.te_threshold,
                                metric_name=metric_display_name, round_name=round_name
                            )
                            print(f"    Transfer Entropy network plot created.")
                        else:
                            print(f"    Transfer Entropy analysis did not produce results for {metric_name}, {round_name} (Consolidated).")

                    except Exception as e:
                        print(f"    Error during Transfer Entropy for {metric_name}, {round_name} (Consolidated): {e}")
                        import traceback
                        traceback.print_exc()
            gc.collect() # After all rounds/phases for a metric

    # Convergent Cross Mapping (CCM)
    if args.run_ccm:
        print("\nRunning Convergent Cross Mapping (CCM)...")
        for metric_name, rounds_or_phases_data in all_metrics_data.items():
            print(f"Processing metric: {metric_name} for CCM")
            if not isinstance(rounds_or_phases_data, dict):
                print(f"  Skipping metric {metric_name}: Expected dict, got {type(rounds_or_phases_data)}.")
                continue

            for round_name, phases_or_metric_df in rounds_or_phases_data.items():
                if run_per_phase_analysis:
                    if not isinstance(phases_or_metric_df, dict):
                        print(f"  Skipping round {round_name} for {metric_name}: Expected dict, got {type(phases_or_metric_df)}.")
                        continue
                    for phase_name, original_phase_df in phases_or_metric_df.items():
                        phase_display_name = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
                        metric_display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
                        print(f"  Processing round: {round_name}, phase: {phase_display_name} for CCM on metric: {metric_display_name}")
                        
                        if not isinstance(original_phase_df, pd.DataFrame) or original_phase_df.empty:
                            print(f"    Skipping CCM: DataFrame is not valid or empty for {metric_name}, {round_name}, {phase_name}.")
                            continue
                        
                        if 'experiment_elapsed_time' not in original_phase_df.columns:
                            print(f"    Skipping CCM for {metric_name}, {round_name}, {phase_name}: 'experiment_elapsed_time' column missing.")
                            continue

                        try:
                            current_df_processed = original_phase_df.copy()
                            current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                            
                            pivoted_df = current_df_processed.pivot_table(
                                index='experiment_elapsed_time', 
                                columns='tenant', 
                                values='value'
                            )
                            pivoted_df.dropna(axis=1, how='all', inplace=True) 
                            pivoted_df.dropna(axis=0, how='any', inplace=True)   

                            # Check if enough data points for max embedding dimension
                            max_embed_dim = max(args.ccm_embed_dimensions) if args.ccm_embed_dimensions else 1
                            required_len = (max_embed_dim - 1) * args.ccm_tau + 1
                            if pivoted_df.shape[0] < required_len or pivoted_df.shape[1] < 2:
                                print(f"    Skipping CCM: Not enough data after pivoting (shape: {pivoted_df.shape}, E_max: {max_embed_dim}, tau: {args.ccm_tau}, required_len: {required_len}) for {metric_name}, {round_name}, {phase_name}.")
                                continue
                            
                            ccm_results_dict = calculate_pairwise_ccm(
                                data=pivoted_df,
                                E_range=args.ccm_embed_dimensions,
                                tau=args.ccm_tau
                            )

                            context_suffix = f"_{phase_name}"
                            base_filename_part = f"{metric_name}_{round_name}{context_suffix}_ccm_tau{args.ccm_tau}"

                            if ccm_results_dict:
                                all_ccm_summaries = []
                                for (target, library), ccm_run_results in ccm_results_dict.items():
                                    # Plot CCM convergence for each pair and embedding dimension
                                    for E_val, lib_lens, rho_vals in ccm_run_results:
                                        plot_ccm_convergence(
                                            lib_lens=lib_lens,
                                            rho_vals=rho_vals,
                                            target_col=target,
                                            library_col=library,
                                            E=E_val,
                                            tau=args.ccm_tau,
                                            title=f"CCM Convergence {library} -> {target} (E={E_val}) - {metric_display_name} ({phase_display_name})",
                                            output_dir=ccm_plots_dir,
                                            filename=f"{base_filename_part}_E{E_val}_{library}_to_{target}_convergence.png",
                                            metric_name=metric_display_name, round_name=round_name, phase_name=phase_display_name
                                        )
                                    # Summarize results for this pair (e.g., pick best E or last rho)
                                    # The summarize_ccm_results function needs to be designed to handle this structure
                                    # For now, let's assume it can take the dict for a pair and return a summary Series/DataFrame
                                    # This part might need adjustment based on summarize_ccm_results actual signature and output
                                    pair_summary = summarize_ccm_results(ccm_run_results, target, library, args.ccm_threshold) # Pass threshold here
                                    if pair_summary is not None:
                                        all_ccm_summaries.append(pair_summary)
                                
                                if all_ccm_summaries:
                                    # Combine all pair summaries into a single DataFrame for heatmap and saving
                                    # This assumes summarize_ccm_results returns a dict or Series that can be made into a DataFrame
                                    # For a heatmap, we usually need a matrix form. The current causality.py summarize_ccm_results returns a list of dicts.
                                    # Let's adapt to create a DataFrame from this list of dicts first.
                                    summary_df = pd.DataFrame(all_ccm_summaries)
                                    if not summary_df.empty:
                                        ccm_summary_table_filename = f"{base_filename_part}_summary_thresh{args.ccm_threshold}.csv"
                                        export_to_csv(summary_df, ccm_tables_dir, ccm_summary_table_filename)
                                        print(f"    CCM summary table saved to {ccm_summary_table_filename}")

                                        # Plot CCM causality heatmap from the summary_df
                                        # plot_ccm_causality_heatmap expects a matrix, so we might need to pivot summary_df
                                        # Assuming summary_df has columns like 'target', 'library', 'best_rho', 'significant'
                                        # We need to pivot it to create a matrix of rho values for the heatmap.
                                        if 'target' in summary_df.columns and 'library' in summary_df.columns and 'best_rho' in summary_df.columns:
                                            try:
                                                heatmap_data = summary_df.pivot(index='target', columns='library', values='best_rho')
                                                plot_ccm_causality_heatmap(
                                                    heatmap_data,
                                                    title=f"CCM Causality - {metric_display_name} ({phase_display_name}) (Thresh {args.ccm_threshold})",
                                                    output_dir=ccm_plots_dir,
                                                    filename=f"{base_filename_part}_causality_heatmap_thresh{args.ccm_threshold}.png",
                                                    metric_name=metric_display_name, round_name=round_name, phase_name=phase_display_name
                                                )
                                                print(f"    CCM causality heatmap created.")
                                            except Exception as e_pivot:
                                                print(f"    Could not pivot CCM summary for heatmap: {e_pivot}")
                                        else:
                                            print("    CCM summary DataFrame missing required columns for heatmap pivoting.")
                                    else:
                                        print(f"    No CCM summary data to save or plot for {metric_name}, {round_name}, {phase_name}.")       
                                else:
                                    print(f"    No CCM summaries generated for {metric_name}, {round_name}, {phase_name}.")
                            else:
                                print(f"    CCM analysis did not produce results for {metric_name}, {round_name}, {phase_name}.")

                        except Exception as e:
                            print(f"    Error during CCM for {metric_name}, {round_name}, {phase_name}: {e}")
                            import traceback
                            traceback.print_exc()
                else: # Consolidated CCM
                    original_metric_df = phases_or_metric_df
                    metric_display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
                    print(f"  Processing round: {round_name} for CCM on metric: {metric_display_name} (Consolidated)")

                    if not isinstance(original_metric_df, pd.DataFrame) or original_metric_df.empty:
                        print(f"    Skipping CCM: DataFrame is not valid or empty for {metric_name}, {round_name} (Consolidated).")
                        continue
                    
                    if 'experiment_elapsed_time' not in original_metric_df.columns:
                        print(f"    Skipping CCM for {metric_name}, {round_name} (Consolidated): 'experiment_elapsed_time' column missing.")
                        continue
                    
                    try:
                        current_df_processed = original_metric_df.copy()
                        current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                        
                        pivoted_df = current_df_processed.pivot_table(
                            index='experiment_elapsed_time', 
                            columns='tenant', 
                            values='value'
                        )
                        pivoted_df.dropna(axis=1, how='all', inplace=True)
                        pivoted_df.dropna(axis=0, how='any', inplace=True)

                        max_embed_dim = max(args.ccm_embed_dimensions) if args.ccm_embed_dimensions else 1
                        required_len = (max_embed_dim - 1) * args.ccm_tau + 1
                        if pivoted_df.shape[0] < required_len or pivoted_df.shape[1] < 2:
                            print(f"    Skipping CCM: Not enough data after pivoting (shape: {pivoted_df.shape}, E_max: {max_embed_dim}, tau: {args.ccm_tau}, required_len: {required_len}) for {metric_name}, {round_name} (Consolidated).")
                            continue
                        
                        ccm_results_dict = calculate_pairwise_ccm(
                            data=pivoted_df,
                            E_range=args.ccm_embed_dimensions,
                            tau=args.ccm_tau
                        )

                        context_suffix = "_consolidated"
                        base_filename_part = f"{metric_name}_{round_name}{context_suffix}_ccm_tau{args.ccm_tau}"

                        if ccm_results_dict:
                            all_ccm_summaries = []
                            for (target, library), ccm_run_results in ccm_results_dict.items():
                                for E_val, lib_lens, rho_vals in ccm_run_results:
                                    plot_ccm_convergence(
                                        lib_lens=lib_lens,
                                        rho_vals=rho_vals,
                                        target_col=target,
                                        library_col=library,
                                        E=E_val,
                                        tau=args.ccm_tau,
                                        title=f"CCM Convergence {library} -> {target} (E={E_val}) - {metric_display_name} (Consolidated)",
                                        output_dir=ccm_plots_dir,
                                        filename=f"{base_filename_part}_E{E_val}_{library}_to_{target}_convergence.png",
                                        metric_name=metric_display_name, round_name=round_name
                                    )
                                pair_summary = summarize_ccm_results(ccm_run_results, target, library, args.ccm_threshold)
                                if pair_summary is not None:
                                    all_ccm_summaries.append(pair_summary)
                            
                            if all_ccm_summaries:
                                summary_df = pd.DataFrame(all_ccm_summaries)
                                if not summary_df.empty:
                                    ccm_summary_table_filename = f"{base_filename_part}_summary_thresh{args.ccm_threshold}.csv"
                                    export_to_csv(summary_df, ccm_tables_dir, ccm_summary_table_filename)
                                    print(f"    CCM summary table saved to {ccm_summary_table_filename}")

                                    if 'target' in summary_df.columns and 'library' in summary_df.columns and 'best_rho' in summary_df.columns:
                                        try:
                                            heatmap_data = summary_df.pivot(index='target', columns='library', values='best_rho')
                                            plot_ccm_causality_heatmap(
                                                heatmap_data,
                                                title=f"CCM Causality - {metric_display_name} (Consolidated) (Thresh {args.ccm_threshold})",
                                                output_dir=ccm_plots_dir,
                                                filename=f"{base_filename_part}_causality_heatmap_thresh{args.ccm_threshold}.png",
                                                metric_name=metric_display_name, round_name=round_name
                                            )
                                            print(f"    CCM causality heatmap created.")
                                        except Exception as e_pivot:
                                            print(f"    Could not pivot CCM summary for heatmap (Consolidated): {e_pivot}")
                                    else:
                                        print("    CCM summary DataFrame missing required columns for heatmap pivoting (Consolidated).")
                                else:
                                    print(f"    No CCM summary data to save or plot for {metric_name}, {round_name} (Consolidated).")
                            else:
                                print(f"    No CCM summaries generated for {metric_name}, {round_name} (Consolidated).")
                        else:
                            print(f"    CCM analysis did not produce results for {metric_name}, {round_name} (Consolidated).")

                    except Exception as e:
                        print(f"    Error during CCM for {metric_name}, {round_name} (Consolidated): {e}")
                        import traceback
                        traceback.print_exc()
            gc.collect() # After all rounds/phases for a metric

    # Granger Causality
    if args.run_granger:
        print("\nRunning Granger Causality Analysis...")
        for metric_name, rounds_or_phases_data in all_metrics_data.items():
            metric_display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
            print(f"Processing metric: {metric_display_name} for Granger Causality")
            if not isinstance(rounds_or_phases_data, dict):
                print(f"  Skipping metric {metric_name}: Expected dict, got {type(rounds_or_phases_data)}.")
                continue

            for round_name, phases_or_metric_df in rounds_or_phases_data.items():
                if run_per_phase_analysis:
                    if not isinstance(phases_or_metric_df, dict):
                        print(f"  Skipping round {round_name} for {metric_name}: Expected dict, got {type(phases_or_metric_df)}.")
                        continue
                    for phase_name, original_phase_df in phases_or_metric_df.items():
                        phase_display_name = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
                        print(f"  Processing round: {round_name}, phase: {phase_display_name} for Granger on metric: {metric_display_name}")

                        if not isinstance(original_phase_df, pd.DataFrame) or original_phase_df.empty:
                            print(f"    Skipping Granger: DataFrame is not valid or empty for {metric_name}, {round_name}, {phase_name}.")
                            continue
                        
                        if 'experiment_elapsed_time' not in original_phase_df.columns or 'tenant' not in original_phase_df.columns or 'value' not in original_phase_df.columns:
                            print(f"    Skipping Granger for {metric_name}, {round_name}, {phase_name}: missing required columns ('experiment_elapsed_time', 'tenant', 'value').")
                            continue

                        try:
                            current_df_processed = original_phase_df.copy()
                            current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                            
                            pivoted_df = current_df_processed.pivot_table(
                                index='experiment_elapsed_time', 
                                columns='tenant', 
                                values='value'
                            )
                            pivoted_df.dropna(axis=1, how='all', inplace=True) # Drop tenants with no data
                            pivoted_df.interpolate(method='linear', axis=0, limit_direction='both', inplace=True) # Interpolate NaNs
                            pivoted_df.dropna(axis=0, how='any', inplace=True) # Drop rows that still have NaNs (e.g., if all tenants were NaN at a time)
                            pivoted_df.dropna(axis=1, how='any', inplace=True) # Drop tenants that are all NaN after interpolation (should not happen if dropna(how='all') worked)


                            if pivoted_df.shape[0] < args.granger_max_lag + 5 or pivoted_df.shape[1] < 2: # Need enough observations for lags and at least 2 series
                                print(f"    Skipping Granger: Not enough data after pivoting/cleaning (shape: {pivoted_df.shape}, max_lag: {args.granger_max_lag}) for {metric_name}, {round_name}, {phase_name}.")
                                continue
                            
                            granger_results_dict = calculate_pairwise_granger_causality(
                                pivoted_df, 
                                max_lag=args.granger_max_lag, 
                                criterion=args.granger_criterion
                            )

                            context_suffix = f"_{phase_name}"
                            title_base = f"Granger Causality (Lag {args.granger_max_lag}, {args.granger_criterion}) - {metric_display_name} ({phase_display_name})"

                            if granger_results_dict and not granger_results_dict['p_values'].empty:
                                # Save tables
                                p_values_filename = f"{metric_name}_{round_name}{context_suffix}_granger_pvalues_lag{args.granger_max_lag}_{args.granger_criterion}.csv"
                                export_to_csv(granger_results_dict['p_values'], granger_tables_dir, p_values_filename)
                                print(f"    Granger p-values saved to {p_values_filename}")

                                lags_filename = f"{metric_name}_{round_name}{context_suffix}_granger_optlags_lag{args.granger_max_lag}_{args.granger_criterion}.csv"
                                export_to_csv(granger_results_dict['optimal_lags'], granger_tables_dir, lags_filename)
                                print(f"    Granger optimal lags saved to {lags_filename}")

                                fstats_filename = f"{metric_name}_{round_name}{context_suffix}_granger_fstats_lag{args.granger_max_lag}_{args.granger_criterion}.csv"
                                export_to_csv(granger_results_dict['f_statistics'], granger_tables_dir, fstats_filename)
                                print(f"    Granger F-statistics saved to {fstats_filename}")

                                # Plot Granger heatmap
                                plot_granger_causality_heatmap(
                                    granger_results_dict,
                                    title=title_base,
                                    output_dir=granger_plots_dir,
                                    filename=f"{metric_name}_{round_name}{context_suffix}_granger_heatmap_lag{args.granger_max_lag}_{args.granger_criterion}.png",
                                    metric_name=metric_display_name, round_name=round_name, phase_name=phase_display_name,
                                    alpha=args.granger_alpha
                                )
                                print(f"    Granger causality heatmap created.")

                                # Plot Granger network
                                plot_granger_causality_network(
                                    granger_results_dict,
                                    title=f"{title_base} - Network (alpha {args.granger_alpha})",
                                    output_dir=granger_plots_dir,
                                    filename=f"{metric_name}_{round_name}{context_suffix}_granger_network_lag{args.granger_max_lag}_{args.granger_criterion}_alpha{args.granger_alpha}.png",
                                    metric_name=metric_display_name, round_name=round_name, phase_name=phase_display_name,
                                    alpha=args.granger_alpha,
                                    f_stat_threshold=args.granger_fstat_threshold 
                                )
                                print(f"    Granger causality network plot created.")
                            else:
                                print(f"    Granger causality analysis did not produce results for {metric_name}, {round_name}, {phase_name}.")

                        except Exception as e:
                            print(f"    Error during Granger Causality for {metric_name}, {round_name}, {phase_name}: {e}")
                            import traceback
                            traceback.print_exc()
                else: # Consolidated Granger
                    original_metric_df = phases_or_metric_df
                    print(f"  Processing round: {round_name} for Granger on metric: {metric_display_name} (Consolidated)")

                    if not isinstance(original_metric_df, pd.DataFrame) or original_metric_df.empty:
                        print(f"    Skipping Granger: DataFrame is not valid or empty for {metric_name}, {round_name} (Consolidated).")
                        continue
                    
                    if 'experiment_elapsed_time' not in original_metric_df.columns or 'tenant' not in original_metric_df.columns or 'value' not in original_metric_df.columns:
                        print(f"    Skipping Granger for {metric_name}, {round_name} (Consolidated): missing required columns ('experiment_elapsed_time', 'tenant', 'value').")
                        continue
                    
                    try:
                        current_df_processed = original_metric_df.copy()
                        current_df_processed['value'] = pd.to_numeric(current_df_processed['value'], errors='coerce')
                        
                        pivoted_df = current_df_processed.pivot_table(
                            index='experiment_elapsed_time', 
                            columns='tenant', 
                            values='value'
                        )
                        pivoted_df.dropna(axis=1, how='all', inplace=True)
                        pivoted_df.interpolate(method='linear', axis=0, limit_direction='both', inplace=True)
                        pivoted_df.dropna(axis=0, how='any', inplace=True)
                        pivoted_df.dropna(axis=1, how='any', inplace=True)


                        if pivoted_df.shape[0] < args.granger_max_lag + 5 or pivoted_df.shape[1] < 2:
                            print(f"    Skipping Granger: Not enough data after pivoting/cleaning (shape: {pivoted_df.shape}, max_lag: {args.granger_max_lag}) for {metric_name}, {round_name} (Consolidated).")
                            continue
                            
                        granger_results_dict = calculate_pairwise_granger_causality(
                            pivoted_df, 
                            max_lag=args.granger_max_lag, 
                            criterion=args.granger_criterion
                        )

                        context_suffix = "_consolidated"
                        title_base = f"Granger Causality (Lag {args.granger_max_lag}, {args.granger_criterion}) - {metric_display_name} (Consolidated)"

                        if granger_results_dict and not granger_results_dict['p_values'].empty:
                            p_values_filename = f"{metric_name}_{round_name}{context_suffix}_granger_pvalues_lag{args.granger_max_lag}_{args.granger_criterion}.csv"
                            export_to_csv(granger_results_dict['p_values'], granger_tables_dir, p_values_filename)
                            print(f"    Granger p-values saved to {p_values_filename}")

                            lags_filename = f"{metric_name}_{round_name}{context_suffix}_granger_optlags_lag{args.granger_max_lag}_{args.granger_criterion}.csv"
                            export_to_csv(granger_results_dict['optimal_lags'], granger_tables_dir, lags_filename)
                            print(f"    Granger optimal lags saved to {lags_filename}")

                            fstats_filename = f"{metric_name}_{round_name}{context_suffix}_granger_fstats_lag{args.granger_max_lag}_{args.granger_criterion}.csv"
                            export_to_csv(granger_results_dict['f_statistics'], granger_tables_dir, fstats_filename)
                            print(f"    Granger F-statistics saved to {fstats_filename}")
                            
                            plot_granger_causality_heatmap(
                                granger_results_dict,
                                title=title_base,
                                output_dir=granger_plots_dir,
                                filename=f"{metric_name}_{round_name}{context_suffix}_granger_heatmap_lag{args.granger_max_lag}_{args.granger_criterion}.png",
                                metric_name=metric_display_name, round_name=round_name,
                                alpha=args.granger_alpha
                            )
                            print(f"    Granger causality heatmap created.")

                            plot_granger_causality_network(
                                granger_results_dict,
                                title=f"{title_base} - Network (alpha {args.granger_alpha})",
                                output_dir=granger_plots_dir,
                                filename=f"{metric_name}_{round_name}{context_suffix}_granger_network_lag{args.granger_max_lag}_{args.granger_criterion}_alpha{args.granger_alpha}.png",
                                metric_name=metric_display_name, round_name=round_name,
                                alpha=args.granger_alpha,
                                f_stat_threshold=args.granger_fstat_threshold
                            )
                            print(f"    Granger causality network plot created.")
                        else:
                            print(f"    Granger causality analysis did not produce results for {metric_name}, {round_name} (Consolidated).")
                            
                    except Exception as e:
                        print(f"    Error during Granger Causality for {metric_name}, {round_name} (Consolidated): {e}")
                        import traceback
                        traceback.print_exc()
            gc.collect()

    close_all_figures() # Close any open matplotlib figures
    print("\nAnalysis pipeline finished.")

if __name__ == '__main__':
    main()
