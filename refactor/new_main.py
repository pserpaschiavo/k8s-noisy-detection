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
    plot_dtw_distance_heatmap
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

    # Argument for analysis scope
    parser.add_argument('--consolidated-analysis', action='store_true',
                        help='Run analysis consolidated across all phases (old behavior). If not set, analysis will be per-phase.')

    return parser.parse_args()


def setup_output_directories(output_dir):
    """Configures output directories."""
    plots_dir = os.path.join(output_dir, 'plots')
    tables_dir = os.path.join(output_dir, 'tables')
    descriptive_stats_plots_dir = os.path.join(plots_dir, "descriptive_stats")
    descriptive_stats_tables_dir = os.path.join(tables_dir, "descriptive_stats")
    correlation_plots_dir = os.path.join(plots_dir, "correlation")
    correlation_tables_dir = os.path.join(tables_dir, "correlation")
    covariance_plots_dir = os.path.join(plots_dir, "covariance")
    covariance_tables_dir = os.path.join(tables_dir, "covariance")
    multivariate_dir = os.path.join(tables_dir, "multivariate")  # Directory within tables for PCA/ICA
    pca_output_dir = os.path.join(multivariate_dir, "pca")
    ica_output_dir = os.path.join(multivariate_dir, "ica")
    comparison_output_dir = os.path.join(multivariate_dir, "comparison")  # Directory for PCA vs ICA comparison
    
    # Directories for causal analysis
    causality_dir = os.path.join(tables_dir, "causality")
    sem_output_dir = os.path.join(causality_dir, "sem")
    
    # Create plots directories for multivariate analysis (PCA/ICA)
    multivariate_plots_dir = os.path.join(plots_dir, "multivariate")
    pca_plots_dir = os.path.join(multivariate_plots_dir, "pca")
    ica_plots_dir = os.path.join(multivariate_plots_dir, "ica")
    comparison_plots_dir = os.path.join(multivariate_plots_dir, "comparison")
    
    # Create plots directory for causality analysis
    causality_plots_dir = os.path.join(plots_dir, "causality")
    sem_plots_dir = os.path.join(causality_plots_dir, "sem")
    
    # New directories for additional causal analysis techniques
    te_output_dir = os.path.join(causality_dir, "transfer_entropy")
    te_plots_dir = os.path.join(causality_plots_dir, "transfer_entropy")
    
    ccm_output_dir = os.path.join(causality_dir, "ccm")
    ccm_plots_dir = os.path.join(causality_plots_dir, "ccm")
    
    granger_output_dir = os.path.join(causality_dir, "granger")
    granger_plots_dir = os.path.join(causality_plots_dir, "granger")
    
    # Create the main similarity directory structure
    similarity_plots_dir = os.path.join(plots_dir, "similarity")
    similarity_tables_dir = os.path.join(tables_dir, "similarity")
    os.makedirs(similarity_plots_dir, exist_ok=True)
    os.makedirs(similarity_tables_dir, exist_ok=True)
    
    # Directory for Distance Correlation analysis
    dcor_plots_dir = os.path.join(similarity_plots_dir, "distance_correlation")
    dcor_tables_dir = os.path.join(similarity_tables_dir, "distance_correlation")
    os.makedirs(dcor_plots_dir, exist_ok=True)
    os.makedirs(dcor_tables_dir, exist_ok=True)
    
    # Directory for Cosine Similarity analysis
    cosine_plots_dir = os.path.join(similarity_plots_dir, "cosine_similarity")
    cosine_tables_dir = os.path.join(similarity_tables_dir, "cosine_similarity")
    os.makedirs(cosine_plots_dir, exist_ok=True)
    os.makedirs(cosine_tables_dir, exist_ok=True)
    
    # Directory for DTW analysis
    dtw_plots_dir = os.path.join(similarity_plots_dir, "dtw")
    dtw_tables_dir = os.path.join(similarity_tables_dir, "dtw")
    os.makedirs(dtw_plots_dir, exist_ok=True)
    os.makedirs(dtw_tables_dir, exist_ok=True)

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(descriptive_stats_plots_dir, exist_ok=True)
    os.makedirs(descriptive_stats_tables_dir, exist_ok=True)
    os.makedirs(correlation_plots_dir, exist_ok=True)
    os.makedirs(correlation_tables_dir, exist_ok=True)
    os.makedirs(covariance_plots_dir, exist_ok=True)
    os.makedirs(covariance_tables_dir, exist_ok=True)
    os.makedirs(multivariate_dir, exist_ok=True)  # Create multivariate directory
    os.makedirs(pca_output_dir, exist_ok=True)  # Create PCA output directory
    os.makedirs(ica_output_dir, exist_ok=True)  # Create ICA output directory
    os.makedirs(comparison_output_dir, exist_ok=True)  # Create comparison directory
    os.makedirs(causality_dir, exist_ok=True)  # Create causality directory
    os.makedirs(sem_output_dir, exist_ok=True)  # Create SEM output directory
    os.makedirs(causality_plots_dir, exist_ok=True)  # Create causality plots directory
    os.makedirs(sem_plots_dir, exist_ok=True)  # Create SEM plots directory
    
    # Create directories for new causal analysis techniques
    os.makedirs(te_output_dir, exist_ok=True)  # Create Transfer Entropy output directory
    os.makedirs(te_plots_dir, exist_ok=True)  # Create Transfer Entropy plots directory
    os.makedirs(ccm_output_dir, exist_ok=True)  # Create CCM output directory
    os.makedirs(ccm_plots_dir, exist_ok=True)  # Create CCM plots directory
    os.makedirs(granger_output_dir, exist_ok=True)  # Create Granger Causality output directory
    os.makedirs(granger_plots_dir, exist_ok=True)  # Create Granger Causality plots directory

    return plots_dir, tables_dir, pca_output_dir, ica_output_dir, comparison_output_dir, sem_output_dir, sem_plots_dir, te_output_dir, te_plots_dir, ccm_output_dir, ccm_plots_dir, granger_output_dir, granger_plots_dir, dcor_plots_dir, dcor_tables_dir, cosine_plots_dir, cosine_tables_dir, dtw_plots_dir, dtw_tables_dir


def main():
    """Main function to run the refactored analysis pipeline."""
    args = parse_arguments()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    plots_dir, tables_dir, pca_output_dir, ica_output_dir, comparison_output_dir, sem_output_dir, sem_plots_dir, te_output_dir, te_plots_dir, ccm_output_dir, ccm_plots_dir, granger_output_dir, granger_plots_dir, dcor_plots_dir, dcor_tables_dir, cosine_plots_dir, cosine_tables_dir, dtw_plots_dir, dtw_tables_dir = setup_output_directories(args.output_dir)

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

if __name__ == '__main__':
    main()
