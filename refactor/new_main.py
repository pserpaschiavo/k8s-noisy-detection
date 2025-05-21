import os
import sys

# Add project root to sys.path to allow direct execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import pandas as pd
import matplotlib.pyplot as plt

"""
New main script for the refactored noisy neighbor experiment analysis pipeline.
"""

# Refactored modules
from refactor.data_handling.loader import load_experiment_data
from refactor.data_handling.save_results import export_to_csv, save_figure
from refactor.analysis_modules.correlation_covariance import (
    calculate_inter_tenant_correlation_per_metric, # Changed from calculate_correlation_matrix
    calculate_covariance_matrix, # Not used in this basic setup yet
    calculate_inter_tenant_covariance_per_metric # Not used in this basic setup yet
)
# Corrected import for plot_correlation_heatmap
from refactor.visualization.new_plots import plot_correlation_heatmap, plot_covariance_heatmap # Added plot_covariance_heatmap

# Existing pipeline modules (will be gradually replaced or integrated)
from refactor.data_handling.new_time_normalization import add_experiment_elapsed_time, add_phase_markers # For potential future use
from refactor.new_config import (
    DEFAULT_DATA_DIR, DEFAULT_METRICS, METRIC_DISPLAY_NAMES,
    VISUALIZATION_CONFIG, TENANT_COLORS, DEFAULT_NOISY_TENANT # TENANT_COLORS, VISUALIZATION_CONFIG might be needed by plots
)
# Add other necessary imports from pipeline.config or other modules as needed


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Refactored pipeline for noisy neighbor experiment analysis.')
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
    parser.add_argument('--noisy-tenant', type=str, default=DEFAULT_NOISY_TENANT,
                        help='Tenant considered to be the noisy one.')
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

    # Argument for analysis scope
    parser.add_argument('--consolidated-analysis', action='store_true',
                        help='Run analysis consolidated across all phases (old behavior). If not set, analysis will be per-phase.')

    return parser.parse_args()


def setup_output_directories(output_dir):
    """Configures output directories."""
    plots_dir = os.path.join(output_dir, 'plots')
    tables_dir = os.path.join(output_dir, 'tables')
    # Add more specific directories as needed (e.g., advanced, anomalies)
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    return plots_dir, tables_dir

def main():
    """Main function to run the refactored analysis pipeline."""
    args = parse_arguments()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    plots_dir, tables_dir = setup_output_directories(args.output_dir)

    # Resolve data directory path
    experiment_data_dir_input = args.data_dir
    experiment_data_dir = ""

    if os.path.isabs(experiment_data_dir_input):
        experiment_data_dir = experiment_data_dir_input
    else:
        cwd = os.getcwd()
        path_from_cwd = os.path.join(cwd, experiment_data_dir_input)
        # Attempt to check relative to workspace root if not found directly under cwd
        # This logic assumes new_main.py is in refactor/ and DEFAULT_DATA_DIR is relative to project root
        path_from_project_root_perspective = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), experiment_data_dir_input)


        if os.path.isdir(path_from_cwd):
            experiment_data_dir = path_from_cwd
        elif os.path.isdir(path_from_project_root_perspective) and not os.path.isdir(path_from_cwd):
             experiment_data_dir = os.path.normpath(path_from_project_root_perspective)
        else: # Default to CWD relative if project root relative also fails or is same
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
    # Determine if analysis is per-phase or consolidated based on the new argument
    run_per_phase_analysis = not args.consolidated_analysis
    if run_per_phase_analysis:
        print("Data loading mode: Per-Phase")
    else:
        print("Data loading mode: Consolidated")

    all_metrics_data = load_experiment_data(
        experiment_dir=experiment_data_dir, # Corrected: data_dir to experiment_dir
        metrics=args.metrics,               # Corrected: specific_metrics to metrics
        tenants=args.tenants,               # Corrected: specific_tenants to tenants
        rounds=args.rounds,                  # Corrected: specific_rounds to rounds
        group_by_phase=run_per_phase_analysis # Pass the flag to the loader
    )

    if not all_metrics_data:
        print("No data loaded. Exiting.")
        return

    print(f"Loaded metrics: {list(all_metrics_data.keys())}")

    # --- Example: Correlation Analysis ---
    if args.run_correlation:
        print("\nRunning Correlation Analysis...")
        correlation_plots_dir = os.path.join(plots_dir, 'correlation') # Specific subdir for correlation plots
        os.makedirs(correlation_plots_dir, exist_ok=True)
        correlation_tables_dir = os.path.join(tables_dir, 'correlation') # Specific subdir for correlation tables
        os.makedirs(correlation_tables_dir, exist_ok=True)

        methods_to_run = []
        if args.run_all_correlation_methods:
            methods_to_run = ['pearson', 'spearman', 'kendall']
            print(f"  Running for ALL methods: {methods_to_run} (due to --run-all-correlation-methods)")
        elif args.correlation_methods:
            methods_to_run = args.correlation_methods
            print(f"  Running for specified method(s): {methods_to_run}")
        else: # Should not happen if default is set for correlation_methods, but as a fallback
            methods_to_run = ['pearson']
            print(f"  Running for default method: {methods_to_run}")

        for current_method in methods_to_run:
            print(f"\n  Processing with method: {current_method.upper()}")
            for metric_name, rounds_or_phases_data in all_metrics_data.items(): 
                print(f"  Processing metric: {metric_name}")
                if not isinstance(rounds_or_phases_data, dict):
                    print(f"    Skipping metric {metric_name}: Expected a dictionary of round/phase data, got {type(rounds_or_phases_data)}.")
                    continue

                for round_name, phases_or_metric_df in rounds_or_phases_data.items():
                    if run_per_phase_analysis: # New: Loop through phases if group_by_phase was True
                        if not isinstance(phases_or_metric_df, dict):
                            print(f"    Skipping round {round_name} for metric {metric_name}: Expected a dictionary of phase data, got {type(phases_or_metric_df)}.")
                            continue
                        for phase_name, metric_df_original in phases_or_metric_df.items():
                            print(f"    Processing round: {round_name}, phase: {phase_name} for metric: {metric_name}")
                            # --- Start of per-phase processing logic (extracted and reused) ---
                            if not isinstance(metric_df_original, pd.DataFrame):
                                print(f"      Skipping round {round_name}, phase {phase_name} for metric {metric_name}: Expected a DataFrame, got {type(metric_df_original)}.")
                                continue
                            if metric_df_original.empty or 'value' not in metric_df_original.columns or 'tenant' not in metric_df_original.columns:
                                print(f"      Skipping correlation for metric {metric_name}, round {round_name}, phase {phase_name} due to missing data or columns.")
                                continue
                            
                            metric_df = metric_df_original.copy()
                            print(f"      Calculating correlation ({current_method}) for metric: {metric_name}, round: {round_name}, phase: {phase_name}")
                            try:
                                if 'timestamp' not in metric_df.columns:
                                    print(f"      Skipping {metric_name}, round {round_name}, phase {phase_name}: 'timestamp' column not found.")
                                    continue
                                if not pd.api.types.is_datetime64_any_dtype(metric_df['timestamp']):
                                    metric_df['timestamp'] = pd.to_datetime(metric_df['timestamp'], format='%Y%m%d_%H%M%S')
                                if 'experiment_elapsed_seconds' not in metric_df.columns:
                                    metric_df_sorted = metric_df.sort_values(by=['timestamp'])
                                    min_timestamp_overall = metric_df_sorted['timestamp'].min()
                                    metric_df['experiment_elapsed_seconds'] = (metric_df['timestamp'] - min_timestamp_overall).dt.total_seconds()

                                correlation_matrix_df = calculate_inter_tenant_correlation_per_metric(
                                    metric_df, method=current_method, time_col='timestamp'
                                )
                                if correlation_matrix_df is not None and not correlation_matrix_df.empty:
                                    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
                                    plot_filename = f"{metric_name}_{round_name}_{phase_name}_{current_method}_correlation_heatmap.png"
                                    plot_title = f"Inter-Tenant Correlation ({current_method.capitalize()}): {display_metric_name} (Round: {round_name}, Phase: {phase_name})"
                                    fig = plot_correlation_heatmap(
                                        correlation_matrix_df, title=plot_title, output_dir=correlation_plots_dir, filename=plot_filename
                                    )
                                    if fig: 
                                        print(f"      Correlation heatmap for {metric_name}, round {round_name}, phase {phase_name} ({current_method}) generated and saved.")
                                        plt.close(fig) 
                                    csv_filename_name_only = f"{metric_name}_{round_name}_{phase_name}_{current_method}_correlation_matrix.csv"
                                    full_csv_path = os.path.join(correlation_tables_dir, csv_filename_name_only)
                                    export_to_csv(correlation_matrix_df, full_csv_path)
                                    print(f"      Correlation matrix for {metric_name}, round {round_name}, phase {phase_name} ({current_method}) saved to {full_csv_path}")
                                else:
                                    print(f"      Skipping plot/save for {metric_name}, round {round_name}, phase {phase_name} ({current_method}): Correlation matrix is empty or could not be calculated.")
                            except Exception as e:
                                print(f"      Error during correlation analysis for metric {metric_name}, round {round_name}, phase {phase_name} ({current_method}): {e}")
                                import traceback
                                traceback.print_exc()
                            # --- End of per-phase processing logic ---
                    else: # Original consolidated processing
                        metric_df_original = phases_or_metric_df # In consolidated mode, this is the DataFrame
                        print(f"    Processing round: {round_name} for metric: {metric_name} (Consolidated)")
                        # --- Start of consolidated processing logic (similar to above but without phase) ---
                        if not isinstance(metric_df_original, pd.DataFrame):
                            print(f"      Skipping round {round_name} for metric {metric_name}: Expected a DataFrame, got {type(metric_df_original)}.")
                            continue
                        if metric_df_original.empty or 'value' not in metric_df_original.columns or 'tenant' not in metric_df_original.columns:
                            print(f"      Skipping correlation for metric {metric_name}, round {round_name} due to missing data or columns.")
                            continue
                        
                        metric_df = metric_df_original.copy()
                        print(f"      Calculating correlation ({current_method}) for metric: {metric_name}, round: {round_name}")
                        try:
                            if 'timestamp' not in metric_df.columns:
                                print(f"      Skipping {metric_name}, round {round_name}: 'timestamp' column not found.")
                                continue
                            if not pd.api.types.is_datetime64_any_dtype(metric_df['timestamp']):
                                metric_df['timestamp'] = pd.to_datetime(metric_df['timestamp'], format='%Y%m%d_%H%M%S')
                            if 'experiment_elapsed_seconds' not in metric_df.columns:
                                metric_df_sorted = metric_df.sort_values(by=['timestamp'])
                                min_timestamp_overall = metric_df_sorted['timestamp'].min()
                                metric_df['experiment_elapsed_seconds'] = (metric_df['timestamp'] - min_timestamp_overall).dt.total_seconds()

                            correlation_matrix_df = calculate_inter_tenant_correlation_per_metric(
                                metric_df, method=current_method, time_col='timestamp'
                            )
                            if correlation_matrix_df is not None and not correlation_matrix_df.empty:
                                display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
                                plot_filename = f"{metric_name}_{round_name}_{current_method}_correlation_heatmap.png"
                                plot_title = f"Inter-Tenant Correlation ({current_method.capitalize()}): {display_metric_name} (Round: {round_name})"
                                fig = plot_correlation_heatmap(
                                    correlation_matrix_df, title=plot_title, output_dir=correlation_plots_dir, filename=plot_filename
                                )
                                if fig: 
                                    print(f"      Correlation heatmap for {metric_name}, round {round_name} ({current_method}) generated and saved.")
                                    plt.close(fig) 
                                csv_filename_name_only = f"{metric_name}_{round_name}_{current_method}_correlation_matrix.csv"
                                full_csv_path = os.path.join(correlation_tables_dir, csv_filename_name_only)
                                export_to_csv(correlation_matrix_df, full_csv_path)
                                print(f"      Correlation matrix for {metric_name}, round {round_name} ({current_method}) saved to {full_csv_path}")
                            else:
                                print(f"      Skipping plot/save for {metric_name}, round {round_name} ({current_method}): Correlation matrix is empty or could not be calculated.")
                        except Exception as e:
                            print(f"      Error during correlation analysis for metric {metric_name}, round {round_name} ({current_method}): {e}")
                            import traceback
                            traceback.print_exc()
                        # --- End of consolidated processing logic ---

    # --- Covariance Analysis ---
    if args.run_covariance:
        print("\nRunning Covariance Analysis...")
        covariance_plots_dir = os.path.join(plots_dir, 'covariance')
        os.makedirs(covariance_plots_dir, exist_ok=True)
        covariance_tables_dir = os.path.join(tables_dir, 'covariance')
        os.makedirs(covariance_tables_dir, exist_ok=True)

        for metric_name, rounds_or_phases_data in all_metrics_data.items(): # Adjusted variable name
            print(f"  Processing metric: {metric_name}")
            if not isinstance(rounds_or_phases_data, dict):
                print(f"    Skipping metric {metric_name}: Expected a dictionary of round/phase data, got {type(rounds_or_phases_data)}.")
                continue

            for round_name, phases_or_metric_df in rounds_or_phases_data.items(): # Adjusted variable name
                if run_per_phase_analysis: # New: Loop through phases
                    if not isinstance(phases_or_metric_df, dict):
                        print(f"    Skipping round {round_name} for metric {metric_name}: Expected a dictionary of phase data, got {type(phases_or_metric_df)}.")
                        continue
                    for phase_name, metric_df_original in phases_or_metric_df.items():
                        print(f"    Processing round: {round_name}, phase: {phase_name} for metric: {metric_name}")
                        # --- Start of per-phase covariance processing ---
                        if not isinstance(metric_df_original, pd.DataFrame):
                            print(f"      Skipping round {round_name}, phase {phase_name} for metric {metric_name}: Expected a DataFrame, got {type(metric_df_original)}.")
                            continue
                        if metric_df_original.empty or 'value' not in metric_df_original.columns or 'tenant' not in metric_df_original.columns:
                            print(f"      Skipping covariance for metric {metric_name}, round {round_name}, phase {phase_name} due to missing data or columns.")
                            continue
                        
                        metric_df = metric_df_original.copy()
                        print(f"      Calculating covariance for metric: {metric_name}, round: {round_name}, phase: {phase_name}")
                        try:
                            if 'timestamp' not in metric_df.columns:
                                print(f"      Skipping {metric_name}, round {round_name}, phase {phase_name}: 'timestamp' column not found.")
                                continue
                            if not pd.api.types.is_datetime64_any_dtype(metric_df['timestamp']):
                                metric_df['timestamp'] = pd.to_datetime(metric_df['timestamp'], format='%Y%m%d_%H%M%S')
                            if 'experiment_elapsed_seconds' not in metric_df.columns:
                                metric_df_sorted = metric_df.sort_values(by=['timestamp'])
                                min_timestamp_overall = metric_df_sorted['timestamp'].min()
                                metric_df['experiment_elapsed_seconds'] = (metric_df['timestamp'] - min_timestamp_overall).dt.total_seconds()

                            covariance_matrix_df = calculate_inter_tenant_covariance_per_metric(
                                metric_df, time_col='timestamp'
                            )
                            if covariance_matrix_df is not None and not covariance_matrix_df.empty:
                                display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
                                plot_filename = f"{metric_name}_{round_name}_{phase_name}_covariance_heatmap.png"
                                plot_title = f"Inter-Tenant Covariance: {display_metric_name} (Round: {round_name}, Phase: {phase_name})"
                                fig = plot_covariance_heatmap(
                                    covariance_matrix_df, title=plot_title, output_dir=covariance_plots_dir, filename=plot_filename
                                )
                                if fig:
                                    print(f"      Covariance heatmap for {metric_name}, round {round_name}, phase {phase_name} generated and saved.")
                                    plt.close(fig)
                                csv_filename_name_only = f"{metric_name}_{round_name}_{phase_name}_covariance_matrix.csv"
                                full_csv_path = os.path.join(covariance_tables_dir, csv_filename_name_only)
                                export_to_csv(covariance_matrix_df, full_csv_path)
                                print(f"      Covariance matrix for {metric_name}, round {round_name}, phase {phase_name} saved to {full_csv_path}")
                            else:
                                print(f"      Skipping save for {metric_name}, round {round_name}, phase {phase_name}: Covariance matrix is empty or could not be calculated.")
                        except Exception as e:
                            print(f"      Error during covariance analysis for metric {metric_name}, round {round_name}, phase {phase_name}: {e}")
                            import traceback
                            traceback.print_exc()
                        # --- End of per-phase covariance processing ---
                else: # Original consolidated processing
                    metric_df_original = phases_or_metric_df # In consolidated mode, this is the DataFrame
                    print(f"    Processing round: {round_name} for metric: {metric_name} (Consolidated)")
                    # --- Start of consolidated covariance processing ---
                    if not isinstance(metric_df_original, pd.DataFrame):
                        print(f"      Skipping round {round_name} for metric {metric_name}: Expected a DataFrame, got {type(metric_df_original)}.")
                        continue
                    if metric_df_original.empty or 'value' not in metric_df_original.columns or 'tenant' not in metric_df_original.columns:
                        print(f"      Skipping covariance for metric {metric_name}, round {round_name} due to missing data or columns.")
                        continue

                    metric_df = metric_df_original.copy()
                    print(f"      Calculating covariance for metric: {metric_name}, round: {round_name}")
                    try:
                        if 'timestamp' not in metric_df.columns:
                            print(f"      Skipping {metric_name}, round {round_name}: 'timestamp' column not found.")
                            continue
                        if not pd.api.types.is_datetime64_any_dtype(metric_df['timestamp']):
                            metric_df['timestamp'] = pd.to_datetime(metric_df['timestamp'], format='%Y%m%d_%H%M%S')
                        if 'experiment_elapsed_seconds' not in metric_df.columns:
                            metric_df_sorted = metric_df.sort_values(by=['timestamp'])
                            min_timestamp_overall = metric_df_sorted['timestamp'].min()
                            metric_df['experiment_elapsed_seconds'] = (metric_df['timestamp'] - min_timestamp_overall).dt.total_seconds()

                        covariance_matrix_df = calculate_inter_tenant_covariance_per_metric(
                            metric_df, time_col='timestamp'
                        )
                        if covariance_matrix_df is not None and not covariance_matrix_df.empty:
                            display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
                            plot_filename = f"{metric_name}_{round_name}_covariance_heatmap.png"
                            plot_title = f"Inter-Tenant Covariance: {display_metric_name} (Round: {round_name})"
                            fig = plot_covariance_heatmap(
                                covariance_matrix_df, title=plot_title, output_dir=covariance_plots_dir, filename=plot_filename
                            )
                            if fig:
                                print(f"      Covariance heatmap for {metric_name}, round {round_name} generated and saved.")
                                plt.close(fig)
                            csv_filename_name_only = f"{metric_name}_{round_name}_covariance_matrix.csv"
                            full_csv_path = os.path.join(covariance_tables_dir, csv_filename_name_only)
                            export_to_csv(covariance_matrix_df, full_csv_path)
                            print(f"      Covariance matrix for {metric_name}, round {round_name} saved to {full_csv_path}")
                        else:
                            print(f"      Skipping save for {metric_name}, round {round_name}: Covariance matrix is empty or could not be calculated.")
                    except Exception as e:
                        print(f"      Error during covariance analysis for metric {metric_name}, round {round_name}: {e}")
                        import traceback
                        traceback.print_exc()
                    # --- End of consolidated covariance processing ---

    print("\nRefactored pipeline processing finished.")

if __name__ == '__main__':
    main()
