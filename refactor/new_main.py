import os
import sys

# Add project root to sys.path to allow direct execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import itertools  # Added import

"""
New main script for the refactored noisy neighbor experiment analysis pipeline.
"""

# Refactored modules
from refactor.data_handling.loader import load_experiment_data
from refactor.data_handling.save_results import export_to_csv, save_figure
from refactor.analysis_modules.correlation_covariance import (
    calculate_inter_tenant_correlation_per_metric,  # Changed from calculate_correlation_matrix
    calculate_covariance_matrix,  # Not used in this basic setup yet
    calculate_inter_tenant_covariance_per_metric  # Not used in this basic setup yet
)
from refactor.analysis_modules.descritive_statistics import calculate_descriptive_statistics  # Added import
# Import new placeholder modules
from refactor.analysis_modules.anomaly_detection import run_anomaly_detection_analysis
from refactor.analysis_modules.tenant_analysis import run_tenant_specific_analysis
from refactor.analysis_modules.advanced_analysis import run_advanced_pipeline_analysis
# Corrected import for plot_correlation_heatmap
from refactor.visualization.new_plots import (
    plot_correlation_heatmap, plot_covariance_heatmap, plot_scatter_comparison,
    plot_descriptive_stats_lineplot, plot_descriptive_stats_boxplot, plot_descriptive_stats_catplot_mean
)
from refactor.data_handling.new_time_normalization import add_experiment_elapsed_time  # Added import

# Existing pipeline modules (will be gradually replaced or integrated)
from refactor.new_config import (
    DEFAULT_DATA_DIR, DEFAULT_METRICS, METRIC_DISPLAY_NAMES,
    VISUALIZATION_CONFIG, TENANT_COLORS, DEFAULT_NOISY_TENANT  # TENANT_COLORS, VISUALIZATION_CONFIG might be needed by plots
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

    # Argument for Descriptive Statistics Analysis
    parser.add_argument('--run-descriptive-stats', action='store_true', help='Run descriptive statistics analysis.')  # Added argument

    # Arguments for other analysis types from pipeline/analysis/
    parser.add_argument('--run-anomaly-detection', action='store_true', help='Run anomaly detection analysis.')
    parser.add_argument('--run-tenant-analysis', action='store_true', help='Run tenant-specific analysis.')
    parser.add_argument('--run-advanced-analysis', action='store_true', help='Run advanced analysis modules.')

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
        if run_per_phase_analysis:
            for round_name, phases_data in rounds_or_phases_data.items():
                for phase_name, metric_df in phases_data.items():
                    if isinstance(metric_df, pd.DataFrame) and 'datetime' in metric_df.columns:
                        # Group by round for experiment_elapsed_time context, even in per-phase
                        all_metrics_data[metric_name][round_name][phase_name] = add_experiment_elapsed_time(
                            metric_df, group_by=['round']
                        )
                    else:
                        print(f"    Skipping time normalization for {metric_name}, {round_name}, {phase_name}: DataFrame not found or 'datetime' column missing.")
        else:  # Consolidated analysis
            for round_name, metric_df in rounds_or_phases_data.items():
                if isinstance(metric_df, pd.DataFrame) and 'datetime' in metric_df.columns:
                    all_metrics_data[metric_name][round_name] = add_experiment_elapsed_time(
                        metric_df, group_by=['round']
                    )
                else:
                    print(f"    Skipping time normalization for {metric_name}, {round_name} (Consolidated): DataFrame not found or 'datetime' column missing.")

    # --- Example: Correlation Analysis ---
    if args.run_correlation:
        print("\nRunning Correlation Analysis...")
        correlation_plots_dir = os.path.join(plots_dir, 'correlation')
        os.makedirs(correlation_plots_dir, exist_ok=True)
        correlation_tables_dir = os.path.join(tables_dir, 'correlation')
        os.makedirs(correlation_tables_dir, exist_ok=True)
        correlation_scatter_plots_dir = os.path.join(correlation_plots_dir, 'scatter_comparisons')
        os.makedirs(correlation_scatter_plots_dir, exist_ok=True)

        methods_to_run = []
        if args.run_all_correlation_methods:
            methods_to_run = ['pearson', 'spearman', 'kendall']
            print(f"  Running for ALL methods: {methods_to_run} (due to --run-all-correlation-methods)")
        elif args.correlation_methods:
            methods_to_run = args.correlation_methods
            print(f"  Running for specified method(s): {methods_to_run}")
        else:
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
                    method_correlation_dfs_for_current_scope = {}

                    if run_per_phase_analysis:
                        if not isinstance(phases_or_metric_df, dict):
                            print(f"    Skipping round {round_name} for metric {metric_name}: Expected a dictionary of phase data, got {type(phases_or_metric_df)}.")
                            continue
                        for phase_name, metric_df_original in phases_or_metric_df.items():
                            print(f"    Processing round: {round_name}, phase: {phase_name} for metric: {metric_name}")
                            method_correlation_dfs_for_current_scope_phase = {}
                            for current_method_inner_loop in methods_to_run:
                                if not isinstance(metric_df_original, pd.DataFrame):
                                    print(f"      Skipping round {round_name}, phase {phase_name} for metric {metric_name} ({current_method_inner_loop}): Expected a DataFrame, got {type(metric_df_original)}.")
                                    continue
                                if metric_df_original.empty or 'value' not in metric_df_original.columns or 'tenant' not in metric_df_original.columns:
                                    print(f"      Skipping correlation for metric {metric_name}, round {round_name}, phase {phase_name} ({current_method_inner_loop}) due to missing data or columns.")
                                    continue

                                metric_df = metric_df_original.copy()
                                print(f"      Calculating correlation ({current_method_inner_loop}) for metric: {metric_name}, round: {round_name}, phase: {phase_name}")
                                try:
                                    if 'timestamp' not in metric_df.columns:
                                        print(f"      Skipping {metric_name}, round {round_name}, phase {phase_name} ({current_method_inner_loop}): 'timestamp' column not found.")
                                        continue

                                    correlation_matrix_df = calculate_inter_tenant_correlation_per_metric(
                                        metric_df, method=current_method_inner_loop, time_col='timestamp'
                                    )
                                    if correlation_matrix_df is not None and not correlation_matrix_df.empty:
                                        method_correlation_dfs_for_current_scope_phase[current_method_inner_loop] = correlation_matrix_df
                                        display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
                                        plot_filename = f"{metric_name}_{round_name}_{phase_name}_{current_method_inner_loop}_correlation_heatmap.png"
                                        plot_title = f"Inter-Tenant Correlation ({current_method_inner_loop.capitalize()}): {display_metric_name} (Round: {round_name}, Phase: {phase_name})"
                                        fig = plot_correlation_heatmap(
                                            correlation_matrix_df, title=plot_title, output_dir=correlation_plots_dir, filename=plot_filename
                                        )
                                        if fig:
                                            print(f"      Correlation heatmap for {metric_name}, round {round_name}, phase {phase_name} ({current_method_inner_loop}) generated and saved.")
                                            plt.close(fig)
                                        csv_filename_name_only = f"{metric_name}_{round_name}_{phase_name}_{current_method_inner_loop}_correlation_matrix.csv"
                                        full_csv_path = os.path.join(correlation_tables_dir, csv_filename_name_only)
                                        export_to_csv(correlation_matrix_df, full_csv_path)
                                        print(f"      Correlation matrix for {metric_name}, round {round_name}, phase {phase_name} ({current_method_inner_loop}) saved to {full_csv_path}")
                                    else:
                                        print(f"      Skipping plot/save for {metric_name}, round {round_name}, phase {phase_name} ({current_method_inner_loop}): Correlation matrix is empty or could not be calculated.")
                                except Exception as e:
                                    print(f"      Error during correlation analysis for metric {metric_name}, round {round_name}, phase {phase_name} ({current_method_inner_loop}): {e}")
                                    import traceback
                                    traceback.print_exc()

                            if len(method_correlation_dfs_for_current_scope_phase) > 1:
                                print(f"    Generating scatter plot comparisons for metric: {metric_name}, round: {round_name}, phase: {phase_name}")
                                for method1_name, method2_name in itertools.combinations(method_correlation_dfs_for_current_scope_phase.keys(), 2):
                                    df1 = method_correlation_dfs_for_current_scope_phase[method1_name]
                                    df2 = method_correlation_dfs_for_current_scope_phase[method2_name]

                                    common_tenants = sorted(list(df1.index.intersection(df2.index)))
                                    values1, values2 = [], []
                                    tenant_pairs_for_plot = []

                                    for i in range(len(common_tenants)):
                                        for j in range(i + 1, len(common_tenants)):
                                            t1, t2 = common_tenants[i], common_tenants[j]
                                            if t1 in df1.columns and t2 in df1.columns and \
                                               t1 in df2.columns and t2 in df2.columns:
                                                val1 = df1.loc[t1, t2]
                                                val2 = df2.loc[t1, t2]
                                                if pd.notna(val1) and pd.notna(val2):
                                                    values1.append(val1)
                                                    values2.append(val2)
                                                    tenant_pairs_for_plot.append(f"{t1}_vs_{t2}")

                                    if values1 and values2:
                                        scatter_data_dict = {f'{method1_name.capitalize()} vs {method2_name.capitalize()}': (values1, values2)}
                                        display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
                                        scatter_plot_filename = f"scatter_{metric_name}_{round_name}_{phase_name}_{method1_name}_vs_{method2_name}.png"
                                        scatter_plot_title = f"Comparison: {method1_name.capitalize()} vs {method2_name.capitalize()} Corr.\n{display_metric_name} (Round: {round_name}, Phase: {phase_name})"

                                        fig_scatter = plot_scatter_comparison(
                                            data_dict=scatter_data_dict,
                                            x_label=f'{method1_name.capitalize()} Correlation',
                                            y_label=f'{method2_name.capitalize()} Correlation',
                                            title=scatter_plot_title,
                                            output_dir=correlation_scatter_plots_dir,
                                            filename=scatter_plot_filename
                                        )
                                        if fig_scatter:
                                            print(f"      Scatter plot comparing {method1_name} and {method2_name} for {metric_name}, round {round_name}, phase {phase_name} saved.")
                                            plt.close(fig_scatter)
                                        else:
                                            print(f"      Could not generate scatter plot for {method1_name} vs {method2_name} for {metric_name}, round {round_name}, phase {phase_name}.")
                                    else:
                                        print(f"      Skipping scatter plot for {method1_name} vs {method2_name} for {metric_name}, round {round_name}, phase {phase_name}: No common comparable data points.")
                    else:
                        metric_df_original = phases_or_metric_df
                        print(f"    Processing round: {round_name} for metric: {metric_name} (Consolidated)")
                        method_correlation_dfs_for_current_scope_consolidated = {}
                        for current_method_inner_loop in methods_to_run:
                            if not isinstance(metric_df_original, pd.DataFrame):
                                print(f"      Skipping round {round_name} for metric {metric_name} ({current_method_inner_loop}): Expected a DataFrame, got {type(metric_df_original)}.")
                                continue
                            if metric_df_original.empty or 'value' not in metric_df_original.columns or 'tenant' not in metric_df_original.columns:
                                print(f"      Skipping correlation for metric {metric_name}, round {round_name} ({current_method_inner_loop}) due to missing data or columns.")
                                continue

                            metric_df = metric_df_original.copy()
                            print(f"      Calculating correlation ({current_method_inner_loop}) for metric: {metric_name}, round: {round_name}")
                            try:
                                if 'timestamp' not in metric_df.columns:
                                    print(f"      Skipping {metric_name}, round {round_name} ({current_method_inner_loop}): 'timestamp' column not found.")
                                    continue

                                correlation_matrix_df = calculate_inter_tenant_correlation_per_metric(
                                    metric_df, method=current_method_inner_loop, time_col='timestamp'
                                )
                                if correlation_matrix_df is not None and not correlation_matrix_df.empty:
                                    method_correlation_dfs_for_current_scope_consolidated[current_method_inner_loop] = correlation_matrix_df
                                    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
                                    plot_filename = f"{metric_name}_{round_name}_{current_method_inner_loop}_correlation_heatmap.png"
                                    plot_title = f"Inter-Tenant Correlation ({current_method_inner_loop.capitalize()}): {display_metric_name} (Round: {round_name})"
                                    fig = plot_correlation_heatmap(
                                        correlation_matrix_df, title=plot_title, output_dir=correlation_plots_dir, filename=plot_filename
                                    )
                                    if fig:
                                        print(f"      Correlation heatmap for {metric_name}, round {round_name} ({current_method_inner_loop}) generated and saved.")
                                        plt.close(fig)
                                    csv_filename_name_only = f"{metric_name}_{round_name}_{current_method_inner_loop}_correlation_matrix.csv"
                                    full_csv_path = os.path.join(correlation_tables_dir, csv_filename_name_only)
                                    export_to_csv(correlation_matrix_df, full_csv_path)
                                    print(f"      Correlation matrix for {metric_name}, round {round_name} ({current_method_inner_loop}) saved to {full_csv_path}")
                                else:
                                    print(f"      Skipping plot/save for {metric_name}, round {round_name} ({current_method_inner_loop}): Correlation matrix is empty or could not be calculated.")
                            except Exception as e:
                                print(f"      Error during correlation analysis for metric {metric_name}, round {round_name} ({current_method_inner_loop}): {e}")
                                import traceback
                                traceback.print_exc()

                        if len(method_correlation_dfs_for_current_scope_consolidated) > 1:
                            print(f"    Generating scatter plot comparisons for metric: {metric_name}, round: {round_name} (Consolidated)")
                            for method1_name, method2_name in itertools.combinations(method_correlation_dfs_for_current_scope_consolidated.keys(), 2):
                                df1 = method_correlation_dfs_for_current_scope_consolidated[method1_name]
                                df2 = method_correlation_dfs_for_current_scope_consolidated[method2_name]

                                common_tenants = sorted(list(df1.index.intersection(df2.index)))
                                values1, values2 = [], []
                                tenant_pairs_for_plot = []

                                for i in range(len(common_tenants)):
                                    for j in range(i + 1, len(common_tenants)):
                                        t1, t2 = common_tenants[i], common_tenants[j]
                                        if t1 in df1.columns and t2 in df1.columns and \
                                           t1 in df2.columns and t2 in df2.columns:
                                            val1 = df1.loc[t1, t2]
                                            val2 = df2.loc[t1, t2]
                                            if pd.notna(val1) and pd.notna(val2):
                                                values1.append(val1)
                                                values2.append(val2)
                                                tenant_pairs_for_plot.append(f"{t1}_vs_{t2}")

                                if values1 and values2:
                                    scatter_data_dict = {f'{method1_name.capitalize()} vs {method2_name.capitalize()}': (values1, values2)}
                                    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
                                    scatter_plot_filename = f"scatter_{metric_name}_{round_name}_consolidated_{method1_name}_vs_{method2_name}.png"
                                    scatter_plot_title = f"Comparison: {method1_name.capitalize()} vs {method2_name.capitalize()} Corr.\n{display_metric_name} (Round: {round_name}, Consolidated)"

                                    fig_scatter = plot_scatter_comparison(
                                        data_dict=scatter_data_dict,
                                        x_label=f'{method1_name.capitalize()} Correlation',
                                        y_label=f'{method2_name.capitalize()} Correlation',
                                        title=scatter_plot_title,
                                        output_dir=correlation_scatter_plots_dir,
                                        filename=scatter_plot_filename
                                    )
                                    if fig_scatter:
                                        print(f"      Scatter plot comparing {method1_name} and {method2_name} for {metric_name}, round {round_name} (Consolidated) saved.")
                                        plt.close(fig_scatter)
                                    else:
                                        print(f"      Could not generate scatter plot for {method1_name} vs {method2_name} for {metric_name}, round {round_name} (Consolidated).")
                                else:
                                    print(f"      Skipping scatter plot for {method1_name} vs {method2_name} for {metric_name}, round {round_name} (Consolidated): No common comparable data points.")

    # --- Covariance Analysis ---
    if args.run_covariance:
        print("\nRunning Covariance Analysis...")
        covariance_plots_dir = os.path.join(plots_dir, 'covariance')
        os.makedirs(covariance_plots_dir, exist_ok=True)
        covariance_tables_dir = os.path.join(tables_dir, 'covariance')
        os.makedirs(covariance_tables_dir, exist_ok=True)

        for metric_name, rounds_or_phases_data in all_metrics_data.items():
            print(f"  Processing metric: {metric_name}")
            if not isinstance(rounds_or_phases_data, dict):
                print(f"    Skipping metric {metric_name}: Expected a dictionary of round/phase data, got {type(rounds_or_phases_data)}.")
                continue

            for round_name, phases_or_metric_df in rounds_or_phases_data.items():
                if run_per_phase_analysis:
                    if not isinstance(phases_or_metric_df, dict):
                        print(f"    Skipping round {round_name} for metric {metric_name}: Expected a dictionary of phase data, got {type(phases_or_metric_df)}.")
                        continue
                    for phase_name, metric_df_original in phases_or_metric_df.items():
                        print(f"    Processing round: {round_name}, phase: {phase_name} for metric: {metric_name}")
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
                else:
                    metric_df_original = phases_or_metric_df
                    print(f"    Processing round: {round_name} for metric: {metric_name} (Consolidated)")
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

    # --- Descriptive Statistics Analysis ---
    if args.run_descriptive_stats:
        print("\nRunning Descriptive Statistics Analysis...")
        output_tables_descriptive_stats_dir = os.path.join(tables_dir, "descriptive_stats")
        os.makedirs(output_tables_descriptive_stats_dir, exist_ok=True)
        output_plots_descriptive_stats_dir = os.path.join(plots_dir, "descriptive_stats")  # New directory for plots
        os.makedirs(output_plots_descriptive_stats_dir, exist_ok=True)

        for metric_name, rounds_or_phases_data in all_metrics_data.items():
            print(f"  Processing metric: {metric_name}")
            if not isinstance(rounds_or_phases_data, dict):
                print(f"    Skipping metric {metric_name}: Expected a dictionary of round/phase data, got {type(rounds_or_phases_data)}.")
                continue

            for round_name, phases_or_metric_df in rounds_or_phases_data.items():
                if run_per_phase_analysis:
                    if not isinstance(phases_or_metric_df, dict):
                        print(f"    Skipping round {round_name} for metric {metric_name}: Expected a dictionary of phase data, got {type(phases_or_metric_df)}.")
                        continue
                    for phase_name, phase_df in phases_or_metric_df.items():
                        print(f"    Processing round: {round_name}, phase: {phase_name} for metric: {metric_name}")
                        if not isinstance(phase_df, pd.DataFrame):
                            print(f"      Skipping round {round_name}, phase {phase_name} for metric {metric_name}: Expected a DataFrame, got {type(phase_df)}.")
                            continue
                        if phase_df.empty or 'value' not in phase_df.columns or 'tenant' not in phase_df.columns:
                            print(f"      Skipping descriptive statistics for metric {metric_name}, round {round_name}, phase {phase_name} due to missing data or columns.")
                            continue

                        # CSV Output
                        groupby_cols_desc = ['tenant'] if 'tenant' in phase_df.columns else None
                        descriptive_stats_df = calculate_descriptive_statistics(
                            phase_df,
                            metric_column='value',
                            groupby_cols=groupby_cols_desc
                        )
                        if descriptive_stats_df is not None and not descriptive_stats_df.empty:
                            filename_csv = f"{metric_name}_{round_name}_{phase_name}_descriptive_stats.csv"
                            full_csv_path = os.path.join(output_tables_descriptive_stats_dir, filename_csv)
                            export_to_csv(descriptive_stats_df, full_csv_path)
                            print(f"      Descriptive statistics for {metric_name}, round {round_name}, phase {phase_name} saved to {full_csv_path}")
                        else:
                            print(f"      No descriptive statistics generated for {metric_name}, round {round_name}, phase {phase_name}.")

                        # Plotting

                        # Line Plot
                        filename_lineplot = f"{metric_name}_{round_name}_{phase_name}_lineplot.png"
                        plot_descriptive_stats_lineplot(
                            data_df=phase_df,
                            metric_name=metric_name,
                            value_column='value',
                            title=f"Metric Over Time by Tenant",
                            output_dir=output_plots_descriptive_stats_dir,
                            filename=filename_lineplot,
                            round_name=round_name,
                            phase_name=phase_name
                        )

                        # Box Plot
                        filename_boxplot = f"{metric_name}_{round_name}_{phase_name}_boxplot.png"
                        plot_descriptive_stats_boxplot(
                            data_df=phase_df,
                            metric_name=metric_name,
                            value_column='value',
                            title=f"Metric Distribution by Tenant",
                            output_dir=output_plots_descriptive_stats_dir,
                            filename=filename_boxplot,
                            round_name=round_name,
                            phase_name=phase_name
                        )

                        # Catplot (Bar plot of mean)
                        filename_catplot = f"{metric_name}_{round_name}_{phase_name}_catplot_mean.png"
                        if descriptive_stats_df is not None and not descriptive_stats_df.empty:
                            plot_descriptive_stats_catplot_mean(
                                stats_df=descriptive_stats_df,  # CORRECTED: Using descriptive stats
                                metric_name=metric_name,
                                value_column='mean',  # CORRECTED: Catplot should plot the 'mean'
                                title=f"Mean Metric Value by Tenant",
                                output_dir=output_plots_descriptive_stats_dir,
                                filename=filename_catplot,
                                round_name=round_name,
                                phase_name=phase_name
                            )
                        else:
                            print(f"      Skipping catplot for {metric_name}, round {round_name}, phase {phase_name} due to missing/empty descriptive stats.")
                else:  # Consolidated Analysis
                    metric_df_consolidated = phases_or_metric_df  # In consolidated, this is the DataFrame for the round
                    print(f"    Processing round: {round_name} for metric: {metric_name} (Consolidated)")
                    if not isinstance(metric_df_consolidated, pd.DataFrame):
                        print(f"      Skipping round {round_name} for metric {metric_name} (Consolidated): Expected a DataFrame, got {type(metric_df_consolidated)}.")
                        continue
                    if metric_df_consolidated.empty or 'value' not in metric_df_consolidated.columns or 'tenant' not in metric_df_consolidated.columns:
                        print(f"      Skipping descriptive statistics for metric {metric_name}, round {round_name} (Consolidated) due to missing data or columns.")
                        continue

                    # CSV Output
                    groupby_cols_desc = ['tenant'] if 'tenant' in metric_df_consolidated.columns else None
                    descriptive_stats_df = calculate_descriptive_statistics(
                        metric_df_consolidated,
                        metric_column='value',
                        groupby_cols=groupby_cols_desc
                    )
                    if descriptive_stats_df is not None and not descriptive_stats_df.empty:
                        filename_csv = f"{metric_name}_{round_name}_consolidated_descriptive_stats.csv"
                        full_csv_path = os.path.join(output_tables_descriptive_stats_dir, filename_csv)
                        export_to_csv(descriptive_stats_df, full_csv_path)
                        print(f"      Descriptive statistics for {metric_name}, round {round_name} (Consolidated) saved to {full_csv_path}")
                    else:
                        print(f"      No descriptive statistics generated for {metric_name}, round {round_name} (Consolidated).")

                    # Plotting

                    # Line Plot
                    filename_lineplot = f"{metric_name}_{round_name}_consolidated_lineplot.png"
                    plot_descriptive_stats_lineplot(
                        data_df=metric_df_consolidated,
                        metric_name=metric_name,
                        value_column='value',
                        title=f"Metric Over Time by Tenant (Consolidated)",
                        output_dir=output_plots_descriptive_stats_dir,
                        filename=filename_lineplot,
                        round_name=round_name
                    )

                    # Box Plot
                    filename_boxplot = f"{metric_name}_{round_name}_consolidated_boxplot.png"
                    plot_descriptive_stats_boxplot(
                        data_df=metric_df_consolidated,
                        metric_name=metric_name,
                        value_column='value',
                        title=f"Metric Distribution by Tenant (Consolidated)",
                        output_dir=output_plots_descriptive_stats_dir,
                        filename=filename_boxplot,
                        round_name=round_name
                    )

                    # Catplot (Bar plot of mean)
                    filename_catplot = f"{metric_name}_{round_name}_consolidated_catplot_mean.png"
                    if descriptive_stats_df is not None and not descriptive_stats_df.empty:
                        plot_descriptive_stats_catplot_mean(
                            stats_df=descriptive_stats_df,  # CORRECTED: Using descriptive stats
                            metric_name=metric_name,
                            value_column='mean',  # CORRECTED: Catplot should plot the 'mean'
                            title=f"Mean Metric Value by Tenant (Consolidated)",
                            output_dir=output_plots_descriptive_stats_dir,
                            filename=filename_catplot,
                            round_name=round_name
                        )
                    else:
                        print(f"      Skipping catplot for consolidated {metric_name}, round {round_name} due to missing/empty descriptive stats.")

    # --- Anomaly Detection Analysis ---
    if args.run_anomaly_detection:
        print("\nRunning Anomaly Detection Analysis...")
        anomaly_output_dir = os.path.join(args.output_dir, 'anomaly_detection')
        os.makedirs(anomaly_output_dir, exist_ok=True)
        # This is a placeholder call. You'll need to adapt how data is passed
        # and how results are handled based on the actual implementation.
        run_anomaly_detection_analysis(all_metrics_data, anomaly_output_dir, args)
        print("  Anomaly Detection (placeholder) finished.")

    # --- Tenant-Specific Analysis ---
    if args.run_tenant_analysis:
        print("\nRunning Tenant-Specific Analysis...")
        tenant_output_dir = os.path.join(args.output_dir, 'tenant_analysis')
        os.makedirs(tenant_output_dir, exist_ok=True)
        # Placeholder call
        run_tenant_specific_analysis(all_metrics_data, tenant_output_dir, args)
        print("  Tenant-Specific Analysis (placeholder) finished.")

    # --- Advanced Analysis ---
    if args.run_advanced_analysis:
        print("\nRunning Advanced Analysis...")
        advanced_output_dir = os.path.join(args.output_dir, 'advanced_analysis')
        os.makedirs(advanced_output_dir, exist_ok=True)
        # Placeholder call
        run_advanced_pipeline_analysis(all_metrics_data, advanced_output_dir, args)
        print("  Advanced Analysis (placeholder) finished.")

    print("\nRefactored pipeline processing finished.")


if __name__ == '__main__':
    main()
