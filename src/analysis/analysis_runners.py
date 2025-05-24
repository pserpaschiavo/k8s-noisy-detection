"""
Analysis runners module - Modular analysis execution functions.

This module contains the refactored analysis functions that were previously
embedded in the large main() function, making the code more maintainable
and testable.

The module provides six main analysis runner functions:
1. run_descriptive_statistics_analysis - Basic statistical summaries
2. run_correlation_covariance_analysis - Inter-tenant correlations and covariances
3. run_causality_analysis - SEM-based causality analysis
4. run_similarity_analysis - Distance correlation, cosine similarity, mutual information
5. run_multivariate_analysis - PCA, ICA, and dimensionality reduction
6. run_root_cause_analysis - Root cause analysis using correlation patterns

Each function follows a consistent pattern:
- Takes all_metrics_data as the primary input
- Accepts specific output directories for plots and tables
- Handles both per-phase and consolidated analysis modes
- Includes comprehensive error handling and logging
- Uses the standardized export and visualization functions

Example usage:
    from .analysis_runners import run_descriptive_statistics_analysis
    
    run_descriptive_statistics_analysis(
        all_metrics_data=metrics_dict,
        desc_stats_tables_dir="output/tables/descriptive",
        desc_stats_plots_dir="output/plots/descriptive", 
        run_per_phase_analysis=True
    )

Dependencies:
    - analysis.* modules for core analysis functions
    - visualization.plots for plotting functions
    - data.io_utils for export utilities
    - utils.common for shared utilities
"""

import logging
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from ..utils.common import plt
from ..data.io_utils import export_to_csv
from .descriptive_statistics import calculate_descriptive_statistics
from .correlation_covariance import (
    calculate_inter_tenant_correlation_per_metric, 
    calculate_inter_tenant_covariance_per_metric
)
from .causality import perform_sem_analysis, plot_sem_path_diagram, plot_sem_fit_indices
from .similarity import (
    calculate_pairwise_distance_correlation,
    calculate_pairwise_cosine_similarity,
    calculate_pairwise_mutual_information,
    plot_distance_correlation_heatmap,
    plot_cosine_similarity_heatmap,
    plot_mutual_information_heatmap
)
from .multivariate import (
    perform_pca, perform_ica, get_top_features_per_component,
    perform_kpca, perform_tsne
)
from .root_cause import perform_complete_root_cause_analysis, calculate_inter_tenant_correlation_per_metric as rca_correlation
from ..visualization.plots import (
    plot_correlation_heatmap,
    plot_covariance_heatmap,
    plot_descriptive_stats_boxplot
)


def run_descriptive_statistics_analysis(all_metrics_data: Dict, 
                                       desc_stats_tables_dir: str,
                                       desc_stats_plots_dir: str,
                                       run_per_phase_analysis: bool) -> None:
    """
    Execute descriptive statistics analysis module.
    
    Args:
        all_metrics_data: Dictionary containing metrics data
        desc_stats_tables_dir: Directory for CSV outputs
        desc_stats_plots_dir: Directory for plot outputs
        run_per_phase_analysis: Whether to run per-phase analysis
    """
    logging.info("Running descriptive statistics module...")
    
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
            logging.debug(f"No data for descriptive stats of {metric_name}, skipping.")
            continue
            
        metric_df = pd.concat(dfs, ignore_index=True)
        
        # Calculate descriptive statistics
        stats_df = calculate_descriptive_statistics(metric_df, metric_column='value')
        
        # Export CSV
        csv_out = os.path.join(desc_stats_tables_dir, f"{metric_name}_descriptive_stats.csv")
        export_to_csv(stats_df, csv_out)
        
        # Generate plots
        try:
            plot_descriptive_stats_boxplot(
                metric_df,
                metric='value',
                title=f"Descriptive Statistics: {metric_name}",
                output_dir=desc_stats_plots_dir
            )
        except Exception as e:
            logging.warning(f"Failed to generate plot for {metric_name}: {e}")


def run_correlation_covariance_analysis(all_metrics_data: Dict,
                                       correlation_tables_dir: str,
                                       correlation_plots_dir: str,
                                       covariance_tables_dir: str,
                                       covariance_plots_dir: str,
                                       run_per_phase_analysis: bool,
                                       correlation_methods: List[str]) -> None:
    """
    Execute correlation and covariance analysis module.
    
    Args:
        all_metrics_data: Dictionary containing metrics data
        correlation_tables_dir: Directory for correlation CSV outputs
        correlation_plots_dir: Directory for correlation plot outputs
        covariance_tables_dir: Directory for covariance CSV outputs
        covariance_plots_dir: Directory for covariance plot outputs
        run_per_phase_analysis: Whether to run per-phase analysis
        correlation_methods: List of correlation methods to use
    """
    logging.info("Running correlation and covariance analysis module...")
    
    for metric_name, rounds_data in all_metrics_data.items():
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
                corr_df = calculate_inter_tenant_correlation_per_metric(
                    df_metric, method=correlation_methods[0], time_col='datetime'
                )
                cov_df = calculate_inter_tenant_covariance_per_metric(
                    df_metric, time_col='datetime'
                )
                
                # Export and plot if not empty
                label = f"{round_key}" + (f"_{phase_key}" if phase_key else "")
                
                if not corr_df.empty:
                    # CSV
                    corr_csv = os.path.join(correlation_tables_dir, f"{metric_name}_{label}_correlation.csv")
                    export_to_csv(corr_df, corr_csv)
                    
                    # Plot
                    try:
                        plot_correlation_heatmap(
                            corr_df,
                            title=f"{metric_name} Correlation ({label})",
                            output_dir=correlation_plots_dir
                        )
                    except Exception as e:
                        logging.warning(f"Failed to generate correlation plot for {metric_name}_{label}: {e}")
                
                if not cov_df.empty:
                    # CSV
                    cov_csv = os.path.join(covariance_tables_dir, f"{metric_name}_{label}_covariance.csv")
                    export_to_csv(cov_df, cov_csv)
                    
                    # Plot
                    try:
                        plot_covariance_heatmap(
                            cov_df,
                            title=f"{metric_name} Covariance ({label})",
                            output_dir=covariance_plots_dir
                        )
                    except Exception as e:
                        logging.warning(f"Failed to generate covariance plot for {metric_name}_{label}: {e}")


def run_causality_analysis(all_metrics_data: Dict,
                          sem_plots_dir: str,
                          sem_tables_dir: str,
                          run_per_phase_analysis: bool,
                          sem_model_spec: str,
                          sem_exog_vars: Optional[List[str]] = None) -> None:
    """
    Execute causal analysis (SEM) module.
    
    Args:
        all_metrics_data: Dictionary containing metrics data
        sem_plots_dir: Directory for SEM plot outputs
        sem_tables_dir: Directory for SEM table outputs
        run_per_phase_analysis: Whether to run per-phase analysis
        sem_model_spec: SEM model specification string
        sem_exog_vars: List of exogenous variables
    """
    logging.info("Running causal analysis module...")
    
    if not sem_model_spec:
        logging.error("SEM model spec is required to run causal analysis. Skipping SEM block.")
        return
        
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
                    
                # Prepare data for SEM: select numeric columns
                sem_data = df_metric.select_dtypes(include=[np.number]).dropna()
                
                try:
                    sem_results = perform_sem_analysis(
                        sem_data,
                        sem_model_spec,
                        exog_vars=sem_exog_vars or []
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
                
                try:
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
                except Exception as e:
                    logging.warning(f"Failed to generate SEM plots for {metric_name}_{round_key}_{phase_key}: {e}")


def run_similarity_analysis(all_metrics_data: Dict,
                           sim_plots_dir: str,
                           sim_tables_dir: str,
                           run_per_phase_analysis: bool) -> None:
    """
    Execute similarity analysis module.
    
    Args:
        all_metrics_data: Dictionary containing metrics data
        sim_plots_dir: Directory for similarity plot outputs
        sim_tables_dir: Directory for similarity table outputs
        run_per_phase_analysis: Whether to run per-phase analysis
    """
    logging.info("Running similarity analysis module...")
    
    time_col_default = 'datetime'
    metric_col_default = 'value'
    group_col_default = 'tenant'

    for metric_name, rounds_data in all_metrics_data.items():
        for round_key, rd_data in rounds_data.items():
            phase_data_map = {}
            if run_per_phase_analysis and isinstance(rd_data, dict):
                phase_data_map = rd_data
            elif not run_per_phase_analysis and isinstance(rd_data, pd.DataFrame):
                phase_data_map = {None: rd_data}
            else:
                logging.debug(f"Skipping similarity for {metric_name} {round_key} due to unexpected data structure")
                continue
            
            for phase_key, df_metric_current in phase_data_map.items():
                if not isinstance(df_metric_current, pd.DataFrame) or df_metric_current.empty:
                    continue

                # Determine actual column names to use
                current_time_col = time_col_default
                if time_col_default not in df_metric_current.columns and 'timestamp' in df_metric_current.columns:
                    current_time_col = 'timestamp'
                
                current_metric_col = metric_col_default
                current_group_col = group_col_default

                if not all(col in df_metric_current.columns for col in [current_time_col, current_metric_col, current_group_col]):
                    logging.warning(f"Missing required columns for similarity analysis in {metric_name} {round_key} {phase_key}")
                    continue

                # Calculate similarity metrics
                distance_corr_df = calculate_pairwise_distance_correlation(
                    df_metric_current, time_col=current_time_col, 
                    metric_col=current_metric_col, group_col=current_group_col
                )
                cosine_sim_df = calculate_pairwise_cosine_similarity(
                    df_metric_current, time_col=current_time_col, 
                    metric_col=current_metric_col, group_col=current_group_col
                )
                mutual_info_df = calculate_pairwise_mutual_information(
                    df_metric_current, time_col=current_time_col, 
                    metric_col=current_metric_col, group_col=current_group_col
                )
                
                label_elements = [metric_name, round_key]
                if phase_key:
                    label_elements.append(phase_key)
                label = "_".join(str(elem) for elem in label_elements if elem is not None)

                # Export and plot results
                _export_and_plot_similarity_results(
                    distance_corr_df, cosine_sim_df, mutual_info_df,
                    label, metric_name, round_key, phase_key,
                    sim_plots_dir, sim_tables_dir
                )


def _export_and_plot_similarity_results(distance_corr_df: pd.DataFrame,
                                       cosine_sim_df: pd.DataFrame,
                                       mutual_info_df: pd.DataFrame,
                                       label: str,
                                       metric_name: str,
                                       round_key: str,
                                       phase_key: Optional[str],
                                       sim_plots_dir: str,
                                       sim_tables_dir: str) -> None:
    """Helper function to export and plot similarity analysis results."""
    
    if not distance_corr_df.empty:
        dist_csv = os.path.join(sim_tables_dir, f"{label}_distance_correlation.csv")
        export_to_csv(distance_corr_df, dist_csv)
        try:
            plot_distance_correlation_heatmap(
                distance_corr_df,
                title=f"{metric_name} Distance Correlation ({round_key}{f'-{phase_key}' if phase_key else ''})",
                output_dir=sim_plots_dir,
                filename=f"{label}_distance_correlation.png",
                tables_dir=sim_tables_dir
            )
        except Exception as e:
            logging.warning(f"Failed to generate distance correlation plot for {label}: {e}")
    
    if not cosine_sim_df.empty:
        cos_csv = os.path.join(sim_tables_dir, f"{label}_cosine_similarity.csv")
        export_to_csv(cosine_sim_df, cos_csv)
        try:
            plot_cosine_similarity_heatmap(
                cosine_sim_df,
                title=f"{metric_name} Cosine Similarity ({round_key}{f'-{phase_key}' if phase_key else ''})",
                output_dir=sim_plots_dir,
                filename=f"{label}_cosine_similarity.png",
                tables_dir=sim_tables_dir
            )
        except Exception as e:
            logging.warning(f"Failed to generate cosine similarity plot for {label}: {e}")
    
    if not mutual_info_df.empty:
        mi_csv = os.path.join(sim_tables_dir, f"{label}_mutual_information.csv")
        export_to_csv(mutual_info_df, mi_csv)
        try:
            plot_mutual_information_heatmap(
                mutual_info_df,
                title=f"{metric_name} Mutual Information ({round_key}{f'-{phase_key}' if phase_key else ''})",
                output_dir=sim_plots_dir,
                filename=f"{label}_mutual_information.png",
                tables_dir=sim_tables_dir
            )
        except Exception as e:
            logging.warning(f"Failed to generate mutual information plot for {label}: {e}")


def run_multivariate_analysis(all_metrics_data: Dict,
                             pca_plots_output_dir: str,
                             pca_tables_output_dir: str,
                             ica_plots_output_dir: str,
                             ica_tables_output_dir: str,
                             comparison_tables_output_dir: str,
                             run_per_phase_analysis: bool,
                             args: Any) -> None:
    """
    Execute multivariate analysis (PCA/ICA) module.
    
    Args:
        all_metrics_data: Dictionary containing metrics data
        pca_plots_output_dir: Directory for PCA plot outputs
        pca_tables_output_dir: Directory for PCA table outputs
        ica_plots_output_dir: Directory for ICA plot outputs
        ica_tables_output_dir: Directory for ICA table outputs
        comparison_tables_output_dir: Directory for comparison table outputs
        run_per_phase_analysis: Whether to run per-phase analysis
        args: Command line arguments object
    """
    logging.info("Running multivariate analysis module...")
    
    if not all_metrics_data:
        logging.warning("Skipping multivariate analysis because all_metrics_data is empty.")
        return
        
    for metric_name, rounds_or_phases_data in all_metrics_data.items():
        if not rounds_or_phases_data:
            continue
            
        for round_name, phases_or_metric_df_data in rounds_or_phases_data.items():
            current_data_for_analysis = None
            analysis_label_suffix = ""

            if run_per_phase_analysis:
                if not isinstance(phases_or_metric_df_data, dict):
                    continue
                
                # Loop through phases within the round for per-phase analysis
                for phase_name, phase_df in phases_or_metric_df_data.items():
                    if not isinstance(phase_df, pd.DataFrame) or phase_df.empty:
                        continue
                    
                    current_data_for_analysis = phase_df
                    analysis_label_suffix = f"{phase_name}"
                    
                    _perform_multivariate_analysis_on_df(
                        current_data_for_analysis, metric_name, analysis_label_suffix, 
                        round_name, args, pca_plots_output_dir, pca_tables_output_dir,
                        ica_plots_output_dir, ica_tables_output_dir, comparison_tables_output_dir
                    )
            else:
                # Consolidated Analysis (per round)
                if not isinstance(phases_or_metric_df_data, pd.DataFrame) or phases_or_metric_df_data.empty:
                    continue
                
                current_data_for_analysis = phases_or_metric_df_data
                analysis_label_suffix = "consolidated"
                
                _perform_multivariate_analysis_on_df(
                    current_data_for_analysis, metric_name, analysis_label_suffix,
                    round_name, args, pca_plots_output_dir, pca_tables_output_dir,
                    ica_plots_output_dir, ica_tables_output_dir, comparison_tables_output_dir
                )


def _perform_multivariate_analysis_on_df(df: pd.DataFrame,
                                        metric_name: str,
                                        analysis_label_suffix: str,
                                        round_name: str,
                                        args: Any,
                                        pca_plots_output_dir: str,
                                        pca_tables_output_dir: str,
                                        ica_plots_output_dir: str,
                                        ica_tables_output_dir: str,
                                        comparison_tables_output_dir: str) -> None:
    """Helper function to perform multivariate analysis on a single DataFrame."""
    
    # Select numeric columns for analysis
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.empty or numeric_df.shape[1] < 2:
        logging.warning(f"Insufficient numeric data for multivariate analysis: {metric_name}_{round_name}_{analysis_label_suffix}")
        return

    label = f"{metric_name}_{round_name}_{analysis_label_suffix}"

    # PCA Analysis
    if args.run_pca:
        try:
            pca_df = perform_pca(
                numeric_df,
                n_components=args.pca_n_components,
                variance_threshold=args.pca_variance_threshold
            )
            
            # Export PCA results
            pca_csv = os.path.join(pca_tables_output_dir, f"{label}_pca.csv")
            export_to_csv(pca_df, pca_csv)
            
            # Create PCA plots
            _create_pca_plots(pca_df, label, pca_plots_output_dir)
            
        except Exception as e:
            logging.error(f"PCA analysis failed for {label}: {e}")

    # ICA Analysis
    if args.run_ica:
        try:
            ica_df = perform_ica(
                numeric_df,
                n_components=args.ica_n_components
            )
            
            # Export ICA results
            ica_csv = os.path.join(ica_tables_output_dir, f"{label}_ica.csv")
            export_to_csv(ica_df, ica_csv)
            
            # Create ICA plots
            _create_ica_plots(ica_df, label, ica_plots_output_dir)
            
        except Exception as e:
            logging.error(f"ICA analysis failed for {label}: {e}")


def _create_pca_plots(pca_df: pd.DataFrame, label: str, output_dir: str) -> None:
    """Create PCA visualization plots."""
    try:
        if pca_df.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], alpha=0.7)
            ax.set_xlabel(pca_df.columns[0])
            ax.set_ylabel(pca_df.columns[1])
            ax.set_title(f'PCA Scatter - {label}')
            
            fig_path = os.path.join(output_dir, f"{label}_pca_scatter.png")
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
    except Exception as e:
        logging.warning(f"Failed to create PCA plots for {label}: {e}")


def _create_ica_plots(ica_df: pd.DataFrame, label: str, output_dir: str) -> None:
    """Create ICA visualization plots."""
    try:
        if ica_df.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(ica_df.iloc[:, 0], ica_df.iloc[:, 1], alpha=0.7)
            ax.set_xlabel(ica_df.columns[0])
            ax.set_ylabel(ica_df.columns[1])
            ax.set_title(f'ICA Scatter - {label}')
            
            fig_path = os.path.join(output_dir, f"{label}_ica_scatter.png")
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
    except Exception as e:
        logging.warning(f"Failed to create ICA plots for {label}: {e}")


def run_root_cause_analysis(all_metrics_data: Dict,
                           root_cause_plots_dir: str,
                           root_cause_tables_dir: str,
                           run_per_phase_analysis: bool,
                           rca_impact_threshold: float) -> None:
    """
    Execute root cause analysis module.
    
    Args:
        all_metrics_data: Dictionary containing metrics data
        root_cause_plots_dir: Directory for RCA plot outputs
        root_cause_tables_dir: Directory for RCA table outputs
        run_per_phase_analysis: Whether to run per-phase analysis
        rca_impact_threshold: Threshold for determining significant impact
    """
    logging.info("Running root cause analysis module...")
    
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
                tenant_names = []
                if 'tenant' in df_metric.columns:
                    tenant_names = df_metric['tenant'].unique().tolist()
                else:
                    # Try to infer tenant names from column names
                    try:
                        tenant_columns = [col for col in df_metric.columns 
                                        if col not in ['datetime', 'timestamp', 'value', 'metric']]
                        if tenant_columns:
                            tenant_names = tenant_columns
                        else:
                            logging.warning(f"Could not determine tenant names for {metric_name} {round_key} {phase_key}")
                            continue
                    except Exception as e:
                        logging.error(f"Error determining tenant names: {e}")
                        continue
                
                # Calculate impact matrix from correlation matrix
                try:
                    corr_df = rca_correlation(df_metric, method='pearson', time_col='datetime')
                    impact_matrix = np.abs(corr_df.values)  # Use absolute correlation values
                    
                    # Apply the threshold
                    impact_matrix = np.where(impact_matrix < rca_impact_threshold, 0, impact_matrix)
                    
                except Exception as e:
                    logging.error(f"Error calculating impact matrix for {metric_name} {round_key} {phase_key}: {e}")
                    continue
                
                # Create result directory
                result_dir = f"{round_key}" + (f"_{phase_key}" if phase_key else "")
                output_dir = os.path.join(root_cause_plots_dir, metric_name, result_dir)
                os.makedirs(output_dir, exist_ok=True)
                
                # Perform root cause analysis
                try:
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
                        })
                        
                        conf_csv = os.path.join(root_cause_tables_dir, f"{metric_name}_{result_dir}_confidence_ranking.csv")
                        export_to_csv(conf_df, conf_csv)
                        
                except Exception as e:
                    logging.error(f"Root cause analysis failed for {metric_name} {round_key} {phase_key}: {e}")
