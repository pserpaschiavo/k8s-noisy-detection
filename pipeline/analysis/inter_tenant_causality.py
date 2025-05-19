"""
Module for inter-tenant causality analysis.

This module contains functions to investigate and quantify cause-and-effect relationships
between the behavior of different tenants in a Kubernetes environment, especially
in noisy neighbor scenarios.
"""
import logging
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
import matplotlib.pyplot as plt
import os
from ..config import (
    DEFAULT_CAUSALITY_MAX_LAG, DEFAULT_CAUSALITY_THRESHOLD_P_VALUE, 
    DEFAULT_METRICS_FOR_CAUSALITY, CAUSALITY_METRIC_COLORS, METRIC_DISPLAY_NAMES,
    CAUSALITY_FIGURE_SIZE # Added import for CAUSALITY_FIGURE_SIZE
)

logger = logging.getLogger(__name__)

def prepare_data_for_granger_causality(data, metric_column_name, phase_name, config):
    """
    Prepares data for Granger causality testing.
    Filters data for the specified phase, pivots it to have time as index and tenants as columns.
    Handles missing values by interpolation.

    Args:
        data (pd.DataFrame): DataFrame with 'timestamp', 'tenant', and metric_column_name.
        metric_column_name (str): The name of the metric column to be used as values.
        phase_name (str): The name of the phase to filter data for (used for logging).
        config: Configuration object.

    Returns:
        tuple: (pd.DataFrame, list): Pivoted DataFrame ready for causality testing, and list of tenants.
               Returns (empty DataFrame, empty list) if data is unsuitable.
    """
    min_obs_granger = config.GRANGER_MIN_OBSERVATIONS

    # Ensure required columns exist
    required_cols = ['timestamp', 'tenant', metric_column_name]
    if not all(col in data.columns for col in required_cols):
        logger.warning(
            f"Data for phase '{phase_name}', metric value column '{metric_column_name}' is missing one or more required columns. "
            f"Required: {required_cols}. Available: {data.columns.tolist()}. "
            f"Skipping Granger causality preparation."
        )
        return pd.DataFrame(), []

    # Data is assumed to be already filtered by phase and metric_name before calling this function.
    data_to_pivot = data.copy()

    if data_to_pivot.empty:
        logger.info(f"No data provided for metric value column '{metric_column_name}' in phase '{phase_name}' for pivoting.")
        return pd.DataFrame(), []

    # Pivot table: time as index, tenants as columns, metric_column_name as values
    try:
        pivot_df = data_to_pivot.pivot_table(index='timestamp', columns='tenant', values=metric_column_name)
    except KeyError as e:
        logger.error(f"KeyError during pivot for metric value column '{metric_column_name}', phase '{phase_name}': {e}. "
                     f"This might happen if 'timestamp', 'tenant', or '{metric_column_name}' are missing in the slice despite initial checks.")
        return pd.DataFrame(), []
    except Exception as e:
        logger.error(f"An unexpected error occurred during pivot for metric value column '{metric_column_name}', phase '{phase_name}': {e}")
        return pd.DataFrame(), []

    # Handle missing values
    pivot_df.interpolate(method='linear', axis=0, limit_direction='both', inplace=True) # Interpolate
    pivot_df.fillna(method='bfill', inplace=True) # Backfill for any leading NaNs after interpolation
    pivot_df.fillna(method='ffill', inplace=True) # Forwardfill for any remaining NaNs

    # Drop tenants (columns) that are still all NaN after filling
    pivot_df.dropna(axis=1, how='all', inplace=True)
    # Drop timestamps (rows) that have any NaN if critical (e.g., if a tenant appeared late and has leading NaNs not filled)
    pivot_df.dropna(axis=0, how='any', inplace=True)

    tenants = pivot_df.columns.tolist()

    if pivot_df.empty or len(tenants) < 2:
        logger.info(
            f"Pivot table for metric value column '{metric_column_name}', phase '{phase_name}' is empty or has fewer than 2 tenants ({len(tenants)}) after NaN handling. "
            f"Cannot perform Granger causality."
        )
        return pd.DataFrame(), tenants # Return tenants found, even if empty df

    if len(pivot_df) < min_obs_granger:
        logger.info(
            f"Insufficient observations for Granger causality for metric value column '{metric_column_name}', phase '{phase_name}'. "
            f"Need {min_obs_granger}, got {len(pivot_df)}. Table shape: {pivot_df.shape}"
        )
        return pd.DataFrame(), tenants # Return tenants, even if data is insufficient

    for tenant_col in tenants:
        if pivot_df[tenant_col].nunique() == 1:
            logger.warning(
                f"Tenant '{tenant_col}' has constant values for metric value column '{metric_column_name}', phase '{phase_name}'. "
                f"This may cause issues with Granger causality (e.g., perfect multicollinearity or singular matrix)."
            )
    return pivot_df, tenants


def granger_causality_test(data_df, tenants, max_lag, config, metric_name, phase_name):
    """
    Performs Granger causality tests between all pairs of tenants.

    Args:
        data_df (pd.DataFrame): Time-series data with tenants as columns.
        tenants (list): List of tenant names (columns in data_df).
        max_lag (int): Maximum lag for the Granger causality test.
        config: Configuration object.
        metric_name (str): Name of the metric being analyzed (for logging).
        phase_name (str): Name of the phase being analyzed (for logging).

    Returns:
        pd.DataFrame: DataFrame with significant causal links, or empty if none.
    """
    causal_links = []
    significance_level = config.GRANGER_SIGNIFICANCE_LEVEL

    if len(tenants) < 2:
        logger.info(f"Not enough tenants ({len(tenants)}) to perform Granger causality for metric '{metric_name}', phase '{phase_name}'.")
        return pd.DataFrame()

    for target_tenant in tenants:
        for source_tenant in tenants:
            if target_tenant == source_tenant:
                continue

            test_data = data_df[[target_tenant, source_tenant]].copy()
            
            # Check for NaNs or Infs that might have slipped through or resulted from operations
            if test_data.isnull().values.any() or np.isinf(test_data.values).any():
                logger.warning(f"NaN or Inf values found in data for {source_tenant} -> {target_tenant} for metric '{metric_name}', phase '{phase_name}'. Skipping this pair.")
                # Attempt to drop rows with NaNs for this specific pair test
                test_data.dropna(inplace=True)
                if test_data.shape[0] < config.GRANGER_MIN_OBSERVATIONS: # Check length again after dropna
                    logger.warning(f"Insufficient data after dropping NaNs for {source_tenant} -> {target_tenant}. Skipping.")
                    continue

            # Check for constant series again, as this is a common issue for statsmodels
            if test_data[source_tenant].nunique() == 1 or test_data[target_tenant].nunique() == 1:
                logger.warning(f"Constant series detected for {source_tenant} or {target_tenant} (metric '{metric_name}', phase '{phase_name}'). Skipping Granger for this pair.")
                continue
            
            if test_data.shape[0] < max_lag + 1: # Or some other minimum, e.g., config.GRANGER_MIN_OBSERVATIONS
                 logger.warning(f"Insufficient data points ({test_data.shape[0]}) for max_lag {max_lag} for {source_tenant} -> {target_tenant} (metric '{metric_name}', phase '{phase_name}'). Skipping.")
                 continue

            try:
                gc_results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
                for lag in range(1, max_lag + 1):
                    p_value = gc_results[lag][0]['ssr_ftest'][1] # F-test p-value
                    if p_value < significance_level:
                        causal_links.append({
                            'Source Tenant': source_tenant,
                            'Target Tenant': target_tenant,
                            'Lag (Granger)': lag,
                            'P-Value (Granger)': p_value,
                            'Metric': metric_name,
                            'Phase': phase_name
                        })
                        logger.debug(f"Granger causality: {source_tenant} -> {target_tenant} at lag {lag} (p={p_value:.4f}) for metric '{metric_name}', phase '{phase_name}'")
                        break # Found significance at this lag, no need to check further lags for this pair
            except Exception as e:
                # Common errors: "LinAlgError: Singular matrix", "ValueError: The number of observations is not sufficient"
                logger.error(f"Error in Granger causality test for {source_tenant} -> {target_tenant} (metric '{metric_name}', phase '{phase_name}'): {e}")

    if not causal_links:
        logger.info(f"No significant Granger causal links found for metric '{metric_name}', phase '{phase_name}' with max_lag={max_lag} and p<{significance_level}.")
    return pd.DataFrame(causal_links)


def perform_inter_tenant_causality_analysis(processed_data_df, aggregated_data_df, config, output_dir, experiment_name, use_aggregated_data=False):
    """
    Performs inter-tenant causality analysis.
    If use_aggregated_data is True, it plots tenant nodes without causality links,
    as aggregated data (summary stats) is not suitable for time-series Granger causality.
    Otherwise, it uses processed_data_df for Granger causality.

    Args:
        processed_data_df (pd.DataFrame): DataFrame with processed time-series data (per round).
                                          Expected columns: 'timestamp', 'tenant', 'metric_name', 'phase', 'value'.
        aggregated_data_df (pd.DataFrame): DataFrame with aggregated data across rounds.
                                           Expected columns: 'tenant', 'metric_name', 'phase', plus summary stats.
        config: Configuration object.
        output_dir (str): Directory to save causality graphs.
        experiment_name (str): Name of the experiment for graph titles/filenames.
        use_aggregated_data (bool): If True, uses aggregated_data_df; otherwise, uses processed_data_df.
    """
    logger.info(f"Starting inter-tenant causality analysis for {experiment_name} (use_aggregated_data={use_aggregated_data})...")
    os.makedirs(output_dir, exist_ok=True)

    input_df = aggregated_data_df if use_aggregated_data else processed_data_df

    if input_df is None or input_df.empty:
        logger.warning("Input data for causality analysis is None or empty. Skipping.")
        return

    # Essential columns check
    required_cols_general = ['metric_name', 'phase', 'tenant']
    if not all(col in input_df.columns for col in required_cols_general):
        logger.error(f"Input DataFrame is missing one or more essential columns: {required_cols_general}. "
                     f"Available columns: {input_df.columns.tolist()}. Cannot proceed.")
        return

    if not use_aggregated_data and 'value' not in input_df.columns:
        logger.error("'value' column not found in processed_data_df, which is required for time-series causality. Cannot proceed.")
        return
    if not use_aggregated_data and 'timestamp' not in input_df.columns: # Timestamp needed for prepare_data
        logger.error("'timestamp' column not found in processed_data_df, which is required for time-series causality. Cannot proceed.")
        return

    # Determine the max_lag for Granger causality tests
    max_lag = config.granger_max_lag if hasattr(config, 'granger_max_lag') and config.granger_max_lag is not None else DEFAULT_CAUSALITY_MAX_LAG
    # Determine the p-value threshold for significance
    threshold_p_value = config.granger_threshold_p_value if hasattr(config, 'granger_threshold_p_value') and config.granger_threshold_p_value is not None else DEFAULT_CAUSALITY_THRESHOLD_P_VALUE
    # Determine metrics to use for causality analysis
    metrics_for_causality = config.metrics_for_causality if hasattr(config, 'metrics_for_causality') and config.metrics_for_causality else DEFAULT_METRICS_FOR_CAUSALITY

    metrics_to_analyze = config.GRANGER_METRICS_TO_ANALYZE if hasattr(config, 'GRANGER_METRICS_TO_ANALYZE') and config.GRANGER_METRICS_TO_ANALYZE else \
                         sorted(list(input_df['metric_name'].unique()))
    phases_to_analyze = config.GRANGER_PHASES_TO_ANALYZE if hasattr(config, 'GRANGER_PHASES_TO_ANALYZE') and config.GRANGER_PHASES_TO_ANALYZE else \
                        sorted(list(input_df['phase'].unique()))
    
    if not metrics_to_analyze:
        logger.warning("No metrics found or specified to analyze for causality. Skipping.")
        return
    if not phases_to_analyze:
        logger.warning("No phases found or specified to analyze for causality. Skipping.")
        return

    for metric in metrics_to_analyze:
        for phase_name in phases_to_analyze:
            output_dir_phase_metric = os.path.join(output_dir, metric, phase_name)
            os.makedirs(output_dir_phase_metric, exist_ok=True)

            logger.info(f"Analyzing causality for metric: '{metric}', phase: '{phase_name}'")

            data_for_metric_phase = input_df[
                (input_df['metric_name'] == metric) & (input_df['phase'] == phase_name)
            ]

            if data_for_metric_phase.empty:
                logger.info(f"No data available for metric '{metric}' in phase '{phase_name}'. Skipping.")
                continue
            
            current_tenants_for_graph = sorted(data_for_metric_phase['tenant'].unique().tolist())
            if not current_tenants_for_graph:
                logger.info(f"No tenants found in data for metric '{metric}' in phase '{phase_name}'. Skipping.")
                continue

            causal_links_df = pd.DataFrame() # Initialize for this metric/phase

            if use_aggregated_data:
                logger.warning(
                    f"Causality analysis on AGGREGATED data for metric '{metric}', phase '{phase_name}' is not performed "
                    f"as it represents summary statistics, not time series. "
                    f"A graph with tenant nodes (if any) but no causal links will be generated."
                )
                # causal_links_df remains empty. Nodes will be drawn based on current_tenants_for_graph.
            else:
                # Using PROCESSED (per-round, time-series) data
                value_column_for_timeseries = 'value' # This column should contain the time series values

                if value_column_for_timeseries not in data_for_metric_phase.columns:
                    logger.error(f"TimeSeries value column '{value_column_for_timeseries}' not found in the filtered data "
                                 f"for metric '{metric}', phase '{phase_name}'. Skipping Granger causality test for this combination.")
                    # causal_links_df remains empty
                else:
                    prepared_data_df, tenants_in_prepared_data = prepare_data_for_granger_causality(
                        data_for_metric_phase, value_column_for_timeseries, phase_name, config
                    )

                    if prepared_data_df.empty or len(tenants_in_prepared_data) < 2:
                        logger.info(
                            f"No suitable time series data from prepare_data_for_granger_causality or insufficient distinct tenants "
                            f"({len(tenants_in_prepared_data)}) for Granger causality for metric '{metric}', phase '{phase_name}'. "
                            f"Skipping Granger causality test."
                        )
                        # causal_links_df remains empty
                    else:
                        logger.info(f"Running Granger causality test for metric '{metric}', phase '{phase_name}' with tenants: {tenants_in_prepared_data}")
                        causal_links_df = granger_causality_test(
                            prepared_data_df, tenants_in_prepared_data, max_lag, config, metric, phase_name
                        )
            
            visualize_causal_graph(
                causal_links_df=causal_links_df,
                all_tenants_for_graph=current_tenants_for_graph,
                metric=metric,
                phase=phase_name,
                output_dir=output_dir_phase_metric,
                experiment_name=experiment_name,
                config=config
            )

    logger.info(f"Inter-tenant causality analysis for {experiment_name} completed.")


def visualize_causal_graph(causal_links_df, all_tenants_for_graph, metric, phase, output_dir, experiment_name, config):
    """
    Visualizes the causal graph and saves it as a PNG file.
    Nodes are created based on all_tenants_for_graph, ensuring all relevant tenants are shown.

    Args:
        causal_links_df (pd.DataFrame): DataFrame with causal links ('Source Tenant', 'Target Tenant', 'Lag (Granger)', 'P-Value (Granger)').
                                        Can be empty if no links found or if analysis was skipped (e.g. for aggregated data).
        all_tenants_for_graph (list): List of all tenant names that should be included as nodes in the graph.
        metric (str): Name of the metric.
        phase (str): Name of the phase.
        output_dir (str): Directory to save the graph.
        experiment_name (str): Name of the experiment for the graph title.
        config: Configuration object.
    """
    # Define a default filename suffix if not provided in config
    filename_suffix = getattr(config, 'CAUSALITY_FILENAME_SUFFIX', 'causal_graph')

    # Construct filename
    filepath = os.path.join(output_dir, f"{experiment_name}_{metric}_{phase}_{filename_suffix}.png")
    
    G = nx.DiGraph()

    if not all_tenants_for_graph:
        logger.warning(f"No tenants provided for graph of metric '{metric}', phase '{phase}'. Graph will be empty but saved.")
    else:
        # Add all tenants that were present in the data for this metric/phase as nodes
        G.add_nodes_from(sorted(list(set(all_tenants_for_graph)))) # Use set for uniqueness, then sort

    if not causal_links_df.empty:
        logger.info(f"Found {len(causal_links_df)} causal links to draw for metric '{metric}', phase '{phase}'.")
        for _, row in causal_links_df.iterrows():
            source = row['Source Tenant']
            target = row['Target Tenant']
            lag = row.get('Lag (Granger)', 'N/A') # Use .get for safety
            p_value = row.get('P-Value (Granger)', np.nan)
            
            # Ensure nodes exist before adding edge (should be guaranteed by all_tenants_for_graph if data is consistent)
            if source not in G: G.add_node(source) # Should not happen if all_tenants_for_graph is comprehensive
            if target not in G: G.add_node(target) # Should not happen

            G.add_edge(source, target, lag=lag, p_value=p_value)
    else:
        if G.nodes(): # Nodes might exist even if no links
             logger.info(f"No causal links found or computed to draw for metric '{metric}', phase '{phase}'. Graph will show {len(G.nodes())} tenant(s) without edges.")
        # If no nodes and no links, it will be an empty graph.

    # Determine figure size
    fig_size = getattr(config, 'CAUSALITY_FIGURE_SIZE', CAUSALITY_FIGURE_SIZE) # Use imported default

    plt.figure(figsize=fig_size)
    
    title = f"Causality: {metric} - Phase: {phase}\nExperiment: {experiment_name}"
    if not G.nodes():
        logger.info(f"Graph for metric '{metric}', phase '{phase}' has no nodes. Plot will be empty.")
        plt.title(title + " (No Nodes)", fontsize=config.PLOT_TITLE_FONTSIZE)
    else:
        pos = nx.circular_layout(G) # Consider other layouts: spring_layout, kamada_kawai_layout
        
        nx.draw(G, pos, with_labels=True, node_size=config.CAUSALITY_NODE_SIZE, 
                node_color=config.CAUSALITY_NODE_COLOR, font_size=config.CAUSALITY_FONT_SIZE, 
                font_weight='bold', arrows=True, arrowstyle='-|>', arrowsize=config.CAUSALITY_ARROW_SIZE,
                connectionstyle='arc3,rad=0.1') # Curved arrows

        if G.edges():
            edge_labels = {}
            for u, v, d in G.edges(data=True):
                lag_val = d.get('lag', 'N/A')
                pval_val = d.get('p_value', np.nan)
                if pd.notna(pval_val):
                    edge_labels[(u,v)] = f"lag: {lag_val}\np: {pval_val:.3f}"
                else:
                    edge_labels[(u,v)] = f"lag: {lag_val}"
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=config.CAUSALITY_EDGE_FONT_SIZE)
        
        plt.title(title, fontsize=config.PLOT_TITLE_FONTSIZE)

    try:
        plt.tight_layout()
        plt.savefig(filepath)
        logger.info(f"Causality graph saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving causality graph {filepath}: {e}")
    finally:
        plt.close()
