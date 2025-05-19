"""
Module for inter-tenant causality analysis.

This module contains functions to investigate and quantify cause-and-effect relationships
between the behavior of different tenants in a Kubernetes environment, especially
in noisy neighbor scenarios.
"""
import pandas as pd
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np

def calculate_causal_impact_between_tenants(
    metric_df: pd.DataFrame,
    source_tenant: str,
    target_tenant: str,
    metric_name: str,
    time_column: str = 'experiment_elapsed_seconds',
    value_column: str = 'value',
    control_tenants: Optional[List[str]] = None,
    max_lag: int = 5,
    verbose: bool = False,
    **kwargs
) -> Dict:
    """
    Calculates the causal impact of one tenant (source) on another tenant (target)
    for a specific metric using Granger Causality.

    Args:
        metric_df (pd.DataFrame): DataFrame containing the time series of metrics.
        source_tenant (str): Identifier of the source tenant.
        target_tenant (str): Identifier of the target tenant.
        metric_name (str): Name of the metric.
        time_column (str): Name of the time column.
        value_column (str): Name of the metric value column.
        control_tenants (Optional[List[str]]): Not used in this implementation.
        max_lag (int): The maximum number of lags to test in Granger Causality.
        verbose (bool): If True, prints the results of the Granger test.
        **kwargs: Additional arguments (currently unused).

    Returns:
        Dict: A dictionary containing the results of the Granger Causality analysis.
              Includes 'min_p_value' and 'best_lag' with the lowest p-value.
    """
    if control_tenants:
        print("Warning: control_tenants are not used in this Granger causality implementation.")

    series_target, series_source = prepare_data_for_granger_causality(
        metric_df,
        target_tenant, # First argument to prepare_data is the 'dependent' variable (effect)
        source_tenant, # Second argument is the 'independent' variable (cause)
        metric_name,
        time_column,
        value_column
    )

    if series_source.empty or series_target.empty:
        return {
            "source_tenant": source_tenant,
            "target_tenant": target_tenant,
            "metric_name": metric_name,
            "error": "One or both series are empty after preparation.",
            "min_p_value": None,
            "best_lag": None,
            "all_p_values": {}
        }

    # Granger causality test requires a DataFrame with both series
    data_for_granger = pd.DataFrame({
        'target': series_target,
        'source': series_source
    })
    data_for_granger = data_for_granger.dropna() # Ensure no NaNs before test

    if len(data_for_granger) < 3 * max_lag: # Heuristic: need enough data points
        return {
            "source_tenant": source_tenant,
            "target_tenant": target_tenant,
            "metric_name": metric_name,
            "error": f"Not enough data for Granger test with max_lag={max_lag}. Data available: {len(data_for_granger)}",
            "min_p_value": None,
            "best_lag": None,
            "all_p_values": {}
        }

    try:
        # The first variable in the DataFrame is the one being caused (target/effect)
        # The second variable is the one causing (source/cause)
        # So, we test if 'source' Granger-causes 'target'
        results = grangercausalitytests(data_for_granger[['target', 'source']], maxlag=max_lag, verbose=verbose)
        
        min_p_value = 1.0
        best_lag = -1
        all_p_values = {}

        for lag in results:
            # Test results are a tuple; p-value is typically the second element of the first test ('ssr_ftest')
            # Example: results[lag][0] is a dict like {'ssr_ftest': (f_stat, p_val, df_num, df_den), ...}
            p_value = results[lag][0]['ssr_ftest'][1]
            all_p_values[lag] = p_value
            if p_value < min_p_value:
                min_p_value = p_value
                best_lag = lag
        
        return {
            "source_tenant": source_tenant,
            "target_tenant": target_tenant,
            "metric_name": metric_name,
            "min_p_value": min_p_value,
            "best_lag": best_lag,
            "all_p_values": all_p_values,
            "message": "Granger Causality test completed."
        }
    except Exception as e:
        # Catch specific exceptions if possible, e.g., LinAlgError for singular matrix
        return {
            "source_tenant": source_tenant,
            "target_tenant": target_tenant,
            "metric_name": metric_name,
            "error": f"Error executing Granger test: {str(e)}",
            "min_p_value": None,
            "best_lag": None,
            "all_p_values": {}
        }

def identify_causal_chains(
    metric_df: pd.DataFrame,
    all_tenants: List[str],
    metric_names: List[str],
    causality_threshold: float = 0.05,
    max_lag_granger: int = 5,
    verbose_granger: bool = False,
    **kwargs
) -> List[Tuple[str, str, str, float, int]]:
    """
    Identifies chains of causal relationships between multiple tenants for a set of metrics,
    using Granger Causality.

    Args:
        metric_df (pd.DataFrame): DataFrame with all time series.
        all_tenants (List[str]): List of all tenants to be considered.
        metric_names (List[str]): List of metrics for which causality is sought.
        causality_threshold (float): Threshold (p-value) to consider a causal relationship as significant.
        max_lag_granger (int): Maximum lag for the Granger test.
        verbose_granger (bool): If True, prints the results of the Granger test.
        **kwargs: Additional arguments for `calculate_causal_impact_between_tenants`.

    Returns:
        List[Tuple[str, str, str, float, int]]: A list of tuples, where each tuple represents
                                           a significant causal link:
                                           (source_tenant, target_tenant, metric_name, p_value, best_lag)
    """
    causal_links = []
    if len(all_tenants) < 2:
        print("At least two tenants are required for causality analysis.")
        return causal_links

    for metric in metric_names:
        print(f"Analyzing causality for metric: {metric}")
        for i in range(len(all_tenants)):
            for j in range(len(all_tenants)):
                if i == j:
                    continue # Do not test causality of a tenant to itself

                source_tenant = all_tenants[i]
                target_tenant = all_tenants[j]

                print(f"  Testing: {source_tenant} -> {target_tenant} for {metric}")
                
                # Pass max_lag and verbose to the calculation function
                causality_result = calculate_causal_impact_between_tenants(
                    metric_df=metric_df,
                    source_tenant=source_tenant,
                    target_tenant=target_tenant,
                    metric_name=metric,
                    max_lag=max_lag_granger,
                    verbose=verbose_granger,
                    **kwargs 
                )

                if "error" in causality_result:
                    print(f"    Error in causality test for {source_tenant} -> {target_tenant} ({metric}): {causality_result['error']}")
                    continue

                p_value = causality_result.get("min_p_value")
                best_lag = causality_result.get("best_lag")

                if p_value is not None and p_value < causality_threshold:
                    print(f"    Significant causality found: {source_tenant} -> {target_tenant} for {metric} (p-value: {p_value:.4f}, lag: {best_lag})")
                    causal_links.append((source_tenant, target_tenant, metric, p_value, best_lag))
                elif p_value is not None:
                    print(f"    Causality not significant: {source_tenant} -> {target_tenant} for {metric} (p-value: {p_value:.4f})")
                else:
                    print(f"    Invalid causality test result for {source_tenant} -> {target_tenant} ({metric}).")


    if not causal_links:
        print("No significant causal links found with the current criteria.")
    else:
        print(f"Total significant causal links found: {len(causal_links)}")
        
    return causal_links

def visualize_causal_graph(
    causal_links: List[Tuple[str, str, str, float, int]], # Added best_lag to tuple
    output_path: Optional[str] = None,
    metric_colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (15, 12),
    node_size: int = 4500,
    font_size: int = 16, # MODIFIED default font_size from 14 to 16
    arrow_size: int = 20,
    layout_type: str = 'spring', # e.g., spring, circular, kamada_kawai
    **kwargs
) -> object:
    """
    Generates a visualization (graph) of the identified causal relationships.

    Args:
        causal_links (List[Tuple[str, str, str, float, int]]):
            List of causal links. Each tuple: 
            (source_tenant, target_tenant, metric_name, p_value, best_lag).
        output_path (Optional[str]): Path to save the graph image.
        metric_colors (Optional[Dict[str, str]]): Dictionary mapping metric names to colors.
        figsize (Tuple[int, int]): Size of the plot figure.
        node_size (int): Size of the nodes in the graph.
        font_size (int): Font size for labels.
        arrow_size (int): Size of the edge arrows.
        layout_type (str): Type of graph layout (e.g., 'spring', 'circular', 'kamada_kawai', 'shell').
        **kwargs: Additional arguments for the visualization library (NetworkX).

    Returns:
        object: Generated matplotlib figure object, or None if saved to file.
    """
    if not causal_links:
        print("No causal links to visualize.")
        return None

    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print("Please install networkx and matplotlib to visualize the graph: pip install networkx matplotlib")
        return {"error": "Missing dependencies: networkx, matplotlib"}

    G = nx.DiGraph()
    
    # Add nodes first to ensure all tenants are present
    all_tenants_in_links = set()
    for source, target, _, _, _ in causal_links:
        all_tenants_in_links.add(source)
        all_tenants_in_links.add(target)
    for tenant in all_tenants_in_links:
        G.add_node(tenant)

    # Map metrics to colors if not provided
    if metric_colors is None:
        unique_metrics = sorted(list(set(link[2] for link in causal_links)))
        # Generate distinct colors (can be improved with a color palette)
        default_colors = plt.cm.get_cmap('tab10', len(unique_metrics) if len(unique_metrics) > 0 else 1)
        metric_colors = {metric: default_colors(i) for i, metric in enumerate(unique_metrics)}
    
    # Add edges with attributes
    edge_labels = {}
    for source, target, metric, p_value, lag in causal_links:
        G.add_edge(source, target, label=f"{metric}\np={p_value:.2f}, lag={lag}", 
                   color=metric_colors.get(metric, 'gray'), weight=(1-p_value), metric=metric)

    # Choose layout
    if layout_type == 'spring':
        pos = nx.spring_layout(G, k=kwargs.get('k', 1.1), iterations=kwargs.get('iterations', 75), seed=kwargs.get('seed', 42)) # MODIFIED k
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, weight='weight') # Use weight for layout
    elif layout_type == 'shell':
        shells = None # Could be [[tenant1, tenant2], [tenant3, tenant4]]
        if 'shells' in kwargs:
            shells = kwargs['shells']
        elif len(all_tenants_in_links) > 0:
            shells = [list(all_tenants_in_links)]
        if shells:
             pos = nx.shell_layout(G, nlist=shells)
        else:
            pos = nx.shell_layout(G) # Fallback if shells cannot be determined
    else:
        pos = nx.spring_layout(G, seed=42) # Default fallback

    plt.figure(figsize=figsize)
    
    edge_colors = [G[u][v]['color'] for u,v in G.edges()]
    
    nx.draw(G, pos, with_labels=True, node_size=node_size, node_color="skyblue", 
            font_size=font_size, font_weight='bold',
            arrows=True, arrowstyle='-|>', arrowsize=arrow_size,
            edge_color=edge_colors, width=2, # width is the thickness of the edge line
            connectionstyle='arc3,rad=0.1')

    current_edge_labels = {(u,v): d['label'] for u,v,d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=current_edge_labels, font_size=font_size-4, # MODIFIED font_size
                                 label_pos=0.35, # ADDED label_pos to shift labels
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))

    plt.title("Directed Causality Graph between Tenants", fontsize=font_size + 4)
    
    if metric_colors:
        patches = [plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=color, 
                        label=f"{metric}")[0]  for metric, color in metric_colors.items() if metric in [d['metric'] for u,v,d in G.edges(data=True)]]
        if patches:
            plt.legend(handles=patches, title="Metrics", loc="best", fontsize=font_size)

    plt.tight_layout()

    if output_path:
        try:
            plt.savefig(output_path, bbox_inches='tight', dpi=kwargs.get('dpi', 300))
            print(f"Causality graph saved to {output_path}")
            plt.close() # Close the figure to free up memory
            return None
        except Exception as e:
            print(f"Error saving graph: {e}")
            try:
                fig = plt.gcf()
                plt.show()
                return fig
            except Exception as e_show:
                 print(f"Error displaying graph: {e_show}")
                 return {"error": f"Error saving and displaying graph: {e}, {e_show}"}

    else:
        fig = plt.gcf()
        return fig

def prepare_data_for_granger_causality(
    metric_df: pd.DataFrame,
    tenant_dependent: str,
    tenant_independent: str,
    metric_name: str,
    time_column: str = 'experiment_elapsed_seconds',
    value_column: str = 'value'
) -> Tuple[pd.Series, pd.Series]:
    """
    Prepares time series of two tenants for Granger Causality analysis.
    The first series returned is the dependent variable (effect).
    The second series returned is the independent variable (cause).

    Args:
        metric_df (pd.DataFrame): DataFrame containing the time series.
        tenant_dependent (str): Identifier of the tenant experiencing the effect.
        tenant_independent (str): Identifier of the tenant causing the effect.
        metric_name (str): Name of the metric.
        time_column (str): Name of the time column.
        value_column (str): Name of the value column.

    Returns:
        Tuple[pd.Series, pd.Series]: Two time series (metric values for tenant_dependent and tenant_independent).
    """
    series_dependent = metric_df[
        (metric_df['tenant'] == tenant_dependent) & (metric_df['metric_name'] == metric_name)
    ].set_index(time_column)[value_column].sort_index()

    series_independent = metric_df[
        (metric_df['tenant'] == tenant_independent) & (metric_df['metric_name'] == metric_name)
    ].set_index(time_column)[value_column].sort_index()

    # Ensure series have the same time index and no NaNs
    aligned_series_dependent, aligned_series_independent = series_dependent.align(series_independent, join='inner')
    
    # Handle NaNs that may arise from alignment or already exist
    aligned_series_dependent = aligned_series_dependent.astype(np.float64).fillna(method='ffill').fillna(method='bfill')
    aligned_series_independent = aligned_series_independent.astype(np.float64).fillna(method='ffill').fillna(method='bfill')
    
    # Remove any remaining NaNs that could not be filled
    common_index = aligned_series_dependent.dropna().index.intersection(aligned_series_independent.dropna().index)
    aligned_series_dependent = aligned_series_dependent.loc[common_index]
    aligned_series_independent = aligned_series_independent.loc[common_index]

    return aligned_series_dependent, aligned_series_independent
