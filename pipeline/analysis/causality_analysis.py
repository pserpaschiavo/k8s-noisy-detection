"""
Module for performing causality analysis using Transfer Entropy, Convergent Cross Mapping, and Granger Causality.
"""
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
import matplotlib.pyplot as plt
import pyinform
from pyinform.transferentropy import transfer_entropy
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Placeholder for Convergent Cross Mapping specific imports if needed later
# For example, if using a library like skccm, you would import it here.
# from skccm.utilities import train_test_split
# from skccm import CCM

def perform_inter_tenant_causality_analysis(
    data, metrics_for_causality, noisy_tenant, other_tenants, 
    max_lag=5, test='ssr_chi2test', significance_level=0.05, 
    verbose=False, min_observations=30,
    time_col='datetime', value_col='value', tenant_col='tenant', 
    phase_col='phase', round_col='round', current_round='round-1', phase_filter=None
):
    """
    Performs inter-tenant Granger causality analysis.
    (Original function from advanced_analysis.py, adapted)
    """
    results = []
    if data.empty:
        print("Input data for causality analysis is empty. Skipping.")
        return pd.DataFrame(results)

    # Ensure 'datetime' is the index if it exists and is not already
    if time_col in data.columns and data.index.name != time_col:
        data = data.set_index(time_col)
    elif data.index.name != time_col: # If time_col is not in columns and not index name
        print(f"Time column '{time_col}' not found in data. Causality analysis might be incorrect.")
        # Attempt to use the existing index if it's a DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            print("Index is not a DatetimeIndex. Cannot proceed with causality analysis without a proper time column.")
            return pd.DataFrame(results)

    for metric in metrics_for_causality:
        metric_data_frames = {}
        all_tenants_for_metric = [noisy_tenant] + other_tenants

        for tenant in all_tenants_for_metric:
            # Filter data for the current tenant, metric, round, and phase
            tenant_metric_data = data[
                (data[tenant_col] == tenant) &
                (data['metric_name'] == metric)  # Assuming 'metric_name' column exists
            ]
            
            if phase_filter and phase_col in tenant_metric_data.columns:
                tenant_metric_data = tenant_metric_data[tenant_metric_data[phase_col] == phase_filter]

            if round_col in tenant_metric_data.columns:
                 tenant_metric_data = tenant_metric_data[tenant_metric_data[round_col] == current_round]

            # Select and rename the value column to the tenant's name for pivoting
            # Ensure we are working with a copy to avoid SettingWithCopyWarning
            ts = tenant_metric_data[[value_col]].copy()
            ts.rename(columns={value_col: tenant}, inplace=True)
            
            if not ts.empty:
                metric_data_frames[tenant] = ts

        if not metric_data_frames or len(metric_data_frames) < 2:
            # print(f"Not enough tenant data for metric {metric} to perform causality analysis. Skipping.")
            continue

        # Combine all tenant time series for the current metric
        # Ensure all series are aligned by their DatetimeIndex
        combined_ts = pd.concat(metric_data_frames.values(), axis=1) # Pass a list of DataFrames

        # Handle missing values - forward fill then backward fill
        combined_ts.ffill(inplace=True)
        combined_ts.bfill(inplace=True)
        combined_ts.dropna(inplace=True) # Drop any remaining NaNs (e.g. if a series was all NaN)


        if combined_ts.shape[0] < min_observations:
            # print(f"Not enough observations for metric {metric} after alignment and NaN handling ({combined_ts.shape[0]} < {min_observations}). Skipping.")
            continue
        
        # Perform Granger causality from noisy_tenant to each other_tenant
        if noisy_tenant not in combined_ts.columns:
            # print(f"Noisy tenant '{noisy_tenant}' not found in combined data for metric {metric}. Skipping.")
            continue

        for target_tenant in other_tenants:
            if target_tenant not in combined_ts.columns:
                # print(f"Target tenant '{target_tenant}' not found in combined data for metric {metric}. Skipping.")
                continue

            # Test causality from noisy_tenant to target_tenant
            try:
                gc_test_result = grangercausalitytests(
                    combined_ts[[target_tenant, noisy_tenant]], 
                    maxlag=max_lag, verbose=False
                )
                min_p_value = min(gc_test_result[lag][0][test][1] for lag in gc_test_result)
                significant = min_p_value < significance_level
                results.append({
                    'metric': metric,
                    'source_tenant': noisy_tenant,
                    'target_tenant': target_tenant,
                    'lag': max_lag, # This is max_lag tested, not necessarily the optimal lag
                    'p_value': min_p_value,
                    'significant': significant,
                    'direction': f"{noisy_tenant} -> {target_tenant}"
                })
                if verbose:
                    print(f"Granger causality ({noisy_tenant} -> {target_tenant}) for {metric}: p-value={min_p_value:.4f} (Significant: {significant})")
            except Exception as e:
                if verbose:
                    print(f"Error in Granger causality test ({noisy_tenant} -> {target_tenant}) for {metric}: {e}")

            # Test causality from target_tenant to noisy_tenant (reverse direction)
            try:
                gc_test_result_reverse = grangercausalitytests(
                    combined_ts[[noisy_tenant, target_tenant]], 
                    maxlag=max_lag, verbose=False
                )
                min_p_value_reverse = min(gc_test_result_reverse[lag][0][test][1] for lag in gc_test_result_reverse)
                significant_reverse = min_p_value_reverse < significance_level
                results.append({
                    'metric': metric,
                    'source_tenant': target_tenant, # Reversed
                    'target_tenant': noisy_tenant,  # Reversed
                    'lag': max_lag,
                    'p_value': min_p_value_reverse,
                    'significant': significant_reverse,
                    'direction': f"{target_tenant} -> {noisy_tenant}"
                })
                if verbose:
                    print(f"Granger causality ({target_tenant} -> {noisy_tenant}) for {metric}: p-value={min_p_value_reverse:.4f} (Significant: {significant_reverse})")
            except Exception as e:
                if verbose:
                    print(f"Error in Granger causality test ({target_tenant} -> {noisy_tenant}) for {metric}: {e}")

    return pd.DataFrame(results)

def visualize_causal_graph(causality_results_df, output_path, title='Causal Graph', significance_level=0.05):
    """
    Visualizes the causal relationships as a directed graph.
    (Original function from advanced_analysis.py, adapted)
    """
    if causality_results_df.empty:
        print("Causality results are empty. Cannot generate graph.")
        return

    G = nx.DiGraph()
    
    # Filter for significant relationships
    significant_results = causality_results_df[causality_results_df['p_value'] < significance_level]

    if significant_results.empty:
        print(f"No significant causal relationships found at p < {significance_level}. Graph will be empty or show no edges.")
        # Add all unique tenants as nodes even if no edges
        all_tenants_in_results = pd.unique(causality_results_df[['source_tenant', 'target_tenant']].values.ravel('K'))
        for tenant in all_tenants_in_results:
            if tenant: G.add_node(str(tenant)) # Ensure nodes are strings
    else:
        for _, row in significant_results.iterrows():
            source = str(row['source_tenant'])
            target = str(row['target_tenant'])
            metric = str(row['metric'])
            p_value = row['p_value']
            
            G.add_node(source)
            G.add_node(target)
            
            # Add edge with metric and p-value as attributes
            # If an edge already exists, this will update its attributes (NetworkX behavior for DiGraph)
            # Or, if you want to represent multiple metrics, you might need a MultiDiGraph
            # For simplicity, let's assume one dominant metric or label with the strongest p-value if multiple exist
            
            # Check if edge exists and if new p-value is smaller (more significant)
            if G.has_edge(source, target):
                if p_value < G[source][target].get('p_value', float('inf')):
                    G[source][target]['label'] = f"{metric}\n(p={p_value:.3f})"
                    G[source][target]['p_value'] = p_value
                    G[source][target]['weight'] = 1 / (p_value + 1e-6) # Weight inversely proportional to p-value
            else:
                G.add_edge(source, target, label=f"{metric}\n(p={p_value:.3f})", p_value=p_value, weight=1/(p_value + 1e-6))

    if not G.nodes():
        print("No nodes to draw in the causal graph.")
        return

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50) # k adjusts distance, iterations for stability
    
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # Draw edges
    edges = G.edges(data=True)
    if edges:
        nx.draw_networkx_edges(G, pos, edgelist=[(u,v) for u,v,d in edges],
                               arrowstyle='-|>', arrowsize=20, edge_color='gray', alpha=0.6)
        edge_labels = {(u,v): d['label'] for u,v,d in edges if 'label' in d}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3)

    plt.title(title, fontsize=15)
    plt.axis('off')
    try:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Causal graph saved to {output_path}")
    except Exception as e:
        print(f"Error saving causal graph: {e}")
    plt.close()

def save_causality_results_to_csv(causality_results_df, output_path):
    """
    Saves the causality analysis results to a CSV file.
    """
    try:
        causality_results_df.to_csv(output_path, index=False)
        print(f"Causality results saved to {output_path}")
    except Exception as e:
        print(f"Error saving causality results to CSV: {e}")

def calculate_transfer_entropy(series1: pd.Series | np.ndarray, series2: pd.Series | np.ndarray, k: int = 1) -> float:
    """
    Calculates the Transfer Entropy from series2 to series1.
    This measures the amount of information flowing from series2 to series1.

    Args:
        series1: The target time series (pandas Series or NumPy array).
        series2: The source time series (pandas Series or NumPy array).
        k: The history length (embedding dimension).

    Returns:
        The calculated Transfer Entropy.
    """
    if len(series1) != len(series2):
        raise ValueError("Series must have the same length for Transfer Entropy calculation.")
    if len(series1) < k + 1:
        # pyinform requires len(series) > k
        # print(f"Warning: Series length ({len(series1)}) must be greater than k ({k}). Returning NaN.")
        return np.nan

    s1 = np.asarray(series1)
    s2 = np.asarray(series2)

    # Discretize data if it's continuous. PyInform works best with discrete data.
    scaler = MinMaxScaler()
    s1_scaled = scaler.fit_transform(s1.reshape(-1, 1)).flatten()
    s2_scaled = scaler.fit_transform(s2.reshape(-1, 1)).flatten()
    
    n_bins = 3 
    s1_discrete = pd.cut(s1_scaled, bins=n_bins, labels=False, include_lowest=True, duplicates='drop')
    s2_discrete = pd.cut(s2_scaled, bins=n_bins, labels=False, include_lowest=True, duplicates='drop')
    
    if np.isnan(s1_discrete).any() or np.isnan(s2_discrete).any():
        return np.nan
    
    # Ensure data is integer type for pyinform
    s1_discrete = s1_discrete.astype(int)
    s2_discrete = s2_discrete.astype(int)

    try:
        te = transfer_entropy(s1_discrete, s2_discrete, k=k)
    except Exception as e:
        te = np.nan
    return te

def calculate_convergent_cross_mapping(series1: pd.Series | np.ndarray, series2: pd.Series | np.ndarray, embed_dim: int = 2, tau: int = 1, lib_len: int | None = None, pred_len: int | None = None):
    """
    Placeholder for Convergent Cross Mapping (CCM) calculation.
    CCM assesses causality by measuring if the historical record of one variable
    can reliably estimate the state of another.

    NOTE: This is a placeholder. A full CCM implementation requires a library like `skccm`
    or a custom implementation. `pyinform` does not directly provide CCM.

    Args:
        series1: The first time series (potential cause or effect).
        series2: The second time series (potential cause or effect).
        embed_dim: Embedding dimension for state space reconstruction.
        tau: Time delay for state space reconstruction.
        lib_len: Length of the library (training) segment.
        pred_len: Length of the prediction (testing) segment.

    Returns:
        A dictionary with CCM scores (currently np.nan).
    """
    print("Convergent Cross Mapping (calculate_convergent_cross_mapping) is a placeholder and not fully implemented. It will return NaN values. A dedicated library like skccm or a custom implementation is needed.")
    # Example structure if using skccm:
    # s1_np = np.asarray(series1).flatten()
    # s2_np = np.asarray(series2).flatten()
    # if lib_len is None: lib_len = int(0.75 * len(s1_np))
    # ccm = CCM(X=s1_np, Y=s2_np, tau=tau, E=embed_dim, L=lib_len)
    # score_s1_xmap_s2 = ccm.score()
    # ccm = CCM(X=s2_np, Y=s1_np, tau=tau, E=embed_dim, L=lib_len)
    # score_s2_xmap_s1 = ccm.score()
    # return {
    #     "ccm_s1_xmaps_s2": np.mean(score_s1_xmap_s2) if score_s1_xmap_s2 is not None else np.nan,
    #     "ccm_s2_xmaps_s1": np.mean(score_s2_xmap_s1) if score_s2_xmap_s1 is not None else np.nan
    # }
    return {"ccm_s1_xmaps_s2": np.nan, "ccm_s2_xmaps_s1": np.nan}
