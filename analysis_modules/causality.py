"""
Module for causal analysis techniques on Kubernetes metrics data.

This module implements various causal inference methods to analyze
relationships between metrics and detect potential causal links
that might indicate anomalies or performance issues.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from semopy import Model
from semopy.stats import calc_stats
import networkx as nx
import re
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity

# Set plot aesthetic
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)


def perform_sem_analysis(data, model_spec, exog_vars=None, standardize=True):
    """
    Performs Structural Equation Modeling analysis on the given data.
    
    SEM allows for analyzing complex relationships between variables,
    including direct and indirect effects, and can handle latent variables.
    
    Args:
        data: DataFrame with observed variables
        model_spec: String specification of the SEM model in semopy syntax
        exog_vars: List of exogenous variables (if any)
        standardize: Whether to standardize variables before analysis
        
    Returns:
        Dictionary with:
            - model: The fitted semopy Model object
            - estimates: Parameter estimates
            - stats: Model fit statistics
    """
    # Filter data to include only variables specified in model_spec and any exogenous vars
    # Extract variable names from model_spec
    tokens = re.split(r'[~\+\;:,\*\|\-<>]', model_spec)
    spec_vars = {tok.strip() for tok in tokens if tok.strip()}
    # Include exogenous variables explicitly
    if exog_vars:
        spec_vars.update(exog_vars)
    # Keep only columns present in the data
    data = data.loc[:, data.columns.intersection(spec_vars)]
    if data.empty:
        raise ValueError(f"No data columns match SEM spec vars: {spec_vars}")

    if standardize:
        # Standardize data (mean=0, std=1) for better numerical stability
        # and easier interpretation of results
        data = (data - data.mean()) / data.std()
    
    # Create and fit the model
    model = Model(model_spec)
    model.fit(data)
    
    # Get parameter estimates
    estimates = model.inspect()
    
    # Get model fit statistics
    stats = calc_stats(model)
    # Ensure stats is a dict for consistency
    if not isinstance(stats, dict):
        try:
            stats = stats.to_dict()
        except Exception:
            stats = dict(stats)

    return {
        'model': model,
        'estimates': estimates,
        'stats': stats
    }


def plot_sem_path_diagram(sem_results, title, output_dir, filename, 
                         metric_name=None, round_name=None, phase_name=None,
                         node_labels=None, edge_labels=None, figsize=(12, 10)):
    """
    Creates a path diagram visualization for the SEM model results.
    
    Args:
        sem_results: Dictionary with SEM analysis results
        title: Title of the plot
        output_dir: Directory to save the plot
        filename: Filename to save the plot
        metric_name: Name of the metric analyzed
        round_name: Name of the round analyzed
        phase_name: Name of the phase analyzed
        node_labels: Dictionary mapping variable names to display labels
        edge_labels: Dictionary mapping edges to display labels
        figsize: Figure size as (width, height) tuple
        
    Returns:
        Figure object
    """
    model = sem_results['model']
    estimates = sem_results['estimates']
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    variables = set()
    for _, row in estimates.iterrows():
        variables.add(row['lval'])
        variables.add(row['rval'])
    
    # Use custom labels if provided, otherwise use variable names
    if node_labels is None:
        node_labels = {var: var for var in variables}
    
    # Add nodes to graph
    for var in variables:
        G.add_node(var, label=node_labels.get(var, var))
    
    # Add edges with weights based on estimated coefficients
    for _, row in estimates.iterrows():
        if row['op'] == '~':  # Regression paths
            G.add_edge(row['rval'], row['lval'], 
                      weight=row['Estimate'],
                      pvalue=row.get('p-value', np.nan),
                      std_err=row.get('Std. Err', np.nan))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, 
                          node_color='lightblue', 
                          alpha=0.8, 
                          ax=ax)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, ax=ax)
    
    # Draw edges with width proportional to coefficient magnitude
    # and color indicating positive (green) or negative (red) effect
    edges = G.edges(data=True)
    weights = [abs(data['weight']) * 2 for _, _, data in edges]
    colors = ['green' if data['weight'] > 0 else 'red' for _, _, data in edges]
    
    nx.draw_networkx_edges(G, pos, width=weights, edge_color=colors,
                          arrowsize=20, ax=ax)
    
    # Edge labels showing coefficient values
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)
    
    # Title and styling
    plt.title(title, fontsize=14)
    
    # Add subtitle with additional info if provided
    subtitle = ""
    if metric_name:
        subtitle += f"Metric: {metric_name}"
    if round_name:
        subtitle += f" | Round: {round_name}"
    if phase_name:
        subtitle += f" | Phase: {phase_name}"
    
    if subtitle:
        plt.figtext(0.5, 0.01, subtitle, ha='center', fontsize=12)
    
    # Fit stats text
    stats = sem_results['stats']
    stats_text = "Model fit:\n"
    
    # Add chi-square and p-value if available
    if 'chisq' in stats and 'pvalue' in stats:
        stats_text += f"Chi-sq: {stats['chisq']:.2f} (p={stats['pvalue']:.3f})\n"
    
    # Add CFI and RMSEA (usually available)
    if 'cfi' in stats:
        stats_text += f"CFI: {stats['cfi']:.3f}\n"
    if 'rmsea' in stats:
        stats_text += f"RMSEA: {stats['rmsea']:.3f}"
    
    plt.figtext(0.05, 0.05, stats_text, fontsize=10)
    
    # Remove axis
    plt.axis('off')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    
    return fig


def plot_sem_fit_indices(sem_results, title, output_dir, filename,
                        metric_name=None, round_name=None, phase_name=None,
                        figsize=(10, 8)):
    """
    Creates a visual summary of SEM fit indices.
    
    Args:
        sem_results: Dictionary with SEM analysis results
        title: Title of the plot
        output_dir: Directory to save the plot
        filename: Filename to save the plot
        metric_name: Name of the metric analyzed
        round_name: Name of the round analyzed
        phase_name: Name of the phase analyzed
        figsize: Figure size as (width, height) tuple
        
    Returns:
        Figure object
    """
    stats = sem_results['stats']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Select key fit indices
    indices = ['cfi', 'tli', 'rmsea', 'srmr', 'gfi', 'aic', 'bic']
    values = [stats.get(idx, np.nan) for idx in indices]
    
    # Define thresholds for good fit
    thresholds = {
        'cfi': 0.95,   # CFI > 0.95 indicates good fit
        'tli': 0.95,   # TLI > 0.95 indicates good fit
        'rmsea': 0.06, # RMSEA < 0.06 indicates good fit
        'srmr': 0.08   # SRMR < 0.08 indicates good fit
    }
    
    # Create bar chart
    bars = ax.bar(indices, values, color='lightblue', edgecolor='black')
    
    # Add threshold lines for indices with established guidelines
    for idx, threshold in thresholds.items():
        if idx in indices:
            pos = indices.index(idx)
            if idx in ['rmsea', 'srmr']:  # Lower is better
                ax.axhline(y=threshold, xmin=pos/len(indices), xmax=(pos+1)/len(indices), 
                          color='red', linestyle='--', linewidth=2)
            else:  # Higher is better
                ax.axhline(y=threshold, xmin=pos/len(indices), xmax=(pos+1)/len(indices), 
                          color='green', linestyle='--', linewidth=2)
    
    # Add value labels on top of bars
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f"{val:.3f}", ha='center', va='bottom', rotation=0, fontsize=10)
    
    # Title and styling
    plt.title(title, fontsize=14)
    plt.ylabel('Value', fontsize=12)
    plt.xlabel('Fit Index', fontsize=12)
    
    # Add subtitle with additional info if provided
    subtitle = ""
    if metric_name:
        subtitle += f"Metric: {metric_name}"
    if round_name:
        subtitle += f" | Round: {round_name}"
    if phase_name:
        subtitle += f" | Phase: {phase_name}"
    
    if subtitle:
        plt.figtext(0.5, 0.01, subtitle, ha='center', fontsize=12)
    
    # Add explanation text
    explanation = (
        "CFI/TLI > 0.95: Good fit\n"
        "RMSEA < 0.06: Good fit\n"
        "SRMR < 0.08: Good fit\n"
        "AIC/BIC: Lower values indicate better model"
    )
    plt.figtext(0.02, 0.02, explanation, fontsize=10)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    
    return fig


def plot_sem_coefficient_heatmap(sem_results, title, output_dir, filename,
                               metric_name=None, round_name=None, phase_name=None,
                               figsize=(12, 10)):
    """
    Creates a heatmap of path coefficients from the SEM model.
    
    Args:
        sem_results: Dictionary with SEM analysis results
        title: Title of the plot
        output_dir: Directory to save the plot
        filename: Filename to save the plot
        metric_name: Name of the metric analyzed
        round_name: Name of the round analyzed
        phase_name: Name of the phase analyzed
        figsize: Figure size as (width, height) tuple
        
    Returns:
        Figure object
    """
    estimates = sem_results['estimates']
    
    # Filter for regression paths
    reg_paths = estimates[estimates['op'] == '~'].copy()
    
    # Pivot to create a matrix of coefficients
    matrix = pd.pivot_table(reg_paths, values='Estimate', 
                           index='lval', columns='rval', fill_value=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with a diverging colormap
    heatmap = sns.heatmap(matrix, cmap='coolwarm', center=0, annot=True, 
                        fmt='.2f', linewidths=0.5, ax=ax)
    
    # Title and styling
    plt.title(title, fontsize=14)
    plt.ylabel('Dependent Variable', fontsize=12)
    plt.xlabel('Predictor Variable', fontsize=12)
    
    # Add subtitle with additional info if provided
    subtitle = ""
    if metric_name:
        subtitle += f"Metric: {metric_name}"
    if round_name:
        subtitle += f" | Round: {round_name}"
    if phase_name:
        subtitle += f" | Phase: {phase_name}"
    
    if subtitle:
        plt.figtext(0.5, 0.01, subtitle, ha='center', fontsize=12)
    
    # Add colorbar label
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Path Coefficient', rotation=270, labelpad=20)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    
    return fig


def create_sem_model_from_correlation(correlation_matrix, threshold=0.3):
    """
    Generates a SEM model specification string from a correlation matrix.
    
    Args:
        correlation_matrix (pd.DataFrame): DataFrame of correlations between variables.
        threshold (float): Absolute correlation threshold to include a path in the model.
        
    Returns:
        str: SEM model specification string for semopy.
             Returns an empty string if no significant correlations are found or if the input is invalid.
    """
    if not isinstance(correlation_matrix, pd.DataFrame) or correlation_matrix.empty:
        print("Warning: Invalid or empty correlation matrix provided to create_sem_model_from_correlation.")
        return "" # Return empty string for invalid input

    model_spec = []
    variables = correlation_matrix.columns
    
    # For each variable pair with correlation above threshold
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i != j and abs(correlation_matrix.loc[var1, var2]) >= threshold:
                # If correlation is positive, var1 "causes" var2
                # If correlation is negative, var2 "causes" var1
                # This is a simple heuristic and not necessarily accurate
                if correlation_matrix.loc[var1, var2] > 0:
                    model_spec.append(f"{var2} ~ {var1}")
                else:
                    model_spec.append(f"{var1} ~ {var2}")
    
    return "\n".join(model_spec)


def calculate_transfer_entropy(source, target, k=1, bins=None, bandwidth=0.1):
    """
    Calculate Transfer Entropy from source to target time series.
    
    Transfer Entropy quantifies the information flow from source to target, 
    providing a measure of causal influence. It's defined as:
    TE(X→Y) = H(Y_t+1|Y_t^(k)) - H(Y_t+1|Y_t^(k),X_t^(k))
    where H is the Shannon entropy and Y_t^(k) represents the past k values of Y.
    
    Args:
        source: time series data (potential cause)
        target: time series data (potential effect)
        k: history length to consider (default: 1)
        bins: number of bins for discretization (default: None, adaptive binning)
        bandwidth: bandwidth for kernel density estimation if bins=None
        
    Returns:
        transfer_entropy: The TE value from source to target
    """
    # Convert to numpy arrays and ensure 1D
    source = np.asarray(source).flatten()
    target = np.asarray(target).flatten()
    
    if len(source) != len(target):
        raise ValueError("Source and target must have the same length")
    
    # Create lagged versions of source and target
    source_past = np.roll(source, k)
    target_past = np.roll(target, k)
    target_future = target
    
    # Remove the first k samples (which are invalid due to rolling)
    source_past = source_past[k:]
    target_past = target_past[k:]
    target_future = target_future[k:]
    source = source[k:]  # Current source values
    
    if bins is None:
        # Use kernel density estimation for probability estimation
        def estimate_entropy_kde(x, y=None, z=None):
            if z is not None:
                # For H(X|Y,Z)
                data = np.vstack([x, y, z]).T
            elif y is not None:
                # For H(X|Y)
                data = np.vstack([x, y]).T
            else:
                # For H(X)
                data = x.reshape(-1, 1)
            
            kde = KernelDensity(bandwidth=bandwidth).fit(data)
            log_prob = kde.score_samples(data)
            return -np.mean(log_prob)
        
        # Calculate entropies for TE
        h_target_given_past = estimate_entropy_kde(target_future, target_past)
        h_target_given_past_source = estimate_entropy_kde(
            target_future, target_past, source
        )
        
        # TE = H(Y_t+1|Y_t) - H(Y_t+1|Y_t,X_t)
        te = h_target_given_past - h_target_given_past_source
    else:
        # Discretize the data into bins
        edges = np.linspace(
            min(np.min(source), np.min(target_past), np.min(target_future)),
            max(np.max(source), np.max(target_past), np.max(target_future)),
            bins + 1
        )
        source_binned = np.digitize(source, edges)
        target_past_binned = np.digitize(target_past, edges) 
        target_future_binned = np.digitize(target_future, edges)
        
        # Calculate joint probability distributions using histogram counts
        def calculate_joint_entropy(x, y=None, z=None):
            if z is not None:
                # For H(X,Y,Z)
                joint_hist, _ = np.histogramdd([x, y, z], bins=bins)
            elif y is not None:
                # For H(X,Y)
                joint_hist, _ = np.histogramdd([x, y], bins=bins)
            else:
                # For H(X)
                joint_hist, _ = np.histogram(x, bins=bins)
            
            # Normalize to get probabilities
            joint_prob = joint_hist / np.sum(joint_hist)
            # Remove zero probabilities to avoid log(0)
            joint_prob = joint_prob[joint_prob > 0]
            # Calculate entropy
            return -np.sum(joint_prob * np.log2(joint_prob))
        
        # Calculate entropies for TE: TE(X→Y) = H(Y_t+1|Y_t) - H(Y_t+1|Y_t,X_t)
        h_target_given_past = (
            calculate_joint_entropy(target_future_binned, target_past_binned) - 
            calculate_joint_entropy(target_past_binned)
        )
        h_target_given_past_source = (
            calculate_joint_entropy(target_future_binned, target_past_binned, source_binned) -
            calculate_joint_entropy(target_past_binned, source_binned)
        )
        
        # TE = H(Y_t+1|Y_t) - H(Y_t+1|Y_t,X_t)
        te = h_target_given_past - h_target_given_past_source
    
    return max(0, te)  # Ensure non-negative result


def calculate_pairwise_transfer_entropy(data_df, time_col=None, k=1, bins=None, 
                                       bandwidth=0.1, min_observations=30):
    """
    Calculate pairwise Transfer Entropy between all variables in the dataset.
    
    Args:
        data_df: DataFrame with time series data
        time_col: Name of the time column, if None assumes data is already ordered
        k: History length to consider (default: 1)
        bins: Number of bins for discretization (default: None, adaptive binning)
        bandwidth: Bandwidth for kernel density estimation if bins=None
        min_observations: Minimum number of observations required
        
    Returns:
        DataFrame: DataFrame with Transfer Entropy values for each variable pair
    """
    # Sort by time if time column is provided
    if time_col is not None and time_col in data_df.columns:
        data_df = data_df.sort_values(by=time_col)
    
    # Get numeric columns only
    numeric_cols = data_df.select_dtypes(include=np.number).columns
    
    # Exclude time column if specified
    if time_col is not None and time_col in numeric_cols:
        numeric_cols = numeric_cols.drop(time_col)
    
    # Check if we have enough observations
    if len(data_df) < min_observations:
        print(f"Not enough observations for Transfer Entropy: {len(data_df)} < {min_observations}")
        return pd.DataFrame()  # Return empty DataFrame instead of dictionary
    
    # Initialize results list to store individual results
    te_results_list = []
    
    # Calculate pairwise Transfer Entropy
    for i, source_col in enumerate(numeric_cols):
        for j, target_col in enumerate(numeric_cols):
            if i != j:  # Skip self-causality
                source_series = data_df[source_col].values
                target_series = data_df[target_col].values
                
                # Skip if too many NaNs
                if (np.isnan(source_series).sum() > len(source_series) * 0.2 or 
                    np.isnan(target_series).sum() > len(target_series) * 0.2):
                    continue
                
                # Fill remaining NaNs with interpolation
                source_series = pd.Series(source_series).interpolate().bfill().ffill().values
                target_series = pd.Series(target_series).interpolate().bfill().ffill().values
                
                try:
                    # Calculate Transfer Entropy source -> target
                    te_value = calculate_transfer_entropy(
                        source_series, target_series, 
                        k=k, bins=bins, bandwidth=bandwidth
                    )
                    
                    # Add results to the list instead of dictionary
                    te_results_list.append({
                        'source': source_col,
                        'target': target_col,
                        'transfer_entropy': te_value,
                        'direction': f'{source_col} -> {target_col}'
                    })
                except Exception as e:
                    print(f"Error calculating Transfer Entropy for {source_col} -> {target_col}: {e}")
                    continue
    
    # Convert the list of dictionaries to a DataFrame
    if not te_results_list:
        return pd.DataFrame()  # Return empty DataFrame if no results
    
    return pd.DataFrame(te_results_list)


def plot_transfer_entropy_heatmap(te_results, title, output_dir, filename,
                                 metric_name=None, round_name=None, phase_name=None,
                                 figsize=(12, 10), threshold=0.01):
    """
    Create a heatmap visualization of Transfer Entropy results.
    
    Args:
        te_results: DataFrame or dictionary with Transfer Entropy results
        title: Title of the plot
        output_dir: Directory to save the plot
        filename: Filename to save the plot
        metric_name: Name of the metric analyzed
        round_name: Name of the round analyzed
        phase_name: Name of the phase analyzed
        figsize: Figure size as (width, height) tuple
        threshold: Minimum TE value to display
        
    Returns:
        Figure object
    """
    # Handle empty result case
    if isinstance(te_results, pd.DataFrame) and te_results.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No Transfer Entropy results available",
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        return fig
    elif isinstance(te_results, dict) and not te_results:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No Transfer Entropy results available",
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        return fig
    
    # Convert dictionary to DataFrame if needed
    if isinstance(te_results, dict):
        results_list = []
        for key, result in te_results.items():
            results_list.append(result)
        te_df = pd.DataFrame(results_list)
    else:
        # Already a DataFrame
        te_df = te_results
    
    # Extract all variable names
    all_vars = set()
    for _, row in te_df.iterrows():
        all_vars.add(row['source'])
        all_vars.add(row['target'])
    all_vars = sorted(list(all_vars))
    
    # Create matrix
    n_vars = len(all_vars)
    te_matrix = np.zeros((n_vars, n_vars))
    
    for _, row in te_df.iterrows():
        source_idx = all_vars.index(row['source'])
        target_idx = all_vars.index(row['target'])
        te_value = row['transfer_entropy']
        
        if te_value >= threshold:
            te_matrix[source_idx, target_idx] = te_value
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with mask for values below threshold
    mask = (te_matrix < threshold)
    
    heatmap = sns.heatmap(te_matrix, cmap='viridis', 
                        annot=True, fmt='.3f', 
                        linewidths=0.5, ax=ax,
                        mask=mask,
                        xticklabels=all_vars,
                        yticklabels=all_vars)
    
    # Title and styling
    plt.title(title, fontsize=14)
    plt.ylabel('Source (Cause)', fontsize=12)
    plt.xlabel('Target (Effect)', fontsize=12)
    
    # Add subtitle with additional info if provided
    subtitle = ""
    if metric_name:
        subtitle += f"Metric: {metric_name}"
    if round_name:
        subtitle += f" | Round: {round_name}"
    if phase_name:
        subtitle += f" | Phase: {phase_name}"
    
    if subtitle:
        plt.figtext(0.5, 0.01, subtitle, ha='center', fontsize=12)
    
    # Add colorbar label
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Transfer Entropy', rotation=270, labelpad=20)
    
    # Add explanation
    plt.figtext(0.02, 0.02, 
                f"Transfer Entropy measures information flow from source to target.\n"
                f"Only values ≥ {threshold} are shown. Higher values indicate stronger causal influence.", 
                fontsize=9)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    
    return fig


def plot_transfer_entropy_network(te_results, title, output_dir, filename,
                                 metric_name=None, round_name=None, phase_name=None,
                                 figsize=(14, 12), threshold=0.01):
    """
    Create a network visualization of Transfer Entropy relationships.
    
    Args:
        te_results: DataFrame or dictionary with Transfer Entropy results
        title: Title of the plot
        output_dir: Directory to save the plot
        filename: Filename to save the plot
        metric_name: Name of the metric analyzed
        round_name: Name of the round analyzed
        phase_name: Name of the phase analyzed
        figsize: Figure size as (width, height) tuple
        threshold: Minimum TE value for edge inclusion
        
    Returns:
        Figure object
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Handle empty result case
    if isinstance(te_results, pd.DataFrame) and te_results.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No Transfer Entropy results available",
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        return fig
    elif isinstance(te_results, dict) and not te_results:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No Transfer Entropy results available",
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        return fig
    
    # Convert dictionary to DataFrame if needed
    if isinstance(te_results, dict):
        results_list = []
        for key, result in te_results.items():
            results_list.append(result)
        te_df = pd.DataFrame(results_list)
    else:
        # Already a DataFrame
        te_df = te_results
    
    # Add nodes and edges
    for _, row in te_df.iterrows():
        source = row['source']
        target = row['target']
        te_value = row['transfer_entropy']
        
        # Add nodes
        G.add_node(source)
        G.add_node(target)
        
        # Add edge if above threshold
        if te_value >= threshold:
            G.add_edge(source, target, 
                      weight=te_value,
                      transfer_entropy=te_value)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if we have any edges
    if len(G.edges()) == 0:
        ax.text(0.5, 0.5, f"No significant Transfer Entropy found (threshold ≥ {threshold})",
                ha='center', va='center', fontsize=14)
        plt.axis('off')
    else:
        # Layout
        pos = nx.spring_layout(G, seed=42, k=1.2/np.sqrt(len(G.nodes())))
        
        # Get edge weights for width and color
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(weights) if weights else 1
        weights_normalized = np.array(weights) / max_weight
        
        # Use a color map based on weights
        edge_colors = plt.cm.viridis(weights_normalized)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=3000, 
                            node_color='lightblue', 
                            alpha=0.8, 
                            ax=ax)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)
        
        # Draw edges with width proportional to Transfer Entropy
        nx.draw_networkx_edges(G, pos, width=weights_normalized*5, 
                             edge_color=edge_colors,
                             connectionstyle='arc3,rad=0.15',
                             arrowsize=20, ax=ax)
        
        # Edge labels showing TE values
        edge_labels = {(u, v): f"TE={G[u][v]['transfer_entropy']:.3f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                   font_size=9, ax=ax)
        
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                norm=plt.Normalize(0, max_weight))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Transfer Entropy', rotation=270, labelpad=20)
    
    # Title and styling
    plt.title(title, fontsize=14)
    
    # Add subtitle with additional info if provided
    subtitle = ""
    if metric_name:
        subtitle += f"Metric: {metric_name}"
    if round_name:
        subtitle += f" | Round: {round_name}"
    if phase_name:
        subtitle += f" | Phase: {phase_name}"
    
    if subtitle:
        plt.figtext(0.5, 0.01, subtitle, ha='center', fontsize=12)
    
    # Information text
    plt.figtext(0.05, 0.01, 
                f"Shows Transfer Entropy relationships with TE ≥ {threshold}.\n"
                "Direction: Source → Target. Stronger information flow has thicker arrows.", 
                fontsize=10)
    
    # Remove axis
    plt.axis('off')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    
    return fig


def compare_causal_analysis_methods(data_df, time_col=None, output_dir=None, 
                                   sem_model_spec=None, significance_threshold=0.05):
    """
    Compare results from different causal analysis methods.
    
    Args:
        data_df: DataFrame with time series data
        time_col: Name of the time column
        output_dir: Directory to save comparison plots
        sem_model_spec: SEM model specification (if None, auto-generated)
        significance_threshold: Threshold for considering relationships significant
        
    Returns:
        Dictionary with comparison results
    """
    print("Running comparative causal analysis...")
    
    results = {
        'transfer_entropy': {},
        'ccm': {},
        'granger': {},
        'sem': {},
        'consensus': {}
    }
    
    # Sort by time if time column is provided
    if time_col is not None and time_col in data_df.columns:
        data_df = data_df.sort_values(by=time_col)
    
    # Get numeric data
    numeric_cols = data_df.select_dtypes(include=np.number).columns
    if time_col is not None and time_col in numeric_cols:
        numeric_cols = numeric_cols.drop(time_col)
    
    numeric_df = data_df[numeric_cols].dropna()
    
    try:
        # Transfer Entropy
        print("  Calculating Transfer Entropy...")
        te_results = calculate_pairwise_transfer_entropy(data_df, time_col=time_col)
        results['transfer_entropy'] = te_results
    except Exception as e:
        print(f"  Transfer Entropy failed: {e}")
    
    try:
        # CCM
        print("  Calculating Convergent Cross Mapping...")
        ccm_results = calculate_pairwise_ccm(data_df, time_col=time_col)
        ccm_summary = {}
        for pair, ccm_df in ccm_results.items():
            if not ccm_df.empty:
                ccm_summary[pair] = ccm_df
        results['ccm'] = ccm_summary
    except Exception as e:
        print(f"  CCM failed: {e}")
    
    try:
        # Granger Causality
        print("  Calculating Granger Causality...")
        granger_results = calculate_pairwise_granger_causality(data_df, time_col=time_col)
        results['granger'] = granger_results
    except Exception as e:
        print(f"  Granger Causality failed: {e}")
    
    try:
        # SEM (if model specification provided)
        if sem_model_spec is not None:
            print("  Performing SEM analysis...")
            sem_results = perform_sem_analysis(numeric_df, sem_model_spec)
            results['sem'] = sem_results
        else:
            print("  Skipping SEM (no model specification provided)")
    except Exception as e:
        print(f"  SEM failed: {e}")
    
    # Create consensus analysis
    print("  Creating consensus analysis...")
    consensus_matrix = _create_consensus_matrix(results, numeric_cols, significance_threshold)
    results['consensus'] = consensus_matrix
    
    # Save comparison plot if output directory provided
    if output_dir is not None:
        try:
            _plot_causal_comparison(results, output_dir)
        except Exception as e:
            print(f"  Error creating comparison plot: {e}")
    
    return results


def _create_consensus_matrix(results, variables, threshold):
    """Helper function to create consensus matrix from multiple causal methods."""
    n_vars = len(variables)
    var_to_idx = {var: i for i, var in enumerate(variables)}
    
    # Initialize consensus matrix
    consensus = np.zeros((n_vars, n_vars))
    method_count = np.zeros((n_vars, n_vars))
    
    # Transfer Entropy contributions
    if 'transfer_entropy' in results and results['transfer_entropy'] is not None:
        te_results = results['transfer_entropy']
        
        # Handle both DataFrame and dictionary formats for transfer entropy
        if isinstance(te_results, pd.DataFrame) and not te_results.empty:
            for _, row in te_results.iterrows():
                source_idx = var_to_idx.get(row['source'])
                target_idx = var_to_idx.get(row['target'])
                if source_idx is not None and target_idx is not None:
                    if row['transfer_entropy'] > threshold:
                        consensus[source_idx, target_idx] += 1
                    method_count[source_idx, target_idx] += 1
        elif isinstance(te_results, dict) and te_results:
            for result in te_results.values():
                source_idx = var_to_idx.get(result['source'])
                target_idx = var_to_idx.get(result['target'])
                if source_idx is not None and target_idx is not None:
                    if result['transfer_entropy'] > threshold:
                        consensus[source_idx, target_idx] += 1
                    method_count[source_idx, target_idx] += 1
    
    # Granger Causality contributions
    if 'granger' in results and 'p_values' in results['granger']:
        p_values = results['granger']['p_values']
        for source in p_values.index:
            for target in p_values.columns:
                source_idx = var_to_idx.get(source)
                target_idx = var_to_idx.get(target)
                if source_idx is not None and target_idx is not None and source != target:
                    if p_values.loc[source, target] < 0.05:
                        consensus[source_idx, target_idx] += 1
                    method_count[source_idx, target_idx] += 1
    
    # CCM contributions (simplified)
    if 'ccm' in results and results['ccm']:
        for pair, ccm_df in results['ccm'].items():
            if not ccm_df.empty:
                source, target = pair
                source_idx = var_to_idx.get(source)
                target_idx = var_to_idx.get(target)
                if source_idx is not None and target_idx is not None:
                    # Check if there's evidence of causality in either direction
                    max_skill = ccm_df['prediction_skill'].max()
                    if max_skill > 0.3:  # Arbitrary threshold for CCM
                        consensus[source_idx, target_idx] += 0.5
                        consensus[target_idx, source_idx] += 0.5
                    method_count[source_idx, target_idx] += 1
                    method_count[target_idx, source_idx] += 1
    
    # SEM contributions
    if 'sem' in results and 'estimates' in results['sem']:
        estimates = results['sem']['estimates']
        # Filter for regression paths
        reg_paths = estimates[estimates['op'] == '~']
        if not reg_paths.empty:
            for _, row in reg_paths.iterrows():
                source = row['rval']  # Right-side variable (predictor)
                target = row['lval']  # Left-side variable (dependent)
                source_idx = var_to_idx.get(source)
                target_idx = var_to_idx.get(target)
                
                if source_idx is not None and target_idx is not None:
                    estimate = row['Estimate']
                    p_value = row.get('p-value', 1.0)  # Default to 1.0 if p-value not available
                    
                    # Add to consensus if significant
                    if p_value < 0.05 and abs(estimate) > threshold:
                        consensus[source_idx, target_idx] += 1
                    
                    # Count this evaluation
                    method_count[source_idx, target_idx] += 1
    
    # Normalize by number of methods that evaluated each pair
    consensus_normalized = np.where(method_count > 0, consensus / method_count, 0)
    
    return pd.DataFrame(consensus_normalized, index=variables, columns=variables)


def _plot_causal_comparison(results, output_dir):
    """Helper function to plot comparison of causal analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Transfer Entropy
    if results['transfer_entropy']:
        # Create simplified TE matrix for visualization
        all_vars = set()
        for result in results['transfer_entropy'].values():
            all_vars.add(result['source'])
            all_vars.add(result['target'])
        
        all_vars = sorted(list(all_vars))
        n_vars = len(all_vars)
        var_to_idx = {var: i for i, var in enumerate(all_vars)}
        
        # Create TE matrix
        te_matrix = np.zeros((n_vars, n_vars))
        for result in results['transfer_entropy'].values():
            source_idx = var_to_idx.get(result['source'])
            target_idx = var_to_idx.get(result['target'])
            if source_idx is not None and target_idx is not None:
                te_matrix[source_idx, target_idx] = result['transfer_entropy']
        
        # Plot heatmap
        sns.heatmap(te_matrix, cmap='viridis', ax=axes[0, 0], 
                   xticklabels=all_vars, yticklabels=all_vars, 
                   annot=True, fmt='.3f')
        axes[0, 0].set_title("Transfer Entropy")
        axes[0, 0].set_xlabel("Target (Effect)")
        axes[0, 0].set_ylabel("Source (Cause)")
    else:
        axes[0, 0].text(0.5, 0.5, "Transfer Entropy\nNo Results", 
                       ha='center', va='center', fontsize=12)
        axes[0, 0].set_title("Transfer Entropy")
    
    # Granger Causality
    if 'granger' in results and 'p_values' in results['granger']:
        p_values = results['granger']['p_values']
        sns.heatmap(-np.log10(p_values), cmap='viridis', ax=axes[0, 1], 
                   annot=True, fmt='.1f')
        axes[0, 1].set_title("Granger Causality (-log10 p-values)")
        axes[0, 1].set_xlabel("Target (Effect)")
        axes[0, 1].set_ylabel("Source (Cause)")
    else:
        axes[0, 1].text(0.5, 0.5, "Granger Causality\nNo Results", 
                       ha='center', va='center', fontsize=12)
        axes[0, 1].set_title("Granger Causality")
    
    # CCM
    if results['ccm']:
        # Summarize CCM results for visualization
        all_pairs = list(results['ccm'].keys())
        all_vars = set()
        for source, target in all_pairs:
            all_vars.add(source)
            all_vars.add(target)
        
        all_vars = sorted(list(all_vars))
        n_vars = len(all_vars)
        var_to_idx = {var: i for i, var in enumerate(all_vars)}
        
        # Create CCM matrix (simplified - max prediction skill)
        ccm_matrix = np.zeros((n_vars, n_vars))
        for (source, target), ccm_df in results['ccm'].items():
            if not ccm_df.empty:
                source_idx = var_to_idx.get(source)
                target_idx = var_to_idx.get(target)
                if source_idx is not None and target_idx is not None:
                    ccm_matrix[source_idx, target_idx] = ccm_df['prediction_skill'].max()
        
        # Plot heatmap
        sns.heatmap(ccm_matrix, cmap='viridis', ax=axes[1, 0], 
                   xticklabels=all_vars, yticklabels=all_vars, 
                   annot=True, fmt='.3f')
        axes[1, 0].set_title("Convergent Cross Mapping (Max Prediction Skill)")
        axes[1, 0].set_xlabel("Target Series")
        axes[1, 0].set_ylabel("Shadow Manifold")
    else:
        axes[1, 0].text(0.5, 0.5, "CCM\nNo Results", 
                       ha='center', va='center', fontsize=12)
        axes[1, 0].set_title("Convergent Cross Mapping")
    
    # Consensus
    if 'consensus' in results and not results['consensus'].empty:
        sns.heatmap(results['consensus'], cmap='viridis', ax=axes[1, 1], 
                   annot=True, fmt='.2f')
        axes[1, 1].set_title("Consensus Analysis")
        axes[1, 1].set_xlabel("Target (Effect)")
        axes[1, 1].set_ylabel("Source (Cause)")
    else:
        axes[1, 1].text(0.5, 0.5, "Consensus\nNo Results", 
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_title("Consensus Analysis")
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "causal_analysis_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

def embed_time_series(time_series, E, tau):
    """
    Create time-delayed embedding of a time series.
    
    Args:
        time_series: 1D array of time series data
        E: embedding dimension
        tau: time delay
        
    Returns:
        embedded_series: array with embedded vectors
    """
    N = len(time_series)
    embedded = np.zeros((N - (E-1)*tau, E))
    
    for i in range(N - (E-1)*tau):
        for j in range(E):
            embedded[i, j] = time_series[i + j*tau]
    
    return embedded


def find_nearest_neighbors(embedded_series, query_point, k, exclude_self=True):
    """
    Find k nearest neighbors to a point in the embedded space.
    
    Args:
        embedded_series: array with embedded vectors
        query_point: the query point (either an index into embedded_series or a point vector)
        k: number of nearest neighbors to find
        exclude_self: whether to exclude the query point itself
        
    Returns:
        indices: indices of nearest neighbors
        distances: distances to nearest neighbors
    """
    if k+1 > len(embedded_series):
        k = len(embedded_series) - 1
        
    # Calculate Euclidean distances
    if isinstance(query_point, (int, np.integer)):
        # query_point is an index
        point = embedded_series[query_point]
    else:
        # query_point is a vector
        point = query_point
        
    distances = np.sqrt(np.sum((embedded_series - point)**2, axis=1))
    
    # Sort distances and get indices
    sorted_indices = np.argsort(distances)
    
    if exclude_self and distances[sorted_indices[0]] < 1e-10:
        # Remove self from neighbors (should be at index 0 with distance 0)
        sorted_indices = sorted_indices[1:k+1]
        sorted_distances = distances[sorted_indices]
    else:
        sorted_indices = sorted_indices[:k]
        sorted_distances = distances[sorted_indices]
    
    return sorted_indices, sorted_distances


def predict_using_neighbors(target_series, neighbor_indices, neighbor_distances):
    """
    Predict a value using nearest neighbors with distance weighting.
    
    Args:
        target_series: the series to predict from
        neighbor_indices: indices of nearest neighbors
        neighbor_distances: distances to nearest neighbors
        
    Returns:
        prediction: predicted value
    """
    # Handle zero distances (exact matches)
    if np.any(neighbor_distances == 0):
        zero_dist_idxs = np.where(neighbor_distances == 0)[0]
        return np.mean(target_series[neighbor_indices[zero_dist_idxs]])
    
    # Use exponential weight decay
    weights = np.exp(-neighbor_distances)
    weights /= np.sum(weights)  # Normalize
    
    # Weighted average
    prediction = np.sum(weights * target_series[neighbor_indices])
    
    return prediction


def calculate_ccm(X, Y, E_range=None, tau=1, library_sizes=None, num_predictions=100):
    """
    Calculate Convergent Cross Mapping between X and Y time series.
    
    CCM tests for causality by measuring how well the historical values of one variable
    can predict the other variable. If X causes Y, then Y's attractor manifold should
    contain information about X, allowing Y's history to predict X's current state.
    
    Args:
        X: first time series (potential effect)
        Y: second time series (potential cause)
        E_range: range of embedding dimensions to try (default: [2,3,4])
        tau: time delay for embedding
        library_sizes: list of library sizes to use for prediction
        num_predictions: number of predictions to make for each library size
        
    Returns:
        result_df: DataFrame with library sizes and prediction skills
    """
    X = np.asarray(X).flatten()
    Y = np.asarray(Y).flatten()
    
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length")
    
    if E_range is None:
        E_range = [2, 3, 4]
    
    # Ensure E_range is compatible with data length
    max_possible_E = (len(X) // 2) - 1
    E_range = [E for E in E_range if E <= max_possible_E]
    if not E_range:
        E_range = [2]  # Default to minimum if all values are too large
    
    if library_sizes is None:
        max_library_size = len(X) - max(E_range)*tau - 1
        max_library_size = max(10, max_library_size)
        library_sizes = np.unique(np.round(np.linspace(10, max_library_size, 10)).astype(int))
    
    # Find best embedding dimension using simplex projection
    best_E = E_range[0]
    best_skill = -float('inf')
    
    for E in E_range:
        # Embed X series
        embedded_X = embed_time_series(X, E, tau)
        
        # Create train/test split
        train_size = int(0.8 * len(embedded_X))
        if train_size >= len(embedded_X):
            train_size = len(embedded_X) - 1
        
        # Ensure we don't try to use more test points than available
        max_test_points = min(50, len(embedded_X) - train_size)
        if max_test_points <= 0:
            continue
            
        test_indices = np.random.choice(range(train_size, len(embedded_X)), max_test_points, replace=False)
        
        # Make predictions
        predictions = []
        actuals = []
        
        for idx in test_indices:
            # Find nearest neighbors
            nn_indices, nn_distances = find_nearest_neighbors(embedded_X[:train_size], embedded_X[idx], k=E+1)
            
            # Predict next value - check index bounds
            if idx + E*tau < len(X):
                prediction = predict_using_neighbors(X[E*tau:], nn_indices, nn_distances)
                predictions.append(prediction)
                actuals.append(X[idx + E*tau])
        
        # Calculate prediction skill (correlation)
        if len(predictions) > 1:
            skill = np.corrcoef(predictions, actuals)[0, 1]
            
            if skill > best_skill:
                best_skill = skill
                best_E = E
    
    results = []
    
    # Test X|M_Y (X predicted from Y manifold) - if Y causes X
    for lib_size in library_sizes:
        # Embed Y series with best dimension
        embedded_Y = embed_time_series(Y, best_E, tau)
        
        if lib_size >= len(embedded_Y):
            continue
        
        # Ensure we don't predict beyond available data
        predict_end = min(len(embedded_Y), lib_size + num_predictions)
        if predict_end <= lib_size:
            continue
            
        # Randomly select prediction points
        predict_indices = np.random.choice(
            range(lib_size, predict_end), 
            min(num_predictions, predict_end - lib_size), 
            replace=False
        )
        
        Y_predict_X = []
        X_actual = []
        
        for p_idx in predict_indices:
            # Find nearest neighbors in Y manifold
            nn_indices, nn_distances = find_nearest_neighbors(
                embedded_Y[:lib_size], embedded_Y[p_idx], k=best_E+1
            )
            
            # Ensure we don't index outside available data
            offset = (best_E-1)*tau
            if p_idx + offset >= len(X):
                continue
                
            # Use those neighbors to predict X
            prediction = predict_using_neighbors(X[offset:], nn_indices, nn_distances)
            Y_predict_X.append(prediction)
            X_actual.append(X[p_idx + offset])
        
        # Calculate prediction skill
        if len(Y_predict_X) > 2:  # Need at least 3 points for correlation
            prediction_skill = np.corrcoef(Y_predict_X, X_actual)[0, 1]
            results.append({
                'library_size': lib_size, 
                'prediction_skill': prediction_skill,
                'direction': 'Y causes X'
            })
    
    # Test Y|M_X (Y predicted from X manifold) - if X causes Y
    for lib_size in library_sizes:
        # Embed X series with best dimension
        embedded_X = embed_time_series(X, best_E, tau)
        
        if lib_size >= len(embedded_X):
            continue
        
        # Ensure we don't predict beyond available data
        predict_end = min(len(embedded_X), lib_size + num_predictions)
        if predict_end <= lib_size:
            continue
            
        # Randomly select prediction points
        predict_indices = np.random.choice(
            range(lib_size, predict_end),
            min(num_predictions, predict_end - lib_size), 
            replace=False
        )
        
        X_predict_Y = []
        Y_actual = []
        
        for p_idx in predict_indices:
            # Find nearest neighbors in X manifold
            nn_indices, nn_distances = find_nearest_neighbors(
                embedded_X[:lib_size], embedded_X[p_idx], k=best_E+1
            )
            
            # Ensure we don't index outside available data
            offset = (best_E-1)*tau
            if p_idx + offset >= len(Y):
                continue
                
            # Use those neighbors to predict Y
            prediction = predict_using_neighbors(Y[offset:], nn_indices, nn_distances)
            X_predict_Y.append(prediction)
            Y_actual.append(Y[p_idx + offset])
        
        # Calculate prediction skill
        if len(X_predict_Y) > 2:  # Need at least 3 points for correlation
            prediction_skill = np.corrcoef(X_predict_Y, Y_actual)[0, 1]
            results.append({
                'library_size': lib_size, 
                'prediction_skill': prediction_skill,
                'direction': 'X causes Y'
            })
    
    return pd.DataFrame(results)


def calculate_pairwise_ccm(data_df, time_col=None, E_range=None, tau=1, 
                          library_sizes=None, num_predictions=100, min_observations=30):
    """
    Calculate pairwise CCM between all variables in the dataset.
    
    Args:
        data_df: DataFrame with time series data
        time_col: Name of the time column, if None assumes data is already ordered
        E_range: Range of embedding dimensions to try
        tau: Time delay for embedding
        library_sizes: List of library sizes to use
        num_predictions: Number of predictions for each library size
        min_observations: Minimum number of observations required
        
    Returns:
        Dict: Dictionary with results for each variable pair
    """
    # Sort by time if time column is provided
    if time_col is not None and time_col in data_df.columns:
        data_df = data_df.sort_values(by=time_col)
    
    # Get numeric columns only
    numeric_cols = data_df.select_dtypes(include=np.number).columns
    
    # Exclude time column if specified
    if time_col is not None and time_col in numeric_cols:
        numeric_cols = numeric_cols.drop(time_col)
    
    # Check if we have enough observations
    if len(data_df) < min_observations:
        print(f"Not enough observations for CCM: {len(data_df)} < {min_observations}")
        return {}
    
    # Initialize results dictionary
    ccm_results = {}
    
    # Calculate pairwise CCM
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i < j:  # Calculate each pair only once
                X = data_df[col1].values
                Y = data_df[col2].values
                
                # Skip if too many NaNs
                if np.isnan(X).sum() > len(X) * 0.2 or np.isnan(Y).sum() > len(Y) * 0.2:
                    print(f"Skipping {col1}-{col2} pair due to too many NaNs")
                    continue
                
                # Fill remaining NaNs with interpolation
                X = pd.Series(X).interpolate().bfill().ffill().values
                Y = pd.Series(Y).interpolate().bfill().ffill().values
                
                try:
                    pair_results = calculate_ccm(
                        X, Y, 
                        E_range=E_range, 
                        tau=tau, 
                        library_sizes=library_sizes,
                        num_predictions=num_predictions
                    )
                    
                    pair_results['X'] = col1
                    pair_results['Y'] = col2
                    
                    ccm_results[(col1, col2)] = pair_results
                except Exception as e:
                    print(f"Error calculating CCM for {col1} and {col2}: {e}")
    
    return ccm_results


def plot_ccm_convergence(ccm_results, title, output_dir, filename,
                       metric_name=None, round_name=None, phase_name=None,
                       figsize=(12, 8)):
    """
    Plot CCM convergence to visualize causality between variables.
    
    Args:
        ccm_results: Dictionary with CCM results from calculate_pairwise_ccm
        title: Title of the plot
        output_dir: Directory to save the plot
        filename: Filename to save the plot
        metric_name: Name of the metric analyzed
        round_name: Name of the round analyzed
        phase_name: Name of the phase analyzed
        figsize: Figure size as (width, height) tuple
        
    Returns:
        Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each pair
    if not ccm_results:
        ax.text(0.5, 0.5, "No CCM results available",
                ha='center', va='center', fontsize=14)
    else:
        colors = plt.cm.tab10.colors
        color_idx = 0
        
        for pair, results_df in ccm_results.items():
            X_name, Y_name = pair
            
            # Get results for each direction
            X_causes_Y = results_df[results_df['direction'] == 'X causes Y']
            Y_causes_X = results_df[results_df['direction'] == 'Y causes X']
            
            if not X_causes_Y.empty:
                # Sort by library size to ensure continuous line
                X_causes_Y = X_causes_Y.sort_values('library_size')
                ax.plot(X_causes_Y['library_size'], X_causes_Y['prediction_skill'], 
                       'o-', color=colors[color_idx % len(colors)],
                       label=f"{X_name} causes {Y_name}")
                color_idx += 1
            
            if not Y_causes_X.empty:
                # Sort by library size to ensure continuous line
                Y_causes_X = Y_causes_X.sort_values('library_size')
                ax.plot(Y_causes_X['library_size'], Y_causes_X['prediction_skill'], 
                       'o-', color=colors[color_idx % len(colors)],
                       label=f"{Y_name} causes {X_name}")
                color_idx += 1
    
    # Format plot
    ax.set_xlabel('Library Size (L)', fontsize=12)
    ax.set_ylabel('Prediction Skill (ρ)', fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if ccm_results:
        ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.8)
    
    # Title and styling
    plt.title(title, fontsize=14)
    
    # Add subtitle with additional info if provided
    subtitle = ""
    if metric_name:
        subtitle += f"Metric: {metric_name}"
    if round_name:
        subtitle += f" | Round: {round_name}"
    if phase_name:
        subtitle += f" | Phase: {phase_name}"
    
    if subtitle:
        plt.figtext(0.5, 0.01, subtitle, ha='center', fontsize=12)
    
    # Add explanation
    plt.figtext(0.02, 0.02, 
                "CCM tests causality by reconstructing attractor manifolds. If variable X causes Y,\n"
                "prediction skill increases with library size and converges. Stronger line indicates stronger causality.", 
                fontsize=9)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    
    return fig


def summarize_ccm_results(ccm_results, min_library_size=30, significance_threshold=0.3, threshold=None):
    """
    Create a summary matrix showing the causal relationships detected by CCM.
    
    Args:
        ccm_results: Dictionary with CCM results from calculate_pairwise_ccm
        min_library_size: Minimum library size to consider for convergence
        significance_threshold: Threshold for considering a relationship significant
        threshold: Alternative name for significance_threshold (for compatibility)
        
    Returns:
        DataFrame: Matrix of causal strengths
    """
    # If threshold is provided and significance_threshold is not, use threshold
    if threshold is not None and significance_threshold == 0.3:
        significance_threshold = threshold
    if not ccm_results:
        return pd.DataFrame()
    
    # Extract all variable names
    all_vars = set()
    for pair in ccm_results.keys():
        all_vars.add(pair[0])
        all_vars.add(pair[1])
    all_vars = sorted(list(all_vars))
    
    # Initialize result matrix
    n_vars = len(all_vars)
    causality_matrix = np.zeros((n_vars, n_vars))
    
    for pair, results_df in ccm_results.items():
        X_name, Y_name = pair
        X_idx = all_vars.index(X_name)
        Y_idx = all_vars.index(Y_name)
        
        # Get results for each direction
        X_causes_Y = results_df[results_df['direction'] == 'X causes Y']
        Y_causes_X = results_df[results_df['direction'] == 'Y causes X']
        
        # Check if X causes Y (look at prediction skill for large library sizes)
        if not X_causes_Y.empty:
            large_lib_results = X_causes_Y[X_causes_Y['library_size'] >= min_library_size]
            if not large_lib_results.empty:
                max_skill = large_lib_results['prediction_skill'].max()
                if max_skill > significance_threshold:
                    causality_matrix[X_idx, Y_idx] = max_skill
        
        # Check if Y causes X
        if not Y_causes_X.empty:
            large_lib_results = Y_causes_X[Y_causes_X['library_size'] >= min_library_size]
            if not large_lib_results.empty:
                max_skill = large_lib_results['prediction_skill'].max()
                if max_skill > significance_threshold:
                    causality_matrix[Y_idx, X_idx] = max_skill
    
    # Create DataFrame with variable names
    causality_df = pd.DataFrame(causality_matrix, index=all_vars, columns=all_vars)
    
    return causality_df


def plot_ccm_causality_heatmap(causality_matrix, title, output_dir, filename,
                             metric_name=None, round_name=None, phase_name=None,
                             figsize=(12, 10), threshold=None):
    """
    Create a heatmap visualization of the causality matrix from CCM.
    
    Args:
        causality_matrix: DataFrame with causality strengths
        title: Title of the plot
        output_dir: Directory to save the plot
        filename: Filename to save the plot
        metric_name: Name of the metric analyzed
        round_name: Name of the round analyzed
        phase_name: Name of the phase analyzed
        figsize: Figure size as (width, height) tuple
        threshold: Minimum causality strength to display (optional).
                   If None or 0, only exact zeros are masked.
                   If a value is given, values < threshold are also masked.
        
    Returns:
        Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with a sequential colormap
    if causality_matrix.empty:
        ax.text(0.5, 0.5, "No significant causal relationships found",
                ha='center', va='center', fontsize=14)
        plt.axis('off')
    else:
        # Mask cells with zero values or values below threshold
        mask = (causality_matrix == 0) 
        if threshold is not None and threshold > 0: # Apply threshold if it's a positive value
            mask = mask | (causality_matrix < threshold)
            
        heatmap = sns.heatmap(causality_matrix, cmap='viridis', annot=True, 
                            fmt='.2f', linewidths=0.5, ax=ax, mask=mask)
        
        # Title and styling
        plt.ylabel('Cause', fontsize=12)
        plt.xlabel('Effect', fontsize=12)
        
        # Add colorbar label
        if hasattr(heatmap, 'collections') and heatmap.collections: # Check if heatmap was actually drawn
            cbar = heatmap.collections[0].colorbar
            cbar.set_label('Causal Strength (Prediction Skill)', rotation=270, labelpad=20)
    
    # Title
    plt.title(title, fontsize=14)
    
    # Add subtitle with additional info if provided
    subtitle = ""
    if metric_name:
        subtitle += f"Metric: {metric_name}"
    if round_name:
        subtitle += f" | Round: {round_name}"
    if phase_name:
        subtitle += f" | Phase: {phase_name}"
    
    if subtitle:
        plt.figtext(0.5, 0.01, subtitle, ha='center', fontsize=12)
    
    # Add explanation
    explanation_base = "Rows are causes, columns are effects. Values represent prediction skill from CCM.\n"
    explanation_threshold = ""
    if threshold is not None and threshold > 0:
        explanation_threshold = f"Only relationships with strength >= {threshold:.2f} are shown.\n"
    explanation_suffix = "Empty cells indicate no significant causal relationship was detected or below threshold."
    
    plt.figtext(0.02, 0.02, 
                explanation_base + explanation_threshold + explanation_suffix, 
                fontsize=9)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    
    return fig


def perform_granger_causality_test(x, y, max_lag=5, criterion='aic'):
    """
    Perform Granger Causality test between two time series.
    
    Granger causality is based on the idea that if X causes Y, then past values of X
    should contain information that helps predict Y beyond the information contained
    in past values of Y alone.
    
    Args:
        x: First time series (potential cause)
        y: Second time series (potential effect)
        max_lag: Maximum lag to test (default: 5)
        criterion: Information criterion for model selection ('aic', 'bic', 'fpe')
        
    Returns:
        results: Dictionary with test results
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    
    # Convert to numpy arrays and ensure 1D
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    # Combine the series for the test
    data = np.column_stack([y, x])
    
    # Check for NaN values
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values")
    
    # Determine max lag to test
    actual_max_lag = min(max_lag, int(len(data) / 5))  # Rule of thumb: max 20% of data length
    if actual_max_lag < 1:
        actual_max_lag = 1
    
    # Perform the Granger causality test (x -> y)
    try:
        gc_results = grangercausalitytests(data, maxlag=actual_max_lag, verbose=False)
        
        # Extract p-values and F-statistics, handling potential errors for specific lags
        p_values = {}
        f_statistics = {}
        for lag, result_tuple in gc_results.items():
            # result_tuple is like ({'ssr_ftest': (F, p, df_denom, df_num), ...}, [params...])
            # We need to access the first element of the tuple, then the dictionary for the criterion
            lag_results = result_tuple[0]
            if criterion in lag_results:
                # lag_results[criterion] is (statistic, p_value, critical_value)
                # For F-test based criteria like 'ssr_ftest', 'lrtest', 'params_ftest'
                # For information criteria like 'aic', 'bic', 'hqic', 'fpe', it's (value, None, None) or similar
                # The p-value is the second element for test statistics.
                # For AIC/BIC, the p-value is not directly given, we rely on the F-test p-value.
                # Let's try to get the F-test p-value as the primary source.
                if 'ssr_ftest' in lag_results:  # Sum of squared residuals F-test
                    p_values[lag] = lag_results['ssr_ftest'][1]
                    f_statistics[lag] = lag_results['ssr_ftest'][0]
                elif criterion in lag_results and len(lag_results[criterion]) > 1 and lag_results[criterion][1] is not None:
                    # Fallback if ssr_ftest is not present but criterion gives a p-value
                    p_values[lag] = lag_results[criterion][1]
                    f_statistics[lag] = lag_results[criterion][0]
        
        if not p_values:  # If no p-values could be extracted
            print(f"Warning: Could not extract p-values for Granger test (x -> y) with criterion '{criterion}'. Available keys: {list(gc_results.keys()) if gc_results else 'None'}")
            return {'error': f"No p-values found for criterion '{criterion}'", 'causal_direction': 'x -> y'}
        
        # Find smallest p-value and corresponding lag
        min_p_value = min(p_values.values())
        optimal_lag = min([lag for lag, p in p_values.items() if p == min_p_value])
        
        # Get F-statistic for the optimal lag
        f_statistic_val = f_statistics.get(optimal_lag, np.nan)
        
        result = {
            'p_value': min_p_value,
            'optimal_lag': optimal_lag,
            'test_statistic': f_statistic_val, # Changed key name to 'test_statistic'
            'significant': min_p_value < 0.05,
            'causal_direction': 'x -> y',
            'p_values_by_lag': p_values
        }
    except KeyError as e:
        print(f"KeyError in Granger test (x -> y) processing results for criterion '{criterion}': {e}. Results structure: {gc_results if 'gc_results' in locals() else 'unavailable'}")
        result = {
            'error': f"KeyError: {e} for criterion '{criterion}'", 
            'causal_direction': 'x -> y', 
            'test_statistic': np.nan, # Ensure test_statistic is present
            'p_value': np.nan, 
            'optimal_lag': np.nan,
            'significant': False
        }
    except Exception as e:
        print(f"Error in Granger test (x -> y): {e}")
        result = {
            'error': str(e), 
            'causal_direction': 'x -> y', 
            'test_statistic': np.nan, # Ensure test_statistic is present
            'p_value': np.nan, 
            'optimal_lag': np.nan,
            'significant': False
        }
    
    return result


def calculate_pairwise_granger_causality(data_df, time_col=None, max_lag=5, criterion='aic'):
    """
    Calculate pairwise Granger causality between all variables in the dataset.
    
    Args:
        data_df: DataFrame with time series data
        time_col: Name of the time column, if None assumes data is already ordered
        max_lag: Maximum lag to test for Granger causality
        criterion: Information criterion for model selection
        
    Returns:
        DataFrame: Matrix of p-values for Granger causality
    """
    # Sort by time if time column is provided
    if time_col is not None and time_col in data_df.columns:
        data_df = data_df.sort_values(by=time_col)
    
    # Get numeric columns only
    numeric_cols = data_df.select_dtypes(include=np.number).columns
    
    # Exclude time column if specified
    if time_col is not None and time_col in numeric_cols:
        numeric_cols = numeric_cols.drop(time_col)
    
    # Initialize results matrix (p-values)
    n_vars = len(numeric_cols)
    p_value_matrix = np.ones((n_vars, n_vars))  # Default: p=1 (no causality)
    lag_matrix = np.zeros((n_vars, n_vars), dtype=int)
    f_stat_matrix = np.zeros((n_vars, n_vars))
    
    # Calculate pairwise Granger causality
    for i, cause_col in enumerate(numeric_cols):
        for j, effect_col in enumerate(numeric_cols):
            if i != j:  # Skip self-causality
                cause_series = data_df[cause_col]
                effect_series = data_df[effect_col]
                
                # Skip if too many NaNs
                if (np.isnan(cause_series).sum() > len(cause_series) * 0.2 or 
                    np.isnan(effect_series).sum() > len(effect_series) * 0.2):
                    continue
                
                # Fill remaining NaNs with interpolation
                cause_series = cause_series.interpolate().bfill().ffill()
                effect_series = effect_series.interpolate().bfill().ffill()
                
                try:
                    # Test cause_col -> effect_col
                    result = perform_granger_causality_test(
                        cause_series, effect_series, 
                        max_lag=max_lag, 
                        criterion=criterion
                    )
                    
                    if 'error' not in result:
                        p_value_matrix[i, j] = result['p_value']
                        lag_matrix[i, j] = result['optimal_lag']
                        f_stat_matrix[i, j] = result['test_statistic']
                except Exception as e:
                    print(f"Error in Granger test {cause_col} -> {effect_col}: {e}")
    
    # Create DataFrames with variable names
    p_value_df = pd.DataFrame(p_value_matrix, index=numeric_cols, columns=numeric_cols)
    lag_df = pd.DataFrame(lag_matrix, index=numeric_cols, columns=numeric_cols)
    f_stat_df = pd.DataFrame(f_stat_matrix, index=numeric_cols, columns=numeric_cols)
    
    return {
        'p_values': p_value_df,
        'optimal_lags': lag_df,
        'f_statistics': f_stat_df
    }


def plot_granger_causality_heatmap(result_dict, title, output_dir, filename,
                                 metric_name=None, round_name=None, phase_name=None,
                                 figsize=(12, 10), alpha=0.05):
    """
    Create a heatmap visualization of Granger causality test results.
    
    Args:
        result_dict: Dictionary with Granger causality test results
        title: Title of the plot
        output_dir: Directory to save the plot
        filename: Filename to save the plot
        metric_name: Name of the metric analyzed
        round_name: Name of the round analyzed
        phase_name: Name of the phase analyzed
        figsize: Figure size as (width, height) tuple
        alpha: Significance level for highlighting significant results
        
    Returns:
        Figure object
    """
    p_value_df = result_dict['p_values']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a mask for non-significant results
    mask = p_value_df > alpha
    
    # Use -log10(p) for better visualization (larger values = more significant)
    log_p_df = -np.log10(p_value_df)
    log_p_df = log_p_df.mask(mask)  # Mask non-significant results
    
    # Create heatmap
    heatmap = sns.heatmap(log_p_df, cmap='viridis', 
                        annot=p_value_df.applymap(lambda x: f"{x:.3f}" if x <= alpha else ""),
                        fmt="", linewidths=0.5, ax=ax)
    
    # Title and styling
    plt.title(title, fontsize=14)
    plt.ylabel('Cause', fontsize=12)
    plt.xlabel('Effect', fontsize=12)
    
    # Add subtitle with additional info if provided
    subtitle = ""
    if metric_name:
        subtitle += f"Metric: {metric_name}"
    if round_name:
        subtitle += f" | Round: {round_name}"
    if phase_name:
        subtitle += f" | Phase: {phase_name}"
    
    if subtitle:
        plt.figtext(0.5, 0.01, subtitle, ha='center', fontsize=12)
    
    # Add colorbar label
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('-log10(p-value)', rotation=270, labelpad=20)
    
    # Add significance threshold line on colorbar
    if alpha < 1:
        log_alpha = -np.log10(alpha)
        if log_alpha <= log_p_df.max().max():
            cbar.ax.axhline(y=cbar.ax.get_position().y0 + 
                          log_alpha/log_p_df.max().max() * cbar.ax.get_position().height,
                          color='red', linestyle='--', linewidth=1)
            cbar.ax.text(cbar.ax.get_position().x1 + 0.1, 
                       cbar.ax.get_position().y0 + 
                       log_alpha/log_p_df.max().max() * cbar.ax.get_position().height,
                       f"p={alpha}", va='center')
    
    # Add explanation
    plt.figtext(0.02, 0.02, 
                f"Granger causality tests if past values of one variable help predict another.\n"
                f"Rows: potential causes, Columns: potential effects. Only p-values < {alpha} are shown.\n"
                f"Brighter colors indicate stronger evidence for causality.", 
                fontsize=9)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    
    return fig


def plot_granger_causality_network(result_dict, title, output_dir, filename,
                                metric_name=None, round_name=None, phase_name=None,
                                figsize=(14, 12), alpha=0.05, f_stat_threshold=None):
    """
    Create a network visualization of significant Granger causality relationships.
    
    Args:
        result_dict: Dictionary with Granger causality test results
        title: Title of the plot
        output_dir: Directory to save the plot
        filename: Filename to save the plot
        metric_name: Name of the metric analyzed
        round_name: Name of the round analyzed
        phase_name: Name of the phase analyzed
        figsize: Figure size as (width, height) tuple
        alpha: Significance level for including edges
        f_stat_threshold: Minimum F-statistic value for edge inclusion
        
    Returns:
        Figure object
    """
    p_value_df = result_dict['p_values']
    f_stat_df = result_dict['f_statistics']
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for var in p_value_df.columns:
        G.add_node(var)
    
    # Add edges for significant causal relationships
    for cause in p_value_df.index:
        for effect in p_value_df.columns:
            p_value = p_value_df.loc[cause, effect]
            f_stat = f_stat_df.loc[cause, effect]
            
            if (cause != effect and 
                p_value <= alpha and 
                (f_stat_threshold is None or f_stat >= f_stat_threshold)):
                G.add_edge(cause, effect, 
                          weight=1/p_value,  # Use inverse p-value as weight
                          p_value=p_value,
                          f_stat=f_stat)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if we have any edges
    if len(G.edges()) == 0:
        ax.text(0.5, 0.5, f"No significant Granger causality found (p <= {alpha})",
                ha='center', va='center', fontsize=14)
        plt.axis('off')
    else:
        # Layout - use spring_layout for better visualization of directed graphs
        pos = nx.spring_layout(G, seed=42, k=1.2/np.sqrt(len(G.nodes())))
        
        # Get edge weights for width and color
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        weights_normalized = np.array(weights) / max(weights)
        
        # Use a color map based on weights
        edge_colors = plt.cm.viridis(weights_normalized)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=3000, 
                            node_color='lightblue', 
                            alpha=0.8, 
                            ax=ax)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)
        
        # Draw edges with width proportional to statistical significance
        nx.draw_networkx_edges(G, pos, width=weights_normalized*5, 
                             edge_color=edge_colors,
                             connectionstyle='arc3,rad=0.15',  # Add some curvature
                             arrowsize=20, ax=ax)
        
        # Edge labels showing p-values
        edge_labels = {(u, v): f"p={G[u][v]['p_value']:.3f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                   font_size=9, ax=ax)
        
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                norm=plt.Normalize(0, max(weights_normalized)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Causal Strength (1/p-value)', rotation=270, labelpad=20)
    
    # Title and styling
    plt.title(title, fontsize=14)
    
    # Add subtitle with additional info if provided
    subtitle = ""
    if metric_name:
        subtitle += f"Metric: {metric_name}"
    if round_name:
        subtitle += f" | Round: {round_name}"
    if phase_name:
        subtitle += f" | Phase: {phase_name}"
    
    if subtitle:
        plt.figtext(0.5, 0.01, subtitle, ha='center', fontsize=12)
    
    # Information text
    plt.figtext(0.05, 0.01, 
                f"Shows Granger causal relationships with p-value < {alpha}.\n"
                "Direction: Source (cause) → Target (effect). Stronger causal effects have thicker arrows.", 
                fontsize=10)
    
    # Remove axis
    plt.axis('off')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    
    return fig