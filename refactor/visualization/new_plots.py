import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from refactor.new_config import VISUALIZATION_CONFIG, METRIC_DISPLAY_NAMES, TENANT_COLORS, PHASE_DISPLAY_NAMES
from refactor.data_handling.save_results import save_figure

def set_publication_style():
    """Sets the style for publication-quality visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (VISUALIZATION_CONFIG.get('figure_width', 10), 
                                      VISUALIZATION_CONFIG.get('figure_height', 6))
    plt.rcParams['figure.dpi'] = VISUALIZATION_CONFIG.get('dpi', 300)
    plt.rcParams['font.size'] = VISUALIZATION_CONFIG.get('font_size', 12)
    plt.rcParams['axes.labelsize'] = VISUALIZATION_CONFIG.get('label_size', 14)
    plt.rcParams['axes.titlesize'] = VISUALIZATION_CONFIG.get('title_size', 16)
    plt.rcParams['xtick.labelsize'] = VISUALIZATION_CONFIG.get('tick_size', 12)
    plt.rcParams['ytick.labelsize'] = VISUALIZATION_CONFIG.get('tick_size', 12)
    plt.rcParams['legend.fontsize'] = VISUALIZATION_CONFIG.get('legend_size', 12)
    plt.rcParams['legend.title_fontsize'] = VISUALIZATION_CONFIG.get('legend_title_size', 14)

def plot_correlation_heatmap(correlation_matrix, title='Correlation Heatmap', cmap='vlag', cbar_label='Correlation Coefficient', output_dir=None, filename=None):
    """
    Plots a correlation matrix as a heatmap.

    Args:
        correlation_matrix (pd.DataFrame): The correlation matrix to plot.
        title (str): The title of the heatmap.
        cmap (str): The colormap to use.
        cbar_label (str): Label for the color bar.
        output_dir (str, optional): Directory to save the plot. If None, plot is not saved.
        filename (str, optional): Filename for the saved plot. If None, a default is generated.
    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('heatmap_figsize', (10, 8)))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap=cmap, center=0, linewidths=.5, 
                cbar_kws={'label': cbar_label}, ax=ax, vmin=-1, vmax=1)
    ax.set_title(title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        full_path = os.path.join(output_dir, filename)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"Heatmap saved to {full_path}")
        except Exception as e:
            print(f"Error saving heatmap to {full_path}: {e}")
    return fig

def plot_covariance_heatmap(covariance_matrix, title='Covariance Heatmap', cmap='coolwarm', cbar_label='Covariance', output_dir=None, filename=None):
    """
    Plots a covariance matrix as a heatmap.

    Args:
        covariance_matrix (pd.DataFrame): The covariance matrix to plot.
        title (str): The title of the heatmap.
        cmap (str): The colormap to use.
        cbar_label (str): Label for the color bar.
        output_dir (str, optional): Directory to save the plot. If None, plot is not saved.
        filename (str, optional): Filename for the saved plot. If None, a default is generated.
    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('heatmap_figsize', (10, 8)))
    sns.heatmap(covariance_matrix, annot=True, fmt=".2f", cmap=cmap, center=0, linewidths=.5,
                cbar_kws={'label': cbar_label}, ax=ax)
    ax.set_title(title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"Covariance heatmap saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving covariance heatmap to {os.path.join(output_dir, filename)}: {e}")
    return fig

def plot_cross_correlation(cross_corr_series: pd.Series, title: str = 'Cross-Correlation Plot', output_dir: str = None, filename: str = None):
    """
    Plots the results of a cross-correlation analysis.

    Args:
        cross_corr_series (pd.Series): Series containing cross-correlation values, with lags as index.
        title (str): Title for the plot.
        output_dir (str, optional): Directory to save the plot.
        filename (str, optional): Filename for the saved plot.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('cross_corr_figsize', (10, 6)))
    
    cross_corr_series.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_xlabel("Lag")
    ax.set_ylabel("Cross-Correlation Coefficient")
    ax.set_title(title)
    ax.axhline(0, color='black', linewidth=0.5)
    plt.tight_layout()

    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        full_path = os.path.join(output_dir, filename)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"Cross-correlation plot saved to {full_path}")
        except Exception as e:
            print(f"Error saving cross-correlation plot to {full_path}: {e}")
            
    return fig

def plot_metric_with_anomalies(df, metric_name, time_column='experiment_elapsed_seconds', value_column='value', 
                               anomaly_column='is_anomaly', change_points=None,
                               phases=None, tenants=None, show_phase_markers=True, 
                               output_dir=None, filename=None, 
                               figsize=None, use_total_duration=False, total_duration_seconds=None,
                               show_as_percentage=False, node_config=None):
    """
    Creates a line plot for a metric, highlighting anomalies and change points.
    Adapted for the refactored structure.
    
    Args:
        df (pd.DataFrame): DataFrame with metric data and anomaly column.
        metric_name (str): Name of the metric (key for METRIC_DISPLAY_NAMES).
        time_column (str): Column with time values for the X-axis. Default: 'experiment_elapsed_seconds'.
        value_column (str): Column with metric values.
        anomaly_column (str): Boolean column indicating if a point is an anomaly.
        change_points (list): List of timestamps/indices where significant changes occur.
        phases (list): List of phases to include (None = all).
        tenants (list): List of tenants to include (None = all).
        show_phase_markers (bool): Whether to show vertical lines marking phases.
        output_dir (str, optional): Directory to save the plot. If None, plot is not saved.
        filename (str, optional): Filename for the saved plot. If None, a default is generated.
        figsize (tuple): Figure size (optional, uses config if None).
        use_total_duration (bool): If True, the X-axis will use 'total_elapsed_seconds'.
        total_duration_seconds (float): Total duration of the experiment in seconds.
        show_as_percentage (bool): If True, display values as percentages of total capacity.
        node_config (dict): Configuration with node resource capacities.
        
    Returns:
        matplotlib.figure.Figure: Matplotlib figure object.
    """
    set_publication_style()
    
    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    
    data = df.copy()
    if phases:
        data = data[data['phase'].isin(phases)]
    if tenants:
        data = data[data['tenant'].isin(tenants)]
    
    if figsize is None:
        figsize = (VISUALIZATION_CONFIG.get('figure_width', 12), VISUALIZATION_CONFIG.get('figure_height', 8))
    fig, ax = plt.subplots(figsize=figsize)
    
    anomaly_label_added = False
    for tenant, group in data.groupby('tenant'):
        group = group.sort_values(time_column)
        color = TENANT_COLORS.get(tenant, None)
        
        ax.plot(group[time_column], group[value_column], 
                label=tenant, linewidth=1.5, color=color, alpha=0.8)
        
        anomalies = group[group[anomaly_column] == True]
        if not anomalies.empty:
            label_anom = 'Anomalies' if not anomaly_label_added else None
            ax.scatter(anomalies[time_column], anomalies[value_column], 
                       color='red', s=30, label=label_anom, 
                       marker='x', zorder=5, alpha=0.7)
            anomaly_label_added = True

    if show_phase_markers and 'phase' in data.columns:
        phase_starts = {}
        for phase_val, group_data in data.groupby('phase'):
            min_time = group_data[time_column].min()
            phase_display_name = PHASE_DISPLAY_NAMES.get(phase_val, phase_val)
            if min_time not in phase_starts:
                phase_starts[min_time] = []
            if phase_display_name not in phase_starts[min_time]:
                 phase_starts[min_time].append(phase_display_name)
        
        for time, phase_list_names in phase_starts.items():
            if data[time_column].min() < time < data[time_column].max():
                ax.axvline(x=time, color='gray', linestyle='--', alpha=0.7)
                ax.text(time + (data[time_column].max() * 0.01) , ax.get_ylim()[1] * 0.95, ', '.join(phase_list_names), 
                        rotation=90, verticalalignment='top', alpha=0.7, fontsize=VISUALIZATION_CONFIG.get('annotation_fontsize', 10))

    if use_total_duration:
        ax.set_xlabel('Total Experiment Time (seconds)')
        if total_duration_seconds:
            ax.set_xlim(0, total_duration_seconds)
    else:
        time_unit = "seconds" if "seconds" in time_column else time_column.split('_')[-1]
        ax.set_xlabel(f'Elapsed time ({time_unit})')
    
    if show_as_percentage:
        if node_config:
            if metric_name == 'cpu_usage' and 'CPUS' in node_config:
                unit_info = f"% of {node_config['CPUS']} CPU cores"
            elif metric_name == 'memory_usage' and 'MEMORY_GB' in node_config:
                unit_info = f"% of {node_config['MEMORY_GB']} GB memory"
            elif metric_name == 'disk_throughput_total':
                unit_info = "% of 500 MB/s theoretical throughput"
            elif metric_name == 'network_total_bandwidth':
                unit_info = "% of 1 Gbps network interface"
            else:
                unit_info = "%"
            ax.set_ylabel(f"{display_metric_name} ({unit_info})")
        else:
            ax.set_ylabel(f"{display_metric_name} (%)")
    else:
        ax.set_ylabel(display_metric_name)
    
    ax.set_title(f'{display_metric_name} with Anomalies by Tenant')
    
    handles, labels = ax.get_legend_handles_labels()
    filtered_handles = []
    filtered_labels = []
    for handle, label in zip(handles, labels):
        if label is not None:
            filtered_handles.append(handle)
            filtered_labels.append(label)
    
    by_label = dict(zip(filtered_labels, filtered_handles)) 
    ax.legend(by_label.values(), by_label.keys(), title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=VISUALIZATION_CONFIG.get('x_axis_nbins', 10)))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"Plot saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving plot to {os.path.join(output_dir, filename)}: {e}")
    
    return fig

def plot_entropy_heatmap(entropy_results_df, metric_name, round_name, output_dir=None, filename=None):
    """
    Generates a heatmap of entropy (mutual information) results between tenant pairs.
    Adapted for the refactored structure.

    Args:
        entropy_results_df (pd.DataFrame): DataFrame with entropy results. 
                                           Expected columns: 'tenant1', 'tenant2', 'mutual_information'.
        metric_name (str): The name of the metric for which entropy was calculated.
        round_name (str): The name of the round/experiment.
        output_dir (str, optional): Directory to save the plot. If None, plot is not saved.
        filename (str, optional): Filename for the saved plot. If None, a default is generated.
    """
    set_publication_style()
    
    if entropy_results_df.empty:
        print(f"No entropy data to plot for heatmap (Metric: {metric_name}, Round: {round_name}).")
        return None # Return None if no plot generated

    heatmap_data = entropy_results_df.pivot(index='tenant1', columns='tenant2', values='mutual_information')
    
    all_tenants = sorted(list(set(entropy_results_df['tenant1']).union(set(entropy_results_df['tenant2']))))
    heatmap_data = heatmap_data.reindex(index=all_tenants, columns=all_tenants)
    
    for i in range(len(all_tenants)):
        for j in range(i + 1, len(all_tenants)):
            t1, t2 = all_tenants[i], all_tenants[j]
            val_t1_t2 = heatmap_data.loc[t1, t2] if t1 in heatmap_data.index and t2 in heatmap_data.columns and pd.notna(heatmap_data.loc[t1, t2]) else np.nan
            val_t2_t1 = heatmap_data.loc[t2, t1] if t2 in heatmap_data.index and t1 in heatmap_data.columns and pd.notna(heatmap_data.loc[t2, t1]) else np.nan

            if pd.isna(val_t1_t2) and not pd.isna(val_t2_t1):
                heatmap_data.loc[t1, t2] = val_t2_t1
            elif not pd.isna(val_t1_t2) and pd.isna(val_t2_t1):
                heatmap_data.loc[t2, t1] = val_t1_t2
                
    heatmap_data = heatmap_data.fillna(0)

    fig, ax = plt.subplots(figsize=(VISUALIZATION_CONFIG.get('figure_width', 10), 
                               VISUALIZATION_CONFIG.get('figure_height', 8)))
    
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="vlag", center=0, linewidths=.5,
                cbar_kws={'label': 'Mutual Information'}, ax=ax)
    
    ax.set_title(f'Mutual Information between Tenant Pairs\nMetric: {metric_name} - Round: {round_name}')
    ax.set_xlabel('Tenant 2')
    ax.set_ylabel('Tenant 1')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"Entropy heatmap saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving entropy heatmap to {os.path.join(output_dir, filename)}: {e}")
    
    return fig

def plot_entropy_top_pairs_barplot(entropy_results_df, metric_name, round_name, output_dir=None, filename=None, top_n=10):
    """
    Generates a bar plot of the top N tenant pairs with the highest mutual information.
    Adapted for the refactored structure.

    Args:
        entropy_results_df (pd.DataFrame): DataFrame with entropy results.
                                           Expected columns: 'tenant1', 'tenant2', 'mutual_information'.
        metric_name (str): The name of the metric for which entropy was calculated.
        round_name (str): The name of the round/experiment.
        output_dir (str, optional): Directory to save the plot. If None, plot is not saved.
        filename (str, optional): Filename for the saved plot. If None, a default is generated.
        top_n (int): Number of top pairs to display.
    """
    set_publication_style()
    
    if entropy_results_df.empty:
        print(f"No entropy data to plot for top pairs barplot (Metric: {metric_name}, Round: {round_name}).")
        return None # Return None if no plot generated

    entropy_results_df['pair'] = entropy_results_df.apply(
        lambda row: tuple(sorted((row['tenant1'], row['tenant2']))), axis=1
    )
    top_pairs = entropy_results_df.sort_values('mutual_information', ascending=False)\
                                  .drop_duplicates(subset=['pair'])\
                                  .head(top_n)
    
    if top_pairs.empty:
        print(f"No entropy data to plot for top pairs barplot (Metric: {metric_name}, Round: {round_name}).")
        return None # Return None if no plot generated

    fig_bar, ax = plt.subplots(figsize=(VISUALIZATION_CONFIG.get('figure_width', 12), 
                                     VISUALIZATION_CONFIG.get('figure_height', 7)))
    
    pair_labels = [f"{p[0]} - {p[1]}" for p in top_pairs['pair']]
    
    barplot = sns.barplot(x='mutual_information', y=pair_labels, data=top_pairs, palette='mako', orient='h', ax=ax)
    
    ax.set_xlabel('Mutual Information')
    ax.set_ylabel('Tenant Pair')
    ax.set_title(f'Top {top_n} Tenant Pairs by Mutual Information\nMetric: {metric_name} - Round: {round_name}')
    
    for index, value in enumerate(top_pairs['mutual_information']):
        barplot.text(value, index, f'{value:.3f}', color='black', ha="left", va="center", fontsize=10)
        
    plt.tight_layout()
    
    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig_bar, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"Entropy top pairs barplot saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving entropy top pairs barplot to {os.path.join(output_dir, filename)}: {e}")
    
    return fig_bar
