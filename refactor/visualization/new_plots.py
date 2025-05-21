import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle # Ensure Rectangle is imported
from matplotlib.lines import Line2D # Import Line2D
from typing import Sequence # Import Sequence for type hinting

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

def plot_correlation_heatmap(correlation_matrix, title='Correlation Heatmap', cmap='vlag', cbar_label='Correlation Coefficient', output_dir: str | None = None, filename: str | None = None):
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

def plot_covariance_heatmap(covariance_matrix, title='Covariance Heatmap', cmap='coolwarm', cbar_label='Covariance', output_dir: str | None = None, filename: str | None = None):
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

def plot_cross_correlation(cross_corr_series: pd.Series, title: str = 'Cross-Correlation Plot', output_dir: str | None = None, filename: str | None = None):
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
                               output_dir: str | None = None, filename: str | None = None, 
                               figsize: tuple[float, float] | None = None, use_total_duration=False, total_duration_seconds=None,
                               show_as_percentage=False, node_config=None):
    """
    Creates a line plot for a metric, highlighting anomalies and change points.
    Adapted for the refactored structure
    
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
        output_dir (str | None, optional): Directory to save the plot. If None, plot is not saved.
        filename (str | None, optional): Filename for the saved plot. If None, a default is generated.
        figsize (tuple): Figure size (optional, uses config if None).
        use_total_duration (bool): If True, the X-axis will use 'total_elapsed_seconds'.
        total_duration_seconds (float): Total duration of the experiment in seconds.
        show_as_percentage (bool): If True, display values as percentages of total capacity.
        node_config (dict): Configuration with node resource capacities.
        
    Returns:
        matplotlib.figure.Figure: Matplotlib figure object.
    """
    set_publication_style()
    
    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, str(metric_name)) # Ensure string
    
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
    ax.legend(by_label.values(), by_label.keys(), title='Legend', loc='best')
    
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

def plot_entropy_heatmap(entropy_results_df, metric_name, round_name: str | None = None, output_dir: str | None = None, filename: str | None = None):
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

def plot_entropy_top_pairs_barplot(entropy_results_df, metric_name, round_name: str | None = None, output_dir: str | None = None, filename: str | None = None, top_n=10):
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

def plot_scatter_comparison(
    x_values: list | pd.Series,
    y_values: list | pd.Series,
    x_label: str, 
    y_label: str, 
    title: str, 
    output_dir: str, 
    filename: str,
    point_labels: list | pd.Series | None = None, # Moved to be after required arguments
    add_identity_line: bool = True
):
    """
    Generates a scatter plot comparing two sets of values.

    Args:
        x_values (list | pd.Series): Data for the x-axis.
        y_values (list | pd.Series): Data for the y-axis.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
        output_dir (str): Directory to save the plot.
        filename (str): Filename for the saved plot.
        point_labels (list | pd.Series, optional): Labels for each point, used for annotation.
                                                   If provided, points will be annotated. Defaults to None.
        add_identity_line (bool): If True, adds a y=x identity line. Defaults to True.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('scatter_figsize', (8, 8)))

    x_data = pd.Series(x_values) if not isinstance(x_values, pd.Series) else x_values.copy()
    y_data = pd.Series(y_values) if not isinstance(y_values, pd.Series) else y_values.copy()

    if len(x_data) != len(y_data):
        print(f"Warning: x_values (len {len(x_data)}) and y_values (len {len(y_data)}) have different lengths for scatter plot '{title}'. Skipping.")
        plt.close(fig)
        return None
    
    if x_data.empty or y_data.empty:
        print(f"Warning: No data to plot for scatter plot '{title}'. Skipping.")
        plt.close(fig)
        return None

    # Reset index to ensure alignment if they are series with different indices but same length
    x_data.index = pd.RangeIndex(len(x_data))
    y_data.index = pd.RangeIndex(len(y_data))

    sns.scatterplot(x=x_data, y=y_data, ax=ax, 
                    s=VISUALIZATION_CONFIG.get('scatter_marker_size', 50), 
                    alpha=VISUALIZATION_CONFIG.get('scatter_marker_alpha', 0.7),
                    edgecolor=VISUALIZATION_CONFIG.get('scatter_marker_edgecolor', 'k'),
                    linewidth=VISUALIZATION_CONFIG.get('scatter_marker_linewidth', 0.5))
    
    all_x_values_list = x_data.tolist() # Renamed to avoid conflict
    all_y_values_list = y_data.tolist() # Renamed to avoid conflict

    ax.set_xlabel(x_label, fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    ax.set_ylabel(y_label, fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    ax.set_title(title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))

    if add_identity_line:
        all_plot_values = pd.Series(all_x_values_list + all_y_values_list).dropna()
        if not all_plot_values.empty:
            min_val = all_plot_values.min()
            max_val = all_plot_values.max()
            buffer = (max_val - min_val) * 0.05 if (max_val - min_val) > 1e-9 else 0.1 
            lims = [min_val - buffer, max_val + buffer]
            
            ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='y=x (Identity)')
            ax.set_xlim(lims[0], lims[1]) # Pass elements separately
            ax.set_ylim(lims[0], lims[1]) # Pass elements separately
            ax.legend(fontsize=VISUALIZATION_CONFIG.get('legend_size', 12)) 
        else:
            print("Warning: Cannot draw identity line as no valid data points found.")

    if point_labels is not None:
        labels_series = pd.Series(point_labels) if not isinstance(point_labels, pd.Series) else point_labels
        if len(labels_series) == len(x_data):
            labels_series.index = pd.RangeIndex(len(labels_series)) 
            annot_count = 0
            max_annots = VISUALIZATION_CONFIG.get('scatter_max_annotations', 20)
            
            for idx in x_data.index: 
                if annot_count >= max_annots:
                    print(f"Warning: Reached max annotations ({max_annots}) for scatter plot. Not all points annotated.")
                    break
                # Ensure idx is valid for all series before attempting to access
                if idx < len(x_data) and idx < len(y_data) and idx < len(labels_series):
                    ax.text(x_data.iloc[idx], y_data.iloc[idx], # Use .iloc for position-based access
                            str(labels_series.iloc[idx]), fontsize=VISUALIZATION_CONFIG.get('annotation_fontsize_small', 8),
                            ha='left', va='bottom')
                    annot_count +=1
        else:
            print(f"Warning: Length of point_labels (len {len(labels_series)}) does not match data length (len {len(x_data)}). Skipping annotations.")
    
    plt.tight_layout()

    if not all_x_values_list or not all_y_values_list: 
        print(f"Warning: No data was plotted for scatter plot '{title}'. Skipping save.")
        plt.close(fig)
        return None

    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
        except Exception as e:
            print(f"Error saving scatter plot to {os.path.join(output_dir, filename)}: {e}")
    
    return fig

# --- New plotting functions for Descriptive Statistics ---

def plot_descriptive_stats_lineplot(
    data_df: pd.DataFrame, 
    metric_name: str, 
    value_column: str, 
    title: str, 
    output_dir: str | None = None, 
    filename: str | None = None,
    round_name: str | None = None,
    phase_name: str | None = None
):
    """
    Generates a line plot of a metric over time for each tenant.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('lineplot_figsize', (12, 7)))

    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    plot_title = f"{title}"
    if phase_name:
        plot_title += f" - Phase: {PHASE_DISPLAY_NAMES.get(phase_name, phase_name)}"
    if round_name:
        plot_title += f" (Round: {round_name})"

    ax.set_title(plot_title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))

    time_col = 'experiment_elapsed_seconds' if 'experiment_elapsed_seconds' in data_df.columns else data_df.index.name

    if 'tenant' in data_df.columns and data_df['tenant'].nunique() > 0:
        unique_tenants = sorted(data_df['tenant'].unique())
        for tenant in unique_tenants:
            tenant_df = data_df[data_df['tenant'] == tenant]
            if not tenant_df.empty:
                sns.lineplot(x=time_col, y=value_column, data=tenant_df, ax=ax, 
                             label=str(tenant), color=TENANT_COLORS.get(str(tenant), '#333333'),
                             marker=VISUALIZATION_CONFIG.get('lineplot_marker', 'o'), 
                             linestyle=VISUALIZATION_CONFIG.get('lineplot_linestyle', '-'))
        # Position legend automatically for best location
        ax.legend(title='Tenant', loc='best', 
                  fontsize=VISUALIZATION_CONFIG.get('legend_size', 10)) 
        legend = ax.get_legend()
        if legend: # Check if legend exists
            plt.setp(legend.get_title(), fontsize=VISUALIZATION_CONFIG.get('legend_title_size', 12))
    else:
        # Plot without tenant breakdown
        sns.lineplot(x=time_col, y=value_column, data=data_df, ax=ax,
                     marker=VISUALIZATION_CONFIG.get('lineplot_marker', 'o'), 
                     linestyle=VISUALIZATION_CONFIG.get('lineplot_linestyle', '-'))

    ax.set_xlabel("Time (seconds)", fontsize=VISUALIZATION_CONFIG.get('label_size', 12))
    ax.set_ylabel(f"{display_metric_name} ({value_column.capitalize()})", fontsize=VISUALIZATION_CONFIG.get('label_size', 12))
    ax.tick_params(axis='both', which='major', labelsize=VISUALIZATION_CONFIG.get('tick_size', 10))
    ax.grid(bool(VISUALIZATION_CONFIG.get('grid_enabled', True)), # Cast to bool
            linestyle=VISUALIZATION_CONFIG.get('grid_linestyle', '--'), 
            alpha=VISUALIZATION_CONFIG.get('grid_alpha', 0.7))

    plt.tight_layout()

    if output_dir and filename:
        save_figure(fig, output_dir, filename)
        print(f"Descriptive stats lineplot saved to {os.path.join(output_dir, filename)}")
    plt.close(fig)
    return fig

def plot_descriptive_stats_boxplot(
    data_df: pd.DataFrame, 
    metric_name: str, 
    value_column: str, 
    title: str, 
    output_dir: str | None = None, 
    filename: str | None = None,
    round_name: str | None = None,
    phase_name: str | None = None
):
    """
    Generates a box plot of a metric for each tenant.
    Handles FutureWarning from Seaborn regarding palette and hue.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('boxplot_figsize', (10, 7)))

    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, str(metric_name))
    plot_title = f"{title}"
    if phase_name:
        plot_title += f" - Phase: {PHASE_DISPLAY_NAMES.get(phase_name, str(phase_name))}"
    if round_name:
        plot_title += f" (Round: {round_name})"
    
    ax.set_title(plot_title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    unique_tenants = [] # Initialize unique_tenants

    if 'tenant' in data_df.columns and data_df['tenant'].nunique() > 0:
        unique_tenants = sorted(data_df['tenant'].unique())
        palette_list = [TENANT_COLORS.get(str(tenant), '#333333') for tenant in unique_tenants]
        sns.boxplot(x='tenant', y=value_column, hue='tenant', data=data_df, ax=ax, 
                    palette=palette_list, hue_order=unique_tenants, legend=False) 
        ax.set_xlabel("Tenant")
    else:
        default_palette_str = VISUALIZATION_CONFIG.get('boxplot_palette', 'Set2')
        # Ensure palette is a string or a compatible type for seaborn
        current_palette = str(default_palette_str) if not isinstance(default_palette_str, (list, dict)) else default_palette_str
        if 'tenant' in data_df.columns and data_df['tenant'].nunique() == 1:
            sns.boxplot(x='tenant', y=value_column, data=data_df, ax=ax, palette=current_palette)
            ax.set_xlabel("Tenant")
        else: 
            sns.boxplot(y=value_column, data=data_df, ax=ax, palette=current_palette)
            ax.set_xlabel("") 

    ax.set_ylabel(f"{display_metric_name} ({value_column.capitalize()})")
    ax.tick_params(axis='x', rotation=45)

    if unique_tenants: # Check if unique_tenants is populated
        handles = [Rectangle((0,0),1,1, color=TENANT_COLORS.get(str(t), '#333333')) for t in unique_tenants]
        labels = [str(t) for t in unique_tenants]
        
        existing_legend = ax.get_legend()
        if existing_legend:
            existing_legend.remove()

        new_legend = ax.legend(handles, labels, title='Tenant', loc='best', 
                               fontsize=VISUALIZATION_CONFIG.get('legend_size', 10))
        
        if new_legend:
            plt.setp(new_legend.get_title(), fontsize=VISUALIZATION_CONFIG.get('legend_title_size', 12))

    plt.tight_layout()

    if output_dir and filename:
        save_figure(fig, output_dir, filename)
        print(f"Descriptive stats boxplot saved to {os.path.join(output_dir, filename)}")
    plt.close(fig)
    return fig

def plot_descriptive_stats_catplot_mean(
    stats_df: pd.DataFrame, 
    metric_name: str, 
    value_column: str, # This is 'mean'
    title: str, 
    output_dir: str | None = None, 
    filename: str | None = None,
    round_name: str | None = None,
    phase_name: str | None = None,
    x_var: str = 'tenant', 
    hue_var: str = 'tenant'
):
    set_publication_style()
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('catplot_figsize', (10, 6)))

    plot_data = stats_df.copy()
    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    
    plot_title_parts = [title]
    if phase_name:
        # Ensure phase_name is a string or number for PHASE_DISPLAY_NAMES
        phase_display = PHASE_DISPLAY_NAMES.get(str(phase_name), str(phase_name))
        plot_title_parts.append(f"Phase: {phase_display}")
    if round_name:
        plot_title_parts.append(f"(Round: {round_name})")
    
    # Join title parts, ensuring "Descriptive Statistics for..." is not duplicated if already in title
    if not title.startswith("Descriptive Statistics for"):
        final_title = f"Descriptive Statistics for {display_metric_name}: Mean Values"
        if plot_title_parts[0] != title : # If title was more specific initially
             final_title = title 
        plot_title_parts = [final_title] + plot_title_parts[1:]


    ax.set_title(" - ".join(plot_title_parts), fontsize=VISUALIZATION_CONFIG.get('title_size', 16))

    unique_hues = sorted(plot_data[hue_var].unique())
    num_hues = len(unique_hues)
    
    palette_to_use = None
    use_tenant_colors_flag = False
    if hue_var == 'tenant':
        all_tenants_in_config = True
        for tenant_name in unique_hues:
            if tenant_name not in TENANT_COLORS:
                all_tenants_in_config = False
                break
        if all_tenants_in_config:
            palette_to_use = [TENANT_COLORS.get(t, '#000000') for t in unique_hues] # Fallback to black if somehow missing
            use_tenant_colors_flag = True

    if not use_tenant_colors_flag:
        if num_hues <= 10:
            palette_to_use = sns.color_palette(n_colors=num_hues)
        elif num_hues <= 20:
            palette_to_use = sns.color_palette("tab20", n_colors=num_hues)
        else:
            palette_to_use = sns.color_palette("husl", n_colors=num_hues)

    # Ensure plot_data is sorted by x_var to match `order` and `hue_order`
    # This helps in aligning manual error bars if get_xticklabels() is not robust enough initially.
    # However, using order and hue_order in barplot and then categories_on_plot from xticklabels is preferred.
    # For simplicity and robustness, ensure unique_hues (which is sorted) is used for order.
    
    barplot = sns.barplot(x=x_var, y=value_column, hue=hue_var, data=plot_data, ax=ax,
                          palette=palette_to_use, errorbar=None, legend=False, 
                          order=unique_hues, hue_order=unique_hues)

    # Manually add error bars using the 'std' column from stats_df
    categories_on_plot = [tick.get_text() for tick in ax.get_xticklabels()]
    
    if not categories_on_plot and unique_hues: # Fallback if xticklabels are not yet populated
        categories_on_plot = unique_hues

    if categories_on_plot:
        plot_data_indexed = plot_data.set_index(x_var)
        
        # Ensure all categories_on_plot exist in plot_data_indexed
        valid_categories = [cat for cat in categories_on_plot if cat in plot_data_indexed.index]
        
        if valid_categories:
            means_for_errorbar = plot_data_indexed.loc[valid_categories, value_column].values
            # Assuming the standard deviation column is named 'std'
            stds_for_errorbar = plot_data_indexed.loc[valid_categories, 'std'].values 
            
            # X positions for error bars: 0, 1, 2, ... corresponding to valid_categories
            x_positions = np.arange(len(valid_categories))

            ax.errorbar(x=x_positions, y=means_for_errorbar, yerr=stds_for_errorbar,
                        fmt='none', color='black', 
                        capsize=3, elinewidth=1, capthick=1,
                        label='_nolegend_') # Use _nolegend_ to hide from automatic legend

    ax.set_xlabel(METRIC_DISPLAY_NAMES.get(x_var, x_var))
    ax.set_ylabel(f"Mean {display_metric_name} (with Std Dev)")

    if hue_var and plot_data[hue_var].nunique() > 0 : # Show legend if there are hues
        # Create legend for hues
        if palette_to_use: # Ensure palette_to_use is not None
            legend_handles = [Rectangle((0,0),1,1, color=palette_to_use[i]) for i in range(len(unique_hues))]
            ax.legend(legend_handles, unique_hues, title=METRIC_DISPLAY_NAMES.get(hue_var, hue_var), loc='best', 
                      fontsize=VISUALIZATION_CONFIG.get('legend_size', 10),
                      title_fontsize=VISUALIZATION_CONFIG.get('legend_title_size', 12))
        else:
            # Fallback or warning if palette_to_use is None but legend is expected
            print(f"Warning: palette_to_use is None for catplot '{title}'. Legend may be incorrect.")
            ax.legend(title=METRIC_DISPLAY_NAMES.get(hue_var, hue_var), loc='best') 

    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        full_path = os.path.join(output_dir, filename)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"Descriptive stats catplot (mean with std error bars) saved to {full_path}")
        except Exception as e:
            print(f"Error saving descriptive stats catplot to {full_path}: {e}")
    
    plt.close(fig)
    return fig

def plot_descriptive_stats_histogram(
    data_df: pd.DataFrame, 
    metric_name: str, 
    value_column: str, 
    title: str, 
    output_dir: str | None = None, 
    filename: str | None = None,
    round_name: str | None = None,
    phase_name: str | None = None,
    bins: int = 30
):
    """
    Generates a histogram of a metric for each tenant, with optional KDE overlay.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('histogram_figsize', (10, 6)))

    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    plot_title = f"{title}"
    if phase_name:
        plot_title += f" - Phase: {PHASE_DISPLAY_NAMES.get(phase_name, phase_name)}"
    if round_name:
        plot_title += f" (Round: {round_name})"

    ax.set_title(plot_title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))

    time_col = 'experiment_elapsed_seconds' if 'experiment_elapsed_seconds' in data_df.columns else data_df.index.name

    if 'tenant' in data_df.columns and data_df['tenant'].nunique() > 0:
        unique_tenants = sorted(data_df['tenant'].unique())
        for tenant in unique_tenants:
            tenant_df = data_df[data_df['tenant'] == tenant]
            if not tenant_df.empty:
                sns.histplot(tenant_df, x=value_column, bins=bins, ax=ax, 
                             label=str(tenant), color=TENANT_COLORS.get(str(tenant), '#333333'),
                             kde=True, stat="density", common_norm=False, 
                             element="step", linewidth=2, alpha=0.7)
        # Position legend automatically for best location
        ax.legend(title='Tenant', loc='best', 
                  fontsize=VISUALIZATION_CONFIG.get('legend_size', 10)) 
        legend = ax.get_legend()
        if legend: # Check if legend exists
            plt.setp(legend.get_title(), fontsize=VISUALIZATION_CONFIG.get('legend_title_size', 12))
    else:
        # Plot without tenant breakdown
        sns.histplot(data_df, x=value_column, bins=bins, ax=ax, 
                     kde=True, stat="density", common_norm=False, 
                     element="step", linewidth=2, alpha=0.7)

    ax.set_xlabel(f"{display_metric_name} ({value_column.capitalize()})", fontsize=VISUALIZATION_CONFIG.get('label_size', 12))
    ax.set_ylabel('Density', fontsize=VISUALIZATION_CONFIG.get('label_size', 12))
    ax.tick_params(axis='both', which='major', labelsize=VISUALIZATION_CONFIG.get('tick_size', 10))
    ax.grid(bool(VISUALIZATION_CONFIG.get('grid_enabled', True)), # Cast to bool
            linestyle=VISUALIZATION_CONFIG.get('grid_linestyle', '--'), 
            alpha=VISUALIZATION_CONFIG.get('grid_alpha', 0.7))

    plt.tight_layout()

    if output_dir and filename:
        save_figure(fig, output_dir, filename)
        print(f"Descriptive stats histogram saved to {os.path.join(output_dir, filename)}")
    plt.close(fig)
    return fig

def plot_histogram_with_kde(
    data_series: pd.Series, 
    title: str, 
    xlabel: str, 
    output_dir: str | None = None, 
    filename: str | None = None, 
    bins: int = 30, 
    color: str = 'skyblue', 
    kde_color: str = 'red',
    round_name: str | None = None,
    phase_name: str | None = None
):
    """
    Generates a histogram with an optional KDE overlay for a given data series.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('histogram_figsize', (10, 6)))

    sns.histplot(data_series, bins=bins, ax=ax, color=color, kde=False, stat="density", 
                 element="step", linewidth=2, alpha=0.7)

    # Overlay KDE if requested
    if VISUALIZATION_CONFIG.get('kde_enabled', True):
        sns.kdeplot(data_series, ax=ax, color=kde_color, linewidth=2.5, alpha=0.8, label='KDE')

    ax.set_title(title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    ax.set_xlabel(xlabel, fontsize=VISUALIZATION_CONFIG.get('label_size', 12))
    ax.set_ylabel('Density', fontsize=VISUALIZATION_CONFIG.get('label_size', 12))
    ax.tick_params(axis='both', which='major', labelsize=VISUALIZATION_CONFIG.get('tick_size', 10))
    ax.grid(bool(VISUALIZATION_CONFIG.get('grid_enabled', True)), # Cast to bool
            linestyle=VISUALIZATION_CONFIG.get('grid_linestyle', '--'), 
            alpha=VISUALIZATION_CONFIG.get('grid_alpha', 0.7))

    plt.tight_layout()

    if output_dir and filename:
        save_figure(fig, output_dir, filename)
        print(f"Histogram with KDE saved to {os.path.join(output_dir, filename)}")
    plt.close(fig)
    return fig

def plot_pca_scree_plot(
    explained_variance_ratio: np.ndarray, 
    cumulative_variance_ratio: np.ndarray, 
    output_dir: str | None = None, 
    filename: str | None = None, 
    title: str = 'Scree Plot for PCA'
):
    """
    Generates a Scree Plot for PCA, showing explained variance by each component.

    Args:
        explained_variance_ratio (np.ndarray): Array of explained variance ratio for each PC.
        cumulative_variance_ratio (np.ndarray): Array of cumulative explained variance ratio.
        title (str): Title of the plot.
        output_dir (str): Directory to save the plot.
        filename (str): Filename for the saved plot.
    """
    set_publication_style()
    fig, ax1 = plt.subplots(figsize=VISUALIZATION_CONFIG.get('scree_plot_figsize', (10, 7)))

    num_components = len(explained_variance_ratio)
    pc_labels = [f'PC{i+1}' for i in range(num_components)]

    # Bar plot for individual explained variance
    color_bar = VISUALIZATION_CONFIG.get('scree_bar_color', 'steelblue')
    ax1.bar(pc_labels, explained_variance_ratio, alpha=0.7, color=color_bar, label='Individual Explained Variance')
    ax1.set_xlabel('Principal Component', fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    ax1.set_ylabel('Explained Variance Ratio', fontsize=VISUALIZATION_CONFIG.get('label_size', 14), color=color_bar)
    ax1.tick_params(axis='y', labelcolor=color_bar)
    ax1.tick_params(axis='x', rotation=45)

    # Line plot for cumulative explained variance on a secondary y-axis
    ax2 = ax1.twinx()
    color_line = VISUALIZATION_CONFIG.get('scree_line_color', 'darkred')
    ax2.plot(pc_labels, cumulative_variance_ratio, color=color_line, marker='o', linestyle='-', linewidth=2, label='Cumulative Explained Variance')
    ax2.set_ylabel('Cumulative Explained Variance Ratio', fontsize=VISUALIZATION_CONFIG.get('label_size', 14), color=color_line)
    ax2.tick_params(axis='y', labelcolor=color_line)
    ax2.set_ylim(0, 1.05) # Ensure y-axis for cumulative goes up to just above 1.0

    # Add a horizontal line at 0.9 or 0.95 for reference (common thresholds)
    threshold_values = VISUALIZATION_CONFIG.get('scree_variance_threshold_lines', [0.9, 0.95])
    if not isinstance(threshold_values, list):
        threshold_values = [threshold_values] # Ensure it's a list

    for thresh in threshold_values:
        if isinstance(thresh, (int, float)) and cumulative_variance_ratio.max() >= thresh:
            ax2.axhline(y=thresh, color='grey', linestyle=':', linewidth=1, label=f'{thresh*100:.0f}% Variance Threshold')

    fig.suptitle(title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right', fontsize=VISUALIZATION_CONFIG.get('legend_size', 10))

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle

    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"Scree plot saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving scree plot to {os.path.join(output_dir, filename)}: {e}")
    
    plt.close(fig)
    return fig

def plot_pca_loadings_heatmap(
    loadings_df: pd.DataFrame, 
    title: str = 'PCA Component Loadings', 
    output_dir: str | None = None, 
    filename: str | None = None, 
    cmap: str = 'viridis'
):
    """
    Generates a heatmap of PCA component loadings.

    Args:
        loadings_df (pd.DataFrame): DataFrame with PCA loadings. 
                                     Expected index: principal components, columns: features.
        title (str): The title of the heatmap.
        output_dir (str, optional): Directory to save the plot. If None, plot is not saved.
        filename (str, optional): Filename for the saved plot. If None, a default is generated.
        cmap (str): Colormap for the heatmap.
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('heatmap_figsize', (10, 8)))
    sns.heatmap(loadings_df, annot=True, fmt=".2f", cmap=cmap, center=0, linewidths=.5, 
                cbar_kws={'label': 'Loading Value'}, ax=ax)
    ax.set_title(title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"PCA loadings heatmap saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving PCA loadings heatmap to {os.path.join(output_dir, filename)}: {e}")
    
    return fig

def plot_pca_score_plot(
    pca_scores: pd.DataFrame, 
    pc_x: int = 0, 
    pc_y: int = 1, 
    title: str = 'PCA Score Plot', 
    output_dir: str | None = None, 
    filename: str | None = None,
    hue_data: pd.Series | None = None, 
    palette: str | dict | None = None 
):
    set_publication_style()
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('scatter_figsize', (8, 8)))

    pc_x_values = pca_scores.iloc[:, pc_x].values
    pc_y_values = pca_scores.iloc[:, pc_y].values

    var_pc_x = np.var(pc_x_values, ddof=1) if len(pc_x_values) > 1 else 0.0
    var_pc_y = np.var(pc_y_values, ddof=1) if len(pc_y_values) > 1 else 0.0
    
    var_pc_x = float(var_pc_x)
    var_pc_y = float(var_pc_y)

    sum_of_plot_variances = var_pc_x + var_pc_y

    if sum_of_plot_variances > 1e-9:
        percent_var_x = (var_pc_x / sum_of_plot_variances) * 100
        percent_var_y = (var_pc_y / sum_of_plot_variances) * 100
        ax.set_xlabel(f'PC{pc_x+1} ({percent_var_x:.1f}% of plot variance)', fontsize=VISUALIZATION_CONFIG.get('label_size', 12))
        ax.set_ylabel(f'PC{pc_y+1} ({percent_var_y:.1f}% of plot variance)', fontsize=VISUALIZATION_CONFIG.get('label_size', 12))
    else:
        ax.set_xlabel(f'PC{pc_x+1}', fontsize=VISUALIZATION_CONFIG.get('label_size', 12))
        ax.set_ylabel(f'PC{pc_y+1}', fontsize=VISUALIZATION_CONFIG.get('label_size', 12))

    scatter_kwargs = {
        's': VISUALIZATION_CONFIG.get('scatter_marker_size', 50),
        'alpha': VISUALIZATION_CONFIG.get('scatter_marker_alpha', 0.7),
        'edgecolor': VISUALIZATION_CONFIG.get('scatter_marker_edgecolor', 'k'),
        'linewidth': VISUALIZATION_CONFIG.get('scatter_marker_linewidth', 0.5)
    }
    
    legend_title = 'Category' # Default legend title
    aligned_hue_data = None # Initialize aligned_hue_data

    if hue_data is not None and not hue_data.empty:
        aligned_hue_data = hue_data.reset_index(drop=True) if isinstance(hue_data, pd.Series) else hue_data
        
        if len(aligned_hue_data) == len(pca_scores):
            if isinstance(hue_data, pd.Series) and hasattr(hue_data, 'name') and hue_data.name:
                legend_title = str(hue_data.name)

            if palette:
                scatter_kwargs['hue'] = aligned_hue_data
                scatter_kwargs['palette'] = palette
            elif isinstance(aligned_hue_data, pd.Series) and all(item in TENANT_COLORS for item in aligned_hue_data.unique()):
                scatter_kwargs['c'] = aligned_hue_data.map(TENANT_COLORS)
            else: 
                scatter_kwargs['hue'] = aligned_hue_data 
        else:
            print(f"Warning: hue_data length ({len(aligned_hue_data)}) does not match pca_scores length ({len(pca_scores)}). Plotting without hue.")
            aligned_hue_data = None # Reset if not usable

    sns.scatterplot(x=pca_scores.iloc[:, pc_x], y=pca_scores.iloc[:, pc_y], ax=ax, **scatter_kwargs)
    ax.set_title(title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    
    if aligned_hue_data is not None: # Check if aligned_hue_data was successfully set
        if 'hue' in scatter_kwargs:
            current_legend = ax.get_legend()
            if current_legend:
                current_legend.set_title(legend_title)
        elif 'c' in scatter_kwargs and isinstance(aligned_hue_data, pd.Series):
            unique_hues = sorted(aligned_hue_data.unique())
            handles = [Line2D([0], [0], marker='o', color='w', label=str(h),
                              markerfacecolor=TENANT_COLORS.get(str(h), '#333333'), 
                              markersize=VISUALIZATION_CONFIG.get('legend_marker_size', 8)) 
                       for h in unique_hues if str(h) in TENANT_COLORS]
            if handles:
                 ax.legend(handles=handles, title=legend_title, 
                           loc='best', fontsize=VISUALIZATION_CONFIG.get('legend_size', 10))

    plt.tight_layout()

    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"PCA score plot saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving PCA score plot to {os.path.join(output_dir, filename)}: {e}")
    
    plt.close(fig)
    return fig

def plot_ica_loadings_heatmap(
    unmixing_matrix_df: pd.DataFrame, 
    title: str = 'ICA Unmixing Matrix (Loadings)', 
    output_dir: str | None = None, 
    filename: str | None = None, 
    cmap: str = 'cividis'
):
    """
    Generates a heatmap of ICA unmixing matrix (loadings).

    Args:
        unmixing_matrix_df (pd.DataFrame): DataFrame with ICA unmixing matrix. 
                                            Expected index: components, columns: features.
        title (str): The title of the heatmap.
        output_dir (str, optional): Directory to save the plot. If None, plot is not saved.
        filename (str, optional): Filename for the saved plot. If None, a default is generated.
        cmap (str): Colormap for the heatmap.
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('heatmap_figsize', (10, 8)))
    sns.heatmap(unmixing_matrix_df, annot=True, fmt=".2f", cmap=cmap, center=0, linewidths=.5, 
                cbar_kws={'label': 'Loading Value'}, ax=ax)
    ax.set_title(title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"ICA loadings heatmap saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving ICA loadings heatmap to {os.path.join(output_dir, filename)}: {e}")
    
    return fig

def plot_ica_score_plot(
    ica_scores: pd.DataFrame, 
    ic_x: int = 0, 
    ic_y: int = 1, 
    title: str = 'ICA Score Plot', 
    output_dir: str | None = None, 
    filename: str | None = None,
    hue_data: pd.Series | None = None, 
    palette: str | dict | None = None
):
    set_publication_style()
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('scatter_figsize', (8, 8)))

    ic_x_values = ica_scores.iloc[:, ic_x].values
    ic_y_values = ica_scores.iloc[:, ic_y].values

    var_ic_x = np.var(ic_x_values, ddof=1) if len(ic_x_values) > 1 else 0.0
    var_ic_y = np.var(ic_y_values, ddof=1) if len(ic_y_values) > 1 else 0.0

    var_ic_x = float(var_ic_x)
    var_ic_y = float(var_ic_y)
    
    sum_of_plot_variances = var_ic_x + var_ic_y

    if sum_of_plot_variances > 1e-9:
        percent_var_x = (var_ic_x / sum_of_plot_variances) * 100
        percent_var_y = (var_ic_y / sum_of_plot_variances) * 100
        ax.set_xlabel(f'IC{ic_x+1} ({percent_var_x:.1f}% of plot variance)', fontsize=VISUALIZATION_CONFIG.get('label_size', 12))
        ax.set_ylabel(f'IC{ic_y+1} ({percent_var_y:.1f}% of plot variance)', fontsize=VISUALIZATION_CONFIG.get('label_size', 12))
    else:
        ax.set_xlabel(f'IC{ic_x+1}', fontsize=VISUALIZATION_CONFIG.get('label_size', 12))
        ax.set_ylabel(f'IC{ic_y+1}', fontsize=VISUALIZATION_CONFIG.get('label_size', 12))

    scatter_kwargs = {
        's': VISUALIZATION_CONFIG.get('scatter_marker_size', 50),
        'alpha': VISUALIZATION_CONFIG.get('scatter_marker_alpha', 0.7),
        'edgecolor': VISUALIZATION_CONFIG.get('scatter_marker_edgecolor', 'k'),
        'linewidth': VISUALIZATION_CONFIG.get('scatter_marker_linewidth', 0.5)
    }
    
    legend_title = 'Category' # Default legend title
    aligned_hue_data = None # Initialize aligned_hue_data

    if hue_data is not None and not hue_data.empty:
        aligned_hue_data = hue_data.reset_index(drop=True) if isinstance(hue_data, pd.Series) else hue_data
        if len(aligned_hue_data) == len(ica_scores):
            if isinstance(hue_data, pd.Series) and hasattr(hue_data, 'name') and hue_data.name:
                legend_title = str(hue_data.name)

            if palette:
                scatter_kwargs['hue'] = aligned_hue_data
                scatter_kwargs['palette'] = palette
            elif isinstance(aligned_hue_data, pd.Series) and all(item in TENANT_COLORS for item in aligned_hue_data.unique()):
                scatter_kwargs['c'] = aligned_hue_data.map(TENANT_COLORS)
            else:
                scatter_kwargs['hue'] = aligned_hue_data
        else:
            print(f"Warning: hue_data length ({len(aligned_hue_data)}) does not match ica_scores length ({len(ica_scores)}). Plotting without hue.")
            aligned_hue_data = None # Reset if not usable

    sns.scatterplot(x=ica_scores.iloc[:, ic_x], y=ica_scores.iloc[:, ic_y], ax=ax, **scatter_kwargs)
    ax.set_title(title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))

    if aligned_hue_data is not None: # Check if aligned_hue_data was successfully set
        if 'hue' in scatter_kwargs:
            current_legend = ax.get_legend()
            if current_legend:
                current_legend.set_title(legend_title)
        elif 'c' in scatter_kwargs and isinstance(aligned_hue_data, pd.Series):
            unique_hues = sorted(aligned_hue_data.unique())
            handles = [Line2D([0], [0], marker='o', color='w', label=str(h),
                              markerfacecolor=TENANT_COLORS.get(str(h), '#333333'),
                              markersize=VISUALIZATION_CONFIG.get('legend_marker_size', 8))
                       for h in unique_hues if str(h) in TENANT_COLORS]
            if handles:
                ax.legend(handles=handles, title=legend_title,
                          loc='best', fontsize=VISUALIZATION_CONFIG.get('legend_size', 10))

    plt.tight_layout()

    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"ICA score plot saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving ICA score plot to {os.path.join(output_dir, filename)}: {e}")
    
    plt.close(fig)
    return fig

def plot_correlation_scatter_matrix(df: pd.DataFrame, title: str = 'Correlation Scatter Matrix', output_dir: str | None = None, filename: str | None = None, hue_column: str | None = None):
    set_publication_style()
    if df.empty:
        print(f"DataFrame is empty. Cannot generate scatter matrix for '{title}'.")
        return None

    # Select only numeric columns for the pairplot, otherwise it can fail or be very slow
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        print(f"Not enough numeric columns ({numeric_df.shape[1]}) to generate a scatter matrix for '{title}'. Need at least 2.")
        return None
    
    # If hue_column is specified and exists, add it back to numeric_df for coloring
    # Ensure hue_column data is not accidentally converted/dropped if it was non-numeric
    final_df_for_plot = numeric_df
    if hue_column and hue_column in df.columns:
        final_df_for_plot = pd.concat([numeric_df, df[hue_column]], axis=1)
        # If hue_column was numeric and already in numeric_df, it might be duplicated.
        # sns.pairplot handles duplicated column names by appending _x, _y, so it should be fine.
        # Or, ensure it's not duplicated if it was already numeric:
        if hue_column in numeric_df.columns and df[hue_column].dtype == numeric_df[hue_column].dtype:
             # it was already there, no need to concat again, just use numeric_df and specify hue_column
             final_df_for_plot = numeric_df
        else: # it was not numeric or was dropped, so concat is needed.
             final_df_for_plot = pd.concat([numeric_df, df[hue_column]], axis=1)

    # Ensure scatter_matrix_figsize is a tuple
    figsize_config = VISUALIZATION_CONFIG.get('scatter_matrix_figsize', (12, 12))
    if isinstance(figsize_config, int): # If it's an int, make it a tuple (e.g. (val, val))
        figsize_tuple: tuple[float, float] | None = (float(figsize_config), float(figsize_config))
    elif isinstance(figsize_config, Sequence) and len(figsize_config) == 2:
        figsize_tuple = (float(figsize_config[0]), float(figsize_config[1]))
    else: # Default or fallback
        figsize_tuple = (12.0, 12.0)

    plt.figure(figsize=figsize_tuple) # Control figure size
    
    pairplot_kwargs = {
        'diag_kind': VISUALIZATION_CONFIG.get('scatter_matrix_diag_kind', 'kde'),
        'plot_kws': {'alpha': VISUALIZATION_CONFIG.get('scatter_matrix_plot_alpha', 0.6),
                     's': VISUALIZATION_CONFIG.get('scatter_matrix_plot_s', 20), 
                     'edgecolor': VISUALIZATION_CONFIG.get('scatter_matrix_plot_edgecolor', 'k')},
        'diag_kws': {'alpha': VISUALIZATION_CONFIG.get('scatter_matrix_diag_alpha', 0.7)}
    }
    
    if hue_column and hue_column in final_df_for_plot.columns:
        unique_hues = final_df_for_plot[hue_column].nunique()
        if unique_hues <= VISUALIZATION_CONFIG.get('scatter_matrix_max_hue_categories', 10):
            pairplot_kwargs['hue'] = hue_column
            # Try to use TENANT_COLORS if hue_column is 'tenant' or similar and palette is not explicitly set
            if hue_column.lower() in ['tenant', 'tenants'] and 'palette' not in pairplot_kwargs:
                # Create a palette mapping for the unique values in the hue column
                custom_palette = {str(val): TENANT_COLORS.get(str(val), '#333333') for val in final_df_for_plot[hue_column].unique()}
                pairplot_kwargs['palette'] = custom_palette
            elif 'palette' not in pairplot_kwargs: # Default palette if not tenant and no palette specified
                pairplot_kwargs['palette'] = VISUALIZATION_CONFIG.get('scatter_matrix_palette', 'viridis')
        else:
            print(f"Warning: Too many unique values in hue_column '{hue_column}' ({unique_hues}). Plotting without hue.")
            if 'hue' in pairplot_kwargs: del pairplot_kwargs['hue'] # remove hue if too many categories
            hue_column = None # Reset hue_column so title doesn't mention it if not used

    try:
        g = sns.pairplot(final_df_for_plot, **pairplot_kwargs)
    except Exception as e:
        print(f"Error during sns.pairplot for '{title}': {e}. Skipping matrix.")
        # Potentially close any open figures if pairplot partially drew something before erroring
        if plt.gcf().get_axes(): # Check if current figure has axes
            plt.close(plt.gcf())
        return None

    # Using a more robust way to set titles for each subplot
    for i, row_ax in enumerate(g.axes):
        for j, ax in enumerate(row_ax):
            if i == j: # Diagonal plots
                ax.set_title(final_df_for_plot.columns[i], fontsize=VISUALIZATION_CONFIG.get('scatter_matrix_subplot_title_size', 10))
            # You can add more customization for off-diagonal titles if needed
            # ax.xaxis.label.set_size(VISUALIZATION_CONFIG.get('tick_size', 10))
            # ax.yaxis.label.set_size(VISUALIZATION_CONFIG.get('tick_size', 10))
            ax.tick_params(labelsize=VISUALIZATION_CONFIG.get('scatter_matrix_tick_size', 8))


    main_title = title
    if hue_column and pairplot_kwargs.get('hue'): # Check if hue was actually used
        main_title += f" (Colored by {hue_column})"
    g.fig.suptitle(main_title, y=1.02, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle

    fig = g.fig # Get the figure object from the PairGrid

    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"Scatter matrix saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving scatter matrix to {os.path.join(output_dir, filename)}: {e}")
    
    return fig # Return the figure object
