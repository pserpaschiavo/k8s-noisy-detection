"""
Module for generating high-quality visualizations for the noisy neighbors experiment.

This module provides functions to create graphs and visualizations with quality
suitable for academic publications, following good data visualization practices.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from pipeline.config import TENANT_COLORS, METRIC_DISPLAY_NAMES, VISUALIZATION_CONFIG, PHASE_DISPLAY_NAMES
import matplotlib.container  # Ensure this is imported


# Global settings for publication-quality plots
def set_publication_style():
    """Sets the style for publication-quality visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')  # Maintains a base style
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


def plot_metric_with_anomalies(df, metric_name, time_column='experiment_elapsed_seconds', value_column='value', 
                               anomaly_column='is_anomaly', change_points=None,
                               phases=None, tenants=None, show_phase_markers=True, 
                               figsize=None, use_total_duration=False, total_duration_seconds=None,
                               show_as_percentage=False, node_config=None):
    """
    Creates a line plot for a metric, highlighting anomalies and change points.
    
    Args:
        df (DataFrame): DataFrame with metric data and anomaly column.
        metric_name (str): Name of the metric (key for METRIC_DISPLAY_NAMES).
        time_column (str): Column with time values for the X-axis. Default: 'experiment_elapsed_seconds'.
        value_column (str): Column with metric values.
        anomaly_column (str): Boolean column indicating if a point is an anomaly.
        change_points (list): List of timestamps/indices where significant changes occur.
        phases (list): List of phases to include (None = all).
        tenants (list): List of tenants to include (None = all).
        show_phase_markers (bool): Whether to show vertical lines marking phases.
        figsize (tuple): Figure size (optional, uses config if None).
        use_total_duration (bool): If True, the X-axis will use 'total_elapsed_seconds'.
        total_duration_seconds (float): Total duration of the experiment in seconds.
        show_as_percentage (bool): If True, display values as percentages of total capacity.
        node_config (dict): Configuration with node resource capacities.
        
    Returns:
        Figure: Matplotlib figure object.
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
    
    anomaly_label_added = False  # Flag to ensure anomaly label is added only once
    for tenant, group in data.groupby('tenant'):
        group = group.sort_values(time_column)
        color = TENANT_COLORS.get(tenant, None)
        
        ax.plot(group[time_column], group[value_column], 
                label=tenant, linewidth=1.5, color=color, alpha=0.8)
        
        # Highlight anomalies
        anomalies = group[group[anomaly_column] == True]
        if not anomalies.empty:
            label_anom = 'Anomalies' if not anomaly_label_added else None
            ax.scatter(anomalies[time_column], anomalies[value_column], 
                       color='red', s=30, label=label_anom, 
                       marker='x', zorder=5, alpha=0.7)  # Reduced size, added alpha
            anomaly_label_added = True

    if show_phase_markers:
        phase_starts = {}
        for phase, group_data in data.groupby('phase'):
            min_time = group_data[time_column].min()
            if min_time not in phase_starts:
                phase_starts[min_time] = []
            phase_starts[min_time].append(phase)
        
        for time, phase_list in phase_starts.items():
            if time > data[time_column].min():
                ax.axvline(x=time, color='gray', linestyle='--', alpha=0.7)
                ax.text(time, ax.get_ylim()[1] * 0.95, ', '.join(phase_list), 
                        rotation=90, verticalalignment='top', alpha=0.7)

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
    ax.legend(by_label.values(), by_label.keys(), title='Legend')
    
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.tight_layout()
    
    return fig


def plot_metric_by_phase(df, metric_name, time_column='experiment_elapsed_seconds', value_column='value', 
                         phases=None, tenants=None, show_phase_markers=True, 
                         figsize=None, use_total_duration=False, total_duration_seconds=None,
                         show_as_percentage=False, node_config=None, use_formatted_values=True):
    """
    Creates a line plot for a metric, showing the different phases of the experiment.
    
    Args:
        df (DataFrame): DataFrame with metric data.
        metric_name (str): Name of the metric (key for METRIC_DISPLAY_NAMES).
        time_column (str): Column with time values for the X-axis. Default: 'experiment_elapsed_seconds'.
        value_column (str): Column with metric values.
        phases (list): List of phases to include (None = all).
        tenants (list): List of tenants to include (None = all).
        show_phase_markers (bool): Whether to show vertical lines marking phases.
        figsize (tuple): Figure size (optional, uses config if None).
        use_total_duration (bool): If True, the X-axis will use 'total_elapsed_seconds'.
        total_duration_seconds (float): Total duration of the experiment in seconds (required if use_total_duration=True).
        show_as_percentage (bool): If True, show values as percentage of total cluster resources.
        node_config (dict): Configuration with node resources (required if show_as_percentage=True).
        use_formatted_values (bool): If True, use formatted values and appropriate units.
        
    Returns:
        Figure: Matplotlib figure object.
    """
    set_publication_style()
    
    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    
    # Filter data if necessary
    data = df.copy()
    if phases:
        data = data[data['phase'].isin(phases)]
    if tenants:
        data = data[data['tenant'].isin(tenants)]
    
    # Obter lista de todos os tenants disponíveis nos dados
    all_tenants = data['tenant'].unique().tolist()
    
    # Filtrar por tenants específicos se solicitado
    if tenants is not None:
        all_tenants = [t for t in all_tenants if t in tenants]
    
    # Create figure and axis
    if figsize is None:
        figsize = (VISUALIZATION_CONFIG.get('figure_width', 10), 
                   VISUALIZATION_CONFIG.get('figure_height', 6))
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if we have quota or normalized info
    has_normalized_data = 'normalized_value' in data.columns
    has_unit_info = 'unit' in data.columns and not data['unit'].isna().all()
    has_formatted_values = 'value_formatted' in data.columns
    has_quota_info = 'quota_limit_formatted' in data.columns
    
    # Determine which column to plot and how to label it
    y_column = value_column
    if use_formatted_values and show_as_percentage and has_normalized_data:
        y_column = 'normalized_value'
    elif use_formatted_values and value_column == 'value' and has_formatted_values:
        # When we're using the 'value' column but we have formatted values available
        # we don't change the column but will use better labels
        pass
    
    # Plot lines for each tenant
    for tenant in sorted(all_tenants):
        tenant_data = data[data['tenant'] == tenant]
        if tenant_data.empty:
            continue
            
        # Get color for tenant
        color = TENANT_COLORS.get(tenant, 'gray')
        
        # Plot tenant line - sem destaque especial para o tenant-b
        line = ax.plot(tenant_data[time_column], tenant_data[y_column], 
                       label=tenant, color=color, linewidth=2)
    
    # Add phase markers if requested
    if show_phase_markers and 'phase_start' in data.columns:
        phase_starts = data[['phase', 'phase_start']].drop_duplicates()
        
        for _, row in phase_starts.iterrows():
            phase_name = row['phase']
            phase_start = row['phase_start']
            
            # Only add if within plot range
            if phase_start >= data[time_column].min() and phase_start <= data[time_column].max():
                ax.axvline(x=phase_start, color='black', linestyle='--', alpha=0.7)
                ax.text(phase_start, ax.get_ylim()[1]*0.95, f' {phase_name}', 
                        rotation=90, verticalalignment='top')
    
    # Configure axis labels
    ax.set_xlabel('Tempo (segundos)')
    
    # Use proper Y label based on available information
    if y_column == 'normalized_value':
        # Usar descrição da normalização se disponível
        if 'normalized_description' in data.columns and not data['normalized_description'].isna().all():
            description = data['normalized_description'].iloc[0]
            ax.set_ylabel(f"{display_metric_name} ({description})")
        else:
            ax.set_ylabel(f"{display_metric_name} (% of total capacity)")
    elif has_unit_info:
        # Usar informação de unidade diretamente
        unit = data['unit'].iloc[0]
        ax.set_ylabel(f"{display_metric_name} ({unit})")
    elif has_quota_info:
        # Usar informação de quota formatada
        quota_info = data['quota_limit_formatted'].iloc[0]
        if y_column == 'normalized_value':
            ax.set_ylabel(f"{display_metric_name} (% of {quota_info})")
        else:
            ax.set_ylabel(f"{display_metric_name} (units relative to {quota_info})")
    else:
        # Fallback para nome da métrica sem unidade específica
        ax.set_ylabel(display_metric_name)
    
    # Add legend and title
    title_parts = []
    if metric_name:
        title_parts.append(display_metric_name)
    if phases and len(phases) == 1:
        title_parts.append(f"- Fase: {phases[0]}")
    
    # Adicionar informações de formatação ao título se disponível
    if has_formatted_values and use_formatted_values:
        # Encontrar o valor máximo e sua representação formatada
        max_idx = data[y_column].idxmax()
        if max_idx is not None and 'value_formatted' in data.columns:
            max_formatted = data.loc[max_idx, 'value_formatted']
            title_parts.append(f"(máx: {max_formatted})")
    
    ax.set_title(" ".join(title_parts))
    ax.legend()
    
    # Grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    fig.tight_layout()
    
    return fig


def plot_phase_comparison(df, metric_name, value_column='mean', 
                         error_column=None, figsize=None,
                         show_as_percentage=False, node_config=None):
    """
    Creates a bar chart comparing different phases for each tenant.
    
    Args:
        df (DataFrame): DataFrame with data aggregated by phase and tenant.
        metric_name (str): Name of the metric (key for METRIC_DISPLAY_NAMES).
        value_column (str): Column with mean metric values.
        error_column (str): Column with error/standard deviation values (optional).
        figsize (tuple): Figure size (optional, uses config if None).
        show_as_percentage (bool): If True, show values as percentage of total cluster resources.
        node_config (dict): Configuration with node resources (required if show_as_percentage=True).
        
    Returns:
        Figure: Matplotlib figure object.
    """
    set_publication_style()
    
    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    
    if figsize is None:
        figsize = (VISUALIZATION_CONFIG.get('figure_width', 14), VISUALIZATION_CONFIG.get('figure_height', 8))
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for the plot
    if 'phase_name' in df.columns:
        pivot = df.pivot_table(index='tenant', columns='phase_name', values=value_column)
    else:
        pivot = df.pivot_table(index='tenant', columns='phase', values=value_column)
    
    yerr = None
    if error_column:
        if 'phase_name' in df.columns:
            error_pivot = df.pivot_table(index='tenant', columns='phase_name', values=error_column)
        else:
            error_pivot = df.pivot_table(index='tenant', columns='phase', values=error_column)
        yerr = error_pivot.values.T
    
    pivot.plot(kind='bar', ax=ax, yerr=yerr, capsize=5, width=0.8,
               colormap=LinearSegmentedColormap.from_list('phases', ['#4C78A8', '#F58518', '#72B7B2']))
    
    ax.set_xlabel('Tenant')
    
    if show_as_percentage:
        if node_config:
            if metric_name == 'cpu_usage' and 'CPUS' in node_config:
                unit_info = f"% of {node_config['CPUS']} CPU cores"
            elif metric_name == 'memory_usage' and 'MEMORY_GB' in node_config:
                unit_info = f"% of {node_config['MEMORY_GB']} GB memory"
            elif metric_name == 'disk_throughput_total':
                unit_info = "% of theoretical max throughput"
            elif metric_name == 'network_total_bandwidth':
                unit_info = "% of 1 Gbps bandwidth"
            else:
                unit_info = "% of total"
                
            ax.set_ylabel(f"{display_metric_name} ({unit_info})")
            ax.set_title(f'Comparison of {display_metric_name} by Tenant and Phase ({unit_info})')
        else:
            ax.set_ylabel(f"{display_metric_name} (% of total)")
            ax.set_title(f'Comparison of {display_metric_name} by Tenant and Phase (% of total)')
    else:
        ax.set_ylabel(display_metric_name)
        ax.set_title(f'Comparison of {display_metric_name} by Tenant and Phase')
    
    ax.legend(title='Phase')
    ax.grid(axis='y', alpha=0.3)
    
    num_bar_containers = len(pivot.columns)
    
    for i in range(num_bar_containers):
        if i < len(ax.containers) and isinstance(ax.containers[i], matplotlib.container.BarContainer):
            if show_as_percentage:
                ax.bar_label(ax.containers[i], fmt='%.1f%%', padding=3, fontsize=10)
            else:
                ax.bar_label(ax.containers[i], fmt='%.2f', padding=3, fontsize=10)
    
    plt.tight_layout()
    
    return fig


def plot_tenant_impact_heatmap(impact_df, metric_name, value_column='impact_percent',
                              cmap='RdYlGn_r', figsize=None, 
                              show_as_percentage=False, node_config=None):
    """
    Creates a heatmap showing the percentage impact on each tenant during the attack phase.
    
    Args:
        impact_df (DataFrame): DataFrame with impact calculated per tenant.
        metric_name (str): Name of the metric (key for METRIC_DISPLAY_NAMES).
        value_column (str): Column with percentage impact values.
        cmap (str): Colormap name for the heatmap.
        figsize (tuple): Figure size (optional, uses config if None).
        show_as_percentage (bool): If True, values are already in percentage format.
        node_config (dict): Configuration with node resource capacities.
        
    Returns:
        Figure: Matplotlib figure object.
    """
    set_publication_style()
    
    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    
    if figsize is None:
        figsize = (VISUALIZATION_CONFIG.get('figure_width', 12), VISUALIZATION_CONFIG.get('figure_height', 8))
    fig, ax = plt.subplots(figsize=figsize)
    
    if 'round' in impact_df.columns and len(impact_df['round'].unique()) > 1:
        pivot = impact_df.pivot_table(index='tenant', columns='round', values=value_column)
    else:
        pivot = impact_df.set_index('tenant')[[value_column]]
    
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap, center=0,
                linewidths=.5, ax=ax, cbar_kws={'label': 'Impact (%)'})
    
    title_prefix = 'Impact on'
    title_suffix = 'during Attack Phase'
    
    if show_as_percentage:
        if node_config and metric_name in ['cpu_usage', 'memory_usage', 'disk_throughput_total', 'network_total_bandwidth']:
            if metric_name == 'cpu_usage' and 'CPUS' in node_config:
                display_metric_name += f" (% of {node_config['CPUS']} CPU cores)"
            elif metric_name == 'memory_usage' and 'MEMORY_GB' in node_config:
                display_metric_name += f" (% of {node_config['MEMORY_GB']} GB)"
            elif metric_name == 'disk_throughput_total':
                display_metric_name += " (% of 500 MB/s)"
            elif metric_name == 'network_total_bandwidth':
                display_metric_name += " (% of 1 Gbps)"
    
    ax.set_title(f'{title_prefix} {display_metric_name} {title_suffix}')
    
    if 'round' in impact_df.columns and len(impact_df['round'].unique()) > 1:
        ax.set_xlabel('Round')
    else:
        ax.set_xlabel('')
    
    ax.set_ylabel('Tenant')
    
    plt.tight_layout()
    
    return fig


def plot_recovery_effectiveness(recovery_df, metric_name, figsize=None, 
                                baseline_phase_name=None, 
                                attack_phase_name=None, 
                                recovery_phase_name=None,
                                show_as_percentage=False, node_config=None):
    """
    Creates a bar chart showing recovery effectiveness.
    
    Args:
        recovery_df (DataFrame): DataFrame with recovery data. 
                                 Must contain columns with phase names and 'tenant'.
        metric_name (str): Name of the metric (key for METRIC_DISPLAY_NAMES).
        figsize (tuple): Figure size (optional, uses config if None).
        baseline_phase_name (str, optional): Column name for the baseline phase. 
                                             If None, uses the default from PHASE_DISPLAY_NAMES.
        attack_phase_name (str, optional): Column name for the attack phase.
                                           If None, uses the default from PHASE_DISPLAY_NAMES.
        recovery_phase_name (str, optional): Column name for the recovery phase.
                                             If None, uses the default from PHASE_DISPLAY_NAMES.
        show_as_percentage (bool): If True, display values as percentages of total capacity.
        node_config (dict): Configuration with node resource capacities.
                                             
    Returns:
        Figure: Matplotlib figure object.
    """
    set_publication_style()
    
    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)

    if baseline_phase_name is None:
        baseline_phase_name = "1 - Baseline"
    if attack_phase_name is None:
        attack_phase_name = "2 - Attack"
    if recovery_phase_name is None:
        recovery_phase_name = "3 - Recovery"

    required_phase_cols = [baseline_phase_name, attack_phase_name, recovery_phase_name, 'tenant', 'recovery_percent']
    for col in required_phase_cols:
        if col not in recovery_df.columns:
            raise KeyError(f"Column '{col}' not found in recovery_df. Available columns: {recovery_df.columns.tolist()}")

    if figsize is None:
        figsize = (VISUALIZATION_CONFIG.get('figure_width', 12), VISUALIZATION_CONFIG.get('figure_height', 7))
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    tenants = recovery_df['tenant'].unique()
    x = np.arange(len(tenants))
    width = 0.25

    plot_data = recovery_df.groupby('tenant', as_index=False).agg(
        {baseline_phase_name: 'mean', 
         attack_phase_name: 'mean', 
         recovery_phase_name: 'mean',
         'recovery_percent': 'mean'
        }
    )
    plot_data = pd.DataFrame({'tenant': tenants}).merge(plot_data, on='tenant', how='left')

    rects1 = ax1.bar(x - width, plot_data[baseline_phase_name], width, label=baseline_phase_name, color='#4C78A8')
    rects2 = ax1.bar(x, plot_data[attack_phase_name], width, label=attack_phase_name, color='#F58518')
    rects3 = ax1.bar(x + width, plot_data[recovery_phase_name], width, label=recovery_phase_name, color='#72B7B2')
    
    ax1.set_xlabel('Tenant')
    
    if show_as_percentage:
        unit_info = "%"
        if node_config:
            if metric_name == 'cpu_usage' and 'CPUS' in node_config:
                unit_info = f"% of {node_config['CPUS']} CPU cores"
            elif metric_name == 'memory_usage' and 'MEMORY_GB' in node_config:
                unit_info = f"% of {node_config['MEMORY_GB']} GB memory"
            elif metric_name == 'disk_throughput_total':
                unit_info = "% of 500 MB/s theoretical throughput"
            elif metric_name == 'network_total_bandwidth':
                unit_info = "% of 1 Gbps network interface"
        
        ax1.set_ylabel(f'Mean Value - {display_metric_name} ({unit_info})')
    else:
        ax1.set_ylabel(f'Mean Value - {display_metric_name}')
    
    ax1.set_title(f'Recovery Effectiveness - {display_metric_name}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tenants, rotation=45, ha="right")
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = ax1.twinx()
    
    bars = ax2.bar(x, plot_data['recovery_percent'], width=width*2.5, alpha=0.3, 
                   color=[('#72B7B2' if val >= 0 else '#F58518') for val in plot_data['recovery_percent']])
    
    ax2.set_ylabel('Recovery Percentage (%)')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(correlation_matrix, title='Correlation between metrics', 
                            figsize=None, cmap='viridis', clean_labels=True, annot=True,
                            fmt='.2f'):
    """
    Creates a correlation heatmap between metrics of different tenants.
    
    Args:
        correlation_matrix (DataFrame): Correlation matrix.
        title (str): Plot title.
        figsize (tuple): Figure size (optional, uses config if None).
        cmap (str): Colormap name for the heatmap.
        clean_labels (bool): If True, clean metric_tenant names to more readable format.
        annot (bool): If True, write the data value in each cell.
        fmt (str): String formatting code to use when adding annotations.
        
    Returns:
        Figure: Matplotlib figure object.
    """
    set_publication_style()
    
    if figsize is None:
        figsize = (VISUALIZATION_CONFIG.get('figure_width', 14), VISUALIZATION_CONFIG.get('figure_height', 12))
    fig, ax = plt.subplots(figsize=figsize)
    
    if clean_labels:
        cm = correlation_matrix.copy()
        def clean_name(name):
            parts = name.split('_')
            if len(parts) > 1:
                metric = METRIC_DISPLAY_NAMES.get(parts[0], parts[0])
                tenant = parts[-1] if 'tenant-' in parts[-1] else ' '.join(parts[1:])
                return f"{metric} ({tenant})"
            return name
            
        cm.columns = [clean_name(col) for col in cm.columns]
        cm.index = [clean_name(idx) for idx in cm.index]
    else:
        cm = correlation_matrix
    
    sns.heatmap(cm, annot=annot, fmt=fmt, cmap=cmap, center=0,
               linewidths=0.5, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
    
    ax.set_title(title)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    fig.tight_layout()
    
    return fig


def plot_metric_multi_tenant_facet(df, metric_name, time_column='experiment_elapsed_seconds', value_column='value',
                                  phases=None, tenants=None, figsize=None, 
                                  use_total_duration=False, total_duration_seconds=None,
                                  show_as_percentage=False, node_config=None):
    """
    Creates a facet grid of plots showing the same metric for different tenants.
    
    Args:
        df (DataFrame): DataFrame with metric data.
        metric_name (str): Name of the metric (key for METRIC_DISPLAY_NAMES).
        time_column (str): Column with time values for the X-axis. Default: 'experiment_elapsed_seconds'.
        value_column (str): Column with metric values.
        phases (list): List of phases to include (None = all).
        tenants (list): List of tenants to include (None = all).
        figsize (tuple): Figure size (optional, uses config if None).
        use_total_duration (bool): If True, the X-axis will use 'total_elapsed_seconds'.
        total_duration_seconds (float): Total duration of the experiment in seconds (required if use_total_duration=True).
        show_as_percentage (bool): If True, display values as percentages of total capacity.
        node_config (dict): Configuration with node resource capacities.
        
    Returns:
        Figure: Matplotlib figure object.
    """
    set_publication_style()
    
    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    
    data = df.copy()
    if phases:
        data = data[data['phase'].isin(phases)]
    if tenants:
        data = data[data['tenant'].isin(tenants)]
    else:
        tenants = sorted(data['tenant'].unique())
    
    n_tenants = len(tenants)
    n_cols = 2
    n_rows = (n_tenants + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (VISUALIZATION_CONFIG.get('figure_width', 16), VISUALIZATION_CONFIG.get('figure_height', 12))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten() if n_tenants > 1 else [axes]
    
    for i, tenant in enumerate(tenants):
        if i >= len(axes):
            break
            
        ax = axes[i]
        tenant_data = data[data['tenant'] == tenant]
        
        for phase, phase_data in tenant_data.groupby('phase'):
            phase_data = phase_data.sort_values(time_column)
            ax.plot(phase_data[time_column], phase_data[value_column], 
                   label=phase, linewidth=1.5)
        
        ax.set_title(f'{tenant}')
        ax.grid(True, alpha=0.3)
        
        if i == len(tenants) - 1:
            ax.legend(title='Phase')
    
    for i in range(n_tenants, len(axes)):
        fig.delaxes(axes[i])
    
    if use_total_duration:
        fig.text(0.5, 0.02, 'Total Experiment Time (seconds)', ha='center', fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
        if total_duration_seconds:
            for ax_item in axes:
                if hasattr(ax_item, 'set_xlim'):
                    ax_item.set_xlim(0, total_duration_seconds)
    else:
        time_unit = "seconds" if "seconds" in time_column else time_column.split('_')[-1]
        fig.text(0.5, 0.02, f'Elapsed time ({time_unit})', ha='center', fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    
    y_axis_label = display_metric_name
    if show_as_percentage:
        unit_info = "%"
        if node_config:
            if metric_name == 'cpu_usage' and 'CPUS' in node_config:
                unit_info = f"% of {node_config['CPUS']} CPU cores"
            elif metric_name == 'memory_usage' and 'MEMORY_GB' in node_config:
                unit_info = f"% of {node_config['MEMORY_GB']} GB memory"
            elif metric_name == 'disk_throughput_total':
                unit_info = "% of 500 MB/s theoretical throughput"
            elif metric_name == 'network_total_bandwidth':
                unit_info = "% of 1 Gbps network interface"
        
        y_axis_label = f"{display_metric_name} ({unit_info})"
    
    fig.text(0.02, 0.5, y_axis_label, va='center', rotation='vertical', fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    fig.suptitle(f'{display_metric_name} over time by Tenant', fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    
    return fig


def plot_impact_score_barplot(impact_score_df, target_tenant=None, figsize=None, palette=None):
    """
    Creates a bar chart for the normalized impact score by tenant.

    Args:
        impact_score_df (DataFrame): DataFrame with impact score. 
                                     Expected columns: ['tenant', 'aggregated_impact_score', 'average_normalized_impact']
                                     Optionally 'round' for facetting.
        target_tenant (str, optional): Specific tenant to highlight in the plot.
        figsize (tuple): Figure size (optional, uses config if None).
        palette (str or list, optional): Color palette for the plot.
    
    Returns:
        Figure: Matplotlib figure object.
    """
    set_publication_style()
    
    if figsize is None:
        figsize = (VISUALIZATION_CONFIG.get('figure_width', 10), VISUALIZATION_CONFIG.get('figure_height', 6))

    fig, ax = plt.subplots(figsize=figsize)

    plot_df = impact_score_df.sort_values(by='aggregated_impact_score', ascending=False)

    bars = sns.barplot(x='tenant', y='aggregated_impact_score', data=plot_df, ax=ax, palette=palette or 'viridis')

    ax.set_ylabel('Aggregated Normalized Impact Score')
    ax.set_xlabel('Tenant')
    
    title = 'Aggregated Impact Score'
    if target_tenant:
        title += f' for {target_tenant}'
    ax.set_title(title)
    
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars.patches:
        ax.text(bar.get_x() + bar.get_width() / 2, 
                bar.get_height(), 
                f'{bar.get_height():.2f}', 
                ha='center', 
                va='bottom' if bar.get_height() >= 0 else 'top',
                fontsize=VISUALIZATION_CONFIG.get('font_size', 10) * 0.8)

    plt.tight_layout()
    return fig


def plot_impact_score_trend(impact_score_df, target_tenant=None, figsize=None, palette=None):
    """
    Creates a line plot for the impact score trend over rounds.

    Args:
        impact_score_df (DataFrame): DataFrame with impact score.
                                     Expected columns: ['tenant', 'round', 'aggregated_impact_score']
        target_tenant (str, optional): Specific tenant to highlight.
        figsize (tuple): Figure size (optional, uses config if None).
        palette (str or list, optional): Color palette for the plot.

    Returns:
        Figure: Matplotlib figure object.
    """
    set_publication_style()

    if figsize is None:
        figsize = (VISUALIZATION_CONFIG.get('figure_width', 10), VISUALIZATION_CONFIG.get('figure_height', 6))
    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(data=impact_score_df, x='round', y='aggregated_impact_score', hue='tenant', 
                 marker='o', ax=ax, palette=palette)

    if target_tenant and target_tenant in impact_score_df['tenant'].unique():
        target_data = impact_score_df[impact_score_df['tenant'] == target_tenant]
        sns.lineplot(data=target_data, x='round', y='aggregated_impact_score', 
                     marker='o', ax=ax, color=TENANT_COLORS.get(target_tenant, 'black'), 
                     linewidth=2.5, label=f'{target_tenant} (Highlighted)')

    ax.set_xlabel('Experiment Round')
    ax.set_ylabel('Aggregated Impact Score')
    ax.set_title('Impact Score Trend by Tenant Over Rounds')
    ax.legend(title='Tenant')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_change_points(df, metric_name, change_points, time_column='elapsed_minutes', value_column='value', figsize=None):
    """
    Creates a line plot for a metric, highlighting detected change points.

    Args:
        df (DataFrame): DataFrame with metric data. Must contain tenant, time, and value columns.
        metric_name (str): Name of the metric (key for METRIC_DISPLAY_NAMES).
        change_points (list): List of timestamps/values in the time column where changes occur.
        time_column (str): Column with time values for the X-axis.
        value_column (str): Column with metric values.
        figsize (tuple): Figure size (optional, uses config if None).

    Returns:
        Figure: Matplotlib figure object.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize or (VISUALIZATION_CONFIG.get('figure_width', 12), VISUALIZATION_CONFIG.get('figure_height', 6)))
    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)

    if 'tenant' in df.columns:
        for tenant, group in df.groupby('tenant'):
            group = group.sort_values(time_column)
            color = TENANT_COLORS.get(tenant, None)
            ax.plot(group[time_column], group[value_column],
                    label=f'{tenant}', linewidth=1.5, color=color, alpha=0.8)
    else:
        df_sorted = df.sort_values(time_column)
        ax.plot(df_sorted[time_column], df_sorted[value_column], 
                label=display_metric_name, linewidth=1.5, alpha=0.8)

    if change_points:
        cp_label_added = False
        for cp_time in change_points:
            label = 'Change Point' if not cp_label_added else None
            ax.axvline(x=cp_time, color='green', linestyle='--', linewidth=1, alpha=0.5, label=label)
            if label:
                cp_label_added = True

    time_unit = time_column.split('_')[-1]
    if time_unit == 'seconds':
        time_unit_label = 'seconds'
    elif time_unit == 'minutes':
        time_unit_label = 'minutes'
    else:
        time_unit_label = time_unit
        
    ax.set_xlabel(f'Time ({time_unit_label})')
    ax.set_ylabel(display_metric_name)
    ax.set_title(f'Detected Change Points in {display_metric_name}')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Legend')
    
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.tight_layout()
    return fig


def create_heatmap(data, title, figsize=None, cmap='viridis'):
    """
    Placeholder for create_heatmap function.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize or (VISUALIZATION_CONFIG.get('figure_width', 10), VISUALIZATION_CONFIG.get('figure_height', 8)))
    ax.set_title(f'Placeholder for Heatmap - {title}')
    ax.text(0.5, 0.5, 'create_heatmap not fully implemented', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    print(f"Placeholder create_heatmap called for {title}")
    return fig


def plot_multivariate_anomalies(df, features, anomaly_column='is_anomaly', figsize=None):
    """
    Placeholder for plot_multivariate_anomalies function.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize or (VISUALIZATION_CONFIG.get('figure_width', 12), VISUALIZATION_CONFIG.get('figure_height', 6)))
    ax.set_title('Placeholder for Multivariate Anomalies Plot')
    ax.text(0.5, 0.5, 'plot_multivariate_anomalies not fully implemented', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    print(f"Placeholder plot_multivariate_anomalies called with features: {features}")
    return fig


if __name__ == '__main__':
    pass
