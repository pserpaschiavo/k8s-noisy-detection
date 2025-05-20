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
        # Allow phase keys or display names for filtering
        phase_keys_to_filter = []
        for p in phases:
            if p in PHASE_DISPLAY_NAMES.values(): # if it's a display name
                phase_keys_to_filter.extend([k for k,v in PHASE_DISPLAY_NAMES.items() if v == p])
            else: # assume it's a key
                phase_keys_to_filter.append(p)
        data = data[data['phase'].isin(phase_keys_to_filter)]

    if tenants:
        data = data[data['tenant'].isin(tenants)]
    
    # Get list of all available tenants in the filtered data
    all_tenants_in_data = data['tenant'].unique().tolist()
    
    # Create figure and axis
    if figsize is None:
        figsize = (VISUALIZATION_CONFIG.get('figure_width', 10), 
                   VISUALIZATION_CONFIG.get('figure_height', 6))
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check for available data types
    has_normalized_data = 'normalized_value' in data.columns
    has_unit_info = 'unit' in data.columns and not data['unit'].isna().all()
    has_formatted_values = 'value_formatted' in data.columns
    has_quota_info = 'quota_limit_formatted' in data.columns
    
    # Determine which column to plot
    y_column_to_plot = value_column
    if use_formatted_values and show_as_percentage and has_normalized_data:
        y_column_to_plot = 'normalized_value'
    
    # Plot lines for each tenant
    for tenant_id in sorted(all_tenants_in_data):
        tenant_data = data[data['tenant'] == tenant_id]
        if tenant_data.empty:
            continue
            
        color = TENANT_COLORS.get(tenant_id, 'gray')
        ax.plot(tenant_data[time_column], tenant_data[y_column_to_plot], 
                label=tenant_id, color=color, linewidth=2)
    
    # Add phase markers if requested
    if show_phase_markers and 'phase_start' in data.columns and 'phase' in data.columns:
        # Get unique phase starts from the *original* df to ensure all markers are potentially drawn
        # then filter by what's actually in the current `data` plot range.
        # This handles cases where filtering might remove the first point of a phase.
        phase_starts_info = df[['phase', 'phase_start']].drop_duplicates().sort_values('phase_start')
        
        # Get min/max time from the currently plotted data for marker relevance
        min_plot_time = data[time_column].min()
        max_plot_time = data[time_column].max()

        for _, row in phase_starts_info.iterrows():
            phase_key = row['phase']
            phase_start_time = row['phase_start']
            phase_display = PHASE_DISPLAY_NAMES.get(phase_key, phase_key)
            
            # Only add if within plot range and not the very first point (unless it's the only phase)
            if phase_start_time >= min_plot_time and phase_start_time <= max_plot_time:
                # Avoid drawing a line at the very beginning if it's the start of the first plotted phase
                # unless it's the only phase being shown or all phases start at the same time.
                is_first_plotted_phase_start = (phase_start_time == min_plot_time)
                
                if not is_first_plotted_phase_start or len(phase_starts_info) == 1 or len(data['phase'].unique()) == 1:
                    ax.axvline(x=phase_start_time, color='black', linestyle='--', alpha=0.7)
                    ax.text(phase_start_time + (max_plot_time - min_plot_time) * 0.01, 
                            ax.get_ylim()[1]*0.95, f' {phase_display}', 
                            rotation=90, verticalalignment='top', horizontalalignment='left')
    
    # Configure axis labels
    x_label = 'Time (seconds)'
    if use_total_duration:
        x_label = 'Total Experiment Time (seconds)'
        if total_duration_seconds:
            ax.set_xlim(0, total_duration_seconds)
    elif "minutes" in time_column:
        x_label = 'Time (minutes)'
    ax.set_xlabel(x_label)
    
    # Determine Y-axis label
    y_axis_label = display_metric_name
    if y_column_to_plot == 'normalized_value':
        if 'normalized_description' in data.columns and not data['normalized_description'].isna().all():
            description = data['normalized_description'].iloc[0]
            y_axis_label = f"{display_metric_name} ({description})"
        else:
            y_axis_label = f"{display_metric_name} (% of Total Capacity)"
    elif has_unit_info and use_formatted_values:
        unit = data['unit'].dropna().iloc[0] if not data['unit'].dropna().empty else "units"
        y_axis_label = f"{display_metric_name} ({unit})"
    elif has_quota_info and use_formatted_values:
        quota_info = data['quota_limit_formatted'].dropna().iloc[0] if not data['quota_limit_formatted'].dropna().empty else "quota"
        if y_column_to_plot == 'normalized_value': # Should be caught by first condition
            y_axis_label = f"{display_metric_name} (% of {quota_info})"
        else:
            y_axis_label = f"{display_metric_name} (Units Relative to {quota_info})"
    ax.set_ylabel(y_axis_label)
    
    # Add legend and title
    title_parts = [display_metric_name]
    if phases and len(phases) == 1:
        # Use display name if a single phase is specified
        single_phase_display = PHASE_DISPLAY_NAMES.get(phases[0], phases[0])
        title_parts.append(f"- Phase: {single_phase_display}")
    
    # Add formatted value info to title if applicable
    if use_formatted_values and has_formatted_values and y_column_to_plot == 'value':
        if not data[y_column_to_plot].empty:
            max_idx = data[y_column_to_plot].idxmax()
            if max_idx is not None and 'value_formatted' in data.columns and pd.notna(data.loc[max_idx, 'value_formatted']):
                max_formatted = data.loc[max_idx, 'value_formatted']
                title_parts.append(f"(Max: {max_formatted})")
    
    ax.set_title(" ".join(title_parts))
    ax.legend(title='Tenant')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    
    return fig


def plot_phase_comparison(df, metric_name, value_column='mean', 
                         error_column=None, figsize=None,
                         show_as_percentage=False, node_config=None):
    """
    Creates a bar chart comparing different phases for each tenant.
    
    Args:
        df (DataFrame): DataFrame with data aggregated by phase and tenant.
                        Expected columns: 'tenant', 'phase' (or 'phase_name'), value_column.
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
    
    # Use 'phase_display' if available, otherwise 'phase_name' or 'phase'
    phase_col_options = ['phase_display', 'phase_name', 'phase']
    phase_col_to_use = next((col for col in phase_col_options if col in df.columns), None)

    if not phase_col_to_use:
        raise ValueError(f"DataFrame must contain one of the phase columns: {phase_col_options}")

    # Ensure phase order is correct using PHASE_DISPLAY_NAMES values
    ordered_phases = [PHASE_DISPLAY_NAMES[k] for k in sorted(PHASE_DISPLAY_NAMES.keys())]
    
    # Filter df to include only phases present in ordered_phases and pivot
    # This also ensures the columns in pivot table are in the desired order.
    pivot = df[df[phase_col_to_use].isin(ordered_phases)].pivot_table(
        index='tenant', 
        columns=phase_col_to_use, 
        values=value_column
    )[ordered_phases] # Reorder columns according to ordered_phases
    
    yerr = None
    if error_column and error_column in df.columns:
        error_pivot = df[df[phase_col_to_use].isin(ordered_phases)].pivot_table(
            index='tenant', 
            columns=phase_col_to_use, 
            values=error_column
        )[ordered_phases]
        yerr = error_pivot.reindex(columns=pivot.columns).values.T # Ensure yerr matches pivot columns and order
    
    # Define colors based on PHASE_DISPLAY_NAMES order if possible, or use a colormap
    # This part needs careful handling if not all phases are present or if using a generic colormap
    phase_colors = [VISUALIZATION_CONFIG.get('phase_colors', {}).get(p, '#CCCCCC') for p in pivot.columns]
    if len(phase_colors) != len(pivot.columns):
        # Fallback if color mapping is incomplete
        cmap = LinearSegmentedColormap.from_list('phases_fallback', ['#4C78A8', '#F58518', '#72B7B2'])
        colors_to_plot = cmap
    else:
        colors_to_plot = phase_colors

    pivot.plot(kind='bar', ax=ax, yerr=yerr, capsize=5, width=0.8, color=colors_to_plot)
    
    ax.set_xlabel('Tenant')
    
    y_axis_label = display_metric_name
    title = f'Comparison of {display_metric_name} by Tenant and Phase'
    if show_as_percentage:
        unit_info = "% of Total"
        if node_config:
            if metric_name == 'cpu_usage' and 'CPUS' in node_config:
                unit_info = f"% of {node_config['CPUS']} CPU Cores"
            elif metric_name == 'memory_usage' and 'MEMORY_GB' in node_config:
                unit_info = f"% of {node_config['MEMORY_GB']} GB Memory"
            elif metric_name == 'disk_throughput_total':
                unit_info = "% of Theoretical Max Throughput"
            elif metric_name == 'network_total_bandwidth':
                unit_info = "% of 1 Gbps Bandwidth"
        y_axis_label = f"{display_metric_name} ({unit_info})"
        title = f'Comparison of {display_metric_name} by Tenant and Phase ({unit_info})'
    
    ax.set_ylabel(y_axis_label)
    ax.set_title(title)
    
    ax.legend(title='Phase')
    ax.grid(axis='y', alpha=0.3)
    
    # Add bar labels
    for container in ax.containers:
        if isinstance(container, matplotlib.container.BarContainer):
            fmt_str = '%.1f%%' if show_as_percentage else '%.2f'
            ax.bar_label(container, fmt=fmt_str, padding=3, fontsize=10)
    
    plt.xticks(rotation=45, ha="right")
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
                                   This influences the title more than the data processing here.
        node_config (dict): Configuration with node resource capacities (for title enrichment).
        
    Returns:
        Figure: Matplotlib figure object.
    """
    set_publication_style()
    
    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    
    if figsize is None:
        figsize = (VISUALIZATION_CONFIG.get('figure_width', 12), VISUALIZATION_CONFIG.get('figure_height', 8))
    fig, ax = plt.subplots(figsize=figsize)
    
    # Pivot if 'round' column exists and has multiple unique values
    if 'round' in impact_df.columns and impact_df['round'].nunique() > 1:
        pivot_data = impact_df.pivot_table(index='tenant', columns='round', values=value_column)
        x_label = 'Round'
    else:
        # Ensure 'tenant' is the index for single-round or aggregated data
        pivot_data = impact_df.set_index('tenant')[[value_column]]
        x_label = '' # No x-axis label if not by round
    
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap=cmap, center=0,
                linewidths=.5, ax=ax, cbar_kws={'label': 'Impact (%)'})
    
    title_metric_part = display_metric_name
    if show_as_percentage: # This flag is more about interpreting the input value_column
        if node_config and metric_name in ['cpu_usage', 'memory_usage', 'disk_throughput_total', 'network_total_bandwidth']:
            if metric_name == 'cpu_usage' and 'CPUS' in node_config:
                title_metric_part += f" (% of {node_config['CPUS']} CPU Cores)"
            elif metric_name == 'memory_usage' and 'MEMORY_GB' in node_config:
                title_metric_part += f" (% of {node_config['MEMORY_GB']} GB)"
            elif metric_name == 'disk_throughput_total':
                title_metric_part += " (% of 500 MB/s)" # Example, make this configurable
            elif metric_name == 'network_total_bandwidth':
                title_metric_part += " (% of 1 Gbps)" # Example, make this configurable
    
    ax.set_title(f'Impact on {title_metric_part} During Attack Phase')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Tenant')
    
    plt.tight_layout()
    return fig


def plot_recovery_effectiveness(recovery_df, metric_name, figsize=None, 
                                baseline_phase_key="baseline", 
                                attack_phase_key="attack", 
                                recovery_phase_key="recovery",
                                show_as_percentage=False, node_config=None):
    """
    Creates a bar chart showing recovery effectiveness.
    Uses PHASE_DISPLAY_NAMES to get display names for phases from keys.
    
    Args:
        recovery_df (DataFrame): DataFrame with recovery data. 
                                 Must contain columns for mean values of each phase (e.g., 'mean_baseline'), 
                                 'tenant', and 'recovery_percent'.
        metric_name (str): Name of the metric (key for METRIC_DISPLAY_NAMES).
        figsize (tuple): Figure size (optional, uses config if None).
        baseline_phase_key (str): Key in recovery_df for baseline phase mean values (e.g., 'mean_baseline').
        attack_phase_key (str): Key in recovery_df for attack phase mean values (e.g., 'mean_attack').
        recovery_phase_key (str): Key in recovery_df for recovery phase mean values (e.g., 'mean_recovery').
        show_as_percentage (bool): If True, display primary y-axis values as percentages of total capacity.
        node_config (dict): Configuration with node resource capacities.
                                             
    Returns:
        Figure: Matplotlib figure object.
    """
    set_publication_style()
    
    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)

    # Get display names for phases from config, fallback to keys if not found
    baseline_display = PHASE_DISPLAY_NAMES.get(baseline_phase_key.split('_')[-1].capitalize(), baseline_phase_key) 
    attack_display = PHASE_DISPLAY_NAMES.get(attack_phase_key.split('_')[-1].capitalize(), attack_phase_key)
    recovery_display = PHASE_DISPLAY_NAMES.get(recovery_phase_key.split('_')[-1].capitalize(), recovery_phase_key)

    # Ensure required columns are present (using the provided keys for mean values)
    required_cols = [baseline_phase_key, attack_phase_key, recovery_phase_key, 'tenant', 'recovery_percent']
    for col in required_cols:
        if col not in recovery_df.columns:
            # Try to find a column that ends with the key (e.g. mean_baseline for baseline_phase_key='baseline')
            # This is a bit fragile and assumes a naming convention like 'mean_<phase_key>'
            potential_col = next((c for c in recovery_df.columns if c.endswith(f'_{col}') or c == col), None)
            if not potential_col:
                 raise KeyError(f"Column for phase '{col}' not found in recovery_df. Expected one of: {col} or ending with _{col}. Available: {recovery_df.columns.tolist()}")
            # Update the key to the found column name for data access
            if col == baseline_phase_key: baseline_phase_key = potential_col
            elif col == attack_phase_key: attack_phase_key = potential_col
            elif col == recovery_phase_key: recovery_phase_key = potential_col
            # Re-check after potential update
            if potential_col not in recovery_df.columns:
                 raise KeyError(f"Column '{potential_col}' (derived from '{col}') still not found. Available: {recovery_df.columns.tolist()}")

    if figsize is None:
        figsize = (VISUALIZATION_CONFIG.get('figure_width', 12), VISUALIZATION_CONFIG.get('figure_height', 7))
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Ensure tenants are sorted for consistent plotting if data isn't pre-sorted
    plot_data = recovery_df.sort_values('tenant').reset_index(drop=True)
    tenants = plot_data['tenant'].unique() # Get unique tenants from sorted data
    x = np.arange(len(tenants))
    width = 0.25

    # Use the (potentially updated) phase keys to access data
    rects1 = ax1.bar(x - width, plot_data[baseline_phase_key], width, label=baseline_display, color='#4C78A8')
    rects2 = ax1.bar(x, plot_data[attack_phase_key], width, label=attack_display, color='#F58518')
    rects3 = ax1.bar(x + width, plot_data[recovery_phase_key], width, label=recovery_display, color='#72B7B2')
    
    ax1.set_xlabel('Tenant')
    
    y_label_main = f'Mean Value - {display_metric_name}'
    if show_as_percentage:
        unit_info = "%"
        if node_config:
            if metric_name == 'cpu_usage' and 'CPUS' in node_config:
                unit_info = f"% of {node_config['CPUS']} CPU Cores"
            elif metric_name == 'memory_usage' and 'MEMORY_GB' in node_config:
                unit_info = f"% of {node_config['MEMORY_GB']} GB Memory"
            elif metric_name == 'disk_throughput_total':
                unit_info = "% of 500 MB/s Theoretical Throughput"
            elif metric_name == 'network_total_bandwidth':
                unit_info = "% of 1 Gbps Network Interface"
        y_label_main = f'Mean Value - {display_metric_name} ({unit_info})'
    ax1.set_ylabel(y_label_main)
    
    ax1.set_title(f'Recovery Effectiveness - {display_metric_name}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tenants, rotation=45, ha="right")
    ax1.legend(loc='upper left', title='Phase')
    ax1.grid(axis='y', alpha=0.3)
    
    # Twin axis for recovery percentage
    ax2 = ax1.twinx()
    recovery_percentages = plot_data['recovery_percent']
    # Ensure colors match positive/negative recovery
    bar_colors = ['#72B7B2' if val >= 0 else '#E15759' for val in recovery_percentages] # Green for positive, Red for negative
    
    # Plot recovery percentage bars. Ensure alignment with tenant groups.
    # The x positions for these bars should align with the tenant groups on ax1.
    # Since ax2 shares x-axis, using 'x' directly is correct.
    ax2.bar(x, recovery_percentages, width=width*0.8, alpha=0.35, 
            color=bar_colors, label='Recovery %') # Reduced width slightly, adjusted alpha
    
    ax2.set_ylabel('Recovery Percentage (%)')
    # Optional: Add a horizontal line at 0% or 100% for reference
    ax2.axhline(0, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)
    ax2.axhline(100, color='blue', linestyle=':', linewidth=0.7, alpha=0.5) # e.g. for 100% recovery
    
    # Add labels to recovery percentage bars
    for i, val in enumerate(recovery_percentages):
        ax2.text(x[i], val + (5 if val >=0 else -10), f'{val:.1f}%', 
                 ha='center', va='bottom' if val >=0 else 'top', fontsize=9, color='black')

    # Ensure legend for ax2 is handled if needed, or integrated with ax1's legend
    # For simplicity, if only one item in ax2 legend, it might not be necessary or can be described in title/caption.
    # handles1, labels1 = ax1.get_legend_handles_labels()
    # handles2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(handles1 + handles2, labels1 + labels2, loc='best', title='Legend')

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(correlation_matrix, title='Correlation Heatmap',
                            figsize=None, cmap='vlag', clean_labels=True, annot=True,
                            fmt='.2f', cbar_label='Correlation Coefficient'):
    """
    Creates a correlation heatmap. Can also be used for covariance matrices by adjusting title and cbar_label.
    Uses a colorblind-friendly diverging palette (vlag - Blue-White-Red).
    
    Args:
        correlation_matrix (DataFrame): Matrix to plot (correlation or covariance).
        title (str): Plot title.
        figsize (tuple): Figure size (optional, uses config if None).
        cmap (str): Colormap name for the heatmap.
        clean_labels (bool): If True, clean metric_tenant names to more readable format.
        annot (bool): If True, write the data value in each cell.
        fmt (str): String formatting code to use when adding annotations.
        cbar_label (str): Label for the color bar.
        
    Returns:
        Figure: Matplotlib figure object.
    """
    set_publication_style()
    
    if figsize is None:
        figsize = (VISUALIZATION_CONFIG.get('figure_width', 14), VISUALIZATION_CONFIG.get('figure_height', 12))
    fig, ax = plt.subplots(figsize=figsize)
    
    # Apply label cleaning if requested
    matrix_to_plot = correlation_matrix.copy()
    if clean_labels:
        def clean_name(name):
            # Attempt to split by common delimiters or patterns
            parts = name.replace('_', ' ').replace('-', ' ').split()
            # Heuristic: find metric display name and tenant part
            # This is a simplified approach and might need refinement based on actual label formats
            cleaned_parts = []
            potential_metric = ""
            tenant_id = ""
            for i, p_part in enumerate(parts):
                # Check if part (or combination with previous) is a known metric key
                current_check = "_".join(parts[:i+1]) # e.g. cpu, cpu_usage
                if current_check in METRIC_DISPLAY_NAMES:
                    potential_metric = METRIC_DISPLAY_NAMES[current_check]
                elif p_part.lower().startswith("tenant") and len(parts) > i+1:
                    tenant_id = " ".join(parts[i:]) # tenant-a, tenant b
                    break # Assume rest is tenant ID
                elif i == 0: # First part, could be a metric or part of it
                    potential_metric = METRIC_DISPLAY_NAMES.get(p_part, p_part.capitalize())
            
            if potential_metric and tenant_id:
                return f"{potential_metric} ({tenant_id})"
            elif potential_metric:
                return potential_metric
            # Fallback if specific parsing fails
            return name.replace('_', ' ').title()
            
        matrix_to_plot.columns = [clean_name(col) for col in matrix_to_plot.columns]
        matrix_to_plot.index = [clean_name(idx) for idx in matrix_to_plot.index]
    
    sns.heatmap(matrix_to_plot, annot=annot, fmt=fmt, cmap=cmap, center=0,
               linewidths=0.5, ax=ax, cbar_kws={'label': cbar_label})
    
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
    Each facet (subplot) represents a tenant, and lines within the subplot represent phases.
    
    Args:
        df (DataFrame): DataFrame with metric data.
        metric_name (str): Name of the metric (key for METRIC_DISPLAY_NAMES).
        time_column (str): Column with time values for the X-axis. Default: 'experiment_elapsed_seconds'.
        value_column (str): Column with metric values.
        phases (list): List of phase keys or display names to include (None = all).
        tenants (list): List of tenants to include (None = all).
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
    # Filter by phases (accepts keys or display names)
    if phases:
        phase_keys_to_filter = []
        for p in phases:
            if p in PHASE_DISPLAY_NAMES.values(): # if it's a display name
                phase_keys_to_filter.extend([k for k,v in PHASE_DISPLAY_NAMES.items() if v == p])
            else: # assume it's a key
                phase_keys_to_filter.append(p)
        data = data[data['phase'].isin(phase_keys_to_filter)]

    # Determine tenants to plot
    if tenants:
        data = data[data['tenant'].isin(tenants)]
        tenants_to_plot = sorted([t for t in tenants if t in data['tenant'].unique()])
    else:
        tenants_to_plot = sorted(data['tenant'].unique())
    
    if not tenants_to_plot:
        print(f"No data to plot for metric {metric_name} with specified filters.")
        # Return an empty figure or handle error appropriately
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data to plot", ha='center', va='center')
        return fig

    n_tenants = len(tenants_to_plot)
    n_cols = VISUALIZATION_CONFIG.get('facet_num_cols', 2) # Default to 2 columns
    n_rows = (n_tenants + n_cols - 1) // n_cols
    
    if figsize is None:
        # Adjust figsize based on number of rows/cols for better readability
        base_w, base_h = VISUALIZATION_CONFIG.get('figure_width', 16), VISUALIZATION_CONFIG.get('figure_height_per_facet_row', 5)
        figsize = (base_w, base_h * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    axes = np.array(axes).flatten() # Ensure axes is always a flat array for easy iteration
    
    # Get ordered phase display names for consistent legend and plotting
    ordered_phase_displays = [PHASE_DISPLAY_NAMES[k] for k in sorted(PHASE_DISPLAY_NAMES.keys())]
    phase_color_map = VISUALIZATION_CONFIG.get('phase_colors', {})

    for i, tenant_id in enumerate(tenants_to_plot):
        if i >= len(axes):
            break # Should not happen if n_rows, n_cols are calculated correctly
            
        ax = axes[i]
        tenant_data = data[data['tenant'] == tenant_id]
        
        # Plot data for each phase for the current tenant
        for phase_key in sorted(PHASE_DISPLAY_NAMES.keys()): # Iterate in defined order
            phase_display_name = PHASE_DISPLAY_NAMES[phase_key]
            phase_data_for_plot = tenant_data[tenant_data['phase'] == phase_key]
            if not phase_data_for_plot.empty:
                phase_data_for_plot = phase_data_for_plot.sort_values(time_column)
                color = phase_color_map.get(phase_display_name, None) # Use phase-specific color
                ax.plot(phase_data_for_plot[time_column], phase_data_for_plot[value_column], 
                       label=phase_display_name, linewidth=1.5, color=color)
        
        ax.set_title(f'{tenant_id}')
        ax.grid(True, alpha=0.3)
        
        # Add legend to the last plotted axis or a dedicated legend area if complex
        if i == 0: # Add legend to the first plot, or handle globally later
            handles, labels = ax.get_legend_handles_labels()
            # Ensure legend order matches `ordered_phase_displays`
            ordered_handles_labels = sorted(zip(handles, labels), key=lambda x: ordered_phase_displays.index(x[1]) if x[1] in ordered_phase_displays else -1)
            if ordered_handles_labels:
                h_sorted, l_sorted = zip(*ordered_handles_labels)
                fig.legend(h_sorted, l_sorted, title='Phase', loc='upper right', bbox_to_anchor=(0.99, 0.98))
    
    # Remove unused subplots
    for j in range(n_tenants, len(axes)):
        fig.delaxes(axes[j])
    
    # Common X-axis label
    x_axis_label = f'Elapsed Time ({time_column.split("_")[-1].capitalize()})'
    if use_total_duration:
        x_axis_label = 'Total Experiment Time (seconds)'
        if total_duration_seconds:
            for ax_item in axes[:n_tenants]: # Only set xlim for used axes
                ax_item.set_xlim(0, total_duration_seconds)
    fig.supxlabel(x_axis_label, fontsize=VISUALIZATION_CONFIG.get('label_size', 14))

    # Common Y-axis label
    y_axis_label = display_metric_name
    if show_as_percentage:
        unit_info = "%"
        if node_config:
            # (Logic for unit_info based on metric_name and node_config as in other functions)
            if metric_name == 'cpu_usage' and 'CPUS' in node_config: unit_info = f"% of {node_config['CPUS']} CPU Cores"
            elif metric_name == 'memory_usage' and 'MEMORY_GB' in node_config: unit_info = f"% of {node_config['MEMORY_GB']} GB Memory"
            # ... other metrics
        y_axis_label = f"{display_metric_name} ({unit_info})"
    fig.supylabel(y_axis_label, fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    
    # Super title for the entire figure
    fig.suptitle(f'{display_metric_name} Over Time by Tenant and Phase', fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95]) # Adjust rect to make space for suptitle, suplabels, and legend
    
    return fig


def plot_impact_score_barplot(impact_score_df, score_col='normalized_impact_score', tenant_col='tenant', target_tenant=None, figsize=None, palette=None):
    """
    Creates a bar chart for the impact score by tenant.
    The title was previously "Aggregated Impact Score", now simplified to "Impact Score".
    Y-axis was "Aggregated Normalized Impact Score", now "Normalized Impact Score".

    Args:
        impact_score_df (DataFrame): DataFrame with impact score. 
                                     Expected columns: [tenant_col, score_col]
        score_col (str): Name of the column containing the score to plot.
        tenant_col (str): Name of the column containing the tenant identifier.
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

    plot_df = impact_score_df.sort_values(by=score_col, ascending=False)

    # Use tenant_col for x-axis and also for hue to assign different colors if no specific palette is given
    # and then hide the legend as it would be redundant.
    # If a specific palette is provided, it will be used.
    # If target_tenant is specified, we might want to adjust colors.
    
    # Determine colors: default, or highlight target_tenant
    colors = None
    if palette:
        colors = palette
    elif target_tenant:
        colors = [TENANT_COLORS.get(t, '#CCCCCC') if t == target_tenant else '#AAAAAA' 
                  for t in plot_df[tenant_col]]
    else: # Default behavior if no palette and no target_tenant, use TENANT_COLORS or a default seaborn palette
        # Check if all tenants in plot_df are in TENANT_COLORS
        if all(t in TENANT_COLORS for t in plot_df[tenant_col]):
            colors = [TENANT_COLORS[t] for t in plot_df[tenant_col]]
        else: # Fallback to a seaborn palette if not all tenants have defined colors
            colors = sns.color_palette(n_colors=len(plot_df[tenant_col]))

    bars = sns.barplot(x=tenant_col, y=score_col, data=plot_df, ax=ax, 
                       palette=colors, hue=tenant_col, legend=False) # hue for individual coloring, legend off

    ax.set_ylabel('Normalized Impact Score') 
    ax.set_xlabel('Tenant') 
    
    title = 'Impact Score by Tenant' # Simplified title
    # if target_tenant: # This part might be redundant if highlighting is done by color primarily
    #     title += f' (Highlighting {target_tenant})'
    ax.set_title(title)
    
    ax.tick_params(axis='x', rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels on bars
    for bar in bars.patches:
        ax.text(bar.get_x() + bar.get_width() / 2, 
                bar.get_height(), 
                f'{bar.get_height():.2f}', 
                ha='center', 
                va='bottom' if bar.get_height() >= 0 else 'top',
                fontsize=VISUALIZATION_CONFIG.get('font_size', 10) * 0.8)

    plt.tight_layout()
    return fig


def plot_impact_score_trend(impact_score_df, score_col='normalized_impact_score', round_col='round', tenant_col='tenant', target_tenant=None, figsize=None, palette=None):
    """
    Creates a line plot for the impact score trend over rounds.
    Y-axis was "Aggregated Impact Score", now "Impact Score".

    Args:
        impact_score_df (DataFrame): DataFrame with impact score.
                                     Expected columns: [tenant_col, round_col, score_col]
        score_col (str): Name of the column containing the score.
        round_col (str): Name of the column containing the round identifier.
        tenant_col (str): Name of the column containing the tenant identifier.
        target_tenant (str, optional): Specific tenant to highlight.
        figsize (tuple): Figure size (optional, uses config if None).
        palette (dict or list, optional): Color palette for tenants.

    Returns:
        Figure: Matplotlib figure object.
    """
    set_publication_style()

    if figsize is None:
        figsize = (VISUALIZATION_CONFIG.get('figure_width', 10), VISUALIZATION_CONFIG.get('figure_height', 6))
    fig, ax = plt.subplots(figsize=figsize)

    # Use a palette that maps tenants to colors, falling back to TENANT_COLORS
    # If palette is None, sns.lineplot will use its default.
    # If palette is a list, it will cycle through it.
    # If palette is a dict, it will map tenant names to colors.
    final_palette = palette
    if not final_palette:
        # Create a dict from TENANT_COLORS for tenants present in the data
        present_tenants = impact_score_df[tenant_col].unique()
        final_palette = {t: TENANT_COLORS.get(t) for t in present_tenants if TENANT_COLORS.get(t)}
        if not final_palette: # If still empty (e.g. no tenants in TENANT_COLORS)
            final_palette = None # Let seaborn choose

    sns.lineplot(data=impact_score_df, x=round_col, y=score_col, hue=tenant_col, 
                 marker='o', ax=ax, palette=final_palette)

    if target_tenant and target_tenant in impact_score_df[tenant_col].unique():
        target_data = impact_score_df[impact_score_df[tenant_col] == target_tenant]
        # Ensure the highlighted line stands out
        highlight_color = TENANT_COLORS.get(target_tenant, 'black') # Default to black if not in TENANT_COLORS
        sns.lineplot(data=target_data, x=round_col, y=score_col, 
                     marker='o', ax=ax, color=highlight_color, 
                     linewidth=2.5, label=f'{target_tenant} (Highlighted)', zorder=10)

    ax.set_xlabel('Experiment Round')
    ax.set_ylabel('Impact Score') # Simplified Y-axis label
    ax.set_title('Impact Score Trend by Tenant Over Rounds')
    
    # Improve legend handling: combine and remove duplicates if target_tenant was plotted separately
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Tenant')
    
    ax.grid(True, alpha=0.3)
    # Ensure rounds are treated as categorical or discrete if they are not numeric sequence
    if impact_score_df[round_col].dtype == 'object':
        ax.set_xticks(impact_score_df[round_col].unique()) # Show all round names if they are strings
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # For numeric rounds, ensure integer ticks

    plt.tight_layout()
    return fig


def create_heatmap(data, title, figsize=None, cmap='viridis', cbar_label='Value'): # Added cbar_label
    """
    Generic heatmap plotting function.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize or (VISUALIZATION_CONFIG.get('figure_width', 10), VISUALIZATION_CONFIG.get('figure_height', 8)))
    
    if not isinstance(data, pd.DataFrame):
        print("Warning: Heatmap data is not a DataFrame. Attempting to convert.")
        try:
            data = pd.DataFrame(data)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: Could not convert data to DataFrame for heatmap.\n{e}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red')
            return fig
            
    if data.empty:
        ax.text(0.5, 0.5, 'No data provided for heatmap.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    else:
        sns.heatmap(data, annot=True, fmt=".2f", cmap=cmap, ax=ax, cbar_kws={'label': cbar_label})
    
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_entropy_heatmap(entropy_results_df, metric_name, round_name, output_path):
    """
    Generates a heatmap of entropy (mutual information) results between tenant pairs.

    Args:
        entropy_results_df (pd.DataFrame): DataFrame with entropy results. 
                                           Expected columns: 'tenant1', 'tenant2', 'mutual_information'.
        metric_name (str): The name of the metric for which entropy was calculated.
        round_name (str): The name of the round/experiment or phase identifier.
        output_path (str): Path to save the generated heatmap.
    """
    set_publication_style()
    
    if entropy_results_df.empty:
        print(f"No entropy data to plot for heatmap (Metric: {metric_name}, Context: {round_name}).")
        return

    heatmap_data = entropy_results_df.pivot(index='tenant1', columns='tenant2', values='mutual_information')
    heatmap_data = heatmap_data.fillna(0)

    all_tenants = sorted(list(set(entropy_results_df['tenant1']).union(set(entropy_results_df['tenant2']))))
    heatmap_data = heatmap_data.reindex(index=all_tenants, columns=all_tenants, fill_value=0)
    
    # Symmetrize the matrix as MI(A,B) = MI(B,A)
    for i in range(len(all_tenants)):
        for j in range(i + 1, len(all_tenants)):
            t1, t2 = all_tenants[i], all_tenants[j]
            val = heatmap_data.loc[t1, t2] if t1 in heatmap_data.index and t2 in heatmap_data.columns else heatmap_data.loc[t2,t1]
            heatmap_data.loc[t1, t2] = heatmap_data.loc[t2, t1] = val if pd.notna(val) else 0
                
    plt.figure(figsize=(VISUALIZATION_CONFIG.get('figure_width', 10), 
                        VISUALIZATION_CONFIG.get('figure_height', 8)))
    
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", # Viridis is good for non-negative MI
                linewidths=.5, cbar_kws={'label': 'Mutual Information'})
    
    display_metric = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    # Context can be a round name or a phase description like "Aggregated - Attack Phase"
    context_description = round_name.replace("phase_", "").replace("all_aggregated", "All Phases").replace("_", " ").title()
    if "Round" not in context_description and not any(p.lower() in context_description.lower() for p in ["baseline", "attack", "recovery", "all phases"]):
        context_description = f"Round: {context_description}"
        
    title = f'Mutual Information between Tenant Pairs\nMetric: {display_metric} - {context_description}'
    plt.title(title)
    plt.xlabel('Tenant B') # Changed from Tenant 2
    plt.ylabel('Tenant A') # Changed from Tenant 1
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
        print(f"Entropy heatmap saved to {output_path}")
    except Exception as e:
        print(f"Error saving entropy heatmap to {output_path}: {e}")
    plt.close()


def plot_entropy_top_pairs_barplot(entropy_results_df, metric_name, round_name, output_path, top_n=6):
    """
    Generates a bar plot of the top N tenant pairs with the highest mutual information.

    Args:
        entropy_results_df (pd.DataFrame): DataFrame with entropy results.
                                           Expected columns: 'tenant1', 'tenant2', 'mutual_information'.
        metric_name (str): The name of the metric for which entropy was calculated.
        round_name (str): The name of the round/experiment or phase identifier.
        output_path (str): Path to save the generated bar plot.
        top_n (int): Number of top pairs to display.
    """
    set_publication_style()
    
    if entropy_results_df.empty:
        print(f"No entropy data to plot for top pairs barplot (Metric: {metric_name}, Context: {round_name}).")
        return

    # Create a canonical representation of the pair (sorted tuple) to avoid duplicates
    entropy_results_df['pair_tuple'] = entropy_results_df.apply(
        lambda row: tuple(sorted((str(row['tenant1']), str(row['tenant2'])))), axis=1
    )
    # Sort by mutual information, drop duplicates by canonical pair, then take top N
    top_pairs = entropy_results_df.sort_values('mutual_information', ascending=False)\
                                  .drop_duplicates(subset=['pair_tuple'])\
                                  .head(top_n)
    
    if top_pairs.empty:
        print(f"No unique top pairs to plot for (Metric: {metric_name}, Context: {round_name}).")
        return

    plt.figure(figsize=(VISUALIZATION_CONFIG.get('figure_width', 12), 
                        VISUALIZATION_CONFIG.get('figure_height_per_item', 0.5) * len(top_pairs)))
    
    # Create string labels for pairs for the y-axis
    top_pairs['pair_label'] = top_pairs['pair_tuple'].apply(lambda p: f"{p[0]} - {p[1]}")
    
    # Sort by MI for plotting (descending for horizontal bar plot)
    top_pairs_sorted = top_pairs.sort_values('mutual_information', ascending=True)

    barplot = sns.barplot(x='mutual_information', y='pair_label', hue='pair_label', 
                          data=top_pairs_sorted, palette='mako_r', orient='h', legend=False) # mako_r for darker high values
    
    plt.xlabel('Mutual Information')
    plt.ylabel('Tenant Pair')
    
    display_metric = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    context_description = round_name.replace("phase_", "").replace("all_aggregated", "All Phases").replace("_", " ").title()
    if "Round" not in context_description and not any(p.lower() in context_description.lower() for p in ["baseline", "attack", "recovery", "all phases"]):
        context_description = f"Round: {context_description}"

    title = f'Top {len(top_pairs_sorted)} Tenant Pairs by Mutual Information\nMetric: {display_metric} - {context_description}'
    plt.title(title)
    
    # Add text labels for values on bars
    for index, row in top_pairs_sorted.iterrows():
        value = row['mutual_information']
        # Find the y-position of the bar. This can be tricky if not just using index.
        # For horizontal barplot with categorical y, index usually works.
        y_pos = list(top_pairs_sorted['pair_label']).index(row['pair_label'])
        barplot.text(value + (top_pairs_sorted['mutual_information'].max() * 0.01), # Small offset from bar end
                     y_pos, 
                     f'{value:.3f}', 
                     color='black', ha="left", va="center", fontsize=10)

    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
        print(f"Entropy top pairs barplot saved to {output_path}")
    except Exception as e:
        print(f"Error saving entropy top pairs barplot to {output_path}: {e}")
    plt.close()


if __name__ == '__main__':
    pass
