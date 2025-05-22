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


def plot_scatter_comparison(
    data_dict: dict, 
    x_label: str, 
    y_label: str, 
    title: str, 
    output_dir: str, 
    filename: str,
    add_identity_line: bool = True,
    annotate_points: bool = False,
    annotation_subset: list = None 
):
    """
    Generates a scatter plot comparing one or more pairs of data series.

    Args:
        data_dict (dict): Dictionary where keys are labels for the legend and 
                          values are tuples of (pd.Series_x, pd.Series_y).
                          Example: {"Method1 vs Method2": (series_x1, series_y1), ...}
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
        output_dir (str): Directory to save the plot.
        filename (str): Filename for the saved plot.
        add_identity_line (bool): If True, adds a y=x identity line.
        annotate_points (bool): If True, annotates points (can be slow for many points).
        annotation_subset (list, optional): A subset of indices from x_data/y_data to annotate.
                                            If None and annotate_points is True, all points are attempted.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('scatter_figsize', (8, 8)))

    all_x_values = []
    all_y_values = []

    for label, (x_data_orig, y_data_orig) in data_dict.items():
        x_data = pd.Series(x_data_orig) if not isinstance(x_data_orig, pd.Series) else x_data_orig.copy()
        y_data = pd.Series(y_data_orig) if not isinstance(y_data_orig, pd.Series) else y_data_orig.copy()

        common_index = x_data.index.intersection(y_data.index)
        if common_index.empty and (len(x_data) == len(y_data)):
            x_data.index = pd.RangeIndex(len(x_data))
            y_data.index = pd.RangeIndex(len(y_data))
            common_index = x_data.index
        elif common_index.empty:
             print(f"Warning: No common data points or mismatched lengths for '{label}' in scatter plot '{title}'. Skipping this series pair.")
             continue

        x_data = x_data.loc[common_index]
        y_data = y_data.loc[common_index]

        if x_data.empty or y_data.empty:
            print(f"Warning: No data to plot for series pair '{label}' in scatter plot '{title}'. Skipping.")
            continue

        sns.scatterplot(x=x_data, y=y_data, ax=ax, 
                        s=VISUALIZATION_CONFIG.get('scatter_marker_size', 50), 
                        alpha=VISUALIZATION_CONFIG.get('scatter_marker_alpha', 0.7),
                        edgecolor=VISUALIZATION_CONFIG.get('scatter_marker_edgecolor', 'k'),
                        linewidth=VISUALIZATION_CONFIG.get('scatter_marker_linewidth', 0.5),
                        label=label if len(data_dict) > 1 else None)
        
        all_x_values.extend(x_data.tolist())
        all_y_values.extend(y_data.tolist())

    ax.set_xlabel(x_label, fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    ax.set_ylabel(y_label, fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    ax.set_title(title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))

    if add_identity_line:
        all_values = pd.Series(all_x_values + all_y_values).dropna()
        if not all_values.empty:
            min_val = all_values.min()
            max_val = all_values.max()
            buffer = (max_val - min_val) * 0.05 if (max_val - min_val) > 1e-6 else 0.1
            lims = [min_val - buffer, max_val + buffer]
            
            ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='y=x (Identity)')
            ax.set_xlim(lims)
            ax.set_ylim(lims)
        else:
            print("Warning: Cannot draw identity line as no valid data points found across all series.")

    if len(data_dict) > 1 or (add_identity_line and all_x_values and all_y_values):
        ax.legend(fontsize=VISUALIZATION_CONFIG.get('legend_size', 12))

    if annotate_points and len(data_dict) == 1:
        label_key = list(data_dict.keys())[0]
        x_data_for_annot, y_data_for_annot = data_dict[label_key]
        x_data_for_annot = pd.Series(x_data_for_annot) if not isinstance(x_data_for_annot, pd.Series) else x_data_for_annot
        y_data_for_annot = pd.Series(y_data_for_annot) if not isinstance(y_data_for_annot, pd.Series) else y_data_for_annot
        
        common_index_annot = x_data_for_annot.index.intersection(y_data_for_annot.index)
        if common_index_annot.empty and (len(x_data_for_annot) == len(y_data_for_annot)):
            x_data_for_annot.index = pd.RangeIndex(len(x_data_for_annot))
            y_data_for_annot.index = pd.RangeIndex(len(y_data_for_annot))
            common_index_annot = x_data_for_annot.index

        if not common_index_annot.empty:
            x_data_final_annot = x_data_for_annot.loc[common_index_annot]
            y_data_final_annot = y_data_for_annot.loc[common_index_annot]

            points_to_annotate = annotation_subset if annotation_subset else x_data_final_annot.index
            annot_count = 0
            max_annots = VISUALIZATION_CONFIG.get('scatter_max_annotations', 20)
            
            for point_label in points_to_annotate:
                if annot_count >= max_annots:
                    print(f"Warning: Reached max annotations ({max_annots}) for scatter plot. Not all points annotated.")
                    break
                if point_label in x_data_final_annot.index and point_label in y_data_final_annot.index:
                    ax.text(x_data_final_annot[point_label], y_data_final_annot[point_label], 
                            str(point_label), fontsize=VISUALIZATION_CONFIG.get('annotation_fontsize_small', 8),
                            ha='left', va='bottom')
                    annot_count +=1
    
    plt.tight_layout()

    if not all_x_values or not all_y_values:
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
    output_dir: str, 
    filename: str,
    round_name: str = None,
    phase_name: str = None
):
    """
    Generates a line plot of a metric over time for each tenant.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('lineplot_figsize', (12, 7)))

    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    plot_title = f"{title}"
    if round_name and phase_name:
        plot_title += f" : {round_name} - {PHASE_DISPLAY_NAMES.get(phase_name, phase_name)}"
    elif phase_name:
        plot_title += f" : {PHASE_DISPLAY_NAMES.get(phase_name, phase_name)}"
    elif round_name:
        plot_title += f" : {round_name}"

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
        ax.legend(title='Tenants', loc='best', 
                  fontsize=VISUALIZATION_CONFIG.get('legend_size', 10)) 
        if ax.get_legend():
            plt.setp(ax.get_legend().get_title(), fontsize=VISUALIZATION_CONFIG.get('legend_title_size', 12))
    else:
        # Plot without tenant breakdown
        sns.lineplot(x=time_col, y=value_column, data=data_df, ax=ax,
                     marker=VISUALIZATION_CONFIG.get('lineplot_marker', 'o'), 
                     linestyle=VISUALIZATION_CONFIG.get('lineplot_linestyle', '-'))

    ax.set_xlabel("Time (seconds)", fontsize=VISUALIZATION_CONFIG.get('label_size', 12))
    ax.set_ylabel(f"{display_metric_name}", fontsize=VISUALIZATION_CONFIG.get('label_size', 12))
    ax.tick_params(axis='both', which='major', labelsize=VISUALIZATION_CONFIG.get('tick_size', 10))
    ax.grid(VISUALIZATION_CONFIG.get('grid_enabled', True), 
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
    output_dir: str, 
    filename: str,
    round_name: str = None,
    phase_name: str = None
):
    """
    Generates a box plot of a metric for each tenant.
    Handles FutureWarning from Seaborn regarding palette and hue.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('boxplot_figsize', (10, 7)))

    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    plot_title = f"{title}"
    if round_name and phase_name:
        plot_title += f" : {round_name} - {PHASE_DISPLAY_NAMES.get(phase_name, phase_name)}"
    elif phase_name:
        plot_title += f" : {PHASE_DISPLAY_NAMES.get(phase_name, phase_name)}"
    elif round_name:
        plot_title += f" : {round_name}"
    
    ax.set_title(plot_title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))

    if 'tenant' in data_df.columns and data_df['tenant'].nunique() > 0:
        unique_tenants = sorted(data_df['tenant'].unique())
        # Create a list of colors for the palette argument when hue is used
        palette_list = [TENANT_COLORS.get(tenant, '#333333') for tenant in unique_tenants]
        sns.boxplot(x='tenant', y=value_column, hue='tenant', data=data_df, ax=ax, 
                    palette=palette_list, hue_order=unique_tenants, legend=False) # MODIFIED
        ax.set_xlabel("Tenants")
    else:
        # No tenant column or single/no tenants for distinct boxes by tenant
        default_palette_str = VISUALIZATION_CONFIG.get('boxplot_palette', 'Set2')
        if 'tenant' in data_df.columns and data_df['tenant'].nunique() == 1:
            # Plotting a single tenant's box, x-axis can still be 'tenant'
            sns.boxplot(x='tenant', y=value_column, data=data_df, ax=ax, palette=default_palette_str)
            ax.set_xlabel("Tenants")
        else: # No 'tenant' column for x-axis, or it's not appropriate
            sns.boxplot(y=value_column, data=data_df, ax=ax, palette=default_palette_str)
            ax.set_xlabel("") # No specific x-label or a generic one

    ax.set_ylabel(f"{display_metric_name}")
    ax.tick_params(axis='x', rotation=45)

    # Legend handling - this should work fine with legend=False in sns.boxplot
    if 'tenant' in data_df.columns and data_df['tenant'].nunique() > 0 :
        unique_tenants_for_legend = sorted(data_df['tenant'].unique())
        handles = [plt.Rectangle((0,0),1,1, color=TENANT_COLORS.get(t, '#333333')) for t in unique_tenants_for_legend]
        labels = unique_tenants_for_legend
        
        current_legend = ax.get_legend()
        if current_legend: # If seaborn made one despite legend=False (unlikely) or if one existed
            current_legend.remove()

        ax.legend(handles, labels, title='Tenants', loc='best', 
                  fontsize=VISUALIZATION_CONFIG.get('legend_size', 10))
        if ax.get_legend():
            plt.setp(ax.get_legend().get_title(), fontsize=VISUALIZATION_CONFIG.get('legend_title_size', 12))

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
    output_dir: str, 
    filename: str,
    round_name: str = None,
    phase_name: str = None,
    x_var: str = 'tenant', 
    hue_var: str = 'tenant'
):
    set_publication_style()
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('catplot_figsize', (10, 6)))

    plot_data = stats_df.copy()
    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    
    plot_title_parts = [title]
    if round_name and phase_name:
        phase_display = PHASE_DISPLAY_NAMES.get(str(phase_name), str(phase_name))
        plot_title_parts.append(f"{round_name} - {phase_display}")
    elif phase_name:
        phase_display = PHASE_DISPLAY_NAMES.get(str(phase_name), str(phase_name))
        plot_title_parts.append(f"{phase_display}")
    elif round_name:
        plot_title_parts.append(f"{round_name}")
    
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
    ax.set_ylabel(f"Mean {display_metric_name}")

    if hue_var and plot_data[hue_var].nunique() > 0 : # Show legend if there are hues
        # Create legend for hues
        legend_handles = [plt.Rectangle((0,0),1,1, color=palette_to_use[i]) for i in range(len(unique_hues))]
        ax.legend(legend_handles, unique_hues, title=METRIC_DISPLAY_NAMES.get(hue_var, hue_var), loc='best', 
                  fontsize=VISUALIZATION_CONFIG.get('legend_size', 10),
                  title_fontsize=VISUALIZATION_CONFIG.get('legend_title_size', 12))
    
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
