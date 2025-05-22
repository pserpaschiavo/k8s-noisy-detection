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

def plot_pca_explained_variance(
    explained_variance_ratio: np.ndarray,
    cumulative_variance: np.ndarray = None,
    title: str = "PCA Explained Variance",
    output_dir: str = None,
    filename: str = None,
    metric_name: str = None,
    round_name: str = None,
    phase_name: str = None,
    threshold_line: float = 0.8
):
    """
    Generates a scree plot showing explained variance by principal components.

    Args:
        explained_variance_ratio (np.ndarray): Array containing the explained variance ratio for each principal component.
        cumulative_variance (np.ndarray, optional): Array containing the cumulative explained variance.
                                                   If None, it will be computed from explained_variance_ratio.
        title (str): Base title for the plot.
        output_dir (str, optional): Directory to save the plot.
        filename (str, optional): Filename for the saved plot.
        metric_name (str, optional): Name of the metric being analyzed.
        round_name (str, optional): Name of the experiment round.
        phase_name (str, optional): Name of the experiment phase.
        threshold_line (float, optional): Value for drawing a horizontal threshold line (0-1).

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """
    set_publication_style()
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=VISUALIZATION_CONFIG.get('pca_scree_figsize', (10, 6)))
    ax2 = ax1.twinx()
    
    # Calculate number of components
    n_components = len(explained_variance_ratio)
    x_indices = np.arange(1, n_components + 1)
    
    # Calculate cumulative variance if not provided
    if cumulative_variance is None:
        cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Plot individual explained variance (bar chart)
    bars = ax1.bar(x_indices, explained_variance_ratio * 100, 
                   alpha=0.7, color=VISUALIZATION_CONFIG.get('pca_bar_color', '#3498db'),
                   label='Individual Explained Variance')
    ax1.set_xlabel('Principal Component', fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    ax1.set_ylabel('Explained Variance (%)', fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    ax1.set_xlim(0.5, n_components + 0.5)
    ax1.set_ylim(0, max(explained_variance_ratio * 100) * 1.1)
    
    # Plot cumulative explained variance (line chart)
    line = ax2.plot(x_indices, cumulative_variance * 100, 'o-', color=VISUALIZATION_CONFIG.get('pca_line_color', '#e74c3c'),
                   linewidth=2, markersize=6, label='Cumulative Explained Variance')
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    ax2.set_ylim(0, 105)
    
    # Add horizontal threshold line if specified
    if threshold_line is not None and 0 < threshold_line < 1:
        ax2.axhline(y=threshold_line * 100, color='gray', linestyle='--', 
                   label=f'{int(threshold_line*100)}% Variance Threshold')
        
        # Find the number of components needed to reach threshold
        components_for_threshold = np.argmax(cumulative_variance >= threshold_line) + 1
        ax2.axvline(x=components_for_threshold, color='gray', linestyle=':', alpha=0.7)
        
        # Add annotation for threshold
        ax2.annotate(f'{components_for_threshold} PCs',
                    xy=(components_for_threshold, threshold_line * 100),
                    xytext=(components_for_threshold + 0.5, threshold_line * 100 - 10),
                    arrowprops=dict(arrowstyle='->'))
    
    # Customize plot
    ax1.set_xticks(x_indices)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Create custom title with metric, round, and phase information
    plot_title = title
    if metric_name:
        display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
        plot_title = f"{title}: {display_metric_name}"
    
    if round_name and phase_name:
        round_number = round_name.split('-')[-1]
        phase_display = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
        plot_title += f" - Round {round_number}: {phase_display}"
    elif round_name:
        round_number = round_name.split('-')[-1]
        plot_title += f" - Round {round_number}"
    elif phase_name:
        phase_display = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
        plot_title += f" - {phase_display}"
    
    plt.title(plot_title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    
    # Add legends for both axes
    legend1 = ax1.legend(loc='upper left', fontsize=VISUALIZATION_CONFIG.get('legend_size', 10))
    legend2 = ax2.legend(loc='lower right', fontsize=VISUALIZATION_CONFIG.get('legend_size', 10))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure if output directory and filename are provided
    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"PCA explained variance plot saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving PCA explained variance plot to {os.path.join(output_dir, filename)}: {e}")
    
    return fig

def plot_pca_biplot(
    pca_results: pd.DataFrame,
    pca_components: pd.DataFrame,
    x_component: int = 1,
    y_component: int = 2,
    scale_arrows: float = 1.0,
    sample_groups: pd.Series = None,
    palette: dict = None,
    title: str = "PCA Biplot",
    output_dir: str = None,
    filename: str = None,
    metric_name: str = None,
    round_name: str = None,
    phase_name: str = None,
    max_features_to_show: int = 15,
    arrow_alpha: float = 0.5
):
    """
    Generates a biplot of PCA results, showing both samples and feature loadings.

    Args:
        pca_results (pd.DataFrame): DataFrame with PCA transformed data. Columns should be PC1, PC2, etc.
        pca_components (pd.DataFrame): DataFrame with PCA loadings/components. Index should be PC1, PC2, etc.
        x_component (int): Component number for x-axis (1-based index).
        y_component (int): Component number for y-axis (1-based index).
        scale_arrows (float): Scaling factor for the feature arrows.
        sample_groups (pd.Series, optional): Series with group labels for samples, used for coloring.
        palette (dict, optional): Dictionary mapping group labels to colors.
        title (str): Base title for the plot.
        output_dir (str, optional): Directory to save the plot.
        filename (str, optional): Filename for the saved plot.
        metric_name (str, optional): Name of the metric being analyzed.
        round_name (str, optional): Name of the experiment round.
        phase_name (str, optional): Name of the experiment phase.
        max_features_to_show (int, optional): Maximum number of feature arrows to display.
        arrow_alpha (float, optional): Transparency for arrows.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """
    set_publication_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('pca_biplot_figsize', (12, 10)))
    
    # Get column names for specified components
    pc_x_name = f"PC{x_component}"
    pc_y_name = f"PC{y_component}"
    
    # Check if the specified components exist
    if pc_x_name not in pca_results.columns or pc_y_name not in pca_results.columns:
        raise ValueError(f"Specified components ({pc_x_name}, {pc_y_name}) not found in PCA results")
    if pc_x_name not in pca_components.index or pc_y_name not in pca_components.index:
        raise ValueError(f"Specified components ({pc_x_name}, {pc_y_name}) not found in PCA components")
    
    # Extract data for the selected components
    x_data = pca_results[pc_x_name]
    y_data = pca_results[pc_y_name]
    
    # Plot samples
    if sample_groups is not None:
        # Use groups for coloring samples
        unique_groups = sample_groups.unique()
        
        # Use tenant colors if available and applicable
        if palette is None and all(group in TENANT_COLORS for group in unique_groups):
            color_palette = {group: TENANT_COLORS[group] for group in unique_groups}
        elif palette is not None:
            color_palette = palette
        else:
            # Generate a color palette
            colors = sns.color_palette('tab10', n_colors=len(unique_groups))
            color_palette = {group: colors[i] for i, group in enumerate(unique_groups)}
        
        # Plot each group separately to create legend
        for group in unique_groups:
            mask = sample_groups == group
            ax.scatter(
                x_data[mask], y_data[mask],
                alpha=0.7, s=VISUALIZATION_CONFIG.get('scatter_marker_size', 50),
                color=color_palette.get(group, '#333333'),
                edgecolor='k', linewidth=0.5,
                label=group
            )
        
        # Add legend for sample groups
        ax.legend(title="Tenants" if all(group in TENANT_COLORS for group in unique_groups) else "Groups",
                 loc='best', fontsize=VISUALIZATION_CONFIG.get('legend_size', 10))
    else:
        # No grouping, use a single color for all samples
        ax.scatter(
            x_data, y_data,
            alpha=0.7, s=VISUALIZATION_CONFIG.get('scatter_marker_size', 50),
            color=VISUALIZATION_CONFIG.get('biplot_sample_color', '#3498db'),
            edgecolor='k', linewidth=0.5
        )
    
    # Get feature loadings
    loadings_x = pca_components.loc[pc_x_name]
    loadings_y = pca_components.loc[pc_y_name]
    
    # Calculate scaling factor for arrows
    # This makes arrows visible on the same scale as the samples
    if scale_arrows is None:
        # Auto-scaling based on data range and loadings
        x_range = x_data.max() - x_data.min()
        y_range = y_data.max() - y_data.min()
        loading_max = max(loadings_x.abs().max(), loadings_y.abs().max())
        scale_arrows = min(x_range, y_range) * 0.4 / loading_max
    
    # If there are too many features, select the most important ones
    if len(loadings_x) > max_features_to_show:
        # Calculate the magnitude of each loading vector
        loading_magnitudes = np.sqrt(loadings_x**2 + loadings_y**2)
        
        # Get indices of the top features
        top_indices = loading_magnitudes.nlargest(max_features_to_show).index
        
        # Filter to show only the most important features
        loadings_x = loadings_x[top_indices]
        loadings_y = loadings_y[top_indices]
    
    # Plot feature loadings as arrows
    for i, feature in enumerate(loadings_x.index):
        ax.arrow(
            0, 0,  # Start at origin
            loadings_x[feature] * scale_arrows,
            loadings_y[feature] * scale_arrows,
            head_width=0.05 * scale_arrows,
            head_length=0.1 * scale_arrows,
            fc=VISUALIZATION_CONFIG.get('biplot_arrow_color', 'red'),
            ec=VISUALIZATION_CONFIG.get('biplot_arrow_color', 'red'),
            alpha=arrow_alpha,
            length_includes_head=True
        )
        
        # Add feature label at arrow tip
        ax.text(
            loadings_x[feature] * scale_arrows * 1.1,
            loadings_y[feature] * scale_arrows * 1.1,
            feature,
            color=VISUALIZATION_CONFIG.get('biplot_text_color', 'black'),
            ha='center', va='center',
            fontsize=VISUALIZATION_CONFIG.get('biplot_label_size', 8)
        )
    
    # Add labels and title
    ax.set_xlabel(f"Principal Component {x_component}", fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    ax.set_ylabel(f"Principal Component {y_component}", fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    
    # Create custom title with metric, round, and phase information
    plot_title = title
    if metric_name:
        display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
        plot_title = f"{title}: {display_metric_name}"
    
    if round_name and phase_name:
        round_number = round_name.split('-')[-1]
        phase_display = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
        plot_title += f" - Round {round_number}: {phase_display}"
    elif round_name:
        round_number = round_name.split('-')[-1]
        plot_title += f" - Round {round_number}"
    elif phase_name:
        phase_display = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
        plot_title += f" - {phase_display}"
    
    plt.title(plot_title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    
    # Add origin lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # Make the plot more balanced/symmetric if needed
    ax.axis('equal')
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save figure if output directory and filename are provided
    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"PCA biplot saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving PCA biplot to {os.path.join(output_dir, filename)}: {e}")
    
    return fig

def plot_pca_loadings_heatmap(
    pca_components: pd.DataFrame,
    title: str = "PCA Loadings",
    output_dir: str = None,
    filename: str = None,
    metric_name: str = None,
    round_name: str = None,
    phase_name: str = None,
    cmap: str = 'coolwarm',
    n_components: int = None
):
    """
    Generates a heatmap of PCA loadings to visualize feature contributions.

    Args:
        pca_components (pd.DataFrame): DataFrame with PCA loadings/components.
                                       Index should be PC1, PC2, etc., and columns should be features.
        title (str): Base title for the plot.
        output_dir (str, optional): Directory to save the plot.
        filename (str, optional): Filename for the saved plot.
        metric_name (str, optional): Name of the metric being analyzed.
        round_name (str, optional): Name of the experiment round.
        phase_name (str, optional): Name of the experiment phase.
        cmap (str, optional): Colormap to use for the heatmap.
        n_components (int, optional): Number of components to include in the heatmap. If None, all components are included.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """
    set_publication_style()
    
    # Limit the number of components if specified
    if n_components is not None and n_components < len(pca_components):
        components_subset = pca_components.iloc[:n_components]
    else:
        components_subset = pca_components
    
    # Create figure
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('pca_heatmap_figsize', (12, 8)))
    
    # Generate heatmap
    sns.heatmap(components_subset, annot=True, fmt=".2f", cmap=cmap, center=0, linewidths=.5,
                cbar_kws={'label': 'Loading Value'}, ax=ax)
    
    # Create custom title with metric, round, and phase information
    plot_title = title
    if metric_name:
        display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
        plot_title = f"{title}: {display_metric_name}"
    
    if round_name and phase_name:
        round_number = round_name.split('-')[-1]
        phase_display = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
        plot_title += f" - Round {round_number}: {phase_display}"
    elif round_name:
        round_number = round_name.split('-')[-1]
        plot_title += f" - Round {round_number}"
    elif phase_name:
        phase_display = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
        plot_title += f" - {phase_display}"
    
    ax.set_title(plot_title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    ax.set_ylabel('Principal Components', fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    ax.set_xlabel('Features', fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure if output directory and filename are provided
    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"PCA loadings heatmap saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving PCA loadings heatmap to {os.path.join(output_dir, filename)}: {e}")
    
    return fig

def plot_ica_components_heatmap(
    ica_components: pd.DataFrame,
    title: str = "ICA Components Heatmap",
    output_dir: str = None,
    filename: str = None,
    metric_name: str = None,
    round_name: str = None,
    phase_name: str = None,
    cmap: str = 'coolwarm',
    n_components: int = None
):
    """
    Generates a heatmap of ICA components to visualize feature contributions.

    Args:
        ica_components (pd.DataFrame): DataFrame with ICA components/unmixing matrix.
                                      Index should be IC1, IC2, etc., and columns should be features.
        title (str): Base title for the plot.
        output_dir (str, optional): Directory to save the plot.
        filename (str, optional): Filename for the saved plot.
        metric_name (str, optional): Name of the metric being analyzed.
        round_name (str, optional): Name of the experiment round.
        phase_name (str, optional): Name of the experiment phase.
        cmap (str, optional): Colormap to use for the heatmap.
        n_components (int, optional): Number of components to include in the heatmap. If None, all components are included.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """
    set_publication_style()
    
    # Limit the number of components if specified
    if n_components is not None and n_components < len(ica_components):
        components_subset = ica_components.iloc[:n_components]
    else:
        components_subset = ica_components
    
    # Create figure
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('ica_heatmap_figsize', (12, 8)))
    
    # Generate heatmap
    sns.heatmap(components_subset, annot=True, fmt=".2f", cmap=cmap, center=0, linewidths=.5,
                cbar_kws={'label': 'Component Weight'}, ax=ax)
    
    # Create custom title with metric, round, and phase information
    plot_title = title
    if metric_name:
        display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
        plot_title = f"{title}: {display_metric_name}"
    
    if round_name and phase_name:
        round_number = round_name.split('-')[-1]
        phase_display = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
        plot_title += f" - Round {round_number}: {phase_display}"
    elif round_name:
        round_number = round_name.split('-')[-1]
        plot_title += f" - Round {round_number}"
    elif phase_name:
        phase_display = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
        plot_title += f" - {phase_display}"
    
    ax.set_title(plot_title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    ax.set_ylabel('Independent Components', fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    ax.set_xlabel('Features', fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure if output directory and filename are provided
    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"ICA components heatmap saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving ICA components heatmap to {os.path.join(output_dir, filename)}: {e}")
    
    return fig

def plot_ica_time_series(
    ica_results: pd.DataFrame,
    title: str = "ICA Time Series",
    output_dir: str = None,
    filename: str = None,
    metric_name: str = None,
    round_name: str = None,
    phase_name: str = None,
    max_components: int = 4
):
    """
    Plots the time series of independent components.

    Args:
        ica_results (pd.DataFrame): DataFrame containing the independent components time series.
                                   Columns should be IC1, IC2, etc.
        title (str): Base title for the plot.
        output_dir (str, optional): Directory to save the plot.
        filename (str, optional): Filename for the saved plot.
        metric_name (str, optional): Name of the metric being analyzed.
        round_name (str, optional): Name of the experiment round.
        phase_name (str, optional): Name of the experiment phase.
        max_components (int, optional): Maximum number of components to plot.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """
    set_publication_style()
    
    # Determine how many components to plot
    n_components = min(max_components, len(ica_results.columns))
    
    # Set up the figure
    fig, axes = plt.subplots(n_components, 1, figsize=(10, 2.5 * n_components), sharex=True)
    if n_components == 1:
        axes = [axes]  # Make axes indexable if only one subplot
    
    # Create custom title with metric, round, and phase information
    plot_title = title
    if metric_name:
        display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
        plot_title = f"{title}: {display_metric_name}"
    
    if round_name and phase_name:
        round_number = round_name.split('-')[-1]
        phase_display = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
        plot_title += f" - Round {round_number}: {phase_display}"
    elif round_name:
        round_number = round_name.split('-')[-1]
        plot_title += f" - Round {round_number}"
    elif phase_name:
        phase_display = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
        plot_title += f" - {phase_display}"
    
    fig.suptitle(plot_title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    
    # Plot each component
    for i, component in enumerate(ica_results.columns[:n_components]):
        # Check if the index is a DatetimeIndex
        if isinstance(ica_results.index, pd.DatetimeIndex):
            axes[i].plot(ica_results.index, ica_results[component], 
                         label=component, linewidth=1.5)
        else:
            axes[i].plot(ica_results[component].values, 
                         label=component, linewidth=1.5)
        
        axes[i].set_title(f"Component {component}", fontsize=VISUALIZATION_CONFIG.get('subtitle_size', 14))
        axes[i].set_ylabel("Value", fontsize=VISUALIZATION_CONFIG.get('label_size', 12))
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Set the x-axis label only for the bottom plot
    if isinstance(ica_results.index, pd.DatetimeIndex):
        axes[-1].set_xlabel("Time", fontsize=VISUALIZATION_CONFIG.get('label_size', 12))
    else:
        axes[-1].set_xlabel("Sample", fontsize=VISUALIZATION_CONFIG.get('label_size', 12))
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)  # Adjust to make room for suptitle
    
    # Save figure if output directory and filename are provided
    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"ICA time series plot saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving ICA time series plot to {os.path.join(output_dir, filename)}: {e}")
    
    return fig

def plot_ica_scatter(
    ica_results: pd.DataFrame,
    x_component: int = 1,
    y_component: int = 2,
    title: str = "ICA Scatter Plot",
    output_dir: str = None,
    filename: str = None,
    metric_name: str = None,
    round_name: str = None,
    phase_name: str = None,
    sample_groups: pd.Series = None,
    palette: dict = None
):
    """
    Creates a scatter plot of two independent components.

    Args:
        ica_results (pd.DataFrame): DataFrame containing the independent components.
                                   Columns should be IC1, IC2, etc.
        x_component (int): The component number for the x-axis (1-based).
        y_component (int): The component number for the y-axis (1-based).
        title (str): Base title for the plot.
        output_dir (str, optional): Directory to save the plot.
        filename (str, optional): Filename for the saved plot.
        metric_name (str, optional): Name of the metric being analyzed.
        round_name (str, optional): Name of the experiment round.
        phase_name (str, optional): Name of the experiment phase.
        sample_groups (pd.Series, optional): Series with group labels for each sample (e.g., tenants).
                                            Index should match ica_results.index.
        palette (dict, optional): Color mapping for groups.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('scatter_figsize', (8, 8)))
    
    x_col = f'IC{x_component}'
    y_col = f'IC{y_component}'
    
    if x_col not in ica_results.columns or y_col not in ica_results.columns:
        raise ValueError(f"Components IC{x_component} or IC{y_component} not found in results dataframe. Available components: {ica_results.columns.tolist()}")
    
    # Prepare data
    x_data = ica_results[x_col]
    y_data = ica_results[y_col]
    
    # Color by groups if provided
    if sample_groups is not None:
        # Ensure sample_groups has the same index as ica_results
        common_idx = ica_results.index.intersection(sample_groups.index)
        if len(common_idx) == 0:
            print(f"Warning: No common indices between ICA results and sample groups. Plotting without groups.")
            ax.scatter(x_data, y_data, alpha=0.7, 
                       s=VISUALIZATION_CONFIG.get('scatter_marker_size', 50),
                       color=VISUALIZATION_CONFIG.get('scatter_color', '#3498db'),
                       edgecolor='k', linewidth=0.5)
        else:
            x_data = x_data.loc[common_idx]
            y_data = y_data.loc[common_idx]
            sample_groups = sample_groups.loc[common_idx]
            
            # Get unique groups
            unique_groups = sample_groups.unique()
            
            # Create color mapping
            if palette is None and all(group in TENANT_COLORS for group in unique_groups):
                color_palette = {group: TENANT_COLORS[group] for group in unique_groups}
            elif palette is not None:
                color_palette = palette
            else:
                # Generate a color palette
                colors = sns.color_palette('tab10', n_colors=len(unique_groups))
                color_palette = {group: colors[i] for i, group in enumerate(unique_groups)}
            
            # Plot each group separately
            for group in unique_groups:
                mask = sample_groups == group
                ax.scatter(
                    x_data[mask], y_data[mask],
                    alpha=0.7, s=VISUALIZATION_CONFIG.get('scatter_marker_size', 50),
                    color=color_palette.get(group, '#333333'),
                    edgecolor='k', linewidth=0.5,
                    label=group
                )
            
            # Add legend for sample groups
            ax.legend(title="Tenants" if all(group in TENANT_COLORS for group in unique_groups) else "Groups",
                     loc='best', fontsize=VISUALIZATION_CONFIG.get('legend_size', 10))
    else:
        # No grouping, use a single color for all samples
        ax.scatter(
            x_data, y_data,
            alpha=0.7, s=VISUALIZATION_CONFIG.get('scatter_marker_size', 50),
            color=VISUALIZATION_CONFIG.get('scatter_color', '#3498db'),
            edgecolor='k', linewidth=0.5
        )
    
    # Add labels and title
    ax.set_xlabel(f"Independent Component {x_component}", fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    ax.set_ylabel(f"Independent Component {y_component}", fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    
    # Create custom title with metric, round, and phase information
    plot_title = title
    if metric_name:
        display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
        plot_title = f"{title}: {display_metric_name}"
    
    if round_name and phase_name:
        round_number = round_name.split('-')[-1]
        phase_display = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
        plot_title += f" - Round {round_number}: {phase_display}"
    elif round_name:
        round_number = round_name.split('-')[-1]
        plot_title += f" - Round {round_number}"
    elif phase_name:
        phase_display = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
        plot_title += f" - {phase_display}"
    
    plt.title(plot_title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    
    # Add origin lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # Make the plot more balanced/symmetric if needed
    ax.axis('equal')
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save figure if output directory and filename are provided
    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"ICA scatter plot saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving ICA scatter plot to {os.path.join(output_dir, filename)}: {e}")
    
    return fig

def plot_pca_vs_ica_comparison(
    pca_components: pd.DataFrame,
    ica_components: pd.DataFrame,
    feature_subset: list = None,
    n_components: int = 2,
    title: str = "PCA vs ICA Feature Importance Comparison",
    output_dir: str = None,
    filename: str = None,
    metric_name: str = None,
    round_name: str = None,
    phase_name: str = None
):
    """
    Creates a bar chart comparing feature importances between PCA and ICA.

    Args:
        pca_components (pd.DataFrame): DataFrame with PCA loadings/components.
        ica_components (pd.DataFrame): DataFrame with ICA components/unmixing matrix.
        feature_subset (list, optional): List of feature names to include in the comparison.
                                        If None, all common features are used.
        n_components (int, optional): Number of components to include in the comparison.
        title (str): Base title for the plot.
        output_dir (str, optional): Directory to save the plot.
        filename (str, optional): Filename for the saved plot.
        metric_name (str, optional): Name of the metric being analyzed.
        round_name (str, optional): Name of the experiment round.
        phase_name (str, optional): Name of the experiment phase.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """
    set_publication_style()
    
    # Limit to the number of components
    pca_subset = pca_components.iloc[:n_components] if len(pca_components) >= n_components else pca_components
    ica_subset = ica_components.iloc[:n_components] if len(ica_components) >= n_components else ica_components
    
    # Get common features
    common_features = set(pca_subset.columns).intersection(set(ica_subset.columns))
    
    if feature_subset:
        # Filter to only include specified features
        features_to_use = [f for f in feature_subset if f in common_features]
        if not features_to_use:
            raise ValueError("None of the specified feature_subset features are found in both PCA and ICA results.")
    else:
        # If not specified, compute feature importance and take top features
        pca_importance = pca_subset.abs().mean(axis=0)
        ica_importance = ica_subset.abs().mean(axis=0)
        
        # Get the most important features across both methods
        combined_importance = pca_importance.add(ica_importance, fill_value=0)
        features_to_use = combined_importance.nlargest(10).index.tolist()
        features_to_use = [f for f in features_to_use if f in common_features]
    
    # Filter components to only include selected features
    pca_filtered = pca_subset[features_to_use].copy()
    ica_filtered = ica_subset[features_to_use].copy()
    
    # Create a figure for the comparison
    fig, axes = plt.subplots(n_components, 1, figsize=(10, 3.5 * n_components))
    if n_components == 1:
        axes = [axes]  # Make axes indexable if only one subplot
    
    # Create custom title with metric, round, and phase information
    plot_title = title
    if metric_name:
        display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
        plot_title = f"{title}: {display_metric_name}"
    
    if round_name and phase_name:
        round_number = round_name.split('-')[-1]
        phase_display = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
        plot_title += f" - Round {round_number}: {phase_display}"
    elif round_name:
        round_number = round_name.split('-')[-1]
        plot_title += f" - Round {round_number}"
    elif phase_name:
        phase_display = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
        plot_title += f" - {phase_display}"
    
    fig.suptitle(plot_title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    
    # Define bar width
    bar_width = 0.35
    
    # Create a plot for each component
    for i in range(min(n_components, len(pca_filtered), len(ica_filtered))):
        # Get current axis
        ax = axes[i]
        
        # Set up indices for bars
        indices = np.arange(len(features_to_use))
        
        # Create a dataframe for easier plotting
        comp_df = pd.DataFrame({
            'Feature': features_to_use,
            'PCA': pca_filtered.iloc[i].abs(),
            'ICA': ica_filtered.iloc[i].abs()
        })
        
        # Sort by combined importance
        comp_df['Combined'] = comp_df['PCA'] + comp_df['ICA']
        comp_df = comp_df.sort_values('Combined', ascending=False)
        
        # Plot bars
        ax.bar(indices - bar_width/2, comp_df['PCA'], bar_width, label='PCA', color='blue', alpha=0.7)
        ax.bar(indices + bar_width/2, comp_df['ICA'], bar_width, label='ICA', color='red', alpha=0.7)
        
        # Add labels and formatting
        ax.set_xticks(indices)
        ax.set_xticklabels(comp_df['Feature'], rotation=45, ha='right')
        ax.set_ylabel('Absolute Coefficient Value')
        ax.set_title(f"Component {i+1} (PC{i+1} vs IC{i+1})")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend()
        
        # Set the same y-axis limits for both PCA and ICA for easier comparison
        y_max = max(comp_df['PCA'].max(), comp_df['ICA'].max()) * 1.1
        ax.set_ylim(0, y_max)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)  # Adjust to make room for suptitle
    
    # Save figure if output directory and filename are provided
    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"PCA vs ICA comparison plot saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving PCA vs ICA comparison plot to {os.path.join(output_dir, filename)}: {e}")
    
    return fig

def plot_pca_vs_ica_overlay_scatter(
    pca_results: pd.DataFrame,
    ica_results: pd.DataFrame,
    x_component: int = 1,
    y_component: int = 2,
    title: str = "PCA vs ICA Overlay Scatter Plot",
    output_dir: str = None,
    filename: str = None,
    metric_name: str = None,
    round_name: str = None,
    phase_name: str = None,
    sample_groups: pd.Series = None
):
    """
    Creates a scatter plot overlaying PCA and ICA results for the same samples.

    Args:
        pca_results (pd.DataFrame): DataFrame containing the principal components.
        ica_results (pd.DataFrame): DataFrame containing the independent components.
        x_component (int): The component number for the x-axis (1-based).
        y_component (int): The component number for the y-axis (1-based).
        title (str): Base title for the plot.
        output_dir (str, optional): Directory to save the plot.
        filename (str, optional): Filename for the saved plot.
        metric_name (str, optional): Name of the metric being analyzed.
        round_name (str, optional): Name of the experiment round.
        phase_name (str, optional): Name of the experiment phase.
        sample_groups (pd.Series, optional): Series with group labels for each sample (e.g., tenants).

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG.get('scatter_figsize', (10, 8)))
    
    pca_x_col = f'PC{x_component}'
    pca_y_col = f'PC{y_component}'
    ica_x_col = f'IC{x_component}'
    ica_y_col = f'IC{y_component}'
    
    if pca_x_col not in pca_results.columns or pca_y_col not in pca_results.columns:
        raise ValueError(f"Components PC{x_component} or PC{y_component} not found in PCA results.")
    
    if ica_x_col not in ica_results.columns or ica_y_col not in ica_results.columns:
        raise ValueError(f"Components IC{x_component} or IC{y_component} not found in ICA results.")
    
    # Find common samples
    common_idx = pca_results.index.intersection(ica_results.index)
    if len(common_idx) == 0:
        print(f"Warning: No common indices between PCA and ICA results. Cannot create overlay plot.")
        return None
    
    # Filter to common samples
    pca_x = pca_results.loc[common_idx, pca_x_col]
    pca_y = pca_results.loc[common_idx, pca_y_col]
    ica_x = ica_results.loc[common_idx, ica_x_col]
    ica_y = ica_results.loc[common_idx, ica_y_col]
    
    # Normalize to the same scale for visual comparison (only for display purposes)
    # This helps compare the clustering patterns rather than the absolute values
    def normalize(series):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(0, index=series.index)
        return (series - min_val) / (max_val - min_val) * 2 - 1  # Scale to [-1, 1]
    
    pca_x_norm = normalize(pca_x)
    pca_y_norm = normalize(pca_y)
    ica_x_norm = normalize(ica_x)
    ica_y_norm = normalize(ica_y)
    
    # Color by groups if provided
    if sample_groups is not None and not sample_groups.empty:
        # Ensure sample_groups has common indices with the results
        common_group_idx = common_idx.intersection(sample_groups.index)
        if len(common_group_idx) == 0:
            print(f"Warning: No common indices between results and sample groups. Plotting without groups.")
            # Plot without group coloring
            ax.scatter(pca_x_norm, pca_y_norm, alpha=0.7, marker='o', s=70, label='PCA', edgecolor='k', linewidth=0.5)
            ax.scatter(ica_x_norm, ica_y_norm, alpha=0.7, marker='x', s=70, label='ICA', edgecolor='k', linewidth=0.5)
        else:
            # Filter to common indices between results and groups
            pca_x_norm = pca_x_norm.loc[common_group_idx]
            pca_y_norm = pca_y_norm.loc[common_group_idx]
            ica_x_norm = ica_x_norm.loc[common_group_idx]
            ica_y_norm = ica_y_norm.loc[common_group_idx]
            sample_groups_filtered = sample_groups.loc[common_group_idx]
            
            # Get unique groups
            unique_groups = sorted(sample_groups_filtered.unique())
            
            # Create color mapping
            if all(group in TENANT_COLORS for group in unique_groups):
                color_palette = {group: TENANT_COLORS[group] for group in unique_groups}
            else:
                colors = sns.color_palette('tab10', n_colors=len(unique_groups))
                color_palette = {group: colors[i] for i, group in enumerate(unique_groups)}
            
            # Plot each group separately
            for group in unique_groups:
                mask = sample_groups_filtered == group
                
                # Plot PCA
                ax.scatter(
                    pca_x_norm[mask], pca_y_norm[mask],
                    alpha=0.7, marker='o', s=70,
                    color=color_palette.get(group, '#333333'),
                    edgecolor='k', linewidth=0.5,
                    label=f'{group} (PCA)'
                )
                
                # Plot ICA
                ax.scatter(
                    ica_x_norm[mask], ica_y_norm[mask],
                    alpha=0.7, marker='x', s=70,
                    color=color_palette.get(group, '#333333'),
                    # Removed edgecolor for marker='x' as it causes warnings
                    linewidth=1.0,
                    label=f'{group} (ICA)'
                )
    else:
        # No grouping information, use different markers for PCA and ICA
        ax.scatter(pca_x_norm, pca_y_norm, alpha=0.7, marker='o', s=70, 
                  color='blue', label='PCA', edgecolor='k', linewidth=0.5)
        ax.scatter(ica_x_norm, ica_y_norm, alpha=0.7, marker='x', s=70, 
                  color='red', label='ICA', 
                  # Removed edgecolor for marker='x' as it causes warnings
                  linewidth=1.0)
    
    # Add labels and title
    ax.set_xlabel("Component 1 (Normalized Scale)", fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    ax.set_ylabel("Component 2 (Normalized Scale)", fontsize=VISUALIZATION_CONFIG.get('label_size', 14))
    
    # Create custom title with metric, round, and phase information
    plot_title = title
    if metric_name:
        display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
        plot_title = f"{title}: {display_metric_name}"
    
    if round_name and phase_name:
        round_number = round_name.split('-')[-1]
        phase_display = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
        plot_title += f" - Round {round_number}: {phase_display}"
    elif round_name:
        round_number = round_name.split('-')[-1]
        plot_title += f" - Round {round_number}"
    elif phase_name:
        phase_display = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
        plot_title += f" - {phase_display}"
    
    plt.title(plot_title, fontsize=VISUALIZATION_CONFIG.get('title_size', 16))
    
    # Add legend - if we have sample groups, make the legend more compact
    if sample_groups is not None and not sample_groups.empty:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=VISUALIZATION_CONFIG.get('legend_size', 10))
    else:
        ax.legend(loc='best', fontsize=VISUALIZATION_CONFIG.get('legend_size', 12))
    
    # Add grid and axis lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Make the plot more balanced/symmetric
    ax.axis('equal')
    
    plt.tight_layout()
    
    # Save figure if output directory and filename are provided
    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            save_figure(fig, output_dir, filename, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
            print(f"PCA vs ICA overlay scatter plot saved to {os.path.join(output_dir, filename)}")
        except Exception as e:
            print(f"Error saving PCA vs ICA overlay scatter plot to {os.path.join(output_dir, filename)}: {e}")
    
    return fig
