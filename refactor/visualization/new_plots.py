\
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

from pipeline.config import VISUALIZATION_CONFIG, METRIC_DISPLAY_NAMES # TODO: May need to pass config or use a relative import if config is also refactored
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

# TODO: Add other relevant plotting functions from pipeline/visualization/plots.py
# For example:
# - plot_metric_by_phase (if still needed and generic enough)
# - plot_tenant_impact_heatmap (if still needed and generic enough)
# - etc.
