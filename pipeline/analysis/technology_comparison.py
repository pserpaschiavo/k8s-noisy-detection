"""
Module for comparison between experiments with different technologies (vCluster, Kata Containers, etc.).

This module implements functions for comparative analysis of different
containerization technologies, focusing on performance, isolation, and interference metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import os

def normalize_metrics_between_experiments(exp1_data, exp2_data, metrics_list=None):
    """
    Normalizes metrics between two experiments to allow fair comparison.
    
    Args:
        exp1_data (dict): Data from the first experiment (dictionary of DataFrames).
        exp2_data (dict): Data from the second experiment.
        metrics_list (list): List of metrics to normalize (None = all common).
        
    Returns:
        tuple: Dictionary of normalized metrics for each experiment and normalization statistics.
    """
    # Identify common metrics if not specified
    if metrics_list is None:
        metrics_list = [m for m in exp1_data.keys() if m in exp2_data.keys()]
    
    # Dictionaries for normalized results
    exp1_normalized = {}
    exp2_normalized = {}
    
    # Statistics for normalization
    normalization_stats = {}
    
    for metric in metrics_list:
        if metric not in exp1_data or metric not in exp2_data:
            continue
            
        # Get DataFrames
        df1 = exp1_data[metric]
        df2 = exp2_data[metric]
        
        # Global statistics for normalization
        combined_values = pd.concat([df1['value'], df2['value']])
        global_min = combined_values.min()
        global_max = combined_values.max()
        global_mean = combined_values.mean()
        global_std = combined_values.std()
        
        # Save statistics
        normalization_stats[metric] = {
            'min': global_min,
            'max': global_max,
            'mean': global_mean,
            'std': global_std
        }
        
        # Normalize using Min-Max or Z-score as needed
        # We use Min-Max for most metrics (maintains proportions)
        if global_max > global_min:
            # Create copies to avoid modifying originals
            df1_norm = df1.copy()
            df2_norm = df2.copy()
            
            # Min-Max normalization (values between 0 and 1)
            df1_norm['value'] = (df1['value'] - global_min) / (global_max - global_min)
            df2_norm['value'] = (df2['value'] - global_min) / (global_max - global_min)
            
            exp1_normalized[metric] = df1_norm
            exp2_normalized[metric] = df2_norm
        else:
            # If min=max (constant values), keep originals
            exp1_normalized[metric] = df1.copy()
            exp2_normalized[metric] = df2.copy()
    
    return exp1_normalized, exp2_normalized, normalization_stats

def calculate_relative_efficiency(exp1_data, exp2_data, metrics_list=None, tenants_list=None, 
                                 phase_filter=None, rounds_filter: Optional[Union[str, List[str]]] = None, 
                                 exp1_name="Experiment 1", exp2_name="Experiment 2"):
    """
    Calculates relative efficiency metrics between two experiments.
    
    Args:
        exp1_data (dict): Data from the first experiment.
        exp2_data (dict): Data from the second experiment.
        metrics_list (list): List of metrics to compare.
        tenants_list (list): List of tenants to include.
        phase_filter (str): Optional filter for a specific phase.
        rounds_filter (Optional[Union[str, List[str]]]): Optional filter for specific round(s).
        exp1_name (str): Name of the first experiment for results.
        exp2_name (str): Name of the second experiment for results.
        
    Returns:
        DataFrame: DataFrame with relative efficiency metrics.
    """
    results = []
    
    # Identify common metrics if not specified
    if metrics_list is None:
        metrics_list = [m for m in exp1_data.keys() if m in exp2_data.keys()]
    
    for metric_name in metrics_list:
        if metric_name not in exp1_data or metric_name not in exp2_data:
            continue
            
        df1 = exp1_data[metric_name].copy() # Use .copy() to avoid modifying original dict data
        df2 = exp2_data[metric_name].copy() # Use .copy() to avoid modifying original dict data
        
        # Apply phase filter if specified
        if phase_filter:
            if 'phase' in df1.columns:
                df1 = df1[df1['phase'] == phase_filter]
            if 'phase' in df2.columns:
                df2 = df2[df2['phase'] == phase_filter]

        # Apply round filter if specified
        if rounds_filter:
            if isinstance(rounds_filter, str):
                rounds_to_keep = [rounds_filter]
            else: # Assuming it's a list
                rounds_to_keep = rounds_filter
            
            if 'round' in df1.columns:
                df1 = df1[df1['round'].isin(rounds_to_keep)]
            if 'round' in df2.columns:
                df2 = df2[df2['round'].isin(rounds_to_keep)]
        
        # Filter tenants if specified
        if tenants_list and 'tenant' in df1.columns and 'tenant' in df2.columns:
            df1_filtered_tenants = df1[df1['tenant'].isin(tenants_list)]
            df2_filtered_tenants = df2[df2['tenant'].isin(tenants_list)]
            
            # Load configurations
            from pipeline.config import DEFAULT_NOISY_TENANT
            
            # Determine the noisy tenant
            noisy_tenant = DEFAULT_NOISY_TENANT
                
            # Check if the noisy tenant is present in both experiments
            has_noisy_tenant_exp1 = noisy_tenant in df1_filtered_tenants['tenant'].unique()
            has_noisy_tenant_exp2 = noisy_tenant in df2_filtered_tenants['tenant'].unique()
            
            # For each tenant, calculate statistics
            for tenant in tenants_list:
                # Special handling for the noisy tenant
                if tenant == noisy_tenant:
                    tenant_df1 = df1_filtered_tenants[df1_filtered_tenants['tenant'] == tenant] if has_noisy_tenant_exp1 else pd.DataFrame()
                    tenant_df2 = df2_filtered_tenants[df2_filtered_tenants['tenant'] == tenant] if has_noisy_tenant_exp2 else pd.DataFrame()
                    
                    # If the noisy tenant does not exist in an experiment but is in the tenants list
                    if tenant_df1.empty and noisy_tenant in tenants_list:
                        other_tenant_df1 = df1_filtered_tenants[df1_filtered_tenants['tenant'] != noisy_tenant]
                        if not other_tenant_df1.empty:
                            template_df = other_tenant_df1.iloc[[0]].copy()
                            template_df['tenant'] = noisy_tenant
                            template_df['value'] = 0 
                            tenant_df1 = template_df
                    
                    if tenant_df2.empty and noisy_tenant in tenants_list:
                        other_tenant_df2 = df2_filtered_tenants[df2_filtered_tenants['tenant'] != noisy_tenant]
                        if not other_tenant_df2.empty:
                            template_df = other_tenant_df2.iloc[[0]].copy()
                            template_df['tenant'] = noisy_tenant
                            template_df['value'] = 0
                            tenant_df2 = template_df
                else:
                    tenant_df1 = df1_filtered_tenants[df1_filtered_tenants['tenant'] == tenant]
                    tenant_df2 = df2_filtered_tenants[df2_filtered_tenants['tenant'] == tenant]
                
                # Skip if not enough data
                if tenant_df1.empty or tenant_df2.empty or len(tenant_df1['value'].dropna()) < 2 or len(tenant_df2['value'].dropna()) < 2:
                    continue
                
                # Basic statistics
                mean1 = tenant_df1['value'].mean()
                mean2 = tenant_df2['value'].mean()
                std1 = tenant_df1['value'].std()
                std2 = tenant_df2['value'].std()
                
                # Calculation of relative differences
                if mean2 != 0:
                    percent_diff = ((mean1 - mean2) / abs(mean2)) * 100
                elif mean1 != 0:
                    percent_diff = np.inf * np.sign(mean1)
                else:
                    percent_diff = 0.0
                
                # T-test for statistical significance
                t_stat, p_value = stats.ttest_ind(
                    tenant_df1['value'].dropna(), 
                    tenant_df2['value'].dropna(),
                    equal_var=False,
                    nan_policy='omit'
                )
                
                # Add results
                results.append({
                    'metric': metric_name,
                    'tenant': tenant,
                    'phase': phase_filter if phase_filter else 'all',
                    'round': ",".join(rounds_filter) if isinstance(rounds_filter, list) else rounds_filter if rounds_filter else 'all',
                    f'{exp1_name}_mean': mean1,
                    f'{exp2_name}_mean': mean2,
                    f'{exp1_name}_std': std1,
                    f'{exp2_name}_std': std2,
                    'difference': mean1 - mean2,
                    'percent_difference': percent_diff,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'statistically_significant': p_value < 0.05,
                    'better_experiment': exp1_name if mean1 < mean2 else exp2_name if mean2 < mean1 else 'equal' # Assuming lower is better
                })
        else:
            # If no tenant list is provided, aggregate across all tenants
            if df1.empty or df2.empty or len(df1['value'].dropna()) < 2 or len(df2['value'].dropna()) < 2:
                continue

            mean1 = df1['value'].mean()
            mean2 = df2['value'].mean()
            std1 = df1['value'].std()
            std2 = df2['value'].std()
            
            if mean2 != 0:
                percent_diff = ((mean1 - mean2) / abs(mean2)) * 100
            elif mean1 != 0:
                percent_diff = np.inf * np.sign(mean1)
            else:
                percent_diff = 0.0

            t_stat, p_value = stats.ttest_ind(
                df1['value'].dropna(), 
                df2['value'].dropna(),
                equal_var=False,
                nan_policy='omit'
            )
            
            results.append({
                'metric': metric_name,
                'tenant': 'all',
                'phase': phase_filter if phase_filter else 'all',
                'round': ",".join(rounds_filter) if isinstance(rounds_filter, list) else rounds_filter if rounds_filter else 'all',
                f'{exp1_name}_mean': mean1,
                f'{exp2_name}_mean': mean2,
                f'{exp1_name}_std': std1,
                f'{exp2_name}_std': std2,
                'difference': mean1 - mean2,
                'percent_difference': percent_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'statistically_significant': p_value < 0.05,
                'better_experiment': exp1_name if mean1 < mean2 else exp2_name if mean2 < mean1 else 'equal' # Assuming lower is better
            })
    
    return pd.DataFrame(results)

def plot_experiment_comparison(efficiency_data, exp1_name, exp2_name, 
                              metric_filter=None, tenant_filter=None,
                              figsize=(14, 10)):
    """
    Plots visual comparison between experiments with different technologies.
    
    Args:
        efficiency_data (DataFrame): DataFrame with relative efficiency metrics.
        exp1_name (str): Name of the first experiment.
        exp2_name (str): Name of the second experiment.
        metric_filter (str): Filter for a specific metric.
        tenant_filter (str): Filter for a specific tenant.
        figsize (tuple): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Figure with the comparison.
    """
    plot_data = efficiency_data.copy()
    
    if metric_filter:
        plot_data = plot_data[plot_data['metric'] == metric_filter]
        
    if tenant_filter:
        plot_data = plot_data[plot_data['tenant'] == tenant_filter]
    
    if plot_data.empty:
        print("Filtered data insufficient to generate visualization.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data to plot", ha='center', va='center')
        return fig

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Comparison between {exp1_name} and {exp2_name}', fontsize=16)
    
    ax1 = axes[0, 0]
    
    if 'tenant' not in plot_data.columns or plot_data['tenant'].nunique() <= 1:
        # Simplified plot if only one tenant or tenant column is missing
        plot_data_pivot = plot_data.set_index('metric')[[f'{exp1_name}_mean', f'{exp2_name}_mean']]
    else:
        plot_data_pivot = plot_data.pivot_table(
            index='metric', 
            columns='tenant', 
            values=[f'{exp1_name}_mean', f'{exp2_name}_mean']
        )
        if isinstance(plot_data_pivot.columns, pd.MultiIndex):
            plot_data_pivot.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in plot_data_pivot.columns]
    
    sns.set_style("whitegrid")
    plot_data_pivot.plot(kind='bar', ax=ax1)
    ax1.set_title('Mean Comparison by Metric')
    ax1.set_ylabel('Mean Value')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Tenant / Mean')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = axes[0, 1]
    plot_data['abs_percent_diff'] = plot_data['percent_difference'].abs()
    significant = plot_data[plot_data['statistically_significant']]
    not_significant = plot_data[~plot_data['statistically_significant']]
    
    if not significant.empty:
        sns.barplot(
            data=significant,
            x='metric',
            y='percent_difference',
            hue='tenant' if 'tenant' in significant.columns and significant['tenant'].nunique() > 1 else None,
            ax=ax2,
            alpha=0.8
        )
    
    if not not_significant.empty:
        metric_categories = plot_data['metric'].unique()
        metric_map = {metric: i for i, metric in enumerate(metric_categories)}

        for _, row in not_significant.iterrows():
            x_pos = metric_map.get(row['metric'], 0)
            # Adjust x_pos slightly if hue is used to avoid overlap, or handle differently
            # For simplicity, this example doesn't perfectly align scatter with barplot hues.
            ax2.scatter(
                x_pos,
                row['percent_difference'],
                marker='o', s=100, color='gray', alpha=0.5,
                label='Not Significant' if 'Not Significant' not in [h.get_label() for h in ax2.get_legend_handles_labels()[0]] else ""
            )
        ax2.set_xticks(range(len(metric_categories)))
        ax2.set_xticklabels(metric_categories, rotation=45)

    ax2.set_title('Percentage Difference (%)')
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    handles, labels = ax2.get_legend_handles_labels()
    unique_labels = {}
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels[l] = h
    ax2.legend(unique_labels.values(), unique_labels.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax3 = axes[1, 0]
    
    if 'tenant' in plot_data.columns and plot_data['tenant'].nunique() > 1:
        try:
            heatmap_data = plot_data.pivot_table(
                index='metric',
                columns='tenant',
                values='percent_difference'
            )
            
            cmap = sns.diverging_palette(240, 10, as_cmap=True) # Blue-Red diverging palette
            sns.heatmap(
                heatmap_data,
                cmap=cmap,
                center=0,
                annot=True,
                fmt=".1f",
                linewidths=.5,
                ax=ax3,
                cbar_kws={'label': 'Percentage Difference (%)'}
            )
            ax3.set_title('Heatmap of Differences by Tenant and Metric')
        except Exception as e:
            ax3.text(0.5, 0.5, f'Error generating heatmap:\n{e}', 
                horizontalalignment='center', verticalalignment='center')
            ax3.set_title('Heatmap of Differences (Error)')

    else:
        ax3.text(0.5, 0.5, 'Insufficient data for heatmap\n(requires multiple tenants with valid data)', 
                horizontalalignment='center', verticalalignment='center')
        ax3.set_title('Heatmap of Differences (unavailable)')
    
    ax4 = axes[1, 1]
    if not plot_data.empty and 'p_value' in plot_data.columns and 'percent_difference' in plot_data.columns:
        sns.scatterplot(
            data=plot_data,
            x='p_value',
            y='percent_difference',
            hue='metric',
            size='abs_percent_diff',
            sizes=(20, 200),
            alpha=0.7,
            ax=ax4
        )
        
        ax4.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax4.axvline(x=0.05, color='r', linestyle='--', alpha=0.3) # Significance line
        ax4.set_title('Statistical Significance vs. Difference')
        ax4.set_xlabel('p-value (lower = more significant)')
        ax4.set_ylabel('Percentage Difference (%)')
        ax4.set_xscale('log') # p-values are often skewed, log scale helps
        
        # Add text for significance regions if data exists
        min_p_val_for_text = max(plot_data['p_value'].min() * 0.9, 1e-10) if not plot_data['p_value'].empty else 0.001
        y_pos_text = ax4.get_ylim()[1]*0.8 if ax4.get_ylim()[1] > ax4.get_ylim()[0] else 0.8

        ax4.text(min_p_val_for_text if min_p_val_for_text < 0.05 else 0.001, y_pos_text, 'Statistically\nSignificant', fontsize=9, ha='left')
        ax4.text(0.06, y_pos_text, 'Not\nSignificant', fontsize=9, ha='left')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Metric')
    else:
        ax4.text(0.5, 0.5, "Insufficient data for scatter plot", ha='center', va='center')
        ax4.set_title('Statistical Significance vs. Difference (unavailable)')

    plt.tight_layout()
    fig.subplots_adjust(top=0.92) # Adjust for suptitle
    
    return fig

def compare_technologies(exp1_data, exp2_data, metrics_list=None, tenants_list=None,
                        rounds_list: Optional[Union[str, List[str]]] = None, 
                        exp1_name="Technology 1", exp2_name="Technology 2",
                        output_dir=None, generate_plots=True):
    """
    Main function for comparison between experiments with different technologies.
    
    Args:
        exp1_data (dict): Data from the first experiment (dictionary of DataFrames by metric).
        exp2_data (dict): Data from the second experiment (dictionary of DataFrames by metric).
        metrics_list (list): List of metrics for analysis.
        tenants_list (list): List of tenants to filter.
        rounds_list (Optional[Union[str, List[str]]]): List of rounds to filter or a specific round.
        exp1_name (str): Name of the first technology.
        exp2_name (str): Name of the second technology.
        output_dir (str): Directory to save results.
        generate_plots (bool): If True, generates visualizations.
        
    Returns:
        dict: Comparison results.
    """
    results = {}
    
    # Normalize data first
    exp1_norm, exp2_norm, norm_stats = normalize_metrics_between_experiments(
        exp1_data, exp2_data, metrics_list
    )
    
    results['normalized_data'] = {
        exp1_name: exp1_norm,
        exp2_name: exp2_norm
    }
    results['normalization_stats'] = norm_stats
    
    # Define phase names based on typical experiment structure
    # This could be made more dynamic if phases vary greatly
    phase_names = ['1 - Baseline', '2 - Attack', '3 - Recovery'] 
    
    # Calculate efficiency across all specified rounds and phases (overall comparison)
    all_metrics_eff = calculate_relative_efficiency(
        exp1_data, exp2_data, metrics_list, tenants_list,
        phase_filter=None, # Analyze across all phases together
        rounds_filter=rounds_list, 
        exp1_name=exp1_name, exp2_name=exp2_name
    )
    
    # Calculate efficiency for each phase separately, across specified rounds
    phase_metrics_eff = {}
    for phase in phase_names:
        phase_metrics_eff[phase] = calculate_relative_efficiency(
            exp1_data, exp2_data, metrics_list, tenants_list,
            phase_filter=phase, 
            rounds_filter=rounds_list, 
            exp1_name=exp1_name, exp2_name=exp2_name
        )
    
    results['efficiency_metrics'] = {
        'all_phases_and_rounds_specified': all_metrics_eff,
        'by_phase_with_rounds_specified': phase_metrics_eff
    }
    
    if generate_plots and output_dir:
        plots_dir = os.path.join(output_dir, 'technology_comparison_plots') # Changed folder name for clarity
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create a suffix for filenames based on rounds to avoid overwriting
        rounds_suffix = ""
        if rounds_list:
            if isinstance(rounds_list, list):
                # Ensure unique, sorted round names for consistent suffix
                rounds_suffix = "_rounds_" + "_".join(sorted(list(set(str(r) for r in rounds_list))))
            else:
                rounds_suffix = "_round_" + str(rounds_list)
        
        # Plot for overall comparison (all phases)
        fig_all = plot_experiment_comparison(
            all_metrics_eff, exp1_name, exp2_name,
            figsize=(16, 12) # Slightly larger figure for better readability
        )
        
        if fig_all:
            plot_filename = f'comparison_{exp1_name}_vs_{exp2_name}_all_phases{rounds_suffix}.png'
            try:
                fig_all.savefig(os.path.join(plots_dir, plot_filename))
                print(f"Overall comparison plot saved to: {os.path.join(plots_dir, plot_filename)}")
            except Exception as e:
                print(f"Error saving overall comparison plot: {e}")
            plt.close(fig_all) # Close figure to free memory
        
        # Plots for each phase
        for phase, phase_data in phase_metrics_eff.items():
            if phase_data.empty:
                print(f"No data to plot for phase: {phase}{rounds_suffix}")
                continue

            phase_label = phase.replace(' ', '_').replace('-','').lower() # Sanitize phase name for filename
            fig_phase = plot_experiment_comparison(
                phase_data, exp1_name, exp2_name,
                figsize=(16, 12)
            )
            if fig_phase:
                plot_filename = f'comparison_{exp1_name}_vs_{exp2_name}_{phase_label}{rounds_suffix}.png'
                try:
                    fig_phase.savefig(os.path.join(plots_dir, plot_filename))
                    print(f"Plot for phase '{phase}' saved to: {os.path.join(plots_dir, plot_filename)}")
                except Exception as e:
                    print(f"Error saving plot for phase '{phase}': {e}")
                plt.close(fig_phase)
                
    return results
