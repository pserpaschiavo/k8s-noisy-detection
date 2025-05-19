"""
Module for aggregating data across multiple experiment rounds.

This module provides functions to consolidate data from different rounds,
allowing for an average view of tenant behavior across multiple executions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def aggregate_metrics_across_rounds(metrics_data, value_column='value', include_std=True):
    """
    Aggregates metrics across multiple rounds, calculating means and standard deviations.
    Time data is maintained relative to the global experiment time.
    
    Args:
        metrics_data (dict): Dictionary with DataFrames for each metric.
        value_column (str): Name of the column with values to be aggregated.
        include_std (bool): If True, includes standard deviation in the results.
        
    Returns:
        dict: Dictionary with aggregated DataFrames for each metric.
    """
    aggregated_metrics = {}
    
    for metric_name, df in metrics_data.items():
        if 'round' not in df.columns:
            print(f"Warning: 'round' column not found in data for metric {metric_name}. Skipping aggregation for this metric.")
            continue
        
        df_clean = df.copy()
        
        # Ensure required columns are present
        required_cols_for_agg = ['phase', 'tenant', 'experiment_elapsed_seconds', value_column]
        if not all(col in df_clean.columns for col in required_cols_for_agg):
            print(f"Warning: Missing one or more required columns ({required_cols_for_agg}) for metric {metric_name}. Skipping.")
            continue

        # Discretize global experiment time for alignment across rounds
        df_clean['experiment_elapsed_seconds'] = pd.to_numeric(df_clean['experiment_elapsed_seconds'], errors='coerce')
        df_clean.dropna(subset=['experiment_elapsed_seconds'], inplace=True)
        if df_clean.empty:
            print(f"Warning: Data for metric {metric_name} became empty after handling NaNs in 'experiment_elapsed_seconds'. Skipping.")
            continue
        
        # Use global experiment time for binning
        df_clean['time_bin_global'] = (df_clean['experiment_elapsed_seconds'] // 5) * 5
            
        df_clean[value_column] = pd.to_numeric(df_clean[value_column], errors='coerce')
        df_clean.dropna(subset=[value_column], inplace=True)
        if df_clean.empty:
            print(f"Warning: Data for metric {metric_name} became empty after handling NaNs in '{value_column}'. Skipping.")
            continue

        # Keep 'phase' for grouping, but time is global
        grouping_cols = ['phase', 'tenant', 'time_bin_global']
        if 'labels' in df_clean.columns:
            grouping_cols.append('labels')

        all_tenants_for_this_metric = df_clean['tenant'].unique() # Use tenants present in the current df_clean

        if not df_clean.empty:
            levels_for_product = []
            
            phases_in_data = df_clean['phase'].unique()
            levels_for_product.append(phases_in_data)
            
            levels_for_product.append(all_tenants_for_this_metric)
            
            time_bins_in_data = df_clean['time_bin_global'].unique()
            levels_for_product.append(time_bins_in_data)
            
            if 'labels' in grouping_cols:
                labels_in_data = df_clean['labels'].unique()
                levels_for_product.append(labels_in_data)
            
            if not all(len(level) > 0 for level in levels_for_product):
                print(f"Warning: Cannot create full product index for metric {metric_name} due to empty levels. Skipping zero-filling.")
            else:
                full_multi_idx = pd.MultiIndex.from_product(levels_for_product, names=grouping_cols)
                df_template = pd.DataFrame(index=full_multi_idx).reset_index()
                df_clean = pd.merge(df_template, df_clean, on=grouping_cols, how='left')
        
        grouped = df_clean.groupby(grouping_cols)[value_column]
        
        agg_funcs = ['mean', 'count', 'min', 'max']
        if include_std:
            agg_funcs.append('std')
        
        agg_results = grouped.agg(agg_funcs).reset_index()
        
        cols_to_fill_na_with_zero = ['mean', 'min', 'max']
        if include_std and 'std' in agg_results.columns:
            cols_to_fill_na_with_zero.append('std')

        for col_name in cols_to_fill_na_with_zero:
            if col_name in agg_results.columns:
                agg_results[col_name] = agg_results[col_name].fillna(0)
        
        if include_std and 'std' in agg_results.columns and 'count' in agg_results.columns:
            valid_counts = agg_results['count'].replace(0, np.nan).fillna(np.nan)
            agg_results['ci_95'] = 1.96 * agg_results['std'] / np.sqrt(valid_counts)
            
            if 'ci_95' not in agg_results:
                agg_results['ci_95'] = np.nan # Ensure column exists
            agg_results.loc[:, 'ci_95'] = agg_results['ci_95'].fillna(0)
        
        # The time column for plotting is now 'time_bin_global'
        agg_results['experiment_elapsed_seconds'] = agg_results['time_bin_global']
        # Remove the 'time_bin_global' column if no longer needed directly,
        # or keep it if useful for other analyses.
        # agg_results.drop(columns=['time_bin_global'], inplace=True, errors='ignore')
            
        aggregated_metrics[metric_name] = agg_results
            
    return aggregated_metrics

def plot_aggregated_metrics(aggregated_data_map, metric_name, figsize=(15, 8), 
                            show_confidence=True, highlight_tenant=None, 
                            all_metrics_data_for_phases=None):
    """
    Plots aggregated metrics across rounds, showing mean and variability over global experiment time.
    
    Args:
        aggregated_data_map (dict): Dictionary of aggregated DataFrames.
        metric_name (str): Name of the metric for the plot title.
        figsize (tuple): Size of the plot.
        show_confidence (bool): If True, shows confidence interval.
        highlight_tenant (str): Tenant to be highlighted.
        all_metrics_data_for_phases (dict): Original data dictionary (before aggregation)
                                            to get global phase start times.
    
    Returns:
        matplotlib.figure.Figure: Created figure object, or None.
    """
    from pipeline.config import TENANT_COLORS, PHASE_DISPLAY_NAMES, METRIC_DISPLAY_NAMES
    
    if metric_name not in aggregated_data_map or aggregated_data_map[metric_name].empty:
        print(f"No aggregated data found for metric: {metric_name}")
        return None
    
    df_agg = aggregated_data_map[metric_name]
    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    
    if 'mean' not in df_agg.columns or 'experiment_elapsed_seconds' not in df_agg.columns:
        print(f"Warning: 'mean' or 'experiment_elapsed_seconds' column not found in aggregated data for {metric_name}. Cannot plot.")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    
    # Get global phase start times from the original (non-aggregated) dataset
    # This is important because aggregation can change time granularity.
    phase_start_times_global = {}
    if all_metrics_data_for_phases and metric_name in all_metrics_data_for_phases:
        # Concatenate all rounds for the metric to find global minimum phase times
        # Assuming all_metrics_data_for_phases[metric_name] is a dict of DFs per round
        # or a single DF if no rounds (though for aggregation, we expect multiple rounds)
        metric_all_rounds_df_list = []
        if isinstance(all_metrics_data_for_phases[metric_name], dict):
            for round_df_original in all_metrics_data_for_phases[metric_name].values():
                if isinstance(round_df_original, pd.DataFrame) and 'phase' in round_df_original.columns and 'experiment_elapsed_seconds' in round_df_original.columns:
                    metric_all_rounds_df_list.append(round_df_original[['phase', 'experiment_elapsed_seconds']])
        elif isinstance(all_metrics_data_for_phases[metric_name], pd.DataFrame):
             metric_df_original = all_metrics_data_for_phases[metric_name]
             if 'phase' in metric_df_original.columns and 'experiment_elapsed_seconds' in metric_df_original.columns:
                metric_all_rounds_df_list.append(metric_df_original[['phase', 'experiment_elapsed_seconds']])

        if metric_all_rounds_df_list:
            combined_original_df = pd.concat(metric_all_rounds_df_list).drop_duplicates()
            phase_starts_raw = combined_original_df.groupby('phase')['experiment_elapsed_seconds'].min().to_dict()
            # Use PHASE_DISPLAY_NAMES to sort and name correctly
            for phase_key, display_name in PHASE_DISPLAY_NAMES.items():
                if phase_key in phase_starts_raw:
                    phase_start_times_global[display_name] = phase_starts_raw[phase_key]
        else:
            print(f"Warning: Could not determine global phase start times for {metric_name} from all_metrics_data_for_phases.")

    # Plot aggregated data
    legend_tenants_added = set()
    unique_phases_in_agg = sorted(df_agg['phase'].unique(), key=lambda p: list(PHASE_DISPLAY_NAMES.keys()).index(p) if p in PHASE_DISPLAY_NAMES else float('inf'))

    for tenant_val in sorted(df_agg['tenant'].unique()):
        tenant_data_full = df_agg[df_agg['tenant'] == tenant_val].sort_values('experiment_elapsed_seconds')
        if tenant_data_full.empty:
            continue

        line_width = 2.5 if tenant_val == highlight_tenant else 1.5
        color = TENANT_COLORS.get(tenant_val, 'gray')
        
        # Plot the entire line for the tenant across all phases
        ax.plot(tenant_data_full['experiment_elapsed_seconds'], tenant_data_full['mean'], 
               label=tenant_val if tenant_val not in legend_tenants_added else "_nolegend_",
               color=color, linewidth=line_width, alpha=1.0)
        legend_tenants_added.add(tenant_val)
        
        if show_confidence and 'ci_95' in tenant_data_full.columns:
            ax.fill_between(tenant_data_full['experiment_elapsed_seconds'],
                           tenant_data_full['mean'] - tenant_data_full['ci_95'],
                           tenant_data_full['mean'] + tenant_data_full['ci_95'],
                           color=color, alpha=0.2)

    # Add global phase markers
    if phase_start_times_global:
        sorted_phase_starts = sorted(phase_start_times_global.items(), key=lambda item: item[1])
        for i, (phase_display_name, start_time) in enumerate(sorted_phase_starts):
            if i > 0: # Do not mark the start of the first phase if it's time 0
                ax.axvline(x=start_time, color='black', linestyle=':', alpha=0.8, linewidth=1.2)
            ax.text(start_time + 5, ax.get_ylim()[1] * 0.98, phase_display_name, 
                    rotation=0, verticalalignment='top', horizontalalignment='left', 
                    fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5, pad=0.2, edgecolor='none'))
    
    ax.set_title(f"Average {display_metric_name} Across Rounds (Global Experiment Time)")
    ax.set_xlabel("Experiment Time (seconds)")
    ax.set_ylabel(f"Average Value of {display_metric_name}")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    handles, labels = ax.get_legend_handles_labels()
    filtered_handles_labels = [(h, l) for h, l in zip(handles, labels) if l != "_nolegend_"]
    if filtered_handles_labels:
        handles_final, labels_final = zip(*filtered_handles_labels)
        by_label = dict(zip(labels_final, handles_final))
        ax.legend(by_label.values(), by_label.keys(), title="Tenant", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust to make space for the external legend
    return fig

def plot_aggregated_metrics_boxplot(aggregated_data_map, metric_name, output_dir_base, figsize=(12, 8)):
    """
    Generates a single box plot per metric, showing the distribution by tenant for each phase,
    with phases on the X-axis and tenants differentiated by color.

    Args:
        aggregated_data_map (dict): Dictionary of aggregated DataFrames.
        metric_name (str): Name of the metric for the plot title.
        output_dir_base (str): Base directory to save plots (e.g., advanced_plots_dir).
        figsize (tuple): Size of the plot.
    """
    from pipeline.config import TENANT_COLORS, PHASE_DISPLAY_NAMES, METRIC_DISPLAY_NAMES, VISUALIZATION_CONFIG
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt

    if metric_name not in aggregated_data_map or aggregated_data_map[metric_name].empty:
        print(f"No aggregated data found for metric: {metric_name} for boxplot.")
        return

    df_agg = aggregated_data_map[metric_name].copy() # Use a copy to avoid SettingWithCopyWarning
    display_metric_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)

    if 'mean' not in df_agg.columns or 'phase' not in df_agg.columns or 'tenant' not in df_agg.columns:
        print(f"Warning: 'mean', 'phase', or 'tenant' column not found in aggregated data for {metric_name}. Cannot create boxplot.")
        return

    # Create subdirectory for aggregated boxplots if it doesn't exist
    boxplot_output_dir = os.path.join(output_dir_base, "aggregated_boxplots")
    os.makedirs(boxplot_output_dir, exist_ok=True)

    # Map 'phase' to 'phase_display' using PHASE_DISPLAY_NAMES
    # and ensure correct order of phases in the plot.
    df_agg['phase_display'] = df_agg['phase'].map(PHASE_DISPLAY_NAMES)
    
    # Filter any rows where phase_display might be NaN (if a phase in data is not in PHASE_DISPLAY_NAMES)
    df_agg_filtered = df_agg.dropna(subset=['phase_display'])
    if df_agg_filtered.empty:
        print(f"No data with displayable phases for metric {metric_name} after filtering. Skipping boxplot.")
        return

    # Determine the order of phases based on keys of PHASE_DISPLAY_NAMES present in the data
    # and whose mapped values (display names) exist in df_agg_filtered['phase_display']
    phase_keys_in_data = df_agg_filtered['phase'].unique()
    ordered_phase_display_names = [
        PHASE_DISPLAY_NAMES[key] for key in PHASE_DISPLAY_NAMES.keys()
        if key in phase_keys_in_data and PHASE_DISPLAY_NAMES[key] in df_agg_filtered['phase_display'].unique()
    ]

    if not ordered_phase_display_names:
        # Fallback if explicit ordering fails: use unique display names, sorted alphabetically
        print(f"Warning: Could not determine explicit order for phases in metric {metric_name}. Using unique phase display names sorted alphabetically.")
        ordered_phase_display_names = sorted(df_agg_filtered['phase_display'].unique())

    if not ordered_phase_display_names:
        print(f"No phases to plot for metric {metric_name}. Skipping boxplot.")
        return

    plt.figure(figsize=figsize)
    sns.boxplot(x='phase_display', y='mean', hue='tenant', data=df_agg_filtered,
                order=ordered_phase_display_names, palette=TENANT_COLORS,
                showfliers=True) # legend is controlled by hue automatically

    plt.title(f"Distribution of {display_metric_name}\nBy Phase and Tenant")
    plt.xlabel("Experiment Phase")
    plt.ylabel(f"Aggregated Mean Value of {display_metric_name}")
    plt.xticks(rotation=15, ha="right") # Slight rotation for better visualization
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust legend to be outside the plot
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles: # Only add legend if there is something to legend
        # Filter duplicate legend entries if sns.boxplot creates them
        unique_entries = {}
        for handle, label in zip(handles, labels):
            if label not in unique_entries:
                unique_entries[label] = handle
        
        if unique_entries:
             plt.legend(unique_entries.values(), unique_entries.keys(), title="Tenant", bbox_to_anchor=(1.02, 1), loc='upper left')
             plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust to make space for the external legend
        else:
             plt.tight_layout() # If no legend to show
    else:
        plt.tight_layout()

    plot_filename = f"boxplot_aggregated_{metric_name}_all_phases_by_tenant.png"
    plot_path = os.path.join(boxplot_output_dir, plot_filename)

    try:
        plt.savefig(plot_path, dpi=VISUALIZATION_CONFIG.get('dpi', 300))
        print(f"    Aggregated boxplot for {metric_name} (all phases) saved to: {plot_path}")
    except Exception as e:
        print(f"    Error saving aggregated boxplot for {metric_name} (all phases): {e}")
    plt.close()

def test_for_significant_differences(metrics_data_dict, metric_name, phase_name, 
                                     tenant1, tenant2, alpha=0.05, value_column='value'):
    """
    Tests if there are statistically significant differences between two tenants for a metric and phase,
    using data that may come from multiple rounds (requires 'round' and 'value_column').
    
    Args:
        metrics_data_dict (dict): Dictionary where keys are metric names and values are DataFrames.
                                  Each DataFrame must contain 'round', 'phase', 'tenant', and 'value_column' columns.
        metric_name (str): Name of the metric to test.
        phase_name (str): Name of the phase to test.
        tenant1 (str): First tenant for comparison.
        tenant2 (str): Second tenant for comparison.
        alpha (float): Significance level for the test.
        value_column (str): Column containing the values to compare.
        
    Returns:
        dict: Test results, including p-value and whether there is a significant difference.
    """
    if metric_name not in metrics_data_dict:
        return {"error": f"Metric {metric_name} not found in the provided data."}
    
    df_metric = metrics_data_dict[metric_name]
    
    required_cols = ['round', 'phase', 'tenant', value_column]
    if not all(col in df_metric.columns for col in required_cols):
        return {"error": f"DataFrame for {metric_name} does not contain all required columns: {required_cols}."}

    data_t1_all_rounds = df_metric[
        (df_metric['phase'] == phase_name) & (df_metric['tenant'] == tenant1)
    ][value_column].dropna()
    
    data_t2_all_rounds = df_metric[
        (df_metric['phase'] == phase_name) & (df_metric['tenant'] == tenant2)
    ][value_column].dropna()
    
    if data_t1_all_rounds.empty or data_t2_all_rounds.empty:
        return {
            "error": f"Insufficient data for one or both tenants ({tenant1}, {tenant2}) in phase {phase_name} for metric {metric_name}.",
            "tenant1_count": len(data_t1_all_rounds),
            "tenant2_count": len(data_t2_all_rounds)
        }
    
    t_stat, p_value = stats.ttest_ind(data_t1_all_rounds, data_t2_all_rounds, equal_var=False) # Welch's t-test
    
    mean1, mean2 = data_t1_all_rounds.mean(), data_t2_all_rounds.mean()
    std1, std2 = data_t1_all_rounds.std(), data_t2_all_rounds.std()
    n1, n2 = len(data_t1_all_rounds), len(data_t2_all_rounds)
    
    if n1 + n2 - 2 <= 0: # Degrees of freedom for pooled_std must be > 0
        cohen_d = float('nan')
        effect_size_interpretation = "N/A (pooled std calculation not possible)"
    else:
        # Calculate pooled standard deviation for Cohen's d
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        if pooled_std == 0: # Avoid division by zero
             cohen_d = float('inf') if abs(mean1 - mean2) > 0 else 0.0
        else:
            cohen_d = abs(mean1 - mean2) / pooled_std
        
        if pd.isna(cohen_d):
            effect_size_interpretation = "N/A (Cohen's d is NaN)"
        elif cohen_d < 0.2:
            effect_size_interpretation = "trivial"
        elif cohen_d < 0.5:
            effect_size_interpretation = "small"
        elif cohen_d < 0.8:
            effect_size_interpretation = "medium"
        else:
            effect_size_interpretation = "large"
            
    return {
        "metric": metric_name,
        "phase": phase_name,
        "tenant1": tenant1,
        "tenant2": tenant2,
        "mean_tenant1": mean1,
        "mean_tenant2": mean2,
        "std_tenant1": std1,
        "std_tenant2": std2,
        "n_tenant1": n1,
        "n_tenant2": n2,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant_at_alpha_" + str(alpha): p_value < alpha,
        "mean_difference": mean1 - mean2,
        "cohen_d": cohen_d,
        "effect_size_interpretation": effect_size_interpretation
    }
