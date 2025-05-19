"""
Module for comparing different noisy neighbor experiments.

This module implements functions to compare metrics, patterns, and anomalies
across multiple noisy neighbor experiments.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any

from ..data_processing.time_normalization import normalize_time
from ..data_processing.aggregation import aggregate_by_time

def load_multiple_experiments(experiment_paths: List[str]) -> Dict[str, Dict]:
    """
    Loads data from multiple experiments for comparison.
    
    Args:
        experiment_paths (List[str]): List of paths to experiment directories.
        
    Returns:
        Dict[str, Dict]: Dictionary with experiment data and metadata.
    """
    from ..data_processing.consolidation import load_experiment_data
    
    experiments = {}
    
    for i, path in enumerate(experiment_paths):
        try:
            metrics_data, exp_info = load_experiment_data(path)
            
            # Use experiment name if available, otherwise use an ID
            exp_name = exp_info.get('name', f'experiment_{i+1}')
            
            # Store data and metadata
            experiments[exp_name] = {
                'metrics': metrics_data,
                'info': exp_info,
                'path': path
            }
            
        except Exception as e:
            print(f"Error loading experiment at {path}: {str(e)}")
    
    return experiments


def preprocess_experiments(
    experiments: Dict[str, Dict],
    metrics_of_interest: List[str],
    normalize_timestamps: bool = True,
    aggregate_data: bool = True,
    agg_freq: str = '1min',
    rounds_filter: Optional[List[str]] = None,
    tenants_filter: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Preprocesses experiment data for comparison.
    
    Args:
        experiments (Dict[str, Dict]): Dictionary with experiment data and metadata.
        metrics_of_interest (List[str]): List of metrics of interest for comparison.
        normalize_timestamps (bool): If True, normalizes timestamps.
        aggregate_data (bool): If True, aggregates data.
        agg_freq (str): Aggregation frequency (e.g., '1min', '5min').
        rounds_filter (Optional[List[str]]): List of rounds to filter.
        tenants_filter (Optional[List[str]]): List of tenants to filter.
        
    Returns:
        Dict[str, Dict]: Dictionary with preprocessed data.
    """
    # Make a copy to avoid modifying originals
    processed_experiments = {}
    
    for exp_name, exp_data in experiments.items():
        # Filter metrics of interest
        selected_metrics = {k: v for k, v in exp_data['metrics'].items() 
                           if k in metrics_of_interest}
        
        # Process each metric
        processed_metrics = {}
        for metric_name, rounds_data_dict in selected_metrics.items():
            processed_rounds = {}
            if not isinstance(rounds_data_dict, dict):
                print(f"Warning: Expected a dictionary of round DataFrames for metric '{metric_name}' in experiment '{exp_name}', but got {type(rounds_data_dict)}. Skipping this metric.")
                continue

            for round_name, df_round in rounds_data_dict.items():
                # Normalize time if requested
                if normalize_timestamps:
                    if not isinstance(df_round, pd.DataFrame):
                        print(f"Skipping normalization for metric '{metric_name}', round '{round_name}' in exp '{exp_name}': df_round is not a DataFrame, but {type(df_round)}")
                        processed_df_round = df_round 
                        continue
                    processed_df_round = normalize_time(df_round.copy())
                else:
                    processed_df_round = df_round.copy()

                # Apply tenant filter, if provided and 'tenant' column exists
                if tenants_filter and 'tenant' in processed_df_round.columns:
                    processed_df_round = processed_df_round[processed_df_round['tenant'].isin(tenants_filter)]
                
                # Apply round filter, if provided
                if rounds_filter:
                    if 'round' in processed_df_round.columns:
                        processed_df_round = processed_df_round[processed_df_round['round'].isin(rounds_filter)]
                    elif round_name not in rounds_filter:
                        continue

                if processed_df_round.empty:
                    continue
                
                # Add elapsed time in minutes column
                if 'datetime' not in processed_df_round.columns:
                    print(f"Warning: 'datetime' column not found for metric '{metric_name}', round '{round_name}' in exp '{exp_name}'. Skipping elapsed time calculation.")
                else:
                    processed_df_round['elapsed_minutes'] = (
                        pd.to_datetime(processed_df_round['datetime']) - pd.to_datetime(processed_df_round['datetime']).min()
                    ).dt.total_seconds() / 60
                
                # Aggregate data if requested
                if aggregate_data:
                    if 'elapsed_minutes' not in processed_df_round.columns:
                        print(f"Warning: 'elapsed_minutes' column not found for metric '{metric_name}', round '{round_name}' in exp '{exp_name}'. Skipping aggregation.")
                    else:
                        value_col_for_agg = 'value'
                        if value_col_for_agg not in processed_df_round.columns:
                             print(f"Warning: Default value column '{value_col_for_agg}' not found for aggregation in metric '{metric_name}', round '{round_name}', exp '{exp_name}'. Columns: {processed_df_round.columns}. Skipping aggregation.")
                        else:
                            processed_df_round = aggregate_by_time(
                                processed_df_round,
                                time_column='elapsed_minutes',
                                agg_interval=agg_freq,
                                value_column=value_col_for_agg
                            )
                
                processed_rounds[round_name] = processed_df_round
            
            if processed_rounds:
                processed_metrics[metric_name] = processed_rounds
        
        # Store results
        processed_experiments[exp_name] = {
            'processed_metrics': processed_metrics,
            'info': exp_data['info'],
            'path': exp_data.get('path', '')
        }
    
    return processed_experiments


def calculate_statistics_summary(
    experiments: Dict[str, Dict], 
    metrics: List[str],
    group_by: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Calculates summary statistics for comparison between experiments.
    
    Args:
        experiments (Dict[str, Dict]): Dictionary with preprocessed data.
        metrics (List[str]): List of metrics for analysis.
        group_by (List[str]): Columns to group data by (e.g., ['tenant', 'phase']).
        
    Returns:
        Dict[str, DataFrame]: DataFrames with statistics per metric.
    """
    results = {}
    
    for metric in metrics:
        all_stats_for_metric = []
        
        for exp_name, exp_data in experiments.items():
            if metric in exp_data['processed_metrics']:
                rounds_data_dict = exp_data['processed_metrics'][metric]
                if not isinstance(rounds_data_dict, dict):
                    print(f"Warning: Expected a dictionary of round DataFrames for metric '{metric}' in experiment '{exp_name}', but got {type(rounds_data_dict)}. Skipping this metric for this experiment.")
                    continue

                for round_name, df_round in rounds_data_dict.items():
                    if not isinstance(df_round, pd.DataFrame):
                        print(f"Warning: Expected a DataFrame for round '{round_name}' of metric '{metric}' in experiment '{exp_name}', but got {type(df_round)}. Skipping this round.")
                        continue
                    
                    if df_round.empty:
                        continue

                    value_col_to_use = None
                    if 'mean' in df_round.columns:
                        value_col_to_use = 'mean'
                    elif 'value' in df_round.columns:
                        value_col_to_use = 'value'
                    else:
                        print(f"Warning: Neither 'mean' nor 'value' column found in DataFrame for round '{round_name}', metric '{metric}', experiment '{exp_name}'. Columns: {df_round.columns}. Skipping stats for this round.")
                        continue

                    current_group_by = group_by.copy() if group_by else []
                    valid_group_by = [gb for gb in current_group_by if gb in df_round.columns]
                    if not valid_group_by and current_group_by:
                        print(f"Warning: Group_by columns {current_group_by} not found in df_round for metric '{metric}', round '{round_name}'. Calculating global stats.")

                    if valid_group_by:
                        stats_df = df_round.groupby(valid_group_by)[value_col_to_use].agg([
                            ('mean', 'mean'),
                            ('median', 'median'),
                            ('std_dev', 'std'),
                            ('min', 'min'),
                            ('max', 'max'),
                            ('count', 'count')
                        ]).reset_index()
                    else:
                        stats_df = pd.DataFrame({
                            'mean': [df_round[value_col_to_use].mean()],
                            'median': [df_round[value_col_to_use].median()],
                            'std_dev': [df_round[value_col_to_use].std()],
                            'min': [df_round[value_col_to_use].min()],
                            'max': [df_round[value_col_to_use].max()],
                            'count': [df_round[value_col_to_use].count()]
                        })
                    
                    stats_df['experiment'] = exp_name
                    stats_df['round'] = round_name
                    
                    all_stats_for_metric.append(stats_df)
        
        if all_stats_for_metric:
            results[metric] = pd.concat(all_stats_for_metric, ignore_index=True)
    
    return results


def compare_distributions(
    experiments: Dict[str, Dict],
    metric: str,
    tenants_filter: Optional[List[str]] = None,
    phase: Optional[str] = None,
    test_method: str = 'ks',
    rounds_filter: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], Dict[str, Dict]]:
    """
    Compares statistical distributions between experiments for a metric.
    
    Args:
        experiments (Dict[str, Dict]): Dictionary with preprocessed data.
        metric (str): Metric for comparison.
        tenants_filter (Optional[List[str]]): List of specific tenants to filter.
        phase (str, optional): Specific phase to filter.
        test_method (str): Statistical test method ('ks' or 'mw').
        rounds_filter (Optional[List[str]]): List of rounds to filter.
        
    Returns:
        Tuple[Dict[str, Any], Dict[str, Dict]]: 
            - Structured data for plotting.
            - Results of statistical tests.
    """
    plot_data = {'series': [], 'labels': []}
    test_results = {}
    
    # Collect time series for comparison
    for exp_name, exp_data in experiments.items():
        if metric in exp_data.get('processed_metrics', {}):
            rounds_data_dict = exp_data['processed_metrics'][metric]
            if not isinstance(rounds_data_dict, dict):
                continue

            all_rounds_df_list = []
            for round_name, df_round in rounds_data_dict.items():
                if not isinstance(df_round, pd.DataFrame):
                    continue
                
                df_round_filtered = df_round.copy()
                if tenants_filter and 'tenant' in df_round_filtered.columns:
                    df_round_filtered = df_round_filtered[df_round_filtered['tenant'].isin(tenants_filter)]
                
                if rounds_filter:
                    if 'round' in df_round_filtered.columns:
                        df_round_filtered = df_round_filtered[df_round_filtered['round'].isin(rounds_filter)]
                    elif round_name not in rounds_filter:
                        continue 

                if phase and 'phase' in df_round_filtered.columns:
                    df_round_filtered = df_round_filtered[df_round_filtered['phase'] == phase]
                
                if not df_round_filtered.empty:
                    all_rounds_df_list.append(df_round_filtered)
            
            if not all_rounds_df_list:
                continue
            
            df_combined = pd.concat(all_rounds_df_list, ignore_index=True)
            value_col_for_dist = None
            if 'mean' in df_combined.columns:
                value_col_for_dist = 'mean'
            elif 'value' in df_combined.columns:
                value_col_for_dist = 'value'
            else:
                continue

            if not df_combined.empty and value_col_for_dist in df_combined.columns:
                if not df_combined[value_col_for_dist].isnull().all():
                    series_to_add = df_combined[value_col_for_dist].dropna()
                    if not series_to_add.empty:
                        plot_data['series'].append(series_to_add)
                        plot_data['labels'].append(exp_name)
    
    # Perform statistical tests between pairs of experiments
    n_series = len(plot_data['series'])
    if n_series >= 2:
        for i in range(n_series):
            for j in range(i+1, n_series):
                exp1_label = plot_data['labels'][i]
                exp2_label = plot_data['labels'][j]
                
                series1 = plot_data['series'][i]
                series2 = plot_data['series'][j]

                if series1.empty or series2.empty:
                    stat, pval = np.nan, np.nan
                    test_name = "Skipped (empty series)"
                elif test_method == 'ks':
                    stat, pval = stats.ks_2samp(series1, series2)
                    test_name = "Kolmogorov-Smirnov"
                elif test_method == 'mw':
                    stat, pval = stats.mannwhitneyu(series1, series2, alternative='two-sided')
                    test_name = "Mann-Whitney U"
                else:
                    stat, pval = stats.ks_2samp(series1, series2)
                    test_name = "Kolmogorov-Smirnov (defaulted)"
                
                test_results[f"{exp1_label}_vs_{exp2_label}"] = {
                    "test_method": test_name,
                    "statistic": stat,
                    "p_value": pval,
                    "significant_difference": pval < 0.05
                }
    
    return plot_data, test_results


def detect_anomalies_across_experiments(
    experiments: Dict[str, Dict],
    metrics: List[str],
    contamination: float = 0.05,
    group_by: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Detects and compares anomalies across experiments.
    
    Args:
        experiments (Dict[str, Dict]): Dictionary with preprocessed data.
        metrics (List[str]): List of metrics for analysis.
        contamination (float): Expected proportion of anomalies in the data.
        group_by (List[str]): Columns to group data by (e.g., ['tenant', 'phase']).
        
    Returns:
        Dict[str, Dict[str, Any]]: Anomaly detection results per experiment and metric.
    """
    from .anomaly_detection import detect_anomalies_ensemble
    
    anomaly_results = {}
    
    for exp_name, exp_data in experiments.items():
        anomaly_results[exp_name] = {}
        
        for metric_name in metrics:
            if metric_name in exp_data.get('processed_metrics', {}):
                metric_data_for_exp = exp_data['processed_metrics'][metric_name]
                
                df_for_anomaly_detection = None
                if isinstance(metric_data_for_exp, dict):
                    all_rounds_list = []
                    for round_df_iter in metric_data_for_exp.values():
                        if isinstance(round_df_iter, pd.DataFrame) and not round_df_iter.empty:
                            all_rounds_list.append(round_df_iter)
                    if all_rounds_list:
                        df_for_anomaly_detection = pd.concat(all_rounds_list, ignore_index=True)
                    else:
                        continue 
                elif isinstance(metric_data_for_exp, pd.DataFrame):
                    df_for_anomaly_detection = metric_data_for_exp
                else:
                    continue

                if df_for_anomaly_detection is None or df_for_anomaly_detection.empty:
                    continue

                value_col_for_anomaly = 'mean' if 'mean' in df_for_anomaly_detection.columns else 'value'
                if value_col_for_anomaly not in df_for_anomaly_detection.columns:
                    continue
                
                if 'elapsed_minutes' not in df_for_anomaly_detection.columns:
                    continue

                try:
                    df_with_anomalies, change_info = detect_anomalies_ensemble(
                        df_for_anomaly_detection, 
                        metric_column=value_col_for_anomaly,
                        time_column='elapsed_minutes',
                        contamination=contamination,
                        group_by=group_by
                    )
                    
                    anomaly_results[exp_name][metric_name] = {
                        'dataframe_with_anomalies': df_with_anomalies,
                        'change_point_info': change_info
                    }
                except Exception as e:
                    print(f"Error during anomaly detection for metric '{metric_name}' in experiment '{exp_name}': {e}")
    
    return anomaly_results


def summarize_anomalies(anomaly_results: Dict[str, Dict[str, Dict[str, Any]]]) -> pd.DataFrame:
    """
    Generates a summary of detected anomalies for comparison.
    
    Args:
        anomaly_results (Dict): Anomaly detection results from detect_anomalies_across_experiments.
        
    Returns:
        DataFrame: Summary of anomalies per experiment, metric, and group.
    """
    anomaly_summary = []
    
    for exp_name, metrics_results in anomaly_results.items():
        for metric, results in metrics_results.items():
            df = results['dataframe_with_anomalies']
            
            total_points = len(df)
            anomaly_if_count = df['is_anomaly_if'].sum() if 'is_anomaly_if' in df.columns else 0
            anomaly_lof_count = df['is_anomaly_lof'].sum() if 'is_anomaly_lof' in df.columns else 0
            anomaly_ensemble_count = df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0
            change_point_count = df['is_change_point'].sum() if 'is_change_point' in df.columns else 0
            
            anomaly_summary.append({
                'experiment': exp_name,
                'metric': metric,
                'group': 'global',
                'total_points': total_points,
                'anomalies_if': anomaly_if_count,
                'anomalies_lof': anomaly_lof_count,
                'anomalies_ensemble': anomaly_ensemble_count,
                'change_points': change_point_count,
                'pct_anomalies': (anomaly_ensemble_count / total_points * 100) if total_points > 0 else 0
            })
            
            for tenant in df['tenant'].unique():
                tenant_df = df[df['tenant'] == tenant]
                total_points = len(tenant_df)
                anomaly_if_count = tenant_df['is_anomaly_if'].sum() if 'is_anomaly_if' in tenant_df.columns else 0
                anomaly_lof_count = tenant_df['is_anomaly_lof'].sum() if 'is_anomaly_lof' in tenant_df.columns else 0
                anomaly_ensemble_count = tenant_df['is_anomaly'].sum() if 'is_anomaly' in tenant_df.columns else 0
                change_point_count = tenant_df['is_change_point'].sum() if 'is_change_point' in tenant_df.columns else 0
                
                anomaly_summary.append({
                    'experiment': exp_name,
                    'metric': metric,
                    'group': tenant,
                    'total_points': total_points,
                    'anomalies_if': anomaly_if_count,
                    'anomalies_lof': anomaly_lof_count,
                    'anomalies_ensemble': anomaly_ensemble_count,
                    'change_points': change_point_count,
                    'pct_anomalies': (anomaly_ensemble_count / total_points * 100) if total_points > 0 else 0
                })
    
    return pd.DataFrame(anomaly_summary)


def compare_experiment_phases(
    experiments: Dict[str, Dict],
    metrics: List[str],
    phases_to_compare: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compares the impact of different phases between experiments.
    
    Args:
        experiments (Dict[str, Dict]): Dictionary with preprocessed data.
        metrics (List[str]): List of metrics for analysis.
        phases_to_compare (List[str], optional): Specific list of phases to compare.
        
    Returns:
        Dict[str, Dict[str, Any]]: Results of phase comparisons.
    """
    from pipeline.analysis.phase_analysis import compare_phases
    
    phase_comparisons = {}
    
    for exp_name, exp_data in experiments.items():
        phase_comparisons[exp_name] = {}
        
        for metric in metrics:
            if metric in exp_data['processed_metrics']:
                df = exp_data['processed_metrics'][metric]
                
                if phases_to_compare is None:
                    phases = sorted(df['phase'].unique())
                else:
                    phases = [p for p in phases_to_compare if p in df['phase'].unique()]
                
                if len(phases) < 2:
                    continue
                
                try:
                    phase_comparison_results = compare_phases(
                        df, phases=phases, metric_column='value'
                    )
                    
                    phase_comparisons[exp_name][metric] = phase_comparison_results
                except Exception as e:
                    print(f"Error comparing phases for {metric} in {exp_name}: {str(e)}")
    
    return phase_comparisons


def compare_time_series_similarity(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    method: str = 'rmse',
    time_column: str = 'elapsed_minutes',
    value_column: str = 'value',
    tenant_column: str = 'tenant'
) -> float:
    """
    Compares the similarity between two time series.
    
    Args:
        df1 (DataFrame): First DataFrame with time series data.
        df2 (DataFrame): Second DataFrame with time series data.
        method (str): Similarity method ('rmse', 'mae', 'correlation').
        time_column (str): Name of the time column.
        value_column (str): Name of the value column.
        tenant_column (str): Name of the tenant column.
        
    Returns:
        float: Similarity value.
    """
    tenants1 = set(df1[tenant_column].unique())
    tenants2 = set(df2[tenant_column].unique())
    common_tenants = list(tenants1.intersection(tenants2))
    
    if not common_tenants:
        return np.nan
    
    tenant_similarities = []
    
    for tenant in common_tenants:
        series1 = df1[df1[tenant_column] == tenant].sort_values(time_column)
        series2 = df2[df2[tenant_column] == tenant].sort_values(time_column)
        
        if len(series1) < 5 or len(series2) < 5:
            continue
        
        min_time = max(series1[time_column].min(), series2[time_column].min())
        max_time = min(series1[time_column].max(), series2[time_column].max())
        
        if min_time >= max_time:
            continue
            
        time_grid = np.linspace(min_time, max_time, 100)
        
        values1 = np.interp(time_grid, series1[time_column], series1[value_column])
        values2 = np.interp(time_grid, series2[time_column], series2[value_column])
        
        if method == 'rmse':
            similarity = mean_squared_error(values1, values2, squared=False)
        elif method == 'mae':
            similarity = mean_absolute_error(values1, values2)
        elif method == 'correlation':
            similarity = np.corrcoef(values1, values2)[0, 1]
        else:
            raise ValueError(f"Unknown similarity method: {method}")
            
        tenant_similarities.append(similarity)
    
    if not tenant_similarities:
        return np.nan
    
    return np.nanmean(tenant_similarities)


def find_common_elements(experiments: Dict[str, Dict], key: str) -> List[Any]:
    """
    Finds common elements across experiments (e.g., tenants, phases).
    
    Args:
        experiments (Dict[str, Dict]): Dictionary with experiment data.
        key (str): Key to search in metadata ('tenants', 'phases', etc.).
        
    Returns:
        List[Any]: List of common elements.
    """
    common_elements = None
    
    for exp_data in experiments.values():
        if key in exp_data.get('info', {}):
            elements = set(exp_data['info'][key])
            
            if common_elements is None:
                common_elements = elements
            else:
                common_elements = common_elements.intersection(elements)
    
    return list(common_elements) if common_elements else []
