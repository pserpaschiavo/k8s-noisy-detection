#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Analysis for Kubernetes Noisy Neighbours Lab
This module provides statistical analysis of metrics data.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller, kpss
import logging
from pathlib import Path

class StatsAnalyzer:
    def __init__(self, output_dir=None):
        """
        Initialize the statistical analyzer.
        
        Args:
            output_dir (str): Directory to save results
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        logging.info(f"Initialized StatsAnalyzer, output directory: {self.output_dir}")
    
    def get_descriptive_stats(self, data):
        """
        Calculate descriptive statistics for the data.
        
        Args:
            data (DataFrame): Input data
            
        Returns:
            DataFrame: DataFrame with descriptive statistics
        """
        if data is None or data.empty:
            return pd.DataFrame()
        
        # Handle DataFrames or Series
        if isinstance(data, pd.DataFrame):
            # Calculate descriptive statistics
            stats_df = data.describe()
            
            # Add additional statistics
            for col in data.columns:
                numeric_data = pd.to_numeric(data[col], errors='coerce')
                stats_df.loc['skewness', col] = stats.skew(numeric_data.dropna())
                stats_df.loc['kurtosis', col] = stats.kurtosis(numeric_data.dropna())
                stats_df.loc['median', col] = numeric_data.median()
                stats_df.loc['iqr', col] = numeric_data.quantile(0.75) - numeric_data.quantile(0.25)
                stats_df.loc['cv', col] = numeric_data.std() / numeric_data.mean() if numeric_data.mean() != 0 else np.nan
        else:
            # If input is a Series, convert to DataFrame
            numeric_data = pd.to_numeric(data, errors='coerce').dropna()
            stats_dict = {
                'count': len(numeric_data),
                'mean': numeric_data.mean(),
                'std': numeric_data.std(),
                'min': numeric_data.min(),
                '25%': numeric_data.quantile(0.25),
                'median': numeric_data.median(),
                '75%': numeric_data.quantile(0.75),
                'max': numeric_data.max(),
                'skewness': stats.skew(numeric_data),
                'kurtosis': stats.kurtosis(numeric_data),
                'iqr': numeric_data.quantile(0.75) - numeric_data.quantile(0.25),
                'cv': numeric_data.std() / numeric_data.mean() if numeric_data.mean() != 0 else np.nan
            }
            stats_df = pd.DataFrame(stats_dict, index=[data.name if hasattr(data, 'name') else 'value']).T
        
        return stats_df
    
    def analyze_phase_metrics(self, phase_data, phase_name):
        """
        Analyze all metrics for a specific phase.
        
        Args:
            phase_data (dict): Dictionary with component names as keys and data dictionaries as values
            phase_name (str): Name of the phase
            
        Returns:
            dict: Dictionary with component names as keys and statistics DataFrames as values
        """
        results = {}
        
        for component, metrics in phase_data.items():
            logging.info(f"Analyzing metrics for {phase_name}/{component}")
            component_stats = {}
            
            for metric_name, metric_data in metrics.items():
                stats_df = self.get_descriptive_stats(metric_data)
                component_stats[metric_name] = stats_df
            
            results[component] = component_stats
        
        return results
    
    def test_stationarity(self, data, column=None):
        """
        Test for stationarity using ADF and KPSS tests.
        
        Args:
            data (DataFrame or Series): Input data
            column (str): Column name if data is DataFrame
            
        Returns:
            dict: Dictionary with test results
        """
        if data is None or (isinstance(data, pd.DataFrame) and data.empty) or (isinstance(data, pd.Series) and len(data) == 0):
            return {
                'adf_statistic': None,
                'adf_pvalue': None,
                'adf_is_stationary': None,
                'kpss_statistic': None,
                'kpss_pvalue': None,
                'kpss_is_stationary': None
            }
        
        # Extract the series to test
        if isinstance(data, pd.DataFrame) and column:
            series = data[column]
        else:
            series = data if isinstance(data, pd.Series) else data.iloc[:, 0]
        
        # Convert to numeric and drop NaNs
        series = pd.to_numeric(series, errors='coerce').dropna()
        
        # If not enough data points, return None
        if len(series) < 10:
            return {
                'adf_statistic': None,
                'adf_pvalue': None,
                'adf_is_stationary': None,
                'kpss_statistic': None,
                'kpss_pvalue': None,
                'kpss_is_stationary': None
            }
        
        # ADF test (null hypothesis: series has a unit root, i.e., non-stationary)
        try:
            adf_result = adfuller(series)
            adf_statistic, adf_pvalue = adf_result[0], adf_result[1]
            adf_is_stationary = adf_pvalue < 0.05  # Reject null hypothesis if p-value < 0.05
        except Exception as e:
            logging.warning(f"Error in ADF test: {e}")
            adf_statistic, adf_pvalue, adf_is_stationary = None, None, None
        
        # KPSS test (null hypothesis: series is stationary)
        try:
            kpss_result = kpss(series)
            kpss_statistic, kpss_pvalue = kpss_result[0], kpss_result[1]
            kpss_is_stationary = kpss_pvalue >= 0.05  # Fail to reject null hypothesis if p-value >= 0.05
        except Exception as e:
            logging.warning(f"Error in KPSS test: {e}")
            kpss_statistic, kpss_pvalue, kpss_is_stationary = None, None, None
        
        return {
            'adf_statistic': adf_statistic,
            'adf_pvalue': adf_pvalue,
            'adf_is_stationary': adf_is_stationary,
            'kpss_statistic': kpss_statistic,
            'kpss_pvalue': kpss_pvalue,
            'kpss_is_stationary': kpss_is_stationary
        }
    
    def analyze_stationarity(self, phase_data):
        """
        Analyze stationarity of all metrics for a specific phase.
        
        Args:
            phase_data (dict): Dictionary with component names as keys and data dictionaries as values
            
        Returns:
            dict: Dictionary with component names as keys and stationarity test results as values
        """
        results = {}
        
        for component, metrics in phase_data.items():
            component_results = {}
            
            for metric_name, metric_data in metrics.items():
                # For DataFrames, test each value column
                if isinstance(metric_data, pd.DataFrame) and len(metric_data.columns) > 0:
                    for col in metric_data.columns:
                        if metric_data[col].dtype in [np.float64, np.int64, np.int32, np.float32]:
                            test_result = self.test_stationarity(metric_data, col)
                            component_results[f"{metric_name}_{col}"] = test_result
                # For Series, test the series
                else:
                    test_result = self.test_stationarity(metric_data)
                    component_results[metric_name] = test_result
            
            results[component] = component_results
        
        return results
    
    def detect_anomalies(self, data, threshold=3.0):
        """
        Detect anomalies in the data using Z-score method.
        
        Args:
            data (DataFrame): Input data
            threshold (float): Z-score threshold for anomalies
            
        Returns:
            DataFrame: DataFrame with boolean mask of anomalies
        """
        if data is None or data.empty:
            return pd.DataFrame()
        
        z_scores = (data - data.mean()) / data.std()
        return z_scores.abs() > threshold
    
    def compare_phases(self, metrics_across_phases, metric_name):
        """
        Compare the same metric across different phases.
        
        Args:
            metrics_across_phases (dict): Dictionary with phase names as keys and DataFrames as values
            metric_name (str): Name of the metric
            
        Returns:
            dict: Dictionary with comparison results
        """
        if not metrics_across_phases or len(metrics_across_phases) < 2:
            return {}
        
        phases = list(metrics_across_phases.keys())
        results = {
            'metric_name': metric_name,
            'phases': phases,
            'phase_stats': {},
            'comparisons': {}
        }
        
        # Calculate statistics for each phase
        for phase, data in metrics_across_phases.items():
            results['phase_stats'][phase] = self.get_descriptive_stats(data)
        
        # Compare each pair of phases
        for i, phase1 in enumerate(phases):
            for j, phase2 in enumerate(phases):
                if i >= j:
                    continue  # Skip self-comparison and redundant comparisons
                
                data1 = metrics_across_phases[phase1]
                data2 = metrics_across_phases[phase2]
                
                # Handle DataFrames vs Series
                if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
                    # For simplicity, assume we're comparing the first value column in each DataFrame
                    value_col1 = data1.columns[0] if len(data1.columns) > 0 else None
                    value_col2 = data2.columns[0] if len(data2.columns) > 0 else None
                    
                    if value_col1 and value_col2:
                        series1 = pd.to_numeric(data1[value_col1], errors='coerce').dropna()
                        series2 = pd.to_numeric(data2[value_col2], errors='coerce').dropna()
                    else:
                        continue
                else:
                    # If one or both are Series, use directly
                    series1 = pd.to_numeric(data1, errors='coerce').dropna() if isinstance(data1, pd.Series) else None
                    series2 = pd.to_numeric(data2, errors='coerce').dropna() if isinstance(data2, pd.Series) else None
                
                if series1 is None or series2 is None or len(series1) == 0 or len(series2) == 0:
                    continue
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(series1, series2, equal_var=False)
                
                # Calculate effect size (Cohen's d)
                mean1, std1 = series1.mean(), series1.std()
                mean2, std2 = series2.mean(), series2.std()
                pooled_std = np.sqrt(((len(series1) - 1) * std1**2 + (len(series2) - 1) * std2**2) / (len(series1) + len(series2) - 2))
                cohens_d = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                
                # Calculate percent change
                baseline = mean1
                change = mean2 - mean1
                percent_change = (change / baseline) * 100 if baseline != 0 else np.nan
                
                # Store results
                results['comparisons'][f"{phase1}_vs_{phase2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'is_significant': p_value < 0.05,
                    'effect_size': cohens_d,
                    'effect_magnitude': self._interpret_effect_size(cohens_d),
                    'mean_difference': mean2 - mean1,
                    'percent_change': percent_change
                }
        
        return results
    
    def _interpret_effect_size(self, d):
        """
        Interpret Cohen's d effect size.
        
        Args:
            d (float): Cohen's d value
            
        Returns:
            str: Interpretation of effect size
        """
        if d < 0.2:
            return 'negligible'
        elif d < 0.5:
            return 'small'
        elif d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def save_results_to_csv(self, results, filename):
        """
        Save results to CSV file.
        
        Args:
            results (dict or DataFrame): Results to save
            filename (str): Filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.output_dir is None:
            logging.warning("Output directory not specified")
            return False
        
        file_path = self.output_dir / filename
        
        try:
            # If results is a DataFrame, save directly
            if isinstance(results, pd.DataFrame):
                results.to_csv(file_path)
                return True
            
            # If results is a dict, try to convert to DataFrame
            if isinstance(results, dict):
                # Flatten nested dictionary for CSV output
                flattened = self._flatten_dict(results)
                pd.DataFrame(flattened).to_csv(file_path)
                return True
            
            logging.error(f"Unsupported results type: {type(results)}")
            return False
        except Exception as e:
            logging.error(f"Error saving results to {file_path}: {e}")
            return False
    
    def _flatten_dict(self, d, parent_key='', sep='_'):
        """
        Flatten a nested dictionary.
        
        Args:
            d (dict): Dictionary to flatten
            parent_key (str): Parent key
            sep (str): Separator for keys
            
        Returns:
            dict: Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict) and not any(isinstance(x, pd.DataFrame) for x in v.values()):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def save_results_to_latex(self, results, filename):
        """
        Save results to LaTeX file.
        
        Args:
            results (DataFrame): Results to save
            filename (str): Filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.output_dir is None:
            logging.warning("Output directory not specified")
            return False
        
        file_path = self.output_dir / filename
        
        try:
            if isinstance(results, pd.DataFrame):
                with open(file_path, 'w') as f:
                    f.write(results.to_latex())
                return True
                
            logging.error(f"Unsupported results type for LaTeX export: {type(results)}")
            return False
        except Exception as e:
            logging.error(f"Error saving results to {file_path}: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Set up paths
    base_path = "/home/phil/Projects/k8s-noisy-lab-data-pipe"
    experiment_name = "2025-05-11/16-58-00/default-experiment-1"
    round_number = "round-1"
    output_dir = f"{base_path}/results/stats_results"
    
    # Load data
    data_loader = DataLoader(base_path, experiment_name, round_number)
    data = data_loader.load_all_phases()
    
    # Initialize stats analyzer
    stats_analyzer = StatsAnalyzer(output_dir)
    
    # Example: Analyze baseline phase
    baseline_phase = "1 - Baseline"
    if baseline_phase in data:
        baseline_stats = stats_analyzer.analyze_phase_metrics(data[baseline_phase], baseline_phase)
        
        # Print sample stats for one component
        component = list(baseline_stats.keys())[0]
        metric = list(baseline_stats[component].keys())[0]
        print(f"\nStats for {baseline_phase}/{component}/{metric}:")
        print(baseline_stats[component][metric])
        
        # Save to CSV
        stats_analyzer.save_results_to_csv(baseline_stats[component][metric], f"{component}_{metric}_stats.csv")
    else:
        print(f"Phase {baseline_phase} not found in data")
