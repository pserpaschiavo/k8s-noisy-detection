#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Analysis for Kubernetes Noisy Neighbours Lab
This module provides advanced time series analysis of metrics data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf, grangercausalitytests
import nolds
import pyinform
from pathlib import Path
import logging

class TimeSeriesAnalyzer:
    def __init__(self, output_dir=None):
        """
        Initialize the time series analyzer.
        
        Args:
            output_dir (str): Directory to save results and plots
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Create subdirectory for time series analysis plots
        self.plots_dir = self.output_dir / "time_series_analysis" if self.output_dir else None
        if self.plots_dir and not self.plots_dir.exists():
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            
        logging.info(f"Initialized TimeSeriesAnalyzer, output directory: {self.output_dir}")
    
    def cross_correlation(self, series1, series2, max_lag=20, title=None, series1_name=None, series2_name=None):
        """
        Calculate and plot cross-correlation between two time series.
        
        Args:
            series1 (Series): First time series
            series2 (Series): Second time series
            max_lag (int): Maximum lag to consider
            title (str): Plot title
            series1_name (str): Name of the first series
            series2_name (str): Name of the second series
            
        Returns:
            tuple: (cross-correlation array, lags array)
        """
        # Convert to numeric
        series1 = pd.to_numeric(series1, errors='coerce').dropna()
        series2 = pd.to_numeric(series2, errors='coerce').dropna()
        
        # Ensure series are the same length
        min_len = min(len(series1), len(series2))
        if min_len < 2:
            logging.warning("Series too short for cross-correlation analysis")
            return None, None
            
        series1 = series1[:min_len]
        series2 = series2[:min_len]
        
        # Get names for labels
        s1_name = series1_name if series1_name else getattr(series1, 'name', 'Series 1')
        s2_name = series2_name if series2_name else getattr(series2, 'name', 'Series 2')
        
        # Calculate cross-correlation
        cross_corr = ccf(series1, series2, adjusted=False)
        lags = np.arange(-max_lag, max_lag + 1)
        
        # Get valid range of lags
        valid_lags_range = min(max_lag, len(cross_corr) // 2)
        cross_corr = cross_corr[len(cross_corr) // 2 - valid_lags_range:len(cross_corr) // 2 + valid_lags_range + 1]
        lags = lags[-len(cross_corr):]
        
        # Plot if output directory is set
        if self.plots_dir:
            plt.figure(figsize=(10, 6))
            plt.stem(lags, cross_corr)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.axhline(y=1.96/np.sqrt(len(series1)), color='r', linestyle='--', alpha=0.7)
            plt.axhline(y=-1.96/np.sqrt(len(series1)), color='r', linestyle='--', alpha=0.7)
            plt.xlabel('Lag')
            plt.ylabel('Cross-Correlation')
            plt.title(title or f'Cross-Correlation: {s1_name} vs {s2_name}')
            plt.grid(True)
            
            # Save plot
            filename = f"cross_corr_{s1_name}_{s2_name}.png"
            # Clean filename
            filename = filename.replace('/', '_').replace(' ', '_').lower()
            plt.savefig(self.plots_dir / filename, bbox_inches='tight', dpi=300)
            plt.close()
        
        return cross_corr, lags
    
    def lag_analysis(self, series1, series2, max_lag=20, title=None, series1_name=None, series2_name=None):
        """
        Analyze the optimal lag between two time series.
        
        Args:
            series1 (Series): First time series
            series2 (Series): Second time series
            max_lag (int): Maximum lag to consider
            title (str): Plot title
            series1_name (str): Name of the first series
            series2_name (str): Name of the second series
            
        Returns:
            dict: Dictionary with optimal lag and correlation
        """
        cross_corr, lags = self.cross_correlation(series1, series2, max_lag, title, series1_name, series2_name)
        
        if cross_corr is None or len(cross_corr) == 0:
            return {'optimal_lag': None, 'correlation': None}
        
        # Find optimal lag (maximum absolute correlation)
        abs_cross_corr = np.abs(cross_corr)
        optimal_idx = np.argmax(abs_cross_corr)
        optimal_lag = lags[optimal_idx]
        correlation = cross_corr[optimal_idx]
        
        # Plot the correlation vs. lag if output directory is set
        if self.plots_dir:
            # Get names for labels
            s1_name = series1_name if series1_name else getattr(series1, 'name', 'Series 1')
            s2_name = series2_name if series2_name else getattr(series2, 'name', 'Series 2')
            
            plt.figure(figsize=(10, 6))
            plt.plot(lags, cross_corr, marker='o')
            plt.axhline(y=0, color='r', linestyle='-')
            plt.axvline(x=optimal_lag, color='g', linestyle='--', 
                        label=f'Optimal Lag: {optimal_lag} ({correlation:.3f})')
            plt.xlabel('Lag')
            plt.ylabel('Correlation')
            plt.title(title or f'Lag Analysis: {s1_name} vs {s2_name}')
            plt.grid(True)
            plt.legend()
            
            # Save plot
            filename = f"lag_analysis_{s1_name}_{s2_name}.png"
            # Clean filename
            filename = filename.replace('/', '_').replace(' ', '_').lower()
            plt.savefig(self.plots_dir / filename, bbox_inches='tight', dpi=300)
            plt.close()
        
        return {
            'optimal_lag': optimal_lag,
            'correlation': correlation
        }
    
    def granger_causality(self, series1, series2, maxlag=5, series1_name=None, series2_name=None):
        """
        Test for Granger causality between two time series.
        
        Args:
            series1 (Series): First time series
            series2 (Series): Second time series
            maxlag (int): Maximum lag to consider
            series1_name (str): Name of the first series
            series2_name (str): Name of the second series
            
        Returns:
            dict: Dictionary with test results
        """
        # Convert to numeric
        series1 = pd.to_numeric(series1, errors='coerce').dropna()
        series2 = pd.to_numeric(series2, errors='coerce').dropna()
        
        # Ensure series are the same length
        min_len = min(len(series1), len(series2))
        if min_len <= maxlag + 1:
            logging.warning("Series too short for Granger causality test")
            return None
        
        series1 = series1[:min_len]
        series2 = series2[:min_len]
        
        # Get names for reporting
        s1_name = series1_name if series1_name else getattr(series1, 'name', 'Series 1')
        s2_name = series2_name if series2_name else getattr(series2, 'name', 'Series 2')
        
        # Combine series into a dataframe
        data = pd.DataFrame({s1_name: series1, s2_name: series2})
        
        try:
            # Test 1: series1 causes series2
            gc_1_to_2 = grangercausalitytests(data[[s1_name, s2_name]], maxlag=maxlag, verbose=False)
            
            # Test 2: series2 causes series1
            gc_2_to_1 = grangercausalitytests(data[[s2_name, s1_name]], maxlag=maxlag, verbose=False)
            
            # Extract p-values for each lag
            results_1_to_2 = {lag: gc_1_to_2[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag + 1)}
            results_2_to_1 = {lag: gc_2_to_1[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag + 1)}
            
            # Find minimum p-values
            min_p_1_to_2 = min(results_1_to_2.values())
            min_p_2_to_1 = min(results_2_to_1.values())
            min_lag_1_to_2 = min(results_1_to_2.items(), key=lambda x: x[1])[0]
            min_lag_2_to_1 = min(results_2_to_1.items(), key=lambda x: x[1])[0]
            
            return {
                'granger_1_to_2': {
                    'direction': f"{s1_name} → {s2_name}",
                    'p_values': results_1_to_2,
                    'min_p_value': min_p_1_to_2,
                    'min_lag': min_lag_1_to_2,
                    'significant': min_p_1_to_2 < 0.05
                },
                'granger_2_to_1': {
                    'direction': f"{s2_name} → {s1_name}",
                    'p_values': results_2_to_1,
                    'min_p_value': min_p_2_to_1,
                    'min_lag': min_lag_2_to_1,
                    'significant': min_p_2_to_1 < 0.05
                }
            }
        except Exception as e:
            logging.error(f"Error in Granger causality test: {e}")
            return None
    
    def calculate_entropy(self, series, method='sample', embed_dim=2, tolerance=None, series_name=None):
        """
        Calculate entropy of a time series.
        
        Args:
            series (Series): Time series
            method (str): Entropy method ('sample', 'approximate', 'shannon')
            embed_dim (int): Embedding dimension
            tolerance (float): Tolerance for sample entropy
            series_name (str): Name of the series
            
        Returns:
            dict: Dictionary with entropy results
        """
        # Ensure we have a valid pandas Series or numpy array
        if series is None:
            logging.warning("Cannot calculate entropy: series is None")
            return {
                'entropy_value': None,
                'method': method
            }
            
        try:
            # Convert to numeric and handle NaN values
            if hasattr(series, 'values'):  # If it's a pandas Series
                series_values = pd.to_numeric(series, errors='coerce').dropna()
            elif isinstance(series, (list, tuple, np.ndarray)):  # If it's already a list, tuple or array
                series_values = pd.to_numeric(pd.Series(series), errors='coerce').dropna()
            else:
                logging.error(f"Unsupported type for entropy calculation: {type(series)}")
                return {
                    'entropy_value': None,
                    'method': method,
                    'error': f"Unsupported type: {type(series)}"
                }
        except Exception as e:
            logging.error(f"Error converting series to numeric: {e}")
            return {
                'entropy_value': None,
                'method': method,
                'error': str(e)
            }
        
        # Need sufficient data points
        if len(series_values) < 10:
            logging.warning(f"Series too short for entropy calculation: {len(series_values)} points")
            return {
                'entropy_value': None,
                'method': method
            }
        
        # Get series name for plotting
        s_name = series_name if series_name else getattr(series, 'name', 'Series')
        
        # Calculate entropy
        try:
            if method == 'sample':
                # Determine tolerance if not provided (0.2 * std is a common choice)
                if tolerance is None:
                    tolerance = 0.2 * series_values.std()
                entropy_value = nolds.sampen(series_values.values, emb_dim=embed_dim, tolerance=tolerance)
            elif method == 'approximate':
                # Determine tolerance if not provided
                if tolerance is None:
                    tolerance = 0.2 * series_values.std()
                # Using sampen (Sample Entropy) instead of ap_entropy (Approximate Entropy)
                entropy_value = nolds.sampen(series_values.values, emb_dim=embed_dim, tolerance=tolerance)
            elif method == 'shannon':
                # Discretize the series (into 8 bins by default)
                hist, bin_edges = np.histogram(series_values, bins=8)
                # Normalize to get probabilities
                p = hist / float(sum(hist))
                # Remove zero probabilities
                p = p[p > 0]
                # Calculate Shannon entropy
                entropy_value = -np.sum(p * np.log2(p))
            else:
                logging.error(f"Unknown entropy method: {method}")
                return {
                    'entropy_value': None,
                    'method': method
                }
        except Exception as e:
            logging.error(f"Error calculating {method} entropy: {e}")
            return {
                'entropy_value': None,
                'method': method,
                'error': str(e)
            }
        
        # Plot histogram and normalized signal if output directory is set
        if self.plots_dir:
            plt.figure(figsize=(12, 8))
            
            # Top plot: Time series
            plt.subplot(2, 1, 1)
            plt.plot(series)
            plt.title(f'{s_name}: {method.capitalize()} Entropy = {entropy_value:.4f}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.grid(True)
            
            # Bottom plot: Histogram
            plt.subplot(2, 1, 2)
            plt.hist(series, bins=20, alpha=0.7)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            filename = f"entropy_{method}_{s_name}.png"
            # Clean filename
            filename = filename.replace('/', '_').replace(' ', '_').lower()
            plt.savefig(self.plots_dir / filename, bbox_inches='tight', dpi=300)
            plt.close()
        
        return {
            'entropy_value': entropy_value,
            'method': method
        }
    
    def compare_entropy_across_phases(self, series_by_phase, method='sample', embed_dim=2, 
                                     tolerance=None, series_name=None):
        """
        Compare entropy across different phases.
        
        Args:
            series_by_phase (dict): Dictionary with phase names as keys and series as values
            method (str): Entropy method ('sample', 'approximate', 'shannon')
            embed_dim (int): Embedding dimension
            tolerance (float): Tolerance for sample entropy
            series_name (str): Name of the series
            
        Returns:
            dict: Dictionary with entropy comparison results
        """
        if not series_by_phase or len(series_by_phase) < 2:
            return {}
        
        # Calculate entropy for each phase
        entropy_results = {}
        for phase, series in series_by_phase.items():
            entropy_results[phase] = self.calculate_entropy(
                series, method, embed_dim, tolerance, f"{series_name} ({phase})"
            )
        
        # Plot comparison if output directory is set
        if self.plots_dir:
            # Extract entropy values and filter out None values
            phases = []
            entropy_values = []
            for phase, result in entropy_results.items():
                if result['entropy_value'] is not None:
                    phases.append(phase)
                    entropy_values.append(result['entropy_value'])
            
            # Only create the plot if we have valid values
            if phases and entropy_values:
                # Create a bar chart for comparison
                plt.figure(figsize=(10, 6))
                bars = plt.bar(phases, entropy_values, alpha=0.7)
                
                # Add value labels on top of bars
                for bar, value in zip(bars, entropy_values):
                    plt.text(bar.get_x() + bar.get_width()/2, value + 0.01,
                            f'{value:.4f}', ha='center', va='bottom')
                
                plt.title(f'Comparison of {method.capitalize()} Entropy: {series_name}')
                plt.xlabel('Phase')
                plt.ylabel(f'{method.capitalize()} Entropy')
                plt.grid(True, axis='y')
                
                # Save plot
                s_name = series_name if series_name else 'unknown'
                filename = f"entropy_comparison_{method}_{s_name}.png"
                # Clean filename
                filename = filename.replace('/', '_').replace(' ', '_').lower()
                plt.savefig(self.plots_dir / filename, bbox_inches='tight', dpi=300)
                plt.close()
            else:
                logging.warning(f"No valid entropy values to plot for {series_name}")
        
        return entropy_results
    
    def analyze_time_series_group(self, series_group, group_name=""):
        """
        Analyze a group of related time series.
        
        Args:
            series_group (dict): Dictionary with series names as keys and series as values
            group_name (str): Name of the group
            
        Returns:
            dict: Dictionary with analysis results
        """
        results = {
            'group_name': group_name,
            'entropy': {},
            'cross_correlations': {},
            'lag_analysis': {},
            'granger_causality': {}
        }
        
        # Calculate entropy for each series
        for name, series in series_group.items():
            # Sample entropy
            results['entropy'][f"{name}_sample"] = self.calculate_entropy(
                series, method='sample', series_name=name
            )
            
            # Approximate entropy
            results['entropy'][f"{name}_approximate"] = self.calculate_entropy(
                series, method='approximate', series_name=name
            )
        
        # Calculate cross-correlations and lag analysis for each pair
        series_names = list(series_group.keys())
        for i, name1 in enumerate(series_names):
            for j, name2 in enumerate(series_names):
                if i >= j:  # Skip self-comparison and redundant pairs
                    continue
                
                # Cross-correlation
                key = f"{name1}_vs_{name2}"
                cross_corr, lags = self.cross_correlation(
                    series_group[name1], series_group[name2], 
                    series1_name=name1, series2_name=name2
                )
                results['cross_correlations'][key] = {
                    'cross_correlation': cross_corr,
                    'lags': lags
                }
                
                # Lag analysis
                results['lag_analysis'][key] = self.lag_analysis(
                    series_group[name1], series_group[name2],
                    series1_name=name1, series2_name=name2
                )
                
                # Granger causality
                results['granger_causality'][key] = self.granger_causality(
                    series_group[name1], series_group[name2],
                    series1_name=name1, series2_name=name2
                )
        
        return results


# Example usage
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
    output_dir = f"{base_path}/results/plots"
    
    # Load data
    data_loader = DataLoader(base_path, experiment_name, round_number)
    data = data_loader.load_all_phases()
    
    # Initialize time series analyzer
    ts_analyzer = TimeSeriesAnalyzer(output_dir)
    
    # Example: Compare CPU usage across phases for tenant-a
    phases = list(data.keys())
    if len(phases) >= 2:
        cpu_usage_across_phases = {}
        for phase in phases:
            if 'tenant-a' in data[phase] and 'cpu_usage' in data[phase]['tenant-a']:
                # Get the 'value' column if it exists, otherwise use the first numerical column
                df = data[phase]['tenant-a']['cpu_usage']
                if 'value' in df.columns:
                    cpu_usage_across_phases[phase] = df['value']
                else:
                    # Use the first column
                    cpu_usage_across_phases[phase] = df.iloc[:, 0]
        
        # Compare entropy across phases
        if len(cpu_usage_across_phases) >= 2:
            entropy_results = ts_analyzer.compare_entropy_across_phases(
                cpu_usage_across_phases, method='sample', 
                series_name='tenant-a_cpu_usage'
            )
            print("\nEntropy comparison results:")
            for phase, result in entropy_results.items():
                print(f"{phase}: {result['entropy_value']:.4f}")
    else:
        print("Not enough phases for comparison.")
