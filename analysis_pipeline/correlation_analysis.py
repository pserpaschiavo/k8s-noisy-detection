#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation Analysis for Kubernetes Noisy Neighbours Lab
This module provides correlation analysis between metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

class CorrelationAnalyzer:
    def __init__(self, output_dir=None):
        """
        Initialize the correlation analyzer.
        
        Args:
            output_dir (str): Directory to save results and plots
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectory for correlation plots
        self.plots_dir = self.output_dir / "correlations" if self.output_dir else None
        if self.plots_dir and not self.plots_dir.exists():
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            
        logging.info(f"Initialized CorrelationAnalyzer, output directory: {self.output_dir}")
    
    def align_time_series(self, series_dict):
        """
        Align multiple time series to the same time index.
        
        Args:
            series_dict (dict): Dictionary with series names as keys and series/dataframes as values
            
        Returns:
            DataFrame: DataFrame with aligned time series
        """
        # Extract series and convert to DataFrame if needed
        aligned_data = {}
        for name, data in series_dict.items():
            if isinstance(data, pd.DataFrame):
                # If DataFrame, use the first value column
                value_cols = [col for col in data.columns if col != 'timestamp' and pd.api.types.is_numeric_dtype(data[col])]
                if value_cols:
                    # Use first value column for alignment
                    aligned_data[name] = data[value_cols[0]]
                else:
                    logging.warning(f"No numeric columns found in DataFrame for {name}")
                    continue
            else:
                # If Series, use directly
                aligned_data[name] = data
        
        # Create a DataFrame from all series
        aligned_df = pd.DataFrame(aligned_data)
        
        return aligned_df
    
    def calculate_correlation(self, series_dict, method='pearson'):
        """
        Calculate correlation between multiple time series.
        
        Args:
            series_dict (dict): Dictionary with series names as keys and series/dataframes as values
            method (str): Correlation method ('pearson', 'kendall', 'spearman')
            
        Returns:
            DataFrame: Correlation matrix
        """
        # Align time series
        aligned_df = self.align_time_series(series_dict)
        
        # Calculate correlation matrix
        corr_matrix = aligned_df.corr(method=method)
        
        return corr_matrix
    
    def plot_correlation_matrix(self, corr_matrix, title=None, filename=None):
        """
        Plot correlation matrix as a heatmap.
        
        Args:
            corr_matrix (DataFrame): Correlation matrix
            title (str): Plot title
            filename (str): Filename to save plot
            
        Returns:
            bool: True if successful, False otherwise
        """
        if corr_matrix is None or corr_matrix.empty:
            logging.warning("Empty correlation matrix, cannot plot")
            return False
        
        # Set default filename if not provided
        if not filename and title:
            filename = f"corr_matrix_{title.replace(' ', '_').lower()}.png"
        elif not filename:
            filename = "correlation_matrix.png"
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Upper triangle mask
        
        # Generate heatmap with better color scheme and annotations
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, 
                    mask=mask, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title(title or "Correlation Matrix")
        plt.tight_layout()
        
        # Save plot if output directory is set
        if self.plots_dir:
            plt.savefig(self.plots_dir / filename, bbox_inches='tight', dpi=300)
            plt.close()
            return True
        else:
            plt.show()
            plt.close()
            return True
    
    def analyze_component_correlations(self, component_data, component_name=None, method='pearson'):
        """
        Analyze correlations between metrics within a component.
        
        Args:
            component_data (dict): Dictionary with metric names as keys and dataframes as values
            component_name (str): Name of the component
            method (str): Correlation method ('pearson', 'kendall', 'spearman')
            
        Returns:
            DataFrame: Correlation matrix
        """
        # Extract series from component data
        series_dict = {}
        for metric_name, metric_data in component_data.items():
            if isinstance(metric_data, pd.DataFrame):
                # Use first value column
                value_cols = [col for col in metric_data.columns if col != 'timestamp' and pd.api.types.is_numeric_dtype(metric_data[col])]
                if value_cols:
                    series_dict[metric_name] = metric_data[value_cols[0]]
            elif isinstance(metric_data, pd.Series):
                series_dict[metric_name] = metric_data
        
        # Calculate correlation
        corr_matrix = self.calculate_correlation(series_dict, method)
        
        # Plot correlation matrix
        title = f"{component_name or 'Component'} Metrics Correlation"
        filename = f"corr_matrix_{component_name or 'component'}.png"
        self.plot_correlation_matrix(corr_matrix, title, filename)
        
        return corr_matrix
    
    def analyze_cross_component_correlations(self, phase_data, components, metrics, method='pearson'):
        """
        Analyze correlations between specific metrics across different components.
        
        Args:
            phase_data (dict): Dictionary with component names as keys and data dictionaries as values
            components (list): List of component names to include
            metrics (list): List of metric names to include
            method (str): Correlation method ('pearson', 'kendall', 'spearman')
            
        Returns:
            DataFrame: Correlation matrix
        """
        # Extract relevant series
        series_dict = {}
        
        for component in components:
            if component not in phase_data:
                logging.warning(f"Component {component} not found in phase data")
                continue
                
            for metric in metrics:
                if metric not in phase_data[component]:
                    logging.warning(f"Metric {metric} not found in component {component}")
                    continue
                
                # Extract data
                metric_data = phase_data[component][metric]
                
                if isinstance(metric_data, pd.DataFrame):
                    # Use first value column
                    value_cols = [col for col in metric_data.columns if col != 'timestamp' and pd.api.types.is_numeric_dtype(metric_data[col])]
                    if value_cols:
                        series_dict[f"{component}_{metric}"] = metric_data[value_cols[0]]
                elif isinstance(metric_data, pd.Series):
                    series_dict[f"{component}_{metric}"] = metric_data
        
        # Calculate correlation
        corr_matrix = self.calculate_correlation(series_dict, method)
        
        # Plot correlation matrix
        title = "Cross-Component Metric Correlations"
        filename = "cross_component_correlations.png"
        self.plot_correlation_matrix(corr_matrix, title, filename)
        
        return corr_matrix
    
    def find_strongest_correlations(self, corr_matrix, threshold=0.7):
        """
        Find the strongest correlations in a correlation matrix.
        
        Args:
            corr_matrix (DataFrame): Correlation matrix
            threshold (float): Correlation threshold
            
        Returns:
            DataFrame: DataFrame with strongest correlations
        """
        if corr_matrix is None or corr_matrix.empty:
            return pd.DataFrame()
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find pairs with correlation above threshold
        strong_pairs = []
        
        for col in upper.columns:
            for idx in upper.index:
                value = upper.at[idx, col]
                if abs(value) >= threshold:
                    strong_pairs.append({
                        'metric1': idx,
                        'metric2': col,
                        'correlation': value,
                        'abs_correlation': abs(value),
                        'relationship': 'positive' if value > 0 else 'negative'
                    })
        
        # Convert to DataFrame and sort by absolute correlation
        if strong_pairs:
            strong_corr = pd.DataFrame(strong_pairs)
            strong_corr = strong_corr.sort_values('abs_correlation', ascending=False)
            return strong_corr
        else:
            return pd.DataFrame(columns=['metric1', 'metric2', 'correlation', 'abs_correlation', 'relationship'])
    
    def plot_scatter_matrix(self, series_dict, title=None, filename=None, n_metrics=None):
        """
        Plot scatter matrix for multiple metrics.
        
        Args:
            series_dict (dict): Dictionary with series names as keys and series as values
            title (str): Plot title
            filename (str): Filename to save plot
            n_metrics (int): Number of metrics to include (limit for readability)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Align time series
        aligned_df = self.align_time_series(series_dict)
        
        # Limit number of metrics if specified
        if n_metrics and len(aligned_df.columns) > n_metrics:
            logging.warning(f"Limiting scatter matrix to {n_metrics} metrics for readability")
            aligned_df = aligned_df.iloc[:, :n_metrics]
        
        # Set default filename if not provided
        if not filename and title:
            filename = f"scatter_matrix_{title.replace(' ', '_').lower()}.png"
        elif not filename:
            filename = "scatter_matrix.png"
        
        # Create scatter matrix
        plt.figure(figsize=(12, 10))
        axes = pd.plotting.scatter_matrix(aligned_df, figsize=(12, 10), diagonal='kde', alpha=0.7)
        
        # Rotate axis labels
        for ax in axes.flatten():
            ax.xaxis.label.set_rotation(45)
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_ha('right')
        
        plt.suptitle(title or "Scatter Matrix Plot")
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save plot if output directory is set
        if self.plots_dir:
            plt.savefig(self.plots_dir / filename, bbox_inches='tight', dpi=300)
            plt.close()
            return True
        else:
            plt.show()
            plt.close()
            return True
    
    def plot_pairwise_correlation(self, series1, series2, name1, name2, title=None, filename=None):
        """
        Plot pairwise correlation between two series.
        
        Args:
            series1 (Series): First time series
            series2 (Series): Second time series
            name1 (str): Name of the first series
            name2 (str): Name of the second series
            title (str): Plot title
            filename (str): Filename to save plot
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Convert to numeric
        series1 = pd.to_numeric(series1, errors='coerce')
        series2 = pd.to_numeric(series2, errors='coerce')
        
        # Create DataFrame for plot
        data = pd.DataFrame({name1: series1, name2: series2}).dropna()
        
        # Calculate correlation
        corr = data.corr().iloc[0, 1]
        
        # Set default filename if not provided
        if not filename:
            filename = f"correlation_{name1}_{name2}.png".replace('/', '_').replace(' ', '_').lower()
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(data[name1], data[name2], alpha=0.7)
        plt.xlabel(name1)
        plt.ylabel(name2)
        plt.title(f"Correlation: {corr:.3f}")
        plt.grid(True)
        
        # Joint plot using seaborn
        plt.subplot(1, 2, 2)
        sns.regplot(x=name1, y=name2, data=data)
        plt.grid(True)
        
        plt.suptitle(title or f"Correlation: {name1} vs {name2}")
        plt.tight_layout()
        
        # Save plot if output directory is set
        if self.plots_dir:
            plt.savefig(self.plots_dir / filename, bbox_inches='tight', dpi=300)
            plt.close()
            return True
        else:
            plt.show()
            plt.close()
            return True
    
    def visualize_strong_correlations(self, corr_matrix, series_dict, threshold=0.7):
        """
        Visualize the strongest correlations with pairwise plots.
        
        Args:
            corr_matrix (DataFrame): Correlation matrix
            series_dict (dict): Dictionary with series names as keys and series as values
            threshold (float): Correlation threshold
            
        Returns:
            list: List of filenames for the generated plots
        """
        # Find strongest correlations
        strong_corr = self.find_strongest_correlations(corr_matrix, threshold)
        
        if strong_corr.empty:
            logging.info(f"No correlations found above threshold {threshold}")
            return []
        
        filenames = []
        
        # Create pairwise plots for strongest correlations
        for _, row in strong_corr.iterrows():
            metric1, metric2 = row['metric1'], row['metric2']
            
            if metric1 in series_dict and metric2 in series_dict:
                title = f"{metric1} vs {metric2} (Correlation: {row['correlation']:.3f})"
                filename = f"strong_corr_{metric1}_{metric2}.png".replace('/', '_').replace(' ', '_').lower()
                
                self.plot_pairwise_correlation(
                    series_dict[metric1], series_dict[metric2],
                    metric1, metric2, title, filename
                )
                
                filenames.append(filename)
        
        return filenames


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
    
    # Initialize correlation analyzer
    corr_analyzer = CorrelationAnalyzer(output_dir)
    
    # Example: Analyze correlations in tenant-a metrics in baseline phase
    baseline_phase = "1 - Baseline"
    if baseline_phase in data and 'tenant-a' in data[baseline_phase]:
        corr_matrix = corr_analyzer.analyze_component_correlations(
            data[baseline_phase]['tenant-a'], component_name="tenant-a"
        )
        
        # Find strongest correlations
        strong_corr = corr_analyzer.find_strongest_correlations(corr_matrix, threshold=0.7)
        print("\nStrongest correlations in tenant-a metrics:")
        print(strong_corr)
