#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Statistical Analysis for Kubernetes Noisy Neighbours Lab
This module provides advanced statistical methods for deeper analysis of experimental data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
import ruptures as rpt  # For change point detection
from scipy.spatial.distance import pdist, squareform
import nolds
import warnings
import logging
from pathlib import Path

# Suppress specific warnings
warnings.filterwarnings("ignore", message="No frequency information was provided")
warnings.filterwarnings("ignore", category=FutureWarning)

class AdvancedAnalyzer:
    def __init__(self, output_dir=None):
        """
        Initialize the advanced statistical analyzer.
        
        Args:
            output_dir (str): Directory to save results
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Create subdirectories
        if self.output_dir:
            self.plots_dir = self.output_dir / "advanced_plots"
            self.plots_dir.mkdir(exist_ok=True)
            
            self.results_dir = self.output_dir / "advanced_results"
            self.results_dir.mkdir(exist_ok=True)
            
            # Create specific subdirectories for different types of analyses
            self.time_series_dir = self.plots_dir / "time_series"
            self.time_series_dir.mkdir(exist_ok=True)
            
            self.multivariate_dir = self.plots_dir / "multivariate"
            self.multivariate_dir.mkdir(exist_ok=True)
            
            self.distribution_dir = self.plots_dir / "distributions"
            self.distribution_dir.mkdir(exist_ok=True)
            
            self.changepoint_dir = self.plots_dir / "changepoints"
            self.changepoint_dir.mkdir(exist_ok=True)
        
        # Set plot style for academic publication quality
        # Use a seaborn style compatible with newer versions
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
        
        logging.info(f"Initialized AdvancedAnalyzer with output directory: {self.output_dir}")

    def time_series_decomposition(self, data, column=None, model='additive', period=None, 
                                 title=None, component_name=None, metric_name=None):
        """
        Decompose a time series into trend, seasonal, and residual components.
        
        Args:
            data (DataFrame or Series): Time series data
            column (str): Column name if data is DataFrame
            model (str): Type of decomposition ('additive' or 'multiplicative')
            period (int): Period for seasonal decomposition, if None will be estimated
            title (str): Plot title
            component_name (str): Name of the component
            metric_name (str): Name of the metric
            
        Returns:
            dict: Dictionary with decomposition results
        """
        # Extract the series
        if isinstance(data, pd.DataFrame) and column:
            series = data[column]
        else:
            series = data if isinstance(data, pd.Series) else data.iloc[:, 0]
            column = series.name if hasattr(series, 'name') else "value"
        
        # Convert to numeric and drop NaNs
        series = pd.to_numeric(series, errors='coerce').dropna()
        
        # If not enough data points, return None
        if len(series) < 10:
            logging.warning("Series too short for decomposition")
            return None
        
        # If series is not indexed by timestamp, create a synthetic index
        if not isinstance(series.index, pd.DatetimeIndex):
            series = pd.Series(series.values, index=pd.date_range('2025-01-01', periods=len(series)))
        
        # Estimate period if not provided
        if period is None:
            # Use a reasonable default based on data length
            period = min(len(series) // 10, 24)  # default to 24 points (e.g., hourly data for a day)
            period = max(period, 2)  # Ensure at least 2 for decomposition
        
        try:
            # Perform decomposition
            result = seasonal_decompose(series, model=model, period=period)
            
            # Generate plot
            if self.plots_dir:
                fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                
                # Original series
                series.plot(ax=axes[0], title='Original')
                axes[0].set_ylabel(column)
                
                # Trend
                result.trend.plot(ax=axes[1], title='Trend')
                axes[1].set_ylabel(column)
                
                # Seasonal
                result.seasonal.plot(ax=axes[2], title=f'Seasonal (period={period})')
                axes[2].set_ylabel(column)
                
                # Residual
                result.resid.plot(ax=axes[3], title='Residual')
                axes[3].set_ylabel(column)
                
                plt.tight_layout()
                
                # Generate filename
                fname = f"decomp_{component_name}_{metric_name}".replace('/', '_').replace(' ', '_').lower()
                filepath = self.time_series_dir / f"{fname}.png"
                plt.savefig(filepath, bbox_inches='tight')
                plt.close()
                
                # Also save as PDF for publications
                filepath_pdf = self.time_series_dir / f"{fname}.pdf"
                fig.savefig(filepath_pdf, format='pdf', bbox_inches='tight')
                
                logging.info(f"Saved time series decomposition plot to {filepath}")
            
            # Return results as dictionary
            return {
                'trend': result.trend,
                'seasonal': result.seasonal,
                'residual': result.resid,
                'period': period,
                'model': model
            }
            
        except Exception as e:
            logging.error(f"Error in time series decomposition: {e}")
            return None

    def detect_change_points(self, data, column=None, method='pelt', model='l2', min_size=10,
                            penalty=None, title=None, component_name=None, metric_name=None):
        """
        Detect change points in time series.
        
        Args:
            data (DataFrame or Series): Time series data
            column (str): Column name if data is DataFrame
            method (str): Change point detection method ('pelt', 'binseg', 'window')
            model (str): Cost model ('l1', 'l2', 'rbf', etc.)
            min_size (int): Minimum segment length
            penalty (float): Penalty term for new change point
            title (str): Plot title
            component_name (str): Name of the component
            metric_name (str): Name of the metric
            
        Returns:
            dict: Dictionary with change points and related information
        """
        # Extract the series
        if isinstance(data, pd.DataFrame) and column:
            series = data[column]
        else:
            series = data if isinstance(data, pd.Series) else data.iloc[:, 0]
            column = series.name if hasattr(series, 'name') else "value"
        
        # Convert to numeric and drop NaNs
        series = pd.to_numeric(series, errors='coerce').dropna()
        
        # If not enough data points, return None
        if len(series) < min_size*2:
            logging.warning(f"Series too short for change point detection (length={len(series)}, min_size={min_size*2})")
            return None
        
        try:
            # Convert to numpy array
            signal = series.values
            
            # Determine penalty if not specified
            if penalty is None:
                # Default penalty depends on the data length (BIC-like)
                penalty = np.log(len(signal)) * np.std(signal) * 0.5
            
            # Choose algorithm
            if method == 'pelt':
                algo = rpt.Pelt(model=model, min_size=min_size, jump=1).fit(signal)
                change_points = algo.predict(pen=penalty)
            elif method == 'binseg':
                # Estimate number of change points
                n_bkps = min(5, len(signal) // (min_size*2))
                algo = rpt.Binseg(model=model, min_size=min_size).fit(signal)
                change_points = algo.predict(n_bkps=n_bkps)
            elif method == 'window':
                algo = rpt.Window(model=model, min_size=min_size, width=40).fit(signal)
                change_points = algo.predict(n_bkps=5)
            else:
                raise ValueError(f"Unknown change point detection method: {method}")
            
            # Generate plot
            if self.plots_dir and len(change_points) > 1:  # Only plot if we found change points
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot original signal
                ax.plot(series.values, 'b-', linewidth=1.5, label='Signal')
                
                # Plot change points
                for cp in change_points[:-1]:  # Last point is series end
                    ax.axvline(x=cp, color='r', linestyle='--', alpha=0.8)
                
                # Add phase labels if available (assuming standard format)
                if any(p in str(series.name).lower() for p in ['baseline', 'attack', 'recovery']):
                    # Try to determine phase boundaries (if the series might span multiple phases)
                    phase_boundaries = []
                    if "1" in str(series.name) or "baseline" in str(series.name).lower():
                        phase_boundaries.append((0, "Baseline"))
                    if "2" in str(series.name) or "attack" in str(series.name).lower():
                        phase_boundaries.append((len(series)//3, "Attack"))
                    if "3" in str(series.name) or "recovery" in str(series.name).lower():
                        phase_boundaries.append((2*len(series)//3, "Recovery"))
                    
                    # Add phase boundary indicators
                    for pos, label in phase_boundaries:
                        ax.axvline(x=pos, color='g', linestyle='-', alpha=0.5)
                        ax.text(pos+5, max(series.values)*0.95, label, rotation=90, alpha=0.7)
                
                ax.set_title(title or f"Change Point Detection: {component_name} - {metric_name}")
                ax.set_xlabel('Time')
                ax.set_ylabel(column)
                plt.tight_layout()
                
                # Generate filename
                fname = f"changepoint_{component_name}_{metric_name}".replace('/', '_').replace(' ', '_').lower()
                filepath = self.changepoint_dir / f"{fname}.png"
                plt.savefig(filepath, bbox_inches='tight')
                
                # Also save as PDF for publications
                filepath_pdf = self.changepoint_dir / f"{fname}.pdf"
                fig.savefig(filepath_pdf, format='pdf', bbox_inches='tight')
                
                plt.close()
                
                logging.info(f"Saved change point detection plot to {filepath}")
            
            # Calculate segments and their statistics
            segments = []
            start = 0
            for end in change_points:
                if end > start:
                    segment = {
                        'start': start,
                        'end': end,
                        'mean': np.mean(signal[start:end]),
                        'std': np.std(signal[start:end]),
                        'median': np.median(signal[start:end]),
                        'min': np.min(signal[start:end]),
                        'max': np.max(signal[start:end]),
                        'length': end - start
                    }
                    segments.append(segment)
                start = end
            
            # Convert to DataFrame
            segments_df = pd.DataFrame(segments)
            
            # Save results
            if self.results_dir:
                # Save segment statistics
                csv_path = self.results_dir / f"changepoint_{component_name}_{metric_name}_segments.csv"
                segments_df.to_csv(csv_path)
                
                # Save LaTeX format
                tex_path = self.results_dir / f"changepoint_{component_name}_{metric_name}_segments.tex"
                with open(tex_path, 'w') as f:
                    f.write(segments_df.to_latex(index=False))
                
                logging.info(f"Saved change point segments to {csv_path}")
            
            return {
                'change_points': change_points,
                'segments': segments_df,
                'method': method,
                'model': model
            }
            
        except Exception as e:
            logging.error(f"Error in change point detection: {e}")
            return None

    def perform_kmeans_clustering(self, data, components=None, n_clusters=3, pca_dims=2, 
                                 title=None, component_name=None, metrics=None):
        """
        Perform K-means clustering on multiple metrics.
        
        Args:
            data (dict or DataFrame): Dictionary with metric names as keys and series as values,
                                      or DataFrame with metrics as columns
            components (list): Component names if multiple components
            n_clusters (int): Number of clusters
            pca_dims (int): Number of dimensions for PCA visualization
            title (str): Plot title
            component_name (str): Name of the component
            metrics (list): List of metric names
            
        Returns:
            dict: Dictionary with clustering results
        """
        # Prepare data for clustering
        if isinstance(data, dict):
            # Convert dict of series to DataFrame
            df_data = pd.DataFrame()
            for name, series in data.items():
                # Convert to numeric and handle NaNs
                if isinstance(series, pd.DataFrame) and len(series.columns) > 0:
                    col = series.columns[0]
                    df_data[name] = pd.to_numeric(series[col], errors='coerce')
                else:
                    df_data[name] = pd.to_numeric(series, errors='coerce')
        else:
            # Use DataFrame directly
            df_data = data.copy()
        
        # Drop rows with any NaN
        df_data = df_data.dropna()
        
        # If not enough data points, return None
        if len(df_data) < n_clusters*2:
            logging.warning(f"Not enough data points for clustering (rows={len(df_data)}, clusters={n_clusters})")
            return None
        
        try:
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_data)
            
            # Perform PCA for visualization
            pca = PCA(n_components=min(pca_dims, df_data.shape[1]))
            pca_result = pca.fit_transform(scaled_data)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Add results to original data
            df_data['cluster'] = clusters
            
            # Create a DataFrame for the PCA results
            pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(pca_dims)])
            pca_df['cluster'] = clusters
            
            # Generate plot
            if self.plots_dir:
                plt.figure(figsize=(10, 8))
                
                # Plot each cluster with different color
                for i in range(n_clusters):
                    mask = (pca_df['cluster'] == i)
                    plt.scatter(pca_df.loc[mask, 'PC1'], 
                               pca_df.loc[mask, 'PC2'], 
                               label=f'Cluster {i+1}',
                               alpha=0.7,
                               s=50)
                
                # Add cluster centers
                centers = pca.transform(kmeans.cluster_centers_)
                plt.scatter(centers[:, 0], centers[:, 1], 
                           marker='x', s=200, color='black', 
                           label='Centroids')
                
                # Add explained variance info
                explained_var = pca.explained_variance_ratio_
                plt.title(title or f"K-means Clustering: {component_name}")
                plt.xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
                plt.ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Generate filename
                fname = f"kmeans_{component_name}".replace('/', '_').replace(' ', '_').lower()
                if metrics:
                    # Add abbreviated metric names to filename
                    metrics_abbr = "_".join([m[:3] for m in metrics[:3]])
                    fname += f"_{metrics_abbr}"
                
                filepath = self.multivariate_dir / f"{fname}.png"
                plt.savefig(filepath, bbox_inches='tight')
                
                # Also save as PDF for publications
                filepath_pdf = self.multivariate_dir / f"{fname}.pdf"
                plt.savefig(filepath_pdf, format='pdf', bbox_inches='tight')
                
                plt.close()
                
                logging.info(f"Saved K-means clustering plot to {filepath}")
                
                # Create a parallel coordinates plot to visualize cluster characteristics
                plt.figure(figsize=(14, 8))
                
                # Create a copy with normalized values for better visualization
                df_viz = df_data.copy()
                for col in df_viz.columns:
                    if col != 'cluster' and pd.api.types.is_numeric_dtype(df_viz[col]):
                        df_viz[col] = (df_viz[col] - df_viz[col].min()) / \
                                      (df_viz[col].max() - df_viz[col].min())
                
                # Plot parallel coordinates
                pd.plotting.parallel_coordinates(df_viz, 'cluster')
                plt.title(f"Cluster Characteristics: {component_name}")
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save parallel coordinates plot
                filepath = self.multivariate_dir / f"{fname}_parallel.png"
                plt.savefig(filepath, bbox_inches='tight')
                
                # Also save as PDF for publications
                filepath_pdf = self.multivariate_dir / f"{fname}_parallel.pdf"
                plt.savefig(filepath_pdf, format='pdf', bbox_inches='tight')
                
                plt.close()
            
            # Calculate cluster statistics
            cluster_stats = df_data.groupby('cluster').agg(['mean', 'std', 'min', 'max', 'count'])
            
            # Compute feature importances using the centroids
            feature_importances = {}
            for i, feature in enumerate(df_data.columns[:-1]):  # Exclude 'cluster'
                # Calculate variance of each feature's centroid values
                importances = np.var(kmeans.cluster_centers_[:, i])
                feature_importances[feature] = importances
            
            # Sort importances
            sorted_importances = {k: v for k, v in sorted(feature_importances.items(), 
                                                         key=lambda item: item[1], 
                                                         reverse=True)}
            
            # Convert to DataFrame
            importance_df = pd.DataFrame(sorted_importances.items(), 
                                        columns=['Feature', 'Importance'])
            importance_df['Relative_Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()
            
            # Save results
            if self.results_dir:
                # Save cluster statistics
                csv_path = self.results_dir / f"kmeans_{component_name}_cluster_stats.csv"
                cluster_stats.to_csv(csv_path)
                
                # Save feature importances
                imp_csv_path = self.results_dir / f"kmeans_{component_name}_feature_imp.csv"
                importance_df.to_csv(imp_csv_path, index=False)
                
                # Save latex format
                tex_path = self.results_dir / f"kmeans_{component_name}_cluster_stats.tex"
                with open(tex_path, 'w') as f:
                    f.write(cluster_stats.to_latex())
                
                imp_tex_path = self.results_dir / f"kmeans_{component_name}_feature_imp.tex"
                with open(imp_tex_path, 'w') as f:
                    f.write(importance_df.to_latex(index=False))
                
                logging.info(f"Saved clustering results to {csv_path}")
            
            return {
                'clusters': clusters,
                'cluster_stats': cluster_stats,
                'pca_results': pca_df,
                'feature_importances': importance_df,
                'pca_variance': pca.explained_variance_ratio_,
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            logging.error(f"Error in K-means clustering: {e}")
            return None

    def detect_anomalies(self, data, column=None, contamination=0.05, method='iforest',
                        title=None, component_name=None, metric_name=None):
        """
        Detect anomalies in time series data.
        
        Args:
            data (DataFrame or Series): Time series data
            column (str): Column name if data is DataFrame
            contamination (float): Expected proportion of anomalies
            method (str): Detection method ('iforest', 'zscore', 'iqr')
            title (str): Plot title
            component_name (str): Name of the component
            metric_name (str): Name of the metric
            
        Returns:
            dict: Dictionary with anomaly detection results
        """
        # Extract the series
        if isinstance(data, pd.DataFrame) and column:
            series = data[column]
        else:
            series = data if isinstance(data, pd.Series) else data.iloc[:, 0]
            column = series.name if hasattr(series, 'name') else "value"
        
        # Convert to numeric and handle NaNs
        series = pd.to_numeric(series, errors='coerce')
        
        # Create DataFrame for analysis
        df = pd.DataFrame({'value': series})
        df = df.dropna()
        
        # If not enough data points, return None
        if len(df) < 10:
            logging.warning(f"Not enough data points for anomaly detection (rows={len(df)})")
            return None
        
        try:
            anomalies = None
            
            # Detect anomalies based on selected method
            if method == 'iforest':
                # Isolation Forest
                model = IsolationForest(contamination=contamination, random_state=42)
                df['anomaly'] = model.fit_predict(df[['value']])
                df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # Convert to binary (1 = anomaly)
                
            elif method == 'zscore':
                # Z-Score method
                threshold = 3.0  # Standard threshold
                df['zscore'] = np.abs((df['value'] - df['value'].mean()) / df['value'].std())
                df['anomaly'] = (df['zscore'] > threshold).astype(int)
                
            elif method == 'iqr':
                # IQR method
                Q1 = df['value'].quantile(0.25)
                Q3 = df['value'].quantile(0.75)
                IQR = Q3 - Q1
                threshold = 1.5
                df['anomaly'] = ((df['value'] < (Q1 - threshold * IQR)) | 
                                (df['value'] > (Q3 + threshold * IQR))).astype(int)
            else:
                raise ValueError(f"Unknown anomaly detection method: {method}")
            
            # Find anomaly timestamps
            anomalies = df[df['anomaly'] == 1].index
            
            # Generate plot
            if self.plots_dir:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot original data
                ax.plot(df.index, df['value'], 'b-', linewidth=1, alpha=0.7, label='Normal')
                
                # Highlight anomalies
                if len(anomalies) > 0:
                    ax.scatter(anomalies, df.loc[anomalies, 'value'], 
                              color='red', s=50, label='Anomalies')
                
                ax.set_title(title or f"Anomaly Detection ({method}): {component_name} - {metric_name}")
                ax.set_xlabel('Time')
                ax.set_ylabel(column)
                ax.legend()
                plt.tight_layout()
                
                # Generate filename
                fname = f"anomaly_{method}_{component_name}_{metric_name}".replace('/', '_').replace(' ', '_').lower()
                filepath = self.time_series_dir / f"{fname}.png"
                plt.savefig(filepath, bbox_inches='tight')
                
                # Also save as PDF for publications
                filepath_pdf = self.time_series_dir / f"{fname}.pdf"
                fig.savefig(filepath_pdf, format='pdf', bbox_inches='tight')
                
                plt.close()
                
                logging.info(f"Saved anomaly detection plot to {filepath}")
            
            # Calculate anomaly statistics
            anomaly_stats = {
                'total_points': len(df),
                'anomalies_count': len(anomalies),
                'anomalies_percentage': (len(anomalies) / len(df)) * 100,
                'method': method
            }
            
            if len(anomalies) > 0:
                anomaly_values = df.loc[anomalies, 'value']
                anomaly_stats.update({
                    'anomalies_mean': anomaly_values.mean(),
                    'anomalies_std': anomaly_values.std(),
                    'anomalies_min': anomaly_values.min(),
                    'anomalies_max': anomaly_values.max()
                })
                
                # Extract anomalies detail
                anomalies_detail = df[df['anomaly'] == 1].copy()
                anomalies_detail['timestamp'] = anomalies_detail.index
                
                # Save results
                if self.results_dir:
                    # Save anomaly details
                    csv_path = self.results_dir / f"anomaly_{method}_{component_name}_{metric_name}.csv"
                    anomalies_detail.to_csv(csv_path)
                    
                    # Save latex format
                    tex_path = self.results_dir / f"anomaly_{method}_{component_name}_{metric_name}.tex"
                    with open(tex_path, 'w') as f:
                        f.write(anomalies_detail.to_latex())
                    
                    logging.info(f"Saved anomaly detection results to {csv_path}")
            
            return {
                'anomalies': anomalies,
                'anomalies_detail': df[df['anomaly'] == 1] if len(anomalies) > 0 else pd.DataFrame(),
                'stats': anomaly_stats,
                'method': method
            }
            
        except Exception as e:
            logging.error(f"Error in anomaly detection: {e}")
            return None

    def distribution_analysis(self, data, column=None, fit_dist=True, bins=30,
                             title=None, component_name=None, metric_name=None):
        """
        Perform distribution analysis and fit theoretical distributions.
        
        Args:
            data (DataFrame or Series): Data for distribution analysis
            column (str): Column name if data is DataFrame
            fit_dist (bool): Whether to fit theoretical distributions
            bins (int): Number of bins for histogram
            title (str): Plot title
            component_name (str): Name of the component
            metric_name (str): Name of the metric
            
        Returns:
            dict: Dictionary with distribution analysis results
        """
        # Extract the series
        if isinstance(data, pd.DataFrame) and column:
            series = data[column]
        else:
            series = data if isinstance(data, pd.Series) else data.iloc[:, 0]
            column = series.name if hasattr(series, 'name') else "value"
        
        # Convert to numeric and handle NaNs
        series = pd.to_numeric(series, errors='coerce').dropna()
        
        # If not enough data points, return None
        if len(series) < 10:
            logging.warning(f"Not enough data points for distribution analysis (points={len(series)})")
            return None
        
        try:
            # Calculate basic statistics
            basic_stats = {
                'count': len(series),
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                '25%': series.quantile(0.25),
                'median': series.median(),
                '75%': series.quantile(0.75),
                'max': series.max(),
                'skew': stats.skew(series),
                'kurtosis': stats.kurtosis(series)
            }
            
            # Create distribution plot
            if self.plots_dir:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Histogram with KDE
                sns.histplot(series, kde=True, bins=bins, ax=ax1)
                ax1.set_title(f"Distribution: {component_name} - {metric_name}")
                ax1.set_xlabel(column)
                ax1.set_ylabel('Frequency')
                
                # QQ plot to check for normality
                stats.probplot(series, dist="norm", plot=ax2)
                ax2.set_title("Q-Q Plot vs. Normal Distribution")
                
                plt.tight_layout()
                
                # Generate filename
                fname = f"dist_{component_name}_{metric_name}".replace('/', '_').replace(' ', '_').lower()
                filepath = self.distribution_dir / f"{fname}.png"
                plt.savefig(filepath, bbox_inches='tight')
                
                # Also save as PDF for publications
                filepath_pdf = self.distribution_dir / f"{fname}.pdf"
                fig.savefig(filepath_pdf, format='pdf', bbox_inches='tight')
                
                plt.close()
                
                logging.info(f"Saved distribution analysis plot to {filepath}")
                
                # If requested, fit theoretical distributions
                if fit_dist:
                    # Distributions to fit
                    distributions = ['norm', 'lognorm', 'expon', 'weibull_min', 'gamma', 't']
                    
                    # Create plot for distribution fitting
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Plot histogram of actual data
                    sns.histplot(series, kde=True, stat='density', bins=bins, ax=ax, alpha=0.6, label='Data')
                    
                    # Fit distributions
                    results = []
                    
                    for dist_name in distributions:
                        try:
                            # Fit distribution
                            dist = getattr(stats, dist_name)
                            params = dist.fit(series)
                            
                            # Calculate PDF
                            x = np.linspace(series.min(), series.max(), 1000)
                            pdf = dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])
                            
                            # Plot PDF
                            ax.plot(x, pdf, label=f'{dist_name}')
                            
                            # Calculate goodness of fit
                            ks_stat, p_value = stats.kstest(series, dist_name, params)
                            
                            results.append({
                                'distribution': dist_name,
                                'params': params,
                                'ks_statistic': ks_stat,
                                'p_value': p_value
                            })
                        except Exception as e:
                            logging.warning(f"Error fitting {dist_name} distribution: {e}")
                    
                    ax.set_title(f"Distribution Fitting: {component_name} - {metric_name}")
                    ax.set_xlabel(column)
                    ax.set_ylabel('Density')
                    ax.legend()
                    plt.tight_layout()
                    
                    # Generate filename
                    fname = f"dist_fit_{component_name}_{metric_name}".replace('/', '_').replace(' ', '_').lower()
                    filepath = self.distribution_dir / f"{fname}.png"
                    plt.savefig(filepath, bbox_inches='tight')
                    
                    # Also save as PDF for publications
                    filepath_pdf = self.distribution_dir / f"{fname}.pdf"
                    plt.savefig(filepath_pdf, format='pdf', bbox_inches='tight')
                    
                    plt.close()
                    
                    # Save distribution fitting results
                    if results and self.results_dir:
                        results_df = pd.DataFrame(results)
                        results_df = results_df.sort_values('ks_statistic')
                        
                        # Clean up params column for CSV export
                        results_df['params'] = results_df['params'].apply(lambda x: str(x))
                        
                        # Save to CSV
                        csv_path = self.results_dir / f"dist_fit_{component_name}_{metric_name}.csv"
                        results_df.to_csv(csv_path, index=False)
                        
                        # Save best fit parameters separately
                        best_fit = results_df.iloc[0].to_dict()
                        best_fit_df = pd.DataFrame([best_fit])
                        
                        best_csv_path = self.results_dir / f"dist_best_fit_{component_name}_{metric_name}.csv"
                        best_fit_df.to_csv(best_csv_path, index=False)
                        
                        # Save as LaTeX
                        tex_path = self.results_dir / f"dist_fit_{component_name}_{metric_name}.tex"
                        with open(tex_path, 'w') as f:
                            clean_df = results_df.copy()
                            # Format p-values for better readability
                            clean_df['p_value'] = clean_df['p_value'].apply(lambda x: f"{x:.4f}")
                            clean_df['ks_statistic'] = clean_df['ks_statistic'].apply(lambda x: f"{x:.4f}")
                            # Remove params column for LaTeX
                            if 'params' in clean_df.columns:
                                clean_df = clean_df.drop('params', axis=1)
                            f.write(clean_df.to_latex(index=False))
                        
                        logging.info(f"Saved distribution fitting results to {csv_path}")
                        
                        return {
                            'basic_stats': basic_stats,
                            'distribution_fits': results_df,
                            'best_fit': best_fit
                        }
            
            return {
                'basic_stats': basic_stats
            }
            
        except Exception as e:
            logging.error(f"Error in distribution analysis: {e}")
            return None

    def analyze_recovery_metrics(self, before_attack, during_attack, after_attack, 
                               column=None, component_name=None, metric_name=None):
        """
        Analyze system recovery metrics after attack.
        
        Args:
            before_attack (DataFrame or Series): Data before attack
            during_attack (DataFrame or Series): Data during attack
            after_attack (DataFrame or Series): Data after attack
            column (str): Column name if data is DataFrame
            component_name (str): Name of the component
            metric_name (str): Name of the metric
            
        Returns:
            dict: Dictionary with recovery analysis results
        """
        # Extract series
        if isinstance(before_attack, pd.DataFrame) and column:
            before_series = before_attack[column]
            during_series = during_attack[column]
            after_series = after_attack[column]
        else:
            before_series = before_attack if isinstance(before_attack, pd.Series) else before_attack.iloc[:, 0]
            during_series = during_attack if isinstance(during_attack, pd.Series) else during_attack.iloc[:, 0]
            after_series = after_attack if isinstance(after_attack, pd.Series) else after_attack.iloc[:, 0]
            column = before_series.name if hasattr(before_series, 'name') else "value"
        
        # Convert to numeric and handle NaNs
        before_series = pd.to_numeric(before_series, errors='coerce').dropna()
        during_series = pd.to_numeric(during_series, errors='coerce').dropna()
        after_series = pd.to_numeric(after_series, errors='coerce').dropna()
        
        # If not enough data points, return None
        if len(before_series) < 5 or len(during_series) < 5 or len(after_series) < 5:
            logging.warning("Not enough data points for recovery analysis")
            return None
        
        try:
            # Calculate baseline statistics (before attack)
            baseline_mean = before_series.mean()
            baseline_std = before_series.std()
            baseline_median = before_series.median()
            
            # Calculate attack impact
            attack_mean = during_series.mean()
            attack_std = during_series.std()
            attack_median = during_series.median()
            
            # Calculate impact percentage
            impact_pct = ((attack_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else float('inf')
            
            # Calculate recovery time (to within 1 standard deviation of baseline)
            recovery_threshold = baseline_mean + baseline_std
            is_above_threshold = impact_pct > 0  # If impact increases the metric, we look for decrease in recovery
            
            recovery_time = None
            recovery_point = None
            recovery_percentage = None
            
            if is_above_threshold:
                # Impact increased the metric, look for decrease during recovery
                for i, value in enumerate(after_series):
                    if value <= recovery_threshold:
                        recovery_time = i
                        recovery_point = value
                        break
            else:
                # Impact decreased the metric, look for increase during recovery
                for i, value in enumerate(after_series):
                    if value >= recovery_threshold:
                        recovery_time = i
                        recovery_point = value
                        break
            
            # Calculate final recovery level
            final_value = after_series.iloc[-1] if len(after_series) > 0 else None
            
            if final_value is not None and baseline_mean != 0:
                recovery_percentage = ((final_value - baseline_mean) / baseline_mean) * 100
            
            # Combine results
            results = {
                'baseline_mean': baseline_mean,
                'baseline_std': baseline_std,
                'baseline_median': baseline_median,
                'attack_mean': attack_mean,
                'attack_std': attack_std,
                'attack_median': attack_median,
                'impact_percentage': impact_pct,
                'recovery_time': recovery_time,
                'recovery_point': recovery_point,
                'recovery_percentage': recovery_percentage,
                'final_value': final_value
            }
            
            # Create recovery visualization
            if self.plots_dir:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                
                # Plot the three phases
                time_before = np.arange(len(before_series))
                time_during = np.arange(len(during_series)) + len(before_series)
                time_after = np.arange(len(after_series)) + len(before_series) + len(during_series)
                
                # Phase 1: Before attack
                ax1.plot(time_before, before_series, 'b-', label='Baseline')
                # Phase 2: During attack
                ax1.plot(time_during, during_series, 'r-', label='Attack')
                # Phase 3: Recovery
                ax1.plot(time_after, after_series, 'g-', label='Recovery')
                
                # Add baseline mean and threshold
                ax1.axhline(y=baseline_mean, color='b', linestyle='--', label='Baseline Mean')
                ax1.axhline(y=recovery_threshold, color='b', linestyle=':', label='Recovery Threshold')
                
                # Mark recovery point if available
                if recovery_time is not None:
                    recovery_x = time_after[recovery_time]
                    ax1.scatter([recovery_x], [recovery_point], color='g', s=100, zorder=5, label='Recovery Point')
                    ax1.axvline(x=recovery_x, color='g', linestyle='--', alpha=0.5)
                
                ax1.set_title(f"Recovery Analysis: {component_name} - {metric_name}")
                ax1.set_ylabel(column)
                ax1.legend(loc='best')
                ax1.grid(True, alpha=0.3)
                
                # Create a percent change plot
                combined_series = pd.concat([before_series, during_series, after_series])
                pct_change = ((combined_series - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else None
                
                if pct_change is not None:
                    time_combined = np.arange(len(combined_series))
                    ax2.plot(time_combined, pct_change, 'k-', alpha=0.7)
                    ax2.axhline(y=0, color='k', linestyle='--')
                    
                    # Add phase boundaries
                    ax2.axvline(x=len(before_series), color='r', linestyle='-', alpha=0.5, label='Attack Starts')
                    ax2.axvline(x=len(before_series) + len(during_series), color='g', linestyle='-', alpha=0.5, label='Recovery Starts')
                    
                    ax2.set_title("Percentage Change from Baseline")
                    ax2.set_ylabel('% Change')
                    ax2.set_xlabel('Time Points')
                    ax2.legend(loc='best')
                    ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Generate filename
                fname = f"recovery_{component_name}_{metric_name}".replace('/', '_').replace(' ', '_').lower()
                filepath = self.time_series_dir / f"{fname}.png"
                plt.savefig(filepath, bbox_inches='tight')
                
                # Also save as PDF for publications
                filepath_pdf = self.time_series_dir / f"{fname}.pdf"
                fig.savefig(filepath_pdf, format='pdf', bbox_inches='tight')
                
                plt.close()
                
                logging.info(f"Saved recovery analysis plot to {filepath}")
            
            # Save results
            if self.results_dir:
                # Convert results to DataFrame
                results_df = pd.DataFrame([results])
                
                # Save to CSV
                csv_path = self.results_dir / f"recovery_{component_name}_{metric_name}.csv"
                results_df.to_csv(csv_path, index=False)
                
                # Save as LaTeX
                tex_path = self.results_dir / f"recovery_{component_name}_{metric_name}.tex"
                with open(tex_path, 'w') as f:
                    f.write(results_df.to_latex(index=False))
                
                logging.info(f"Saved recovery analysis results to {csv_path}")
            
            return results
            
        except Exception as e:
            logging.error(f"Error in recovery analysis: {e}")
            return None


# Example usage
if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Example setup (would need actual data to run this)
    analyzer = AdvancedAnalyzer(output_dir="results/advanced_analysis")
    
    print("Advanced Statistical Analyzer initialized.")
    print("This module provides methods for:")
    print("- Time Series Decomposition")
    print("- Change Point Detection")
    print("- K-means Clustering")
    print("- Anomaly Detection")
    print("- Distribution Analysis")
    print("- Recovery Metrics Analysis")
    print("")
    print("Import this module and use it with your experimental data.")
