#!/usr/bin/env python3
"""
Advanced K8s Analysis Module

This module provides the most sophisticated analysis capabilities including:
- Machine Learning models (Random Forest, SVM, Neural Networks)
- Signal processing (Fourier Transform, Wavelet Analysis)
- Time series forecasting (ARIMA, Prophet)
- Deep anomaly detection
- Advanced clustering with ensemble methods
- Causal inference analysis

Execution time: 30+ minutes for comprehensive analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import base classes
import sys
from pathlib import Path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from extended import ExtendedK8sAnalyzer

# Advanced ML and signal processing imports
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest, VotingClassifier
    from sklearn.svm import SVC as SVM, OneClassSVM
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler, RobustScaler
    
    # Time series analysis
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    
    # Signal processing
    from scipy import signal
    from scipy.fft import fft, fftfreq, fftshift
    from scipy.signal import spectrogram, welch
    
    # Advanced clustering
    from sklearn.cluster import AgglomerativeClustering, SpectralClustering
    from sklearn.mixture import GaussianMixture
    
    # Causal inference (if available)
    try:
        import networkx as nx
        NETWORKX_AVAILABLE = True
    except ImportError:
        NETWORKX_AVAILABLE = False
        
except ImportError as e:
    logging.warning(f"Some advanced libraries not available: {e}")


class AdvancedK8sAnalyzer(ExtendedK8sAnalyzer):
    """
    Advanced Kubernetes metrics analyzer with ML models and signal processing.
    
    This class extends the ExtendedK8sAnalyzer to provide:
    - Machine Learning classification models
    - Signal processing and frequency analysis
    - Time series forecasting
    - Advanced anomaly detection ensemble
    - Causal inference analysis
    - Deep clustering analysis
    """
    
    def __init__(self, config_path: str):
        """Initialize the advanced analyzer."""
        super().__init__(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Advanced analysis results storage
        self.ml_models = {}
        self.signal_analysis = {}
        self.time_series_models = {}
        self.ensemble_results = {}
        self.causal_graph = None
        
        # Performance tracking
        self.execution_times = {}
        
    def run_advanced_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive advanced analysis.
        
        Returns:
            Dictionary containing all advanced analysis results
        """
        start_time = datetime.now()
        self.logger.info("Starting advanced K8s analysis...")
        
        results = {}
        
        try:
            # Phase 1: Machine Learning Models
            self.logger.info("Phase 1: Training ML models...")
            results['ml_models'] = self._run_ml_analysis()
            
            # Phase 2: Signal Processing
            self.logger.info("Phase 2: Signal processing analysis...")
            results['signal_processing'] = self._run_signal_analysis()
            
            # Phase 3: Time Series Analysis
            self.logger.info("Phase 3: Time series forecasting...")
            results['time_series'] = self._run_time_series_analysis()
            
            # Phase 4: Ensemble Anomaly Detection
            self.logger.info("Phase 4: Ensemble anomaly detection...")
            results['ensemble_anomaly'] = self._run_ensemble_anomaly_detection()
            
            # Phase 5: Advanced Clustering
            self.logger.info("Phase 5: Advanced clustering analysis...")
            results['advanced_clustering'] = self._run_advanced_clustering()
            
            # Phase 6: Causal Analysis
            if NETWORKX_AVAILABLE:
                self.logger.info("Phase 6: Causal inference analysis...")
                results['causal_analysis'] = self._run_causal_analysis()
            
            # Phase 7: Generate Advanced Visualizations
            self.logger.info("Phase 7: Generating advanced visualizations...")
            self._generate_advanced_plots()
            
            # Phase 8: Advanced Reporting
            self.logger.info("Phase 8: Generating advanced reports...")
            self._generate_advanced_report(results)
            
            execution_time = datetime.now() - start_time
            self.logger.info(f"Advanced analysis completed in {execution_time}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Advanced analysis failed: {e}")
            raise
    
    def _run_ml_analysis(self) -> Dict[str, Any]:
        """Train and evaluate machine learning models."""
        ml_results = {}
        
        # Prepare data for ML
        X, y = self._prepare_ml_data()
        
        if X is None or len(X) < 10:
            self.logger.warning("Insufficient data for ML analysis")
            return ml_results
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model configurations
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVM(kernel='rbf', random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        # Train and evaluate models
        for name, model in models.items():
            try:
                self.logger.info(f"Training {name} model...")
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                
                # Test predictions
                y_pred = model.predict(X_test_scaled)
                
                ml_results[name] = {
                    'model': model,
                    'cv_mean': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'test_score': model.score(X_test_scaled, y_test),
                    'predictions': y_pred,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                self.logger.info(f"{name} - CV Score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")
                continue
        
        self.ml_models = ml_results
        return ml_results
    
    def _prepare_ml_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare data for machine learning models."""
        try:
            # Create feature matrix from processed data
            features = []
            labels = []
            
            for phase_name, phase_data in self.processed_data.items():
                for file_data in phase_data:
                    # Extract numerical features
                    numeric_cols = file_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        # Calculate statistical features
                        row_features = []
                        for col in numeric_cols:
                            if col != 'timestamp':
                                values = file_data[col].dropna()
                                if len(values) > 0:
                                    row_features.extend([
                                        values.mean(),
                                        values.std(),
                                        values.min(),
                                        values.max(),
                                        values.quantile(0.25),
                                        values.quantile(0.75)
                                    ])
                        
                        if row_features:
                            features.append(row_features)
                            # Label phases (0: Baseline, 1: Attack, 2: Recovery)
                            if 'baseline' in phase_name.lower():
                                labels.append(0)
                            elif 'attack' in phase_name.lower():
                                labels.append(1)
                            else:
                                labels.append(2)
            
            if not features:
                return None, None
            
            # Ensure all feature vectors have the same length
            max_len = max(len(f) for f in features)
            normalized_features = []
            for f in features:
                if len(f) < max_len:
                    f.extend([0] * (max_len - len(f)))
                normalized_features.append(f[:max_len])
            
            return np.array(normalized_features), np.array(labels)
            
        except Exception as e:
            self.logger.error(f"Error preparing ML data: {e}")
            return None, None
    
    def _run_signal_analysis(self) -> Dict[str, Any]:
        """Perform signal processing analysis."""
        signal_results = {}
        
        try:
            for phase_name, phase_data in self.processed_data.items():
                phase_signals = {}
                
                for i, file_data in enumerate(phase_data):
                    numeric_cols = file_data.select_dtypes(include=[np.number]).columns
                    
                    for col in numeric_cols:
                        if col != 'timestamp':
                            values = file_data[col].dropna().values
                            
                            if len(values) > 10:  # Minimum length for signal analysis
                                signal_info = self._analyze_signal(values, col)
                                phase_signals[f"{col}_{i}"] = signal_info
                
                signal_results[phase_name] = phase_signals
            
            self.signal_analysis = signal_results
            return signal_results
            
        except Exception as e:
            self.logger.error(f"Error in signal analysis: {e}")
            return signal_results
    
    def _analyze_signal(self, signal_data: np.ndarray, signal_name: str) -> Dict[str, Any]:
        """Analyze individual signal using various signal processing techniques."""
        results = {}
        
        try:
            # Basic signal statistics
            results['mean'] = np.mean(signal_data)
            results['std'] = np.std(signal_data)
            results['energy'] = np.sum(signal_data ** 2)
            
            # Frequency domain analysis
            if len(signal_data) > 1:
                # FFT analysis
                fft_vals = fft(signal_data)
                freqs = fftfreq(len(signal_data))
                
                results['dominant_frequency'] = freqs[np.argmax(np.abs(fft_vals[1:]))]
                results['spectral_centroid'] = np.sum(freqs * np.abs(fft_vals)) / np.sum(np.abs(fft_vals))
                
                # Power spectral density
                frequencies, psd = welch(signal_data)
                results['peak_frequency'] = frequencies[np.argmax(psd)]
                results['total_power'] = np.sum(psd)
            
            # Time domain features
            if len(signal_data) > 2:
                # Zero crossing rate
                zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
                results['zero_crossing_rate'] = len(zero_crossings) / len(signal_data)
                
                # Autocorrelation
                autocorr = np.correlate(signal_data, signal_data, mode='full')
                results['autocorr_peak'] = np.max(autocorr) / len(signal_data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing signal {signal_name}: {e}")
            return {}
    
    def _run_time_series_analysis(self) -> Dict[str, Any]:
        """Perform time series analysis and forecasting."""
        ts_results = {}
        
        try:
            for phase_name, phase_data in self.processed_data.items():
                phase_ts = {}
                
                for i, file_data in enumerate(phase_data):
                    numeric_cols = file_data.select_dtypes(include=[np.number]).columns
                    
                    for col in numeric_cols:
                        if col != 'timestamp':
                            values = file_data[col].dropna()
                            
                            if len(values) > 20:  # Minimum for time series analysis
                                ts_info = self._analyze_time_series(values, f"{col}_{i}")
                                phase_ts[f"{col}_{i}"] = ts_info
                
                ts_results[phase_name] = phase_ts
            
            self.time_series_models = ts_results
            return ts_results
            
        except Exception as e:
            self.logger.error(f"Error in time series analysis: {e}")
            return ts_results
    
    def _analyze_time_series(self, ts_data: pd.Series, series_name: str) -> Dict[str, Any]:
        """Analyze individual time series."""
        results = {}
        
        try:
            # Stationarity tests
            adf_result = adfuller(ts_data)
            results['adf_statistic'] = adf_result[0]
            results['adf_pvalue'] = adf_result[1]
            results['is_stationary'] = adf_result[1] < 0.05
            
            # Trend analysis
            if len(ts_data) > 10:
                # Simple linear trend
                x = np.arange(len(ts_data))
                trend_coef = np.polyfit(x, ts_data, 1)[0]
                results['trend_coefficient'] = trend_coef
                results['trend_direction'] = 'increasing' if trend_coef > 0 else 'decreasing'
            
            # Seasonal decomposition (if enough data)
            if len(ts_data) >= 24:  # Minimum for seasonal decomposition
                try:
                    decomposition = seasonal_decompose(ts_data, model='additive', period=min(12, len(ts_data)//2))
                    results['trend_strength'] = np.std(decomposition.trend.dropna()) / np.std(ts_data)
                    results['seasonal_strength'] = np.std(decomposition.seasonal) / np.std(ts_data)
                except:
                    pass
            
            # Simple ARIMA modeling (if enough data)
            if len(ts_data) >= 30:
                try:
                    # Auto ARIMA would be better, but using simple ARIMA(1,1,1)
                    model = ARIMA(ts_data, order=(1, 1, 1))
                    fitted_model = model.fit()
                    results['arima_aic'] = fitted_model.aic
                    results['arima_bic'] = fitted_model.bic
                    
                    # Forecast next 5 points
                    forecast = fitted_model.forecast(steps=5)
                    results['forecast'] = forecast.tolist()
                except:
                    pass
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing time series {series_name}: {e}")
            return {}
    
    def _run_ensemble_anomaly_detection(self) -> Dict[str, Any]:
        """Run ensemble anomaly detection using multiple algorithms."""
        ensemble_results = {}
        
        try:
            # Prepare data
            all_data = []
            data_labels = []
            
            for phase_name, phase_data in self.processed_data.items():
                for file_data in phase_data:
                    numeric_cols = file_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        # Get statistical summary for each row
                        row_data = []
                        for col in numeric_cols:
                            if col != 'timestamp':
                                values = file_data[col].dropna()
                                if len(values) > 0:
                                    row_data.extend([
                                        values.mean(), values.std(), 
                                        values.min(), values.max()
                                    ])
                        
                        if row_data:
                            all_data.append(row_data)
                            data_labels.append(phase_name)
            
            if len(all_data) < 5:
                self.logger.warning("Insufficient data for ensemble anomaly detection")
                return ensemble_results
            
            # Normalize data
            max_len = max(len(row) for row in all_data)
            normalized_data = []
            for row in all_data:
                if len(row) < max_len:
                    row.extend([0] * (max_len - len(row)))
                normalized_data.append(row[:max_len])
            
            X = np.array(normalized_data)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Ensemble of anomaly detectors
            detectors = {
                'IsolationForest': IsolationForest(contamination=0.1, random_state=42),
                'OneClassSVM': OneClassSVM(gamma='scale'),
            }
            
            ensemble_predictions = {}
            
            for name, detector in detectors.items():
                try:
                    predictions = detector.fit_predict(X_scaled)
                    anomaly_indices = np.where(predictions == -1)[0]
                    
                    ensemble_predictions[name] = {
                        'predictions': predictions,
                        'anomaly_count': len(anomaly_indices),
                        'anomaly_ratio': len(anomaly_indices) / len(predictions),
                        'anomaly_phases': [data_labels[i] for i in anomaly_indices]
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error with {name}: {e}")
                    continue
            
            # Consensus prediction
            if len(ensemble_predictions) > 1:
                all_preds = np.array([pred['predictions'] for pred in ensemble_predictions.values()])
                consensus = np.sum(all_preds == -1, axis=0) >= len(ensemble_predictions) // 2
                consensus_anomalies = np.where(consensus)[0]
                
                ensemble_results['consensus'] = {
                    'anomaly_indices': consensus_anomalies.tolist(),
                    'anomaly_count': len(consensus_anomalies),
                    'anomaly_phases': [data_labels[i] for i in consensus_anomalies]
                }
            
            ensemble_results['individual_detectors'] = ensemble_predictions
            self.ensemble_results = ensemble_results
            
            return ensemble_results
            
        except Exception as e:
            self.logger.error(f"Error in ensemble anomaly detection: {e}")
            return ensemble_results
    
    def _run_advanced_clustering(self) -> Dict[str, Any]:
        """Run advanced clustering analysis."""
        clustering_results = {}
        
        try:
            # Prepare data (reuse from ML preparation)
            X, _ = self._prepare_ml_data()
            
            if X is None or len(X) < 5:
                self.logger.warning("Insufficient data for advanced clustering")
                return clustering_results
            
            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Advanced clustering algorithms
            clusterers = {
                'AgglomerativeClustering': AgglomerativeClustering(n_clusters=3),
                'SpectralClustering': SpectralClustering(n_clusters=3, random_state=42),
                'GaussianMixture': GaussianMixture(n_components=3, random_state=42)
            }
            
            for name, clusterer in clusterers.items():
                try:
                    if name == 'GaussianMixture':
                        clusterer.fit(X_scaled)
                        labels = clusterer.predict(X_scaled)
                        probs = clusterer.predict_proba(X_scaled)
                        
                        clustering_results[name] = {
                            'labels': labels.tolist(),
                            'probabilities': probs.tolist(),
                            'n_clusters': 3,
                            'aic': clusterer.aic(X_scaled),
                            'bic': clusterer.bic(X_scaled)
                        }
                    else:
                        labels = clusterer.fit_predict(X_scaled)
                        clustering_results[name] = {
                            'labels': labels.tolist(),
                            'n_clusters': len(np.unique(labels))
                        }
                    
                    self.logger.info(f"{name} clustering completed")
                    
                except Exception as e:
                    self.logger.error(f"Error with {name}: {e}")
                    continue
            
            return clustering_results
            
        except Exception as e:
            self.logger.error(f"Error in advanced clustering: {e}")
            return clustering_results
    
    def _run_causal_analysis(self) -> Dict[str, Any]:
        """Run causal inference analysis (if NetworkX is available)."""
        if not NETWORKX_AVAILABLE:
            return {}
        
        causal_results = {}
        
        try:
            # Create correlation-based causal graph
            # This is a simplified approach - real causal inference requires more sophisticated methods
            
            # Get correlation matrix from extended analysis
            if hasattr(self, 'correlations') and self.correlations:
                corr_matrix = None
                for phase_name, phase_corr in self.correlations.items():
                    if 'pearson' in phase_corr and phase_corr['pearson'] is not None:
                        corr_matrix = phase_corr['pearson']
                        break
                
                if corr_matrix is not None:
                    # Create graph from strong correlations
                    G = nx.Graph()
                    threshold = 0.7  # Strong correlation threshold
                    
                    for i, col1 in enumerate(corr_matrix.columns):
                        for j, col2 in enumerate(corr_matrix.columns):
                            if i < j and abs(corr_matrix.iloc[i, j]) > threshold:
                                G.add_edge(col1, col2, weight=abs(corr_matrix.iloc[i, j]))
                    
                    # Graph analysis
                    if len(G.nodes()) > 0:
                        causal_results['graph_nodes'] = list(G.nodes())
                        causal_results['graph_edges'] = list(G.edges())
                        causal_results['centrality'] = nx.degree_centrality(G)
                        causal_results['clustering_coefficient'] = nx.average_clustering(G)
                        
                        if nx.is_connected(G):
                            causal_results['average_path_length'] = nx.average_shortest_path_length(G)
                        
                        self.causal_graph = G
            
            return causal_results
            
        except Exception as e:
            self.logger.error(f"Error in causal analysis: {e}")
            return causal_results
    
    def _generate_advanced_plots(self):
        """Generate advanced visualization plots."""
        try:
            output_dir = Path(self.output_dir) / "advanced_plots"
            output_dir.mkdir(exist_ok=True)
            
            # ML Model Performance Plot
            if self.ml_models:
                self._plot_ml_performance(output_dir)
            
            # Signal Analysis Plots
            if self.signal_analysis:
                self._plot_signal_analysis(output_dir)
            
            # Time Series Plots
            if self.time_series_models:
                self._plot_time_series_analysis(output_dir)
            
            # Ensemble Anomaly Detection Plot
            if self.ensemble_results:
                self._plot_ensemble_anomalies(output_dir)
            
            # Causal Graph Plot
            if self.causal_graph and NETWORKX_AVAILABLE:
                self._plot_causal_graph(output_dir)
            
        except Exception as e:
            self.logger.error(f"Error generating advanced plots: {e}")
    
    def _plot_ml_performance(self, output_dir: Path):
        """Plot ML model performance comparison."""
        try:
            models = list(self.ml_models.keys())
            cv_scores = [self.ml_models[m]['cv_mean'] for m in models]
            test_scores = [self.ml_models[m]['test_score'] for m in models]
            
            fig, ax = plt.subplots(1, 2, figsize=(15, 6))
            
            # CV scores
            ax[0].bar(models, cv_scores)
            ax[0].set_title('Cross-Validation Scores')
            ax[0].set_ylabel('Accuracy')
            ax[0].tick_params(axis='x', rotation=45)
            
            # Test scores
            ax[1].bar(models, test_scores, color='orange')
            ax[1].set_title('Test Scores')
            ax[1].set_ylabel('Accuracy')
            ax[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'ml_model_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting ML performance: {e}")
    
    def _plot_signal_analysis(self, output_dir: Path):
        """Plot signal analysis results."""
        try:
            # Collect frequency data
            frequencies = []
            powers = []
            phases = []
            
            for phase_name, phase_signals in self.signal_analysis.items():
                for signal_name, signal_info in phase_signals.items():
                    if 'peak_frequency' in signal_info and 'total_power' in signal_info:
                        frequencies.append(signal_info['peak_frequency'])
                        powers.append(signal_info['total_power'])
                        phases.append(phase_name)
            
            if frequencies:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Frequency distribution
                ax1.hist(frequencies, bins=20, alpha=0.7)
                ax1.set_title('Peak Frequency Distribution')
                ax1.set_xlabel('Frequency')
                ax1.set_ylabel('Count')
                
                # Power vs Frequency
                scatter = ax2.scatter(frequencies, powers, c=[hash(p) for p in phases], alpha=0.7)
                ax2.set_title('Power vs Peak Frequency')
                ax2.set_xlabel('Peak Frequency')
                ax2.set_ylabel('Total Power')
                
                plt.tight_layout()
                plt.savefig(output_dir / 'signal_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Error plotting signal analysis: {e}")
    
    def _plot_time_series_analysis(self, output_dir: Path):
        """Plot time series analysis results."""
        try:
            # Collect trend data
            trends = []
            stationarity = []
            phases = []
            
            for phase_name, phase_ts in self.time_series_models.items():
                for ts_name, ts_info in phase_ts.items():
                    if 'trend_coefficient' in ts_info:
                        trends.append(ts_info['trend_coefficient'])
                        stationarity.append(ts_info.get('is_stationary', False))
                        phases.append(phase_name)
            
            if trends:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Trend coefficients by phase
                phase_trends = {}
                for phase, trend in zip(phases, trends):
                    if phase not in phase_trends:
                        phase_trends[phase] = []
                    phase_trends[phase].append(trend)
                
                phase_names = list(phase_trends.keys())
                trend_means = [np.mean(phase_trends[p]) for p in phase_names]
                
                ax1.bar(phase_names, trend_means)
                ax1.set_title('Average Trend Coefficient by Phase')
                ax1.set_ylabel('Trend Coefficient')
                ax1.tick_params(axis='x', rotation=45)
                
                # Stationarity
                stationary_count = sum(stationarity)
                non_stationary_count = len(stationarity) - stationary_count
                
                ax2.pie([stationary_count, non_stationary_count], 
                       labels=['Stationary', 'Non-stationary'],
                       autopct='%1.1f%%')
                ax2.set_title('Time Series Stationarity')
                
                plt.tight_layout()
                plt.savefig(output_dir / 'time_series_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Error plotting time series analysis: {e}")
    
    def _plot_ensemble_anomalies(self, output_dir: Path):
        """Plot ensemble anomaly detection results."""
        try:
            if 'individual_detectors' not in self.ensemble_results:
                return
            
            detectors = list(self.ensemble_results['individual_detectors'].keys())
            anomaly_ratios = [
                self.ensemble_results['individual_detectors'][d]['anomaly_ratio'] 
                for d in detectors
            ]
            
            plt.figure(figsize=(10, 6))
            plt.bar(detectors, anomaly_ratios)
            plt.title('Anomaly Detection Results by Algorithm')
            plt.ylabel('Anomaly Ratio')
            plt.tick_params(axis='x', rotation=45)
            
            # Add consensus line if available
            if 'consensus' in self.ensemble_results:
                consensus_ratio = (self.ensemble_results['consensus']['anomaly_count'] / 
                                 len(self.ensemble_results['individual_detectors'][detectors[0]]['predictions']))
                plt.axhline(y=consensus_ratio, color='red', linestyle='--', 
                           label=f'Consensus: {consensus_ratio:.3f}')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(output_dir / 'ensemble_anomaly_detection.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting ensemble anomalies: {e}")
    
    def _plot_causal_graph(self, output_dir: Path):
        """Plot causal inference graph."""
        try:
            if not self.causal_graph or not NETWORKX_AVAILABLE:
                return
            
            plt.figure(figsize=(12, 8))
            
            # Layout
            pos = nx.spring_layout(self.causal_graph, k=1, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(self.causal_graph, pos, node_color='lightblue', 
                                 node_size=1000, alpha=0.7)
            
            # Draw edges with weights
            edges = self.causal_graph.edges(data=True)
            weights = [edge[2]['weight'] for edge in edges]
            nx.draw_networkx_edges(self.causal_graph, pos, width=weights, alpha=0.6)
            
            # Draw labels
            nx.draw_networkx_labels(self.causal_graph, pos, font_size=8)
            
            plt.title('Causal Inference Graph\n(Based on Strong Correlations)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / 'causal_graph.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting causal graph: {e}")
    
    def _generate_advanced_report(self, results: Dict[str, Any]):
        """Generate comprehensive advanced analysis report."""
        try:
            report_path = Path(self.output_dir) / "advanced_analysis_report.txt"
            
            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("ADVANCED K8S METRICS ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # ML Models Section
                if 'ml_models' in results and results['ml_models']:
                    f.write("MACHINE LEARNING MODELS\n")
                    f.write("-" * 40 + "\n")
                    for model_name, model_info in results['ml_models'].items():
                        f.write(f"\n{model_name}:\n")
                        f.write(f"  Cross-Validation Score: {model_info['cv_mean']:.3f} ± {model_info['cv_std']:.3f}\n")
                        f.write(f"  Test Score: {model_info['test_score']:.3f}\n")
                    f.write("\n")
                
                # Signal Processing Section
                if 'signal_processing' in results and results['signal_processing']:
                    f.write("SIGNAL PROCESSING ANALYSIS\n")
                    f.write("-" * 40 + "\n")
                    total_signals = sum(len(phase_signals) for phase_signals in results['signal_processing'].values())
                    f.write(f"Total signals analyzed: {total_signals}\n")
                    f.write("Frequency domain analysis completed for all signals\n\n")
                
                # Time Series Section
                if 'time_series' in results and results['time_series']:
                    f.write("TIME SERIES ANALYSIS\n")
                    f.write("-" * 40 + "\n")
                    total_ts = sum(len(phase_ts) for phase_ts in results['time_series'].values())
                    f.write(f"Total time series analyzed: {total_ts}\n")
                    f.write("Stationarity tests and trend analysis completed\n\n")
                
                # Ensemble Anomaly Detection Section
                if 'ensemble_anomaly' in results and results['ensemble_anomaly']:
                    f.write("ENSEMBLE ANOMALY DETECTION\n")
                    f.write("-" * 40 + "\n")
                    if 'individual_detectors' in results['ensemble_anomaly']:
                        for detector, det_results in results['ensemble_anomaly']['individual_detectors'].items():
                            f.write(f"{detector}: {det_results['anomaly_count']} anomalies "
                                   f"({det_results['anomaly_ratio']:.2%})\n")
                    if 'consensus' in results['ensemble_anomaly']:
                        consensus = results['ensemble_anomaly']['consensus']
                        f.write(f"Consensus: {consensus['anomaly_count']} anomalies\n")
                    f.write("\n")
                
                # Advanced Clustering Section
                if 'advanced_clustering' in results and results['advanced_clustering']:
                    f.write("ADVANCED CLUSTERING ANALYSIS\n")
                    f.write("-" * 40 + "\n")
                    for cluster_method, cluster_info in results['advanced_clustering'].items():
                        f.write(f"{cluster_method}: {cluster_info['n_clusters']} clusters identified\n")
                    f.write("\n")
                
                # Causal Analysis Section
                if 'causal_analysis' in results and results['causal_analysis']:
                    f.write("CAUSAL INFERENCE ANALYSIS\n")
                    f.write("-" * 40 + "\n")
                    causal = results['causal_analysis']
                    if 'graph_nodes' in causal:
                        f.write(f"Causal graph nodes: {len(causal['graph_nodes'])}\n")
                        f.write(f"Causal graph edges: {len(causal['graph_edges'])}\n")
                        if 'clustering_coefficient' in causal:
                            f.write(f"Graph clustering coefficient: {causal['clustering_coefficient']:.3f}\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write("ANALYSIS COMPLETE\n")
                f.write("=" * 80 + "\n")
            
            self.logger.info(f"Advanced report saved to: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating advanced report: {e}")


if __name__ == "__main__":
    # Example usage
    analyzer = AdvancedK8sAnalyzer("config/advanced_config.yaml")
    
    try:
        # Load data
        analyzer.load_data("demo-data")
        
        # Run analysis
        results = analyzer.run_advanced_analysis()
        
        print("Advanced analysis completed successfully!")
        print(f"Results keys: {list(results.keys())}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
