#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loader for Kubernetes Noisy Neighbours Lab
This module loads and preprocesses data from the experiment results.
"""

import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class DataLoader:
    def __init__(self, base_path, experiment_name, round_number=None):
        """
        Initialize the data loader.
        
        Args:
            base_path (str): Path to the results directory
            experiment_name (str): Name of the experiment (e.g. YYYY-MM-DD/HH-MM-SS/default-experiment-1)
            round_number (str or list): Round number(s) (e.g. "round-1" or ["round-1", "round-2"])
                                        If None, all rounds found will be processed
        """
        self.base_path = Path(base_path)
        self.experiment_name = experiment_name
        
        # Check if experiment path points to demo-data for easier testing
        if 'demo-data' in experiment_name:
            self.experiment_path = self.base_path / experiment_name
        else:
            self.experiment_path = self.base_path / 'results' / experiment_name
        
        # Handle round number options
        self.rounds = []
        if round_number is None:
            # Auto-detect rounds
            round_dirs = [d for d in self.experiment_path.glob("round-*") if d.is_dir()]
            self.rounds = [d.name for d in round_dirs]
            self.rounds.sort()  # Sort rounds for consistency
            logging.info(f"Auto-detected rounds: {self.rounds}")
        elif isinstance(round_number, list):
            self.rounds = round_number
        else:
            self.rounds = [round_number]
            
        # Check if rounds exist
        for round_name in self.rounds:
            round_path = self.experiment_path / round_name
            if not round_path.exists():
                logging.error(f"Round path does not exist: {round_path}")
                raise ValueError(f"Round path does not exist: {round_path}")
        
        self.current_round = self.rounds[0] if self.rounds else None
        self.results_path = self.experiment_path / self.current_round if self.current_round else None
        self.data = {}
        self.combined_data = {}  # For storing combined data from multiple rounds
        
        logging.info(f"Initialized DataLoader for {experiment_name}, rounds: {self.rounds}")
        
    def set_current_round(self, round_number):
        """
        Set the current round for loading data.
        
        Args:
            round_number (str): Round number (e.g. "round-1")
            
        Returns:
            bool: True if successful, False otherwise
        """
        if round_number in self.rounds:
            self.current_round = round_number
            self.results_path = self.experiment_path / self.current_round
            return True
        else:
            logging.error(f"Round {round_number} not found in available rounds: {self.rounds}")
            return False
    
    def load_all_phases(self, phases=None):
        """
        Load data from all phases.
        
        Args:
            phases (list): List of phases to load. If None, all phases in the directory will be loaded.
            
        Returns:
            dict: Dictionary with phase names as keys and data dictionaries as values
        """
        # If phases not provided, detect phases from directory structure
        if phases is None:
            phases = [d.name for d in self.results_path.iterdir() if d.is_dir()]
            phases.sort()  # Sort to ensure consistent order (1-baseline, 2-attack, 3-recovery)
        
        # Load data for each phase
        for phase in phases:
            logging.info(f"Loading data for phase: {phase}")
            self.data[phase] = self._load_phase_data(phase)
        
        return self.data
    
    def _load_phase_data(self, phase):
        """
        Load all metrics data for a specific phase.
        
        Args:
            phase (str): Phase name
            
        Returns:
            dict: Dictionary with component/tenant names as keys and data dictionaries as values
        """
        phase_path = self.results_path / phase
        if not phase_path.exists():
            logging.error(f"Phase path does not exist: {phase_path}")
            return {}
        
        phase_data = {}
        
        # Get all component directories (tenant-a, tenant-b, etc.)
        components = [d for d in phase_path.iterdir() if d.is_dir()]
        
        for component in components:
            logging.info(f"Loading data for component: {component.name}")
            component_data = self._load_component_data(component)
            if component_data:
                phase_data[component.name] = component_data
        
        return phase_data
    
    def _load_component_data(self, component_path):
        """
        Load all metrics data for a specific component.
        
        Args:
            component_path (Path): Path to the component directory
            
        Returns:
            dict: Dictionary with metric names as keys and pandas DataFrames as values
        """
        component_data = {}
        
        # Get all metric CSV files
        csv_files = glob.glob(str(component_path / "*.csv"))
        
        for csv_file in csv_files:
            metric_name = Path(csv_file).stem
            try:
                # Load CSV file into pandas DataFrame
                df = pd.read_csv(csv_file)
                
                # Basic preprocessing
                if 'timestamp' in df.columns:
                    try:
                        # Tentar converter o formato específico "YYYYMMDD_HHMMSS"
                        df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y%m%d_%H%M%S")
                    except Exception:
                        try:
                            # Fallback para tentar o parse automático
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                        except Exception as e:
                            logging.warning(f"Failed to parse timestamps for {metric_name}: {e}")
                            # Usar índice numérico quando não conseguir converter timestamp
                            component_data[metric_name] = df
                            continue
                    
                    # Verificar e tratar timestamps duplicados
                    if df['timestamp'].duplicated().any():
                        logging.warning(f"Duplicate timestamps found in {metric_name}, aggregating by mean")
                        # Agregar valores duplicados usando a média
                        df = df.groupby('timestamp').mean(numeric_only=True).reset_index()
                    
                    df = df.set_index('timestamp')
                
                component_data[metric_name] = df
                logging.debug(f"Loaded metric: {metric_name}")
            except Exception as e:
                logging.error(f"Error loading {csv_file}: {e}")
                continue
        
        return component_data
    
    def load_all_rounds(self, phases=None):
        """
        Load data from all rounds and store separately.
        
        Args:
            phases (list): List of phases to load for each round
            
        Returns:
            dict: Dictionary with round names as keys and data dictionaries as values
        """
        rounds_data = {}
        
        for round_name in self.rounds:
            logging.info(f"Loading data for round: {round_name}")
            self.set_current_round(round_name)
            rounds_data[round_name] = self.load_all_phases(phases)
        
        return rounds_data
    
    def combine_rounds(self, phases=None, method='mean'):
        """
        Combine data from multiple rounds, calculating average or other statistics.
        
        Args:
            phases (list): List of phases to combine. If None, all phases will be combined.
            method (str): Method for combining data ('mean', 'median', 'min', 'max')
            
        Returns:
            dict: Dictionary with phase names as keys and combined data as values
        """
        if len(self.rounds) <= 1:
            logging.warning("Only one round available, no combination needed")
            return self.data
        
        # Load data from all rounds if not already loaded
        rounds_data = {}
        for round_name in self.rounds:
            if round_name not in rounds_data:
                self.set_current_round(round_name)
                rounds_data[round_name] = self.load_all_phases(phases)
        
        # Identify common phases across all rounds
        common_phases = set()
        for round_data in rounds_data.values():
            if not common_phases:
                common_phases = set(round_data.keys())
            else:
                common_phases = common_phases.intersection(set(round_data.keys()))
        
        # Filter to specified phases if provided
        if phases:
            common_phases = [phase for phase in phases if phase in common_phases]
        else:
            common_phases = list(common_phases)
        
        logging.info(f"Combining data for phases: {common_phases}")
        
        # Initialize combined data structure
        combined_data = {}
        
        # Process each phase
        for phase in common_phases:
            combined_data[phase] = {}
            
            # Identify common components across all rounds for this phase
            common_components = set()
            for round_data in rounds_data.values():
                if phase in round_data:
                    if not common_components:
                        common_components = set(round_data[phase].keys())
                    else:
                        common_components = common_components.intersection(set(round_data[phase].keys()))
            
            # Process each component
            for component in common_components:
                combined_data[phase][component] = {}
                
                # Identify common metrics across all rounds for this component
                common_metrics = set()
                for round_data in rounds_data.values():
                    if phase in round_data and component in round_data[phase]:
                        if not common_metrics:
                            common_metrics = set(round_data[phase][component].keys())
                        else:
                            common_metrics = common_metrics.intersection(set(round_data[phase][component].keys()))
                
                # Process each metric
                for metric in common_metrics:
                    # Collect dataframes from each round
                    metric_dfs = []
                    for round_data in rounds_data.values():
                        if (phase in round_data and component in round_data[phase] and 
                            metric in round_data[phase][component]):
                            metric_dfs.append(round_data[phase][component][metric])
                    
                    # Combine dataframes
                    if metric_dfs:
                        combined_df = self._combine_metric_dataframes(metric_dfs, method)
                        combined_data[phase][component][metric] = combined_df
        
        self.combined_data = combined_data
        return combined_data
    
    def _combine_metric_dataframes(self, dfs, method='mean'):
        """
        Combine multiple dataframes with the same structure.
        
        Args:
            dfs (list): List of DataFrames to combine
            method (str): Method for combining data ('mean', 'median', 'min', 'max')
            
        Returns:
            DataFrame: Combined DataFrame
        """
        if not dfs:
            return pd.DataFrame()
        
        if len(dfs) == 1:
            return dfs[0]
        
        # Reset indices to be able to align the dataframes
        aligned_dfs = []
        for df in dfs:
            # Make a copy to avoid modifying the original
            df_copy = df.copy()
            if isinstance(df_copy.index, pd.DatetimeIndex):
                # If timestamp index, convert to time since start
                start_time = df_copy.index.min()
                df_copy['time_seconds'] = (df_copy.index - start_time).total_seconds()
                df_copy = df_copy.reset_index()
            aligned_dfs.append(df_copy)
        
        # Identify common columns
        common_cols = set(aligned_dfs[0].columns)
        for df in aligned_dfs[1:]:
            common_cols = common_cols.intersection(set(df.columns))
        
        # Separate timestamp column if it exists
        timestamp_col = None
        if 'timestamp' in common_cols:
            timestamp_col = 'timestamp'
            common_cols.remove('timestamp')
        
        # Ensure we have time_seconds for alignment
        if 'time_seconds' not in common_cols:
            # If we don't have time_seconds, we can't align the dataframes well
            # Just concatenate and group by close timestamps if available
            if timestamp_col:
                combined_df = pd.concat(aligned_dfs, ignore_index=True)
                combined_df = combined_df.sort_values(timestamp_col)
                return combined_df.set_index(timestamp_col)
            else:
                # Without timestamp, just use simple concatenation
                return pd.concat(aligned_dfs, ignore_index=True)
        
        # Use time_seconds for alignment
        common_cols.remove('time_seconds')
        
        # Create interpolation points (use a reasonable number of points)
        min_time = min(df['time_seconds'].min() for df in aligned_dfs)
        max_time = max(df['time_seconds'].max() for df in aligned_dfs)
        
        # Determine a reasonable number of points based on the average sampling rate
        avg_points = int(np.mean([len(df) for df in aligned_dfs]))
        num_points = min(1000, max(100, avg_points))  # Cap between 100 and 1000 points
        
        interp_times = np.linspace(min_time, max_time, num_points)
        
        # Interpolate each DataFrame to common time points
        interp_dfs = []
        for df in aligned_dfs:
            interp_df = pd.DataFrame({'time_seconds': interp_times})
            
            for col in common_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Interpolate numeric columns
                    interp_df[col] = np.interp(
                        interp_times,
                        df['time_seconds'],
                        df[col],
                        left=np.nan,
                        right=np.nan
                    )
            
            interp_dfs.append(interp_df)
        
        # Create a combined DataFrame
        combined_df = pd.DataFrame({'time_seconds': interp_times})
        
        # Apply the specified method to combine values
        for col in common_cols:
            # Check if the column is present in all interpolated dataframes
            values_available = [col in df.columns for df in interp_dfs]
            if not all(values_available):
                continue
                
            # Extract values from each DataFrame
            col_values = np.array([df[col].values for df in interp_dfs if col in df.columns])
            
            # Apply the combination method
            if method == 'mean':
                combined_df[col] = np.nanmean(col_values, axis=0)
            elif method == 'median':
                combined_df[col] = np.nanmedian(col_values, axis=0)
            elif method == 'min':
                combined_df[col] = np.nanmin(col_values, axis=0)
            elif method == 'max':
                combined_df[col] = np.nanmax(col_values, axis=0)
            else:
                # Default to mean
                combined_df[col] = np.nanmean(col_values, axis=0)
        
        # Create timestamp index if we had timestamps originally
        if timestamp_col:
            # Generate timestamps from time_seconds
            start_time = aligned_dfs[0][timestamp_col].min()
            combined_df[timestamp_col] = pd.to_datetime(start_time) + pd.to_timedelta(combined_df['time_seconds'], unit='s')
            combined_df = combined_df.set_index(timestamp_col)
        
        # Remove the time_seconds column from final output
        if 'time_seconds' in combined_df.columns:
            combined_df = combined_df.drop(columns=['time_seconds'])
        
        return combined_df
    
    def get_metric(self, phase, component, metric):
        """
        Get a specific metric data.
        
        Args:
            phase (str): Phase name
            component (str): Component name
            metric (str): Metric name
            
        Returns:
            DataFrame: DataFrame with the metric data
        """
        try:
            return self.data[phase][component][metric]
        except KeyError:
            logging.error(f"Metric not found: {phase}/{component}/{metric}")
            return None
    
    def get_all_metrics_for_component(self, phase, component):
        """
        Get all metrics for a specific component.
        
        Args:
            phase (str): Phase name
            component (str): Component name
            
        Returns:
            dict: Dictionary with metric names as keys and pandas DataFrames as values
        """
        try:
            return self.data[phase][component]
        except KeyError:
            logging.error(f"Component not found: {phase}/{component}")
            return {}
    
    def get_same_metric_across_phases(self, component, metric, phases=None):
        """
        Get the same metric across all phases for comparison.
        
        Args:
            component (str): Component name
            metric (str): Metric name
            phases (list): List of phases to include. If None, all loaded phases will be included.
            
        Returns:
            dict: Dictionary with phase names as keys and pandas DataFrames as values
        """
        if phases is None:
            phases = list(self.data.keys())
        
        result = {}
        for phase in phases:
            try:
                result[phase] = self.data[phase][component][metric]
            except KeyError:
                logging.warning(f"Metric not found: {phase}/{component}/{metric}")
                continue
        
        return result
    
    def get_all_components(self, phase=None):
        """
        Get all component names.
        
        Args:
            phase (str): Phase name. If None, components from the first phase will be returned.
            
        Returns:
            list: List of component names
        """
        if phase is None and self.data:
            phase = list(self.data.keys())[0]
        
        try:
            return list(self.data[phase].keys())
        except KeyError:
            logging.error(f"Phase not found: {phase}")
            return []
    
    def get_all_metrics(self, phase=None, component=None):
        """
        Get all metric names.
        
        Args:
            phase (str): Phase name. If None, metrics from the first phase will be returned.
            component (str): Component name. If None, metrics from the first component will be returned.
            
        Returns:
            list: List of metric names
        """
        if phase is None and self.data:
            phase = list(self.data.keys())[0]
        
        if component is None and phase in self.data:
            component = list(self.data[phase].keys())[0]
        
        try:
            return list(self.data[phase][component].keys())
        except KeyError:
            logging.error(f"Phase or component not found: {phase}/{component}")
            return []
    
    def use_combined_data(self, combined=True):
        """
        Switch between combined data and current round data.
        
        Args:
            combined (bool): If True, use combined data, otherwise use current round data
            
        Returns:
            None
        """
        if combined and self.combined_data:
            self.data = self.combined_data
            logging.info("Switched to using combined data across rounds")
        elif not combined:
            # Reload the current round data
            self.data = {}
            self.load_all_phases()
            logging.info(f"Switched to using data from round: {self.current_round}")
        else:
            logging.warning("Combined data not available. Call combine_rounds() first.")
    
    def load_demo_data(self):
        """
        Método alternativo para carregar dados direto dos arquivos de demonstração.
        Este método usa uma estrutura mais simples e é útil para testes rápidos.
        
        Returns:
            dict: Dados organizados por fase e componente
        """
        data = {}
        
        # Certifique-se de que temos um caminho para o round atual
        if not self.results_path or not self.results_path.exists():
            logging.error(f"Caminho do round não existe: {self.results_path}")
            return data
        
        logging.info(f"Carregando dados de demonstração do diretório: {self.results_path}")
        
        # Para cada diretório de fase
        for phase_dir in self.results_path.glob("*"):
            if phase_dir.is_dir():
                phase_name = phase_dir.name
                data[phase_name] = {}
                
                # Para cada subdiretório que representa um componente
                for component_dir in phase_dir.glob("*"):
                    if component_dir.is_dir():
                        component_name = component_dir.name
                        data[phase_name][component_name] = {}
                        
                        # Carregar cada arquivo CSV como uma métrica
                        for csv_file in component_dir.glob("*.csv"):
                            metric_name = csv_file.stem
                            try:
                                df = pd.read_csv(csv_file)
                                data[phase_name][component_name][metric_name] = df
                                logging.debug(f"Carregado {metric_name} para {component_name} em {phase_name}")
                            except Exception as e:
                                logging.error(f"Erro carregando {csv_file}: {str(e)}")
        
        self.data = data
        return data


# Example usage
if __name__ == "__main__":
    # Example usage
    data_loader = DataLoader(
        base_path="/home/phil/Projects/k8s-noisy-lab-data-pipe",
        experiment_name="2025-05-11/16-58-00/default-experiment-1",
        round_number=["round-1", "round-2", "round-3"]  # Multiple rounds
    )
    
    # Load and combine data from multiple rounds
    combined_data = data_loader.combine_rounds(method='mean')
    
    # Use the combined data
    data_loader.use_combined_data(True)
    
    # Print available components and metrics
    phases = list(data_loader.data.keys())
    if phases:
        first_phase = phases[0]
        print(f"\nComponents in {first_phase} (combined data):")
        components = data_loader.get_all_components(first_phase)
        for component in components:
            print(f"  - {component}")
            metrics = data_loader.get_all_metrics(first_phase, component)
            for metric in metrics[:5]:  # Print only first 5 metrics to avoid clutter
                print(f"    - {metric}")
            if len(metrics) > 5:
                print(f"    - ... and {len(metrics) - 5} more")
    
    # Switch back to individual round data
    data_loader.use_combined_data(False)
    print("\nSwitched to individual round data")
