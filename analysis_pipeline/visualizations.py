#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizations for Kubernetes Noisy Neighbours Lab
This module generates visualizations of metrics data.

Features:
- Time series visualizations
- Distribution plots
- Boxplot generation
- Phase comparison plots
- Tenant comparison with accessibility features
- Colorblind-friendly palettes and visual distinctions

The module includes a colorblind-friendly mode that uses:
- High-contrast color palettes selected for color vision deficiency compatibility
- Different line styles and markers to distinguish data series
- Texture patterns (hatches) to differentiate experiment phases
- Text labels with improved contrast and positioning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

class VisualizationGenerator:
    def __init__(self, output_dir=None, colorblind_friendly=True):
        """
        Initialize the visualization generator.
        
        Args:
            output_dir (str): Directory to save visualizations
            colorblind_friendly (bool): Whether to use colorblind friendly palettes (default: True)
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.colorblind_friendly = colorblind_friendly
        
        if self.output_dir and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        logging.info(f"Initialized VisualizationGenerator, output directory: {self.output_dir}")
        logging.info(f"Colorblind friendly mode: {self.colorblind_friendly}")
        
        # Set default style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 12  # Aumentar um pouco o tamanho da fonte
        
        # Configurações adicionais para melhor legibilidade
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        
        # Se modo amigável para daltônicos estiver ativado, use a paleta apropriada do seaborn
        if self.colorblind_friendly:
            # Define a paleta colorblind-friendly para todo o matplotlib
            sns.set_palette("colorblind")
            logging.info("Using colorblind-friendly color palette")
    
    def plot_time_series(self, data, title=None, y_label=None, filename=None, 
                         phase_name=None, component_name=None, metric_name=None):
        """
        Plot time series data.
        
        Args:
            data (DataFrame or Series): Data to plot
            title (str): Plot title
            y_label (str): Y-axis label
            filename (str): Filename to save plot
            phase_name (str): Name of the phase
            component_name (str): Name of the component
            metric_name (str): Name of the metric
            
        Returns:
            bool: True if successful, False otherwise
        """
        if data is None or (isinstance(data, pd.DataFrame) and data.empty) or (isinstance(data, pd.Series) and len(data) == 0):
            logging.warning("Empty data, cannot plot time series")
            return False
        
        # Generate directory for this phase if not specified
        if self.output_dir and phase_name:
            phase_dir = self.output_dir / phase_name
            if not phase_dir.exists():
                phase_dir.mkdir(parents=True, exist_ok=True)
        else:
            phase_dir = self.output_dir
        
        # Generate filename if not specified
        if not filename:
            if component_name and metric_name:
                clean_comp = component_name.replace('/', '_').replace(' ', '_').lower()
                clean_metric = metric_name.replace('/', '_').replace(' ', '_').lower()
                filename = f"{clean_comp}_{clean_metric}.png"
            else:
                filename = "time_series.png"
        
        plt.figure(figsize=(12, 6))
        
        if isinstance(data, pd.DataFrame):
            # Plot each column as a separate line
            for column in data.columns:
                if pd.api.types.is_numeric_dtype(data[column]):
                    plt.plot(data.index, data[column], label=column)
                    
            if len(data.columns) > 1:
                plt.legend()
        else:
            # Plot the series
            plt.plot(data.index, data)
        
        # Add annotations
        plt.title(title or f"{component_name} - {metric_name}")
        plt.xlabel("Time")
        plt.ylabel(y_label or metric_name or "Value")
        plt.grid(True)
        
        # Rotate x-axis labels if timestamps
        if isinstance(data.index, pd.DatetimeIndex):
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot if output directory is set
        if phase_dir:
            plt.savefig(phase_dir / filename, bbox_inches='tight', dpi=300)
            plt.close()
            return True
        else:
            plt.show()
            plt.close()
            return True
    
    def plot_distribution(self, data, title=None, x_label=None, filename=None,
                         phase_name=None, component_name=None, metric_name=None):
        """
        Plot distribution of data.
        
        Args:
            data (DataFrame or Series): Data to plot
            title (str): Plot title
            x_label (str): X-axis label
            filename (str): Filename to save plot
            phase_name (str): Name of the phase
            component_name (str): Name of the component
            metric_name (str): Name of the metric
            
        Returns:
            bool: True if successful, False otherwise
        """
        if data is None or (isinstance(data, pd.DataFrame) and data.empty) or (isinstance(data, pd.Series) and len(data) == 0):
            logging.warning("Empty data, cannot plot distribution")
            return False
        
        # Generate directory for this phase if not specified
        if self.output_dir and phase_name:
            phase_dir = self.output_dir / phase_name
            if not phase_dir.exists():
                phase_dir.mkdir(parents=True, exist_ok=True)
        else:
            phase_dir = self.output_dir
        
        # Generate filename if not specified
        if not filename:
            if component_name and metric_name:
                clean_comp = component_name.replace('/', '_').replace(' ', '_').lower()
                clean_metric = metric_name.replace('/', '_').replace(' ', '_').lower()
                filename = f"{clean_comp}_{clean_metric}_dist.png"
            else:
                filename = "distribution.png"
        
        plt.figure(figsize=(12, 6))
        
        # Plot distributions
        if isinstance(data, pd.DataFrame):
            # Create a 2x2 subplot for multiple distributions
            n_cols = len([col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])])
            if n_cols > 0:
                n_cols_sqrt = min(2, int(np.ceil(np.sqrt(n_cols))))
                n_rows = int(np.ceil(n_cols / n_cols_sqrt))
                
                fig, axes = plt.subplots(n_rows, n_cols_sqrt, figsize=(12, 8))
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
                
                col_idx = 0
                for column in data.columns:
                    if pd.api.types.is_numeric_dtype(data[column]) and col_idx < len(axes):
                        ax = axes[col_idx]
                        sns.histplot(data[column].dropna(), kde=True, ax=ax)
                        ax.set_title(column)
                        ax.set_xlabel(x_label or column)
                        col_idx += 1
                
                # Hide any unused subplots
                for i in range(col_idx, len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                fig.suptitle(title or f"{component_name} - {metric_name} Distribution", y=1.05)
            else:
                logging.warning("No numeric columns for distribution plot")
                plt.close()
                return False
        else:
            # Plot single series distribution
            sns.histplot(data.dropna(), kde=True)
            plt.title(title or f"{component_name} - {metric_name} Distribution")
            plt.xlabel(x_label or metric_name or "Value")
            plt.ylabel("Frequency")
            plt.tight_layout()
        
        # Save plot if output directory is set
        if phase_dir:
            plt.savefig(phase_dir / filename, bbox_inches='tight', dpi=300)
            plt.close()
            return True
        else:
            plt.show()
            plt.close()
            return True
    
    def plot_boxplot(self, data, title=None, y_label=None, filename=None,
                    phase_name=None, component_name=None, metric_name=None):
        """
        Plot boxplot of data.
        
        Args:
            data (DataFrame or Series): Data to plot
            title (str): Plot title
            y_label (str): Y-axis label
            filename (str): Filename to save plot
            phase_name (str): Name of the phase
            component_name (str): Name of the component
            metric_name (str): Name of the metric
            
        Returns:
            bool: True if successful, False otherwise
        """
        if data is None or (isinstance(data, pd.DataFrame) and data.empty) or (isinstance(data, pd.Series) and len(data) == 0):
            logging.warning("Empty data, cannot plot boxplot")
            return False
        
        # Generate directory for this phase if not specified
        if self.output_dir and phase_name:
            phase_dir = self.output_dir / phase_name
            if not phase_dir.exists():
                phase_dir.mkdir(parents=True, exist_ok=True)
        else:
            phase_dir = self.output_dir
        
        # Generate filename if not specified
        if not filename:
            if component_name and metric_name:
                clean_comp = component_name.replace('/', '_').replace(' ', '_').lower()
                clean_metric = metric_name.replace('/', '_').replace(' ', '_').lower()
                filename = f"{clean_comp}_{clean_metric}_boxplot.png"
            else:
                filename = "boxplot.png"
        
        plt.figure(figsize=(10, 6))
        
        if isinstance(data, pd.DataFrame):
            # Select only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                sns.boxplot(data=numeric_data)
                plt.title(title or f"{component_name} - {metric_name} Boxplot")
                plt.ylabel(y_label or "Value")
                plt.xticks(rotation=45)
            else:
                logging.warning("No numeric columns for boxplot")
                plt.close()
                return False
        else:
            # Plot single series boxplot
            sns.boxplot(y=data.dropna())
            plt.title(title or f"{component_name} - {metric_name} Boxplot")
            plt.ylabel(y_label or metric_name or "Value")
        
        plt.tight_layout()
        
        # Save plot if output directory is set
        if phase_dir:
            plt.savefig(phase_dir / filename, bbox_inches='tight', dpi=300)
            plt.close()
            return True
        else:
            plt.show()
            plt.close()
            return True
    
    def plot_phase_comparison(self, metrics_across_phases, metric_name=None, component_name=None,
                             title=None, y_label=None, filename=None, plot_type='line'):
        """
        Compare the same metric across different phases.
        
        Args:
            metrics_across_phases (dict): Dictionary with phase names as keys and DataFrames/Series as values
            metric_name (str): Name of the metric
            component_name (str): Name of the component
            title (str): Plot title
            y_label (str): Y-axis label
            filename (str): Filename to save plot
            plot_type (str): Type of plot ('line', 'box', or 'violin')
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not metrics_across_phases or len(metrics_across_phases) < 2:
            logging.warning("Not enough phases for comparison")
            return False
        
        # Generate directory for phase comparisons
        if self.output_dir:
            phase_comp_dir = self.output_dir / "comparacao_fases"
            if not phase_comp_dir.exists():
                phase_comp_dir.mkdir(parents=True, exist_ok=True)
        else:
            phase_comp_dir = self.output_dir
        
        # Generate filename if not specified
        if not filename:
            if component_name and metric_name:
                clean_comp = component_name.replace('/', '_').replace(' ', '_').lower()
                clean_metric = metric_name.replace('/', '_').replace(' ', '_').lower()
                filename = f"compare_{clean_comp}_{clean_metric}.png"
            else:
                filename = "phase_comparison.png"
        
        if plot_type == 'line':
            # Line plot
            plt.figure(figsize=(14, 8))
            
            for phase, data in metrics_across_phases.items():
                if isinstance(data, pd.DataFrame):
                    # Use first numeric column
                    num_cols = data.select_dtypes(include=[np.number]).columns
                    if len(num_cols) > 0:
                        # Reset index to create an array index for comparison
                        plot_data = data[num_cols[0]].reset_index(drop=True)
                        plt.plot(plot_data, label=phase)
                else:
                    # Reset index for series
                    plot_data = data.reset_index(drop=True)
                    plt.plot(plot_data, label=phase)
            
            plt.title(title or f"{component_name} - {metric_name} Across Phases")
            plt.xlabel("Time Point")
            plt.ylabel(y_label or metric_name or "Value")
            plt.legend()
            plt.grid(True)
            
        elif plot_type == 'box' or plot_type == 'violin':
            # Prepare data for box/violin plot
            plot_data = []
            labels = []
            
            for phase, data in metrics_across_phases.items():
                if isinstance(data, pd.DataFrame):
                    # Use first numeric column
                    num_cols = data.select_dtypes(include=[np.number]).columns
                    if len(num_cols) > 0:
                        values = data[num_cols[0]].dropna().values
                        plot_data.append(values)
                        labels.append(phase)
                else:
                    values = data.dropna().values
                    plot_data.append(values)
                    labels.append(phase)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            if plot_type == 'box':
                plt.boxplot(plot_data, labels=labels, vert=True, patch_artist=True)
            else:  # violin plot
                plt.violinplot(plot_data, showmeans=True, showmedians=True)
                plt.xticks(range(1, len(labels) + 1), labels)
            
            plt.title(title or f"{component_name} - {metric_name} Across Phases")
            plt.ylabel(y_label or metric_name or "Value")
            plt.grid(True, axis='y')
        
        plt.tight_layout()
        
        # Save plot if output directory is set
        if phase_comp_dir:
            plt.savefig(phase_comp_dir / filename, bbox_inches='tight', dpi=300)
            plt.close()
            return True
        else:
            plt.show()
            plt.close()
            return True
    
    def generate_component_visualizations(self, component_data, component_name, phase_name=None):
        """
        Generate visualizations for all metrics in a component.
        
        Args:
            component_data (dict): Dictionary with metric names as keys and dataframes as values
            component_name (str): Name of the component
            phase_name (str): Name of the phase
            
        Returns:
            int: Number of visualizations generated
        """
        count = 0
        
        for metric_name, metric_data in component_data.items():
            try:
                # Time series plot
                self.plot_time_series(metric_data, component_name=component_name,
                                     metric_name=metric_name, phase_name=phase_name)
                count += 1
                
                # Distribution plot
                self.plot_distribution(metric_data, component_name=component_name,
                                      metric_name=metric_name, phase_name=phase_name)
                count += 1
                
                # Boxplot
                self.plot_boxplot(metric_data, component_name=component_name,
                                 metric_name=metric_name, phase_name=phase_name)
                count += 1
                
            except Exception as e:
                logging.error(f"Error generating visualizations for {component_name}/{metric_name}: {e}")
                continue
        
        return count
    
    def generate_phase_visualizations(self, phase_data, phase_name):
        """
        Generate visualizations for all components in a phase.
        
        Args:
            phase_data (dict): Dictionary with component names as keys and data dictionaries as values
            phase_name (str): Name of the phase
            
        Returns:
            int: Number of visualizations generated
        """
        count = 0
        
        for component_name, component_data in phase_data.items():
            component_count = self.generate_component_visualizations(
                component_data, component_name, phase_name
            )
            count += component_count
            
            logging.info(f"Generated {component_count} visualizations for {phase_name}/{component_name}")
        
        return count
    
    def generate_phase_comparison_visualizations(self, data_loader, components, metrics):
        """
        Generate comparison visualizations across phases.
        
        Args:
            data_loader (DataLoader): Data loader instance
            components (list): List of component names to include
            metrics (list): List of metric names to include
            
        Returns:
            int: Number of visualizations generated
        """
        count = 0
        
        for component in components:
            for metric in metrics:
                try:
                    # Get the same metric across all phases
                    metrics_across_phases = data_loader.get_same_metric_across_phases(component, metric)
                    
                    if len(metrics_across_phases) >= 2:
                        # Line plot comparison
                        self.plot_phase_comparison(
                            metrics_across_phases, metric_name=metric, component_name=component,
                            plot_type='line', filename=f"compare_{component}_{metric}_line.png"
                        )
                        count += 1
                        
                        # Box plot comparison
                        self.plot_phase_comparison(
                            metrics_across_phases, metric_name=metric, component_name=component,
                            plot_type='box', filename=f"compare_{component}_{metric}_box.png"
                        )
                        count += 1
                        
                        # Violin plot comparison
                        self.plot_phase_comparison(
                            metrics_across_phases, metric_name=metric, component_name=component,
                            plot_type='violin', filename=f"compare_{component}_{metric}_violin.png"
                        )
                        count += 1
                except Exception as e:
                    logging.error(f"Error generating comparison for {component}/{metric}: {e}")
                    continue
        
        return count
    
    def generate_tenant_comparison_plots(self, data, tenants, metrics_list=None, output_subdir='tenant_comparison'):
        """
        Generate comparative plots for tenants metrics across experiment phases.
        
        Args:
            data (dict): Dictionary with phase data from DataLoader.load_all_phases()
            tenants (list): List of tenant names to include in comparison
            metrics_list (list, optional): List of metrics to plot as (name, display_title) tuples
            output_subdir (str): Subdirectory under output_dir to save plots
            
        Returns:
            list: Paths to generated plot files
        """
        if not metrics_list:
            # Default metrics to plot
            metrics_list = [
                ("cpu_usage", "CPU Usage (%)"),
                ("memory_usage", "Memory Usage (%)"),
                ("disk_io_total", "Disk I/O Operations"),
                ("network_total_bandwidth", "Network Bandwidth (bytes)")
            ]
        
        # Create output directory
        output_dir = self.output_dir / output_subdir if self.output_dir else Path("./results/tenant_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Generating tenant comparison plots in directory: {output_dir}")
        
        # Log the phases and tenants that were found
        phases = list(data.keys())
        logging.info(f"Processing {len(phases)} phases: {phases}")
        
        for phase, phase_data in data.items():
            components = list(phase_data.keys())
            logging.info(f"Phase {phase} has {len(components)} components: {components}")
        
        # Track generated plot paths
        generated_plots = []
        
        # Generate plots for each metric
        for metric_name, metric_title in metrics_list:
            logging.info(f"Generating comparison plot for {metric_name}")
            plot_path = str(output_dir / f"{metric_name}_comparison.png")
            
            self._plot_tenant_comparison(
                data=data,
                tenants=tenants,
                metric_name=metric_name,
                output_path=plot_path,
                title=f"{metric_title} Comparison Across Tenants and Phases"
            )
            
            generated_plots.append(plot_path)
        
        logging.info(f"All tenant comparison plots saved to {output_dir}")
        return generated_plots
    
    def _merge_phase_data(self, data, tenants, metric_name):
        """
        Merge data from different phases for comparison and convert to elapsed time
        """
        merged_data = {}
        all_timestamps = []
        
        # First pass: collect all timestamps to determine experiment start time
        for tenant in tenants:
            for phase_name, phase_data in data.items():
                if tenant in phase_data and metric_name in phase_data[tenant]:
                    phase_df = phase_data[tenant][metric_name].copy()
                    if not phase_df.empty and isinstance(phase_df.index, pd.DatetimeIndex):
                        all_timestamps.extend(phase_df.index.tolist())
        
        # Determine experiment start time
        if all_timestamps:
            experiment_start = min(all_timestamps)
            logging.info(f"Experiment start time: {experiment_start}")
        else:
            experiment_start = None
            logging.warning("Could not determine experiment start time")
        
        # Second pass: process data with normalized time
        for tenant in tenants:
            tenant_data = pd.DataFrame()
            
            # Process each phase
            for phase_name, phase_data in data.items():
                if tenant in phase_data and metric_name in phase_data[tenant]:
                    # Get phase data for this tenant and metric
                    phase_df = phase_data[tenant][metric_name].copy()
                    
                    # Convert index to datetime if it's not already
                    if not isinstance(phase_df.index, pd.DatetimeIndex):
                        try:
                            phase_df.index = pd.to_datetime(phase_df.index)
                        except:
                            # Keep numeric index if conversion fails
                            pass
                    
                    # If phase dataframe is empty or has no numeric columns, skip
                    if phase_df.empty or not any(pd.api.types.is_numeric_dtype(phase_df[col]) 
                                              for col in phase_df.columns):
                        logging.warning(f"No usable data for {tenant}/{metric_name} in phase {phase_name}")
                        continue
                    
                    # Convert datetime index to elapsed time in seconds if we have experiment start
                    if isinstance(phase_df.index, pd.DatetimeIndex) and experiment_start is not None:
                        # Calculate seconds since start
                        elapsed_seconds = [(ts - experiment_start).total_seconds() for ts in phase_df.index]
                        
                        # Create a new DataFrame with elapsed seconds as index
                        new_df = phase_df.copy()
                        new_df['elapsed_seconds'] = elapsed_seconds
                        new_df = new_df.reset_index()
                        new_df = new_df.set_index('elapsed_seconds')
                        phase_df = new_df
                    
                    # Add phase label column
                    phase_df['phase'] = phase_name
                    
                    # Identify the value column (first numeric column)
                    value_col = next((col for col in phase_df.columns 
                                    if pd.api.types.is_numeric_dtype(phase_df[col]) and 
                                    col not in ['elapsed_seconds', 'index']), None)
                    
                    if value_col:
                        # Rename to 'value' for consistency
                        if value_col != 'value':
                            phase_df['value'] = phase_df[value_col]
                    
                    # Append to tenant data
                    tenant_data = pd.concat([tenant_data, phase_df])
                else:
                    logging.warning(f"No data for {tenant}/{metric_name} in phase {phase_name}")
            
            # Store the merged data for this tenant
            merged_data[tenant] = tenant_data
        
        return merged_data
    
    def _get_phase_boundaries(self, data):
        """
        Get elapsed time points for phase boundaries and deduplicate them
        """
        # Format: (timestamp, source_phase, target_phase)
        raw_boundaries = []
        phase_transitions = set()  # To track unique transitions
        
        # Extract all phase changes
        for tenant, tenant_data in data.items():
            if tenant_data.empty:
                continue
            
            rows = list(tenant_data.iterrows())
            current_phase = None
            
            for i, (timestamp, row) in enumerate(rows):
                if 'phase' in row:
                    phase = row['phase']
                    if phase != current_phase and current_phase is not None:
                        # Record phase transition
                        raw_boundaries.append((timestamp, current_phase, phase))
                    current_phase = phase
        
        # Sort boundaries by timestamp and deduplicate based on transitions
        # (This ensures we only have one boundary per phase change)
        seen_transitions = set()
        boundaries = []
        
        for timestamp, source, target in sorted(raw_boundaries, key=lambda x: x[0]):
            transition = (source, target)
            if transition not in seen_transitions:
                seen_transitions.add(transition)
                boundaries.append((timestamp, source, target))
        
        return boundaries
    
    def _plot_tenant_comparison(self, data, tenants, metric_name, output_path, title=None, ylim=None):
        """
        Create a plot comparing metrics across tenants with elapsed time on x-axis
        """
        plt.figure(figsize=(14, 8))
        
        # Prepare merged data with elapsed time
        merged_data = self._merge_phase_data(data, tenants, metric_name)
        
        # Color map for tenants - using colorblind friendly palette
        # Baseado na paleta 'IBM' do ColorBrewer - amigável para daltônicos
        tenant_colors = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000', '#000000']
        
        # Definir diferentes estilos de linha para melhor diferenciação
        line_styles = ['-', '--', '-.', ':', '-', '--']
        marker_styles = ['o', 's', '^', 'D', 'x', '+']
        
        # Plot data for each tenant
        for i, tenant in enumerate(tenants):
            if tenant not in merged_data or merged_data[tenant].empty:
                logging.warning(f"No data for tenant {tenant}")
                continue
                
            tenant_data = merged_data[tenant]
            
            # Plot the metric value com marcadores espaçados para melhor identificação
            if 'value' in tenant_data.columns:
                plt.plot(tenant_data.index, tenant_data['value'], 
                        color=tenant_colors[i % len(tenant_colors)], 
                        linestyle=line_styles[i % len(line_styles)],
                        marker=marker_styles[i % len(marker_styles)],
                        markevery=int(len(tenant_data) / 10) if len(tenant_data) > 10 else None,  # Adiciona marcadores espaçados
                        markersize=6,
                        label=f"{tenant}", 
                        linewidth=2)
        
        # Get unique phase boundaries (deduplicated)
        data_boundaries = self._get_phase_boundaries(merged_data)
        
        # Add background colors for phases - usando cores com padrões distintos para daltônicos
        phase_colors = {
            '1 - Baseline': '#E6F2FF',  # Azul bem claro com padrão de listras
            '2 - Attack': '#FFE6E6',    # Vermelho bem claro com padrão pontilhado
            '3 - Recovery': '#E6FFE6'   # Verde bem claro com padrão quadriculado
        }
        
        # Adicionamos também padrões (hatches) para distinguir as fases através da textura
        phase_hatches = {
            '1 - Baseline': '/',      # Listras diagonais
            '2 - Attack': '.',        # Pontos
            '3 - Recovery': 'x'       # Padrão xadrez
        }
        
        # If we have phase data, add background shading and boundary lines
        if data_boundaries:
            # First, determine the start and end of experiment
            all_indices = []
            for tenant_data in merged_data.values():
                if not tenant_data.empty:
                    all_indices.extend(tenant_data.index.tolist())
            
            if all_indices:
                experiment_start = min(all_indices)
                experiment_end = max(all_indices)
                
                # Get first phase
                first_phase = data_boundaries[0][1]  # Source phase of first boundary
                
                # Add shading for first phase from start to first boundary
                plt.axvspan(experiment_start, data_boundaries[0][0], 
                          alpha=0.3, color=phase_colors.get(first_phase, 'white'),
                          hatch=phase_hatches.get(first_phase, ''))
                
                # Add boundary lines and shading for the rest
                for i, (timestamp, source_phase, target_phase) in enumerate(data_boundaries):
                    # Add boundary line - using a darker color for better visibility
                    plt.axvline(x=timestamp, color='#333333', linestyle='-', linewidth=1.5, alpha=0.8)
                    
                    # Add label for phase transition - better positioning and contrast
                    y_pos = plt.ylim()[1] * 0.95
                    plt.text(timestamp + 0.5, y_pos, f"{source_phase} → {target_phase}",
                            rotation=90, verticalalignment='top', horizontalalignment='left',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
                    
                    # Add shading for next phase - with hatch patterns for better distinction
                    if i < len(data_boundaries) - 1:  # If not the last boundary
                        next_timestamp = data_boundaries[i+1][0]
                        plt.axvspan(timestamp, next_timestamp, 
                                  alpha=0.3, color=phase_colors.get(target_phase, 'white'),
                                  hatch=phase_hatches.get(target_phase, ''))
                    else:  # Last boundary - shade to the end
                        plt.axvspan(timestamp, experiment_end, 
                                  alpha=0.3, color=phase_colors.get(target_phase, 'white'),
                                  hatch=phase_hatches.get(target_phase, ''))
        
        # Set plot title and labels with improved contrast and readability
        plt.title(title or f"Comparison of {metric_name} across Tenants", fontsize=14, fontweight='bold')
        plt.xlabel("Time Elapsed (seconds)", fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Adiciona legenda com maior contraste e informações sobre as fases
        plt.legend(loc='best', framealpha=0.95, edgecolor='#333333')
        
        # Adiciona uma legenda separada para as fases na parte superior
        from matplotlib.patches import Patch
        from matplotlib.legend import Legend
        
        # Primeiro recuperamos a legenda dos tenants para não sobrescrevê-la
        tenant_legend = plt.legend(loc='best', framealpha=0.95, edgecolor='#333333')
        
        # Adiciona a legenda dos tenants ao plot atual
        plt.gca().add_artist(tenant_legend)
        
        # Cria a legenda para as fases
        phase_legend_elements = [
            Patch(facecolor=phase_colors['1 - Baseline'], hatch=phase_hatches['1 - Baseline'], 
                  edgecolor='#333333', alpha=0.3, label='Baseline Phase'),
            Patch(facecolor=phase_colors['2 - Attack'], hatch=phase_hatches['2 - Attack'], 
                  edgecolor='#333333', alpha=0.3, label='Attack Phase'),
            Patch(facecolor=phase_colors['3 - Recovery'], hatch=phase_hatches['3 - Recovery'], 
                  edgecolor='#333333', alpha=0.3, label='Recovery Phase')
        ]
        
        # Adiciona a segunda legenda para as fases em uma posição diferente
        plt.legend(handles=phase_legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, 1.15), ncol=3, framealpha=0.95)
        
        # Set y-axis limits if provided
        if ylim:
            plt.ylim(ylim)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved tenant comparison plot to {output_path}")
        plt.close()


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
    
    # Initialize visualization generator
    vis_generator = VisualizationGenerator(output_dir)
    
    # Example: Generate visualizations for tenant-a in baseline phase
    baseline_phase = "1 - Baseline"
    if baseline_phase in data and 'tenant-a' in data[baseline_phase]:
        count = vis_generator.generate_component_visualizations(
            data[baseline_phase]['tenant-a'], component_name="tenant-a", phase_name=baseline_phase
        )
        print(f"\nGenerated {count} visualizations for {baseline_phase}/tenant-a")
        
        # Generate phase comparison for CPU usage
        if len(data) >= 2:
            metrics_across_phases = data_loader.get_same_metric_across_phases('tenant-a', 'cpu_usage')
            
            if len(metrics_across_phases) >= 2:
                vis_generator.plot_phase_comparison(
                    metrics_across_phases, metric_name='cpu_usage', component_name='tenant-a',
                    plot_type='line'
                )
                print("\nGenerated phase comparison for tenant-a/cpu_usage")
