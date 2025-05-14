#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tenant Degradation Analysis Module for Kubernetes Noisy Neighbors Lab
This module identifies direct sources of service degradation through cross-tenant analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import networkx as nx
from scipy import stats
from time_series_analysis import TimeSeriesAnalyzer
from correlation_analysis import CorrelationAnalyzer
from data_loader import DataLoader

class TenantDegradationAnalyzer:
    """Analyzes relationships between tenants to identify sources of service degradation."""
    
    def __init__(self, output_dir=None):
        """
        Initialize the tenant degradation analyzer.
        
        Args:
            output_dir (str): Directory to save results and plots
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectory for degradation analysis
        self.plots_dir = self.output_dir / "tenant_degradation" if self.output_dir else None
        if self.plots_dir and not self.plots_dir.exists():
            self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Results directory for CSV and report files
        self.results_dir = self.output_dir / "tenant_degradation_results" if self.output_dir else None
        if self.results_dir and not self.results_dir.exists():
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Time series analysis directory
        self.ts_dir = self.output_dir / "time_series_analysis" if self.output_dir else None
        if self.ts_dir and not self.ts_dir.exists():
            self.ts_dir.mkdir(parents=True, exist_ok=True)
            
        # Correlation analysis directory
        self.corr_dir = self.output_dir / "correlations" if self.output_dir else None
        if self.corr_dir and not self.corr_dir.exists():
            self.corr_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize helper analyzers with appropriate output directories
        self.ts_analyzer = TimeSeriesAnalyzer(self.ts_dir)
        self.corr_analyzer = CorrelationAnalyzer(self.corr_dir)
            
        logging.info(f"Initialized TenantDegradationAnalyzer, output directory: {self.output_dir}")
    
    def align_metrics_across_tenants(self, data, phase, metric_name, tenants):
        """
        Aligns a specific metric across multiple tenants for a particular phase.
        
        Args:
            data (dict): Data dictionary from DataLoader with phase data
            phase (str): Phase name
            metric_name (str): Metric name to analyze
            tenants (list): List of tenants to analyze
            
        Returns:
            DataFrame: DataFrame with aligned metrics
        """
        aligned_data = {}
        
        for tenant in tenants:
            if tenant in data[phase] and metric_name in data[phase][tenant]:
                tenant_data = data[phase][tenant][metric_name]
                
                # Extract the value column (first numeric column)
                value_col = next((col for col in tenant_data.columns 
                                if pd.api.types.is_numeric_dtype(tenant_data[col])), None)
                
                if value_col:
                    aligned_data[f"{tenant}"] = tenant_data[value_col]
                else:
                    logging.warning(f"No numeric column found for {tenant}/{metric_name}")
            else:
                logging.warning(f"No data for {tenant}/{metric_name} in phase {phase}")
        
        if not aligned_data:
            return None
            
        # Create DataFrame with aligned data
        try:
            aligned_df = pd.DataFrame(aligned_data)
            return aligned_df
        except Exception as e:
            logging.error(f"Error aligning tenant metrics: {e}")
            return None
    
    def analyze_cross_tenant_correlations(self, data, phase, metrics_of_interest, tenants):
        """
        Analyze correlations between metrics across tenants for a specific phase.
        
        Args:
            data (dict): Data dictionary from DataLoader with phase data
            phase (str): Phase name
            metrics_of_interest (list): List of metrics to analyze
            tenants (list): List of tenants to analyze
            
        Returns:
            dict: Dictionary of correlation matrices by metric
        """
        correlation_results = {}
        
        for metric in metrics_of_interest:
            logging.info(f"Analyzing cross-tenant correlations for {metric} in phase {phase}")
            
            # Align metrics across tenants
            aligned_df = self.align_metrics_across_tenants(data, phase, metric, tenants)
            
            if aligned_df is None or aligned_df.empty:
                logging.warning(f"Insufficient data for correlation analysis of {metric}")
                continue
            
            # Calculate correlation matrix
            corr_matrix = aligned_df.corr(method='pearson')
            correlation_results[metric] = corr_matrix
            
            # Plot correlation matrix 
            # Save in the tenant_degradation directory for primary visualizations
            if self.plots_dir:
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                           center=0, square=True, fmt='.2f', cbar_kws={'shrink': .8})
                
                plt.title(f'Cross-Tenant Correlation: {metric} ({phase})')
                plt.tight_layout()
                
                # Save plot to tenant_degradation folder
                filename = f"cross_tenant_corr_{phase}_{metric}.png".replace(' ', '_').lower()
                plt.savefig(self.plots_dir / filename, bbox_inches='tight', dpi=300)
                
                # Also save to correlations folder
                if self.corr_dir:
                    plt.savefig(self.corr_dir / filename, bbox_inches='tight', dpi=300)
                
                plt.close()
            
            # Save correlation matrix to CSV
            if self.results_dir:
                csv_file = f"cross_tenant_corr_{phase}_{metric}.csv".replace(' ', '_').lower()
                corr_matrix.to_csv(self.results_dir / csv_file)
                
                # Also save to correlations folder as CSV
                if self.corr_dir:
                    corr_matrix.to_csv(self.corr_dir / csv_file)
        
        return correlation_results
    
    def analyze_granger_causality(self, data, phase, metrics_of_interest, tenants, max_lag=5):
        """
        Analyze Granger causality between tenants for a specific phase and metric.
        
        Args:
            data (dict): Data dictionary from DataLoader with phase data
            phase (str): Phase name
            metrics_of_interest (list): List of metrics to analyze
            tenants (list): List of tenants to analyze
            max_lag (int): Maximum lag to consider for Granger causality
            
        Returns:
            dict: Dictionary of causality results by metric
        """
        causality_results = {}
        
        for metric in metrics_of_interest:
            logging.info(f"Analyzing Granger causality for {metric} in phase {phase}")
            
            # Align metrics across tenants
            aligned_df = self.align_metrics_across_tenants(data, phase, metric, tenants)
            
            if aligned_df is None or aligned_df.empty or aligned_df.shape[1] < 2:
                logging.warning(f"Insufficient data for Granger causality analysis of {metric}")
                continue
            
            # Create causality matrix (directed)
            tenant_names = aligned_df.columns.tolist()
            causality_matrix = pd.DataFrame(0, index=tenant_names, columns=tenant_names)
            p_value_matrix = pd.DataFrame(1.0, index=tenant_names, columns=tenant_names)
            
            # Test causality between each pair of tenants
            causality_pairs = []
            for i, tenant1 in enumerate(tenant_names):
                for j, tenant2 in enumerate(tenant_names):
                    if i == j:
                        continue
                    
                    # Get the two series
                    series1 = aligned_df[tenant1].dropna()
                    series2 = aligned_df[tenant2].dropna()
                    
                    # Skip if either series is too short
                    if len(series1) <= max_lag + 1 or len(series2) <= max_lag + 1:
                        logging.warning(f"Series too short for Granger causality test: {tenant1} → {tenant2}")
                        continue
                    
                    # Test if tenant1 Granger-causes tenant2
                    try:
                        # Set up custom filename for time series analysis results
                        ts_filename = f"granger_causality_{phase}_{tenant1}_to_{tenant2}_{metric}.png".replace(' ', '_').lower()
                        
                        # Use the time series analyzer with specific output file
                        result = self.ts_analyzer.granger_causality(
                            series1, series2, maxlag=max_lag,
                            series1_name=tenant1, series2_name=tenant2
                        )
                        
                        # If we have the time_series_analysis directory, save results there too
                        if result and self.ts_dir:
                            # Save lag analysis plot in time series directory
                            # This code creates a simple lag plot visualization
                            if 'granger_1_to_2' in result and result['granger_1_to_2']['significant']:
                                plt.figure(figsize=(10, 6))
                                lags = list(range(1, max_lag + 1))
                                p_values = [result['granger_1_to_2']['p_values'].get(lag, 1.0) for lag in lags]
                                plt.plot(lags, p_values, marker='o')
                                plt.axhline(y=0.05, color='r', linestyle='--', label='Significance level (0.05)')
                                plt.xlabel('Lag')
                                plt.ylabel('p-value')
                                plt.title(f'Granger Causality: {tenant1} → {tenant2} ({metric}, {phase})')
                                plt.ylim([0, 1])
                                plt.legend()
                                plt.grid(True)
                                plt.savefig(self.ts_dir / ts_filename, bbox_inches='tight', dpi=300)
                                plt.close()
                        
                        if result and 'granger_1_to_2' in result and result['granger_1_to_2']['significant']:
                            # Store result: tenant1 causes tenant2
                            causality_matrix.at[tenant1, tenant2] = result['granger_1_to_2']['min_p_value']
                            p_value_matrix.at[tenant1, tenant2] = result['granger_1_to_2']['min_p_value']
                            
                            # Add to pairs list for summary
                            causality_pairs.append({
                                'source': tenant1,
                                'target': tenant2,
                                'p_value': result['granger_1_to_2']['min_p_value'],
                                'lag': result['granger_1_to_2']['min_lag'],
                                'significant': True
                            })
                    except Exception as e:
                        logging.error(f"Error in Granger causality test {tenant1} → {tenant2}: {e}")
            
            # Store results
            causality_results[metric] = {
                'matrix': causality_matrix,
                'p_values': p_value_matrix,
                'significant_pairs': causality_pairs
            }
            
            # Plot causality network to primary visualization folder
            if self.plots_dir and causality_pairs:
                self.plot_causality_network(
                    causality_pairs, 
                    title=f"Granger Causality Network: {metric} ({phase})",
                    filename=f"causality_network_{phase}_{metric}.png"
                )
                
                # Also save causality matrix to time series folder
                if self.ts_dir:
                    if not causality_matrix.empty:
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(causality_matrix, annot=True, cmap='YlOrRd_r', vmin=0, vmax=0.05, 
                                   square=True, fmt='.4f', cbar_kws={'shrink': .8})
                        plt.title(f'Granger Causality p-values: {metric} ({phase})')
                        plt.tight_layout()
                        
                        filename = f"granger_causality_matrix_{phase}_{metric}.png".replace(' ', '_').lower()
                        plt.savefig(self.ts_dir / filename, bbox_inches='tight', dpi=300)
                        plt.close()
            
            # Save causality results to CSV
            if self.results_dir and causality_pairs:
                pairs_df = pd.DataFrame(causality_pairs)
                csv_file = f"granger_causality_{phase}_{metric}.csv".replace(' ', '_').lower()
                pairs_df.to_csv(self.results_dir / csv_file, index=False)
                
                # Also save to time series folder
                if self.ts_dir:
                    pairs_df.to_csv(self.ts_dir / csv_file, index=False)
                    
                    # Save the causality matrix as CSV
                    causality_matrix.to_csv(self.ts_dir / f"granger_causality_matrix_{phase}_{metric}.csv".replace(' ', '_').lower())
        
        return causality_results

    def plot_causality_network(self, causality_pairs, title=None, filename=None):
        """
        Create a network visualization of causality between tenants.
        
        Args:
            causality_pairs (list): List of dictionaries with causality results
            title (str): Title for the plot
            filename (str): Filename to save the plot
        """
        if not causality_pairs:
            return
            
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for pair in causality_pairs:
            source = pair['source']
            target = pair['target']
            p_value = pair['p_value']
            
            # Add nodes if they don't exist
            if source not in G:
                G.add_node(source)
            if target not in G:
                G.add_node(target)
            
            # Add edge with weight based on p-value (lower p-value = stronger causality)
            weight = 1 - p_value  # Transform p-value to weight
            G.add_edge(source, target, weight=weight, p_value=p_value)
        
        # Only proceed if we have edges
        if not G.edges:
            logging.warning("No significant causal relationships to visualize")
            return
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=0.8, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue', alpha=0.8)
        
        # Draw edges with width proportional to causality strength
        edges = G.edges(data=True)
        edge_widths = [3 * (1 - d['p_value']) for _, _, d in edges]
        nx.draw_networkx_edges(G, pos, arrowsize=20, width=edge_widths, alpha=0.7)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Add edge labels (p-values)
        edge_labels = {(u, v): f"{d['p_value']:.3f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        
        plt.title(title or "Tenant Causality Network")
        plt.axis('off')
        plt.tight_layout()
        
        # Save plot to main visualization directory
        if filename and self.plots_dir:
            plt.savefig(self.plots_dir / filename, bbox_inches='tight', dpi=300)
            
            # Also save to time series directory
            if self.ts_dir:
                plt.savefig(self.ts_dir / filename, bbox_inches='tight', dpi=300)
            
            plt.close()
        else:
            plt.show()
            plt.close()
            
    def identify_degradation_sources(self, data, phases, metrics_of_interest, tenants):
        """
        Identify sources of service degradation across phases.
        
        Args:
            data (dict): Data dictionary from DataLoader with phase data
            phases (list): List of phases to analyze
            metrics_of_interest (list): List of metrics to analyze
            tenants (list): List of tenants to analyze
            
        Returns:
            dict: Dictionary with degradation sources by phase and metric
        """
        degradation_results = {}
        
        # First, check if we have at least baseline and attack phases
        if len(phases) < 2 or '1 - Baseline' not in phases or '2 - Attack' not in phases:
            logging.error("Need at least baseline and attack phases for degradation analysis")
            return {}
            
        # Define phases we're interested in
        baseline_phase = '1 - Baseline'
        attack_phase = '2 - Attack'
        recovery_phase = '3 - Recovery' if '3 - Recovery' in phases else None
        
        # For each metric, identify potential sources of degradation
        for metric in metrics_of_interest:
            degradation_results[metric] = {}
            
            # Step 1: Measure baseline correlation between tenants
            baseline_corr = self.analyze_cross_tenant_correlations(
                data, baseline_phase, [metric], tenants
            )
            
            # Step 2: Measure attack phase correlation between tenants  
            attack_corr = self.analyze_cross_tenant_correlations(
                data, attack_phase, [metric], tenants
            )
            
            # Step 3: Analyze Granger causality in the attack phase
            attack_causality = self.analyze_granger_causality(
                data, attack_phase, [metric], tenants
            )
            
            # Step 4: Identify metrics that changed significantly from baseline to attack
            if metric in baseline_corr and metric in attack_corr:
                baseline_matrix = baseline_corr[metric]
                attack_matrix = attack_corr[metric]
                
                # Calculate difference in correlation patterns
                if not baseline_matrix.empty and not attack_matrix.empty:
                    # Ensure matrices have same dimensions
                    common_tenants = list(set(baseline_matrix.index).intersection(set(attack_matrix.index)))
                    
                    if common_tenants:
                        baseline_filtered = baseline_matrix.loc[common_tenants, common_tenants]
                        attack_filtered = attack_matrix.loc[common_tenants, common_tenants]
                        
                        # Calculate correlation difference (how much relationships changed)
                        corr_diff = attack_filtered - baseline_filtered
                        
                        # Calculate correlation changes
                        degradation_results[metric]['correlation_change'] = corr_diff
                        
                        # Identify significant correlation changes
                        sig_changes = []
                        for i, tenant1 in enumerate(common_tenants):
                            for j, tenant2 in enumerate(common_tenants):
                                if i >= j:  # Skip diagonal and lower triangle
                                    continue
                                    
                                baseline_val = baseline_filtered.at[tenant1, tenant2]
                                attack_val = attack_filtered.at[tenant1, tenant2]
                                change = attack_val - baseline_val
                                
                                # Consider significant if abs change > 0.3
                                if abs(change) > 0.3:
                                    sig_changes.append({
                                        'tenant1': tenant1,
                                        'tenant2': tenant2,
                                        'baseline_corr': baseline_val,
                                        'attack_corr': attack_val,
                                        'change': change
                                    })
                        
                        degradation_results[metric]['significant_correlation_changes'] = sig_changes
            
            # Step 5: Combine with causality analysis
            if metric in attack_causality and attack_causality[metric]['significant_pairs']:
                degradation_results[metric]['causality_pairs'] = attack_causality[metric]['significant_pairs']
                
                # Identify strong candidates for degradation sources
                degradation_sources = []
                
                # Find tenants that cause many others
                tenant_counts = {}
                for pair in attack_causality[metric]['significant_pairs']:
                    source = pair['source']
                    tenant_counts[source] = tenant_counts.get(source, 0) + 1
                
                # Rank by number of targets influenced
                ranked_sources = sorted(tenant_counts.items(), key=lambda x: x[1], reverse=True)
                
                for tenant, count in ranked_sources:
                    if count >= 2:  # If tenant causes at least 2 others
                        degradation_sources.append({
                            'tenant': tenant,
                            'impact_count': count,
                            'evidence': 'Granger-causes multiple other tenants'
                        })
                
                degradation_results[metric]['likely_degradation_sources'] = degradation_sources
                
                # Generate degradation source report
                self.generate_degradation_report(degradation_results[metric], metric)
        
        return degradation_results
        
    def generate_degradation_report(self, results, metric_name):
        """
        Generate a human-readable report of degradation sources.
        
        Args:
            results (dict): Results dictionary for a specific metric
            metric_name (str): Name of the metric
        """
        if not self.results_dir:
            return
            
        # Create report file in main results directory
        report_file = self.results_dir / f"degradation_report_{metric_name}.txt".replace(' ', '_').lower()
        
        # Create report content
        report_content = [f"# Degradation Analysis Report: {metric_name}\n\n"]
            
        # Add likely degradation sources
        if 'likely_degradation_sources' in results and results['likely_degradation_sources']:
            report_content.append("## Likely Sources of Service Degradation\n\n")
            
            for source in results['likely_degradation_sources']:
                report_content.append(f"* **{source['tenant']}** - Impacts {source['impact_count']} other tenants\n")
                report_content.append(f"  - Evidence: {source['evidence']}\n\n")
        else:
            report_content.append("## No clear degradation sources identified\n\n")
        
        # Add significant correlation changes
        if 'significant_correlation_changes' in results and results['significant_correlation_changes']:
            report_content.append("## Significant Relationship Changes\n\n")
            
            for change in results['significant_correlation_changes']:
                direction = "increased" if change['change'] > 0 else "decreased"
                report_content.append(f"* Correlation between **{change['tenant1']}** and **{change['tenant2']}** {direction} by {abs(change['change']):.2f}\n")
                report_content.append(f"  - Baseline: {change['baseline_corr']:.2f}, Attack: {change['attack_corr']:.2f}\n\n")
        
        # Add causality evidence
        if 'causality_pairs' in results and results['causality_pairs']:
            report_content.append("## Causal Relationships Detected\n\n")
            
            for pair in results['causality_pairs']:
                report_content.append(f"* **{pair['source']}** → **{pair['target']}**  (p-value: {pair['p_value']:.4f}, lag: {pair['lag']})\n")
            
            report_content.append("\nNote: Arrows indicate the direction of causality (X → Y means X likely causes changes in Y)\n\n")
        
        # Add recommendation section
        report_content.append("\n## Recommendation\n\n")
        if 'likely_degradation_sources' in results and results['likely_degradation_sources']:
            sources = [s['tenant'] for s in results['likely_degradation_sources']]
            report_content.append(f"Based on the analysis, the most likely source(s) of degradation for {metric_name} are: {', '.join(sources)}.\n")
            report_content.append("Consider limiting resources for these tenants or isolating them to prevent impacts on other services.\n")
        else:
            report_content.append("No clear single source of degradation was identified. The issues may be systemic or related to overall resource constraints rather than a specific noisy neighbor.\n")
        
        # Join content into full report text
        report_text = "".join(report_content)
        
        # Write to main results directory
        with open(report_file, 'w') as f:
            f.write(report_text)
            
        # Also write to correlations directory if it exists
        if self.corr_dir:
            with open(self.corr_dir / f"degradation_report_{metric_name}.txt".replace(' ', '_').lower(), 'w') as f:
                f.write(report_text)
                
        # Also write to time series directory if it exists
        if self.ts_dir:
            with open(self.ts_dir / f"degradation_report_{metric_name}.txt".replace(' ', '_').lower(), 'w') as f:
                f.write(report_text)
        
        logging.info(f"Generated degradation report for {metric_name} at {report_file}")

def analyze_tenant_degradation(data_loader, output_dir):
    """
    Run tenant degradation analysis.
    
    Args:
        data_loader (DataLoader): DataLoader with loaded data
        output_dir (Path): Output directory for results
    """
    try:
        # Initialize analyzer
        analyzer = TenantDegradationAnalyzer(output_dir)
        
        # Get data and phases
        data = data_loader.data
        phases = list(data.keys())
        
        # Find tenants in the data
        tenants = []
        for phase_name, phase_data in data.items():
            for component in phase_data.keys():
                if component.startswith("tenant-"):
                    if component not in tenants:
                        tenants.append(component)
        
        # Define metrics of interest
        metrics_of_interest = [
            "cpu_usage",
            "memory_usage", 
            "network_total_bandwidth",
            "disk_io_total"
        ]
        
        # Run degradation analysis
        results = analyzer.identify_degradation_sources(data, phases, metrics_of_interest, tenants)
        
        return results
        
    except Exception as e:
        logging.error(f"Error in tenant degradation analysis: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    # When run as a script, load default experiment
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Run tenant degradation analysis')
    parser.add_argument('--experiment', type=str, default="2025-05-11/16-58-00/default-experiment-1",
                      help='Path to experiment relative to base path')
    parser.add_argument('--round', type=str, default="round-1",
                      help='Round number to analyze')
    parser.add_argument('--output', type=str, default=None,
                      help='Output directory for results')
                      
    args = parser.parse_args()
    
    # Set up paths
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    if not args.output:
        output_path = os.path.join(base_path, "results", "tenant_degradation_analysis")
    else:
        output_path = args.output
    
    # Initialize data loader and load data
    logging.info(f"Loading data for experiment {args.experiment}, round {args.round}")
    data_loader = DataLoader(base_path, args.experiment, args.round)
    
    # Load all phases
    data = data_loader.load_all_phases()
    
    if data:
        logging.info(f"Running tenant degradation analysis...")
        analyze_tenant_degradation(data_loader, output_path)
        logging.info(f"Analysis complete. Results saved to {output_path}")
    else:
        logging.error("Failed to load experiment data")
