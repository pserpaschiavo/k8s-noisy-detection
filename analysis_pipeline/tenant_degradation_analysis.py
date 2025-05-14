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
import matplotlib.cm as cm
import mplcursors

# Tentar importar as bibliotecas opcionais
try:
    from mpl_chord_diagram import chord_diagram
    CHORD_AVAILABLE = True
except ImportError:
    CHORD_AVAILABLE = False
    logging.warning("mpl_chord_diagram não encontrado. Diagrama circular não estará disponível.")

try:
    import plotly.graph_objects as go
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("plotly não encontrado. Gráfico de Sankey não estará disponível.")

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
                        # This code creates a simple lag plot visualization
                        if result and self.ts_dir:
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
            
            # Plot causality visualizations in various formats
            if self.plots_dir and causality_pairs:
                # Original network plot
                self.plot_causality_network(
                    causality_pairs, 
                    title=f"Granger Causality Network: {metric} ({phase})",
                    filename=f"causality_network_{phase}_{metric}.png"
                )
                
                # Improved network plot for academic publications
                self.plot_improved_causality_network(
                    causality_pairs,
                    title=f"Rede de Causalidade de Granger: {metric} ({phase})",
                    filename=f"academic_causality_network_{phase}_{metric}.png",
                    academic_style=True
                )
                
                # Chord diagram if available
                self.plot_chord_diagram(
                    causality_pairs,
                    title=f"Diagrama de Relações Causais: {metric} ({phase})",
                    filename=f"chord_diagram_{phase}_{metric}.png"
                )
                
                # Sankey diagram if available
                self.plot_sankey_degradation(
                    causality_pairs,
                    title=f"Fluxo de Degradação: {metric} ({phase})",
                    filename=f"sankey_diagram_{phase}_{metric}.png"
                )
                
                # Also save causality matrix to time series folder
                if self.ts_dir and not causality_matrix.empty:
                    # Traditional heatmap
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(causality_matrix, annot=True, cmap='YlOrRd_r', vmin=0, vmax=0.05, 
                               square=True, fmt='.4f', cbar_kws={'shrink': .8})
                    plt.title(f'Granger Causality p-values: {metric} ({phase})')
                    plt.tight_layout()
                    
                    filename = f"granger_causality_matrix_{phase}_{metric}.png".replace(' ', '_').lower()
                    plt.savefig(self.ts_dir / filename, bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    # Improved heatmap
                    self.plot_causality_heatmap(
                        causality_matrix,
                        title=f"Mapa de Calor de Causalidade: {metric} ({phase})",
                        filename=f"causality_heatmap_{phase}_{metric}.png"
                    )
            
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

    def plot_improved_causality_network(self, causality_pairs, title=None, filename=None, academic_style=True):
        """
        Versão aprimorada da visualização de rede de causalidade para publicações acadêmicas.
        
        Args:
            causality_pairs (list): Lista de dicionários com resultados de causalidade
            title (str): Título para o gráfico
            filename (str): Nome do arquivo para salvar o gráfico
            academic_style (bool): Se deve usar estilo acadêmico (True) ou padrão (False)
        """
        if not causality_pairs:
            return
            
        # Criar grafo direcionado
        G = nx.DiGraph()
        
        # Adicionar nós e arestas
        for pair in causality_pairs:
            source = pair['source']
            target = pair['target']
            p_value = pair['p_value']
            lag = pair.get('lag', 1)
            
            # Adicionar nós se não existirem
            if source not in G:
                G.add_node(source)
            if target not in G:
                G.add_node(target)
            
            # Adicionar aresta com peso baseado no p-valor
            weight = 1 - p_value  # Transformar p-valor em peso
            G.add_edge(source, target, weight=weight, p_value=p_value, lag=lag)
        
        # Prosseguir apenas se tivermos arestas
        if not G.edges:
            logging.warning("Sem relações causais significativas para visualizar")
            return
        
        # Configurar estilo para publicação acadêmica
        if academic_style:
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams['font.family'] = 'serif'
            # Desabilitar LaTeX para evitar problemas em sistemas sem LaTeX instalado
            plt.rcParams['text.usetex'] = False
        
        # Criar plot
        plt.figure(figsize=(10, 8), dpi=300)
        
        # Calcular centralidade para definir tamanho dos nós
        in_centrality = nx.in_degree_centrality(G)  # Quem é mais afetado
        out_centrality = nx.out_degree_centrality(G)  # Quem causa mais problemas
        
        # Tamanho do nó proporcional à centralidade de saída (quanto mais causa, maior o nó)
        node_sizes = [1000 * (out_centrality[node] + 0.2) for node in G.nodes()]
        
        # Cor do nó baseada na centralidade de entrada (quão afetado é)
        node_colors = [in_centrality[node] for node in G.nodes()]
        
        # Usar layout hierárquico para melhor visualização de fluxos causais
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            # Fallback para spring layout se kamada_kawai não convergir
            pos = nx.spring_layout(G, k=0.8, seed=42)
        
        # Desenhar nós com bordas mais definidas e cores baseadas na centralidade
        nodes = nx.draw_networkx_nodes(G, pos, 
                              node_size=node_sizes,
                              node_color=node_colors, 
                              cmap=plt.cm.YlOrRd,
                              alpha=0.8, 
                              linewidths=1.5,
                              edgecolors='k')
        
        # Adicionar barra de cores para mostrar o significado das cores dos nós
        cbar = plt.colorbar(nodes, shrink=0.7, label='Grau de Impacto (Centralidade de Entrada)')
        cbar.ax.tick_params(labelsize=8)
        
        # Desenhar arestas com largura proporcional à causalidade
        edges = G.edges(data=True)
        edge_widths = [4 * (1 - d['p_value']) for _, _, d in edges]
        edge_colors = [cm.YlGnBu(1 - d['p_value']) for _, _, d in edges]
        nx.draw_networkx_edges(G, pos, 
                              arrowsize=25, 
                              width=edge_widths, 
                              edge_color=edge_colors,
                              alpha=0.7,
                              connectionstyle='arc3,rad=0.1')  # Curvatura leve para evitar sobreposições
        
        # Adicionar rótulos aos nós com formatação melhorada
        nx.draw_networkx_labels(G, pos, 
                               font_size=14, 
                               font_weight='bold',
                               bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=6))
        
        # Adicionar rótulos às arestas (p-valores)
        edge_labels = {(u, v): f"p={d['p_value']:.3f}\nlag={d['lag']}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, 
                                    edge_labels=edge_labels, 
                                    font_size=11,
                                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))
        
        plt.title(title or "Rede de Causalidade entre Inquilinos", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Adicionar legenda explicativa
        plt.figtext(0.02, 0.02, 
                   "Nota: A espessura das setas indica a força da relação causal.\n"
                   "Valores p menores indicam maior confiança na relação causal.",
                   fontsize=10, ha='left')
        
        # Salvar o plot no diretório principal de visualização
        if filename and self.plots_dir:
            # Salvar como PNG de alta resolução
            plt.savefig(self.plots_dir / filename, bbox_inches='tight', dpi=300)
            
            # Salvar também como PDF para incluir em papers acadêmicos
            pdf_filename = filename.replace('.png', '.pdf')
            plt.savefig(self.plots_dir / pdf_filename, bbox_inches='tight', format='pdf')
            
            # Salvar também no diretório de análise de séries temporais
            if self.ts_dir:
                plt.savefig(self.ts_dir / filename, bbox_inches='tight', dpi=300)
                plt.savefig(self.ts_dir / pdf_filename, bbox_inches='tight', format='pdf')
            
            logging.info(f"Gráfico de rede de causalidade salvo em {filename}")
            plt.close()
        else:
            plt.show()
            plt.close()

    def plot_chord_diagram(self, causality_pairs, title=None, filename=None):
        """
        Cria um diagrama circular (chord diagram) para visualizar relações de causalidade.
        Requer a biblioteca matplotlib-chord.
        
        Args:
            causality_pairs (list): Lista de dicionários com resultados de causalidade
            title (str): Título para o gráfico
            filename (str): Nome do arquivo para salvar o gráfico
        """
        if not CHORD_AVAILABLE:
            logging.error("Não foi possível criar o diagrama circular: biblioteca mpl_chord_diagram não está disponível.")
            logging.error("Instale com: pip install mpl_chord_diagram")
            return
            
        if not causality_pairs:
            logging.warning("Sem dados de causalidade para visualizar.")
            return
        
        # Preparar os dados para o diagrama circular
        sources = []
        targets = []
        values = []
        
        for pair in causality_pairs:
            sources.append(pair['source'])
            targets.append(pair['target'])
            # Converter p-valor para força (1-p)
            values.append(1 - pair['p_value'])
        
        # Criar DataFrame com os dados
        df = pd.DataFrame({
            'source': sources,
            'target': targets,
            'value': values
        })
        
        # Obter lista única de todos os inquilinos
        all_tenants = sorted(list(set(sources + targets)))
        
        # Criar matriz de fluxo
        flow_matrix = np.zeros((len(all_tenants), len(all_tenants)))
        
        # Preencher matriz com valores
        for _, row in df.iterrows():
            source_idx = all_tenants.index(row['source'])
            target_idx = all_tenants.index(row['target'])
            flow_matrix[source_idx, target_idx] = row['value']
        
        # Configuração para estilo acadêmico
        plt.figure(figsize=(12, 10), dpi=300)
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'
        # Desabilitar LaTeX para evitar problemas em sistemas sem LaTeX instalado
        plt.rcParams['text.usetex'] = False
        
        # Criar diagrama circular com cores acessíveis
        chord_diagram(flow_matrix, 
                    names=all_tenants,
                    colors=[plt.cm.Set3(i) for i in np.linspace(0, 1, len(all_tenants))],
                    alpha=0.7,
                    use_gradient=True,
                    width=0.1,
                    gap=0.03,
                    fontsize=14)
        
        plt.title(title or "Relações de Degradação entre Inquilinos", fontsize=16, fontweight='bold')
        
        # Adicionar legenda explicativa
        plt.figtext(0.5, 0.02, 
                   "Nota: A espessura das conexões indica a força da relação causal.\n"
                   "Conexões mais largas indicam maior probabilidade de relação causal.",
                   fontsize=10, ha='center')
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Ajustar para incluir o texto explicativo
        
        # Salvar em alta resolução
        if filename and self.plots_dir:
            # Salvar como PNG
            plt.savefig(self.plots_dir / filename, bbox_inches='tight', dpi=300)
            
            # Salvar como PDF para publicação
            pdf_filename = filename.replace('.png', '.pdf')
            plt.savefig(self.plots_dir / pdf_filename, bbox_inches='tight', format='pdf')
            
            # Salvar também no diretório de análise de séries temporais
            if self.ts_dir:
                plt.savefig(self.ts_dir / filename, bbox_inches='tight', dpi=300)
                plt.savefig(self.ts_dir / pdf_filename, bbox_inches='tight', format='pdf')
                
            logging.info(f"Diagrama circular salvo em {filename}")
            plt.close()
        else:
            plt.show()
            plt.close()

    def plot_causality_heatmap(self, causality_matrix, title=None, filename=None):
        """
        Cria um mapa de calor mostrando as relações causais entre inquilinos.
        O mapa usa um esquema de cor que vai de branco (sem causalidade) a vermelho escuro (forte causalidade).
        
        Args:
            causality_matrix (DataFrame): Matriz de p-valores da causalidade de Granger
            title (str): Título para o gráfico
            filename (str): Nome do arquivo para salvar o gráfico
        """
        if causality_matrix.empty:
            logging.warning("Matriz de causalidade vazia. Não há dados para visualizar.")
            return
        
        # Configurações para estilo acadêmico
        plt.figure(figsize=(12, 10), dpi=300)
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'
        # Desabilitar LaTeX para evitar problemas em sistemas sem LaTeX instalado
        plt.rcParams['text.usetex'] = False
        
        # Transformar p-valores em intensidade de causalidade (1-p)
        causality_strength = 1 - causality_matrix.copy()
        
        # Configurar diagonal para NaN para não mostrar auto-causalidade
        np.fill_diagonal(causality_strength.values, np.nan)
        
        # Criar máscara para valores ausentes
        mask = np.isnan(causality_strength.values)
        
        # Criar mapa de calor com esquema de cores científico
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(causality_strength, 
                        annot=True, 
                        fmt='.3f',
                        cmap='YlOrRd', 
                        mask=mask,
                        cbar_kws={'label': 'Intensidade da Causalidade (1-p)', 'shrink': 0.8},
                        square=True,
                        linewidths=0.5,
                        annot_kws={"size": 10})
        
        # Configurar título e rótulos
        plt.title(title or "Intensidade de Causalidade entre Inquilinos", fontsize=16, fontweight='bold')
        plt.xlabel("Inquilino Causador", fontsize=14)
        plt.ylabel("Inquilino Afetado", fontsize=14)
        
        # Rotacionar rótulos para melhor legibilidade
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        
        # Adicionar nota explicativa
        plt.figtext(0.02, 0.02, 
                   "Nota: Valores mais altos (vermelho) indicam maior confiança na relação causal.\n"
                   "Linhas representam inquilinos causadores, colunas representam inquilinos afetados.",
                   fontsize=10, ha='left')
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Ajustar para incluir o texto explicativo
        
        # Adicionar interatividade básica
        if not plt.isinteractive():
            mplcursors.cursor(hover=True).connect(
                "add", lambda sel: sel.annotation.set_text(
                    f"Causador: {causality_strength.columns[int(sel.target.index)]}\n" +
                    f"Afetado: {causality_strength.index[int(sel.target[1])]}\n" +
                    f"Intensidade: {causality_strength.iloc[int(sel.target[1]), int(sel.target.index)]:.3f}"
                )
            )
        
        # Salvar em alta resolução
        if filename and self.plots_dir:
            # Salvar como PNG
            plt.savefig(self.plots_dir / filename, bbox_inches='tight', dpi=300)
            
            # Salvar em PDF para publicações acadêmicas
            pdf_filename = filename.replace('.png', '.pdf')
            plt.savefig(self.plots_dir / pdf_filename, bbox_inches='tight', format='pdf')
            
            # Salvar também no diretório de análise temporal
            if self.ts_dir:
                plt.savefig(self.ts_dir / filename, bbox_inches='tight', dpi=300)
                plt.savefig(self.ts_dir / pdf_filename, bbox_inches='tight', format='pdf')
            
            logging.info(f"Mapa de calor de causalidade salvo em {filename}")
            plt.close()
        else:
            plt.show()
            plt.close()

    def plot_sankey_degradation(self, causality_pairs, title=None, filename=None):
        """
        Cria um diagrama de Sankey mostrando o fluxo de degradação entre inquilinos.
        Requer a biblioteca plotly.
        
        Args:
            causality_pairs (list): Lista de dicionários com resultados de causalidade
            title (str): Título para o gráfico
            filename (str): Nome do arquivo para salvar o gráfico
        """
        if not PLOTLY_AVAILABLE:
            logging.error("Não foi possível criar o diagrama de Sankey: biblioteca plotly não está disponível.")
            logging.error("Instale com: pip install plotly")
            return
        
        if not causality_pairs:
            logging.warning("Sem dados de causalidade para visualizar.")
            return
        
        # Preparar dados para o Sankey
        sources = []
        targets = []
        values = []
        labels = []
        
        # Obter lista única de todos os inquilinos
        all_tenants = sorted(list(set([p['source'] for p in causality_pairs] + [p['target'] for p in causality_pairs])))
        tenant_indices = {tenant: idx for idx, tenant in enumerate(all_tenants)}
        
        # Popular listas para o Sankey
        for pair in causality_pairs:
            source_idx = tenant_indices[pair['source']]
            target_idx = tenant_indices[pair['target']]
            sources.append(source_idx)
            targets.append(target_idx)
            # Usar 1-p como valor do fluxo (força da causalidade)
            values.append((1 - pair['p_value']) * 10)  # Multiplicar por 10 para melhor visualização
        
        # Criar figura Sankey
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_tenants,
                color="rgba(31,119,180,0.8)"  # Azul semi-transparente
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=[f"rgba(214,39,40,{(1-p['p_value'])*0.8+0.2})" for p in causality_pairs]  # Vermelho com opacidade baseada na força
            ))])
        
        # Adicionar anotação explicativa
        fig.add_annotation(
            x=0.5, y=-0.1,
            xref="paper", yref="paper",
            text="Nota: A espessura das conexões representa a força da relação causal.<br>Cores mais intensas indicam maior confiança na relação.",
            showarrow=False,
            font=dict(family="Times New Roman", size=12),
            align="center",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
        
        # Configurar layout
        fig.update_layout(
            title_text=title or "Fluxo de Degradação entre Inquilinos",
            font_size=14,
            font_family="Times New Roman",
            paper_bgcolor='white',
            plot_bgcolor='white',
            width=1000,
            height=800,
            margin=dict(t=50, l=25, r=25, b=70)  # Mais espaço abaixo para anotação
        )
        
        # Salvar como HTML interativo e como imagem estática
        if filename and self.plots_dir:
            # HTML interativo
            html_filename = filename.replace('.png', '.html')
            plot(fig, filename=str(self.plots_dir / html_filename), auto_open=False)
            
            # Imagem estática de alta qualidade
            fig.write_image(str(self.plots_dir / filename), scale=2)
            
            # PDF para publicação
            pdf_filename = filename.replace('.png', '.pdf')
            fig.write_image(str(self.plots_dir / pdf_filename))
            
            logging.info(f"Diagrama de Sankey salvo em {filename} e {html_filename}")
            
            # Salvar também no diretório de análise temporal, se disponível
            if self.ts_dir:
                plot(fig, filename=str(self.ts_dir / html_filename), auto_open=False)
                fig.write_image(str(self.ts_dir / filename), scale=2)
                fig.write_image(str(self.ts_dir / pdf_filename))
        else:
            # Mostrar no navegador em modo interativo
            fig.show()

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

    def generate_all_visualizations(self, data, phases, metrics_of_interest, tenants, output_subdir='degradation_visualizations'):
        """
        Gera todas as visualizações de degradação disponíveis para os dados fornecidos.
        
        Args:
            data (dict): Dados do DataLoader com fases
            phases (list): Lista de nomes das fases
            metrics_of_interest (list): Lista de métricas para analisar
            tenants (list): Lista de inquilinos para analisar
            output_subdir (str): Subdiretório para salvar as visualizações
        
        Returns:
            dict: Dicionário com caminhos para as visualizações geradas
        """
        if not self.output_dir:
            logging.warning("Diretório de saída não definido. Não é possível salvar visualizações.")
            return {}
            
        # Criar diretório para visualizações
        vis_dir = self.output_dir / output_subdir
        vis_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Gerando visualizações de degradação em: {vis_dir}")
        
        # Verificar fases
        baseline_phase = '1 - Baseline' if '1 - Baseline' in phases else phases[0] if phases else None
        attack_phase = '2 - Attack' if '2 - Attack' in phases else phases[1] if len(phases) > 1 else None
        recovery_phase = '3 - Recovery' if '3 - Recovery' in phases else phases[2] if len(phases) > 2 else None
        
        if not baseline_phase or not attack_phase:
            logging.error("Fases insuficientes para análise de degradação.")
            return {}
            
        generated_visualizations = {}
        
        # Análise de correlação cruzada entre inquilinos
        logging.info("Gerando visualizações de correlação entre inquilinos...")
        
        # Gerar visualizações para fase de baseline e ataque
        for phase in [baseline_phase, attack_phase, recovery_phase]:
            if phase:
                phase_vis = {}
                
                # Correlações
                correlations = self.analyze_cross_tenant_correlations(data, phase, metrics_of_interest, tenants)
                if correlations:
                    phase_vis['correlations'] = correlations
                
                # Causalidade (gera diversas visualizações)
                causality = self.analyze_granger_causality(data, phase, metrics_of_interest, tenants)
                if causality:
                    phase_vis['causality'] = causality
                    
                # Adicionar ao dicionário de visualizações
                generated_visualizations[phase] = phase_vis
        
        # Identificar fontes de degradação e gerar relatórios
        logging.info("Identificando fontes de degradação...")
        degradation_sources = self.identify_degradation_sources(data, phases, metrics_of_interest, tenants)
        generated_visualizations['degradation_sources'] = degradation_sources
        
        logging.info(f"Todas as visualizações de degradação foram geradas em: {vis_dir}")
        return generated_visualizations

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
