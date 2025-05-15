#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tenant Analysis Module for Kubernetes Noisy Neighbours Lab
Este módulo implementa análises focadas em inquilinos (tenants) e suas interações,
integrando análises de degradação, correlação e causalidade entre inquilinos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import networkx as nx
from scipy import stats
from analysis_pipeline.metrics_analysis import MetricsAnalyzer

# Tentar importar bibliotecas opcionais para visualizações avançadas
try:
    from mpl_chord_diagram import chord_diagram
    CHORD_AVAILABLE = True
except ImportError:
    CHORD_AVAILABLE = False
    logging.warning("mpl_chord_diagram não encontrado. Diagrama circular não estará disponível.")

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("plotly não encontrado. Gráfico de Sankey não estará disponível.")

class TenantAnalyzer:
    def __init__(self, output_dir=None):
        """
        Inicializa o analisador de inquilinos.
        
        Args:
            output_dir (str): Diretório para salvar resultados e gráficos
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Criar subdiretórios para análises específicas
        if self.output_dir:
            self.plots_dir = self.output_dir / "tenant_analysis"
            self.plots_dir.mkdir(exist_ok=True)
            self.degradation_dir = self.output_dir / "degradation"
            self.degradation_dir.mkdir(exist_ok=True)
            self.tables_dir = self.output_dir / "tenant_tables"
            self.tables_dir.mkdir(exist_ok=True)
            
        # Inicializar analisador de métricas para reuso de métodos
        self.metrics_analyzer = MetricsAnalyzer(output_dir)
            
        logging.info(f"Inicializado TenantAnalyzer, diretório de saída: {self.output_dir}")
    
    # ========== ANÁLISE DE TENANTS ==========
    
    def compare_tenants(self, tenant_data, metric_name, phase=None):
        """
        Compara métricas entre diferentes inquilinos.
        
        Args:
            tenant_data (dict): Dicionário com dados por inquilino {tenant1: series1, tenant2: series2, ...}
            metric_name (str): Nome da métrica para contexto
            phase (str): Fase do experimento (opcional)
            
        Returns:
            DataFrame: Dados estatísticos comparativos
            
        Outputs:
            - Gráficos comparativos (.png)
            - Tabela comparativa (.csv, .tex)
        """
        if not tenant_data or len(tenant_data) < 2:
            logging.warning("Dados insuficientes para comparação entre inquilinos")
            return None
            
        # Calcular estatísticas por inquilino
        stats_dict = {}
        
        for tenant, data in tenant_data.items():
            # Converter para numérico
            clean_data = pd.to_numeric(data, errors='coerce').dropna()
            
            if len(clean_data) < 5:
                logging.warning(f"Dados insuficientes para o inquilino {tenant}")
                continue
                
            # Calcular estatísticas
            stats_dict[tenant] = {
                'mean': clean_data.mean(),
                'median': clean_data.median(),
                'std': clean_data.std(),
                'min': clean_data.min(),
                'max': clean_data.max(),
                'count': len(clean_data),
                'q25': clean_data.quantile(0.25),
                'q75': clean_data.quantile(0.75)
            }
            
        # Converter para DataFrame
        stats_df = pd.DataFrame.from_dict(stats_dict, orient='index')
        
        # Gerar visualizações comparativas
        self._generate_tenant_boxplot(tenant_data, metric_name, phase)
        self._generate_tenant_barplot(stats_df, metric_name, phase)
        
        # Salvar tabela comparativa
        if self.tables_dir:
            phase_suffix = f"_{phase}" if phase else ""
            safe_name = f"{metric_name}_tenant_comparison{phase_suffix}".replace(" ", "_").replace("/", "_")
            
            # Salvar em CSV
            csv_path = self.tables_dir / f"{safe_name}.csv"
            stats_df.to_csv(csv_path)
            logging.info(f"Tabela comparativa entre inquilinos salva em: {csv_path}")
            
            # Gerar tabela LaTeX
            tex_path = self.tables_dir / f"{safe_name}.tex"
            with open(tex_path, 'w') as f:
                # Adicionar descrição da tabela em LaTeX
                if phase:
                    caption = f"Comparação de {metric_name} entre inquilinos na fase {phase}"
                else:
                    caption = f"Comparação de {metric_name} entre inquilinos"
                
                latex_code = stats_df.to_latex(float_format=lambda x: f"{x:.3f}")
                f.write(latex_code)
            logging.info(f"Tabela LaTeX comparativa entre inquilinos salva em: {tex_path}")
            
        return stats_df
    
    def _generate_tenant_boxplot(self, tenant_data, metric_name, phase=None):
        """Gera boxplot comparando inquilinos"""
        if not self.plots_dir:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Preparar dados para o boxplot
        plot_data = []
        labels = []
        
        for tenant, data in tenant_data.items():
            clean_data = pd.to_numeric(data, errors='coerce').dropna()
            if len(clean_data) >= 5:  # Verificar dados suficientes
                plot_data.append(clean_data)
                labels.append(tenant)
        
        if not plot_data:
            return
            
        # Gerar boxplot
        plt.boxplot(plot_data, labels=labels, patch_artist=True)
        
        # Título e rótulos
        title = f"Comparação de {metric_name} entre Inquilinos"
        if phase:
            title += f" na fase {phase}"
        plt.title(title)
        plt.ylabel(metric_name)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Salvar figura
        phase_suffix = f"_{phase}" if phase else ""
        safe_name = f"{metric_name}_tenant_boxplot{phase_suffix}".replace(" ", "_").replace("/", "_")
        fig_path = self.plots_dir / f"{safe_name}.png"
        plt.savefig(fig_path)
        plt.close()
        logging.info(f"Boxplot comparativo entre inquilinos salvo em: {fig_path}")
    
    def _generate_tenant_barplot(self, stats_df, metric_name, phase=None):
        """Gera gráfico de barras comparando estatísticas entre inquilinos"""
        if not self.plots_dir or stats_df.empty:
            return
            
        plt.figure(figsize=(14, 8))
        
        # Gerar gráfico de barras para média com barras de erro
        means = stats_df['mean']
        stds = stats_df['std']
        
        tenants = stats_df.index
        x_pos = np.arange(len(tenants))
        
        plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, capsize=10)
        plt.xticks(x_pos, tenants, rotation=45)
        plt.grid(True, linestyle='--', alpha=0.5, axis='y')
        
        # Título e rótulos
        title = f"Média de {metric_name} por Inquilino"
        if phase:
            title += f" na fase {phase}"
        plt.title(title)
        plt.ylabel(f"Média de {metric_name}")
        plt.tight_layout()
        
        # Salvar figura
        phase_suffix = f"_{phase}" if phase else ""
        safe_name = f"{metric_name}_tenant_barplot{phase_suffix}".replace(" ", "_").replace("/", "_")
        fig_path = self.plots_dir / f"{safe_name}.png"
        plt.savefig(fig_path)
        plt.close()
        logging.info(f"Gráfico de barras comparativo entre inquilinos salvo em: {fig_path}")
    
    # ========== ANÁLISE DE SÉRIES TEMPORAIS POR TENANT ==========
    
    def analyze_tenant_time_series(self, tenant_data, metric_name, phase=None):
        """
        Analisa séries temporais de múltiplos inquilinos.
        
        Args:
            tenant_data (dict): Dicionário com dados por inquilino {tenant1: series1, tenant2: series2, ...}
            metric_name (str): Nome da métrica para contexto
            phase (str): Fase do experimento (opcional)
            
        Returns:
            dict: Resultados da análise de séries temporais
            
        Outputs:
            - Gráfico de séries temporais (.png)
            - Análises estatísticas
        """
        if not tenant_data or len(tenant_data) < 1:
            logging.warning("Dados insuficientes para análise de séries temporais")
            return None
            
        # Gerar visualizações de séries temporais
        self._plot_tenant_time_series(tenant_data, metric_name, phase)
        
        # Resultado por tenant
        results = {}
        
        for tenant, data in tenant_data.items():
            # Pular séries muito curtas
            if len(data) < 10:
                continue
                
            tenant_result = {}
            
            # Testar estacionariedade
            stationarity = self.metrics_analyzer.check_stationarity(data, f"{tenant}_{metric_name}")
            tenant_result['stationarity'] = stationarity
            
            # Calcular entropia
            try:
                entropy = self.metrics_analyzer.calculate_entropy(data, 'sample', f"{tenant}_{metric_name}")
                tenant_result['entropy'] = entropy
            except:
                tenant_result['entropy'] = None
                
            # Detectar anomalias
            try:
                anomalies = self.metrics_analyzer.detect_anomalies(
                    data, 'zscore', 3.0, f"{tenant}_{metric_name}")
                tenant_result['anomalies_count'] = anomalies.sum()
                tenant_result['anomalies_percent'] = (anomalies.sum() / len(anomalies)) * 100
            except:
                tenant_result['anomalies_count'] = None
                
            results[tenant] = tenant_result
            
        # Salvar resultados como tabela
        if results and self.tables_dir:
            # Converter para um formato mais adequado para DataFrame
            rows = []
            for tenant, tenant_result in results.items():
                row = {'Tenant': tenant}
                
                # Adicionar resultados de estacionariedade
                if 'stationarity' in tenant_result:
                    stat = tenant_result['stationarity']
                    if isinstance(stat, dict) and 'adf_pvalue' in stat:
                        row['ADF_p_value'] = stat['adf_pvalue']
                        row['Is_Stationary'] = stat['adf_pvalue'] < 0.05
                
                # Adicionar entropia
                if 'entropy' in tenant_result and tenant_result['entropy'] is not None:
                    row['Entropy'] = tenant_result['entropy']
                    
                # Adicionar contagem de anomalias
                if 'anomalies_count' in tenant_result and tenant_result['anomalies_count'] is not None:
                    row['Anomalies_Count'] = tenant_result['anomalies_count']
                    row['Anomalies_Percent'] = tenant_result['anomalies_percent']
                    
                rows.append(row)
                
            if rows:
                result_df = pd.DataFrame(rows)
                
                # Salvar em CSV
                phase_suffix = f"_{phase}" if phase else ""
                safe_name = f"{metric_name}_tenant_timeseries{phase_suffix}".replace(" ", "_").replace("/", "_")
                csv_path = self.tables_dir / f"{safe_name}_analysis.csv"
                result_df.to_csv(csv_path, index=False)
                logging.info(f"Resultados de análise de série temporal por inquilino em: {csv_path}")
                
                # Gerar tabela LaTeX
                tex_path = self.tables_dir / f"{safe_name}_analysis.tex"
                with open(tex_path, 'w') as f:
                    f.write(result_df.to_latex(index=False, float_format=lambda x: f"{x:.3f}"))
                
        return results
    
    def _plot_tenant_time_series(self, tenant_data, metric_name, phase=None):
        """Plota séries temporais de múltiplos inquilinos"""
        if not self.plots_dir:
            return
            
        plt.figure(figsize=(14, 8))
        
        # Cores para diferenciar os inquilinos
        colors = plt.cm.tab10.colors
        linestyles = ['-', '--', '-.', ':', '-', '--', '-.']  # Estilos repetidos para mais de 7 tenants
        
        # Plotar cada série temporal
        for i, (tenant, data) in enumerate(tenant_data.items()):
            clean_data = pd.to_numeric(data, errors='coerce')
            plt.plot(clean_data, 
                     label=tenant, 
                     color=colors[i % len(colors)],
                     linestyle=linestyles[i % len(linestyles)])
        
        # Título e rótulos
        title = f"Série Temporal de {metric_name} por Inquilino"
        if phase:
            title += f" na fase {phase}"
        plt.title(title)
        plt.ylabel(metric_name)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Salvar figura
        phase_suffix = f"_{phase}" if phase else ""
        safe_name = f"{metric_name}_tenant_timeseries{phase_suffix}".replace(" ", "_").replace("/", "_")
        fig_path = self.plots_dir / f"{safe_name}.png"
        plt.savefig(fig_path)
        plt.close()
        logging.info(f"Gráfico de séries temporais por inquilino salvo em: {fig_path}")
    
    # ========== ANÁLISE DE CORRELAÇÃO ENTRE TENANTS ==========
    
    def correlation_between_tenants(self, tenant_data, metric_name, phase=None, method='pearson'):
        """
        Calcula e visualiza correlações entre inquilinos para uma métrica.
        
        Args:
            tenant_data (dict): Dicionário com dados por inquilino {tenant1: series1, tenant2: series2, ...}
            metric_name (str): Nome da métrica para contexto
            phase (str): Fase do experimento (opcional)
            method (str): Método de correlação ('pearson', 'spearman', 'kendall')
            
        Returns:
            DataFrame: Matriz de correlação entre inquilinos
            
        Outputs:
            - Heatmap de correlação (.png)
            - Matriz de correlação (.csv, .tex)
        """
        if not tenant_data or len(tenant_data) < 2:
            logging.warning("Dados insuficientes para correlação entre inquilinos")
            return None
            
        # Alinhar séries temporais
        aligned_data = pd.DataFrame()
        
        for tenant, data in tenant_data.items():
            # Converter para numérico
            clean_data = pd.to_numeric(data, errors='coerce')
            
            if len(clean_data) < 5:  # Verificar dados suficientes
                continue
                
            # Adicionar ao DataFrame
            aligned_data[tenant] = clean_data
            
        if aligned_data.shape[1] < 2:
            logging.warning("Dados insuficientes para correlação após alinhamento")
            return None
            
        # Calcular matriz de correlação
        corr_matrix = aligned_data.corr(method=method)
        
        # Gerar heatmap
        if self.plots_dir:
            plt.figure(figsize=(10, 8))
            
            # Usar máscaras para triângulo superior
            mask = np.zeros_like(corr_matrix, dtype=bool)
            mask[np.triu_indices_from(mask)] = True
            
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                      square=True, linewidths=.5, annot=True, fmt=".2f")
            
            # Título
            title = f"Correlação entre Inquilinos para {metric_name}"
            if phase:
                title += f" na fase {phase}"
            plt.title(title)
            plt.tight_layout()
            
            # Salvar figura
            phase_suffix = f"_{phase}" if phase else ""
            safe_name = f"{metric_name}_tenant_correlation{phase_suffix}".replace(" ", "_").replace("/", "_")
            fig_path = self.plots_dir / f"{safe_name}.png"
            plt.savefig(fig_path)
            plt.close()
            logging.info(f"Heatmap de correlação entre inquilinos salvo em: {fig_path}")
            
            # Salvar matriz de correlação
            if self.tables_dir:
                # Salvar em CSV
                csv_path = self.tables_dir / f"{safe_name}.csv"
                corr_matrix.to_csv(csv_path)
                logging.info(f"Matriz de correlação entre inquilinos salva em: {csv_path}")
                
                # Gerar tabela LaTeX
                tex_path = self.tables_dir / f"{safe_name}.tex"
                with open(tex_path, 'w') as f:
                    f.write(corr_matrix.to_latex(float_format=lambda x: f"{x:.3f}"))
                    
        return corr_matrix
    
    # ========== ANÁLISE DE DEGRADAÇÃO ENTRE TENANTS ==========
    
    def analyze_tenant_degradation(self, tenant_metrics, baseline_phase, attack_phase, threshold=0.10):
        """
        Analisa degradação de serviço entre inquilinos durante um ataque.
        
        Args:
            tenant_metrics (dict): Dicionário aninhado com estrutura {tenant: {metric: {phase: data}}}
            baseline_phase (str): Nome da fase de baseline
            attack_phase (str): Nome da fase de ataque
            threshold (float): Limiar de mudança para considerar degradação significativa
            
        Returns:
            dict: Resultados da análise de degradação
            
        Outputs:
            - Gráficos de rede de degradação (.png)
            - Diagrama circular (chord) de degradação (.png)
            - Gráfico de Sankey (.html)
            - Tabelas de degradação (.csv, .tex)
        """
        # Verificar se temos dados suficientes
        if not tenant_metrics or len(tenant_metrics) < 2:
            logging.warning("Dados insuficientes para análise de degradação")
            return None
            
        # Estrutura para armazenar resultados
        degradation_results = {
            'tenant_degradation': {},  # Degradação por inquilino
            'causal_relationships': [],  # Relações causais detectadas
            'degradation_network': None  # Grafo de rede para visualização
        }
        
        # Calcular degradação para cada inquilino e métrica
        for tenant, metrics in tenant_metrics.items():
            tenant_degradation = {}
            
            for metric, phases in metrics.items():
                if baseline_phase not in phases or attack_phase not in phases:
                    continue
                    
                baseline_data = phases[baseline_phase]
                attack_data = phases[attack_phase]
                
                # Verificar dados suficientes
                if len(baseline_data) < 5 or len(attack_data) < 5:
                    continue
                    
                # Calcular estatísticas
                baseline_mean = baseline_data.mean()
                attack_mean = attack_data.mean()
                
                if baseline_mean == 0:
                    percent_change = float('inf') if attack_mean > 0 else 0
                else:
                    percent_change = ((attack_mean - baseline_mean) / abs(baseline_mean)) * 100
                    
                # Calcular significância estatística
                try:
                    t_stat, p_value = stats.ttest_ind(baseline_data, attack_data, equal_var=False)
                    significant = p_value < 0.05
                except:
                    t_stat, p_value, significant = None, None, False
                    
                # Armazenar resultados
                tenant_degradation[metric] = {
                    'baseline_mean': baseline_mean,
                    'attack_mean': attack_mean,
                    'percent_change': percent_change,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': significant
                }
                
            degradation_results['tenant_degradation'][tenant] = tenant_degradation
            
        # Identificar relações causais entre degradações
        if len(degradation_results['tenant_degradation']) >= 2:
            self._analyze_causal_relationships(
                degradation_results, 
                tenant_metrics, 
                baseline_phase, 
                attack_phase, 
                threshold)
                
            # Criar visualizações
            self._visualize_degradation_network(degradation_results)
            
            if CHORD_AVAILABLE:
                self._create_chord_diagram(degradation_results)
                
            if PLOTLY_AVAILABLE:
                self._create_sankey_diagram(degradation_results)
                
        # Gerar tabelas de degradação
        self._export_degradation_tables(degradation_results)
            
        return degradation_results
    
    def _analyze_causal_relationships(self, results, tenant_metrics, baseline_phase, attack_phase, threshold):
        """Analisa relações causais entre degradações de inquilinos"""
        # Identificar inquilinos com degradação significativa
        degraded_metrics = {}
        
        for tenant, metrics in results['tenant_degradation'].items():
            for metric, stats in metrics.items():
                if stats['significant'] and abs(stats['percent_change']) > threshold * 100:
                    if tenant not in degraded_metrics:
                        degraded_metrics[tenant] = []
                    degraded_metrics[tenant].append(metric)
        
        if len(degraded_metrics) < 2:
            return
            
        # Criar grafo direcionado para modelar relações causais
        G = nx.DiGraph()
        
        # Adicionar nós (inquilinos)
        for tenant in degraded_metrics:
            G.add_node(tenant, metrics=degraded_metrics[tenant])
            
        # Testar causalidade entre pares de inquilinos (usando métricas mais degradadas)
        causal_links = []
        
        for tenant1, metrics1 in degraded_metrics.items():
            for tenant2, metrics2 in degraded_metrics.items():
                if tenant1 == tenant2:
                    continue
                    
                # Testar causalidade nos dados de ataque (assumindo que é onde as relações são mais visíveis)
                for metric1 in metrics1[:3]:  # Limitar para as 3 métricas mais degradadas
                    for metric2 in metrics2[:3]:
                        # Obter séries temporais
                        if (metric1 in tenant_metrics[tenant1] and 
                            metric2 in tenant_metrics[tenant2] and
                            attack_phase in tenant_metrics[tenant1][metric1] and
                            attack_phase in tenant_metrics[tenant2][metric2]):
                            
                            series1 = tenant_metrics[tenant1][metric1][attack_phase]
                            series2 = tenant_metrics[tenant2][metric2][attack_phase]
                            
                            # Testar causalidade (tanto Granger quanto Transfer Entropy)
                            try:
                                # Testar causalidade de Granger
                                granger_result = self.metrics_analyzer.granger_causality(
                                    series1, series2, 5, 'ssr_chi2test', 
                                    f"{tenant1}_{metric1}", f"{tenant2}_{metric2}")
                                
                                # Verificar se há causalidade significativa
                                has_causality = False
                                min_p_value = 1.0
                                
                                if isinstance(granger_result, dict) and 'error' not in granger_result:
                                    for lag, lag_result in granger_result.items():
                                        if lag_result['significant']:
                                            has_causality = True
                                            min_p_value = min(min_p_value, lag_result['p_value'])
                                
                                if has_causality:
                                    # Adicionar relação causal ao grafo
                                    weight = 1.0 - min_p_value  # Mais confiante = maior peso
                                    
                                    if G.has_edge(tenant1, tenant2):
                                        # Incrementar peso se já existe
                                        G[tenant1][tenant2]['weight'] += weight
                                        G[tenant1][tenant2]['count'] += 1
                                    else:
                                        G.add_edge(tenant1, tenant2, weight=weight, count=1)
                                        
                                    causal_links.append({
                                        'source': tenant1,
                                        'target': tenant2,
                                        'source_metric': metric1,
                                        'target_metric': metric2,
                                        'p_value': min_p_value,
                                        'weight': weight
                                    })
                            except Exception as e:
                                logging.warning(f"Erro ao testar causalidade: {str(e)}")
        
        # Armazenar resultados
        results['causal_relationships'] = causal_links
        results['degradation_network'] = G
    
    def _visualize_degradation_network(self, results):
        """Gera visualização da rede de degradação"""
        if not self.degradation_dir or 'degradation_network' not in results or results['degradation_network'] is None:
            return
            
        G = results['degradation_network']
        
        if len(G.nodes) < 2:
            return
            
        plt.figure(figsize=(12, 10))
        
        # Calcular layout
        pos = nx.spring_layout(G, seed=42)
        
        # Obter pesos das arestas para espessura
        edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges]
        
        # Desenhar grafo
        nx.draw_networkx_nodes(G, pos, node_size=800, alpha=0.8)
        nx.draw_networkx_labels(G, pos, font_size=12)
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, 
                             edge_color='blue', arrows=True, arrowsize=15, 
                             connectionstyle='arc3,rad=0.1')
        
        # Adicionar rótulos de peso nas arestas
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        
        plt.title("Rede de Degradação entre Inquilinos")
        plt.axis("off")
        plt.tight_layout()
        
        # Salvar figura
        fig_path = self.degradation_dir / "degradation_network.png"
        plt.savefig(fig_path)
        plt.close()
        logging.info(f"Rede de degradação salva em: {fig_path}")
    
    def _create_chord_diagram(self, results):
        """Cria diagrama circular (chord diagram) para visualizar relações de degradação"""
        if not self.degradation_dir or not CHORD_AVAILABLE or 'causal_relationships' not in results:
            return
            
        causal_links = results['causal_relationships']
        
        if not causal_links:
            return
            
        # Extrair inquilinos únicos
        tenants = set()
        for link in causal_links:
            tenants.add(link['source'])
            tenants.add(link['target'])
            
        tenants = sorted(list(tenants))
        n_tenants = len(tenants)
        
        if n_tenants < 2:
            return
            
        # Criar matriz de fluxo
        flow_matrix = np.zeros((n_tenants, n_tenants))
        
        for link in causal_links:
            source_idx = tenants.index(link['source'])
            target_idx = tenants.index(link['target'])
            flow_matrix[source_idx, target_idx] += link['weight']
        
        # Normalizar fluxos
        if flow_matrix.max() > 0:
            flow_matrix = flow_matrix / flow_matrix.max()
        
        # Criar diagrama circular
        plt.figure(figsize=(12, 12))
        chord_diagram(flow_matrix, names=tenants, width=0.1, pad=2, gap=0.03)
        plt.title("Diagrama Circular de Degradação entre Inquilinos")
        
        # Salvar figura
        fig_path = self.degradation_dir / "chord_diagram.png"
        plt.savefig(fig_path)
        plt.close()
        logging.info(f"Diagrama circular salvo em: {fig_path}")
    
    def _create_sankey_diagram(self, results):
        """Cria diagrama de Sankey para visualizar fluxo de degradação"""
        if not self.degradation_dir or not PLOTLY_AVAILABLE or 'causal_relationships' not in results:
            return
            
        causal_links = results['causal_relationships']
        
        if not causal_links:
            return
            
        # Preparar dados para Sankey
        tenants = set()
        for link in causal_links:
            tenants.add(link['source'])
            tenants.add(link['target'])
            
        tenant_ids = {tenant: i for i, tenant in enumerate(sorted(tenants))}
        
        # Preparar nós e links
        sources = []
        targets = []
        values = []
        
        for link in causal_links:
            sources.append(tenant_ids[link['source']])
            targets.append(tenant_ids[link['target']])
            values.append(link['weight'] * 100)  # Escalar para melhor visualização
        
        # Criar figura Sankey
        fig = go.Figure(data=[go.Sankey(
            node = dict(
                pad = 15,
                thickness = 20,
                line = dict(color = "black", width = 0.5),
                label = list(tenant_ids.keys())
            ),
            link = dict(
                source = sources,
                target = targets,
                value = values
            )
        )])
        
        fig.update_layout(
            title="Fluxo de Degradação entre Inquilinos",
            font=dict(size=12)
        )
        
        # Salvar como HTML
        html_path = self.degradation_dir / "sankey_diagram.html"
        fig.write_html(str(html_path))
        logging.info(f"Diagrama de Sankey salvo em: {html_path}")
    
    def _export_degradation_tables(self, results):
        """Exporta tabelas com resultados da análise de degradação"""
        if not self.tables_dir or 'tenant_degradation' not in results:
            return
            
        # Preparar dados para tabela geral de degradação
        rows = []
        
        for tenant, metrics in results['tenant_degradation'].items():
            for metric, stats in metrics.items():
                if stats['significant']:
                    rows.append({
                        'Tenant': tenant,
                        'Metric': metric,
                        'Baseline': stats['baseline_mean'],
                        'Attack': stats['attack_mean'],
                        'Change (%)': stats['percent_change'],
                        'p-value': stats['p_value']
                    })
        
        if not rows:
            return
            
        # Criar DataFrame
        df = pd.DataFrame(rows)
        
        # Ordenar por magnitude de mudança
        df = df.sort_values(by='Change (%)', key=abs, ascending=False)
        
        # Salvar em CSV
        csv_path = self.tables_dir / "tenant_degradation.csv"
        df.to_csv(csv_path, index=False)
        logging.info(f"Tabela de degradação por inquilino salva em: {csv_path}")
        
        # Gerar tabela LaTeX
        tex_path = self.tables_dir / "tenant_degradation.tex"
        with open(tex_path, 'w') as f:
            f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.3f}"))
        
        # Se houver relações causais, criar tabela específica
        if 'causal_relationships' in results and results['causal_relationships']:
            causal_df = pd.DataFrame(results['causal_relationships'])
            
            # Ordenar por confiabilidade (menor p-valor primeiro)
            if 'p_value' in causal_df.columns:
                causal_df = causal_df.sort_values(by='p_value')
            
            # Salvar em CSV
            csv_path = self.tables_dir / "causal_relationships.csv"
            causal_df.to_csv(csv_path, index=False)
            logging.info(f"Tabela de relações causais salva em: {csv_path}")
            
            # Gerar tabela LaTeX
            tex_path = self.tables_dir / "causal_relationships.tex"
            with open(tex_path, 'w') as f:
                f.write(causal_df.to_latex(index=False, float_format=lambda x: f"{x:.3f}"))
    
    # ========== SISTEMA DE SUGESTÕES ==========
    
    def suggest_visualizations(self, tenant_metrics):
        """
        Sugere visualizações relevantes com base nas características dos dados de inquilinos.
        
        Args:
            tenant_metrics (dict): Dicionário aninhado com estrutura {tenant: {metric: {phase: data}}}
            
        Returns:
            dict: Sugestões de visualizações com justificativas
        """
        suggestions = {}
        
        # Verificar dados suficientes
        if not tenant_metrics or len(tenant_metrics) < 2:
            return {"error": "Dados insuficientes para sugestões"}
            
        # Identificar número de inquilinos e métricas
        tenants = list(tenant_metrics.keys())
        num_tenants = len(tenants)
        
        # Identificar métricas comuns a múltiplos inquilinos
        common_metrics = {}
        for tenant, metrics in tenant_metrics.items():
            for metric in metrics:
                if metric not in common_metrics:
                    common_metrics[metric] = set()
                common_metrics[metric].add(tenant)
        
        # Sugerir visualizações para métricas comuns a vários inquilinos
        common_viz = []
        for metric, tenant_set in common_metrics.items():
            if len(tenant_set) >= max(2, num_tenants // 2):  # Pelo menos 2 ou metade dos inquilinos
                common_viz.append({
                    'metric': metric,
                    'tenants': list(tenant_set),
                    'count': len(tenant_set)
                })
                
        # Ordenar por contagem descendente
        common_viz.sort(key=lambda x: x['count'], reverse=True)
        
        # Sugerir visualizações de séries temporais para métricas comuns
        for i, viz in enumerate(common_viz[:5]):  # Top 5 métricas comuns
            metric = viz['metric']
            suggestions[f"timeseries_{metric}"] = {
                "type": "Séries Temporais Comparativas",
                "description": f"Compare {metric} entre {viz['count']} inquilinos",
                "justification": f"Métrica comum a {viz['count']} inquilinos, boa para comparação direta",
                "function": "analyze_tenant_time_series",
                "priority": i + 1
            }
            
        # Verificar se há várias fases para sugerir análise de degradação
        phases_by_tenant = {}
        for tenant, metrics in tenant_metrics.items():
            for metric, phases in metrics.items():
                if tenant not in phases_by_tenant:
                    phases_by_tenant[tenant] = set()
                phases_by_tenant[tenant].update(phases.keys())
        
        tenants_with_baseline_attack = []
        for tenant, phases in phases_by_tenant.items():
            baseline_phases = [p for p in phases if "baseline" in p.lower()]
            attack_phases = [p for p in phases if "attack" in p.lower() or "ataque" in p.lower()]
            
            if baseline_phases and attack_phases:
                tenants_with_baseline_attack.append(tenant)
                
        # Sugerir análise de degradação se temos fases apropriadas
        if len(tenants_with_baseline_attack) >= 2:
            suggestions["degradation_analysis"] = {
                "type": "Análise de Degradação entre Inquilinos",
                "description": "Análise completa de degradação com visualizações de rede causal",
                "justification": f"{len(tenants_with_baseline_attack)} inquilinos têm dados de baseline e ataque",
                "function": "analyze_tenant_degradation",
                "priority": 0
            }
            
        # Sugerir correlação entre inquilinos para métricas populares
        if len(common_viz) > 0:
            top_metric = common_viz[0]['metric']
            suggestions["correlation_analysis"] = {
                "type": "Matriz de Correlação entre Inquilinos",
                "description": f"Análise de correlação de {top_metric} entre inquilinos",
                "justification": f"Métrica {top_metric} presente em {common_viz[0]['count']} inquilinos",
                "function": "correlation_between_tenants",
                "priority": 3
            }
            
        return suggestions