#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suggestion Engine Module for Kubernetes Noisy Neighbours Lab
Este módulo implementa um sistema de sugestões para visualizações e tabelas mais relevantes
com base nas características dos dados analisados.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import scipy.stats as stats

class SuggestionEngine:
    def __init__(self, output_dir=None):
        """
        Inicializa o motor de sugestões.
        
        Args:
            output_dir (str): Diretório para salvar resultados
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Criar subdiretório para sugestões
        if self.output_dir:
            self.suggestions_dir = self.output_dir / "suggestions"
            self.suggestions_dir.mkdir(exist_ok=True)
            
        logging.info(f"Inicializado SuggestionEngine, diretório de saída: {self.output_dir}")
    
    # ========== SUGESTÕES DE VISUALIZAÇÕES ==========
    
    def suggest_metric_plots(self, metric_data, metric_name):
        """
        Sugere visualizações para uma métrica específica.
        
        Args:
            metric_data (Series): Dados da métrica
            metric_name (str): Nome da métrica
            
        Returns:
            dict: Sugestões de visualização com justificativas
        """
        suggestions = {}
        
        # Verificar dados suficientes
        if metric_data is None or len(metric_data) < 5:
            return {"error": "Dados insuficientes para sugestões"}
            
        # Converter para numérico
        try:
            numeric_data = pd.to_numeric(metric_data, errors='coerce').dropna()
            
            if len(numeric_data) < 5:  # Verificar novamente após conversão
                return {"error": "Dados insuficientes para sugestões após conversão numérica"}
        except:
            return {"error": "Erro ao converter dados para formato numérico"}
        
        # Calcular estatísticas para tomada de decisão
        try:
            stats_data = {
                'mean': numeric_data.mean(),
                'median': numeric_data.median(),
                'std': numeric_data.std(),
                'min': numeric_data.min(),
                'max': numeric_data.max(),
                'skewness': stats.skew(numeric_data),
                'kurtosis': stats.kurtosis(numeric_data)
            }
            
            # Analisar índice para verificar se é série temporal
            is_timeseries = isinstance(numeric_data.index, pd.DatetimeIndex)
            
            # Calcular características do índice (para séries temporais)
            if is_timeseries:
                time_gaps = np.diff(numeric_data.index.astype(np.int64) // 10**9)  # em segundos
                time_stats = {
                    'time_gap_mean': np.mean(time_gaps),
                    'time_gap_std': np.std(time_gaps),
                    'total_duration': (numeric_data.index[-1] - numeric_data.index[0]).total_seconds()
                }
                
                # Verificar regularidade da amostragem
                is_regular_sampling = time_stats['time_gap_std'] / time_stats['time_gap_mean'] < 0.1
                stats_data.update(time_stats)
                stats_data['is_regular_sampling'] = is_regular_sampling
                
        except Exception as e:
            logging.warning(f"Erro ao calcular estatísticas para sugestões: {str(e)}")
            stats_data = {}
        
        # Sugerir visualizações com base nas características
        
        # 1. Séries temporais
        if is_timeseries:
            suggestions["timeseries"] = {
                "type": "Série Temporal",
                "description": f"Gráfico de linha mostrando evolução de {metric_name} ao longo do tempo",
                "justification": "Dados possuem índice temporal",
                "function": "plot_timeseries",
                "priority": 1
            }
            
            # Verificar se há variações significativas para sugerir detecção de pontos de mudança
            variation_ratio = stats_data['std'] / stats_data['mean'] if stats_data.get('mean', 0) != 0 else 0
            if variation_ratio > 0.2:  # Variação significativa
                suggestions["change_points"] = {
                    "type": "Detecção de Pontos de Mudança",
                    "description": f"Identificação de mudanças significativas em {metric_name}",
                    "justification": f"Métrica apresenta variação significativa (CV={variation_ratio:.2f})",
                    "function": "detect_change_points",
                    "priority": 3
                }
        
        # 2. Distribuição de dados
        # Verificar assimetria (skewness)
        if 'skewness' in stats_data:
            abs_skew = abs(stats_data['skewness'])
            
            if abs_skew > 1.0:
                suggestions["distribution"] = {
                    "type": "Distribuição de Dados",
                    "description": f"Histograma da distribuição de {metric_name}",
                    "justification": f"Dados apresentam assimetria significativa (skew={abs_skew:.2f})",
                    "function": "plot_histogram",
                    "priority": 2
                }
            else:
                suggestions["distribution"] = {
                    "type": "Distribuição de Dados",
                    "description": f"Histograma da distribuição de {metric_name}",
                    "justification": "Visualização da distribuição de valores",
                    "function": "plot_histogram",
                    "priority": 4
                }
                
        # 3. Boxplot para outliers
        if 'kurtosis' in stats_data and stats_data['kurtosis'] > 2.0:
            suggestions["boxplot"] = {
                "type": "Boxplot",
                "description": f"Boxplot mostrando distribuição e outliers de {metric_name}",
                "justification": f"Dados apresentam kurtosis elevada ({stats_data['kurtosis']:.2f}), indicando possíveis outliers",
                "function": "plot_boxplot",
                "priority": 2
            }
        
        return suggestions
    
    def suggest_phase_comparisons(self, phase_data, metric_name):
        """
        Sugere visualizações para comparação entre fases.
        
        Args:
            phase_data (dict): Dicionário com dados por fase {fase1: series1, fase2: series2, ...}
            metric_name (str): Nome da métrica
            
        Returns:
            dict: Sugestões de visualização com justificativas
        """
        suggestions = {}
        
        # Verificar dados suficientes
        if not phase_data or len(phase_data) < 2:
            return {"error": "Precisa de pelo menos duas fases para comparação"}
            
        # Verificar se existem diferenças significativas entre fases
        phase_stats = {}
        has_significant_diff = False
        
        for phase, data in phase_data.items():
            try:
                numeric_data = pd.to_numeric(data, errors='coerce').dropna()
                phase_stats[phase] = {
                    'mean': numeric_data.mean(),
                    'median': numeric_data.median(),
                    'std': numeric_data.std(),
                    'count': len(numeric_data)
                }
            except:
                continue
        
        # Testar diferenças estatísticas
        phases = list(phase_stats.keys())
        for i in range(len(phases)):
            for j in range(i+1, len(phases)):
                phase1, phase2 = phases[i], phases[j]
                stats1, stats2 = phase_stats[phase1], phase_stats[phase2]
                
                # Verificar diferença estatisticamente significativa
                if stats1['count'] > 5 and stats2['count'] > 5:
                    mean_diff = abs(stats1['mean'] - stats2['mean'])
                    pooled_std = ((stats1['std']**2 / stats1['count']) + (stats2['std']**2 / stats2['count'])) ** 0.5
                    
                    if pooled_std > 0:
                        effect_size = mean_diff / pooled_std
                        if effect_size > 0.5:  # Efeito moderado ou maior
                            has_significant_diff = True
                            break
        
        # Sempre sugerir visualização de boxplot para comparação
        suggestions["boxplot_comparison"] = {
            "type": "Boxplot Comparativo",
            "description": f"Boxplots para comparar {metric_name} entre fases",
            "justification": "Comparação visual da distribuição de valores entre fases",
            "function": "compare_phases",
            "parameters": {"comparison_methods": ["boxplot"]},
            "priority": 1
        }
        
        # Se diferenças significativas, sugerir testes estatísticos
        if has_significant_diff:
            suggestions["statistical_tests"] = {
                "type": "Testes Estatísticos",
                "description": f"Comparação estatística de {metric_name} entre fases",
                "justification": "Diferenças potencialmente significativas detectadas entre fases",
                "function": "compare_phases",
                "parameters": {"comparison_methods": ["stats_test"]},
                "priority": 2
            }
            
            # Sugerir visualização de violino para mostrar distribuições
            suggestions["violin_comparison"] = {
                "type": "Gráfico de Violino",
                "description": f"Comparação detalhada das distribuições de {metric_name} entre fases",
                "justification": "Diferenças significativas merecem análise detalhada das distribuições",
                "function": "compare_phases",
                "parameters": {"comparison_methods": ["violin"]},
                "priority": 3
            }
        
        # Verificar se dados são séries temporais para sugerir sobreposição
        is_timeseries = False
        for data in phase_data.values():
            if isinstance(data.index, pd.DatetimeIndex):
                is_timeseries = True
                break
                
        if is_timeseries:
            suggestions["time_series_comparison"] = {
                "type": "Séries Temporais Sobrepostas",
                "description": f"Gráfico com séries temporais de {metric_name} sobrepostas por fase",
                "justification": "Comparação visual direta da evolução temporal entre fases",
                "function": "compare_phases", 
                "parameters": {"comparison_methods": ["time_series"]},
                "priority": 2 if has_significant_diff else 3
            }
        
        return suggestions
    
    def suggest_tenant_visualizations(self, tenant_data, metric_name):
        """
        Sugere visualizações para comparação entre inquilinos.
        
        Args:
            tenant_data (dict): Dicionário com dados por inquilino {tenant1: series1, tenant2: series2, ...}
            metric_name (str): Nome da métrica
            
        Returns:
            dict: Sugestões de visualização com justificativas
        """
        suggestions = {}
        
        # Verificar dados suficientes
        if not tenant_data or len(tenant_data) < 2:
            return {"error": "Precisa de pelo menos dois inquilinos para comparação"}
            
        # Calcular estatísticas para cada inquilino
        tenant_stats = {}
        
        for tenant, data in tenant_data.items():
            try:
                numeric_data = pd.to_numeric(data, errors='coerce').dropna()
                tenant_stats[tenant] = {
                    'mean': numeric_data.mean(),
                    'median': numeric_data.median(),
                    'std': numeric_data.std(),
                    'count': len(numeric_data)
                }
            except:
                continue
        
        # Verificar diferenças entre inquilinos
        tenants = list(tenant_stats.keys())
        has_significant_diff = False
        
        # Calcular coeficiente de variação entre médias dos inquilinos
        means = [stats['mean'] for stats in tenant_stats.values() if stats['count'] > 5]
        if means:
            cv_means = np.std(means) / np.mean(means) if np.mean(means) != 0 else 0
            has_significant_diff = cv_means > 0.2  # Diferença significativa se CV > 20%
        
        # Sugerir visualizações básicas
        suggestions["tenant_boxplot"] = {
            "type": "Boxplot por Inquilino",
            "description": f"Boxplots comparando {metric_name} entre inquilinos",
            "justification": "Comparação visual da distribuição de valores entre inquilinos",
            "function": "compare_tenants",
            "priority": 1
        }
        
        suggestions["tenant_barplot"] = {
            "type": "Gráfico de Barras",
            "description": f"Comparação de médias de {metric_name} entre inquilinos com barras de erro",
            "justification": "Visualização clara das diferenças entre inquilinos com indicadores de variabilidade",
            "function": "compare_tenants",
            "priority": 2
        }
        
        # Verificar se dados são séries temporais
        is_timeseries = False
        for data in tenant_data.values():
            if isinstance(data.index, pd.DatetimeIndex):
                is_timeseries = True
                break
                
        if is_timeseries:
            suggestions["tenant_timeseries"] = {
                "type": "Séries Temporais por Inquilino",
                "description": f"Gráfico comparando evolução de {metric_name} para diferentes inquilinos",
                "justification": "Visualização da evolução temporal e comportamento relativo",
                "function": "analyze_tenant_time_series",
                "priority": 3
            }
            
        # Se houver muitos inquilinos, sugerir correlação
        if len(tenant_stats) >= 4:
            suggestions["tenant_correlation"] = {
                "type": "Matriz de Correlação entre Inquilinos",
                "description": f"Heatmap mostrando correlações de {metric_name} entre inquilinos",
                "justification": f"Identificação de padrões de comportamento similares entre os {len(tenant_stats)} inquilinos",
                "function": "correlation_between_tenants",
                "priority": 4
            }
        
        return suggestions
    
    # ========== SUGESTÕES DE TABELAS ==========
    
    def suggest_tables(self, data, context=None):
        """
        Sugere tabelas relevantes com base nos dados e contexto.
        
        Args:
            data: Dados para analisar (Series, DataFrame ou dict)
            context (dict): Informações contextuais (tipo de análise, métrica, fase, etc)
            
        Returns:
            dict: Sugestões de tabelas com justificativas
        """
        suggestions = {}
        
        # Verificar tipo de dados
        if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
            return self._suggest_tables_for_dataframe(data, context)
        elif isinstance(data, dict):
            # Check if this is phase or tenant data
            if context and context.get('analysis_type') == 'phase':
                return self._suggest_tables_for_phases(data, context)
            elif context and context.get('analysis_type') == 'tenant':
                return self._suggest_tables_for_tenants(data, context)
            else:
                return self._suggest_tables_generic(data, context)
        else:
            return {"error": "Tipo de dados não suportado para sugestão de tabelas"}
    
    def _suggest_tables_for_dataframe(self, data, context=None):
        """Sugerir tabelas para DataFrame/Series"""
        suggestions = {}
        metric_name = context.get('metric_name', 'métrica') if context else 'métrica'
        
        # Estatísticas descritivas são sempre úteis
        suggestions["descriptive_stats"] = {
            "type": "Estatísticas Descritivas",
            "description": f"Tabela com medidas estatísticas para {metric_name}",
            "justification": "Fornece visão geral dos dados com medidas de posição e dispersão",
            "function": "get_descriptive_stats",
            "formats": ["CSV", "LaTeX"],
            "priority": 1
        }
        
        # Verificar dados suficientes para mais análises
        if len(data) < 10:
            return suggestions
            
        # Verificar se é série temporal
        is_timeseries = (isinstance(data, pd.Series) and isinstance(data.index, pd.DatetimeIndex)) or \
                        (isinstance(data, pd.DataFrame) and isinstance(data.index, pd.DatetimeIndex))
        
        # Tabelas específicas para séries temporais
        if is_timeseries:
            suggestions["stationarity_test"] = {
                "type": "Testes de Estacionariedade",
                "description": f"Resultados de testes ADF e KPSS para {metric_name}",
                "justification": "Importante para análise de séries temporais e modelagem",
                "function": "check_stationarity",
                "formats": ["CSV"],
                "priority": 3
            }
        
        # Se for DataFrame com múltiplas colunas, sugerir correlação
        if isinstance(data, pd.DataFrame) and data.shape[1] > 1:
            suggestions["correlation_matrix"] = {
                "type": "Matriz de Correlação",
                "description": f"Correlações entre variáveis relacionadas a {metric_name}",
                "justification": f"Quantifica relações entre {data.shape[1]} variáveis",
                "function": "analyze_correlations",
                "formats": ["CSV", "LaTeX"],
                "priority": 2
            }
            
        return suggestions
    
    def _suggest_tables_for_phases(self, phase_data, context=None):
        """Sugerir tabelas para comparação entre fases"""
        suggestions = {}
        metric_name = context.get('metric_name', 'métrica') if context else 'métrica'
        
        suggestions["phase_comparison"] = {
            "type": "Tabela Comparativa entre Fases",
            "description": f"Estatísticas comparativas de {metric_name} para diferentes fases",
            "justification": f"Comparação direta de {len(phase_data)} fases em formato tabular",
            "function": "compare_phases",
            "parameters": {"comparison_methods": ["stats_test"]},
            "formats": ["CSV", "LaTeX"],
            "priority": 1
        }
        
        # Verificar dados suficientes para mais análises
        if len(phase_data) < 2:
            return suggestions
            
        # Verificar se temos as fases específicas para análise de recuperação
        phases = list(phase_data.keys())
        has_baseline = any("baseline" in p.lower() for p in phases)
        has_attack = any("attack" in p.lower() or "ataque" in p.lower() for p in phases)
        has_recovery = any("recovery" in p.lower() or "recuperação" in p.lower() for p in phases)
        
        if has_baseline and has_attack and has_recovery:
            suggestions["recovery_analysis"] = {
                "type": "Análise de Recuperação",
                "description": f"Métricas de impacto e recuperação para {metric_name}",
                "justification": "Quantificação do impacto do ataque e eficácia da recuperação",
                "function": "analyze_recovery",
                "formats": ["CSV", "LaTeX"],
                "priority": 2
            }
            
        return suggestions
    
    def _suggest_tables_for_tenants(self, tenant_data, context=None):
        """Sugerir tabelas para comparação entre inquilinos"""
        suggestions = {}
        metric_name = context.get('metric_name', 'métrica') if context else 'métrica'
        
        suggestions["tenant_comparison"] = {
            "type": "Tabela Comparativa entre Inquilinos",
            "description": f"Estatísticas de {metric_name} por inquilino",
            "justification": f"Comparação direta de {len(tenant_data)} inquilinos em formato tabular",
            "function": "compare_tenants",
            "formats": ["CSV", "LaTeX"],
            "priority": 1
        }
        
        # Se houver muitos inquilinos, sugerir correlação
        if len(tenant_data) >= 4:
            suggestions["tenant_correlation"] = {
                "type": "Matriz de Correlação entre Inquilinos",
                "description": f"Correlações de {metric_name} entre inquilinos",
                "justification": f"Quantificação de similaridades entre {len(tenant_data)} inquilinos",
                "function": "correlation_between_tenants",
                "formats": ["CSV", "LaTeX"],
                "priority": 2
            }
            
        # Verificar se há análise de série temporal
        has_timeseries = False
        for data in tenant_data.values():
            if isinstance(data.index, pd.DatetimeIndex):
                has_timeseries = True
                break
                
        if has_timeseries:
            suggestions["tenant_timeseries_stats"] = {
                "type": "Estacionariedade e Entropia por Inquilino",
                "description": f"Análise temporal dos dados de {metric_name} por inquilino",
                "justification": "Caracterização estatística das séries temporais por inquilino",
                "function": "analyze_tenant_time_series",
                "formats": ["CSV", "LaTeX"],
                "priority": 3
            }
            
        return suggestions
    
    def _suggest_tables_generic(self, data_dict, context=None):
        """Sugerir tabelas para dicionário genérico de dados"""
        suggestions = {}
        
        if not data_dict:
            return {"error": "Sem dados para sugestão de tabelas"}
            
        suggestions["summary_table"] = {
            "type": "Tabela de Resumo",
            "description": "Resumo estatístico dos dados",
            "justification": "Visão geral consolidada dos principais dados",
            "function": "generate_summary_table",
            "formats": ["CSV", "LaTeX"],
            "priority": 1
        }
        
        return suggestions

    # ========== GERAÇÃO DE RELATÓRIO DE SUGESTÕES ==========
    
    def generate_suggestion_report(self, all_suggestions, output_file=None):
        """
        Gera um relatório consolidado de todas as sugestões.
        
        Args:
            all_suggestions (dict): Dicionário com todas as sugestões organizadas por categoria
            output_file (str): Nome do arquivo de saída (opcional)
            
        Returns:
            str: Caminho do arquivo de relatório gerado
        """
        if not self.suggestions_dir:
            return None
            
        if output_file is None:
            output_file = "suggestion_report.txt"
            
        report_path = self.suggestions_dir / output_file
        
        # Preparar relatório
        report_lines = [
            "=============================================================",
            "           RELATÓRIO DE SUGESTÕES DE ANÁLISE",
            "=============================================================\n",
            "Este relatório contém sugestões automatizadas para visualizações",
            "e tabelas com base nas características dos dados analisados.\n"
        ]
        
        # Organizar sugestões por categoria e prioridade
        for category, suggestions in all_suggestions.items():
            report_lines.append(f"\n{'=' * 60}")
            report_lines.append(f"CATEGORIA: {category}")
            report_lines.append(f"{'=' * 60}\n")
            
            # Ordenar por prioridade
            items = []
            for key, data in suggestions.items():
                if key != "error":
                    items.append((key, data))
            
            items.sort(key=lambda x: x[1].get('priority', 999))
            
            for key, data in items:
                report_lines.append(f"* {data['type']}")
                report_lines.append(f"  Descrição: {data['description']}")
                report_lines.append(f"  Justificativa: {data['justification']}")
                
                if 'function' in data:
                    report_lines.append(f"  Função: {data['function']}")
                    
                if 'parameters' in data:
                    params = ', '.join(f"{k}={v}" for k, v in data['parameters'].items())
                    report_lines.append(f"  Parâmetros: {params}")
                    
                if 'formats' in data:
                    report_lines.append(f"  Formatos: {', '.join(data['formats'])}")
                    
                report_lines.append("")
                
        # Escrever relatório
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
            
        logging.info(f"Relatório de sugestões salvo em: {report_path}")
        return report_path
