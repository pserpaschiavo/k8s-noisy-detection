#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Análise Causal Integrado (Corrigido) para K8s Noisy Neighbors Lab

Este módulo implementa métodos avançados de análise de causalidade
para detectar relações de causa-efeito entre métricas do Kubernetes,
integrado com a estrutura modular do pipeline completo.

Autor: P. S. Schiavo
Data: 14/05/2025
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import json
import networkx as nx

# Tente importar bibliotecas opcionais
try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    logging.warning("Biblioteca 'ruptures' não disponível. Change Point Impact Analysis será limitada.")

try:
    from pyinform import transfer_entropy
    PYINFORM_AVAILABLE = True
except ImportError:
    PYINFORM_AVAILABLE = False
    logging.warning("Biblioteca 'pyinform' não disponível. Transfer Entropy usará implementação alternativa.")


class CausalAnalysisFixed:
    """
    Classe integrada para análise causal entre métricas de sistema do Kubernetes.
    Implementa métodos avançados para detectar relações de causa-efeito,
    com integração completa com o pipeline de análise.
    """
    
    def __init__(self, output_dir=None):
        """
        Inicializa o analisador causal com diretório de saída opcional.
        
        Args:
            output_dir: Diretório para salvar resultados de análise (Path ou string)
        """
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.results_dir = self.output_dir / 'causal_analysis'
            self.results_dir.mkdir(exist_ok=True)
            
            self.plots_dir = self.output_dir / 'causal_plots'
            self.plots_dir.mkdir(exist_ok=True)
            
            self.tables_dir = self.output_dir / 'causal_tables'
            self.tables_dir.mkdir(exist_ok=True)
        
        # Configurar estilo de visualização
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 12
        
        logging.info(f"Inicializado CausalAnalysisFixed com diretório de saída: {self.output_dir}")

    def determine_optimal_lag(self, x_series, y_series, max_lag=10):
        """
        Determina o lag ideal para análise de séries temporais usando AIC.
        
        Args:
            x_series: Primeira série temporal (DataFrame ou Series)
            y_series: Segunda série temporal (DataFrame ou Series)
            max_lag: Número máximo de defasagens a considerar
            
        Returns:
            int: Defasagem ótima
        """
        # Converter DataFrames para Series se necessário
        if isinstance(x_series, pd.DataFrame):
            x_series = x_series['value'] if 'value' in x_series.columns else x_series.iloc[:, 0]
        
        if isinstance(y_series, pd.DataFrame):
            y_series = y_series['value'] if 'value' in y_series.columns else y_series.iloc[:, 0]
            
        # Remover NaN para análise de lag
        x_clean = x_series.dropna().values
        y_clean = y_series.dropna().values
        
        # Limitar ao tamanho da série mais curta
        min_length = min(len(x_clean), len(y_clean))
        if min_length <= max_lag:
            return 1  # Retorna lag mínimo se dados insuficientes
            
        x_clean = x_clean[:min_length]
        y_clean = y_clean[:min_length]
        
        # Calcular AIC para diferentes lags
        aic_values = []
        
        # O mínimo lag considerado é 1
        max_lag_to_use = min(max_lag, min_length // 5)  # Não usar mais que 20% do tamanho da série
        
        for lag in range(1, max_lag_to_use + 1):
            # Construir modelo autoregressivo
            y_lags = sm.tsa.lagmat(y_clean, lag)
            x_lags = sm.tsa.lagmat(x_clean, lag)
            
            # Truncar para igualar tamanhos
            data_length = len(y_clean) - lag
            X = np.column_stack((y_lags[:data_length, :], x_lags[:data_length, :]))
            y = y_clean[lag:]
            
            # Adicionar constante
            X = sm.add_constant(X)
            
            # Ajustar modelo e obter AIC
            try:
                model = sm.OLS(y, X).fit()
                aic_values.append(model.aic)
            except:
                aic_values.append(np.inf)
        
        # Retornar lag com menor AIC
        if not aic_values or np.all(np.isinf(aic_values)):
            return 1  # Default para casos problemáticos
            
        return np.argmin(aic_values) + 1  # +1 porque começamos de lag=1

    def toda_yamamoto_causality_test(self, x_series, y_series, max_lag=5, alpha=0.05):
        """
        Implementa o teste de causalidade de Toda-Yamamoto para séries não-estacionárias.
        
        Args:
            x_series: Série temporal explicativa (possível causa) - DataFrame ou Series
            y_series: Série temporal resposta (possível efeito) - DataFrame ou Series
            max_lag: Máximo de defasagens a considerar
            alpha: Nível de significância
            
        Returns:
            dict: Resultados da análise de causalidade incluindo estatística do teste,
                  p-valor, e interpretação
        """
        # Converter DataFrames para Series se necessário
        if isinstance(x_series, pd.DataFrame):
            x_series = x_series['value'] if 'value' in x_series.columns else x_series.iloc[:, 0]
        
        if isinstance(y_series, pd.DataFrame):
            y_series = y_series['value'] if 'value' in y_series.columns else y_series.iloc[:, 0]
            
        # Validar dados de entrada
        if x_series.isnull().any() or y_series.isnull().any():
            logging.warning("Séries contêm valores NaN. Realizando limpeza de dados.")
            # Interpolar valores NaN para viabilizar análise
            x_series = x_series.interpolate(method='linear').ffill().bfill()
            y_series = y_series.interpolate(method='linear').ffill().bfill()
        
        # Verificar tamanho da série
        if len(x_series) <= max_lag + 3 or len(y_series) <= max_lag + 3:
            logging.warning("Séries muito curtas para teste de Toda-Yamamoto")
            return {
                'method': 'Toda-Yamamoto',
                'x_causes_y': False,
                'p_value': None,
                'test_statistic': None,
                'lag_order': None,
                'max_integration_order': None,
                'critical_value': None,
                'error': 'Série temporal muito curta para análise'
            }
        
        try:
            # Determinar ordem de integração (d) com teste ADF
            x_adf = adfuller(x_series.dropna())
            y_adf = adfuller(y_series.dropna())
            
            # Se p-valor > 0.05, série não é estacionária
            x_integration_order = 1 if x_adf[1] > 0.05 else 0
            y_integration_order = 1 if y_adf[1] > 0.05 else 0
            
            # Máxima ordem de integração entre as séries
            max_integration_order = max(x_integration_order, y_integration_order)
            
            # Determinar defasagem ótima usando AIC
            optimal_lag = self.determine_optimal_lag(x_series, y_series, max_lag)
            
            # Ajustar modelo VAR com lag adicional
            total_lag = optimal_lag + max_integration_order
            
            # Construir matrizes de defasagens para x e y
            x_values = x_series.values
            y_values = y_series.values
            
            y_lags = sm.tsa.lagmat(y_values, total_lag)
            x_lags = sm.tsa.lagmat(x_values, total_lag)
            
            # Truncar para combinar dimensões
            data_length = len(y_values) - total_lag
            y_dependent = y_values[total_lag:]
            
            # Modelo irrestrito (inclui todas as defasagens de x e y)
            X_unrestricted = np.column_stack((
                np.ones(data_length),  # Constante
                y_lags[:data_length, :],  # Todas as defasagens de y
                x_lags[:data_length, :]   # Todas as defasagens de x
            ))
            
            # Modelo restrito (exclui defasagens de x até optimal_lag)
            X_restricted = np.column_stack((
                np.ones(data_length),  # Constante
                y_lags[:data_length, :],  # Todas as defasagens de y
                x_lags[:data_length, optimal_lag:]  # Apenas defasagens adicionais de x
            ))
            
            # Ajustar modelos
            unrestricted_model = sm.OLS(y_dependent, X_unrestricted).fit()
            restricted_model = sm.OLS(y_dependent, X_restricted).fit()
            
            # Número total de coeficientes: 1 (constante) + total_lag (y) + total_lag (x)
            n_coef = 1 + total_lag + total_lag
            
            # Matriz de restrição R: para cada defasagem de x até optimal_lag
            R = np.zeros((optimal_lag, n_coef))
            for i in range(optimal_lag):
                # A posição dos coeficientes de x começa após a constante e defasagens de y
                pos = 1 + total_lag + i
                R[i, pos] = 1
            
            # Vetor q de zeros (testamos se os coeficientes = 0)
            q = np.zeros(optimal_lag)
            
            # Calcular estatística de teste F manualmente
            ssr_restricted = restricted_model.ssr
            ssr_unrestricted = unrestricted_model.ssr
            
            df1 = optimal_lag  # número de restrições
            df2 = data_length - n_coef  # graus de liberdade do modelo irrestrito
            
            # Estatística F = ((SSR_r - SSR_ur)/df1) / (SSR_ur/df2)
            f_statistic = ((ssr_restricted - ssr_unrestricted) / df1) / (ssr_unrestricted / df2)
            
            # p-valor
            p_value = 1 - stats.f.cdf(f_statistic, df1, df2)
            
            # Calcular valor crítico baseado no nível de significância
            critical_value = stats.f.ppf(1-alpha, optimal_lag, data_length-X_unrestricted.shape[1])
            
            # Interpretação do resultado
            x_causes_y = p_value < alpha
            
            # Resultados
            result = {
                'method': 'Toda-Yamamoto',
                'x_causes_y': bool(x_causes_y),  # Garantir tipo booleano
                'p_value': float(p_value),  # Garantir tipo float
                'test_statistic': float(f_statistic),  # Garantir tipo float
                'lag_order': int(optimal_lag),  # Garantir tipo int
                'max_integration_order': int(max_integration_order),  # Garantir tipo int
                'critical_value': float(critical_value)  # Garantir tipo float
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Erro no teste de causalidade de Toda-Yamamoto: {str(e)}")
            return {
                'method': 'Toda-Yamamoto',
                'error': f"Falha no processamento: {str(e)}",
                'x_causes_y': False,
                'p_value': None,
                'test_statistic': None,
                'lag_order': None
            }

    def run_causal_analysis(self, phase_data, method='toda-yamamoto', metrics_of_interest=None, 
                          components=None, save_results=True):
        """
        Executa análise causal em dados de experimento.
        
        Args:
            phase_data: Dados hierárquicos por fase e componente
            method: Método de causalidade a usar
            metrics_of_interest: Lista de métricas específicas para analisar
            components: Lista de componentes para analisar
            save_results: Se True, salva resultados em CSV
            
        Returns:
            DataFrame: Resultados da análise causal
        """
        results = []
        
        # Coletar todos os componentes se não especificados
        all_components = set()
        
        for phase_name, phase_components in phase_data.items():
            all_components.update(phase_components.keys())
        
        components_to_analyze = components if components else list(all_components)
        
        try:
            # Para cada fase do experimento
            for phase_name, phase_components in phase_data.items():
                logging.info(f"Analisando causalidade na fase: {phase_name}")
                
                # Coletar séries temporais de métricas relevantes
                phase_metrics = {}
                
                for component_name in components_to_analyze:
                    if component_name not in phase_components:
                        continue
                        
                    component_data = phase_components[component_name]
                    
                    # Para cada métrica no componente
                    for metric_name, metric_series in component_data.items():
                        if metrics_of_interest and metric_name not in metrics_of_interest:
                            continue
                            
                        # Armazenar série com identificador completo
                        metric_id = f"{component_name}_{metric_name}"
                        phase_metrics[metric_id] = metric_series
                
                # Se menos de 2 métricas, não há como fazer análise causal
                if len(phase_metrics) < 2:
                    logging.warning(f"Insuficientes métricas para análise causal na fase {phase_name}")
                    continue
                
                # Realizar análise causal entre pares de métricas
                metric_ids = list(phase_metrics.keys())
                
                for i, source_metric_id in enumerate(metric_ids):
                    for target_metric_id in metric_ids[i+1:]:
                        source_series = phase_metrics[source_metric_id]
                        target_series = phase_metrics[target_metric_id]
                        
                        # Verificar tamanho mínimo das séries
                        if len(source_series) < 10 or len(target_series) < 10:
                            logging.warning(f"Séries muito curtas para análise: {source_metric_id}, {target_metric_id}")
                            continue
                        
                        # Aplicar o método de análise causal escolhido
                        if method == 'toda-yamamoto':
                            # Teste na direção source -> target
                            forward_result = self.toda_yamamoto_causality_test(
                                source_series, target_series
                            )
                            
                            # Teste na direção target -> source (bidirecional)
                            reverse_result = self.toda_yamamoto_causality_test(
                                target_series, source_series
                            )
                            
                            # Adicionar resultados
                            if isinstance(forward_result, dict) and 'error' not in forward_result:
                                results.append({
                                    'phase': phase_name,
                                    'source_metric': source_metric_id,
                                    'target_metric': target_metric_id,
                                    'method': 'Toda-Yamamoto',
                                    'test_statistic': forward_result.get('test_statistic'),
                                    'p_value': forward_result.get('p_value'),
                                    'causality': forward_result.get('x_causes_y'),
                                    'lag_order': forward_result.get('lag_order')
                                })
                            
                            if isinstance(reverse_result, dict) and 'error' not in reverse_result:
                                results.append({
                                    'phase': phase_name,
                                    'source_metric': target_metric_id,
                                    'target_metric': source_metric_id,
                                    'method': 'Toda-Yamamoto',
                                    'test_statistic': reverse_result.get('test_statistic'),
                                    'p_value': reverse_result.get('p_value'),
                                    'causality': reverse_result.get('x_causes_y'),
                                    'lag_order': reverse_result.get('lag_order')
                                })
            
            # Converter resultados para DataFrame
            results_df = pd.DataFrame(results)
            
            # Salvar resultados
            if save_results and not results_df.empty and self.output_dir:
                # Criar nome de arquivo baseado no método
                filename = f"causal_analysis_{method.replace('-', '_')}.csv"
                results_df.to_csv(self.results_dir / filename, index=False)
                
                # Também salvar versão LaTeX para publicações acadêmicas
                latex_filename = f"causal_analysis_{method.replace('-', '_')}.tex"
                
                # Selecionar colunas relevantes para LaTeX
                if method == 'toda-yamamoto':
                    latex_cols = ['phase', 'source_metric', 'target_metric', 'p_value', 'causality', 'lag_order']
                else:  # Outras implementações futuras
                    latex_cols = ['phase', 'source_metric', 'target_metric']
                
                # Filtrar colunas que existem
                latex_cols = [col for col in latex_cols if col in results_df.columns]
                
                # Salvar tabela LaTeX
                if latex_cols:
                    with open(self.tables_dir / latex_filename, 'w') as f:
                        f.write(results_df[latex_cols].to_latex(index=False))
            
            return results_df
            
        except Exception as e:
            logging.error(f"Erro na análise causal: {str(e)}")
            return pd.DataFrame()


# Teste básico quando executado diretamente
if __name__ == "__main__":
    import sys
    import argparse
    
    # Configuração de logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    parser = argparse.ArgumentParser(description="Análise causal integrada para métricas do K8s")
    
    parser.add_argument('--method', type=str, default='toda-yamamoto',
                      choices=['toda-yamamoto', 'transfer-entropy', 'change-point-impact'],
                      help='Método de análise causal')
                      
    parser.add_argument('--output', type=str, default=None,
                      help='Diretório para salvar resultados')
    
    args = parser.parse_args()
    
    print(f"Módulo de análise causal corrigido carregado. Método selecionado: {args.method}")
    print("Para usar, importe este módulo em seu script principal.")
    
    sys.exit(0)