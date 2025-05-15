#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Análise Causal Integrado para K8s Noisy Neighbors Lab

Este módulo implementa métodos avançados de análise de causalidade
para detectar relações de causa-efeito entre métricas do Kubernetes,
integrado com a estrutura modular do pipeline completo.

Inclui implementações de:
- Causalidade de Toda-Yamamoto (extensão de Granger para séries não-estacionárias)
- Transfer Entropy (baseado em teoria da informação para relações não-lineares)
- Change Point Impact Analysis (para detectar sequência de impactos através de pontos de mudança)

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


class CausalAnalysisIntegrated:
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
        
        logging.info(f"Inicializado CausalAnalysisIntegrated com diretório de saída: {self.output_dir}")

    def determine_optimal_lag(self, x_series, y_series, max_lag=10):
        """
        Determina o lag ideal para análise de séries temporais usando AIC.
        
        Args:
            x_series: Primeira série temporal
            y_series: Segunda série temporal
            max_lag: Número máximo de defasagens a considerar
            
        Returns:
            int: Defasagem ótima
        """
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
            x_series: Série temporal explicativa (possível causa)
            y_series: Série temporal resposta (possível efeito)
            max_lag: Máximo de defasagens a considerar
            alpha: Nível de significância
            
        Returns:
            dict: Resultados da análise de causalidade incluindo estatística do teste,
                  p-valor, e interpretação
        """
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
                'x_causes_y': x_causes_y,
                'p_value': float(p_value),
                'test_statistic': float(f_statistic),
                'lag_order': int(optimal_lag),
                'max_integration_order': int(max_integration_order),
                'critical_value': float(critical_value)
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

    def transfer_entropy_test(self, source, target, k=1, significance_level=0.05, n_shuffles=100):
        """
        Análise de causalidade usando Transfer Entropy, uma abordagem baseada em teoria da informação
        que pode detectar relações não-lineares.
        
        Args:
            source: Série temporal da fonte (possível causa)
            target: Série temporal do alvo (possível efeito)
            k: Histórico a considerar (valor padrão: 1)
            significance_level: Nível de significância para o teste
            n_shuffles: Número de embaralhamentos para teste de significância
            
        Returns:
            dict: Resultados da análise de causalidade
        """
        # Validar dados de entrada
        if source.isnull().any() or target.isnull().any():
            logging.warning("Séries contêm valores NaN. Realizando limpeza de dados.")
            source = source.interpolate(method='linear').ffill().bfill()
            target = target.interpolate(method='linear').ffill().bfill()
        
        # Verificar tamanho da série
        if len(source) <= (k*3) or len(target) <= (k*3):
            logging.warning("Séries muito curtas para análise de Transfer Entropy")
            return {
                'method': 'Transfer Entropy',
                'source_causes_target': False,
                'transfer_entropy': None,
                'p_value': None,
                'k_history': k,
                'error': 'Série temporal muito curta para análise'
            }
            
        try:
            # Normalizar os dados para terem a mesma escala (importante para entropy)
            source_norm = (source - source.mean()) / source.std()
            target_norm = (target - target.mean()) / target.std()
            
            # Discretizar os dados para análise de entropia
            # Método: converter para quantis e depois para inteiros (8 bins)
            bins = 8
            source_disc = pd.qcut(source_norm, bins, labels=False, duplicates='drop').fillna(0).astype(int)
            target_disc = pd.qcut(target_norm, bins, labels=False, duplicates='drop').fillna(0).astype(int)
            
            # Calcular Transfer Entropy
            if PYINFORM_AVAILABLE:
                # Usar pyinform para cálculo exato
                te_value = transfer_entropy(source_disc.values, target_disc.values, k=k)
            else:
                # Implementação alternativa baseada em entropia condicional
                te_value = self._calculate_transfer_entropy(source_disc.values, target_disc.values, k)
            
            # Teste de significância usando surrogate data
            # Repetidamente embaralhar a série fonte para gerar distribuição nula
            te_shuffled_values = []
            for _ in range(n_shuffles):
                # Criar série temporal embaralhada
                source_shuffled = np.random.permutation(source_disc.values)
                
                if PYINFORM_AVAILABLE:
                    te_shuffled = transfer_entropy(source_shuffled, target_disc.values, k=k)
                else:
                    te_shuffled = self._calculate_transfer_entropy(source_shuffled, target_disc.values, k)
                    
                te_shuffled_values.append(te_shuffled)
                
            # Calcular p-valor empírico
            # (proporção de valores embaralhados com TE maior ou igual ao observado)
            p_value = sum(te_shuffled >= te_value for te_shuffled in te_shuffled_values) / n_shuffles
            
            # Interpretar resultado
            source_causes_target = p_value < significance_level
            
            # Resultados
            result = {
                'method': 'Transfer Entropy',
                'source_causes_target': source_causes_target,
                'transfer_entropy': float(te_value),
                'p_value': float(p_value),
                'k_history': k
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Erro na análise de Transfer Entropy: {str(e)}")
            return {
                'method': 'Transfer Entropy',
                'error': f"Falha no processamento: {str(e)}",
                'source_causes_target': False,
                'transfer_entropy': None,
                'p_value': None,
                'k_history': k
            }
    
    def _calculate_transfer_entropy(self, source, target, k=1):
        """
        Implementação alternativa de Transfer Entropy usando entropia condicional.
        
        Args:
            source: Dados fonte (discretizados)
            target: Dados alvo (discretizados)
            k: Histórico a considerar
            
        Returns:
            float: Valor de Transfer Entropy
        """
        # Criar arrays defasados
        target_future = target[k:]
        target_past = np.array([target[i:-(k-i) if k-i > 0 else None] for i in range(k)]).T
        source_past = np.array([source[i:-(k-i+1) if k-i+1 > 0 else None] for i in range(k)]).T
        
        # Calcular entropia usando frequências empíricas
        H_target_future_target_past = self._entropy_joint(
            np.column_stack((target_future.reshape(-1, 1), target_past))
        )
        
        H_target_past = self._entropy(target_past)
        
        H_target_future_target_past_source_past = self._entropy_joint(
            np.column_stack((target_future.reshape(-1, 1), target_past, source_past))
        )
        
        H_target_past_source_past = self._entropy_joint(
            np.column_stack((target_past, source_past))
        )
        
        # Transfer Entropy = H(target_future|target_past) - H(target_future|target_past, source_past)
        # = H(target_future,target_past) - H(target_past) - H(target_future,target_past,source_past) + H(target_past,source_past)
        return (H_target_future_target_past - H_target_past - 
                H_target_future_target_past_source_past + H_target_past_source_past)
    
    def _entropy(self, data):
        """Calcula entropia de Shannon de dados discretizados."""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        # Converter para tuplas para contagem
        data_tuples = [tuple(row) for row in data]
        
        # Contar frequências
        n = len(data_tuples)
        counts = {}
        for item in data_tuples:
            counts[item] = counts.get(item, 0) + 1
            
        # Calcular entropia
        entropy_val = 0
        for count in counts.values():
            p = count / n
            entropy_val -= p * np.log2(p)
            
        return entropy_val
    
    def _entropy_joint(self, data):
        """Calcula entropia conjunta dos dados."""
        return self._entropy(data)

    def change_point_impact_analysis(self, source_metric, target_metrics, window_size=10, penalty=10):
        """
        Análise de impacto baseada em detecção de pontos de mudança.
        Detecta pontos de mudança na métrica fonte e verifica se causam
        mudanças nas métricas alvo.
        
        Args:
            source_metric: Série temporal da potencial fonte de impacto
            target_metrics: Dicionário de séries temporais de alvos potenciais
            window_size: Janela de tempo para buscar impactos (em pontos de dados)
            penalty: Penalidade para o algoritmo de detecção de pontos de mudança
            
        Returns:
            dict: Resultados da análise, incluindo pontos de mudança e impactos
        """
        # Verificar se temos a biblioteca de detecção de pontos de mudança
        if not RUPTURES_AVAILABLE:
            logging.error("Biblioteca ruptures não disponível para detecção de pontos de mudança")
            return {
                'method': 'Change Point Impact',
                'error': 'Biblioteca ruptures não disponível',
                'source_change_points': [],
                'detailed_results': pd.DataFrame()
            }
        
        # Validar dados de entrada
        if source_metric.isnull().any():
            logging.warning("Série fonte contém NaN. Realizando interpolação.")
            source_metric = source_metric.interpolate(method='linear').ffill().bfill()
        
        if len(source_metric) < window_size*3:
            logging.warning("Série temporal muito curta para análise de pontos de mudança")
            return {
                'method': 'Change Point Impact',
                'error': 'Série temporal muito curta',
                'source_change_points': [],
                'detailed_results': pd.DataFrame()
            }
            
        try:
            # Detectar pontos de mudança na série fonte
            source_data = source_metric.values
            
            # Algoritmo de detecção de pontos de mudança (PELT)
            algo = rpt.Pelt(model="l2").fit(source_data)
            source_change_points = algo.predict(pen=penalty)
            
            # Filtrar pontos de mudança muito próximos do início/fim
            min_distance = max(window_size // 2, 3)
            filtered_change_points = [cp for cp in source_change_points 
                                     if cp > min_distance and cp < len(source_data) - min_distance]
            
            # Se não encontramos pontos de mudança, retornar resultado vazio
            if not filtered_change_points:
                return {
                    'method': 'Change Point Impact',
                    'warning': 'Nenhum ponto de mudança significativo detectado',
                    'source_change_points': [],
                    'detailed_results': pd.DataFrame()
                }
            
            # Analisar impacto em cada métrica alvo
            results = []
            
            for target_name, target_series in target_metrics.items():
                # Pular se a série alvo for idêntica à fonte
                if target_series.equals(source_metric):
                    continue
                    
                # Interpolar valores faltantes
                if target_series.isnull().any():
                    target_series = target_series.interpolate(method='linear').ffill().bfill()
                
                # Se a série for muito curta, pular
                if len(target_series) < window_size*3:
                    continue
                
                # Detectar pontos de mudança na série alvo
                target_data = target_series.values
                algo = rpt.Pelt(model="l2").fit(target_data)
                target_change_points = algo.predict(pen=penalty)
                
                # Para cada ponto de mudança na fonte, verificar se causou mudança no alvo
                for source_cp in filtered_change_points:
                    # Definir janela de busca no alvo
                    search_start = max(0, source_cp)
                    search_end = min(len(target_data), source_cp + window_size)
                    
                    # Encontrar ponto de mudança mais próximo na janela
                    closest_target_cp = None
                    min_distance = float('inf')
                    
                    for tcp in target_change_points:
                        # Verificar se o ponto de mudança do alvo está na janela de busca
                        if search_start <= tcp <= search_end:
                            distance = tcp - source_cp
                            if distance >= 0 and distance < min_distance:
                                min_distance = distance
                                closest_target_cp = tcp
                    
                    # Se encontramos um ponto de mudança no alvo, calcular força do impacto
                    if closest_target_cp is not None:
                        impact_strength = self._calculate_impact_strength(
                            target_data, closest_target_cp, source_cp
                        )
                        
                        significant = impact_strength > 0.3  # Impacto significativo
                        
                        results.append({
                            'source_change_point': int(source_cp),
                            'target_metric': target_name,
                            'target_change_point': int(closest_target_cp),
                            'lag': int(closest_target_cp - source_cp),
                            'impact_strength': float(impact_strength),
                            'significant': bool(significant)
                        })
            
            # Converter resultados para DataFrame
            detailed_results = pd.DataFrame(results) if results else pd.DataFrame()
            
            # Salvar resultados em formato CSV se houver diretório de saída
            if self.output_dir and not detailed_results.empty:
                detailed_results.to_csv(self.results_dir / 'change_point_impact.csv', index=False)
                
                # Também salvar versão para LaTeX
                with open(self.tables_dir / 'change_point_impact.tex', 'w') as f:
                    f.write(detailed_results[['source_change_point', 'target_metric', 
                                            'target_change_point', 'lag', 
                                            'impact_strength', 'significant']].to_latex(index=False))
            
            return {
                'method': 'Change Point Impact',
                'source_change_points': [int(cp) for cp in filtered_change_points],
                'detailed_results': detailed_results
            }
            
        except Exception as e:
            logging.error(f"Erro na análise de impacto por pontos de mudança: {str(e)}")
            return {
                'method': 'Change Point Impact',
                'error': f"Falha no processamento: {str(e)}",
                'source_change_points': [],
                'detailed_results': pd.DataFrame()
            }
    
    def _calculate_impact_strength(self, series, change_point, source_point, window_before=5, window_after=5):
        """
        Calcula a força do impacto de uma mudança comparando janelas antes e depois.
        
        Args:
            series: Série temporal
            change_point: Ponto de mudança na série
            source_point: Ponto de mudança na fonte que potencialmente causou este ponto
            window_before: Tamanho da janela antes do ponto de mudança
            window_after: Tamanho da janela depois do ponto de mudança
            
        Returns:
            float: Magnitude do impacto normalizada [0-1]
        """
        # Ajustar janelas se próximo do início/fim
        window_before = min(window_before, change_point)
        window_after = min(window_after, len(series) - change_point - 1)
        
        # Se janelas muito pequenas, retorna impacto zero
        if window_before < 2 or window_after < 2:
            return 0.0
            
        # Extrair dados das janelas
        before_data = series[change_point - window_before:change_point]
        after_data = series[change_point:change_point + window_after]
        
        # Calcular estatísticas antes/depois
        before_mean = np.mean(before_data)
        after_mean = np.mean(after_data)
        
        # Usar desvio padrão combinado para normalização
        pooled_std = np.sqrt(
            ((window_before - 1) * np.std(before_data)**2 + 
             (window_after - 1) * np.std(after_data)**2) / 
            (window_before + window_after - 2)
        )
        
        # Evitar divisão por zero
        if pooled_std == 0:
            pooled_std = 1e-6
            
        # Calcular tamanho do efeito (similar a Cohen's d)
        effect_size = abs(after_mean - before_mean) / pooled_std
        
        # Normalizar para [0-1] usando função logística
        impact_strength = 2 / (1 + np.exp(-0.5 * effect_size)) - 1
        
        return impact_strength

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
                    # Assegurar que todos os valores são primitivos (não Series/DataFrames)
                    result_dict = {
                        'phase': phase_name,
                        'source_metric': source_metric_id,
                        'target_metric': target_metric_id,
                        'method': 'Toda-Yamamoto'
                    }
                    
                    # Extrair e verificar cada valor
                    for key in ['test_statistic', 'p_value', 'lag_order']:
                        value = forward_result.get(key)
                        if isinstance(value, (pd.Series, pd.DataFrame)):
                            # Converter Series/DataFrame para valor único se possível
                            if len(value) == 1:
                                result_dict[key] = value.iloc[0]
                            else:
                                result_dict[key] = float(value.mean()) if key != 'lag_order' else int(value.mean())
                        else:
                            result_dict[key] = value
                    
                    # Converter causality para boolean primitivo
                    causality = forward_result.get('x_causes_y')
                    if isinstance(causality, (pd.Series, pd.DataFrame)):
                        if len(causality) == 1:
                            result_dict['causality'] = bool(causality.iloc[0])
                        else:
                            result_dict['causality'] = bool(causality.any())
                    else:
                        result_dict['causality'] = bool(causality)
                        
                    results.append(result_dict)
                
                if isinstance(reverse_result, dict) and 'error' not in reverse_result:
                    # Assegurar que todos os valores são primitivos (não Series/DataFrames)
                    result_dict = {
                        'phase': phase_name,
                        'source_metric': target_metric_id,
                        'target_metric': source_metric_id,
                        'method': 'Toda-Yamamoto'
                    }
                    
                    # Extrair e verificar cada valor
                    for key in ['test_statistic', 'p_value', 'lag_order']:
                        value = reverse_result.get(key)
                        if isinstance(value, (pd.Series, pd.DataFrame)):
                            # Converter Series/DataFrame para valor único se possível
                            if len(value) == 1:
                                result_dict[key] = value.iloc[0]
                            else:
                                result_dict[key] = float(value.mean()) if key != 'lag_order' else int(value.mean())
                        else:
                            result_dict[key] = value
                    
                    # Converter causality para boolean primitivo
                    causality = reverse_result.get('x_causes_y')
                    if isinstance(causality, (pd.Series, pd.DataFrame)):
                        if len(causality) == 1:
                            result_dict['causality'] = bool(causality.iloc[0])
                        else:
                            result_dict['causality'] = bool(causality.any())
                    else:
                        result_dict['causality'] = bool(causality)
                        
                    results.append(result_dict)
                                
                        elif method == 'transfer-entropy':
                            # Teste na direção source -> target
                            forward_result = self.transfer_entropy_test(
                                source_series, target_series
                            )
                            
                            # Teste na direção target -> source (bidirecional)
                            reverse_result = self.transfer_entropy_test(
                                target_series, source_series
                            )
                            
                            # Adicionar resultados
                            if 'error' not in forward_result:
                                results.append({
                                    'phase': phase_name,
                                    'source_metric': source_metric_id,
                                    'target_metric': target_metric_id,
                                    'method': 'Transfer Entropy',
                                    'transfer_entropy': forward_result.get('transfer_entropy'),
                                    'p_value': forward_result.get('p_value'),
                                    'causality': forward_result.get('source_causes_target'),
                                    'k_history': forward_result.get('k_history')
                                })
                            
                            if 'error' not in reverse_result:
                                results.append({
                                    'phase': phase_name,
                                    'source_metric': target_metric_id,
                                    'target_metric': source_metric_id,
                                    'method': 'Transfer Entropy',
                                    'transfer_entropy': reverse_result.get('transfer_entropy'),
                                    'p_value': reverse_result.get('p_value'),
                                    'causality': reverse_result.get('source_causes_target'),
                                    'k_history': reverse_result.get('k_history')
                                })
                
                # Análise de impacto de pontos de mudança - tratada separadamente pois analisa múltiplos alvos de uma vez
                if method == 'change-point-impact':
                    for source_metric_id, source_series in phase_metrics.items():
                        # Coletar todas as outras métricas como possíveis alvos
                        target_dict = {mid: phase_metrics[mid] for mid in phase_metrics 
                                      if mid != source_metric_id}
                        
                        # Realizar análise de impacto
                        impact_results = self.change_point_impact_analysis(
                            source_series, target_dict
                        )
                        
                        # Processar resultados
                        if isinstance(impact_results, dict) and 'detailed_results' in impact_results:
                            detailed_df = impact_results['detailed_results']
                            
                            for _, row in detailed_df.iterrows():
                                results.append({
                                    'phase': phase_name,
                                    'source_metric': source_metric_id,
                                    'target_metric': row['target_metric'],
                                    'method': 'Change Point Impact',
                                    'source_change_point': row['source_change_point'],
                                    'target_change_point': row['target_change_point'],
                                    'lag': row['lag'],
                                    'impact_strength': row['impact_strength'],
                                    'causality': row['significant']
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
                elif method == 'transfer-entropy':
                    latex_cols = ['phase', 'source_metric', 'target_metric', 'transfer_entropy', 'p_value', 'causality']
                else:  # change-point-impact
                    latex_cols = ['phase', 'source_metric', 'target_metric', 'lag', 'impact_strength', 'causality']
                
                # Filtrar colunas que existem
                latex_cols = [col for col in latex_cols if col in results_df.columns]
                
                # Salvar tabela LaTeX
                if latex_cols:
                    with open(self.results_dir / latex_filename, 'w') as f:
                        f.write(results_df[latex_cols].to_latex(index=False))
                        
                # Salvar gráfico se tivermos causas significativas
                if not results_df.empty and 'causality' in results_df.columns:
                    significant_results = results_df[results_df['causality'] == True]
                    
                    if len(significant_results) > 0:
                        self._plot_causal_graph(significant_results, method, phase_name)
            
            return results_df
            
        except Exception as e:
            logging.error(f"Erro na análise causal: {str(e)}")
            return pd.DataFrame()
    
    def _plot_causal_graph(self, causal_df, method, phase=None):
        """
        Gera gráfico de rede causal baseado nos resultados da análise.
        
        Args:
            causal_df: DataFrame com resultados de causalidade
            method: Método usado para análise causal
            phase: Fase específica para filtrar (opcional)
        """
        if causal_df.empty:
            return
            
        # Filtrar para fase específica se fornecida
        if phase and 'phase' in causal_df.columns:
            phase_data = causal_df[causal_df['phase'] == phase]
            if phase_data.empty:
                return
        else:
            phase_data = causal_df
        
        # Criar gráfico direcionado
        G = nx.DiGraph()
        
        # Adicionar nós e arestas
        edge_labels = {}
        
        for _, row in phase_data.iterrows():
            source = row['source_metric']
            target = row['target_metric']
            
            # Adicionar nós
            if source not in G:
                G.add_node(source)
            if target not in G:
                G.add_node(target)
            
            # Adicionar aresta
            weight = None
            if 'p_value' in row:
                weight = 1 - row['p_value']  # Converter p-valor em peso (maior = mais significativo)
                label = f"p={row['p_value']:.3f}"
            elif 'transfer_entropy' in row:
                weight = row['transfer_entropy']
                label = f"TE={weight:.3f}"
            elif 'impact_strength' in row:
                weight = row['impact_strength']
                label = f"IS={weight:.3f}"
            else:
                weight = 0.5
                label = "causal"
            
            G.add_edge(source, target, weight=weight)
            edge_labels[(source, target)] = label
        
        # Calcular medidas de centralidade
        try:
            centrality = nx.betweenness_centrality(G)
            # Normalizar centralidade para tamanho dos nós
            max_centrality = max(centrality.values()) if centrality else 1
            node_sizes = {node: (c / max_centrality) * 2000 + 500 for node, c in centrality.items()}
        except:
            # Fallback para tamanhos iguais em caso de erro
            node_sizes = {node: 1000 for node in G.nodes()}
        
        # Criar figura
        plt.figure(figsize=(12, 10))
        
        # Definir layout
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # Desenhar nós
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=[node_sizes[node] for node in G.nodes()],
            node_color='skyblue',
            alpha=0.8
        )
        
        # Desenhar arestas com largura baseada em peso
        edge_widths = [G[u][v]['weight'] * 2 + 0.5 for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=0.7,
            edge_color='gray',
            arrowsize=20
        )
        
        # Adicionar rótulos
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_family='sans-serif'
        )
        
        # Adicionar rótulos nas arestas
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8
        )
        
        # Adicionar título
        phase_str = f" na fase {phase}" if phase else ""
        plt.title(f"Gráfico de Causalidade ({method}){phase_str}", size=14)
        
        # Remover eixos
        plt.axis('off')
        
        # Salvar figura
        method_str = method.replace('-', '_')
        phase_filename = f"_{phase.replace(' ', '_')}" if phase else ""
        filename = f"causal_graph_{method_str}{phase_filename}.png"
        
        if self.plots_dir:
            plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches='tight')
            logging.info(f"Gráfico causal salvo em: {self.plots_dir / filename}")
            
            # Também salvar em PDF para publicações
            pdf_filename = f"causal_graph_{method_str}{phase_filename}.pdf"
            plt.savefig(self.plots_dir / pdf_filename, format='pdf', bbox_inches='tight')
        
        plt.close()


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
    
    print(f"Módulo de análise causal integrada carregado. Método selecionado: {args.method}")
    print("Para usar, importe este módulo em seu script principal.")
    
    sys.exit(0)
