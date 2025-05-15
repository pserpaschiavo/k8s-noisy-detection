#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Análise Causal para K8s Noisy Neighbors Lab

Este módulo implementa métodos alternativos de análise de causalidade
para detectar relações de causa-efeito entre métricas do Kubernetes.
Inclui implementações de:

- Causalidade de Toda-Yamamoto (extensão de Granger para séries não-estacionárias)
- Transfer Entropy (baseado em teoria da informação para relações não-lineares)
- Change Point Impact Analysis (para detectar sequência de impactos através de pontos de mudança)

Autor: P. S. Schiavo
Data: 13/05/2025
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


class CausalAnalyzer:
    """
    Classe para análise causal entre métricas de sistema do Kubernetes.
    Implementa métodos alternativos ao teste de Granger para lidar com
    as características específicas de métricas de sistema em ambientes K8s.
    """
    
    def __init__(self, output_dir=None):
        """
        Inicializa o analisador causal com diretório de saída opcional.
        
        Args:
            output_dir: Diretório para salvar resultados de análise (Path ou string)
        """
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.results_dir = self.output_dir / 'causal_results'
            self.results_dir.mkdir(exist_ok=True)
            
            self.plots_dir = self.output_dir / 'causal_plots'
            self.plots_dir.mkdir(exist_ok=True)
        
        # Configurar estilo de visualização
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 12
        
        logging.info(f"Inicializado CausalAnalyzer com diretório de saída: {self.output_dir}")

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
            
            # Calcular estatística de teste Wald para diferença entre modelos
            ssr_restricted = restricted_model.ssr
            ssr_unrestricted = unrestricted_model.ssr
            
            # Método de Toda-Yamamoto: teste de restrições nos coeficientes
            # Criar matriz de restrição adequada para teste F
            # Para testar se as defasagens de x não Granger-causam y, precisamos criar restrições
            # que fazem os coeficientes das defasagens de x (sem as defasagens adicionais) = 0
            
            # Número total de coeficientes: 1 (constante) + total_lag (y) + total_lag (x)
            n_coef = 1 + total_lag + total_lag
            
            # Matriz de restrição R: para cada defasagem de x até optimal_lag
            # Na matriz R, cada linha representa uma restrição
            # Colocamos 1 nas posições dos coeficientes das defasagens de x que queremos testar
            R = np.zeros((optimal_lag, n_coef))
            for i in range(optimal_lag):
                # A posição dos coeficientes de x começa após a constante e defasagens de y
                pos = 1 + total_lag + i
                R[i, pos] = 1
            
            # Vetor q de zeros (testamos se os coeficientes = 0)
            q = np.zeros(optimal_lag)
            
            # Teste F
            from scipy import stats as scipy_stats
            
            # Calcular estatística de teste F manualmente
            ssr_restricted = restricted_model.ssr
            ssr_unrestricted = unrestricted_model.ssr
            
            df1 = optimal_lag  # número de restrições
            df2 = data_length - n_coef  # graus de liberdade do modelo irrestrito
            
            # Estatística F = ((SSR_r - SSR_ur)/df1) / (SSR_ur/df2)
            f_statistic = ((ssr_restricted - ssr_unrestricted) / df1) / (ssr_unrestricted / df2)
            
            # p-valor
            p_value = 1 - scipy_stats.f.cdf(f_statistic, df1, df2)
            
            # Calcular valor crítico baseado no nível de significância
            critical_value = stats.f.ppf(1-alpha, optimal_lag, data_length-X_unrestricted.shape[1])
            
            # Interpretação do resultado
            x_causes_y = p_value < alpha
            
            # Resultados
            result = {
                'method': 'Toda-Yamamoto',
                'x_causes_y': x_causes_y,
                'p_value': p_value,
                'test_statistic': f_statistic,
                'lag_order': optimal_lag,
                'max_integration_order': max_integration_order,
                'critical_value': critical_value
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Erro no teste de causalidade Toda-Yamamoto: {e}")
            return {
                'method': 'Toda-Yamamoto',
                'x_causes_y': False,
                'p_value': None,
                'test_statistic': None,
                'error': str(e)
            }
    
    def transfer_entropy_test(self, source, target, k=1, significance_level=0.05, n_shuffles=100):
        """
        Calcula a entropia de transferência de source para target
        e realiza teste de significância por bootstrapping.
        
        Args:
            source: Série temporal fonte (possível causa)
            target: Série temporal alvo (possível efeito)
            k: Histórico a considerar
            significance_level: Nível alfa para o teste
            n_shuffles: Número de permutações para bootstrapping
            
        Returns:
            dict: Resultados da análise com entropia de transferência, p-valor e significância
        """
        # Verificar dados de entrada
        if source.isnull().any() or target.isnull().any():
            logging.warning("Séries contêm valores NaN. Realizando limpeza de dados.")
            source = source.interpolate(method='linear').ffill().bfill()
            target = target.interpolate(method='linear').ffill().bfill()
        
        # Verificar tamanho da série
        if len(source) < 30 or len(target) < 30:  # Requer série de tamanho razoável
            logging.warning("Séries muito curtas para análise de entropia de transferência")
            return {
                'method': 'Transfer Entropy',
                'source_causes_target': False,
                'p_value': None,
                'transfer_entropy': None,
                'error': 'Série temporal muito curta para análise'
            }
            
        try:
            # Garantir que as séries tenham o mesmo comprimento
            min_length = min(len(source), len(target))
            source_array = source.iloc[:min_length].values
            target_array = target.iloc[:min_length].values
            
            # Verificar se os arrays têm o mesmo tamanho
            if len(source_array) != len(target_array):
                logging.error(f"Erro: Arrays com tamanhos diferentes após ajuste. Source: {len(source_array)}, Target: {len(target_array)}")
                return {
                    'method': 'Transfer Entropy',
                    'source_causes_target': False,
                    'p_value': None,
                    'transfer_entropy': None,
                    'error': f'Incompatibilidade de tamanhos entre séries temporais: {len(source_array)} vs {len(target_array)}'
                }
            
            # Usar implementação da biblioteca pyinform se disponível
            if PYINFORM_AVAILABLE:
                # Normalizar e discretizar dados para cálculo de entropia
                # Normalizar para [0,1] e discretizar em 10 bins
                source_norm = (source_array - np.min(source_array)) / (np.max(source_array) - np.min(source_array) + 1e-10)
                target_norm = (target_array - np.min(target_array)) / (np.max(target_array) - np.min(target_array) + 1e-10)
                
                source_discrete = np.floor(source_norm * 9).astype(int)  # 0 a 9
                target_discrete = np.floor(target_norm * 9).astype(int)  # 0 a 9
                
                # Verificar se arrays estão vazios ou contêm só zeros
                if len(source_discrete) == 0 or len(target_discrete) == 0 or np.all(source_discrete == 0) or np.all(target_discrete == 0):
                    logging.warning("Arrays para análise de entropia contêm apenas zeros ou estão vazios após discretização.")
                    return {
                        'method': 'Transfer Entropy',
                        'source_causes_target': False,
                        'p_value': None,
                        'transfer_entropy': 0.0,
                        'error': 'Dados insuficientes para análise após normalização'
                    }
                
                # Calcular entropia de transferência original
                te_original = transfer_entropy(source_discrete, target_discrete, k=k)
                
                # Teste de significância por bootstrapping
                te_shuffled = np.zeros(n_shuffles)
                for i in range(n_shuffles):
                    # Permuta os dados da fonte para destruir qualquer causalidade
                    shuffled_source = np.random.permutation(source_discrete)
                    te_shuffled[i] = transfer_entropy(shuffled_source, target_discrete, k=k)
                
                # Calcular p-valor
                p_value = np.mean(te_shuffled >= te_original)
                
            else:
                # Implementação alternativa usando entropia condicional
                # Esta é uma aproximação simplificada quando pyinform não está disponível
                from sklearn.metrics import mutual_info_score
                from sklearn.neighbors import KernelDensity
                
                source_array = source_array.reshape(-1, 1)
                target_array = target_array.reshape(-1, 1)
                
                # Calcular informação mútua entre fonte e alvo defasado
                target_lagged = np.roll(target_array, k)
                target_lagged[:k] = target_lagged[k]  # Evitar valor circular
                
                # Estimar informação mútua usando densidade kernel
                source_kde = KernelDensity(bandwidth=0.1).fit(source_array)
                target_kde = KernelDensity(bandwidth=0.1).fit(target_array)
                
                # Aproximar entropia de transferência pela diferença de informação mútua
                joint_data = np.column_stack([source_array, target_lagged])
                mi = mutual_info_score(None, None, contingency=joint_data)
                
                te_original = mi
                
                # Teste de significância
                te_shuffled = np.zeros(n_shuffles)
                for i in range(n_shuffles):
                    # Permuta os dados da fonte
                    shuffled_source = source_array.copy()
                    np.random.shuffle(shuffled_source)
                    joint_shuffled = np.column_stack([shuffled_source, target_lagged])
                    te_shuffled[i] = mutual_info_score(None, None, contingency=joint_shuffled)
                
                # Calcular p-valor
                p_value = np.mean(te_shuffled >= te_original)
            
            # Determinar se fonte causa alvo
            source_causes_target = p_value < significance_level
            
            result = {
                'method': 'Transfer Entropy',
                'source_causes_target': source_causes_target,
                'transfer_entropy': te_original,
                'p_value': p_value,
                'k_history': k,
                'significance_threshold': significance_level,
                'interpretation': f"{'Evidência de causalidade' if source_causes_target else 'Sem evidência significativa de causalidade'}"
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Erro na análise de entropia de transferência: {e}")
            return {
                'method': 'Transfer Entropy',
                'source_causes_target': False,
                'p_value': None,
                'transfer_entropy': None,
                'error': str(e)
            }
    
    def change_point_impact_analysis(self, source_metric, target_metrics, window_size=10, penalty=10):
        """
        Analisa a propagação de pontos de mudança de uma métrica fonte para métricas alvo.
        
        Args:
            source_metric: Série temporal da métrica fonte (possível causa)
            target_metrics: Dict com nome e série temporal das métricas alvo
            window_size: Janela de tempo para considerar impacto
            penalty: Penalidade para o algoritmo PELT (menor = mais sensível)
            
        Returns:
            Dict com resultados detalhados e resumo estatístico
        """
        if not RUPTURES_AVAILABLE:
            logging.error("Biblioteca 'ruptures' é necessária para análise de impacto de pontos de mudança")
            return pd.DataFrame({
                'error': ["Biblioteca 'ruptures' não disponível"]
            })
            
        # Validar dados de entrada
        if source_metric.isnull().any():
            logging.warning("Série fonte contém valores NaN. Realizando limpeza de dados.")
            source_metric = source_metric.interpolate(method='linear').ffill().bfill()

        try:
            # Detectar pontos de mudança na métrica fonte usando algoritmo PELT (Pruned Exact Linear Time)
            # O modelo RBF (Radial Basis Function) é eficaz para mudanças em variância ou amplitude
            algo_source = rpt.Pelt(model="rbf").fit(source_metric.values.reshape(-1, 1))
            source_change_points = algo_source.predict(pen=penalty)
            
            # Alternativa: usar o algoritmo de segmentação binária
            if not source_change_points or len(source_change_points) <= 1:
                logging.info("Tentando algoritmo alternativo para detecção de pontos de mudança na fonte")
                
                # Algoritmo de mudança de ponto por segmentação binária (mais sensível a mudanças pequenas)
                algo_source = rpt.Binseg(model="rbf").fit(source_metric.values.reshape(-1, 1))
                source_change_points = algo_source.predict(n_bkps=min(5, len(source_metric) // 20))
            
            if not source_change_points or len(source_change_points) <= 1:
                logging.warning("Não foi possível detectar pontos de mudança na métrica fonte")
                
                # Último recurso: tente mudança de modelo para captar outros tipos de alterações
                algo_source = rpt.Pelt(model="l2").fit(source_metric.values.reshape(-1, 1))
                source_change_points = algo_source.predict(pen=penalty/2)  # Diminuir penalidade para aumentar sensibilidade
                
                if not source_change_points or len(source_change_points) <= 1:
                    return pd.DataFrame({
                        'warning': ["Não foram encontrados pontos de mudança significativos na métrica fonte"]
                    })
            
            # Resultados
            impact_results = []
            
            # Para cada métrica alvo, verificar pontos de mudança após os pontos da fonte
            for target_name, target_series in target_metrics.items():
                try:
                    # Limpar dados na métrica alvo
                    if target_series.isnull().any():
                        target_series = target_series.interpolate(method='linear').ffill().bfill()
                    
                    # Garantir que os comprimentos são compatíveis
                    min_length = min(len(source_metric), len(target_series))
                    if min_length != len(target_series):
                        target_series = target_series.iloc[:min_length]
                    
                    # Verificar tamanho mínimo das séries para análise
                    if len(target_series) < 10:
                        logging.warning(f"Série muito curta para detecção de pontos de mudança: {target_name}")
                        continue
                    
                    # Detectar pontos de mudança na métrica alvo com o mesmo algoritmo
                    algo_target = rpt.Pelt(model="rbf").fit(target_series.values.reshape(-1, 1))
                    target_change_points = algo_target.predict(pen=penalty)
                    
                    # Tentar algoritmo alternativo se não encontrar pontos na métrica alvo
                    if not target_change_points or len(target_change_points) <= 1:
                        logging.info(f"Tentando algoritmo alternativo para {target_name}")
                        algo_target = rpt.Binseg(model="rbf").fit(target_series.values.reshape(-1, 1))
                        target_change_points = algo_target.predict(n_bkps=min(5, len(target_series) // 20))
                        
                        if not target_change_points or len(target_change_points) <= 1:
                            logging.info(f"Não foram detectados pontos de mudança em {target_name}")
                            continue
                except Exception as e:
                    logging.warning(f"Erro ao processar pontos de mudança para {target_name}: {e}")
                    continue
                
                # Para cada ponto de mudança na fonte, procurar por impactos
                for source_cp in source_change_points:
                    try:
                        # Ignorar o último ponto que geralmente é o fim da série
                        if source_cp >= len(target_series) - 1 or source_cp >= len(source_metric) - 1:
                            continue
                            
                        # Procurar por pontos de mudança na janela de tempo após ponto de fonte
                        impacts = [cp for cp in target_change_points 
                                if source_cp < cp <= min(source_cp + window_size, len(target_series) - 1)]
                        
                        if impacts:
                            lag = min(impacts) - source_cp
                            
                            # Verificar se o lag é válido
                            if lag <= 0:
                                logging.debug(f"Ignorando lag inválido: {lag} para {target_name}")
                                continue
                            
                            # Calcular a força de impacto considerando vários fatores
                            impact_strength = self._calculate_impact_strength(
                                target_series, min(impacts), source_cp
                            )
                            
                            # Calcular correlação na janela antes e depois do ponto de mudança
                            correlation_before = self._calculate_window_correlation(
                                source_metric, target_series, source_cp, window_before=window_size
                            )
                            
                            correlation_after = self._calculate_window_correlation(
                                source_metric, target_series, source_cp, window_before=0, window_after=window_size
                            )
                            
                            # Calcular a diferença de correlação (medida adicional de impacto)
                            correlation_change = correlation_after - correlation_before
                            
                            # Verificar significância usando múltiplas métricas
                            # Um impacto é significativo se a força de impacto for alta OU houve mudança significativa de correlação
                            is_significant = (impact_strength > 0.5) or (abs(correlation_change) > 0.3)
                            
                            impact_results.append({
                                'source_change_point': source_cp,
                                'target_metric': target_name,
                                'target_change_point': min(impacts),
                                'lag': lag,
                                'impact_strength': impact_strength,
                                'correlation_before': correlation_before,
                                'correlation_after': correlation_after,
                                'correlation_change': correlation_change,
                                'significant': is_significant
                            })
                    except Exception as e:
                        logging.debug(f"Erro ao processar impacto para {target_name} no ponto {source_cp}: {e}")
                        continue
            
            # Criar DataFrame com resultados - usando construtor direto em vez de append()
            result_df = pd.DataFrame(impact_results) if impact_results else pd.DataFrame()
            
            # Se encontrou resultados, adicionar estatísticas de resumo
            if not result_df.empty:
                try:
                    # Calcular estatísticas de lag e força de impacto por métrica alvo
                    summary_stats = result_df.groupby('target_metric').agg({
                        'lag': ['mean', 'min', 'max', 'count'],
                        'impact_strength': ['mean', 'max'],
                        'correlation_change': ['mean', 'min', 'max'],
                        'significant': 'sum'
                    })
                    
                    # Calcular taxa de impactos significativos
                    summary_stats['significant_ratio'] = (
                        summary_stats[('significant', 'sum')] / summary_stats[('lag', 'count')]
                    )
                    
                    # Ordenar por taxa de impacto significativo
                    summary_stats = summary_stats.sort_values(('significant_ratio'), ascending=False)
                    
                    # Se diretório de saída estiver configurado, salva resultados
                    if self.output_dir:
                        # Salvar resultados detalhados
                        result_df.to_csv(self.results_dir / 'change_point_impacts_details.csv', index=False)
                        
                        # Converter formato multi-index para formato plano para salvar em CSV
                        rows_list = []
                        for metric, values in summary_stats.iterrows():
                            row = {'target_metric': metric}
                            for col_tuple, value in values.items():
                                col_name = f"{col_tuple[0]}_{col_tuple[1]}" if isinstance(col_tuple, tuple) else col_tuple
                                row[col_name] = value
                            rows_list.append(row)
                        flat_summary = pd.DataFrame(rows_list)
                            
                        flat_summary.to_csv(self.results_dir / 'change_point_impacts_summary.csv', index=False)
                        
                        # Criar visualização dos impactos
                        self._plot_change_point_impacts(source_metric, target_metrics, result_df)
                    
                    return {
                        'detailed_results': result_df,
                        'summary_statistics': summary_stats
                    }
                except Exception as e:
                    logging.error(f"Erro ao gerar resumo estatístico: {e}")
                    return {
                        'detailed_results': result_df,
                        'error': str(e)
                    }
            
            else:
                logging.info("Não foram encontrados impactos de pontos de mudança")
                return pd.DataFrame({
                    'info': ["Não foram encontrados impactos causais entre os pontos de mudança"]
                })
                
        except Exception as e:
            logging.error(f"Erro na análise de impacto de pontos de mudança: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return pd.DataFrame({
                'error': [f"Erro na análise: {str(e)}"]
            })
    
    def _calculate_impact_strength(self, series, change_point, source_point, window_before=5, window_after=5):
        """
        Calcula a força do impacto de um ponto de mudança usando múltiplas métricas.
        
        Args:
            series: Série temporal
            change_point: Ponto de mudança na série
            source_point: Ponto de mudança na métrica fonte
            window_before: Tamanho da janela antes do ponto
            window_after: Tamanho da janela após o ponto
            
        Returns:
            float: Medida da força do impacto (0 a 1)
        """
        try:
            # Verificar se os pontos são válidos
            if change_point < 0 or source_point < 0:
                logging.warning(f"Pontos de mudança inválidos: change_point={change_point}, source_point={source_point}")
                return 0.0
                
            # Verificar se a série tem valores adequados
            if series.isna().sum() > len(series) * 0.5:  # Mais de 50% dos valores são NaN
                logging.warning("Série tem muitos valores NaN para calcular força de impacto")
                return 0.0
                
            # Garantir que os índices estão dentro dos limites
            start_before = max(0, change_point - window_before)
            end_before = change_point
            
            start_after = change_point
            end_after = min(len(series), change_point + window_after)
            
            # Verificar se temos pontos suficientes para análise
            if end_before <= start_before or end_after <= start_after:
                logging.debug(f"Janelas de análise inválidas: before={start_before}:{end_before}, after={start_after}:{end_after}")
                return 0.0
            
            # Calcular estatísticas antes e depois do ponto de mudança
            before_window = series.iloc[start_before:end_before]
            after_window = series.iloc[start_after:end_after]
            
            # Remover NaN antes de calcular estatísticas
            before_window = before_window.dropna()
            after_window = after_window.dropna()
            
            # Garantir que temos dados suficientes
            if len(before_window) < 2 or len(after_window) < 2:
                logging.debug(f"Dados insuficientes para análise: before={len(before_window)}, after={len(after_window)}")
                return 0.0
            
            # 1. Mudança na média
            mean_before = before_window.mean()
            mean_after = after_window.mean()
            mean_change = abs(mean_after - mean_before)
            
            # 2. Mudança na variância (indicador de estabilidade)
            var_before = before_window.var() if len(before_window) > 1 else 0
            var_after = after_window.var() if len(after_window) > 1 else 0
            var_change = abs(var_after - var_before)
            
            # 3. Calcular a distância do lag como fator temporal
            # Quanto menor o lag, maior o impacto potencial
            lag_distance = max(1, change_point - source_point)  # Garantir lag mínimo de 1
            lag_factor = np.exp(-max(0, lag_distance - 1) / window_after)
            
            # 4. Calcular autocorrelação antes e depois
            try:
                from statsmodels.tsa.stattools import acf
                # Usar safe=True para evitar erros quando os dados não permitem cálculo de autocorrelação
                acf_before = acf(before_window, nlags=1, fft=True)[1] if len(before_window) > 2 else 0
                acf_after = acf(after_window, nlags=1, fft=True)[1] if len(after_window) > 2 else 0
                autocorr_change = abs(acf_after - acf_before)
            except Exception as acf_error:
                logging.debug(f"Erro ao calcular autocorrelação: {acf_error}")
                autocorr_change = 0
            
            # Calcular desvio padrão global da série para normalização
            std_dev = series.std()
            
            if pd.isna(std_dev) or std_dev < 1e-8:  # Evitar divisão por zero ou valores muito pequenos
                logging.debug("Desvio padrão muito pequeno ou NaN")
                # Usar um valor padrão pequeno mas não zero
                std_dev = 1e-8
                
            # Normalizar diferenças pela escala da série
            series_var = series.var()
            if pd.isna(series_var) or series_var < 1e-8:
                series_var = 1e-8
                
            normalized_mean_diff = mean_change / std_dev
            normalized_var_diff = var_change / series_var
            
            # Combinar métricas ponderadas 
            # - Diferença de médias (50%)
            # - Diferença de variâncias (20%)
            # - Fator de lag temporal (20%)
            # - Mudança de autocorrelação (10%)
            combined_score = (0.5 * normalized_mean_diff + 
                             0.2 * normalized_var_diff + 
                             0.2 * lag_factor + 
                             0.1 * autocorr_change)
            
            # Mapear para escala de 0 a 1 com função sigmoide
            impact_strength = 1 / (1 + np.exp(-combined_score + 1))
            
            # Registrar componentes para análise de impacto forte
            if impact_strength > 0.7:
                logging.debug(f"Impacto forte ({impact_strength:.2f}) - Componentes: mean={normalized_mean_diff:.2f}, " 
                             f"var={normalized_var_diff:.2f}, lag={lag_factor:.2f}, acf={autocorr_change:.2f}")
            
            return min(1.0, max(0.0, impact_strength))  # Garantir que está no intervalo [0,1]
            
        except Exception as e:
            logging.error(f"Erro ao calcular força de impacto: {e}")
            import traceback
            logging.debug(traceback.format_exc())
            return 0.0
    
    def _calculate_window_correlation(self, series_a, series_b, center_point, window_before=5, window_after=5):
        """
        Calcula correlação entre séries temporais em uma janela ao redor de um ponto.
        
        Args:
            series_a: Primeira série temporal
            series_b: Segunda série temporal
            center_point: Ponto central para janela
            window_before: Tamanho da janela antes do ponto
            window_after: Tamanho da janela após o ponto
            
        Returns:
            float: Correlação de Pearson entre as séries na janela especificada
        """
        try:
            # Verificar parâmetros de entrada
            if center_point < 0:
                logging.warning(f"Ponto central inválido: {center_point}")
                return 0.0
                
            # Garantir que os índices estão dentro dos limites
            start = max(0, center_point - window_before) 
            end = min(min(len(series_a), len(series_b)), center_point + window_after)
            
            # Garantir que há pontos suficientes para correlação (pelo menos 3)
            if end - start < 3:
                logging.debug(f"Janela muito pequena para correlação: {end-start} pontos")
                return 0.0
                
            # Extrair subconjuntos das séries
            a_window = series_a.iloc[start:end]
            b_window = series_b.iloc[start:end]
            
            # Tratar valores NaN antes de calcular correlação
            valid_data = a_window.notna() & b_window.notna()
            a_valid = a_window[valid_data]
            b_valid = b_window[valid_data]
            
            # Verificar se ainda temos pontos suficientes após remover NaNs
            if len(a_valid) < 3:
                logging.debug(f"Dados insuficientes após remover NaNs: {len(a_valid)} pontos")
                return 0.0
            
            # Verificar variância das séries
            if np.var(a_valid) < 1e-8 or np.var(b_valid) < 1e-8:
                logging.debug(f"Variância muito baixa em uma das séries")
                return 0.0
                
            # Calcular correlação de Pearson
            correlation = np.corrcoef(a_valid, b_valid)[0, 1]
            
            # Tratar valores inválidos
            if np.isnan(correlation) or np.isinf(correlation):
                logging.debug(f"Correlação inválida: {correlation}")
                return 0.0
                
            return correlation
            
        except Exception as e:
            logging.error(f"Erro ao calcular correlação em janela: {e}")
            import traceback
            logging.debug(traceback.format_exc())
            return 0.0
    
    def _plot_change_point_impacts(self, source_metric, target_metrics, impact_df):
        """
        Cria visualização dos pontos de mudança e seus impactos com informações detalhadas.
        
        Args:
            source_metric: Série temporal fonte
            target_metrics: Dict com métricas alvo
            impact_df: DataFrame com resultados de impacto
        """
        if impact_df.empty or not self.plots_dir:
            return
            
        try:
            # Para cada métrica alvo com impactos detectados
            for target_name in impact_df['target_metric'].unique():
                # Filtrar impactos para esta métrica alvo específica
                target_impacts = impact_df[impact_df['target_metric'] == target_name]
                
                # Obter a série temporal da métrica alvo
                target_series = target_metrics[target_name]
                
                # Garantir que podemos indexar corretamente
                min_length = min(len(source_metric), len(target_series))
                source_data = source_metric.iloc[:min_length].values
                target_data = target_series.iloc[:min_length].values
                
                # Criar figura com três subplots para análise detalhada
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1, 0.5]})
                
                # 1. Plot da série fonte com pontos de mudança
                ax1.plot(source_data, label='Métrica Fonte', color='royalblue', linewidth=1.5)
                
                # Anotar pontos de mudança significativos na fonte
                for cp in target_impacts['source_change_point'].unique():
                    if cp < len(source_data):
                        ax1.axvline(x=cp, color='red', linestyle='--', alpha=0.7)
                        ax1.scatter(cp, source_data[cp], color='red', s=80, zorder=5, alpha=0.7)
                
                ax1.set_title(f"Pontos de Mudança na Métrica Fonte", fontsize=12, fontweight='bold')
                ax1.legend(loc='upper right')
                ax1.grid(True, alpha=0.3)
                ax1.set_ylabel("Valor")
                
                # 2. Plot da série alvo com pontos de impacto
                ax2.plot(target_data, label=target_name, color='forestgreen', linewidth=1.5)
                
                # Anotar pontos de impacto na série alvo
                for _, impact in target_impacts.iterrows():
                    source_cp = impact['source_change_point']
                    target_cp = impact['target_change_point']
                    
                    # Verificar se os índices estão dentro dos limites
                    if target_cp >= len(target_data) or source_cp >= len(source_data):
                        continue
                    
                    # Tamanho e cor do marcador baseados na força e significância do impacto
                    marker_size = 50 + 100 * impact['impact_strength']
                    color = 'darkred' if impact['significant'] else 'darkorange'
                    
                    # Destacar o ponto de impacto
                    ax2.scatter(target_cp, target_data[target_cp], 
                               s=marker_size, color=color, zorder=5,
                               edgecolor='black', linewidth=1)
                    
                    # Linha conectando ponto de origem ao impacto com curvatura
                    ax2.annotate('',
                        xy=(target_cp, target_data[target_cp]),
                        xytext=(source_cp, target_data[target_cp]),
                        arrowprops=dict(
                            arrowstyle='->', 
                            color=color, 
                            alpha=0.7, 
                            lw=1.5,
                            connectionstyle='arc3,rad=0.1'
                        ))
                    
                    # Adicionar informação sobre a força do impacto
                    ax2.annotate(f"Impacto: {impact['impact_strength']:.2f}",
                        xy=(target_cp, target_data[target_cp]),
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                
                ax2.set_title(f"Impactos Causais Detectados em {target_name}", fontsize=12, fontweight='bold')
                ax2.set_ylabel("Valor")
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='upper right')
                
                # 3. Plot adicional mostrando a correlação antes/depois para cada ponto de impacto
                if 'correlation_before' in target_impacts.columns and 'correlation_after' in target_impacts.columns:
                    # Extrair dados para o gráfico de barras
                    impact_points = target_impacts['target_change_point'].values
                    corr_before = target_impacts['correlation_before'].values
                    corr_after = target_impacts['correlation_after'].values
                    
                    # Limitar a 5 pontos para legibilidade
                    if len(impact_points) > 5:
                        # Ordenar por força de impacto e pegar os top 5
                        top_indices = target_impacts['impact_strength'].nlargest(5).index
                        impact_points = target_impacts.loc[top_indices, 'target_change_point'].values
                        corr_before = target_impacts.loc[top_indices, 'correlation_before'].values
                        corr_after = target_impacts.loc[top_indices, 'correlation_after'].values
                    
                    # Configurar posições das barras
                    x = np.arange(len(impact_points))
                    width = 0.35
                    
                    # Criar gráfico de barras para correlação
                    ax3.bar(x - width/2, corr_before, width, label='Correlação Antes', color='lightblue', alpha=0.7)
                    ax3.bar(x + width/2, corr_after, width, label='Correlação Depois', color='coral', alpha=0.7)
                    
                    # Configurar eixos e legendas
                    ax3.set_ylabel('Correlação')
                    ax3.set_title('Mudança de Correlação nos Pontos de Impacto', fontsize=12)
                    ax3.set_xticks(x)
                    ax3.set_xticklabels([f"CP{i+1}" for i in range(len(impact_points))])
                    ax3.legend()
                    ax3.grid(True, alpha=0.2, axis='y')
                    
                    # Adicionar linhas horizontais para correlação zero
                    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                    
                    # Definir limites do eixo y
                    ax3.set_ylim(-1, 1)
                
                # Ajustar o layout para melhor visualização
                plt.tight_layout()
                
                # Salvar figuras em vários formatos
                clean_target_name = target_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
                plt.savefig(self.plots_dir / f"causal_impact_{clean_target_name}.png", dpi=300)
                plt.savefig(self.plots_dir / f"causal_impact_{clean_target_name}.pdf", format='pdf')
                plt.close()
                
        except Exception as e:
            logging.error(f"Erro ao criar visualização de impactos: {e}")
            import traceback
            logging.debug(traceback.format_exc())
    
    def run_causal_analysis(self, phase_data, method='toda-yamamoto', metrics_of_interest=None, 
                          components=None, save_results=True):
        """
        Executa análise causal nas métricas especificadas usando o método selecionado.
        
        Args:
            phase_data: Dicionário de dados de fases do K8s
            method: Método de análise causal ('toda-yamamoto', 'transfer-entropy', 'change-point-impact')
            metrics_of_interest: Lista de métricas para analisar (opcional)
            components: Lista de componentes para analisar (opcional)
            save_results: Se True, salva resultados em arquivos
            
        Returns:
            DataFrame com resultados de causalidade
        """
        results = []
        
        # Verificar dados de entrada
        if not phase_data:
            logging.error("Dados de fase vazios para análise causal")
            return pd.DataFrame()
        
        # Se não especificados, usar todos componentes e métricas
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
                            if 'error' not in forward_result:
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
                            
                            if 'error' not in reverse_result:
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
                
                # Filtrar apenas relações causais significativas para tabela LaTeX
                significant_results = results_df[results_df['causality'] == True]
                
                if not significant_results.empty:
                    with open(self.results_dir / latex_filename, 'w') as f:
                        columns_to_export = [col for col in latex_cols if col in significant_results.columns]
                        f.write(significant_results[columns_to_export].to_latex(index=False))
                
            return results_df
            
        except Exception as e:
            logging.error(f"Erro ao executar análise causal: {e}")
            return pd.DataFrame({
                'error': [f"Falha na análise causal: {str(e)}"]
            })


# Função para teste direto do módulo
if __name__ == "__main__":
    import sys
    import argparse
    
    # Configuração de logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    parser = argparse.ArgumentParser(description="Análise causal para métricas do K8s")
    
    parser.add_argument('--method', type=str, default='toda-yamamoto',
                      choices=['toda-yamamoto', 'transfer-entropy', 'change-point-impact'],
                      help='Método de análise causal')
                      
    parser.add_argument('--output', type=str, default=None,
                      help='Diretório para salvar resultados')
    
    args = parser.parse_args()
    
    print(f"Módulo de análise causal carregado. Método selecionado: {args.method}")
    print("Para usar, importe este módulo em seu script principal.")
    
    sys.exit(0)
