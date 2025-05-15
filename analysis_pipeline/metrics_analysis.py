#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics Analysis Module for Kubernetes Noisy Neighbours Lab
Este módulo implementa análises focadas em métricas específicas, integrando análises 
estatísticas, de séries temporais, correlação, causalidade e entropia.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats as stats
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
import warnings

# Tentar importar bibliotecas opcionais
try:
    import nolds
    NOLDS_AVAILABLE = True
except ImportError:
    NOLDS_AVAILABLE = False
    logging.warning("nolds não encontrado. Algumas métricas de entropia não estarão disponíveis.")

try:
    import pyinform
    PYINFORM_AVAILABLE = True
except ImportError:
    PYINFORM_AVAILABLE = False
    logging.warning("pyinform não encontrado. Algumas métricas de entropia não estarão disponíveis.")

try:
    from sklearn.cluster import DBSCAN, KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn não encontrado. Alguns métodos de detecção de anomalias não estarão disponíveis.")

class MetricsAnalyzer:
    def __init__(self, output_dir=None):
        """
        Inicializa o analisador de métricas.
        
        Args:
            output_dir (str): Diretório para salvar resultados e gráficos
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir and not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Criar subdiretórios para análises específicas
        if self.output_dir:
            self.plots_dir = self.output_dir / "metrics_analysis"
            self.plots_dir.mkdir(exist_ok=True)
            self.ts_plots_dir = self.output_dir / "timeseries"
            self.ts_plots_dir.mkdir(exist_ok=True)
            self.corr_dir = self.output_dir / "correlations"
            self.corr_dir.mkdir(exist_ok=True)
            self.stats_dir = self.output_dir / "stats_results"
            self.stats_dir.mkdir(exist_ok=True)
            self.tables_dir = self.output_dir / "tables"
            self.tables_dir.mkdir(exist_ok=True)
            
        logging.info(f"Inicializado MetricsAnalyzer, diretório de saída: {self.output_dir}")
    
    # ========== ANÁLISE ESTATÍSTICA DESCRITIVA ==========
    
    def get_descriptive_stats(self, data, label=None):
        """
        Calcula estatísticas descritivas para uma série temporal ou DataFrame.
        
        Args:
            data (Series/DataFrame): Dados para análise
            label (str): Rótulo para identificar a série no nome do arquivo
            
        Returns:
            DataFrame: Estatísticas descritivas
            
        Outputs:
            - Tabela com estatísticas em .csv e .tex
            - Histograma em .png se for uma Series
        """
        if data is None:
            logging.warning("Dados vazios para estatísticas descritivas")
            return None
        
        # Lidar com Series ou DataFrame
        if isinstance(data, pd.Series):
            # Converter para numérico
            numeric_data = pd.to_numeric(data, errors='coerce').dropna()
            
            # Calcular estatísticas
            stats_dict = {
                'mean': numeric_data.mean(),
                'median': numeric_data.median(),
                'std': numeric_data.std(),
                'cv': (numeric_data.std() / numeric_data.mean()) if numeric_data.mean() != 0 else np.nan,
                'min': numeric_data.min(),
                'max': numeric_data.max(),
                'range': numeric_data.max() - numeric_data.min(),
                'q25': numeric_data.quantile(0.25),
                'q75': numeric_data.quantile(0.75),
                'skewness': numeric_data.skew(),
                'kurtosis': numeric_data.kurtosis(),
                'count': len(numeric_data)
            }
            
            # Criar DataFrame para fácil exportação
            stats_df = pd.DataFrame([stats_dict], index=['value']).T
            stats_df.columns = ['Value']
            
            # Gerar histograma
            if self.plots_dir and label:
                self.plot_histogram(numeric_data, label)
            
        elif isinstance(data, pd.DataFrame):
            # Calcular estatísticas para DataFrame
            stats_df = data.describe()
            
            # Adicionar estatísticas adicionais
            for col in data.columns:
                numeric_data = pd.to_numeric(data[col], errors='coerce').dropna()
                stats_df.loc['skewness', col] = stats.skew(numeric_data) if len(numeric_data) > 0 else np.nan
                stats_df.loc['kurtosis', col] = stats.kurtosis(numeric_data) if len(numeric_data) > 0 else np.nan
                cv = numeric_data.std() / numeric_data.mean() if len(numeric_data) > 0 and numeric_data.mean() != 0 else np.nan
                stats_df.loc['cv', col] = cv
        else:
            logging.warning(f"Tipo de dados não suportado: {type(data)}")
            return None
            
        # Salvar tabela se diretório de saída configurado
        if self.stats_dir and label:
            safe_name = label.replace(" ", "_").replace("/", "_")
            
            # Salvar em CSV
            csv_path = self.stats_dir / f"{safe_name}_stats.csv"
            stats_df.to_csv(csv_path)
            logging.info(f"Estatísticas descritivas salvas em: {csv_path}")
            
            # Gerar tabela LaTeX
            tex_path = self.stats_dir / f"{safe_name}_stats.tex"
            with open(tex_path, 'w') as f:
                f.write(stats_df.to_latex(float_format=lambda x: f"{x:.4f}"))
                
        return stats_df
    
    def plot_histogram(self, data, label=None):
        """
        Gera um histograma da distribuição dos dados.
        
        Args:
            data (Series): Série de dados para análise
            label (str): Rótulo para identificar a série no nome do arquivo
            
        Outputs:
            - Histograma em .png
        """
        if not self.plots_dir or data is None or len(data) < 5:
            return
        
        plt.figure(figsize=(10, 6))
        
        # Criar histograma com densidade KDE
        sns.histplot(data, kde=True)
        
        title = "Distribuição de Dados"
        if label:
            title += f": {label}"
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Salvar figura
        if label:
            safe_name = label.replace(" ", "_").replace("/", "_")
            fig_path = self.plots_dir / f"{safe_name}_histogram.png"
            plt.savefig(fig_path)
            plt.close()
            logging.info(f"Histograma salvo em: {fig_path}")
        else:
            plt.close()
    
    def check_stationarity(self, data, metric_name=None):
        """
        Realiza testes de estacionariedade (ADF e KPSS).
        
        Args:
            data (Series/DataFrame): Dados de entrada
            metric_name (str): Nome da métrica para contexto
            
        Returns:
            dict: Resultados dos testes de estacionariedade
        """
        results = {}
        
        # Testes ADF e KPSS
        if isinstance(data, pd.DataFrame):
            for col in data.columns:
                results[col] = self._run_stationarity_tests(data[col])
        else:
            results = self._run_stationarity_tests(data)
            
        # Salvar resultados
        if self.stats_dir and metric_name:
            pd.DataFrame(results).to_csv(self.stats_dir / f"{metric_name}_stationarity.csv")
            
        return results
    
    def _run_stationarity_tests(self, series):
        """Executa testes ADF e KPSS em uma série"""
        # Remover valores NaN
        clean_series = series.dropna()
        
        if len(clean_series) < 5:  # Verificar se há dados suficientes
            return {"adf_statistic": None, "adf_pvalue": None, "kpss_statistic": None, "kpss_pvalue": None}
            
        try:
            # Teste ADF (Augmented Dickey-Fuller)
            adf_result = adfuller(clean_series)
            
            # Teste KPSS
            kpss_result = kpss(clean_series)
            
            return {
                "adf_statistic": adf_result[0],
                "adf_pvalue": adf_result[1],
                "kpss_statistic": kpss_result[0],
                "kpss_pvalue": kpss_result[1]
            }
        except Exception as e:
            logging.warning(f"Erro ao executar testes de estacionariedade: {str(e)}")
            return {"adf_statistic": None, "adf_pvalue": None, "kpss_statistic": None, "kpss_pvalue": None}
    
    # ========== ANÁLISE DE SÉRIES TEMPORAIS ==========
    
    def decompose_time_series(self, series, metric_name=None, period=None, model='additive'):
        """
        Decompõe uma série temporal em tendência, sazonalidade e resíduo.
        
        Args:
            series (Series): Série temporal para decompor
            metric_name (str): Nome da métrica para contexto
            period (int): Período para decomposição sazonal. Se None, será estimado.
            model (str): Modelo de decomposição ('additive' ou 'multiplicative')
            
        Returns:
            object: Resultado da decomposição
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Remover valores NaN
        clean_series = series.dropna()
        
        # Estimar período se não fornecido
        if period is None:
            # Tentar estimar com base na autocorrelação
            from statsmodels.tsa.stattools import acf
            acf_vals = acf(clean_series, nlags=len(clean_series)//3)
            # Encontrar picos na ACF após o lag 1
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(acf_vals[1:], height=0.1)
            if len(peaks) > 0:
                period = peaks[0] + 1
                logging.info(f"Período estimado: {period}")
            else:
                period = 24  # Valor padrão (assumindo dados horários)
                logging.info(f"Não foi possível estimar o período, usando padrão: {period}")
        
        # Realizar decomposição
        try:
            decomposition = seasonal_decompose(clean_series, model=model, period=period)
            
            # Plotar decomposição
            if self.plots_dir and metric_name:
                fig, axes = plt.subplots(4, 1, figsize=(10, 12))
                axes[0].plot(decomposition.observed)
                axes[0].set_title('Observed')
                axes[1].plot(decomposition.trend)
                axes[1].set_title('Trend')
                axes[2].plot(decomposition.seasonal)
                axes[2].set_title('Seasonal')
                axes[3].plot(decomposition.resid)
                axes[3].set_title('Residual')
                plt.tight_layout()
                
                # Salvar figura
                fig_path = self.plots_dir / f"{metric_name}_decomposition.png"
                plt.savefig(fig_path)
                plt.close()
                logging.info(f"Decomposição de série temporal salva em: {fig_path}")
                
            return decomposition
            
        except Exception as e:
            logging.warning(f"Erro na decomposição da série temporal: {str(e)}")
            return None
    
    # ========== ANÁLISE DE CORRELAÇÃO ==========
    
    def analyze_correlations(self, data_dict, title=None, corr_method='pearson'):
        """
        Analisa correlações entre múltiplas séries temporais.
        
        Args:
            data_dict (dict): Dicionário com séries temporais
            title (str): Título para o gráfico
            corr_method (str): Método de correlação ('pearson', 'spearman', 'kendall')
            
        Returns:
            DataFrame: Matriz de correlação
            
        Outputs:
            - Heatmap de correlação (.png)
            - Arquivo CSV com valores de correlação
        """
        # Verificar se temos dados
        if not data_dict or len(data_dict) < 2:
            logging.warning("Dados insuficientes para análise de correlação")
            return None
            
        # Alinhar séries temporais
        aligned_data = self._align_time_series(data_dict)
        
        if aligned_data is None or aligned_data.empty:
            logging.warning("Não foi possível alinhar os dados para análise de correlação")
            return None
            
        # Calcular matriz de correlação
        corr_matrix = aligned_data.corr(method=corr_method)
        
        # Gerar heatmap
        if self.plots_dir:
            plt.figure(figsize=(10, 8))
            mask = np.zeros_like(corr_matrix, dtype=bool)
            mask[np.triu_indices_from(mask)] = True  # Mascarar triângulo superior
            
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                        square=True, linewidths=.5, annot=True, fmt=".2f")
            
            if title:
                plt.title(f"Correlation Matrix: {title}")
            else:
                plt.title("Correlation Matrix")
                
            # Salvar figura
            safe_title = title.replace(" ", "_").replace("/", "_") if title else "correlation"
            fig_path = self.plots_dir / f"{safe_title}_heatmap.png"
            plt.savefig(fig_path)
            plt.close()
            logging.info(f"Heatmap de correlação salvo em: {fig_path}")
            
            # Salvar matriz de correlação como CSV
            csv_path = self.tables_dir / f"{safe_title}_correlation_matrix.csv" if self.tables_dir else None
            if csv_path:
                corr_matrix.to_csv(csv_path)
                logging.info(f"Matriz de correlação salva em: {csv_path}")
                
                # Gerar tabela LaTeX
                tex_path = self.tables_dir / f"{safe_title}_correlation_matrix.tex"
                with open(tex_path, 'w') as f:
                    f.write(corr_matrix.to_latex(float_format=lambda x: f"{x:.3f}"))
                logging.info(f"Tabela LaTeX salva em: {tex_path}")
        
        return corr_matrix
    
    def _align_time_series(self, series_dict):
        """Alinha múltiplas séries temporais para o mesmo índice de tempo."""
        # Extrair séries e converter para DataFrame se necessário
        aligned_data = {}
        for name, data in series_dict.items():
            if isinstance(data, pd.DataFrame):
                # Se DataFrame, usar a primeira coluna de valores
                value_cols = [col for col in data.columns if col != 'timestamp' and pd.api.types.is_numeric_dtype(data[col])]
                if value_cols:
                    # Usar primeira coluna de valor para alinhamento
                    aligned_data[name] = data[value_cols[0]]
                else:
                    logging.warning(f"Nenhuma coluna de valor encontrada para {name}")
            elif isinstance(data, pd.Series):
                aligned_data[name] = data
            else:
                logging.warning(f"Tipo de dados não suportado para {name}: {type(data)}")
        
        if not aligned_data:
            return None
        
        # Converter para DataFrame
        return pd.DataFrame(aligned_data)
    
    def ccf(self, x, y, adjusted=True):
        """
        Calcula a função de correlação cruzada entre duas séries temporais.
        
        Args:
            x (array-like): Primeira série temporal
            y (array-like): Segunda série temporal
            adjusted (bool): Se True, ajusta para séries de diferentes comprimentos
            
        Returns:
            numpy.ndarray: Array de correlações cruzadas
        """
        # Garantir que x e y são arrays numpy
        x = np.array(x)
        y = np.array(y)
        
        # Normalizar as séries (subtrair média e dividir por desvio padrão)
        x = (x - np.mean(x)) / (np.std(x) * len(x))
        y = (y - np.mean(y)) / np.std(y)
        
        # Calcular correlação cruzada usando convolução
        corr = np.correlate(x, y, mode='full')
        
        # Extrair a parte positiva da correlação (lags positivos)
        middle = len(corr) // 2
        corr_positive_lags = corr[middle:]
        
        return corr_positive_lags
        
    def cross_correlation(self, series1, series2, max_lag=20, title=None, series1_name=None, series2_name=None):
        """
        Calcula e plota correlação cruzada entre duas séries temporais.
        
        Args:
            series1 (Series): Primeira série temporal
            series2 (Series): Segunda série temporal
            max_lag (int): Defasagem máxima a considerar
            title (str): Título do gráfico
            series1_name (str): Nome da primeira série
            series2_name (str): Nome da segunda série
            
        Returns:
            tuple: (array de correlação cruzada, array de defasagens)
        """
        # Converter para numérico
        s1 = pd.to_numeric(series1, errors='coerce').dropna()
        s2 = pd.to_numeric(series2, errors='coerce').dropna()
        
        # Verificar dados suficientes
        if len(s1) < 2 or len(s2) < 2:
            logging.warning("Séries temporais muito curtas para análise de correlação cruzada")
            return None, None
            
        # Calcular correlação cruzada
        cross_corr = self.ccf(s1, s2, adjusted=True)[:max_lag+1]
        lags = np.arange(len(cross_corr))
        
        # Gerar gráfico
        if self.plots_dir:
            plt.figure(figsize=(10, 6))
            plt.stem(lags, cross_corr, use_line_collection=True)
            plt.axhline(y=0, linestyle='--', color='gray')
            plt.xlabel('Lag')
            plt.ylabel('Cross-Correlation')
            
            # Adicionar título
            if title:
                plt.title(title)
            elif series1_name and series2_name:
                plt.title(f"Cross-Correlation: {series1_name} vs {series2_name}")
            else:
                plt.title("Cross-Correlation")
                
            # Marcar a defasagem com maior correlação
            max_idx = np.argmax(np.abs(cross_corr))
            plt.axvline(x=max_idx, color='red', linestyle='--', alpha=0.5)
            plt.annotate(f'Max at lag {max_idx} (r={cross_corr[max_idx]:.2f})', 
                         xy=(max_idx, cross_corr[max_idx]),
                         xytext=(max_idx+1, cross_corr[max_idx]),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
            
            # Salvar figura
            if series1_name and series2_name:
                safe_name = f"{series1_name}_vs_{series2_name}".replace(" ", "_").replace("/", "_")
            else:
                safe_name = "cross_correlation"
                
            fig_path = self.plots_dir / f"{safe_name}_ccf.png"
            plt.savefig(fig_path)
            plt.close()
            logging.info(f"Gráfico de correlação cruzada salvo em: {fig_path}")
            
        return cross_corr, lags
    
    # ========== ANÁLISE DE CAUSALIDADE E ENTROPIA ==========
    
    def granger_causality(self, series1, series2, max_lag=10, test='ssr_chi2test', series1_name=None, series2_name=None):
        """
        Teste de causalidade de Granger entre duas séries temporais.
        
        Args:
            series1 (Series): Primeira série temporal (potencial causa)
            series2 (Series): Segunda série temporal (potencial efeito)
            max_lag (int): Defasagem máxima a testar
            test (str): Tipo de teste ('ssr_chi2test', 'ssr_ftest', 'lrtest', 'params_ftest')
            series1_name (str): Nome da primeira série
            series2_name (str): Nome da segunda série
            
        Returns:
            dict: Resultados do teste de causalidade de Granger
        """
        # Converter para numérico
        s1 = pd.to_numeric(series1, errors='coerce').interpolate().dropna()
        s2 = pd.to_numeric(series2, errors='coerce').interpolate().dropna()
        
        # Verificar dados suficientes
        if len(s1) < max_lag + 2 or len(s2) < max_lag + 2:
            logging.warning("Séries temporais muito curtas para teste de causalidade de Granger")
            return {"error": "Insufficient data"}
            
        # Preparar dados para teste (alinhar índices)
        common_index = s1.index.intersection(s2.index)
        if len(common_index) < max_lag + 2:
            logging.warning("Dados alinhados insuficientes para teste de causalidade de Granger")
            return {"error": "Insufficient aligned data"}
            
        s1_aligned = s1.loc[common_index]
        s2_aligned = s2.loc[common_index]
        
        # Executar teste de causalidade de Granger
        data = pd.DataFrame({'x': s1_aligned, 'y': s2_aligned})
        try:
            result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            
            # Processar e formatar resultados
            processed_results = {}
            for lag, lag_result in result.items():
                test_statistic = lag_result[0][test][0]
                p_value = lag_result[0][test][1]
                processed_results[lag] = {
                    'test_statistic': test_statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
            # Salvar resultados como tabela
            if self.tables_dir and series1_name and series2_name:
                result_df = pd.DataFrame({
                    'lag': list(processed_results.keys()),
                    'test_statistic': [processed_results[lag]['test_statistic'] for lag in processed_results],
                    'p_value': [processed_results[lag]['p_value'] for lag in processed_results],
                    'significant': [processed_results[lag]['significant'] for lag in processed_results]
                })
                
                safe_name = f"{series1_name}_causes_{series2_name}".replace(" ", "_").replace("/", "_")
                csv_path = self.tables_dir / f"{safe_name}_granger.csv"
                result_df.to_csv(csv_path, index=False)
                logging.info(f"Resultados de causalidade de Granger salvos em: {csv_path}")
                
            return processed_results
            
        except Exception as e:
            logging.warning(f"Erro no teste de causalidade de Granger: {str(e)}")
            return {"error": str(e)}
    
    def calculate_entropy(self, series, method='sample', metric_name=None):
        """
        Calcula entropia para uma série temporal.
        
        Args:
            series (Series): Série temporal
            method (str): Método de entropia ('sample', 'approximate', 'shannon', 'permutation')
            metric_name (str): Nome da métrica para contexto
            
        Returns:
            float: Valor da entropia
        """
        # Converter para numérico e remover NaNs
        clean_series = pd.to_numeric(series, errors='coerce').dropna()
        
        if len(clean_series) < 10:
            logging.warning("Série temporal muito curta para cálculo de entropia")
            return None
            
        try:
            # Calcular entropia de acordo com o método
            if method == 'sample':
                entropy_value = nolds.sampen(clean_series)
            elif method == 'approximate':
                entropy_value = nolds.ap_entropy(clean_series, emb_dim=2)
            elif method == 'shannon':
                # Normalizar e discretizar a série para cálculo da entropia de Shannon
                normalized = (clean_series - min(clean_series)) / (max(clean_series) - min(clean_series))
                bins = np.linspace(0, 1, 10)
                discretized = np.digitize(normalized, bins)
                entropy_value = pyinform.shannon.entropy(discretized)
            elif method == 'permutation':
                entropy_value = pyinform.shannon.entropy_rate(clean_series, 3)
            else:
                raise ValueError(f"Método de entropia não suportado: {method}")
            
            # Salvar resultado
            if self.tables_dir and metric_name:
                safe_name = f"{metric_name}".replace(" ", "_").replace("/", "_")
                with open(self.tables_dir / f"{safe_name}_entropy.csv", 'w') as f:
                    f.write(f"method,entropy\n{method},{entropy_value}\n")
                    
            return entropy_value
            
        except Exception as e:
            logging.warning(f"Erro no cálculo de entropia ({method}): {str(e)}")
            return None
    
    # ========== DETECÇÃO DE ANOMALIAS ==========
    
    def detect_anomalies(self, series, method='zscore', threshold=3.0, metric_name=None):
        """
        Detecta anomalias em uma série temporal.
        
        Args:
            series (Series): Série temporal
            method (str): Método de detecção ('zscore', 'iqr', 'iforest')
            threshold (float): Limite para detecção (usado apenas por alguns métodos)
            metric_name (str): Nome da métrica para contexto
            
        Returns:
            Series: Série booleana indicando anomalias
            
        Outputs:
            - Gráfico com anomalias destacadas (.png)
        """
        # Converter para numérico e remover NaNs
        clean_series = pd.to_numeric(series, errors='coerce')
        
        if clean_series.isna().all():
            logging.warning("Série contém apenas valores NA")
            return pd.Series(False, index=series.index)
            
        # Interpolar NaNs para análise
        clean_series = clean_series.interpolate()
        
        # Detectar anomalias baseado no método
        if method == 'zscore':
            mean = clean_series.mean()
            std = clean_series.std()
            anomalies = (abs(clean_series - mean) > threshold * std)
            
        elif method == 'iqr':
            q1 = clean_series.quantile(0.25)
            q3 = clean_series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            anomalies = (clean_series < lower_bound) | (clean_series > upper_bound)
            
        elif method == 'iforest':
            from sklearn.ensemble import IsolationForest
            
            # Reshape para 2D array (exigido pelo scikit-learn)
            X = clean_series.values.reshape(-1, 1)
            
            # Treinar modelo
            model = IsolationForest(contamination=0.05, random_state=42)
            anomaly_scores = model.fit_predict(X)
            anomalies = pd.Series(anomaly_scores == -1, index=clean_series.index)
            
        else:
            raise ValueError(f"Método de detecção de anomalias não suportado: {method}")
            
        # Gerar visualização
        if self.plots_dir and metric_name:
            plt.figure(figsize=(12, 6))
            plt.plot(clean_series, label='Série Original')
            if anomalies.sum() > 0:  # Verificar se há anomalias
                plt.scatter(
                    clean_series[anomalies].index, 
                    clean_series[anomalies], 
                    color='red', 
                    label=f'Anomalias ({anomalies.sum()} pontos)'
                )
            plt.title(f"Detecção de Anomalias: {metric_name} (método: {method})")
            plt.legend()
            plt.grid(True)
            
            # Salvar figura
            safe_name = f"{metric_name}_anomalies_{method}".replace(" ", "_").replace("/", "_")
            fig_path = self.plots_dir / f"{safe_name}.png"
            plt.savefig(fig_path)
            plt.close()
            logging.info(f"Gráfico de detecção de anomalias salvo em: {fig_path}")
            
            # Salvar índices das anomalias
            if anomalies.sum() > 0:
                anomaly_points = clean_series[anomalies].reset_index()
                anomaly_points.columns = ['timestamp', 'value']
                csv_path = self.tables_dir / f"{safe_name}.csv" if self.tables_dir else None
                if csv_path:
                    anomaly_points.to_csv(csv_path, index=False)
                    logging.info(f"Pontos de anomalia salvos em: {csv_path}")
            
        return anomalies
    
    # ========== SISTEMA DE SUGESTÕES ==========
    
    def suggest_plots(self, data, metric_name):
        """
        Sugere visualizações relevantes com base nas características dos dados.
        
        Args:
            data (DataFrame/Series): Dados para análise
            metric_name (str): Nome da métrica
            
        Returns:
            dict: Sugestões de visualizações com justificativas
        """
        suggestions = {}
        
        # Verificar se temos dados suficientes
        if data is None or len(data) < 5:
            return {"error": "Dados insuficientes para sugestões"}
            
        # Converter para numérico se necessário
        if isinstance(data, pd.DataFrame):
            numeric_data = data.apply(pd.to_numeric, errors='coerce')
        else:
            numeric_data = pd.to_numeric(data, errors='coerce')
            
        # Análise de estacionariedade
        try:
            stationarity_results = self.check_stationarity(numeric_data)
            is_stationary = False
            
            if isinstance(stationarity_results, dict):
                # Verificar o p-valor do teste ADF (p < 0.05 indica estacionariedade)
                if 'adf_pvalue' in stationarity_results and stationarity_results['adf_pvalue'] is not None:
                    is_stationary = stationarity_results['adf_pvalue'] < 0.05
                
            # Sugerir decomposição para séries não-estacionárias
            if not is_stationary:
                suggestions["decomposition"] = {
                    "type": "Decomposição de Série Temporal",
                    "justification": "Série não é estacionária, a decomposição ajudará a visualizar tendências e sazonalidades."
                }
        except:
            pass
            
        # Análise de distribuição
        try:
            if isinstance(numeric_data, pd.DataFrame):
                skewness = numeric_data.skew().abs().mean()
                kurtosis = numeric_data.kurtosis().abs().mean()
            else:
                skewness = abs(stats.skew(numeric_data.dropna()))
                kurtosis = abs(stats.kurtosis(numeric_data.dropna()))
                
            # Sugestões baseadas na distribuição
            if skewness > 1.0:
                suggestions["histogram"] = {
                    "type": "Histograma",
                    "justification": f"Distribuição apresenta assimetria significativa (skewness = {skewness:.2f}), um histograma ajudará a visualizar."
                }
                
            if kurtosis > 3.0:
                suggestions["boxplot"] = {
                    "type": "Boxplot",
                    "justification": f"Distribuição apresenta kurtosis elevada ({kurtosis:.2f}), indicando possíveis outliers que podem ser visualizados com boxplot."
                }
        except:
            pass
            
        # Análise de sazonalidade/padrões
        try:
            from statsmodels.tsa.stattools import acf
            
            if isinstance(numeric_data, pd.DataFrame):
                first_col = numeric_data.columns[0]
                acf_values = acf(numeric_data[first_col].dropna(), nlags=min(24, len(numeric_data)//3))
            else:
                acf_values = acf(numeric_data.dropna(), nlags=min(24, len(numeric_data)//3))
                
            # Verificar autocorrelações significativas
            significant_lags = (abs(acf_values[1:]) > 1.96/np.sqrt(len(numeric_data))).sum()
            
            if significant_lags > 2:
                suggestions["acf_pacf"] = {
                    "type": "Autocorrelation Function",
                    "justification": f"Dados apresentam autocorrelação significativa em {significant_lags} lags, sugerindo padrões temporais importantes."
                }
        except:
            pass
            
        # Se for DataFrame, sugerir matriz de correlação
        if isinstance(numeric_data, pd.DataFrame) and numeric_data.shape[1] > 1:
            suggestions["correlation_matrix"] = {
                "type": "Matriz de Correlação",
                "justification": f"Conjunto de dados contém {numeric_data.shape[1]} variáveis, uma matriz de correlação pode revelar relações importantes."
            }
            
        return suggestions

    def suggest_tables(self, data, metric_name):
        """
        Sugere tabelas relevantes com base nas características dos dados.
        
        Args:
            data (DataFrame/Series): Dados para análise
            metric_name (str): Nome da métrica
            
        Returns:
            dict: Sugestões de tabelas com justificativas
        """
        suggestions = {}
        
        # Verificar se temos dados suficientes
        if data is None or len(data) < 5:
            return {"error": "Dados insuficientes para sugestões"}
        
        # Estatísticas descritivas são sempre úteis
        suggestions["descriptive_stats"] = {
            "type": "Estatísticas Descritivas",
            "justification": "Fornece visão geral dos dados com medidas de posição e dispersão.",
            "format": ["CSV", "LaTeX"]
        }
        
        # Verificar se há múltiplas colunas para correlação
        if isinstance(data, pd.DataFrame) and data.shape[1] > 1:
            suggestions["correlation_table"] = {
                "type": "Tabela de Correlação",
                "justification": f"Conjunto de dados contém {data.shape[1]} variáveis, uma tabela de correlação quantifica as relações.",
                "format": ["CSV", "LaTeX"]
            }
            
        # Verificar se tem dados suficientes para análise de percentis
        if len(data) >= 100:
            suggestions["percentiles"] = {
                "type": "Tabela de Percentis",
                "justification": "Conjunto de dados é suficientemente grande para análise de percentis detalhada.",
                "format": ["CSV"]
            }
            
        return suggestions
