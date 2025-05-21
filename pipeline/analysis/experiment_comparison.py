"""
Módulo para comparação entre diferentes experimentos de noisy neighbors.

Este módulo implementa funções para comparar métricas, padrões e anomalias
entre múltiplos experimentos de noisy neighbors.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any

from ..data_processing.time_normalization import normalize_time
from ..data_processing.aggregation import aggregate_by_time
from refactor.data_handling.loader import load_experiment_data # Atualizado

def load_multiple_experiments(experiment_paths: List[str]) -> Dict[str, Dict]:
    """
    Carrega dados de múltiplos experimentos para comparação.
    
    Args:
        experiment_paths (List[str]): Lista de caminhos para os diretórios dos experimentos
        
    Returns:
        Dict[str, Dict]: Dicionário com dados e metadados dos experimentos
    """
    
    experiments = {}
    
    for i, path in enumerate(experiment_paths):
        try:
            metrics_data, exp_info = load_experiment_data(path)
            
            # Usar o nome do experimento se disponível, caso contrário usar um ID
            exp_name = exp_info.get('name', f'experiment_{i+1}')
            
            # Armazenar dados e metadados
            experiments[exp_name] = {
                'metrics': metrics_data,
                'info': exp_info,
                'path': path
            }
            
        except Exception as e:
            print(f"Erro ao carregar experimento em {path}: {str(e)}")
    
    return experiments


def preprocess_experiments(
    experiments: Dict[str, Dict],
    metrics_of_interest: List[str],
    normalize_timestamps: bool = True,
    aggregate_data: bool = True,
    agg_freq: str = '1min',
    rounds_filter: Optional[List[str]] = None,
    tenants_filter: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Preprocessa dados de experimentos para comparação.
    
    Args:
        experiments (Dict[str, Dict]): Dicionário com dados e metadados dos experimentos
        metrics_of_interest (List[str]): Lista de métricas de interesse para comparação
        normalize_timestamps (bool): Se True, normaliza os timestamps
        aggregate_data (bool): Se True, agrega os dados
        agg_freq (str): Frequência de agregação (ex: '1min', '5min')
        rounds_filter (Optional[List[str]]): Lista de rounds para filtrar.
        tenants_filter (Optional[List[str]]): Lista de tenants para filtrar.
        
    Returns:
        Dict[str, Dict]: Dicionário com dados preprocessados
    """
    # Fazer uma cópia para não modificar os originais
    processed_experiments = {}
    
    for exp_name, exp_data in experiments.items():
        # Filtrar métricas de interesse
        selected_metrics = {k: v for k, v in exp_data['metrics'].items() 
                           if k in metrics_of_interest}
        
        # Processar cada métrica
        processed_metrics = {}
        for metric_name, df in selected_metrics.items():
            # Normalizar tempo se solicitado
            if normalize_timestamps:
                processed_df = normalize_time(df.copy(), exp_data['info'])
            else:
                processed_df = df.copy()

            # Aplicar filtro de tenants, se fornecido e a coluna 'tenant' existir
            if tenants_filter and 'tenant' in processed_df.columns:
                processed_df = processed_df[processed_df['tenant'].isin(tenants_filter)]
            
            # Aplicar filtro de rounds, se fornecido e a coluna 'round' existir
            if rounds_filter and 'round' in processed_df.columns:
                processed_df = processed_df[processed_df['round'].isin(rounds_filter)]

            if processed_df.empty: # Pular se o DataFrame ficar vazio após os filtros
                continue
            
            # Adicionar coluna de tempo decorrido em minutos
            processed_df['elapsed_minutes'] = (
                processed_df['datetime'] - processed_df['datetime'].min()
            ).dt.total_seconds() / 60
            
            # Agregar dados se solicitado
            if aggregate_data:
                processed_df = aggregate_by_time(
                    processed_df,
                    time_column='elapsed_minutes',
                    freq=agg_freq
                )
            
            processed_metrics[metric_name] = processed_df
        
        # Armazenar resultados
        processed_experiments[exp_name] = {
            'processed_metrics': processed_metrics,
            'info': exp_data['info'],
            'path': exp_data.get('path', '')
        }
    
    return processed_experiments


def calculate_statistics_summary(
    experiments: Dict[str, Dict], 
    metrics: List[str],
    group_by: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Calcula estatísticas resumidas para comparação entre experimentos.
    
    Args:
        experiments (Dict[str, Dict]): Dicionário com dados preprocessados
        metrics (List[str]): Lista de métricas para análise
        group_by (List[str]): Colunas para agrupar os dados (ex: ['tenant', 'phase'])
        
    Returns:
        Dict[str, DataFrame]: DataFrames com estatísticas por métrica
    """
    results = {}
    
    for metric in metrics:
        all_stats = []
        
        for exp_name, exp_data in experiments.items():
            if metric in exp_data['processed_metrics']:
                df = exp_data['processed_metrics'][metric]
                
                # Estatísticas globais
                stats = df.groupby(group_by)['value'].agg([
                    ('média', 'mean'),
                    ('mediana', 'median'),
                    ('desvio_padrão', 'std'),
                    ('mínimo', 'min'),
                    ('máximo', 'max'),
                    ('contagem', 'count')
                ]) if group_by else pd.DataFrame({
                    'média': [df['value'].mean()],
                    'mediana': [df['value'].median()],
                    'desvio_padrão': [df['value'].std()],
                    'mínimo': [df['value'].min()],
                    'máximo': [df['value'].max()],
                    'contagem': [df['value'].count()]
                })
                
                # Adicionar informações do experimento
                stats = stats.reset_index() if group_by else stats
                stats['experimento'] = exp_name
                
                all_stats.append(stats)
        
        if all_stats:
            results[metric] = pd.concat(all_stats, ignore_index=True)
    
    return results


def compare_distributions(
    experiments: Dict[str, Dict],
    metric: str,
    tenants_filter: Optional[List[str]] = None,
    phase: Optional[str] = None,
    test_method: str = 'ks',
    rounds_filter: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], Dict[str, Dict]]:
    """
    Compara distribuições estatísticas entre experimentos para uma métrica.
    
    Args:
        experiments (Dict[str, Dict]): Dicionário com dados preprocessados
        metric (str): Métrica para comparação
        tenants_filter (Optional[List[str]]): Lista de tenants específicos para filtrar
        phase (str, optional): Fase específica para filtrar
        test_method (str): Método de teste estatístico ('ks' ou 'mw')
        rounds_filter (Optional[List[str]]): Lista de rounds para filtrar.
        
    Returns:
        Tuple[Dict[str, Any], Dict[str, Dict]]: 
            - Dados estruturados para plotagem
            - Resultados dos testes estatísticos
    """
    plot_data = {'series': [], 'labels': []}
    test_results = {}
    
    # Coletar séries temporais para comparação
    for exp_name, exp_data in experiments.items():
        if metric in exp_data['processed_metrics']:
            df = exp_data['processed_metrics'][metric]
            
            # Filtrar por tenants se especificados
            if tenants_filter and 'tenant' in df.columns:
                df = df[df['tenant'].isin(tenants_filter)]
            
            # Filtrar por rounds se especificados e a coluna 'round' existir
            if rounds_filter and 'round' in df.columns:
                df = df[df['round'].isin(rounds_filter)]

            # Filtrar por fase se especificada
            if phase and 'phase' in df.columns:
                df = df[df['phase'] == phase]
            
            if not df.empty:
                plot_data['series'].append(df['value'])
                plot_data['labels'].append(exp_name)
    
    # Realizar testes estatísticos entre pares de experimentos
    n_series = len(plot_data['series'])
    if n_series >= 2:
        for i in range(n_series):
            for j in range(i+1, n_series):
                exp1 = plot_data['labels'][i]
                exp2 = plot_data['labels'][j]
                
                # Executar teste apropriado
                if test_method == 'ks':
                    # Kolmogorov-Smirnov test
                    stat, pval = stats.ks_2samp(plot_data['series'][i], plot_data['series'][j])
                    test_name = "Kolmogorov-Smirnov"
                else:
                    # Mann-Whitney U test
                    stat, pval = stats.mannwhitneyu(
                        plot_data['series'][i], plot_data['series'][j], alternative='two-sided'
                    )
                    test_name = "Mann-Whitney U"
                
                test_results[f"{exp1}_vs_{exp2}"] = {
                    "test": test_name,
                    "statistic": stat,
                    "p_value": pval,
                    "significant_diff": pval < 0.05
                }
    
    return plot_data, test_results


def detect_anomalies_across_experiments(
    experiments: Dict[str, Dict],
    metrics: List[str],
    contamination: float = 0.05,
    group_by: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Detecta e compara anomalias entre experimentos.
    
    Args:
        experiments (Dict[str, Dict]): Dicionário com dados preprocessados
        metrics (List[str]): Lista de métricas para análise
        contamination (float): Proporção esperada de anomalias nos dados
        group_by (List[str]): Colunas para agrupar os dados (ex: ['tenant'])
        
    Returns:
        Dict[str, Dict[str, Any]]: Resultados da detecção de anomalias por experimento e métrica
    """
    from .anomaly_detection import detect_anomalies_ensemble
    
    anomaly_results = {}
    
    for exp_name, exp_data in experiments.items():
        anomaly_results[exp_name] = {}
        
        for metric in metrics:
            if metric in exp_data['processed_metrics']:
                df = exp_data['processed_metrics'][metric]
                
                # Detectar anomalias
                df_with_anomalies, change_info = detect_anomalies_ensemble(
                    df, 
                    metric_column='value',
                    time_column='elapsed_minutes',
                    contamination=contamination,
                    group_by=group_by
                )
                
                # Armazenar resultados
                anomaly_results[exp_name][metric] = {
                    'dataframe': df_with_anomalies,
                    'change_info': change_info
                }
    
    return anomaly_results


def summarize_anomalies(anomaly_results: Dict[str, Dict[str, Dict[str, Any]]]) -> pd.DataFrame:
    """
    Gera um resumo das anomalias detectadas para comparação.
    
    Args:
        anomaly_results (Dict): Resultados da detecção de anomalias
        
    Returns:
        DataFrame: Resumo das anomalias por experimento, métrica e grupo
    """
    anomaly_summary = []
    
    # Iterar por experimentos e métricas
    for exp_name, metrics_results in anomaly_results.items():
        for metric, results in metrics_results.items():
            df = results['dataframe']
            
            # Estatísticas globais
            total_points = len(df)
            anomaly_if_count = df['is_anomaly_if'].sum() if 'is_anomaly_if' in df.columns else 0
            anomaly_lof_count = df['is_anomaly_lof'].sum() if 'is_anomaly_lof' in df.columns else 0
            anomaly_ensemble_count = df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0
            change_point_count = df['is_change_point'].sum() if 'is_change_point' in df.columns else 0
            
            # Adicionar estatísticas globais
            anomaly_summary.append({
                'experimento': exp_name,
                'métrica': metric,
                'grupo': 'global',
                'total_pontos': total_points,
                'anomalias_if': anomaly_if_count,
                'anomalias_lof': anomaly_lof_count,
                'anomalias_ensemble': anomaly_ensemble_count,
                'pontos_mudança': change_point_count,
                'pct_anomalias': (anomaly_ensemble_count / total_points * 100) if total_points > 0 else 0
            })
            
            # Estatísticas por tenant
            for tenant in df['tenant'].unique():
                tenant_df = df[df['tenant'] == tenant]
                total_points = len(tenant_df)
                anomaly_if_count = tenant_df['is_anomaly_if'].sum() if 'is_anomaly_if' in tenant_df.columns else 0
                anomaly_lof_count = tenant_df['is_anomaly_lof'].sum() if 'is_anomaly_lof' in tenant_df.columns else 0
                anomaly_ensemble_count = tenant_df['is_anomaly'].sum() if 'is_anomaly' in tenant_df.columns else 0
                change_point_count = tenant_df['is_change_point'].sum() if 'is_change_point' in tenant_df.columns else 0
                
                # Adicionar estatísticas por tenant
                anomaly_summary.append({
                    'experimento': exp_name,
                    'métrica': metric,
                    'grupo': tenant,
                    'total_pontos': total_points,
                    'anomalias_if': anomaly_if_count,
                    'anomalias_lof': anomaly_lof_count,
                    'anomalias_ensemble': anomaly_ensemble_count,
                    'pontos_mudança': change_point_count,
                    'pct_anomalias': (anomaly_ensemble_count / total_points * 100) if total_points > 0 else 0
                })
    
    return pd.DataFrame(anomaly_summary)


def extract_experiment_features(
    experiments: Dict[str, Dict], 
    metrics: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Extrai características estatísticas dos experimentos para análise dimensional.
    
    Args:
        experiments (Dict[str, Dict]): Dicionário com dados preprocessados
        metrics (List[str]): Lista de métricas para análise
        
    Returns:
        Dict[str, DataFrame]: DataFrames com características por métrica
    """
    features_by_metric = {}
    
    for metric in metrics:
        features = []
        
        for exp_name, exp_data in experiments.items():
            if metric in exp_data['processed_metrics']:
                df = exp_data['processed_metrics'][metric]
                
                # Extrair características por tenant
                for tenant in df['tenant'].unique():
                    tenant_data = df[df['tenant'] == tenant]
                    
                    # Pular se não houver dados suficientes
                    if len(tenant_data) < 5:
                        continue
                    
                    # Calcular estatísticas
                    try:
                        mean_val = tenant_data['value'].mean()
                        std_val = tenant_data['value'].std()
                        max_val = tenant_data['value'].max()
                        min_val = tenant_data['value'].min()
                        median_val = tenant_data['value'].median()
                        skew_val = tenant_data['value'].skew()
                        kurtosis_val = tenant_data['value'].kurtosis()
                        
                        # Estatísticas de autocorrelação
                        autocorr_1 = tenant_data['value'].autocorr(lag=1)
                        autocorr_5 = tenant_data['value'].autocorr(lag=5) if len(tenant_data) > 5 else np.nan
                        
                        # Adicionar ao conjunto de características
                        features.append({
                            'experimento': exp_name,
                            'métrica': metric,
                            'tenant': tenant,
                            'média': mean_val,
                            'desvio_padrão': std_val,
                            'máximo': max_val,
                            'mínimo': min_val,
                            'mediana': median_val,
                            'assimetria': skew_val,
                            'curtose': kurtosis_val,
                            'autocorr_1': autocorr_1,
                            'autocorr_5': autocorr_5
                        })
                    except Exception as e:
                        print(f"Erro ao extrair características para {metric}, {exp_name}, {tenant}: {str(e)}")
        
        if features:
            features_by_metric[metric] = pd.DataFrame(features)
    
    return features_by_metric


def compare_experiment_phases(
    experiments: Dict[str, Dict],
    metrics: List[str],
    phases_to_compare: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compara o impacto de diferentes fases entre experimentos.
    
    Args:
        experiments (Dict[str, Dict]): Dicionário com dados preprocessados
        metrics (List[str]): Lista de métricas para análise
        phases_to_compare (List[str], optional): Lista específica de fases para comparar
        
    Returns:
        Dict[str, Dict[str, Any]]: Resultados da comparação entre fases
    """
    from pipeline.analysis.phase_analysis import compare_phases
    
    phase_comparisons = {}
    
    for exp_name, exp_data in experiments.items():
        phase_comparisons[exp_name] = {}
        
        for metric in metrics:
            if metric in exp_data['processed_metrics']:
                df = exp_data['processed_metrics'][metric]
                
                # Definir fases para comparação
                if phases_to_compare is None:
                    # Usar todas as fases disponíveis
                    phases = sorted(df['phase'].unique())
                else:
                    # Filtrar fases especificadas que existem no dataset
                    phases = [p for p in phases_to_compare if p in df['phase'].unique()]
                
                if len(phases) < 2:
                    continue  # Pular se não houver fases suficientes
                
                # Comparar fases
                try:
                    phase_comparison_results = compare_phases(
                        df, phases=phases, metric_column='value'
                    )
                    
                    phase_comparisons[exp_name][metric] = phase_comparison_results
                except Exception as e:
                    print(f"Erro ao comparar fases para {metric} em {exp_name}: {str(e)}")
    
    return phase_comparisons


def compare_time_series_similarity(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    method: str = 'rmse',
    time_column: str = 'elapsed_minutes',
    value_column: str = 'value',
    tenant_column: str = 'tenant'
) -> float:
    """
    Compara a similaridade entre duas séries temporais.
    
    Args:
        df1 (DataFrame): Primeiro DataFrame com dados da série temporal
        df2 (DataFrame): Segundo DataFrame com dados da série temporal
        method (str): Método de similaridade ('rmse', 'mae', 'correlation')
        time_column (str): Nome da coluna de tempo
        value_column (str): Nome da coluna de valores
        tenant_column (str): Nome da coluna de tenant
        
    Returns:
        float: Valor de similaridade
    """
    # Identificar tenants comuns
    tenants1 = set(df1[tenant_column].unique())
    tenants2 = set(df2[tenant_column].unique())
    common_tenants = list(tenants1.intersection(tenants2))
    
    if not common_tenants:
        return np.nan  # Não há tenants comuns para comparação
    
    # Inicializar lista de similaridades por tenant
    tenant_similarities = []
    
    for tenant in common_tenants:
        series1 = df1[df1[tenant_column] == tenant].sort_values(time_column)
        series2 = df2[df2[tenant_column] == tenant].sort_values(time_column)
        
        # Verificar se temos dados suficientes
        if len(series1) < 5 or len(series2) < 5:
            continue
        
        # Alinhar séries temporais por interpolação em grade comum de tempo
        min_time = max(series1[time_column].min(), series2[time_column].min())
        max_time = min(series1[time_column].max(), series2[time_column].max())
        
        if min_time >= max_time:
            continue  # Não há sobreposição temporal
            
        # Criar grade de tempo comum
        time_grid = np.linspace(min_time, max_time, 100)
        
        # Interpolar valores
        values1 = np.interp(time_grid, series1[time_column], series1[value_column])
        values2 = np.interp(time_grid, series2[time_column], series2[value_column])
        
        # Calcular similaridade
        if method == 'rmse':
            similarity = mean_squared_error(values1, values2, squared=False)
        elif method == 'mae':
            similarity = mean_absolute_error(values1, values2)
        elif method == 'correlation':
            similarity = np.corrcoef(values1, values2)[0, 1]
        else:
            raise ValueError(f"Método de similaridade desconhecido: {method}")
            
        tenant_similarities.append(similarity)
    
    if not tenant_similarities:
        return np.nan
    
    # Retornar média das similaridades por tenant
    return np.nanmean(tenant_similarities)


def find_common_elements(experiments: Dict[str, Dict], key: str) -> List[Any]:
    """
    Encontra elementos comuns entre experimentos (ex: tenants, fases).
    
    Args:
        experiments (Dict[str, Dict]): Dicionário com dados de experimentos
        key (str): Chave para buscar nos metadados ('tenants', 'phases', etc.)
        
    Returns:
        List[Any]: Lista de elementos comuns
    """
    common_elements = None
    
    for exp_data in experiments.values():
        if key in exp_data.get('info', {}):
            elements = set(exp_data['info'][key])
            
            if common_elements is None:
                common_elements = elements
            else:
                common_elements = common_elements.intersection(elements)
    
    return list(common_elements) if common_elements else []
