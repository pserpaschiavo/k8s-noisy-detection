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

def load_multiple_experiments(experiment_paths: List[str]) -> Dict[str, Dict]:
    """
    Carrega dados de múltiplos experimentos para comparação.
    
    Args:
        experiment_paths (List[str]): Lista de caminhos para os diretórios dos experimentos
        
    Returns:
        Dict[str, Dict]: Dicionário com dados e metadados dos experimentos
    """
    from ..data_processing.consolidation import load_experiment_data
    
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
        for metric_name, rounds_data_dict in selected_metrics.items():
            processed_rounds = {}
            if not isinstance(rounds_data_dict, dict):
                pass

            for round_name, df_round in rounds_data_dict.items():
                # DEBUG: Check data for comparison_exp_1 and memory_usage after processing in preprocess_experiments
                if exp_name == "comparison_exp_1" and metric_name == "memory_usage":
                    print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: START of round processing.")
                    print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: Initial df_round empty: {df_round.empty}, shape: {df_round.shape}")
                    if not df_round.empty:
                        print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: Initial df_round columns: {df_round.columns.tolist()}")
                        if 'value' in df_round.columns:
                             print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: Initial df_round 'value' head:\n{df_round['value'].head().to_string()}")

                # Normalizar tempo se solicitado
                if normalize_timestamps:
                    # Ensure df_round is a DataFrame before calling normalize_time
                    if not isinstance(df_round, pd.DataFrame):
                        print(f"Skipping normalization for metric '{metric_name}', round '{round_name}' in exp '{exp_name}': df_round is not a DataFrame, but {type(df_round)}")
                        processed_df_round = df_round # Assign to processed_df_round before continue
                        if exp_name == "comparison_exp_1" and metric_name == "memory_usage":
                            print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: SKIPPED NORMALIZATION (not a DataFrame).")
                        continue
                    # Call normalize_time correctly, assuming default time_column='datetime'
                    processed_df_round = normalize_time(df_round.copy())
                    if exp_name == "comparison_exp_1" and metric_name == "memory_usage":
                        print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: After normalize_time. Empty: {processed_df_round.empty}, Shape: {processed_df_round.shape}")
                        if not processed_df_round.empty:
                            print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: Columns after norm: {processed_df_round.columns.tolist()}")
                else:
                    processed_df_round = df_round.copy()
                    if exp_name == "comparison_exp_1" and metric_name == "memory_usage":
                        print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: Normalization skipped by flag.")

                # Aplicar filtro de tenants, se fornecido e a coluna 'tenant' existir
                if tenants_filter and 'tenant' in processed_df_round.columns:
                    processed_df_round = processed_df_round[processed_df_round['tenant'].isin(tenants_filter)]
                    if exp_name == "comparison_exp_1" and metric_name == "memory_usage":
                        print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: After tenant filter. Empty: {processed_df_round.empty}, Shape: {processed_df_round.shape}")
                
                # Aplicar filtro de rounds, se fornecido e a coluna 'round' existir
                if rounds_filter and 'round' in processed_df_round.columns:
                    processed_df_round = processed_df_round[processed_df_round['round'].isin(rounds_filter)]
                elif rounds_filter and round_name not in rounds_filter:
                    if exp_name == "comparison_exp_1" and metric_name == "memory_usage":
                        print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: SKIPPED due to round_filter (round_name not in list).")
                    continue

                if processed_df_round.empty:
                    if exp_name == "comparison_exp_1" and metric_name == "memory_usage":
                        print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: SKIPPED (empty after filters).")
                    continue
                
                # Adicionar coluna de tempo decorrido em minutos
                if 'datetime' not in processed_df_round.columns:
                    print(f"Warning: 'datetime' column not found for metric '{metric_name}', round '{round_name}' in exp '{exp_name}'. Skipping elapsed time calculation.")
                    if exp_name == "comparison_exp_1" and metric_name == "memory_usage":
                        print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: 'datetime' column missing for elapsed time.")
                else:
                    processed_df_round['elapsed_minutes'] = (
                        processed_df_round['datetime'] - processed_df_round['datetime'].min()
                    ).dt.total_seconds() / 60
                    if exp_name == "comparison_exp_1" and metric_name == "memory_usage":
                        print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: After elapsed_minutes. Empty: {processed_df_round.empty}, Shape: {processed_df_round.shape}")
                        if not processed_df_round.empty and 'elapsed_minutes' in processed_df_round.columns:
                             print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: 'elapsed_minutes' head:\n{processed_df_round['elapsed_minutes'].head().to_string()}")
                
                # Agregar dados se solicitado
                if aggregate_data:
                    if 'elapsed_minutes' not in processed_df_round.columns:
                        print(f"Warning: 'elapsed_minutes' column not found for metric '{metric_name}', round '{round_name}' in exp '{exp_name}'. Skipping aggregation.")
                        if exp_name == "comparison_exp_1" and metric_name == "memory_usage":
                            print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: 'elapsed_minutes' column missing for aggregation.")
                    else:
                        # Ensure 'value' column exists if it's the default for aggregation, or specify another
                        value_col_for_agg = 'value' # Default assumption
                        if value_col_for_agg not in processed_df_round.columns:
                             print(f"Warning: Default value column '{value_col_for_agg}' not found for aggregation in metric '{metric_name}', round '{round_name}', exp '{exp_name}'. Columns: {processed_df_round.columns}. Skipping aggregation.")
                             if exp_name == "comparison_exp_1" and metric_name == "memory_usage":
                                 print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: Default value col '{value_col_for_agg}' missing for aggregation.")
                        else:
                            processed_df_round = aggregate_by_time(
                                processed_df_round,
                                time_column='elapsed_minutes',
                                agg_interval=agg_freq,
                                value_column=value_col_for_agg # Explicitly pass value column
                            )
                            if exp_name == "comparison_exp_1" and metric_name == "memory_usage":
                                print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: After aggregate_by_time. Empty: {processed_df_round.empty}, Shape: {processed_df_round.shape}")
                                if not processed_df_round.empty:
                                    print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: Columns after agg: {processed_df_round.columns.tolist()}")
                else:
                    if exp_name == "comparison_exp_1" and metric_name == "memory_usage":
                        print(f"DEBUG PREPROCESS [{exp_name}][{metric_name}][{round_name}]: Aggregation skipped by flag.")
                
                # DEBUG: Check data for comparison_exp_1 and memory_usage after processing in preprocess_experiments
                if exp_name == "comparison_exp_1" and metric_name == "memory_usage":
                    print(f"DEBUG PREPROCESS FINAL [{exp_name}][{metric_name}][{round_name}]: df after ALL processing. Empty: {processed_df_round.empty}, Shape: {processed_df_round.shape}")
                    if not processed_df_round.empty:
                        print(f"DEBUG PREPROCESS FINAL [{exp_name}][{metric_name}][{round_name}]: Columns: {processed_df_round.columns.tolist()}")
                        if 'mean' in processed_df_round.columns: # After aggregation, 'mean' is expected
                            print(f"DEBUG PREPROCESS FINAL [{exp_name}][{metric_name}][{round_name}]: 'mean' column head:\n{processed_df_round['mean'].head().to_string()}")
                        elif 'value' in processed_df_round.columns: # If no aggregation, 'value' might still be there
                            print(f"DEBUG PREPROCESS FINAL [{exp_name}][{metric_name}][{round_name}]: 'value' column head (no 'mean' found):\n{processed_df_round['value'].head().to_string()}")
                        else:
                            print(f"DEBUG PREPROCESS FINAL [{exp_name}][{metric_name}][{round_name}]: Neither 'mean' nor 'value' column found in final processed_df_round.")
                    else:
                        print(f"DEBUG PREPROCESS FINAL [{exp_name}][{metric_name}][{round_name}]: DataFrame is empty.")

                processed_rounds[round_name] = processed_df_round
            
            if processed_rounds:
                processed_metrics[metric_name] = processed_rounds
        
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
        all_stats_for_metric = []
        
        for exp_name, exp_data in experiments.items():
            if metric in exp_data['processed_metrics']:
                rounds_data_dict = exp_data['processed_metrics'][metric]
                if not isinstance(rounds_data_dict, dict):
                    print(f"Warning: Expected a dictionary of round DataFrames for metric '{metric}' in experiment '{exp_name}', but got {type(rounds_data_dict)}. Skipping this metric for this experiment.")
                    continue

                for round_name, df_round in rounds_data_dict.items():
                    if not isinstance(df_round, pd.DataFrame):
                        print(f"Warning: Expected a DataFrame for round '{round_name}' of metric '{metric}' in experiment '{exp_name}', but got {type(df_round)}. Skipping this round.")
                        continue
                    
                    if df_round.empty:
                        continue

                    value_col_to_use = None
                    if 'mean' in df_round.columns:
                        value_col_to_use = 'mean'
                    elif 'value' in df_round.columns:
                        value_col_to_use = 'value'
                    else:
                        print(f"Warning: Neither 'mean' nor 'value' column found in DataFrame for round '{round_name}', metric '{metric}', experiment '{exp_name}'. Columns: {df_round.columns}. Skipping stats for this round.")
                        continue

                    current_group_by = group_by.copy() if group_by else []
                    valid_group_by = [gb for gb in current_group_by if gb in df_round.columns]
                    if not valid_group_by and current_group_by:
                        print(f"Warning: Group_by columns {current_group_by} not found in df_round for metric '{metric}', round '{round_name}'. Calculating global stats.")

                    if valid_group_by:
                        stats = df_round.groupby(valid_group_by)[value_col_to_use].agg([
                            ('média', 'mean'),
                            ('mediana', 'median'),
                            ('desvio_padrão', 'std'),
                            ('mínimo', 'min'),
                            ('máximo', 'max'),
                            ('contagem', 'count')
                        ]).reset_index()
                    else:
                        stats = pd.DataFrame({
                            'média': [df_round[value_col_to_use].mean()],
                            'mediana': [df_round[value_col_to_use].median()],
                            'desvio_padrão': [df_round[value_col_to_use].std()],
                            'mínimo': [df_round[value_col_to_use].min()],
                            'máximo': [df_round[value_col_to_use].max()],
                            'contagem': [df_round[value_col_to_use].count()]
                        })
                    
                    stats['experimento'] = exp_name
                    stats['round'] = round_name
                    
                    all_stats_for_metric.append(stats)
        
        if all_stats_for_metric:
            results[metric] = pd.concat(all_stats_for_metric, ignore_index=True)
    
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
    print(f"DEBUG COMPARE_DIST: Entering for metric: {metric}, tenants_filter: {tenants_filter}, rounds_filter: {rounds_filter}, phase: {phase}")
    for exp_name, exp_data in experiments.items():
        print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}]: Processing experiment.")
        if metric in exp_data.get('processed_metrics', {}):
            rounds_data_dict = exp_data['processed_metrics'][metric]
            if not isinstance(rounds_data_dict, dict):
                print(f"Warning: Expected a dictionary of round DataFrames for metric '{metric}' in experiment '{exp_name}', but got {type(rounds_data_dict)}. Skipping this metric for this experiment.")
                continue

            all_rounds_df_list = []
            print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}]: rounds_data_dict keys: {list(rounds_data_dict.keys())}")
            for round_name, df_round in rounds_data_dict.items():
                print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}][{round_name}]: Processing round.")
                if not isinstance(df_round, pd.DataFrame):
                    print(f"Warning: Expected a DataFrame for round '{round_name}' of metric '{metric}' in experiment '{exp_name}', but got {type(df_round)}. Skipping this round.")
                    continue
                
                print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}][{round_name}]: Initial df_round (from preprocess_experiments). Empty: {df_round.empty}, Shape: {df_round.shape}, Columns: {df_round.columns.tolist()}")
                if not df_round.empty:
                    # Print head of value or mean column if they exist, before filtering
                    if 'value' in df_round.columns:
                        print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}][{round_name}]: df_round 'value' column head BEFORE filters:\n{df_round['value'].head().to_string()}")
                    elif 'mean' in df_round.columns: 
                        print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}][{round_name}]: df_round 'mean' column head BEFORE filters (value not found):\n{df_round['mean'].head().to_string()}")
                    else:
                        print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}][{round_name}]: Neither 'value' nor 'mean' column found in df_round BEFORE filters.")


                df_round_filtered = df_round.copy()
                if tenants_filter and 'tenant' in df_round_filtered.columns:
                    df_round_filtered = df_round_filtered[df_round_filtered['tenant'].isin(tenants_filter)]
                    print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}][{round_name}]: After tenant filter ({tenants_filter}). Empty: {df_round_filtered.empty}, Shape: {df_round_filtered.shape}")
                
                if rounds_filter:
                    # Check if 'round' column exists for filtering, or if round_name itself should be used
                    if 'round' in df_round_filtered.columns: # If 'round' column was added during preprocessing
                        df_round_filtered = df_round_filtered[df_round_filtered['round'].isin(rounds_filter)]
                        print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}][{round_name}]: After round filter on 'round' column ({rounds_filter}). Empty: {df_round_filtered.empty}, Shape: {df_round_filtered.shape}")
                    elif round_name not in rounds_filter: # Filter based on the round_name (key of the dict)
                        print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}][{round_name}]: Skipping round (round_name '{round_name}' not in rounds_filter {rounds_filter}).")
                        continue 
                    else: # round_name is in rounds_filter, no 'round' column to filter on, so proceed
                         print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}][{round_name}]: round_name '{round_name}' is in rounds_filter {rounds_filter}. No 'round' column for further filtering here.")


                if phase and 'phase' in df_round_filtered.columns:
                    df_round_filtered = df_round_filtered[df_round_filtered['phase'] == phase]
                    print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}][{round_name}]: After phase filter ({phase}). Empty: {df_round_filtered.empty}, Shape: {df_round_filtered.shape}")
                
                if not df_round_filtered.empty:
                    all_rounds_df_list.append(df_round_filtered)
                    print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}][{round_name}]: Appended df_round_filtered to all_rounds_df_list. List size: {len(all_rounds_df_list)}")
                else:
                    print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}][{round_name}]: df_round_filtered is EMPTY after filters. Not appending.")
            
            if not all_rounds_df_list:
                print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}]: all_rounds_df_list is EMPTY. Skipping experiment for this metric.")
                continue
            
            df_combined = pd.concat(all_rounds_df_list, ignore_index=True)
            print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}]: df_combined after concat. Empty: {df_combined.empty}, Shape: {df_combined.shape}, Columns: {df_combined.columns.tolist()}")
            if not df_combined.empty:
                if 'value' in df_combined.columns:
                    print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}]: df_combined 'value' head:\n{df_combined['value'].head().to_string()}")
                if 'mean' in df_combined.columns: 
                    print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}]: df_combined 'mean' head:\n{df_combined['mean'].head().to_string()}")
            
            value_col_for_dist = None
            if 'mean' in df_combined.columns: 
                value_col_for_dist = 'mean'
            elif 'value' in df_combined.columns:
                value_col_for_dist = 'value'
            else:
                print(f"Warning: Neither 'mean' nor 'value' column found for distribution comparison in metric '{metric}', experiment '{exp_name}'. Columns: {df_combined.columns}. Skipping this experiment for this metric.")
                continue
            print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}]: Selected value_col_for_dist: '{value_col_for_dist}'")

            if not df_combined.empty and value_col_for_dist in df_combined.columns:
                # Ensure the selected column actually has data before trying to dropna and append
                if df_combined[value_col_for_dist].isnull().all():
                    print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}]: The selected value_col_for_dist '{value_col_for_dist}' contains ALL NaN values. Not adding to plot_data.")
                else:
                    series_to_add = df_combined[value_col_for_dist].dropna()
                    print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}]: series_to_add (column '{value_col_for_dist}' after dropna). Length: {len(series_to_add)}")
                    if not series_to_add.empty:
                        print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}]: series_to_add head:\n{series_to_add.head().to_string()}")
                        plot_data['series'].append(series_to_add)
                        plot_data['labels'].append(exp_name)
                        print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}]: Added data to plot_data. Current plot_data labels: {plot_data['labels']}, num_series: {len(plot_data['series'])}")
                    else:
                        print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}]: series_to_add is EMPTY after dropna. Not adding to plot_data.")
            else:
                print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}]: df_combined is empty OR value_col_for_dist ('{value_col_for_dist}') not in columns OR column contains all NaNs. df_combined columns: {df_combined.columns.tolist()}. Not adding to plot_data.")
        else:
            print(f"DEBUG COMPARE_DIST [{exp_name}][{metric}]: Metric not found in processed_metrics. Skipping.")
    
    print(f"DEBUG COMPARE_DIST: Exiting for metric: {metric}. Final plot_data labels: {plot_data['labels']}, num_series: {len(plot_data['series'])}")
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
