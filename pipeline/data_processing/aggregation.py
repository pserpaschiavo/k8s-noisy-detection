"""
Módulo de agregação de dados para o experimento de noisy neighbors.

Este módulo fornece funções para agregar e resumir dados de métricas
do experimento por tenant, fase, round, etc.
"""

import pandas as pd
import numpy as np


def calculate_tenant_stats(df, value_column='value', group_columns=['tenant', 'round', 'phase']):
    """
    Calcula estatísticas resumidas para os valores de métricas, agrupados por tenant, round e fase.
    
    Args:
        df (DataFrame): DataFrame com dados de métricas
        value_column (str): Coluna que contém os valores da métrica
        group_columns (list): Colunas para agrupar (por padrão, tenant, round e fase)
        
    Returns:
        DataFrame: DataFrame com estatísticas calculadas
    """
    stats = df.groupby(group_columns)[value_column].agg([
        'mean', 'median', 'std', 'min', 'max', 'count',
        lambda x: np.percentile(x, 25),
        lambda x: np.percentile(x, 75),
        lambda x: np.percentile(x, 95)
    ]).reset_index()
    
    # Renomear as colunas calculadas
    stats = stats.rename(columns={
        '<lambda_0>': 'percentile_25',
        '<lambda_1>': 'percentile_75',
        '<lambda_2>': 'percentile_95'
    })
    
    return stats


def calculate_inter_tenant_impact(df, noisy_tenant='tenant-b', attack_phase='2 - Attack', 
                                  value_column='mean', baseline_phase='1 - Baseline'):
    """
    Calcula o impacto do tenant barulhento nos outros tenants.
    
    Args:
        df (DataFrame): DataFrame com estatísticas resumidas por tenant e fase
        noisy_tenant (str): Nome do tenant barulhento
        attack_phase (str): Nome da fase de ataque
        value_column (str): Nome da coluna com valores de métrica (geralmente 'mean')
        baseline_phase (str): Nome da fase de baseline
        
    Returns:
        DataFrame: DataFrame com o impacto calculado
    """
    # Filtrar apenas as fases de baseline e ataque
    filtered_df = df[df['phase'].isin([baseline_phase, attack_phase])].copy()
    
    # Pivotar para ter colunas para cada fase
    pivot = filtered_df.pivot_table(
        index=['tenant', 'round'],
        columns='phase',
        values=value_column
    ).reset_index()
    
    # Calcular o impacto (variação percentual)
    pivot['impact_percent'] = ((pivot[attack_phase] - pivot[baseline_phase]) / pivot[baseline_phase]) * 100
    
    # Remover o tenant barulhento se necessário
    if noisy_tenant is not None:
        impact_summary = pivot[pivot['tenant'] != noisy_tenant]
    else:
        impact_summary = pivot
    
    return impact_summary


def calculate_recovery_effectiveness(df, value_column='mean', 
                                   baseline_phase='1 - Baseline',
                                   attack_phase='2 - Attack',
                                   recovery_phase='3 - Recovery'):
    """
    Calcula a efetividade da recuperação após a fase de ataque.
    
    Args:
        df (DataFrame): DataFrame com estatísticas resumidas por tenant e fase
        value_column (str): Nome da coluna com valores de métrica (geralmente 'mean')
        baseline_phase (str): Nome da fase de baseline
        attack_phase (str): Nome da fase de ataque
        recovery_phase (str): Nome da fase de recuperação
        
    Returns:
        DataFrame: DataFrame com a efetividade da recuperação calculada
    """
    # Filtrar apenas as fases relevantes
    filtered_df = df[df['phase'].isin([baseline_phase, attack_phase, recovery_phase])].copy()
    
    # Pivotar para ter colunas para cada fase
    pivot = filtered_df.pivot_table(
        index=['tenant', 'round'],
        columns='phase',
        values=value_column
    ).reset_index()
    
    # Calcular a degradação durante o ataque (quanto piorou)
    pivot['attack_degradation'] = ((pivot[attack_phase] - pivot[baseline_phase]) / pivot[baseline_phase]) * 100
    
    # Calcular quanto da degradação foi recuperada
    pivot['recovery_percent'] = ((pivot[recovery_phase] - pivot[attack_phase]) / (pivot[baseline_phase] - pivot[attack_phase])) * 100
    
    # Calcular quanto ainda falta para voltar ao baseline (em percentual)
    pivot['baseline_diff_percent'] = ((pivot[recovery_phase] - pivot[baseline_phase]) / pivot[baseline_phase]) * 100
    
    return pivot


def aggregate_data_by_custom_elements(df, aggregation_keys=None, elements_to_aggregate=None, value_column='value', agg_functions=None):
    """
    Agrega dados com base em chaves de agregação e elementos específicos definidos pelo usuário.

    Args:
        df (DataFrame): DataFrame com dados de métricas.
        aggregation_keys (list): Lista de colunas para agrupar (ex: ["tenant", "phase", "custom_group"]).
        elements_to_aggregate (list or dict): Lista de valores específicos para filtrar na primeira chave de agregação,
                                             ou um dicionário onde as chaves são nomes de colunas de `aggregation_keys`
                                             e os valores são listas de elementos para filtrar nessas colunas.
                                             Se None, todos os elementos são considerados.
        value_column (str): Coluna que contém os valores da métrica.
        agg_functions (list or dict): Lista de funções de agregação (ex: ['mean', 'std']) ou um dicionário
                                      mapeando colunas para funções de agregação.
                                      Padrão é ['mean', 'std', 'count'].

    Returns:
        DataFrame: DataFrame agregado.
    """
    if aggregation_keys is None:
        aggregation_keys = ['tenant', 'phase'] # Default aggregation

    df_filtered = df.copy()

    if elements_to_aggregate:
        if isinstance(elements_to_aggregate, dict):
            for filter_column, filter_values in elements_to_aggregate.items():
                if filter_column in df_filtered.columns and filter_values:
                    df_filtered = df_filtered[df_filtered[filter_column].isin(filter_values)]
        elif isinstance(elements_to_aggregate, list) and aggregation_keys:
            # Mantém o comportamento original se for uma lista: filtra na primeira chave de agregação
            filter_column = aggregation_keys[0]
            if filter_column in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[filter_column].isin(elements_to_aggregate)]
        # Se elements_to_aggregate não for dict nem list, ou se for uma lista vazia, não faz nada aqui.
    
    # Se após a filtragem o DataFrame estiver vazio, retorna um DataFrame vazio.
    if df_filtered.empty:
        return pd.DataFrame(columns=df.columns.tolist() + ['mean', 'std', 'count', 'percentile_25', 'percentile_75', 'percentile_95'])

    # Calcular estatísticas
    agg_stats = df_filtered.groupby(aggregation_keys)[value_column].agg([
        'mean', 'std', 'count',
        lambda x: np.percentile(x, 25),
        lambda x: np.percentile(x, 75),
        lambda x: np.percentile(x, 95)
    ]).reset_index()

    # Renomear as colunas calculadas
    agg_stats = agg_stats.rename(columns={
        '<lambda_0>': 'percentile_25',
        '<lambda_1>': 'percentile_75',
        '<lambda_2>': 'percentile_95'
    })

    return agg_stats


def aggregate_by_time(df, time_column='elapsed_minutes', value_column='value', agg_interval='5T', agg_funcs=None):
    """
    Agrega dados por intervalos de tempo.

    Args:
        df (DataFrame): DataFrame com coluna de tempo (datetime ou timedelta) e valor.
        time_column (str): Nome da coluna de tempo a ser usada para resampling.
                           Deve ser datetime ou convertível para timedelta se for numérico (minutos, segundos).
        value_column (str): Nome da coluna de valor a ser agregada.
        agg_interval (str): String de intervalo de agregação (ex: '1T' para 1 minuto, '5S' para 5 segundos).
                            Usado com pd.Grouper ou resample.
        agg_funcs (list or dict, optional): Funções de agregação a aplicar (ex: ['mean', 'std']).
                                            Defaults to ['mean'].

    Returns:
        DataFrame: DataFrame agregado por tempo.
    """
    if agg_funcs is None:
        agg_funcs = ['mean']

    if df.empty or time_column not in df.columns or value_column not in df.columns:
        # Retorna um DataFrame vazio com colunas esperadas se os dados de entrada não forem válidos
        return pd.DataFrame()

    df_copy = df.copy()

    # Certificar que a coluna de tempo é datetime para usar resample ou pd.Grouper
    if pd.api.types.is_numeric_dtype(df_copy[time_column]):
        df_copy[time_column] = pd.to_timedelta(df_copy[time_column], unit='m')
        base_timestamp = df_copy['datetime'].min() if 'datetime' in df_copy.columns else pd.Timestamp("2000-01-01")
        df_copy['resample_time'] = base_timestamp + df_copy[time_column]
        time_col_for_resample = 'resample_time'
    elif pd.api.types.is_datetime64_any_dtype(df_copy[time_column]):
        time_col_for_resample = time_column
    else:
        raise ValueError(f"A coluna de tempo '{time_column}' deve ser numérica (minutos/segundos) ou datetime.")

    grouping_cols = [col for col in ['tenant', 'phase', 'round'] if col in df_copy.columns]

    if grouping_cols:
        try:
            aggregated_df = df_copy.groupby(
                [pd.Grouper(key=time_col_for_resample, freq=agg_interval)] + grouping_cols
            )[value_column].agg(agg_funcs).reset_index()
        except Exception as e:
            print(f"Erro ao agregar com pd.Grouper e grupos adicionais: {e}. Tentando resample global.")
            df_copy = df_copy.set_index(time_col_for_resample)
            aggregated_df = df_copy[value_column].resample(agg_interval).agg(agg_funcs).reset_index()
    else:
        df_copy = df_copy.set_index(time_col_for_resample)
        aggregated_df = df_copy[value_column].resample(agg_interval).agg(agg_funcs).reset_index()

    return aggregated_df
