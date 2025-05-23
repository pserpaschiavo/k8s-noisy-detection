"""
Módulo de normalização temporal para o experimento.

Este módulo fornece funções para normalizar timestamps e adicionar
campos de tempo decorrido aos DataFrames do experimento.
"""

import pandas as pd


def add_elapsed_time(df, group_by=['round', 'phase']):
    """
    Adiciona colunas de tempo decorrido, calculado desde o início de cada grupo.
    
    Args:
        df (DataFrame): DataFrame com dados do experimento
        group_by (list): Colunas para agrupar ao calcular tempo inicial
        
    Returns:
        DataFrame: DataFrame com colunas adicionais de tempo decorrido
    """
    # Cria uma cópia para não modificar o original
    result = df.copy()
    
    # Encontrar o timestamp inicial para cada grupo
    start_times = df.groupby(group_by)['datetime'].min().reset_index()
    
    # Renomear a coluna para facilitar o merge
    start_times = start_times.rename(columns={'datetime': 'start_time'})
    
    # Mesclar com o DataFrame original
    result = pd.merge(result, start_times, on=group_by)
    
    # Calcular tempo decorrido em segundos desde o início de cada grupo
    result['elapsed_seconds'] = (result['datetime'] - result['start_time']).dt.total_seconds()
    
    # Calcular tempo decorrido em minutos (para facilitar a visualização)
    result['elapsed_minutes'] = result['elapsed_seconds'] / 60.0
    
    return result


def add_experiment_elapsed_time(df, experiment_start_time=None, group_by=None):
    """
    Adiciona coluna de tempo decorrido desde o início do experimento ou round.
    Se experiment_start_time for fornecido, usa ele como referência global.
    Caso contrário, agrupa por `group_by` para encontrar o tempo inicial de cada grupo.

    Args:
        df (DataFrame): DataFrame com dados do experimento.
        experiment_start_time (datetime, optional): Timestamp do início global do experimento.
        group_by (list, optional): Colunas para agrupar ao calcular tempo inicial se 
                                   experiment_start_time não for fornecido (geralmente ['round']).
                                   Padrão é ['round'] se experiment_start_time for None.
        
    Returns:
        DataFrame: DataFrame com coluna adicional de tempo decorrido do experimento.
    """
    result = df.copy()

    if experiment_start_time is not None:
        # Usar o tempo de início global fornecido
        result['experiment_start_time'] = experiment_start_time
    else:
        # Determinar o tempo de início por grupo
        if group_by is None:
            group_by = ['round'] # Padrão se não especificado e experiment_start_time é None
        
        if not all(col in df.columns for col in group_by):
            # Se as colunas de agrupamento não existirem, calcula o tempo global para todo o DataFrame
            # Isso pode acontecer se, por exemplo, os dados já estiverem filtrados para um único round/grupo
            global_start_time = df['datetime'].min()
            result['experiment_start_time'] = global_start_time
        else:
            start_times = df.groupby(group_by)['datetime'].min().reset_index()
            start_times = start_times.rename(columns={'datetime': 'experiment_start_time'})
            result = pd.merge(result, start_times, on=group_by, how='left')

    # Calcular tempo decorrido em segundos desde o início de cada grupo/global
    result['experiment_elapsed_seconds'] = (result['datetime'] - result['experiment_start_time']).dt.total_seconds()
    result['experiment_elapsed_minutes'] = result['experiment_elapsed_seconds'] / 60.0
    
    # Add experiment_elapsed_time as an alias for experiment_elapsed_seconds (for compatibility)
    result['experiment_elapsed_time'] = result['experiment_elapsed_seconds']
    
    return result


def add_phase_markers(df, phase_column='phase', phase_display_names=None):
    """
    Adiciona marcadores de início e fim de fase para facilitar a visualização.
    
    Args:
        df (DataFrame): DataFrame com dados do experimento
        phase_column (str): Nome da coluna que contém o nome da fase
        phase_display_names (dict, optional): Dicionário para mapear nomes de fase brutos para nomes de exibição.
        
    Returns:
        DataFrame: O mesmo DataFrame com coluna adicional de nome simplificado da fase
        dict: Dicionário com marcadores de início de fase para uso em gráficos
    """
    # Extrair nomes simplificados das fases
    if phase_display_names:
        df['phase_name'] = df[phase_column].apply(lambda x: phase_display_names.get(x, x.split('-')[-1].strip()) if isinstance(x, str) else x)
    else:
        df['phase_name'] = df[phase_column].apply(lambda x: x.split('-')[-1].strip() if isinstance(x, str) else x)
    
    # Agrupar por round e calcular início de cada fase
    phase_markers = {}
    
    for round_name, group in df.groupby('round'):
        round_markers = {}
        for phase, phase_df in group.groupby('phase'):
            if len(phase_df) > 0:
                start_time = phase_df['experiment_elapsed_minutes'].min()
                round_markers[phase] = start_time
        
        phase_markers[round_name] = round_markers
    
    return df, phase_markers


def merge_phase_info(df, phase_info=None):
    """
    Adiciona informações sobre duração de fase ao DataFrame.
    
    Args:
        df (DataFrame): DataFrame com dados do experimento
        phase_info (dict): Dicionário com informações sobre as fases
                           (se None, será inferido dos dados)
        
    Returns:
        DataFrame: DataFrame com informações adicionais sobre as fases
    """
    result = df.copy()
    
    if phase_info is None:
        # Inferir informações de fase dos dados
        phase_durations = {}
        
        for (round_name, phase_name), group in df.groupby(['round', 'phase']):
            duration = group['elapsed_minutes'].max() - group['elapsed_minutes'].min()
            if round_name not in phase_durations:
                phase_durations[round_name] = {}
            phase_durations[round_name][phase_name] = duration
    else:
        phase_durations = phase_info
    
    # Adicionar informações de duração
    phase_duration_list = []
    
    for idx, row in df.iterrows():
        round_name = row['round']
        phase_name = row['phase']
        
        if round_name in phase_durations and phase_name in phase_durations[round_name]:
            duration = phase_durations[round_name][phase_name]
            phase_duration_list.append(duration)
        else:
            phase_duration_list.append(None)
    
    result['phase_duration_minutes'] = phase_duration_list
    
    return result


def normalize_time(df, time_column='datetime', group_by=None, method='elapsed_seconds'):
    """
    Normaliza a coluna de tempo de acordo com o método especificado.

    Args:
        df (DataFrame): DataFrame de entrada.
        time_column (str): Nome da coluna de timestamp (datetime objects).
        group_by (list, optional): Colunas para agrupar antes de normalizar.
                                   Se None, normaliza globalmente.
        method (str): Método de normalização:
                      'elapsed_seconds': Segundos desde o primeiro timestamp (no grupo ou global).
                      'elapsed_minutes': Minutos desde o primeiro timestamp.
                      'experiment_elapsed_seconds': Segundos desde o início do experimento (requer 'round' em group_by).
                      'experiment_elapsed_minutes': Minutos desde o início do experimento.

    Returns:
        DataFrame: DataFrame com a nova coluna de tempo normalizado.
    """
    if time_column not in df.columns:
        raise ValueError(f"Coluna de tempo '{time_column}' não encontrada no DataFrame.")
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        raise ValueError(f"Coluna de tempo '{time_column}' deve ser do tipo datetime.")

    df_copy = df.copy()

    if method in ['elapsed_seconds', 'elapsed_minutes']:
        if group_by:
            df_copy['start_time_group'] = df_copy.groupby(group_by)[time_column].transform('min')
            elapsed = (df_copy[time_column] - df_copy['start_time_group']).dt.total_seconds()
            df_copy.drop(columns=['start_time_group'], inplace=True)
        else:
            elapsed = (df_copy[time_column] - df_copy[time_column].min()).dt.total_seconds()
        
        if method == 'elapsed_seconds':
            df_copy['normalized_time'] = elapsed
        else: # elapsed_minutes
            df_copy['normalized_time'] = elapsed / 60.0

    elif method in ['experiment_elapsed_seconds', 'experiment_elapsed_minutes']:
        # Este método é mais específico e geralmente usa add_experiment_elapsed_time
        # Aqui, replicamos uma lógica similar para fins de uma função genérica 'normalize_time'
        # Assumimos que 'round' (ou um agrupamento similar de alto nível) está em group_by
        if not group_by or not any(g_col in df_copy.columns for g_col in group_by):
            # Se não há group_by, calcula desde o início global do dataset
             exp_start_col_name = 'experiment_start_time_calc'
             df_copy[exp_start_col_name] = df_copy[time_column].min()
        else:
            exp_start_col_name = 'experiment_start_time_calc'
            df_copy[exp_start_col_name] = df_copy.groupby(group_by)[time_column].transform('min')

        elapsed_exp = (df_copy[time_column] - df_copy[exp_start_col_name]).dt.total_seconds()
        df_copy.drop(columns=[exp_start_col_name], inplace=True)

        if method == 'experiment_elapsed_seconds':
            df_copy['normalized_time'] = elapsed_exp
        else: # experiment_elapsed_minutes
            df_copy['normalized_time'] = elapsed_exp / 60.0
    else:
        raise ValueError(f"Método de normalização desconhecido: {method}")

    return df_copy
