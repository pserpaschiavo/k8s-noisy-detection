"""
Módulo para detecção avançada de anomalias em dados do experimento de noisy neighbors.

Este módulo implementa algoritmos de machine learning para detecção de anomalias,
pontos de mudança e comportamentos anormais nas métricas coletadas.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
import ruptures as rpt
from tslearn.clustering import TimeSeriesKMeans
import warnings
warnings.filterwarnings('ignore')


def detect_anomalies_isolation_forest(df, metric_column='value', contamination=0.05, group_by=None):
    """
    Detecta anomalias usando o algoritmo Isolation Forest.
    
    Args:
        df (DataFrame): DataFrame com dados da métrica
        metric_column (str): Coluna com os valores da métrica
        contamination (float): Proporção esperada de anomalias nos dados
        group_by (list): Colunas para agrupar os dados (ex: ['tenant', 'phase'])
        
    Returns:
        DataFrame: DataFrame original com coluna adicional 'is_anomaly' e 'anomaly_score'
    """
    # Cria uma cópia para não modificar o original
    result = df.copy()
    
    # Adicionar coluna para anomalias
    result['is_anomaly_if'] = False
    result['anomaly_score_if'] = 0.0
    
    # Se não houver agrupamento, tratar todo o conjunto de dados
    if group_by is None:
        # Criar e treinar o modelo
        X = result[metric_column].values.reshape(-1, 1)
        
        # Pular se houver valores faltantes
        if np.isnan(X).any():
            X = result[metric_column].dropna().values.reshape(-1, 1)
            if len(X) == 0:
                return result
        
        model = IsolationForest(contamination=contamination, random_state=42)
        result['anomaly_score_if'] = model.fit_predict(X)
        result['is_anomaly_if'] = result['anomaly_score_if'] == -1
        
        # Converter score para valores positivos (maior = mais anômalo)
        model_decision = model.decision_function(X) * -1
        result.loc[result.index[~np.isnan(result[metric_column])], 'anomaly_score_if'] = model_decision
    else:
        # Para cada grupo, treinar um modelo separado
        for group_name, group in result.groupby(group_by):
            # Pular se o grupo for muito pequeno
            if len(group) < 10:
                continue
            
            # Preparar os dados
            X = group[metric_column].values.reshape(-1, 1)
            
            # Pular se houver valores faltantes
            if np.isnan(X).any():
                continue
            
            # Criar e treinar o modelo
            model = IsolationForest(contamination=contamination, random_state=42)
            predictions = model.fit_predict(X)
            
            # Atualizar o DataFrame de resultado
            index_in_group = group.index
            result.loc[index_in_group, 'is_anomaly_if'] = (predictions == -1)
            
            # Converter score para valores positivos (maior = mais anômalo)
            model_decision = model.decision_function(X) * -1
            result.loc[index_in_group, 'anomaly_score_if'] = model_decision
    
    return result


def detect_anomalies_local_outlier_factor(df, metric_column='value', n_neighbors=20, contamination=0.05, group_by=None):
    """
    Detecta anomalias usando o algoritmo Local Outlier Factor (LOF).
    
    Args:
        df (DataFrame): DataFrame com dados da métrica
        metric_column (str): Coluna com os valores da métrica
        n_neighbors (int): Número de vizinhos para o algoritmo LOF
        contamination (float): Proporção esperada de anomalias nos dados
        group_by (list): Colunas para agrupar os dados (ex: ['tenant', 'phase'])
        
    Returns:
        DataFrame: DataFrame original com coluna adicional 'is_anomaly_lof' e 'anomaly_score_lof'
    """
    # Cria uma cópia para não modificar o original
    result = df.copy()
    
    # Adicionar coluna para anomalias
    result['is_anomaly_lof'] = False
    result['anomaly_score_lof'] = 0.0
    
    # Se não houver agrupamento, tratar todo o conjunto de dados
    if group_by is None:
        # Criar e treinar o modelo
        X = result[metric_column].values.reshape(-1, 1)
        
        # Pular se houver valores faltantes
        if np.isnan(X).any():
            X = result[metric_column].dropna().values.reshape(-1, 1)
            if len(X) == 0:
                return result
        
        # Ajustar n_neighbors se o conjunto de dados for pequeno
        n_neighbors = min(n_neighbors, len(X) - 1)
        if n_neighbors < 1:
            return result
        
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        result['is_anomaly_lof'] = (model.fit_predict(X) == -1)
        
        # Obter scores de anomalia (negativo da distância de alcance)
        result.loc[result.index[~np.isnan(result[metric_column])], 'anomaly_score_lof'] = -model.negative_outlier_factor_
    else:
        # Para cada grupo, treinar um modelo separado
        for group_name, group in result.groupby(group_by):
            # Pular se o grupo for muito pequeno
            if len(group) <= n_neighbors:
                continue
            
            # Preparar os dados
            X = group[metric_column].values.reshape(-1, 1)
            
            # Pular se houver valores faltantes
            if np.isnan(X).any():
                continue
            
            # Ajustar n_neighbors se o conjunto de dados for pequeno
            local_n_neighbors = min(n_neighbors, len(X) - 1)
            
            # Criar e treinar o modelo
            model = LocalOutlierFactor(n_neighbors=local_n_neighbors, contamination=contamination)
            predictions = model.fit_predict(X)
            
            # Atualizar o DataFrame de resultado
            index_in_group = group.index
            result.loc[index_in_group, 'is_anomaly_lof'] = (predictions == -1)
            result.loc[index_in_group, 'anomaly_score_lof'] = -model.negative_outlier_factor_
    
    return result


def detect_change_points(df, metric_column='value', time_column='elapsed_minutes', 
                        method='pelt', model='l2', min_size=5, penalty=3, group_by=None):
    """
    Detecta pontos de mudança na série temporal usando o algoritmo especificado.
    
    Args:
        df (DataFrame): DataFrame com dados da métrica
        metric_column (str): Coluna com os valores da métrica
        time_column (str): Coluna com os valores de tempo
        method (str): Método de detecção ('pelt', 'binseg', 'window')
        model (str): Modelo de custo ('l1', 'l2', 'rbf', etc.)
        min_size (int): Tamanho mínimo do segmento
        penalty (float): Penalidade para o algoritmo PELT
        group_by (list): Colunas para agrupar os dados (ex: ['tenant'])
        
    Returns:
        DataFrame: DataFrame original com coluna adicional 'is_change_point'
        dict: Informações sobre os pontos de mudança detectados
    """
    # Cria uma cópia para não modificar o original
    result = df.copy()
    
    # Adicionar coluna para pontos de mudança
    result['is_change_point'] = False
    
    changes_info = {}
    
    # Se não houver agrupamento, tratar todo o conjunto de dados
    if group_by is None:
        # Ordenar por tempo
        sorted_df = result.sort_values(time_column)
        
        # Preparar os dados
        signal = sorted_df[metric_column].values
        
        # Pular se o sinal for muito pequeno
        if len(signal) < min_size * 2:
            return result, changes_info
        
        # Configurar o algoritmo
        if method == 'pelt':
            algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
            change_points = algo.predict(pen=penalty)
        elif method == 'binseg':
            algo = rpt.Binseg(model=model, min_size=min_size).fit(signal)
            change_points = algo.predict(n_bkps=5)  # Detectar até 5 pontos
        elif method == 'window':
            algo = rpt.Window(model=model, width=40).fit(signal)
            change_points = algo.predict(n_bkps=5)  # Detectar até 5 pontos
        else:
            # Método padrão
            algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
            change_points = algo.predict(pen=penalty)
        
        # Filtrar change_points para garantir que estão dentro dos limites de iloc
        valid_change_points_indices = [cp for cp in change_points if cp < len(sorted_df)]

        # Marcar os pontos de mudança no DataFrame
        for cp_idx in valid_change_points_indices:
            # cp_idx já é um índice válido para iloc
            idx = sorted_df.iloc[cp_idx].name
            result.loc[idx, 'is_change_point'] = True
        
        # Registrar informações
        # Usar os índices válidos para buscar os tempos
        change_point_times_list = []
        if valid_change_points_indices: # Apenas se houver pontos de mudança válidos
            change_point_times_list = sorted_df.iloc[valid_change_points_indices][time_column].tolist()

        changes_info['all'] = {
            'n_change_points': len(valid_change_points_indices), # Contar apenas os válidos
            'change_point_indices': valid_change_points_indices, # Armazenar os válidos
            'change_point_times': change_point_times_list
        }
    else:
        # Para cada grupo, detectar pontos de mudança separadamente
        for group_name, group in result.groupby(group_by):
            # Ordenar por tempo
            sorted_group = group.sort_values(time_column)
            
            # Pular se o grupo for muito pequeno
            if len(sorted_group) < min_size * 2:
                continue
            
            # Preparar os dados
            signal = sorted_group[metric_column].values
            
            # Configurar o algoritmo
            if method == 'pelt':
                algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
                change_points = algo.predict(pen=penalty)
            elif method == 'binseg':
                algo = rpt.Binseg(model=model, min_size=min_size).fit(signal)
                change_points = algo.predict(n_bkps=5)  # Detectar até 5 pontos
            elif method == 'window':
                algo = rpt.Window(model=model, width=40).fit(signal)
                change_points = algo.predict(n_bkps=5)  # Detectar até 5 pontos
            else:
                # Método padrão
                algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
                change_points = algo.predict(pen=penalty)

            # Filtrar change_points para garantir que estão dentro dos limites de iloc
            valid_change_points_indices_group = [cp for cp in change_points if cp < len(sorted_group)]
            
            # Marcar os pontos de mudança no DataFrame
            for cp_idx in valid_change_points_indices_group:
                idx = sorted_group.iloc[cp_idx].name
                result.loc[idx, 'is_change_point'] = True
            
            # Registrar informações
            change_point_times_list_group = []
            if valid_change_points_indices_group:
                change_point_times_list_group = sorted_group.iloc[valid_change_points_indices_group][time_column].tolist()

            changes_info[group_name] = {
                'n_change_points': len(valid_change_points_indices_group),
                'change_point_indices': valid_change_points_indices_group,
                'change_point_times': change_point_times_list_group
            }
    
    return result, changes_info


def detect_pattern_changes(df, metrics, time_column='elapsed_minutes', 
                         window_size=10, n_clusters=3, group_by=None):
    """
    Detecta mudanças de padrão usando clustering de séries temporais.
    
    Args:
        df (DataFrame): DataFrame com dados
        metrics (list): Lista de nomes de métricas a incluir na análise
        time_column (str): Coluna com os valores de tempo
        window_size (int): Tamanho da janela deslizante para extração de padrões
        n_clusters (int): Número de clusters para agrupar padrões
        group_by (list): Colunas para agrupar os dados (ex: ['tenant'])
        
    Returns:
        DataFrame: DataFrame com informações sobre padrões detectados
    """
    # Verificar se temos métricas suficientes
    if len(metrics) == 0:
        return pd.DataFrame()
    
    results = []
    
    # Se não houver agrupamento, tratar todo o conjunto de dados
    if group_by is None:
        # Ordenar por tempo
        sorted_df = df.sort_values(time_column)
        
        # Extração de características
        X = sorted_df[metrics].values
        
        # Pular se o sinal for muito pequeno
        if len(X) < window_size * 2:
            return pd.DataFrame()
        
        # Normalizar os dados
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Extrair subsequências usando janela deslizante
        subsequences = []
        timestamps = []
        
        for i in range(len(X_scaled) - window_size + 1):
            subseq = X_scaled[i:i+window_size].flatten()
            subsequences.append(subseq)
            timestamps.append(sorted_df.iloc[i+window_size-1][time_column])
        
        # Agrupar subsequências
        if len(subsequences) > n_clusters:
            kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
            clusters = kmeans.fit_predict(np.array(subsequences))
            
            # Detectar mudanças de cluster
            cluster_changes = np.diff(clusters, prepend=clusters[0])
            change_indices = np.where(cluster_changes != 0)[0]
            
            # Registrar informações sobre mudanças de padrão
            for i in change_indices:
                if i > 0 and i < len(timestamps):
                    results.append({
                        'time': timestamps[i],
                        'from_cluster': clusters[i-1],
                        'to_cluster': clusters[i],
                        'group': 'all'
                    })
    else:
        # Para cada grupo, detectar mudanças de padrão separadamente
        for group_name, group in df.groupby(group_by):
            # Ordenar por tempo
            sorted_group = group.sort_values(time_column)
            
            # Pular se o grupo for muito pequeno
            if len(sorted_group) < window_size * 2:
                continue
            
            # Extração de características
            X = sorted_group[metrics].values
            
            # Normalizar os dados
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Extrair subsequências usando janela deslizante
            subsequences = []
            timestamps = []
            
            for i in range(len(X_scaled) - window_size + 1):
                subseq = X_scaled[i:i+window_size].flatten()
                subsequences.append(subseq)
                timestamps.append(sorted_group.iloc[i+window_size-1][time_column])
            
            # Agrupar subsequências
            if len(subsequences) > n_clusters:
                kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
                clusters = kmeans.fit_predict(np.array(subsequences))
                
                # Detectar mudanças de cluster
                cluster_changes = np.diff(clusters, prepend=clusters[0])
                change_indices = np.where(cluster_changes != 0)[0]
                
                # Registrar informações sobre mudanças de padrão
                for i in change_indices:
                    if i > 0 and i < len(timestamps):
                        results.append({
                            'time': timestamps[i],
                            'from_cluster': clusters[i-1],
                            'to_cluster': clusters[i],
                            'group': group_name
                        })
    
    return pd.DataFrame(results)


def detect_anomalies_ensemble(df, metric_column='value', time_column='elapsed_minutes', 
                            contamination=0.05, group_by=None):
    """
    Detecta anomalias usando um conjunto de algoritmos (ensemble).
    
    Args:
        df (DataFrame): DataFrame com dados da métrica
        metric_column (str): Coluna com os valores da métrica
        time_column (str): Coluna com os valores de tempo
        contamination (float): Proporção esperada de anomalias nos dados
        group_by (list): Colunas para agrupar os dados (ex: ['tenant', 'phase'])
        
    Returns:
        DataFrame: DataFrame com resultados consolidados da detecção de anomalias
    """
    # Carregar configurações
    from pipeline.config import DEFAULT_NOISY_TENANT
    
    # Verificar e tratar valores NaN
    df_clean = df.copy()
    
    # Determinar qual é o tenant gerador de ruído a partir de um atributo no DataFrame ou usar o padrão
    noisy_tenant = None
    if hasattr(df, 'noisy_tenant') and df.noisy_tenant:
        noisy_tenant = df.noisy_tenant
    else:
        noisy_tenant = DEFAULT_NOISY_TENANT
    
    # Tratar especialmente o caso do tenant gerador de ruído que pode não existir em certas fases
    if 'tenant' in df_clean.columns and group_by and 'tenant' in group_by:
        # Verificar se há NaNs na coluna de valores para o tenant gerador de ruído
        noisy_tenant_mask = df_clean['tenant'] == noisy_tenant
        if noisy_tenant_mask.any():
            # Substituir NaNs por zeros apenas para o tenant gerador de ruído 
            df_clean.loc[noisy_tenant_mask, metric_column] = df_clean.loc[noisy_tenant_mask, metric_column].fillna(0)
    
    # Preencher quaisquer outros NaNs remanescentes com a média da coluna
    if df_clean[metric_column].isna().any():
        df_clean[metric_column] = df_clean[metric_column].fillna(df_clean[metric_column].mean())
    
    # Aplicar diferentes algoritmos no DataFrame limpo
    result_if = detect_anomalies_isolation_forest(df_clean, metric_column, contamination, group_by)
    
    # Ajustar o número de vizinhos com base no tamanho dos dados
    if group_by is not None:
        # Calcular tamanho médio dos grupos
        group_sizes = df_clean.groupby(group_by).size()
        n_neighbors = max(5, int(group_sizes.mean() * 0.1))  # 10% do tamanho médio
    else:
        n_neighbors = max(5, int(len(df_clean) * 0.1))  # 10% do tamanho total
    
    result_lof = detect_anomalies_local_outlier_factor(df, metric_column, n_neighbors, contamination, group_by)
    
    # Combinar resultados
    result = df.copy()
    result['anomaly_score_if'] = result_if['anomaly_score_if']
    result['is_anomaly_if'] = result_if['is_anomaly_if']
    result['anomaly_score_lof'] = result_lof['anomaly_score_lof']
    result['is_anomaly_lof'] = result_lof['is_anomaly_lof']
    
    # Normalizar scores para permitir combinação
    if 'anomaly_score_if' in result.columns and 'anomaly_score_lof' in result.columns:
        # Função para normalizar entre 0 e 1
        def normalize(series):
            min_val = series.min()
            max_val = series.max()
            if max_val > min_val:
                return (series - min_val) / (max_val - min_val)
            else:
                return series * 0
        
        result['normalized_score_if'] = normalize(result['anomaly_score_if'])
        result['normalized_score_lof'] = normalize(result['anomaly_score_lof'])
        
        # Calcular score combinado (média dos scores normalizados)
        result['anomaly_score_combined'] = (result['normalized_score_if'] + result['normalized_score_lof']) / 2
        
        # Determinar anomalias combinadas (qualquer um dos algoritmos classificou como anomalia)
        result['is_anomaly'] = result['is_anomaly_if'] | result['is_anomaly_lof']
    
    # Adicionar detecção de pontos de mudança
    change_result, change_info = detect_change_points(df, metric_column, time_column, group_by=group_by)
    result['is_change_point'] = change_result['is_change_point']
    
    return result, change_info
