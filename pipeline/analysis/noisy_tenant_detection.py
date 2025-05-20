"""
Módulo para detecção automática do tenant ruidoso (noisy neighbor).

Este módulo fornece funções para identificar automaticamente qual tenant
é provavelmente o gerador de interferência nos outros tenants, sem necessidade
de especificar o tenant ruidoso antecipadamente.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler

def detect_noisy_tenant_from_correlation(correlation_matrix, tenant_metrics):
    """
    Detecta o tenant mais provável de ser ruidoso com base na matriz de correlação.
    Um tenant ruidoso geralmente tem correlações negativas com os outros tenants
    para métricas como CPU e memória.
    
    Args:
        correlation_matrix (pd.DataFrame): Matriz de correlação entre métricas de tenants
        tenant_metrics (List[str]): Lista de strings no formato "metrica_tenant"
        
    Returns:
        Dict[str, float]: Dicionário com scores de cada tenant, maior = mais provável de ser ruidoso
    """
    tenant_scores = {}
    
    # Retornar um dicionário vazio se a matriz de correlação for None ou estiver vazia
    if correlation_matrix is None or correlation_matrix.empty:
        return tenant_scores
        
    # Verificar se tenant_metrics é uma lista válida
    if tenant_metrics is None or len(tenant_metrics) == 0:
        return tenant_scores
    
    # Mapear os tenants diretamente a partir das colunas da matriz de correlação
    tenants = []
    
    for col in correlation_matrix.columns:
        if 'tenant-' in col:
            parts = col.split('tenant-')
            if len(parts) >= 2:
                tenant_name = f"tenant-{parts[1]}"
                if tenant_name not in tenants:
                    tenants.append(tenant_name)
    
    # Se não encontramos tenants no formato "tenant-X", usamos as próprias colunas
    if not tenants:
        tenants = list(correlation_matrix.columns)
    
    for tenant in tenants:
        tenant_score = 0
        count = 0
        
        # Define as métricas que queremos analisar
        metrics_to_analyze = ['cpu_usage', 'memory_usage', 'network_total_bandwidth', 'disk_throughput_total']

        # Para cada métrica que estamos analisando
        for metric in metrics_to_analyze:
            tenant_metric = f"{metric}_{tenant}"
            
            if tenant_metric not in correlation_matrix.columns:
                continue
                
            # Verificar correlações com outros tenants para esta métrica
            for other_tenant in tenants:
                if other_tenant == tenant:
                    continue
                    
                other_metric = f"{metric}_{other_tenant}"
                if other_metric not in correlation_matrix.columns:
                    continue
                
                # Correlação negativa em CPU/memória significa que quando um aumenta, o outro diminui
                # Isso é indicativo de um tenant ruidoso afetando outros
                correlation = correlation_matrix.loc[tenant_metric, other_metric]
                
                # Um valor negativo indica possível relação de noisy neighbor
                if correlation < 0:
                    # Quanto mais negativa a correlação, maior o score
                    tenant_score += abs(correlation)
                count += 1
        
        # Normalizar pelo número de comparações
        if count > 0:
            tenant_scores[tenant] = tenant_score / count
        else:
            tenant_scores[tenant] = 0
    
    return tenant_scores


def detect_noisy_tenant_from_causality(causality_results):
    """
    Detecta o tenant mais provável de ser ruidoso com base nos testes de causalidade de Granger.
    Um tenant ruidoso geralmente causa mudanças nos outros tenants mais do que é afetado por eles.
    
    Args:
        causality_results (List[Dict]): Resultados dos testes de causalidade entre tenants
        
    Returns:
        Dict[str, float]: Dicionário com scores de cada tenant, maior = mais provável de ser ruidoso
    """
    # Se não temos resultados de causalidade, retornar um dicionário vazio
    if causality_results is None or not causality_results:
        return {}
        
    # Verificar o tipo de dados, se for string, retornar um dicionário vazio
    if isinstance(causality_results, str):
        return {}
        
    # Contar quantas vezes cada tenant é identificado como causador
    # e quantas vezes é identificado como afetado
    tenant_causality_scores = {}
    
    for result in causality_results:
        # Verificar se result é um dicionário
        if not isinstance(result, dict):
            continue
            
        # Usar get() para dicionários e acessar diretamente se for um objeto com atributos
        try:
            if hasattr(result, 'get'):
                source = result.get('source_tenant')
                target = result.get('target_tenant')
                p_value = result.get('min_p_value')
            else:
                source = result.source_tenant if hasattr(result, 'source_tenant') else None
                target = result.target_tenant if hasattr(result, 'target_tenant') else None
                p_value = result.min_p_value if hasattr(result, 'min_p_value') else None
        except Exception:
            # Se houver qualquer problema, pular este resultado
            continue
        
        if not source or not target or p_value is None:
            continue
            
        # Inicializar scores se necessário
        if source not in tenant_causality_scores:
            tenant_causality_scores[source] = {
                'caused_others': 0,
                'affected_by_others': 0,
                'total_relations': 0
            }
        
        if target not in tenant_causality_scores:
            tenant_causality_scores[target] = {
                'caused_others': 0,
                'affected_by_others': 0,
                'total_relations': 0
            }
        
        # Verifica se há uma relação causal significativa (p < 0.05)
        if p_value < 0.05:
            # Incrementa o score do tenant causador
            tenant_causality_scores[source]['caused_others'] += 1
            # Incrementa o contador do tenant afetado
            tenant_causality_scores[target]['affected_by_others'] += 1
        
        # Incrementa o contador total de relações analisadas
        tenant_causality_scores[source]['total_relations'] += 1
        tenant_causality_scores[target]['total_relations'] += 1
    
    # Calcular score final para cada tenant
    tenant_scores = {}
    for tenant, scores in tenant_causality_scores.items():
        # Um tenant ruidoso deve causar mais do que ser afetado
        if scores['total_relations'] > 0:
            # Razão entre causar e ser afetado
            causality_ratio = scores['caused_others'] / max(1, scores['affected_by_others'])
            # Multiplicamos pela quantidade de relações causais identificadas
            tenant_scores[tenant] = causality_ratio * scores['caused_others']
        else:
            tenant_scores[tenant] = 0
    
    return tenant_scores


def detect_noisy_tenant_from_anomalies(anomaly_results, phase_df):
    """
    Detecta o tenant mais provável de ser ruidoso com base na quantidade e severidade
    das anomalias detectadas, especialmente durante a fase de ataque.
    
    Args:
        anomaly_results (pd.DataFrame): DataFrame com resultados da detecção de anomalias
        phase_df (pd.DataFrame): DataFrame com informações das fases do experimento
        
    Returns:
        Dict[str, float]: Dicionário com scores de cada tenant, maior = mais provável de ser ruidoso
    """
    tenant_scores = {}
    
    # Verificar se anomaly_results é um DataFrame
    if anomaly_results is None or not isinstance(anomaly_results, pd.DataFrame):
        return tenant_scores
        
    # Verificar se o DataFrame tem as colunas necessárias
    required_columns = ['tenant', 'is_anomaly_if', 'is_anomaly_lof']
    if not all(col in anomaly_results.columns for col in required_columns):
        return tenant_scores
        
    # Verificar se temos a coluna 'phase' ou se precisamos adicionar com base no phase_df
    if 'phase' not in anomaly_results.columns:
        # Se não temos a fase no DataFrame de anomalias, podemos tentar analisar por tenant somente
        # e verificar quais tenants têm mais anomalias em geral
        
        # Agrupar por tenant e contar anomalias
        tenant_anomalies = anomaly_results.groupby('tenant')[['is_anomaly_if', 'is_anomaly_lof']].sum().reset_index()
        
        # Calcular score baseado na quantidade de anomalias
        for _, row in tenant_anomalies.iterrows():
            tenant = row['tenant']
            # Somar os diferentes tipos de anomalias
            total_anomalies = row['is_anomaly_if'] + row['is_anomaly_lof']
            tenant_scores[tenant] = total_anomalies
            
        return tenant_scores
    
    # Se temos a fase, filtrar apenas a fase de ataque
    attack_phase = anomaly_results[anomaly_results['phase'].str.contains('Attack', case=False, na=False)]
    
    if len(attack_phase) == 0:
        return tenant_scores
    
    # Agrupar por tenant e contar anomalias
    tenant_anomalies = attack_phase.groupby('tenant')[['is_anomaly_if', 'is_anomaly_lof']].sum().reset_index()
    
    # Calcular score baseado na quantidade de anomalias
    for _, row in tenant_anomalies.iterrows():
        tenant = row['tenant']
        # Somar os diferentes tipos de anomalias
        total_anomalies = row['is_anomaly_if'] + row['is_anomaly_lof']
        tenant_scores[tenant] = total_anomalies
    
    # Normalizar scores
    if tenant_scores:
        max_score = max(tenant_scores.values())
        if max_score > 0:
            tenant_scores = {k: v / max_score for k, v in tenant_scores.items()}
    
    return tenant_scores


def detect_noisy_tenant_from_impact(impact_scores):
    """
    Detecta o tenant mais provável de ser ruidoso com base no impacto causado em outros tenants.
    
    Args:
        impact_scores (Dict): Dicionário com scores de impacto entre tenants
        
    Returns:
        Dict[str, float]: Dicionário com scores de cada tenant, maior = mais provável de ser ruidoso
    """
    tenant_scores = {}
    
    for tenant, impacts in impact_scores.items():
        # Calcular o impacto médio causado nos outros tenants
        impact_values = [v for k, v in impacts.items() if k != tenant]
        if impact_values:
            tenant_scores[tenant] = sum(impact_values) / len(impact_values)
        else:
            tenant_scores[tenant] = 0
    
    return tenant_scores


def combine_detection_results(correlation_scores, causality_scores, anomaly_scores, impact_scores, weights=None):
    """
    Combina os resultados de diferentes métodos de detecção para obter um score final.
    
    Args:
        correlation_scores (Dict[str, float]): Scores baseados em correlação
        causality_scores (Dict[str, float]): Scores baseados em causalidade
        anomaly_scores (Dict[str, float]): Scores baseados em anomalias
        impact_scores (Dict[str, float]): Scores baseados em impacto
        weights (Dict[str, float]): Pesos para cada método (opcional)
        
    Returns:
        Dict[str, float]: Scores finais combinados
        Dict[str, Dict[str, float]]: Scores detalhados por método
    """
    # Definir pesos padrão se não fornecidos
    if weights is None:
        weights = {
            'correlation': 0.2,
            'causality': 0.3,
            'anomaly': 0.2,
            'impact': 0.3
        }
    
    # Mapear todos os nomes de tenants para o formato padrão (ex: "tenant-a", "tenant-b")
    # isso ajuda a consolidar tenants que podem aparecer com diferentes prefixos em diferentes métodos
    def normalize_tenant_name(tenant_name):
        # Se o nome já estiver no formato "tenant-X", retorná-lo
        if tenant_name.startswith("tenant-"):
            return tenant_name
            
        # Se o nome tiver um prefixo e depois "tenant-X", extrair apenas "tenant-X"
        if "_tenant-" in tenant_name:
            parts = tenant_name.split("_tenant-")
            return f"tenant-{parts[1]}"
            
        # Se não conseguirmos normalizar, retornar o nome original
        return tenant_name
    
    # Normalizar os nomes dos tenants nas pontuações
    normalized_correlation_scores = {normalize_tenant_name(k): v for k, v in correlation_scores.items()}
    normalized_causality_scores = {normalize_tenant_name(k): v for k, v in causality_scores.items()}
    normalized_anomaly_scores = {normalize_tenant_name(k): v for k, v in anomaly_scores.items()}
    normalized_impact_scores = {normalize_tenant_name(k): v for k, v in impact_scores.items()}
    
    # Coletar todos os tenants únicos após normalização
    all_tenants = set()
    all_tenants.update(normalized_correlation_scores.keys())
    all_tenants.update(normalized_causality_scores.keys())
    all_tenants.update(normalized_anomaly_scores.keys())
    all_tenants.update(normalized_impact_scores.keys())
    
    # Normalizar scores dentro de cada método
    methods = {
        'correlation': normalized_correlation_scores,
        'causality': normalized_causality_scores,
        'anomaly': normalized_anomaly_scores,
        'impact': normalized_impact_scores
    }
    
    normalized_scores = {}
    for method, scores in methods.items():
        if not scores:
            continue
        
        # Encontrar o maior score para normalizar
        max_score = max(scores.values()) if scores else 1
        
        # Normalizar scores para [0, 1]
        if max_score > 0:
            normalized_scores[method] = {k: v / max_score for k, v in scores.items()}
        else:
            normalized_scores[method] = scores
    
    # Combinar scores ponderados
    final_scores = {}
    detailed_scores = {}
    
    for tenant in all_tenants:
        weighted_sum = 0
        detailed = {}
        
        for method, weight in weights.items():
            if method in normalized_scores and tenant in normalized_scores[method]:
                score = normalized_scores[method][tenant]
                weighted_sum += score * weight
                detailed[method] = score
            else:
                detailed[method] = 0
        
        final_scores[tenant] = weighted_sum
        detailed_scores[tenant] = detailed
    
    return final_scores, detailed_scores


def identify_noisy_tenant(
    metrics_dict,
    causality_results=None,
    anomaly_results=None,
    impact_scores=None,
    round_name='round-1',
    weights=None,
    real_tenants=None
):
    """
    Identifica automaticamente qual tenant é provavelmente o "noisy neighbor"
    baseado em múltiplos critérios de análise.
    
    Args:
        metrics_dict (Dict): Dicionário com DataFrames para cada métrica
        causality_results (List[Dict]): Resultados de análise de causalidade (opcional)
        anomaly_results (pd.DataFrame): Resultados de detecção de anomalias (opcional)
        impact_scores (Dict): Scores de impacto entre tenants (opcional)
        round_name (str): Round a ser analisado
        weights (Dict[str, float]): Pesos para cada método de detecção
        real_tenants (List[str]): Lista dos nomes reais dos tenants no ambiente
        
    Returns:
        str: Nome do tenant identificado como provável noisy neighbor
        Dict: Scores finais para cada tenant
        Dict: Scores detalhados por método de detecção
    """
    from pipeline.analysis.tenant_analysis import calculate_correlation_matrix
    
    try:
        # Calcular matriz de correlação
        correlation_matrix = calculate_correlation_matrix(metrics_dict, round_name=round_name)
        
        # Extrair nomes das métricas/tenant das colunas da matriz
        tenant_metrics = list(correlation_matrix.columns)
        
        # Detectar tenant ruidoso usando diferentes métodos
        correlation_scores = detect_noisy_tenant_from_correlation(correlation_matrix, tenant_metrics)
    except Exception as e:
        print(f"  Erro ao calcular correlação: {e}")
        correlation_scores = {}
    
    # Inicializar scores para métodos opcionais
    causality_scores = {}
    anomaly_scores = {}
    impact_scores_processed = {}
    
    # Adicionar resultados de causalidade se disponíveis
    try:
        if causality_results:
            causality_scores = detect_noisy_tenant_from_causality(causality_results)
    except Exception as e:
        print(f"  Erro ao processar dados de causalidade: {e}")
        causality_scores = {}
    
    # Adicionar resultados de anomalias se disponíveis
    try:
        if anomaly_results is not None:
            # Precisamos também do DataFrame com as fases
            # Vamos extrair as informações de fase do primeiro DataFrame de métricas
            first_metric_df = next(iter(metrics_dict.values()))
            phase_df = first_metric_df[['datetime', 'phase']].drop_duplicates() if 'phase' in first_metric_df.columns else None
            
            anomaly_scores = detect_noisy_tenant_from_anomalies(anomaly_results, phase_df)
    except Exception as e:
        print(f"  Erro ao processar dados de anomalias: {e}")
        anomaly_scores = {}
    
    # Adicionar resultados de impacto se disponíveis
    try:
        if impact_scores:
            impact_scores_processed = detect_noisy_tenant_from_impact(impact_scores)
    except Exception as e:
        print(f"  Erro ao processar scores de impacto: {e}")
        impact_scores_processed = {}
    
    # Combinar todos os resultados
    final_scores, detailed_scores = combine_detection_results(
        correlation_scores, 
        causality_scores, 
        anomaly_scores, 
        impact_scores_processed,
        weights
    )
    
    # Identificar o tenant com o maior score final
    if final_scores:
        # Se temos a lista de real_tenants, filtramos os resultados para mostrar apenas tenants reais
        if real_tenants:
            filtered_scores = {k: v for k, v in final_scores.items() if k in real_tenants or any(k.endswith(t) for t in real_tenants)}
            if filtered_scores:
                # Se ainda temos scores após a filtragem, usamos apenas os tenants reais
                sorted_tenants = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
                most_likely_noisy_tenant = sorted_tenants[0][0]
                # Se o nome contém um prefixo, vamos tentar extrair apenas o tenant real
                if not most_likely_noisy_tenant.startswith('tenant-') and 'tenant-' in most_likely_noisy_tenant:
                    # Extrair o tenant-X do nome
                    for real_tenant in real_tenants:
                        if real_tenant in most_likely_noisy_tenant:
                            most_likely_noisy_tenant = real_tenant
                            break
            else:
                # Se não temos scores para tenants reais, usamos todos os scores
                sorted_tenants = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
                most_likely_noisy_tenant = sorted_tenants[0][0]
        else:
            # Ordenar do maior para o menor score
            sorted_tenants = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            most_likely_noisy_tenant = sorted_tenants[0][0]
            # Se o tenant identificado tem um formato como "metrica_tenant", tentar extrair apenas o tenant
            if '_tenant-' in most_likely_noisy_tenant:
                most_likely_noisy_tenant = 'tenant-' + most_likely_noisy_tenant.split('_tenant-')[1]
    else:
        most_likely_noisy_tenant = None
    
    return most_likely_noisy_tenant, final_scores, detailed_scores
