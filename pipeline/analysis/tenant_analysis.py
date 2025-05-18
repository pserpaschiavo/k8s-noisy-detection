"""
Módulo para análise comparativa entre tenants para o experimento de noisy neighbors.

Este módulo fornece funções para comparar métricas entre diferentes tenants
e analisar o impacto do tenant barulhento nos demais.
"""

import pandas as pd
import numpy as np
from scipy import stats


def calculate_correlation_matrix(metrics_dict, tenants=None, round_name='round-1', noisy_tenant=None):
    """
    Calcula uma matriz de correlação entre métricas de diferentes tenants.
    
    Args:
        metrics_dict (dict): Dicionário com DataFrames para cada métrica
        tenants (list): Lista de tenants a incluir (None = todos)
        round_name (str): Round a ser analisado
        noisy_tenant (str): Tenant específico que gera ruído (por padrão: DEFAULT_NOISY_TENANT da configuração)
        
    Returns:
        DataFrame: Matriz de correlação entre métricas dos tenants
    """
    from pipeline.config import DEFAULT_NOISY_TENANT
    
    # Determinar qual é o tenant gerador de ruído
    noisy_tenant = noisy_tenant if noisy_tenant else DEFAULT_NOISY_TENANT
    
    # Preparar dados para correlação
    correlation_data = {}
    
    for metric_name, metric_df in metrics_dict.items():
        # Filtrar pelo round especificado
        round_df = metric_df[metric_df['round'] == round_name]
        
        if tenants:
            round_df = round_df[round_df['tenant'].isin(tenants)]
        
        # Verificar se o tenant gerador de ruído está presente
        has_noisy_tenant = noisy_tenant in round_df['tenant'].unique()
        
        # Pivotar para ter uma coluna para cada tenant
        pivot = round_df.pivot_table(
            index='datetime',
            columns='tenant',
            values='value'
        )
        
        # Preencher valores NaN com 0
        pivot.fillna(0, inplace=True)
        
        # Garantir que tenant gerador de ruído esteja presente para consistência nas análises
        if not has_noisy_tenant and noisy_tenant not in pivot.columns:
            pivot[noisy_tenant] = 0  # Adicionar tenant gerador de ruído com valores zero
        
        # Adicionar ao dicionário com prefixo da métrica
        for tenant in pivot.columns:
            correlation_data[f"{metric_name}_{tenant}"] = pivot[tenant]
    
    # Criar DataFrame com todas as séries
    corr_df = pd.DataFrame(correlation_data)
    
    # Calcular correlação
    correlation_matrix = corr_df.corr()
    
    return correlation_matrix


def compare_tenant_metrics(df, baseline_tenant='tenant-a', metric_column='value'):
    """
    Compara métricas de diferentes tenants com um tenant de referência.
    
    Args:
        df (DataFrame): DataFrame com dados de métricas
        baseline_tenant (str): Tenant de referência para comparação
        metric_column (str): Coluna com os valores da métrica
        
    Returns:
        DataFrame: DataFrame com comparações entre tenants
    """
    # Filtrar apenas o tenant de referência
    baseline_df = df[df['tenant'] == baseline_tenant].copy()
    
    # Preparar um DataFrame para armazenar os resultados
    results = []
    
    # Para cada combinação de round e phase
    for (round_name, phase_name), group in df.groupby(['round', 'phase']):
        # Dados do tenant de referência para esta combinação
        base = baseline_df[(baseline_df['round'] == round_name) & 
                          (baseline_df['phase'] == phase_name)]
        
        if len(base) == 0:
            continue  # Skip if no baseline data
            
        base_mean = base[metric_column].mean()
        base_std = base[metric_column].std()
        
        # Para cada tenant
        for tenant, tenant_group in group.groupby('tenant'):
            if tenant == baseline_tenant:
                continue  # Skip baseline tenant
                
            tenant_mean = tenant_group[metric_column].mean()
            tenant_std = tenant_group[metric_column].std()
            
            # Calcular diferença percentual
            percent_diff = ((tenant_mean - base_mean) / base_mean) * 100 if base_mean != 0 else np.nan
            
            # Teste estatístico (t-test para duas amostras independentes)
            t_stat, p_value = stats.ttest_ind(
                base[metric_column].dropna(),
                tenant_group[metric_column].dropna(),
                equal_var=False  # Assume variâncias diferentes (Welch's t-test)
            )
            
            # Adicionar aos resultados
            results.append({
                'round': round_name,
                'phase': phase_name,
                'baseline_tenant': baseline_tenant,
                'compared_tenant': tenant,
                'baseline_mean': base_mean,
                'baseline_std': base_std,
                'tenant_mean': tenant_mean,
                'tenant_std': tenant_std,
                'percent_difference': percent_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_difference': p_value < 0.05  # Significância estatística
            })
    
    return pd.DataFrame(results)
