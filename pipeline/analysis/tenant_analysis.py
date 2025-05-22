"""
Módulo para análise comparativa entre tenants para o experimento de noisy neighbors.

Este módulo fornece funções para comparar métricas entre diferentes tenants
e analisar o impacto do tenant barulhento nos demais.
"""

import pandas as pd
import numpy as np
from scipy import stats


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
