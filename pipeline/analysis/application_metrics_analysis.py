"""
Módulo para análise de métricas de nível de aplicação em cenários de noisy neighbors.

Este módulo expande a análise para além das métricas de infraestrutura,
incorporando métricas de aplicação como latência, taxa de erros e throughput.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats

def analyze_latency_impact(app_metrics: Dict[str, pd.DataFrame], 
                         infrastructure_metrics: Dict[str, pd.DataFrame],
                         noisy_tenant: str) -> Dict[str, Dict[str, float]]:
    """
    Analisa o impacto do tenant ruidoso nas latências de aplicação de outros tenants.
    
    Args:
        app_metrics: Dicionário com métricas de aplicação (latência, erros, etc.)
        infrastructure_metrics: Dicionário com métricas de infraestrutura
        noisy_tenant: O tenant identificado como ruidoso
        
    Returns:
        Dicionário com análise de impacto na latência para cada tenant
    """
    impact_analysis = {}
    
    if 'latency' not in app_metrics:
        return impact_analysis
        
    latency_df = app_metrics['latency']
    
    # Separar por fases
    baseline_df = latency_df[latency_df['phase'].str.contains('Baseline', case=False, na=False)]
    attack_df = latency_df[latency_df['phase'].str.contains('Attack', case=False, na=False)]
    
    # Calcular impacto por tenant
    for tenant in latency_df['tenant'].unique():
        if tenant == noisy_tenant:
            continue
            
        # Latência na baseline
        tenant_baseline = baseline_df[baseline_df['tenant'] == tenant]['value'].mean()
        
        # Latência na fase de ataque
        tenant_attack = attack_df[attack_df['tenant'] == tenant]['value'].mean()
        
        # Calcular variação percentual
        if tenant_baseline > 0:
            latency_increase_pct = ((tenant_attack - tenant_baseline) / tenant_baseline) * 100
        else:
            latency_increase_pct = 0
            
        # Calcular significância estatística
        t_stat, p_value = stats.ttest_ind(
            baseline_df[baseline_df['tenant'] == tenant]['value'],
            attack_df[attack_df['tenant'] == tenant]['value'],
            equal_var=False
        )
        
        impact_analysis[tenant] = {
            'baseline_latency': tenant_baseline,
            'attack_latency': tenant_attack,
            'increase_percentage': latency_increase_pct,
            'p_value': p_value,
            'significant_impact': p_value < 0.05
        }
    
    return impact_analysis

def analyze_error_rate_correlation(app_metrics: Dict[str, pd.DataFrame],
                                 infrastructure_metrics: Dict[str, pd.DataFrame],
                                 noisy_tenant: str) -> Dict[str, float]:
    """
    Analisa a correlação entre utilização de recursos do tenant ruidoso
    e taxas de erro nos outros tenants.
    
    Args:
        app_metrics: Dicionário com métricas de aplicação
        infrastructure_metrics: Dicionário com métricas de infraestrutura
        noisy_tenant: O tenant identificado como ruidoso
        
    Returns:
        Dicionário com correlações entre uso de recursos e taxas de erro
    """
    correlations = {}
    
    if 'error_rate' not in app_metrics or 'cpu_usage' not in infrastructure_metrics:
        return correlations
        
    error_df = app_metrics['error_rate']
    cpu_df = infrastructure_metrics['cpu_usage']
    
    # Filtrar dados para o tenant ruidoso
    noisy_cpu = cpu_df[cpu_df['tenant'] == noisy_tenant]
    
    # Para cada tenant, calcular correlação entre CPU do noisy e taxa de erros
    for tenant in error_df['tenant'].unique():
        if tenant == noisy_tenant:
            continue
            
        tenant_errors = error_df[error_df['tenant'] == tenant]
        
        # Alinhar os índices de tempo para correlação
        if 'datetime' in noisy_cpu.columns and 'datetime' in tenant_errors.columns:
            # Criar séries temporais indexadas
            cpu_series = pd.Series(noisy_cpu['value'].values, index=noisy_cpu['datetime'])
            error_series = pd.Series(tenant_errors['value'].values, index=tenant_errors['datetime'])
            
            # Reamostrar para o mesmo intervalo de tempo
            common_index = pd.date_range(
                start=min(cpu_series.index.min(), error_series.index.min()),
                end=max(cpu_series.index.max(), error_series.index.max()),
                freq='1min'
            )
            
            cpu_resampled = cpu_series.reindex(common_index, method='nearest')
            error_resampled = error_series.reindex(common_index, method='nearest')
            
            # Calcular correlação
            correlation = cpu_resampled.corr(error_resampled)
            correlations[tenant] = correlation
    
    return correlations

def calculate_application_slo_violations(app_metrics: Dict[str, pd.DataFrame],
                                       slo_thresholds: Dict[str, float],
                                       noisy_tenant: str) -> Dict[str, Dict[str, float]]:
    """
    Calcula violações de SLO (Service Level Objectives) causadas pelo tenant ruidoso.
    
    Args:
        app_metrics: Dicionário com métricas de aplicação
        slo_thresholds: Dicionário com limites de SLO por métrica
        noisy_tenant: O tenant identificado como ruidoso
        
    Returns:
        Dicionário com análise de violações de SLO por tenant
    """
    slo_analysis = {}
    
    for metric_name, threshold in slo_thresholds.items():
        if metric_name not in app_metrics:
            continue
            
        df = app_metrics[metric_name]
        
        # Analisar por fase
        baseline_df = df[df['phase'].str.contains('Baseline', case=False, na=False)]
        attack_df = df[df['phase'].str.contains('Attack', case=False, na=False)]
        
        for tenant in df['tenant'].unique():
            if tenant == noisy_tenant:
                continue
                
            # Calcular violações na baseline
            tenant_baseline = baseline_df[baseline_df['tenant'] == tenant]
            baseline_violations = len(tenant_baseline[tenant_baseline['value'] > threshold])
            baseline_violation_rate = baseline_violations / len(tenant_baseline) if len(tenant_baseline) > 0 else 0
            
            # Calcular violações na fase de ataque
            tenant_attack = attack_df[attack_df['tenant'] == tenant]
            attack_violations = len(tenant_attack[tenant_attack['value'] > threshold])
            attack_violation_rate = attack_violations / len(tenant_attack) if len(tenant_attack) > 0 else 0
            
            # Guardar resultados
            if tenant not in slo_analysis:
                slo_analysis[tenant] = {}
                
            slo_analysis[tenant][metric_name] = {
                'baseline_violation_rate': baseline_violation_rate,
                'attack_violation_rate': attack_violation_rate,
                'violation_increase': attack_violation_rate - baseline_violation_rate,
                'violation_increase_factor': attack_violation_rate / baseline_violation_rate if baseline_violation_rate > 0 else float('inf')
            }
    
    return slo_analysis
