"""
Módulo para análise entre diferentes fases do experimento de noisy neighbors.

Este módulo fornece funções para analisar e comparar métricas entre as
diferentes fases (Baseline, Attack, Recovery) do experimento.
"""

import pandas as pd
import numpy as np
from scipy import stats


def compare_phases_ttest(df, metric_column='value', alpha=0.05):
    """
    Compara métricas entre diferentes fases usando teste t de Student.
    
    Args:
        df (DataFrame): DataFrame com dados de métricas
        metric_column (str): Coluna com os valores da métrica
        alpha (float): Nível de significância para o teste estatístico
        
    Returns:
        DataFrame: DataFrame com resultados dos testes estatísticos
    """
    # Lista para armazenar os resultados
    results = []
    
    # Para cada tenant e round
    for (tenant, round_name), group in df.groupby(['tenant', 'round']):
        phases = sorted(group['phase'].unique())
        
        # Comparar cada par de fases
        for i, phase1 in enumerate(phases):
            for phase2 in phases[i+1:]:
                # Dados das duas fases
                data1 = group[group['phase'] == phase1][metric_column].dropna()
                data2 = group[group['phase'] == phase2][metric_column].dropna()
                
                if len(data1) < 2 or len(data2) < 2:
                    continue  # Pular se não houver dados suficientes
                
                # Calcular estatísticas
                mean1, mean2 = data1.mean(), data2.mean()
                std1, std2 = data1.std(), data2.std()
                
                # Teste t para duas amostras independentes
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                
                # Tamanho do efeito (Cohen's d)
                pooled_std = np.sqrt(((len(data1) - 1) * std1**2 + (len(data2) - 1) * std2**2) / 
                                     (len(data1) + len(data2) - 2))
                cohen_d = abs(mean1 - mean2) / pooled_std if pooled_std != 0 else np.nan
                
                # Adicionar resultados
                results.append({
                    'tenant': tenant,
                    'round': round_name,
                    'phase1': phase1,
                    'phase2': phase2,
                    'mean1': mean1,
                    'mean2': mean2,
                    'std1': std1,
                    'std2': std2,
                    'diff': mean2 - mean1,
                    'percent_diff': ((mean2 - mean1) / mean1) * 100 if mean1 != 0 else np.nan,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'effect_size': cohen_d,
                    'effect_magnitude': 'Large' if cohen_d > 0.8 else 
                                        'Medium' if cohen_d > 0.5 else 
                                        'Small' if cohen_d > 0.2 else 'Negligible'
                })
    
    return pd.DataFrame(results)


def analyze_recovery_effectiveness(df, metric_column='value', 
                                  baseline_phase='1 - Baseline', 
                                  attack_phase='2 - Attack', 
                                  recovery_phase='3 - Recovery'):
    """
    Analisa a efetividade da recuperação após a fase de ataque.
    
    Args:
        df (DataFrame): DataFrame com dados de métricas
        metric_column (str): Coluna com os valores da métrica
        baseline_phase (str): Nome da fase de baseline
        attack_phase (str): Nome da fase de ataque
        recovery_phase (str): Nome da fase de recuperação
        
    Returns:
        DataFrame: DataFrame com análise da recuperação
    """
    # Lista para armazenar resultados
    results = []
    
    # Para cada tenant e round
    for (tenant, round_name), group in df.groupby(['tenant', 'round']):
        # Obter dados para cada fase
        baseline_data = group[group['phase'] == baseline_phase][metric_column]
        attack_data = group[group['phase'] == attack_phase][metric_column]
        recovery_data = group[group['phase'] == recovery_phase][metric_column]
        
        if len(baseline_data) == 0 or len(attack_data) == 0 or len(recovery_data) == 0:
            continue  # Pular se alguma fase não tiver dados
        
        # Calcular estatísticas
        baseline_mean = baseline_data.mean()
        attack_mean = attack_data.mean()
        recovery_mean = recovery_data.mean()
        
        # Calcular índices de degradação e recuperação
        degradation = attack_mean - baseline_mean
        recovery_delta = recovery_mean - attack_mean
        
        # Calcular percentual de recuperação
        if degradation != 0:
            recovery_percent = (recovery_delta / abs(degradation)) * 100
        else:
            recovery_percent = np.nan
        
        # Distância do valor de recuperação em relação ao baseline
        baseline_diff = recovery_mean - baseline_mean
        baseline_diff_percent = (baseline_diff / baseline_mean) * 100 if baseline_mean != 0 else np.nan
        
        # Adicionar resultados
        results.append({
            'tenant': tenant,
            'round': round_name,
            'baseline_mean': baseline_mean,
            'attack_mean': attack_mean,
            'recovery_mean': recovery_mean,
            'degradation': degradation,
            'degradation_percent': (degradation / baseline_mean) * 100 if baseline_mean != 0 else np.nan,
            'recovery_delta': recovery_delta,
            'recovery_percent': recovery_percent,
            'baseline_diff': baseline_diff,
            'baseline_diff_percent': baseline_diff_percent,
            'full_recovery': abs(baseline_diff_percent) < 5 if not np.isnan(baseline_diff_percent) else False
        })
    
    return pd.DataFrame(results)
