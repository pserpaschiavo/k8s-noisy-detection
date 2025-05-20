"""
Módulo para análises avançadas, incluindo covariância, causalidade e entropia.

Este módulo implementa métodos avançados para análise de dados do experimento de 
noisy neighbors, focando em relações complexas entre tenants e fases.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from pipeline.config import METRICS_CONFIG, IMPACT_SCORE_WEIGHTS, PHASE_DISPLAY_NAMES


def calculate_covariance_matrix(metrics_dict, tenants=None, phase=None, round_name='round-1'):
    """
    Calcula uma matriz de covariância entre métricas de diferentes tenants.
    
    Args:
        metrics_dict (dict): Dicionário com DataFrames para cada métrica
        tenants (list): Lista de tenants a incluir (None = todos)
        phase (str): Fase específica para análise (None = todas)
        round_name (str): Round a ser analisado
        
    Returns:
        DataFrame: Matriz de covariância entre métricas dos tenants
        DataFrame: Matriz de correlação (para comparação)
    """
    # Preparar dados para covariância
    covariance_data = {}
    
    for metric_name, metric_df in metrics_dict.items():
        # Filtrar pelo round especificado
        round_df = metric_df[metric_df['round'] == round_name]
        
        # Filtrar pela fase se especificada
        if phase:
            round_df = round_df[round_df['phase'] == phase]
            
        if tenants:
            round_df = round_df[round_df['tenant'].isin(tenants)]
        
        # Pivotar para ter uma coluna para cada tenant
        pivot = round_df.pivot_table(
            index='datetime',
            columns='tenant',
            values='value'
        )
        
        # Adicionar ao dicionário com prefixo da métrica
        for tenant in pivot.columns:
            covariance_data[f"{metric_name}_{tenant}"] = pivot[tenant]
    
    # Criar DataFrame com todas as séries
    cov_df = pd.DataFrame(covariance_data)
    
    # Calcular covariância
    covariance_matrix = cov_df.cov()
    
    # Calcular correlação para comparação
    correlation_matrix = cov_df.corr()
    
    return covariance_matrix, correlation_matrix


def calculate_cross_tenant_entropy(df, tenant1, tenant2, metric_column='value'):
    """
    Calcula a entropia cruzada entre dois tenants para uma métrica.
    Usamos informação mútua como uma medida relacionada à entropia cruzada.
    
    Args:
        df (DataFrame): DataFrame com dados da métrica
        tenant1 (str): Primeiro tenant para análise
        tenant2 (str): Segundo tenant para análise
        metric_column (str): Coluna com os valores da métrica
        
    Returns:
        float: Valor de informação mútua (relacionado à entropia cruzada)
        dict: Informações adicionais sobre a relação
    """
    # Filtrar dados para os dois tenants
    data1 = df[df['tenant'] == tenant1].sort_values('datetime')
    data2 = df[df['tenant'] == tenant2].sort_values('datetime')
    
    # Verificar se temos dados suficientes
    if len(data1) < 10 or len(data2) < 10:
        return None, {"error": "Dados insuficientes para análise de entropia"}
    
    # Alinhar as séries temporais (pode requerer interpolação)
    common_times = pd.Series(list(set(data1['datetime']).intersection(set(data2['datetime']))))
    if len(common_times) < 10:
        # Se não houver timestamps comuns suficientes, interpolar
        min_time = min(data1['datetime'].min(), data2['datetime'].min())
        max_time = max(data1['datetime'].max(), data2['datetime'].max())
        
        # Criar índice de tempo regular
        time_index = pd.date_range(start=min_time, end=max_time, freq='1min')
        
        # Reindexar e interpolar
        series1 = data1.set_index('datetime')[metric_column].reindex(time_index).interpolate()
        series2 = data2.set_index('datetime')[metric_column].reindex(time_index).interpolate()
    else:
        # Se houver timestamps comuns suficientes, usar apenas esses
        series1 = data1[data1['datetime'].isin(common_times)].set_index('datetime')[metric_column]
        series2 = data2[data2['datetime'].isin(common_times)].set_index('datetime')[metric_column]
    
    # Normalizar os dados
    scaler = StandardScaler()
    X1 = scaler.fit_transform(series1.values.reshape(-1, 1)).flatten()
    X2 = scaler.fit_transform(series2.values.reshape(-1, 1)).flatten()
    
    # Calcular informação mútua (relacionada à entropia cruzada)
    mi = mutual_info_regression(X1.reshape(-1, 1), X2)[0]
    
    # Calcular estatísticas adicionais
    corr = np.corrcoef(X1, X2)[0, 1]
    
    info = {
        "tenant1": tenant1,
        "tenant2": tenant2,
        "mutual_information": mi,
        "correlation": corr,
        "n_samples": len(X1)
    }
    
    return mi, info


def calculate_entropy_metrics(df, tenants=None, phase=None, metric_column='value'):
    """
    Calcula métricas de entropia para diferentes tenants e fases.
    
    Args:
        df (DataFrame): DataFrame com dados da métrica
        tenants (list): Lista de tenants a incluir (None = todos)
        phase (str): Fase específica para análise (None = todas)
        metric_column (str): Coluna com os valores da métrica
        
    Returns:
        DataFrame: Resultados das métricas de entropia
    """
    if tenants is None:
        tenants = sorted(df['tenant'].unique())
    
    # Filtrar pela fase se especificada
    if phase:
        df = df[df['phase'] == phase]
    
    results = []
    
    # Calcular entropia para cada par de tenants
    for i, tenant1 in enumerate(tenants):
        for tenant2 in tenants[i+1:]:  # Evitar duplicações
            mi, info = calculate_cross_tenant_entropy(df, tenant1, tenant2, metric_column)
            
            if mi is not None:
                # Se a análise for bem-sucedida, adicionar aos resultados
                info["phase"] = phase if phase else "all"
                results.append(info)
    
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame()


def granger_causality_test(df, cause_tenant, effect_tenant, metric_column='value', max_lag=5, alpha=0.05):
    """
    Realiza teste de causalidade de Granger entre dois tenants para uma métrica.
    
    Args:
        df (DataFrame): DataFrame com dados da métrica
        cause_tenant (str): Tenant que potencialmente causa o efeito
        effect_tenant (str): Tenant que potencialmente recebe o efeito
        metric_column (str): Coluna com os valores da métrica
        max_lag (int): Número máximo de lags a considerar
        alpha (float): Nível de significância para o teste
        
    Returns:
        dict: Resultados do teste de causalidade de Granger
    """
    # Filtrar dados para os dois tenants
    data_cause = df[df['tenant'] == cause_tenant].sort_values('datetime')
    data_effect = df[df['tenant'] == effect_tenant].sort_values('datetime')
    
    # Verificar se temos dados suficientes
    if len(data_cause) < max_lag + 1 or len(data_effect) < max_lag + 1:
        return {"error": "Dados insuficientes para análise de causalidade"}
    
    # Alinhar as séries temporais (pode requerer interpolação)
    common_times = pd.Series(list(set(data_cause['datetime']).intersection(set(data_effect['datetime']))))
    
    if len(common_times) < max_lag * 2:
        # Se não houver timestamps comuns suficientes, interpolar
        min_time = min(data_cause['datetime'].min(), data_effect['datetime'].min())
        max_time = max(data_cause['datetime'].max(), data_effect['datetime'].max())
        
        # Criar índice de tempo regular
        time_index = pd.date_range(start=min_time, end=max_time, freq='1min')
        
        # Reindexar e interpolar
        series_cause = data_cause.set_index('datetime')[metric_column].reindex(time_index).interpolate()
        series_effect = data_effect.set_index('datetime')[metric_column].reindex(time_index).interpolate()
    else:
        # Se houver timestamps comuns suficientes, usar apenas esses
        common_times = sorted(common_times)
        series_cause = data_cause[data_cause['datetime'].isin(common_times)].set_index('datetime')[metric_column]
        series_effect = data_effect[data_effect['datetime'].isin(common_times)].set_index('datetime')[metric_column]
    
    # Preparar dados para o teste
    data = pd.DataFrame({
        'cause': series_cause,
        'effect': series_effect
    })
    
    # Realizar teste de causalidade de Granger
    try:
        results = grangercausalitytests(data[['effect', 'cause']], max_lag, verbose=False)
        
        # Formatar os resultados
        granger_results = {}
        for lag in range(1, max_lag + 1):
            p_value = results[lag][0]['ssr_chi2test'][1]
            granger_results[f"lag_{lag}"] = {
                "p_value": p_value,
                "significant": p_value < alpha
            }
        
        # Determinar se há causalidade significativa em qualquer lag
        significant_lags = sum(1 for lag in granger_results if granger_results[lag]["significant"])
        
        summary = {
            "cause_tenant": cause_tenant,
            "effect_tenant": effect_tenant,
            "significant_causal_relationship": significant_lags > 0,
            "significant_lags": significant_lags,
            "max_lag_tested": max_lag,
            "min_p_value": min(granger_results[f"lag_{lag}"]["p_value"] for lag in range(1, max_lag + 1)),
            "detail": granger_results
        }
        
        return summary
    
    except Exception as e:
        return {"error": str(e), "cause_tenant": cause_tenant, "effect_tenant": effect_tenant}


def analyze_causal_relationships(df, tenant_pairs=None, metric_column='value', phase=None):
    """
    Analisa relações causais entre pares de tenants usando causalidade de Granger.
    
    Args:
        df (DataFrame): DataFrame com dados da métrica
        tenant_pairs (list): Lista de tuplas (causa, efeito) de tenants a analisar
                            Se None, analisar todas as combinações possíveis
        metric_column (str): Coluna com os valores da métrica
        phase (str): Fase específica para análise (None = todas)
        
    Returns:
        DataFrame: Resultados da análise causal
    """
    # Filtrar pela fase se especificada
    if phase:
        df = df[df['phase'] == phase]
    
    # Se não houver pares especificados, criar todas as combinações
    if tenant_pairs is None:
        tenants = sorted(df['tenant'].unique())
        tenant_pairs = []
        
        for i, tenant1 in enumerate(tenants):
            for tenant2 in tenants:
                if tenant1 != tenant2:
                    tenant_pairs.append((tenant1, tenant2))
    
    results = []
    
    # Realizar teste para cada par
    for cause, effect in tenant_pairs:
        result = granger_causality_test(df, cause, effect, metric_column)
        
        if "error" not in result:
            # Se a análise for bem-sucedida, adicionar aos resultados
            result["phase"] = phase if phase else "all"
            results.append(result)
    
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame()


def calculate_single_metric_impact(row, baseline_phase, attack_phase, weights, metrics_config):
    # Assuming 'metric' is part of the row's name (index) or a column
    # If row is a Series from a DataFrame grouped by ['tenant', 'round', 'metric'],
    # then row.name will be a tuple like ('tenant-a', 'round-1', 'cpu_usage')
    if isinstance(row.name, tuple) and len(row.name) > 2:
        metric = row.name[2] # metric is the third level in the multi-index
    elif 'metric' in row:
        metric = row['metric']
    else:
        # Fallback or error if metric cannot be determined
        # This part needs to be robust based on how 'row' is structured when passed to apply
        # For now, let's assume it can be found or raise an error / return default
        # print(f"DEBUG: Could not determine metric from row: {row.name}")
        return 0.0 # Or raise an error

    weight = weights.get(metric, 1.0)
    # higher_is_better = metrics_config.get(metric, {}).get('higher_is_better', False)
    # Accessing higher_is_better from the global METRICS_CONFIG directly
    metric_properties = METRICS_CONFIG.get(metric, {})
    higher_is_better = metric_properties.get('higher_is_better', False)

    baseline_value = row[f'{baseline_phase}_value']
    attack_value = row[f'{attack_phase}_value']

    if pd.isna(baseline_value) or pd.isna(attack_value):
        return 0.0

    if baseline_value == 0:
        if attack_value == 0:
            impact_percentage = 0.0
        else:
            # Consider how to handle infinite impact: cap it or use a large number
            impact_percentage = float('inf') if attack_value > 0 else float('-inf') 
    else:
        impact_percentage = (attack_value - baseline_value) / baseline_value

    if higher_is_better:
        normalized_impact = -impact_percentage
    else:
        normalized_impact = impact_percentage
    
    return normalized_impact * weight


def calculate_normalized_impact_score(df, metrics, baseline_phase='baseline', attack_phase='attack', agg_type='mean'):
    """
    Calculates a normalized impact score for each tenant and round.
    The score is weighted by metric and normalized based on whether higher or lower values are better.
    """
    if df.empty:
        print("Aviso: DataFrame de entrada para calculate_normalized_impact_score está vazio.")
        return pd.DataFrame(columns=['tenant', 'round', 'aggregated_impact_score', 'average_normalized_impact'])

    if 'phase_name' in df.columns:
        print(f"DEBUG: Unique phase_name values in input df for impact score: {df['phase_name'].unique()}")
    else:
        print("DEBUG: 'phase_name' column not in input df for impact score.")
        unique_tenant_rounds = df[['tenant', 'round']].drop_duplicates() if 'tenant' in df.columns and 'round' in df.columns else pd.DataFrame(columns=['tenant', 'round'])
        final_scores_df = unique_tenant_rounds.copy()
        final_scores_df['aggregated_impact_score'] = 0.0
        final_scores_df['average_normalized_impact'] = 0.0
        return final_scores_df

    if 'value' not in df.columns:
        print("DEBUG: 'value' column not in input df for pivot_table in impact score calculation.")
        unique_tenant_rounds = df[['tenant', 'round']].drop_duplicates() if 'tenant' in df.columns and 'round' in df.columns else pd.DataFrame(columns=['tenant', 'round'])
        final_scores_df = unique_tenant_rounds.copy()
        final_scores_df['aggregated_impact_score'] = 0.0
        final_scores_df['average_normalized_impact'] = 0.0
        return final_scores_df
        
    try:
        pivot_df = df.pivot_table(index=['tenant', 'round', 'metric'], columns='phase_name', values='value', aggfunc=agg_type)
    except Exception as e:
        print(f"DEBUG: Error during pivot_table in impact score calculation: {e}")
        unique_tenant_rounds = df[['tenant', 'round']].drop_duplicates() if 'tenant' in df.columns and 'round' in df.columns else pd.DataFrame(columns=['tenant', 'round'])
        final_scores_df = unique_tenant_rounds.copy()
        final_scores_df['aggregated_impact_score'] = 0.0
        final_scores_df['average_normalized_impact'] = 0.0
        return final_scores_df

    print(f"DEBUG: Columns in pivot_df for impact score: {pivot_df.columns.tolist()}")

    impact_df = pd.DataFrame(index=pivot_df.index)
    
    # Use PHASE_DISPLAY_NAMES to get the actual column names for baseline and attack
    # Assuming baseline_phase and attack_phase are keys like 'baseline', 'attack'
    actual_baseline_phase_col = PHASE_DISPLAY_NAMES.get(baseline_phase, baseline_phase)
    actual_attack_phase_col = PHASE_DISPLAY_NAMES.get(attack_phase, attack_phase)

    if actual_baseline_phase_col not in pivot_df.columns or actual_attack_phase_col not in pivot_df.columns:
        print(f"Aviso: Fases [{actual_baseline_phase_col} ({baseline_phase}), {actual_attack_phase_col} ({attack_phase})] não encontradas nos dados pivotados. Score de impacto pode ser incompleto.")
        if not df.empty and 'tenant' in df.columns and 'round' in df.columns:
            unique_tenant_rounds = df[['tenant', 'round']].drop_duplicates()
            final_scores_df = unique_tenant_rounds.copy()
            final_scores_df['aggregated_impact_score'] = 0.0
            final_scores_df['average_normalized_impact'] = 0.0
        else:
            final_scores_df = pd.DataFrame(columns=['tenant', 'round', 'aggregated_impact_score', 'average_normalized_impact'])
        return final_scores_df
    else:
        impact_df[f'{actual_baseline_phase_col}_value'] = pivot_df[actual_baseline_phase_col]
        impact_df[f'{actual_attack_phase_col}_value'] = pivot_df[actual_attack_phase_col]

        df_grouped_for_impact_calc = impact_df.reset_index() # Makes 'tenant', 'round', 'metric' columns
        
        if 'metric' in df_grouped_for_impact_calc.columns:
            # Pass the actual column names to calculate_single_metric_impact
            df_grouped_for_impact_calc['normalized_impact'] = df_grouped_for_impact_calc.apply(
                lambda row: calculate_single_metric_impact(row, actual_baseline_phase_col, actual_attack_phase_col, 
                                                         IMPACT_SCORE_WEIGHTS, METRICS_CONFIG), axis=1
            )
        else:
            print("DEBUG: 'metric' column not found after reset_index in impact_df for impact calculation.")
            df_grouped_for_impact_calc['normalized_impact'] = 0.0

    if 'normalized_impact' in df_grouped_for_impact_calc.columns:
        final_scores_df = df_grouped_for_impact_calc.groupby(['tenant', 'round'])['normalized_impact'].sum().reset_index()
        final_scores_df.rename(columns={'normalized_impact': 'aggregated_impact_score'}, inplace=True)
        
        avg_normalized_impact = df_grouped_for_impact_calc.groupby(['tenant', 'round'])['normalized_impact'].mean().reset_index()
        avg_normalized_impact.rename(columns={'normalized_impact': 'average_normalized_impact'}, inplace=True)
        
        final_scores_df = pd.merge(final_scores_df, avg_normalized_impact, on=['tenant', 'round'], how='left')
    else:
        if not df.empty and 'tenant' in df.columns and 'round' in df.columns:
            unique_tenant_rounds = df[['tenant', 'round']].drop_duplicates()
            final_scores_df = unique_tenant_rounds.copy()
        else:
            final_scores_df = pd.DataFrame(columns=['tenant', 'round'])
        final_scores_df['aggregated_impact_score'] = 0.0
        final_scores_df['average_normalized_impact'] = 0.0

    return final_scores_df
