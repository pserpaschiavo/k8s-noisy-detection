"""
Módulo para geração de tabelas em formatos LaTeX e CSV para o experimento de noisy neighbors.

Este módulo fornece funções para criar tabelas bem formatadas com os resultados
do experimento, adequadas para publicações acadêmicas.
"""

import pandas as pd
import numpy as np
import os


def convert_df_to_markdown(df, float_format='.2f', index=False):
    """
    Converte um DataFrame para uma tabela em formato Markdown.
    
    Args:
        df (DataFrame): DataFrame a ser convertido
        float_format (str, optional): Formato para valores de ponto flutuante. Default '.2f'.
        index (bool, optional): Se deve incluir o índice na tabela. Default False.
        
    Returns:
        str: String contendo a tabela em formato Markdown
    """
    # Formatar float columns antes de converter
    formatted_df = df.copy()
    for col in formatted_df.columns:
        if formatted_df[col].dtype in [np.float32, np.float64]:
            formatted_df[col] = formatted_df[col].map(lambda x: f"{x:{float_format}}" if not pd.isna(x) else "")

    # Converter para Markdown
    markdown_table = formatted_df.to_markdown(index=index)
    
    return markdown_table


def create_phase_comparison_table(df, metric_name, phases=None, tenants=None, 
                                 value_column='mean', std_column='std'):
    """
    Cria uma tabela formatada comparando métricas entre fases para cada tenant.
    
    Args:
        df (DataFrame): DataFrame com dados agregados
        metric_name (str): Nome da métrica para o título da tabela
        phases (list): Lista de fases a incluir (None = todas)
        tenants (list): Lista de tenants a incluir (None = todos)
        value_column (str): Nome da coluna com valores médios
        std_column (str): Nome da coluna com desvios padrão
        
    Returns:
        DataFrame: DataFrame formatado para exportação
    """
    # Filtrar dados se necessário
    data = df.copy()
    if phases:
        data = data[data['phase'].isin(phases)]
    if tenants:
        data = data[data['tenant'].isin(tenants)]
    
    # Criar uma tabela formatada
    formatted_table = pd.DataFrame()
    
    # Para cada tenant, adicionar uma linha
    for tenant, tenant_data in data.groupby('tenant'):
        row = {'Tenant': tenant}
        
        # Para cada fase, adicionar média ± desvio padrão
        for phase, phase_data in tenant_data.groupby('phase'):
            if len(phase_data) > 0:
                mean_val = phase_data[value_column].iloc[0]
                std_val = phase_data[std_column].iloc[0] if std_column in phase_data.columns else 0
                
                # Formatar como "média ± desvio"
                row[phase] = f"{mean_val:.2f} ± {std_val:.2f}"
        
        # Adicionar linha ao DataFrame
        formatted_table = pd.concat([formatted_table, pd.DataFrame([row])], ignore_index=True)
    
    return formatted_table


def create_impact_summary_table(impact_df, metric_name, round_column='round', 
                              tenant_column='tenant', impact_column='impact_percent'):
    """
    Cria uma tabela resumida do impacto percentual em cada tenant durante a fase de ataque.
    
    Args:
        impact_df (DataFrame): DataFrame com impacto calculado por tenant
        metric_name (str): Nome da métrica para o título da tabela
        round_column (str): Nome da coluna com os nomes dos rounds
        tenant_column (str): Nome da coluna com os nomes dos tenants
        impact_column (str): Nome da coluna com os valores percentuais de impacto
        
    Returns:
        DataFrame: DataFrame formatado para exportação
    """
    # Criar tabela pivotada com rounds nas colunas e tenants nas linhas
    pivot_table = impact_df.pivot_table(
        index=tenant_column,
        columns=round_column,
        values=impact_column
    )
    
    # Adicionar média de impacto entre rounds
    pivot_table['Média'] = pivot_table.mean(axis=1)
    
    # Formatar a tabela para exportação
    formatted_table = pivot_table.copy()
    
    for col in formatted_table.columns:
        formatted_table[col] = formatted_table[col].apply(lambda x: f"{x:.2f}%" if not pd.isna(x) else "")
    
    return formatted_table


def create_causality_results_table(causal_links_df,
                                   source_col='Source Tenant',
                                   source_metric_col='Source Metric',
                                   target_col='Target Tenant',
                                   target_metric_col='Target Metric',
                                   lag_col='Lag (Granger)',
                                   p_value_col='P-Value (Granger)'):
    """
    Creates a formatted table from a DataFrame of causal links.

    Args:
        causal_links_df (pd.DataFrame): DataFrame containing the causal links.
                                        Expected columns: 'Source Tenant', 'Source Metric', 
                                                        'Target Tenant', 'Target Metric', 
                                                        'Lag (Granger)', 'P-Value (Granger)'.
        source_col (str): Name for the source tenant column in the output table.
        source_metric_col (str): Name for the source metric column in the output table.
        target_col (str): Name for the target tenant column in the output table.
        target_metric_col (str): Name for the target metric column in the output table.
        lag_col (str): Name for the lag column in the output table.
        p_value_col (str): Name for the p-value column in the output table.

    Returns:
        pd.DataFrame: A DataFrame formatted for export, with renamed columns
                      and formatted p-values.
    """
    if causal_links_df.empty:
        # Return an empty DataFrame with expected column names if no links are found
        return pd.DataFrame(columns=[
            source_col, source_metric_col, target_col, target_metric_col,
            lag_col, p_value_col
        ])

    # Select and rename columns
    table_df = causal_links_df[[
        'Source Tenant', 'Source Metric', 'Target Tenant', 'Target Metric',
        'Lag (Granger)', 'P-Value (Granger)'
    ]].copy()

    table_df.rename(columns={
        'Source Tenant': source_col,
        'Source Metric': source_metric_col,
        'Target Tenant': target_col,
        'Target Metric': target_metric_col,
        'Lag (Granger)': lag_col,
        'P-Value (Granger)': p_value_col
    }, inplace=True)

    # Format P-Value
    if p_value_col in table_df.columns:
        table_df[p_value_col] = table_df[p_value_col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    # Sort by P-Value and then by Lag
    table_df.sort_values(by=[p_value_col, lag_col], ascending=[True, True], inplace=True)
    
    return table_df
