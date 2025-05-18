"""
Módulo para geração de tabelas em formatos LaTeX e CSV para o experimento de noisy neighbors.

Este módulo fornece funções para criar tabelas bem formatadas com os resultados
do experimento, adequadas para publicações acadêmicas.
"""

import pandas as pd
import numpy as np
import os
from pipeline.config import TABLE_EXPORT_CONFIG  # Import TABLE_EXPORT_CONFIG


def format_float_columns(df, float_format='.2f'):
    """
    Formata colunas de ponto flutuante em um DataFrame.
    
    Args:
        df (DataFrame): DataFrame a ser formatado
        float_format (str): Formato a ser aplicado (ex: '.2f', '.3f')
        
    Returns:
        DataFrame: DataFrame com colunas formatadas
    """
    # Fazer uma cópia para não modificar o original
    result = df.copy()
    
    # Para cada coluna, verificar se é ponto flutuante
    for col in result.columns:
        if result[col].dtype in [np.float32, np.float64]:
            result[col] = result[col].map(lambda x: f"{x:{float_format}}" if not pd.isna(x) else "")
    
    return result


def convert_df_to_markdown(df, float_format=None, index=None):
    """
    Converte um DataFrame para uma tabela em formato Markdown.
    
    Args:
        df (DataFrame): DataFrame a ser convertido
        float_format (str, optional): Formato para valores de ponto flutuante. Usa config se None.
        index (bool, optional): Se deve incluir o índice na tabela. Usa config se None.
        
    Returns:
        str: String contendo a tabela em formato Markdown
    """
    # Usar valores do TABLE_EXPORT_CONFIG se não fornecidos
    final_float_format = float_format if float_format is not None else TABLE_EXPORT_CONFIG.get('float_format', '.2f')
    final_index = index if index is not None else TABLE_EXPORT_CONFIG.get('include_index', False)

    # Formatar float columns antes de converter
    formatted_df = format_float_columns(df, final_float_format)
    
    # Converter para Markdown
    markdown_table = formatted_df.to_markdown(index=final_index)
    
    return markdown_table


def export_to_latex(df, caption, label, filename, float_format=None, index=None, 
                   column_format=None, escape=True, longtable=None):
    """
    Exporta um DataFrame para uma tabela LaTeX formatada.
    
    Args:
        df (DataFrame): DataFrame a ser exportado
        caption (str): Legenda da tabela
        label (str): Identificador para referência cruzada
        filename (str): Nome do arquivo de saída
        float_format (str, optional): Formato para valores de ponto flutuante. Usa config se None.
        index (bool, optional): Se deve incluir o índice na tabela. Usa config se None.
        column_format (str): Formato de coluna personalizado para LaTeX
        escape (bool): Se deve escapar caracteres especiais
        longtable (bool, optional): Se deve usar o ambiente longtable. Usa config se None.
    """
    # Garantir que o diretório existe
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    # Usar valores do TABLE_EXPORT_CONFIG se não fornecidos
    final_float_format_str = float_format if float_format is not None else TABLE_EXPORT_CONFIG.get('float_format', '.2f')
    final_index = index if index is not None else TABLE_EXPORT_CONFIG.get('include_index', False)
    final_longtable = longtable if longtable is not None else TABLE_EXPORT_CONFIG.get('longtable', False)

    # Formatar colunas de ponto flutuante para string ANTES de to_latex
    formatted_df = format_float_columns(df, final_float_format_str)
    
    # Exportar para LaTeX
    latex_table = formatted_df.to_latex(
        index=final_index,
        caption=caption,
        label=label,
        float_format=None,  # Floats já são strings formatadas
        column_format=column_format,
        escape=escape,
        longtable=final_longtable
    )
    
    # Adicionar pacotes e formatação adicional
    latex_preamble = """\\documentclass{article}
\\usepackage{booktabs}
\\usepackage{siunitx}
\\usepackage{caption}
\\usepackage{longtable}
\\usepackage{multirow}
\\usepackage{array}
\\usepackage{colortbl}
\\begin{document}
"""
    latex_end = "\\end{document}"
    
    with open(filename, 'w') as f:
        f.write(latex_preamble)
        f.write(latex_table)
        f.write(latex_end)
    
    print(f"Tabela exportada como LaTeX para {filename}")


def export_to_csv(df, filename, float_format=None):
    """
    Exporta um DataFrame para um arquivo CSV formatado.
    
    Args:
        df (DataFrame): DataFrame a ser exportado
        filename (str): Nome do arquivo de saída
        float_format (str, optional): Formato para valores de ponto flutuante. Usa config se None.
    """
    # Garantir que o diretório existe
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Usar valor do TABLE_EXPORT_CONFIG se não fornecido
    final_float_format = float_format if float_format is not None else TABLE_EXPORT_CONFIG.get('float_format', '.2f')

    # Formatar float columns antes de exportar
    formatted_df = format_float_columns(df, final_float_format)
    
    # Exportar para CSV
    formatted_df.to_csv(filename, index=False)
    
    print(f"Tabela exportada como CSV para {filename}")


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
