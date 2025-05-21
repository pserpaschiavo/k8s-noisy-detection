\
# filepath: /home/phil/Projects/k8s-noisy-detection/refactor/analysis_modules/descritive_statistics.py
import pandas as pd
import numpy as np

def calculate_descriptive_statistics(data_df: pd.DataFrame, metric_column: str = 'value', groupby_cols: list[str] | None = None):
    """
    Calcula estatísticas descritivas para uma determinada métrica, opcionalmente agrupada.

    Args:
        data_df (pd.DataFrame): DataFrame contendo os dados. 
                                Espera colunas como 'tenant', 'round', 'phase', e a metric_column.
        metric_column (str): Nome da coluna contendo os valores da métrica a ser analisada.
        groupby_cols (list[str] | None): Lista de colunas para agrupar os dados antes de calcular as estatísticas.
                             Ex: ['round', 'phase', 'tenant'] ou ['round', 'tenant'] etc.
                             Se None, calcula para todo o DataFrame.

    Returns:
        pd.DataFrame: DataFrame com as estatísticas descritivas.
    """
    if not isinstance(data_df, pd.DataFrame):
        raise ValueError("Input data_df must be a pandas DataFrame.")
    
    if metric_column not in data_df.columns:
        raise ValueError(f"Metric column '{metric_column}' not found in DataFrame.")

    if groupby_cols:
        if not all(col in data_df.columns for col in groupby_cols):
            missing_cols = [col for col in groupby_cols if col not in data_df.columns]
            raise ValueError(f"Grouping columns not found in DataFrame: {missing_cols}")
        
        grouped = data_df.groupby(groupby_cols)
        descriptive_stats = grouped[metric_column].agg([
            'count', 'mean', 'std', 'min', 
            lambda x: x.quantile(0.25), 
            'median', 
            lambda x: x.quantile(0.75), 
            'max',
            lambda x: x.max() - x.min(), # Range
            'skew', # Assimetria
            lambda x: x.kurtosis()  # Curtose
        ]).reset_index()
        
        # Renomear colunas lambda de forma mais robusta
        rename_map = {}
        lambda_idx = 0
        for col in descriptive_stats.columns:
            if '<lambda' in col:
                if lambda_idx == 0: rename_map[col] = 'q1'
                elif lambda_idx == 1: rename_map[col] = 'q3'
                elif lambda_idx == 2: rename_map[col] = 'range'
                elif lambda_idx == 3: rename_map[col] = 'kurtosis' # Adicionado para curtose
                lambda_idx +=1
        descriptive_stats.rename(columns=rename_map, inplace=True)
        
    else:
        # Calcula estatísticas para todo o DataFrame se groupby_cols não for fornecido
        stats_list = data_df[metric_column].agg([
            'count', 'mean', 'std', 'min', 
            lambda x: x.quantile(0.25), 
            'median', 
            lambda x: x.quantile(0.75), 
            'max',
            lambda x: x.max() - x.min(),
            'skew',
            lambda x: x.kurtosis()
        ])
        descriptive_stats = pd.DataFrame(stats_list).T
        
        # Renomear colunas lambda de forma mais robusta
        rename_map = {}
        # As colunas no DataFrame transposto serão os nomes das agregações
        # e os nomes dos lambdas serão os índices da Series original
        original_lambda_names = [idx for idx in stats_list.index if isinstance(idx, str) and '<lambda' in idx]

        if len(original_lambda_names) > 0: rename_map[original_lambda_names[0]] = 'q1'
        if len(original_lambda_names) > 1: rename_map[original_lambda_names[1]] = 'q3'
        if len(original_lambda_names) > 2: rename_map[original_lambda_names[2]] = 'range'
        if len(original_lambda_names) > 3: rename_map[original_lambda_names[3]] = 'kurtosis' # Adicionado para curtose
        
        descriptive_stats.rename(columns=rename_map, inplace=True)

    return descriptive_stats

# Exemplo de como poderia ser usado (para ser removido ou comentado depois):
# if __name__ == '__main__':
#     # Criar um DataFrame de exemplo
#     example_data = {
#         'round': ['r1', 'r1', 'r1', 'r1', 'r2', 'r2', 'r2', 'r2'],
#         'phase': ['p1', 'p1', 'p2', 'p2', 'p1', 'p1', 'p2', 'p2'],
#         'tenant': ['tA', 'tB', 'tA', 'tB', 'tA', 'tB', 'tA', 'tB'],
#         'value': [10, 12, 15, 16, 11, 13, 14, 15]
#     }
#     df_example = pd.DataFrame(example_data)
#
#     print("--- Estatísticas Descritivas Agrupadas por Round, Phase, Tenant ---")
#     stats_grouped_all = calculate_descriptive_statistics(df_example, 'value', groupby_cols=['round', 'phase', 'tenant'])
#     print(stats_grouped_all)
#     print("\\n")
#
#     print("--- Estatísticas Descritivas Agrupadas por Tenant ---")
#     stats_grouped_tenant = calculate_descriptive_statistics(df_example, 'value', groupby_cols=['tenant'])
#     print(stats_grouped_tenant)
#     print("\\n")
#     
#     print("--- Estatísticas Descritivas Gerais ---")
#     stats_overall = calculate_descriptive_statistics(df_example, 'value')
#     print(stats_overall)

