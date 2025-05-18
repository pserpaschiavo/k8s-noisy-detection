"""
Módulo para agregação de dados entre múltiplos rounds do experimento.

Este módulo fornece funções para consolidar dados entre diferentes rounds,
permitindo uma visão média do comportamento dos tenants em múltiplas execuções.
"""

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats

# def aggregate_metrics_across_rounds(metrics_data, value_column='value', include_std=True):
#     """
#     Agrega métricas entre múltiplos rounds, calculando médias e desvios padrão.
#     
#     Args:
#         metrics_data (dict): Dicionário com DataFrames para cada métrica
#         value_column (str): Nome da coluna com os valores a serem agregados
#         include_std (bool): Se True, inclui o desvio padrão nos resultados
#         
#     Returns:
#         dict: Dicionário com DataFrames agregados para cada métrica
#     """
#     aggregated_metrics = {}
#     
#     for metric_name, df in metrics_data.items():
#         # Verificar se o DataFrame contém a coluna de round
#         if 'round' not in df.columns:
#             continue
#         
#         # Fazer uma cópia para evitar modificar o original
#         df_clean = df.copy()
#         
#         # Verificar se tenant-b está presente nos dados
#         has_tenant_b = 'tenant' in df_clean.columns and 'tenant-b' in df_clean['tenant'].unique()
#         
#         # Preparar dados de tenant-b se necessário
#         if 'tenant' in df_clean.columns and not has_tenant_b:
#             # Verificar quais tenants estão disponíveis
#             available_tenants = df_clean['tenant'].unique()
#             if available_tenants.size > 0:
#                 # Usar o primeiro tenant disponível como referência para criar dados para tenant-b
#                 reference_tenant = available_tenants[0]
#                 template_data = []
#                 
#                 # Para cada combinação de round e phase, criar registros para tenant-b
#                 for (round_name, phase_name), group in df_clean.groupby(['round', 'phase']):
#                     reference_data = group[group['tenant'] == reference_tenant]
#                     
#                     for _, row in reference_data.iterrows():
#                         new_row = row.copy()
#                         new_row['tenant'] = 'tenant-b'
#                         new_row[value_column] = 0  # Valores zerados
#                         template_data.append(new_row)
#                 
#                 # Adicionar os dados simulados para tenant-b ao DataFrame
#                 if template_data:
#                     tenant_b_df = pd.DataFrame(template_data)
#                     df_clean = pd.concat([df_clean, tenant_b_df], ignore_index=True)
#             
#         # Agrupar por fase, tenant e timestamp relativo à fase
#         # Primeiro, criar um identificador para alinhar timestamps entre rounds
#         if 'phase' in df_clean.columns and 'tenant' in df_clean.columns and 'experiment_elapsed_seconds' in df_clean.columns:
#             # Calcular o tempo relativo ao início de cada fase, por round
#             phase_starts = df_clean.groupby(['round', 'phase'])['experiment_elapsed_seconds'].min().reset_index()
#             phase_starts.rename(columns={'experiment_elapsed_seconds': 'phase_start_time'}, inplace=True)
#             
#             # Mesclar com o DataFrame original
#             df_with_phase_start = pd.merge(
#                 df_clean, 
#                 phase_starts, 
#                 on=['round', 'phase'], 
#                 how='left'
#             )
#             
#             # Calcular o tempo relativo à fase
#             df_with_phase_start['time_since_phase_start'] = df_with_phase_start['experiment_elapsed_seconds'] - df_with_phase_start['phase_start_time']
#             
#             # Discretizar o tempo relativo para alinhamento entre rounds (por exemplo, em intervalos de 5 segundos)
#             df_with_phase_start['time_bin'] = (df_with_phase_start['time_since_phase_start'] // 5) * 5
#             
#             # Tratar valores NaN para tenant-b
#             if 'tenant' in df_with_phase_start.columns:
#                 tenant_b_mask = df_with_phase_start['tenant'] == 'tenant-b'
#                 if tenant_b_mask.any():
#                     # Substituir NaNs por zeros apenas para o tenant-b 
#                     df_with_phase_start.loc[tenant_b_mask, value_column] = df_with_phase_start.loc[tenant_b_mask, value_column].fillna(0)
#             
#             # Agrupar e calcular estatísticas
#             grouped = df_with_phase_start.groupby(['phase', 'tenant', 'time_bin'])[value_column]
#             
#             # Calcular média e outros estatísticas
#             agg_results = grouped.agg(['mean', 'std', 'count', 'min', 'max']).reset_index()
#             
#             # Adicionar intervalo de confiança de 95%
#             agg_results['ci_95'] = 1.96 * agg_results['std'] / np.sqrt(agg_results['count'])
#             
#             # Criar tempo contínuo baseado no bin para visualização
#             agg_results['experiment_elapsed_seconds'] = agg_results['time_bin']
#             
#             # Salvar no dicionário de resultados
#             aggregated_metrics[metric_name] = agg_results
#     
#     return aggregated_metrics

# def plot_aggregated_metrics(aggregated_data, metric_name, figsize=(12, 8), 
#                             show_confidence=True, highlight_tenant=None):
#     """
#     Plota métricas agregadas entre rounds, mostrando média e variabilidade.
#     
#     Args:
#         aggregated_data (DataFrame): DataFrame com métricas agregadas
#         metric_name (str): Nome da métrica para o título do gráfico
#         figsize (tuple): Tamanho do gráfico (largura, altura)
#         show_confidence (bool): Se True, mostra intervalo de confiança
#         highlight_tenant (str): Tenant a ser destacado no gráfico
#     
#     Returns:
#         matplotlib.figure.Figure: Objeto figura criado
#     """
#     from pipeline.config import TENANT_COLORS, PHASE_DISPLAY_NAMES
#     
#     if metric_name not in aggregated_data:
#         return None
#     
#     df = aggregated_data[metric_name]
#     
#     # Criar figura
#     fig, ax = plt.subplots(figsize=figsize)
#     
#     # Plotar para cada tenant, separado por fase
#     for phase_name in sorted(df['phase'].unique()):
#         phase_display = PHASE_DISPLAY_NAMES.get(phase_name, phase_name)
#         
#         # Adicionar marcador vertical para separar fases
#         if df[df['phase'] == phase_name]['experiment_elapsed_seconds'].min() > 0:
#             ax.axvline(x=df[df['phase'] == phase_name]['experiment_elapsed_seconds'].min(), 
#                     color='gray', linestyle='--', alpha=0.7)
#         
#         # Plotar dados para cada tenant nesta fase
#         for tenant in sorted(df['tenant'].unique()):
#             tenant_data = df[(df['phase'] == phase_name) & (df['tenant'] == tenant)]
#             
#             # Definir estilo da linha baseado em se é o tenant destacado
#             line_width = 2.5 if tenant == highlight_tenant else 1.5
#             line_style = '-' if tenant == highlight_tenant else '--'
#             alpha_value = 1.0 if tenant == highlight_tenant else 0.7
#             
#             # Cor do tenant
#             color = TENANT_COLORS.get(tenant, 'gray')
#             
#             # Plotar linha média
#             ax.plot(tenant_data['experiment_elapsed_seconds'], tenant_data['mean'], 
#                    label=f"{tenant} ({phase_display})" if tenant == highlight_tenant else tenant,
#                    color=color, linewidth=line_width, linestyle=line_style, alpha=alpha_value)
#             
#             # Adicionar intervalo de confiança/variabilidade se solicitado
#             if show_confidence:
#                 ax.fill_between(tenant_data['experiment_elapsed_seconds'],
#                                tenant_data['mean'] - tenant_data['ci_95'],
#                                tenant_data['mean'] + tenant_data['ci_95'],
#                                color=color, alpha=0.2)
#     
#     # Configurações do gráfico
#     ax.set_title(f"Média entre Rounds: {metric_name}")
#     ax.set_xlabel("Tempo desde Início da Fase (segundos)")
#     ax.set_ylabel(f"Valor Médio de {metric_name}")
#     ax.grid(True, linestyle='--', alpha=0.7)
#     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     
#     plt.tight_layout()
#     return fig

# def test_for_significant_differences(metrics_data, metric_name, phase_name, 
#                                      tenant1, tenant2, alpha=0.05):
#     """
#     Testa se há diferenças estatisticamente significativas entre dois tenants.
#     
#     Args:
#         metrics_data (dict): Dicionário com DataFrames para cada métrica
#         metric_name (str): Nome da métrica a ser testada
#         phase_name (str): Nome da fase a ser testada
#         tenant1 (str): Primeiro tenant para comparação
#         tenant2 (str): Segundo tenant para comparação
#         alpha (float): Nível de significância para o teste
#         
#     Returns:
#         dict: Resultados do teste, incluindo p-valor e se há diferença significativa
#     """
#     if metric_name not in metrics_data:
#         return {"error": f"Métrica {metric_name} não encontrada"}
#     
#     df = metrics_data[metric_name]
#     
#     # Filtrar dados para a fase e tenants específicos
#     tenant1_data = df[(df['phase'] == phase_name) & (df['tenant'] == tenant1)]['value']
#     tenant2_data = df[(df['phase'] == phase_name) & (df['tenant'] == tenant2)]['value']
#     
#     if tenant1_data.empty or tenant2_data.empty:
#         return {
#             "error": f"Dados insuficientes para tenant(s) na fase {phase_name}",
#             "tenant1_count": len(tenant1_data),
#             "tenant2_count": len(tenant2_data)
#         }
#     
#     # Realizar teste t para amostras independentes
#     t_stat, p_value = stats.ttest_ind(tenant1_data, tenant2_data, equal_var=False)
#     
#     # Calcular tamanho do efeito (Cohen's d)
#     mean1, mean2 = tenant1_data.mean(), tenant2_data.mean()
#     std1, std2 = tenant1_data.std(), tenant2_data.std()
#     
#     # Variância agrupada
#     n1, n2 = len(tenant1_data), len(tenant2_data)
#     pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
#     
#     # Cohen's d
#     cohen_d = abs(mean1 - mean2) / pooled_std
#     
#     # Interpretar o efeito
#     effect_size = "small" if cohen_d < 0.5 else "medium" if cohen_d < 0.8 else "large"
#     
#     return {
#         "tenant1": tenant1,
#         "tenant2": tenant2,
#         "phase": phase_name,
#         "t_statistic": t_stat,
#         "p_value": p_value,
#         "significant": p_value < alpha,
#         "mean_difference": mean1 - mean2,
#         "percent_difference": ((mean1 - mean2) / mean2) * 100 if mean2 != 0 else float('inf'),
#         "effect_size": cohen_d,
#         "effect_interpretation": effect_size
#     }
