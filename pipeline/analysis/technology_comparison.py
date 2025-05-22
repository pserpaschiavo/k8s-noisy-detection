"""
Módulo para comparação entre experimentos com diferentes tecnologias (vCluster, Kata Containers, etc).

Este módulo implementa funções para análise comparativa de diferentes tecnologias
de containerização, focando em métricas de desempenho, isolamento e interferência.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import os
from refactor.data_handling.save_results import save_figure

def normalize_metrics_between_experiments(exp1_data, exp2_data, metrics_list=None):
    """
    Normaliza métricas entre dois experimentos para permitir comparação justa.
    
    Args:
        exp1_data (dict): Dados do primeiro experimento (dicionário de DataFrames)
        exp2_data (dict): Dados do segundo experimento  
        metrics_list (list): Lista de métricas para normalizar (None = todas comuns)
        
    Returns:
        tuple: Dicionário de métricas normalizadas para cada experimento
    """
    # Identificar métricas comuns se não especificadas
    if metrics_list is None:
        metrics_list = [m for m in exp1_data.keys() if m in exp2_data.keys()]
    
    # Dicionários para resultados normalizados
    exp1_normalized = {}
    exp2_normalized = {}
    
    # Estatísticas para normalização
    normalization_stats = {}
    
    for metric in metrics_list:
        if metric not in exp1_data or metric not in exp2_data:
            continue
            
        # Obter DataFrames
        df1 = exp1_data[metric]
        df2 = exp2_data[metric]
        
        # Estatísticas globais para normalização
        combined_values = pd.concat([df1['value'], df2['value']])
        global_min = combined_values.min()
        global_max = combined_values.max()
        global_mean = combined_values.mean()
        global_std = combined_values.std()
        
        # Salvar estatísticas
        normalization_stats[metric] = {
            'min': global_min,
            'max': global_max,
            'mean': global_mean,
            'std': global_std
        }
        
        # Normalizar usando Min-Max ou Z-score conforme necessário
        # Usamos Min-Max para a maioria das métricas (mantém proporções)
        if global_max > global_min:
            # Criar cópias para não modificar os originais
            df1_norm = df1.copy()
            df2_norm = df2.copy()
            
            # Normalização Min-Max (valores entre 0 e 1)
            df1_norm['value'] = (df1['value'] - global_min) / (global_max - global_min)
            df2_norm['value'] = (df2['value'] - global_min) / (global_max - global_min)
            
            exp1_normalized[metric] = df1_norm
            exp2_normalized[metric] = df2_norm
        else:
            # Se min=max (valores constantes), manter originais
            exp1_normalized[metric] = df1.copy()
            exp2_normalized[metric] = df2.copy()
    
    return exp1_normalized, exp2_normalized, normalization_stats

def calculate_relative_efficiency(exp1_data, exp2_data, metrics_list=None, tenants_list=None, 
                                 phase_filter=None, rounds_filter: Optional[Union[str, List[str]]] = None, 
                                 exp1_name="Experiment 1", exp2_name="Experiment 2"):
    """
    Calcula métricas de eficiência relativa entre dois experimentos.
    
    Args:
        exp1_data (dict): Dados do primeiro experimento
        exp2_data (dict): Dados do segundo experimento
        metrics_list (list): Lista de métricas a comparar
        tenants_list (list): Lista de tenants a incluir
        phase_filter (str): Filtro opcional para fase específica
        rounds_filter (Optional[Union[str, List[str]]]): Filtro opcional para round(s) específico(s).
        exp1_name (str): Nome do primeiro experimento para resultados
        exp2_name (str): Nome do segundo experimento para resultados
        
    Returns:
        DataFrame: DataFrame com métricas de eficiência relativa
    """
    results = []
    
    # Identificar métricas comuns se não especificadas
    if metrics_list is None:
        metrics_list = [m for m in exp1_data.keys() if m in exp2_data.keys()]
    
    for metric_name in metrics_list:
        if metric_name not in exp1_data or metric_name not in exp2_data:
            continue
            
        df1 = exp1_data[metric_name].copy() # Use .copy() to avoid modifying original dict data
        df2 = exp2_data[metric_name].copy() # Use .copy() to avoid modifying original dict data
        
        # Aplicar filtro de fase se especificado
        if phase_filter:
            if 'phase' in df1.columns:
                df1 = df1[df1['phase'] == phase_filter]
            if 'phase' in df2.columns:
                df2 = df2[df2['phase'] == phase_filter]

        # Aplicar filtro de round se especificado
        if rounds_filter:
            if isinstance(rounds_filter, str):
                rounds_to_keep = [rounds_filter]
            else: # Assumindo que é uma lista
                rounds_to_keep = rounds_filter
            
            if 'round' in df1.columns:
                df1 = df1[df1['round'].isin(rounds_to_keep)]
            if 'round' in df2.columns:
                df2 = df2[df2['round'].isin(rounds_to_keep)]
        
        # Filtrar tenants se especificado
        if tenants_list and 'tenant' in df1.columns and 'tenant' in df2.columns:
            df1_filtered_tenants = df1[df1['tenant'].isin(tenants_list)]
            df2_filtered_tenants = df2[df2['tenant'].isin(tenants_list)]
            
            # Carregar configurações
            from pipeline.config import DEFAULT_NOISY_TENANT
            
            # Determinar o tenant gerador de ruído
            noisy_tenant = DEFAULT_NOISY_TENANT
                
            # Verificar se o tenant gerador de ruído está presente em ambos os experimentos
            has_noisy_tenant_exp1 = noisy_tenant in df1_filtered_tenants['tenant'].unique()
            has_noisy_tenant_exp2 = noisy_tenant in df2_filtered_tenants['tenant'].unique()
            
            # Para cada tenant, calcular estatísticas
            for tenant in tenants_list:
                # Tratamento especial para o tenant gerador de ruído
                if tenant == noisy_tenant:
                    tenant_df1 = df1_filtered_tenants[df1_filtered_tenants['tenant'] == tenant] if has_noisy_tenant_exp1 else pd.DataFrame()
                    tenant_df2 = df2_filtered_tenants[df2_filtered_tenants['tenant'] == tenant] if has_noisy_tenant_exp2 else pd.DataFrame()
                    
                    # Se o tenant gerador de ruído não existe em algum experimento mas está na lista de tenants
                    if tenant_df1.empty and noisy_tenant in tenants_list:
                        other_tenant_df1 = df1_filtered_tenants[df1_filtered_tenants['tenant'] != noisy_tenant]
                        if not other_tenant_df1.empty:
                            template_df = other_tenant_df1.iloc[[0]].copy()
                            template_df['tenant'] = noisy_tenant
                            template_df['value'] = 0 
                            tenant_df1 = template_df
                    
                    if tenant_df2.empty and noisy_tenant in tenants_list:
                        other_tenant_df2 = df2_filtered_tenants[df2_filtered_tenants['tenant'] != noisy_tenant]
                        if not other_tenant_df2.empty:
                            template_df = other_tenant_df2.iloc[[0]].copy()
                            template_df['tenant'] = noisy_tenant
                            template_df['value'] = 0
                            tenant_df2 = template_df
                else:
                    tenant_df1 = df1_filtered_tenants[df1_filtered_tenants['tenant'] == tenant]
                    tenant_df2 = df2_filtered_tenants[df2_filtered_tenants['tenant'] == tenant]
                
                # Pular se não houver dados suficientes
                if tenant_df1.empty or tenant_df2.empty or len(tenant_df1['value'].dropna()) < 2 or len(tenant_df2['value'].dropna()) < 2:
                    continue
                
                # Estatísticas básicas
                mean1 = tenant_df1['value'].mean()
                mean2 = tenant_df2['value'].mean()
                std1 = tenant_df1['value'].std()
                std2 = tenant_df2['value'].std()
                
                # Cálculo de diferenças relativas
                if mean2 != 0:
                    percent_diff = ((mean1 - mean2) / abs(mean2)) * 100
                elif mean1 != 0:
                    percent_diff = np.inf * np.sign(mean1)
                else:
                    percent_diff = 0.0
                
                # T-test para significância estatística
                t_stat, p_value = stats.ttest_ind(
                    tenant_df1['value'].dropna(), 
                    tenant_df2['value'].dropna(),
                    equal_var=False,
                    nan_policy='omit'
                )
                
                # Adicionar resultados
                results.append({
                    'metric': metric_name,
                    'tenant': tenant,
                    'phase': phase_filter if phase_filter else 'all',
                    'round': ",".join(rounds_filter) if isinstance(rounds_filter, list) else rounds_filter if rounds_filter else 'all',
                    f'{exp1_name}_mean': mean1,
                    f'{exp2_name}_mean': mean2,
                    f'{exp1_name}_std': std1,
                    f'{exp2_name}_std': std2,
                    'difference': mean1 - mean2,
                    'percent_difference': percent_diff,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'statistically_significant': p_value < 0.05,
                    'better_experiment': exp1_name if mean1 < mean2 else exp2_name if mean2 < mean1 else 'equal'
                })
        else:
            if df1.empty or df2.empty or len(df1['value'].dropna()) < 2 or len(df2['value'].dropna()) < 2:
                continue

            mean1 = df1['value'].mean()
            mean2 = df2['value'].mean()
            std1 = df1['value'].std()
            std2 = df2['value'].std()
            
            if mean2 != 0:
                percent_diff = ((mean1 - mean2) / abs(mean2)) * 100
            elif mean1 != 0:
                percent_diff = np.inf * np.sign(mean1)
            else:
                percent_diff = 0.0

            t_stat, p_value = stats.ttest_ind(
                df1['value'].dropna(), 
                df2['value'].dropna(),
                equal_var=False,
                nan_policy='omit'
            )
            
            results.append({
                'metric': metric_name,
                'tenant': 'all',
                'phase': phase_filter if phase_filter else 'all',
                'round': ",".join(rounds_filter) if isinstance(rounds_filter, list) else rounds_filter if rounds_filter else 'all',
                f'{exp1_name}_mean': mean1,
                f'{exp2_name}_mean': mean2,
                f'{exp1_name}_std': std1,
                f'{exp2_name}_std': std2,
                'difference': mean1 - mean2,
                'percent_difference': percent_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'statistically_significant': p_value < 0.05,
                'better_experiment': exp1_name if mean1 < mean2 else exp2_name if mean2 < mean1 else 'equal'
            })
    
    return pd.DataFrame(results)

def plot_experiment_comparison(efficiency_data, exp1_name, exp2_name, 
                              metric_filter=None, tenant_filter=None,
                              figsize=(14, 10)):
    """
    Plota comparação visual entre experimentos com diferentes tecnologias.
    
    Args:
        efficiency_data (DataFrame): DataFrame com métricas de eficiência relativa
        exp1_name (str): Nome do primeiro experimento
        exp2_name (str): Nome do segundo experimento
        metric_filter (str): Filtrar para uma métrica específica
        tenant_filter (str): Filtrar para um tenant específico
        figsize (tuple): Tamanho da figura
        
    Returns:
        matplotlib.figure.Figure: Figura com a comparação
    """
    plot_data = efficiency_data.copy()
    
    if metric_filter:
        plot_data = plot_data[plot_data['metric'] == metric_filter]
        
    if tenant_filter:
        plot_data = plot_data[plot_data['tenant'] == tenant_filter]
    
    if plot_data.empty:
        print("Dados filtrados insuficientes para gerar visualização")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Dados insuficientes para plotar", ha='center', va='center')
        return fig

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Comparação entre {exp1_name} e {exp2_name}', fontsize=16)
    
    ax1 = axes[0, 0]
    
    if 'tenant' not in plot_data.columns:
        plot_data_pivot = plot_data.set_index('metric')[[f'{exp1_name}_mean', f'{exp2_name}_mean']]
    else:
        plot_data_pivot = plot_data.pivot_table(
            index='metric', 
            columns='tenant', 
            values=[f'{exp1_name}_mean', f'{exp2_name}_mean']
        )
        if isinstance(plot_data_pivot.columns, pd.MultiIndex):
            plot_data_pivot.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in plot_data_pivot.columns]
    
    sns.set_style("whitegrid")
    plot_data_pivot.plot(kind='bar', ax=ax1)
    ax1.set_title('Comparação de Médias por Métrica')
    ax1.set_ylabel('Valor Médio')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Tenant / Média')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = axes[0, 1]
    plot_data['abs_percent_diff'] = plot_data['percent_difference'].abs()
    significant = plot_data[plot_data['statistically_significant']]
    not_significant = plot_data[~plot_data['statistically_significant']]
    
    if not significant.empty:
        sns.barplot(
            data=significant,
            x='metric',
            y='percent_difference',
            hue='tenant' if 'tenant' in significant.columns and len(significant['tenant'].unique()) > 1 else None,
            ax=ax2,
            alpha=0.8
        )
    
    if not not_significant.empty:
        metric_categories = plot_data['metric'].unique()
        metric_map = {metric: i for i, metric in enumerate(metric_categories)}

        for _, row in not_significant.iterrows():
            x_pos = metric_map.get(row['metric'], 0)
            ax2.scatter(
                x_pos,
                row['percent_difference'],
                marker='o', s=100, color='gray', alpha=0.5,
                label='Não Significativo' if 'Não Significativo' not in [h.get_label() for h in ax2.get_legend_handles_labels()[0]] else ""
            )
        ax2.set_xticks(range(len(metric_categories)))
        ax2.set_xticklabels(metric_categories, rotation=45)

    ax2.set_title('Diferença Percentual (%)')
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    handles, labels = ax2.get_legend_handles_labels()
    unique_labels = {}
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels[l] = h
    ax2.legend(unique_labels.values(), unique_labels.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax3 = axes[1, 0]
    
    if 'tenant' in plot_data.columns and len(plot_data['tenant'].unique()) > 1 and plot_data['tenant'].nunique() > 1:
        try:
            heatmap_data = plot_data.pivot_table(
                index='metric',
                columns='tenant',
                values='percent_difference'
            )
            
            cmap = sns.diverging_palette(240, 10, as_cmap=True)
            sns.heatmap(
                heatmap_data,
                cmap=cmap,
                center=0,
                annot=True,
                fmt=".1f",
                linewidths=.5,
                ax=ax3,
                cbar_kws={'label': 'Diferença Percentual (%)'}
            )
            ax3.set_title('Heatmap de Diferenças por Tenant e Métrica')
        except Exception as e:
            ax3.text(0.5, 0.5, f'Erro ao gerar heatmap:\n{e}', 
                horizontalalignment='center', verticalalignment='center')
            ax3.set_title('Heatmap de Diferenças (Erro)')

    else:
        ax3.text(0.5, 0.5, 'Dados insuficientes para heatmap\n(necessário múltiplos tenants com dados válidos)', 
                horizontalalignment='center', verticalalignment='center')
        ax3.set_title('Heatmap de Diferenças (indisponível)')
    
    ax4 = axes[1, 1]
    if not plot_data.empty:
        sns.scatterplot(
            data=plot_data,
            x='p_value',
            y='percent_difference',
            hue='metric',
            size='abs_percent_diff',
            sizes=(20, 200),
            alpha=0.7,
            ax=ax4
        )
        
        ax4.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax4.axvline(x=0.05, color='r', linestyle='--', alpha=0.3)
        ax4.set_title('Significância Estatística vs Diferença')
        ax4.set_xlabel('p-value (menor = mais significativo)')
        ax4.set_ylabel('Diferença Percentual (%)')
        ax4.set_xscale('log')
        
        min_p_val_for_text = max(plot_data['p_value'].min() * 0.9, 1e-10)
        ax4.text(min_p_val_for_text if min_p_val_for_text < 0.05 else 0.001, ax4.get_ylim()[1]*0.8, 'Estatisticamente\nSignificativo', fontsize=9, ha='left')
        ax4.text(0.06, ax4.get_ylim()[1]*0.8, 'Não\nSignificativo', fontsize=9, ha='left')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Métrica')
    else:
        ax4.text(0.5, 0.5, "Dados insuficientes para gráfico de dispersão", ha='center', va='center')
        ax4.set_title('Significância Estatística vs Diferença (indisponível)')

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    return fig

def compare_technologies(exp1_data, exp2_data, metrics_list=None, tenants_list=None,
                        rounds_list: Optional[Union[str, List[str]]] = None, 
                        exp1_name="Tecnologia 1", exp2_name="Tecnologia 2",
                        output_dir=None, generate_plots=True):
    """
    Função principal para comparação entre experimentos com diferentes tecnologias.
    
    Args:
        exp1_data (dict): Dados do primeiro experimento (dicionário de DataFrames por métrica)
        exp2_data (dict): Dados do segundo experimento (dicionário de DataFrames por métrica)
        metrics_list (list): Lista de métricas para análise
        tenants_list (list): Lista de tenants para filtrar
        rounds_list (Optional[Union[str, List[str]]]): Lista de rounds para filtrar ou round específico.
        exp1_name (str): Nome da primeira tecnologia
        exp2_name (str): Nome da segunda tecnologia
        output_dir (str): Diretório para salvar resultados
        generate_plots (bool): Se True, gera visualizações
        
    Returns:
        dict: Resultados da comparação
    """
    results = {}
    
    exp1_norm, exp2_norm, norm_stats = normalize_metrics_between_experiments(
        exp1_data, exp2_data, metrics_list
    )
    
    results['normalized_data'] = {
        exp1_name: exp1_norm,
        exp2_name: exp2_norm
    }
    results['normalization_stats'] = norm_stats
    
    phase_names = ['1 - Baseline', '2 - Attack', '3 - Recovery'] 
    
    all_metrics_eff = calculate_relative_efficiency(
        exp1_data, exp2_data, metrics_list, tenants_list,
        phase_filter=None, rounds_filter=rounds_list, 
        exp1_name=exp1_name, exp2_name=exp2_name
    )
    
    phase_metrics_eff = {}
    for phase in phase_names:
        phase_metrics_eff[phase] = calculate_relative_efficiency(
            exp1_data, exp2_data, metrics_list, tenants_list,
            phase_filter=phase, rounds_filter=rounds_list, 
            exp1_name=exp1_name, exp2_name=exp2_name
        )
    
    results['efficiency_metrics'] = {
        'all_phases_and_rounds_specified': all_metrics_eff,
        'by_phase_with_rounds_specified': phase_metrics_eff
    }
    
    if generate_plots and output_dir:
        plots_dir = os.path.join(output_dir, 'technology_comparison')
        os.makedirs(plots_dir, exist_ok=True)
        
        rounds_suffix = ""
        if rounds_list:
            if isinstance(rounds_list, list):
                rounds_suffix = "_rounds_" + "_".join(sorted(list(set(rounds_list))))
            else:
                rounds_suffix = "_round_" + rounds_list
        
        fig_all = plot_experiment_comparison(
            all_metrics_eff, exp1_name, exp2_name,
            figsize=(14, 10)
        )
        
        if fig_all:
            plot_filename = f'comparison_{exp1_name}_vs_{exp2_name}_all_phases{rounds_suffix}.png'
            try:
                save_figure(fig_all, plot_filename, plots_dir)
                print(f"Plot geral salvo em: {os.path.join(plots_dir, plot_filename)}")
            except Exception as e:
                print(f"Erro ao salvar plot geral: {e}")
            plt.close(fig_all)
        
        for phase, phase_data in phase_metrics_eff.items():
            phase_label = phase.replace(' ', '_').lower()
            
            fig_phase = plot_experiment_comparison(
                phase_data, exp1_name, exp2_name,
                figsize=(14, 10)
            )
            
            if fig_phase:
                plot_filename = f'comparison_{exp1_name}_vs_{exp2_name}_{phase_label}{rounds_suffix}.png'
                try:
                    save_figure(fig_phase, plot_filename, plots_dir)
                    print(f"Plot da fase {phase} salvo em: {os.path.join(plots_dir, plot_filename)}")
                except Exception as e:
                    print(f"Erro ao salvar plot da fase {phase}: {e}")
                plt.close(fig_phase)
        
        active_metrics_list = metrics_list if metrics_list else list(exp1_data.keys())

        for metric in active_metrics_list:
            metric_specific_eff_data = all_metrics_eff[all_metrics_eff['metric'] == metric]
            if metric_specific_eff_data.empty:
                print(f"Sem dados de eficiência para a métrica {metric} para o plot individual.")
                continue

            fig_metric = plot_experiment_comparison(
                metric_specific_eff_data,
                exp1_name, exp2_name,
                metric_filter=metric,
                figsize=(14, 10)
            )
            
            if fig_metric:
                plot_filename = f'comparison_{exp1_name}_vs_{exp2_name}_{metric}{rounds_suffix}.png'
                try:
                    save_figure(fig_metric, plot_filename, plots_dir)
                    print(f"Plot da métrica {metric} salvo em: {os.path.join(plots_dir, plot_filename)}")
                except Exception as e:
                    print(f"Erro ao salvar plot da métrica {metric}: {e}")
                plt.close(fig_metric)
        
        results['plot_paths'] = {
            'all_phases_and_specified_rounds': os.path.join(plots_dir, f'comparison_{exp1_name}_vs_{exp2_name}_all_phases{rounds_suffix}.png'),
            'by_phase_with_specified_rounds': {
                phase: os.path.join(plots_dir, f'comparison_{exp1_name}_vs_{exp2_name}_{phase.replace(" ", "_").lower()}{rounds_suffix}.png') 
                for phase in phase_names if not phase_metrics_eff.get(phase, pd.DataFrame()).empty
            },
            'by_metric_with_specified_rounds': {
                metric: os.path.join(plots_dir, f'comparison_{exp1_name}_vs_{exp2_name}_{metric}{rounds_suffix}.png')
                for metric in active_metrics_list if not all_metrics_eff[all_metrics_eff['metric'] == metric].empty
            }
        }
    
    return results
