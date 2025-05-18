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
                                 phase_filter=None, exp1_name="Experiment 1", exp2_name="Experiment 2"):
    """
    Calcula métricas de eficiência relativa entre dois experimentos.
    
    Args:
        exp1_data (dict): Dados do primeiro experimento
        exp2_data (dict): Dados do segundo experimento
        metrics_list (list): Lista de métricas a comparar
        tenants_list (list): Lista de tenants a incluir
        phase_filter (str): Filtro opcional para fase específica
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
            
        df1 = exp1_data[metric_name]
        df2 = exp2_data[metric_name]
        
        # Aplicar filtro de fase se especificado
        if phase_filter:
            if 'phase' in df1.columns:
                df1 = df1[df1['phase'] == phase_filter]
            if 'phase' in df2.columns:
                df2 = df2[df2['phase'] == phase_filter]
        
        # Filtrar tenants se especificado
        if tenants_list and 'tenant' in df1.columns and 'tenant' in df2.columns:
            df1 = df1[df1['tenant'].isin(tenants_list)]
            df2 = df2[df2['tenant'].isin(tenants_list)]
            
            # Carregar configurações
            from pipeline.config import DEFAULT_NOISY_TENANT
            
            # Determinar o tenant gerador de ruído
            noisy_tenant = None
            if 'noisy_tenant' in locals():
                noisy_tenant = locals()['noisy_tenant']
            else:
                noisy_tenant = DEFAULT_NOISY_TENANT
                
            # Verificar se o tenant gerador de ruído está presente em ambos os experimentos
            has_noisy_tenant_exp1 = noisy_tenant in df1['tenant'].unique()
            has_noisy_tenant_exp2 = noisy_tenant in df2['tenant'].unique()
            
            # Para cada tenant, calcular estatísticas
            for tenant in tenants_list:
                # Tratamento especial para o tenant gerador de ruído
                if tenant == noisy_tenant:
                    tenant_df1 = df1[df1['tenant'] == tenant] if has_noisy_tenant_exp1 else df1[df1['tenant'] == tenant].copy()
                    tenant_df2 = df2[df2['tenant'] == tenant] if has_noisy_tenant_exp2 else df2[df2['tenant'] == tenant].copy()
                    
                    # Se o tenant gerador de ruído não existe em algum experimento mas está na lista de tenants
                    if (not has_noisy_tenant_exp1 or tenant_df1.empty) and noisy_tenant in tenants_list:
                        # Criar dados simulados para o tenant gerador de ruído no exp1 usando timestamps de outro tenant
                        other_tenant = next((t for t in df1['tenant'].unique() if t != noisy_tenant), None)
                        if other_tenant:
                            template_df = df1[df1['tenant'] == other_tenant].copy()
                            tenant_df1 = template_df.copy()
                            tenant_df1['tenant'] = noisy_tenant
                            tenant_df1['value'] = 0  # Valores zerados
                    
                    if (not has_noisy_tenant_exp2 or tenant_df2.empty) and noisy_tenant in tenants_list:
                        # Criar dados simulados para o tenant gerador de ruído no exp2
                        other_tenant = next((t for t in df2['tenant'].unique() if t != noisy_tenant), None)
                        if other_tenant:
                            template_df = df2[df2['tenant'] == other_tenant].copy()
                            tenant_df2 = template_df.copy()
                            tenant_df2['tenant'] = noisy_tenant
                            tenant_df2['value'] = 0  # Valores zerados
                else:
                    tenant_df1 = df1[df1['tenant'] == tenant]
                    tenant_df2 = df2[df2['tenant'] == tenant]
                
                # Pular se não houver dados suficientes
                if (tenant != 'tenant-b' and (len(tenant_df1) < 5 or len(tenant_df2) < 5)):
                    continue
                
                # Estatísticas básicas
                mean1 = tenant_df1['value'].mean()
                mean2 = tenant_df2['value'].mean()
                std1 = tenant_df1['value'].std()
                std2 = tenant_df2['value'].std()
                
                # Cálculo de diferenças relativas
                if mean2 != 0:
                    percent_diff = ((mean1 - mean2) / abs(mean2)) * 100
                else:
                    percent_diff = np.nan
                
                # T-test para significância estatística
                t_stat, p_value = stats.ttest_ind(
                    tenant_df1['value'].dropna(), 
                    tenant_df2['value'].dropna(),
                    equal_var=False
                )
                
                # Adicionar resultados
                results.append({
                    'metric': metric_name,
                    'tenant': tenant,
                    'phase': phase_filter if phase_filter else 'all',
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
            # Comparação global de métricas sem distinção de tenant
            mean1 = df1['value'].mean()
            mean2 = df2['value'].mean()
            std1 = df1['value'].std()
            std2 = df2['value'].std()
            
            # Cálculo de diferenças relativas
            if mean2 != 0:
                percent_diff = ((mean1 - mean2) / abs(mean2)) * 100
            else:
                percent_diff = np.nan
            
            # T-test para significância estatística
            t_stat, p_value = stats.ttest_ind(
                df1['value'].dropna(), 
                df2['value'].dropna(),
                equal_var=False
            )
            
            # Adicionar resultados
            results.append({
                'metric': metric_name,
                'tenant': 'all',
                'phase': phase_filter if phase_filter else 'all',
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
    # Filtrar dados se necessário
    plot_data = efficiency_data.copy()
    
    if metric_filter:
        plot_data = plot_data[plot_data['metric'] == metric_filter]
        
    if tenant_filter:
        plot_data = plot_data[plot_data['tenant'] == tenant_filter]
    
    if plot_data.empty:
        print("Dados filtrados insuficientes para gerar visualização")
        return None
        
    # Configurar figura
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Comparação entre {exp1_name} e {exp2_name}', fontsize=16)
    
    # Subplot 1: Comparação de médias por métrica
    ax1 = axes[0, 0]
    # Preparar dados para este gráfico
    plot_means = plot_data.pivot_table(
        index='metric', 
        columns='tenant', 
        values=[f'{exp1_name}_mean', f'{exp2_name}_mean']
    ).reset_index()
    
    # Flatten columns em caso de MultiIndex
    if isinstance(plot_means.columns, pd.MultiIndex):
        plot_means.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in plot_means.columns]
    
    # Plotar
    sns.set_style("whitegrid")
    plot_means.set_index('metric').plot(kind='bar', ax=ax1)
    ax1.set_title('Comparação de Médias por Métrica')
    ax1.set_ylabel('Valor Médio')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    
    # Subplot 2: Diferenças percentuais
    ax2 = axes[0, 1]
    plot_data['abs_percent_diff'] = plot_data['percent_difference'].abs()
    significant = plot_data[plot_data['statistically_significant']]
    not_significant = plot_data[~plot_data['statistically_significant']]
    
    # Plotar barras para diferenças significativas
    if not significant.empty:
        sns.barplot(
            data=significant,
            x='metric',
            y='percent_difference',
            hue='tenant',
            ax=ax2,
            alpha=0.8
        )
    
    # Adicionar marcadores para diferenças não significativas
    if not not_significant.empty:
        for _, row in not_significant.iterrows():
            ax2.scatter(
                row['metric'], row['percent_difference'],
                marker='o', s=100, color='gray', alpha=0.5,
                label='Não Significativo' if 'Não Significativo' not in ax2.get_legend_handles_labels()[1] else ""
            )
    
    ax2.set_title('Diferença Percentual (%)')
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Subplot 3: Heatmap de diferenças percentuais
    ax3 = axes[1, 0]
    
    if 'tenant' in plot_data.columns and len(plot_data['tenant'].unique()) > 1:
        # Criar matriz para o heatmap
        heatmap_data = plot_data.pivot_table(
            index='metric',
            columns='tenant',
            values='percent_difference'
        )
        
        # Criar heatmap com escala divergente
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
    else:
        ax3.text(0.5, 0.5, 'Dados insuficientes para heatmap\n(necessário múltiplos tenants)', 
                horizontalalignment='center', verticalalignment='center')
        ax3.set_title('Heatmap de Diferenças (indisponível)')
    
    # Subplot 4: Gráfico de dispersão p-value vs diferença
    ax4 = axes[1, 1]
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
    
    # Adicionar textos explicativos
    ax4.text(0.01, ax4.get_ylim()[1]*0.9, 'Estatisticamente\nSignificativo', fontsize=9)
    ax4.text(0.1, ax4.get_ylim()[1]*0.9, 'Não\nSignificativo', fontsize=9)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)  # Ajustar para título principal
    
    return fig

def compare_technologies(exp1_data, exp2_data, metrics_list=None, tenants_list=None,
                        exp1_name="Tecnologia 1", exp2_name="Tecnologia 2",
                        output_dir=None, generate_plots=True):
    """
    Função principal para comparação entre experimentos com diferentes tecnologias.
    
    Args:
        exp1_data (dict): Dados do primeiro experimento
        exp2_data (dict): Dados do segundo experimento  
        metrics_list (list): Lista de métricas para análise
        tenants_list (list): Lista de tenants para filtrar
        exp1_name (str): Nome da primeira tecnologia
        exp2_name (str): Nome da segunda tecnologia
        output_dir (str): Diretório para salvar resultados
        generate_plots (bool): Se True, gera visualizações
        
    Returns:
        dict: Resultados da comparação
    """
    results = {}
    
    # 1. Normalizar métricas
    exp1_norm, exp2_norm, norm_stats = normalize_metrics_between_experiments(
        exp1_data, exp2_data, metrics_list
    )
    
    results['normalized_data'] = {
        exp1_name: exp1_norm,
        exp2_name: exp2_norm
    }
    results['normalization_stats'] = norm_stats
    
    # 2. Calcular métricas de eficiência para diferentes fases
    phase_names = ['1 - Baseline', '2 - Attack', '3 - Recovery']
    
    all_metrics = calculate_relative_efficiency(
        exp1_data, exp2_data, metrics_list, tenants_list,
        None, exp1_name, exp2_name
    )
    
    phase_metrics = {}
    for phase in phase_names:
        phase_metrics[phase] = calculate_relative_efficiency(
            exp1_data, exp2_data, metrics_list, tenants_list,
            phase, exp1_name, exp2_name
        )
    
    results['efficiency_metrics'] = {
        'all_phases': all_metrics,
        'by_phase': phase_metrics
    }
    
    # 3. Gerar plots se solicitado
    if generate_plots and output_dir:
        plots_dir = os.path.join(output_dir, 'technology_comparison')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot geral
        fig_all = plot_experiment_comparison(
            all_metrics, exp1_name, exp2_name,
            figsize=(14, 10)
        )
        
        if fig_all:
            fig_all.savefig(os.path.join(plots_dir, f'comparison_{exp1_name}_vs_{exp2_name}_all.png'))
            plt.close(fig_all)
        
        # Plots por fase
        for phase, phase_data in phase_metrics.items():
            phase_label = phase.replace(' ', '_').lower()
            
            fig_phase = plot_experiment_comparison(
                phase_data, exp1_name, exp2_name,
                figsize=(14, 10)
            )
            
            if fig_phase:
                fig_phase.savefig(os.path.join(plots_dir, 
                                f'comparison_{exp1_name}_vs_{exp2_name}_{phase_label}.png'))
                plt.close(fig_phase)
        
        # Plots por métrica individual
        for metric in metrics_list:
            if metric not in exp1_data or metric not in exp2_data:
                continue
                
            fig_metric = plot_experiment_comparison(
                all_metrics, exp1_name, exp2_name,
                metric_filter=metric,
                figsize=(14, 10)
            )
            
            if fig_metric:
                fig_metric.savefig(os.path.join(plots_dir, 
                                 f'comparison_{exp1_name}_vs_{exp2_name}_{metric}.png'))
                plt.close(fig_metric)
        
        results['plot_paths'] = {
            'all': os.path.join(plots_dir, f'comparison_{exp1_name}_vs_{exp2_name}_all.png'),
            'by_phase': {phase: os.path.join(plots_dir, 
                                         f'comparison_{exp1_name}_vs_{exp2_name}_{phase.replace(" ", "_").lower()}.png') 
                         for phase in phase_names},
            'by_metric': {metric: os.path.join(plots_dir, 
                                           f'comparison_{exp1_name}_vs_{exp2_name}_{metric}.png')
                        for metric in metrics_list if metric in exp1_data and metric in exp2_data}
        }
    
    return results
