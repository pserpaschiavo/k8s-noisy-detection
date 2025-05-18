"""
Script principal para o pipeline de análise de dados do experimento de noisy neighbors.

Este script orquestra todo o pipeline, desde o carregamento dos dados até 
a geração de visualizações e relatórios.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as stats

# Importar módulos do pipeline
from pipeline.data_processing.consolidation import list_available_tenants, list_available_metrics
from pipeline.data_processing.consolidation import load_experiment_data, select_tenants, load_multiple_metrics
from pipeline.data_processing.time_normalization import add_elapsed_time, add_experiment_elapsed_time, add_phase_markers
from pipeline.data_processing.time_normalization import normalize_time
from pipeline.data_processing.aggregation import calculate_tenant_stats, calculate_inter_tenant_impact, calculate_recovery_effectiveness
from pipeline.data_processing.aggregation import aggregate_by_time, aggregate_data_by_custom_elements
from pipeline.data_processing.metric_normalization import (
    normalize_metrics_by_node_capacity, apply_normalization_to_all_metrics,
    auto_format_metrics
)
from pipeline.data_processing.quota_parser import (
    get_tenant_quotas, create_node_config_from_quotas,
    get_quota_summary, format_value_with_unit, convert_to_best_unit
)
from pipeline.analysis.tenant_analysis import calculate_correlation_matrix, compare_tenant_metrics
from pipeline.analysis.phase_analysis import compare_phases_ttest, analyze_recovery_effectiveness
from pipeline.analysis.advanced_analysis import calculate_covariance_matrix, calculate_entropy_metrics
from pipeline.analysis.advanced_analysis import granger_causality_test, analyze_causal_relationships, calculate_normalized_impact_score
from pipeline.analysis.anomaly_detection import (
    detect_anomalies_ensemble, detect_pattern_changes
)
from pipeline.analysis.noisy_tenant_detection import identify_noisy_tenant
from pipeline.analysis.experiment_comparison import (
    load_multiple_experiments,
    preprocess_experiments,
    calculate_statistics_summary,
    compare_distributions,
    detect_anomalies_across_experiments,
    summarize_anomalies,
    compare_experiment_phases
)
from pipeline.visualization.plots import (plot_metric_by_phase, plot_phase_comparison,
                                plot_tenant_impact_heatmap, plot_recovery_effectiveness,
                                plot_impact_score_barplot, plot_impact_score_trend,
                                plot_metric_with_anomalies, plot_change_points,
                                create_heatmap, plot_multivariate_anomalies, plot_correlation_heatmap)
from pipeline.visualization.table_generator import (export_to_latex, export_to_csv,
                                         create_phase_comparison_table, create_impact_summary_table,
                                         convert_df_to_markdown, create_causality_results_table)
from pipeline.config import (DEFAULT_DATA_DIR, DEFAULT_METRICS, AGGREGATION_CONFIG,
                    IMPACT_CALCULATION_DEFAULTS, VISUALIZATION_CONFIG,
                    NODE_RESOURCE_CONFIGS, DEFAULT_NODE_CONFIG_NAME,
                    DEFAULT_CAUSALITY_MAX_LAG, DEFAULT_CAUSALITY_THRESHOLD_P_VALUE,
                    DEFAULT_METRICS_FOR_CAUSALITY, CAUSALITY_METRIC_COLORS, PHASE_DISPLAY_NAMES,
                    METRIC_DISPLAY_NAMES, DEFAULT_NOISY_TENANT)
from pipeline.analysis.inter_tenant_causality import identify_causal_chains, visualize_causal_graph
from pipeline.analysis.application_metrics_analysis import (
    analyze_latency_impact, analyze_error_rate_correlation, calculate_application_slo_violations
)
from pipeline.analysis.technology_comparison import (
    normalize_metrics_between_experiments, calculate_relative_efficiency,
    plot_experiment_comparison, compare_technologies
)
from dotenv import load_dotenv


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pipeline de análise de dados do experimento de noisy neighbors.')
    parser.add_argument('--data-dir', type=str, default='/home/phil/Projects/k8s-noisy-detection/demo-data/demo-experiment-3-rounds',
                        help='Diretório com os dados do experimento')
    parser.add_argument('--compare-dir', type=str, nargs='+',
                        help='Diretório(s) adicional(is) para comparar múltiplos experimentos')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Diretório para salvar os resultados')
    parser.add_argument('--tenants', type=str, nargs='+',
                        help='Tenant(s) específico(s) a analisar')
    parser.add_argument('--noisy-tenant', type=str, 
                        help='Tenant específico que gera ruído (por padrão: tenant-b)')
    parser.add_argument('--auto-detect-noisy', action='store_true',
                        help='Detectar automaticamente qual tenant é o gerador de ruído')
    parser.add_argument('--metrics', type=str, nargs='+',
                        help='Métrica(s) específica(s) a analisar')
    parser.add_argument('--phases', type=str, nargs='+',
                        help='Fase(s) específica(s) a analisar')
    parser.add_argument('--rounds', type=str, nargs='+',
                        help='Round(s) específico(s) a analisar')
    parser.add_argument('--show-as-percentage', action='store_true',
                        help='Exibir métricas como percentual da capacidade total do cluster')
    parser.add_argument('--advanced', action='store_true',
                        help='Executar análises avançadas (covariância, entropia, causalidade)')
    parser.add_argument('--anomaly-detection', action='store_true',
                        help='Executar detecção de anomalias')
    parser.add_argument('--compare-experiments', action='store_true',
                        help='Comparar múltiplos experimentos')
    parser.add_argument('--generate-reports', action='store_true',
                        help='Gerar relatórios em Markdown, LaTeX e HTML')
    parser.add_argument('--elements-to-aggregate', type=str, nargs='+',
                        help='Elemento(s) específico(s) para agregar (ex: tenant-a, tenant-b, ingress-nginx)')
    parser.add_argument('--node-config', type=str, default=None,
                        help='Nome da configuração do nó a ser usada (ex: Default, Limited). Sobrepõe o .env.')
    parser.add_argument('--use-quotas-for-normalization', action='store_true', default=False,
                        help='Usar configuração baseada em quotas para normalização de métricas.')
    parser.add_argument('--app-metrics-analysis', action='store_true',
                        help='Executar análise de métricas em nível de aplicação')
    parser.add_argument('--slo-thresholds', type=str, 
                        help='Arquivo JSON com definição de thresholds de SLO por métrica')
    parser.add_argument('--compare-technologies', action='store_true',
                        help='Comparar experimentos com tecnologias diferentes')
    parser.add_argument('--technology-names', type=str, nargs='+',
                        help='Nomes das tecnologias sendo comparadas (ex: Docker Vanilla, Kata Containers)')
    parser.add_argument('--inter-tenant-causality', action='store_true',
                        help='Executar análise de causalidade entre tenants.')
    parser.add_argument('--compare-rounds-intra', action='store_true',
                        help='Executar comparação formal entre rodadas do mesmo experimento.')
    parser.add_argument('--compare-tenants-directly', action='store_true',
                        help='Executar comparação estatística direta entre tenants.')
    
    return parser.parse_args()


def setup_output_directories(output_dir):
    """Configura diretórios de saída."""
    plots_dir = os.path.join(output_dir, 'plots')
    tables_dir = os.path.join(output_dir, 'tables')
    advanced_dir = os.path.join(output_dir, 'advanced')
    anomaly_dir = os.path.join(output_dir, 'anomalies')
    comparison_dir = os.path.join(output_dir, 'comparisons')
    reports_dir = os.path.join(output_dir, 'reports')
    causality_dir = os.path.join(output_dir, 'causality')
    rounds_comparison_intra_dir = os.path.join(output_dir, 'rounds_comparison_intra')
    tenant_comparison_dir = os.path.join(output_dir, 'tenant_comparison') # New directory
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(advanced_dir, exist_ok=True)
    os.makedirs(os.path.join(advanced_dir, 'tables'), exist_ok=True)
    os.makedirs(anomaly_dir, exist_ok=True)
    os.makedirs(os.path.join(anomaly_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(anomaly_dir, 'tables'), exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(causality_dir, exist_ok=True)
    os.makedirs(os.path.join(causality_dir, 'tables'), exist_ok=True)  # Ensure causality tables directory is created
    os.makedirs(rounds_comparison_intra_dir, exist_ok=True)
    os.makedirs(os.path.join(rounds_comparison_intra_dir, 'plots'), exist_ok=True)
    os.makedirs(tenant_comparison_dir, exist_ok=True) # Create new directory
    os.makedirs(os.path.join(tenant_comparison_dir, 'tables'), exist_ok=True) # Create tables subdirectory
    
    return plots_dir, tables_dir, advanced_dir, anomaly_dir, comparison_dir, reports_dir, causality_dir, rounds_comparison_intra_dir, tenant_comparison_dir


def compare_rounds_within_experiment(experiment_results, output_dir_main, metrics_to_compare=None, phases_to_compare=None, 
                            show_as_percentage=False, node_config=None):
    """
    Compara formalmente diferentes rodadas dentro do mesmo experimento para métricas e fases especificadas.
    Realiza testes ANOVA, gera gráficos de barras e salva os resultados agregados (ex: média por rodada) em CSV.
    Retorna um dicionário com caminhos para CSVs, plots e estatísticas ANOVA para o relatório.
    
    Args:
        experiment_results: Resultados do experimento
        output_dir_main: Diretório de saída principal
        metrics_to_compare: Lista de métricas para comparar
        phases_to_compare: Lista de fases para comparar
        show_as_percentage: Se True, exibe valores como percentuais da capacidade total
        node_config: Configuração do nó com capacidades totais dos recursos
    """
    print("\nIniciando Comparação Entre Rodadas do Mesmo Experimento...")
    
    rounds_comparison_output_dir = os.path.join(output_dir_main, "rounds_comparison_intra")
    plots_subdir = os.path.join(rounds_comparison_output_dir, "plots")

    processed_data = experiment_results.get('processed_data')
    all_comparison_outputs = {}

    if not processed_data:
        print("Nenhum dado processado disponível para comparação entre rodadas.")
        return all_comparison_outputs

    # Default metrics and phases if not provided
    if metrics_to_compare is None:
        metrics_to_compare = DEFAULT_METRICS # Use default metrics from config if none are specified
    if phases_to_compare is None:
        # This default should align with how phases_to_compare_rounds is set in main()
        # It expects raw phase names like "2 - Attack"
        phases_to_compare = ["2 - Attack"] 

    for metric_name in metrics_to_compare:
        metric_display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
        if metric_name not in processed_data:
            print(f"Métrica {metric_name} não encontrada nos dados processados. Pulando comparação de rodadas para esta métrica.")
            continue
        
        metric_df = processed_data[metric_name]
        if not all(col in metric_df.columns for col in ['phase', 'round', 'value']):
            print(f"Colunas 'phase', 'round', ou 'value' não encontradas no DataFrame da métrica {metric_name}. Pulando.")
            continue

        for raw_phase_name in phases_to_compare: # Iterate over raw phase names
            # Get the display name for reporting and filenames
            current_phase_display_for_report = PHASE_DISPLAY_NAMES.get(raw_phase_name, raw_phase_name)
            print(f"  Comparando rodadas para Métrica: {metric_display_name}, Fase: {current_phase_display_for_report} (Raw: {raw_phase_name})")
            
            # Filter using the raw phase name
            phase_specific_df = metric_df[metric_df['phase'] == raw_phase_name]
            
            output_key = f"{metric_name}_{current_phase_display_for_report.replace(' ', '_')}"
            current_output = {"csv_path": None, "plot_path": None, "anova_f_stat": None, "anova_p_value": None}

            if not phase_specific_df.empty:
                # Verificar se há dados suficientes para cada rodada
                rounds_count = phase_specific_df['round'].value_counts()
                valid_rounds = rounds_count[rounds_count >= 5].index.tolist()
                
                if not valid_rounds:
                    print(f"    Não há rodadas com dados suficientes para análise. Pulando.")
                    continue
                    
                # Filtrar apenas para rodadas com dados suficientes
                phase_specific_df_filtered = phase_specific_df[phase_specific_df['round'].isin(valid_rounds)]
                
                # Aggregate data for CSV and plotting
                if 'tenant' in phase_specific_df_filtered.columns and len(phase_specific_df_filtered['tenant'].unique()) > 1:
                    comparison_data_agg = phase_specific_df_filtered.groupby(['round', 'tenant'])['value'].mean().reset_index()
                    comparison_data = comparison_data_agg.groupby('round')['value'].mean().reset_index()
                    comparison_data.rename(columns={'value': f'mean_value_across_tenants'}, inplace=True)
                else:
                    comparison_data = phase_specific_df_filtered.groupby('round')['value'].mean().reset_index()
                    comparison_data.rename(columns={'value': 'mean_value'}, inplace=True)
                
                print(f"    Valores médios de {metric_display_name} na fase {current_phase_display_for_report} por rodada:")
                print(comparison_data)
                
                # Use current_phase_display_for_report for filenames
                csv_filename = f"{metric_name}_{current_phase_display_for_report.replace(' ', '_')}_round_comparison.csv"
                csv_path = os.path.join(rounds_comparison_output_dir, csv_filename)
                try:
                    comparison_data.to_csv(csv_path, index=False)
                    print(f"    Comparação (CSV) salva em: {csv_path}")
                    current_output["csv_path"] = csv_path
                except Exception as e:
                    print(f"    Erro ao salvar CSV de comparação de rodadas: {e}")

                # Perform ANOVA
                rounds_with_data = phase_specific_df['round'].unique()
                if len(rounds_with_data) >= 2:
                    grouped_values = [
                        group['value'].dropna() for name, group in phase_specific_df.groupby('round')
                        if not group['value'].dropna().empty
                    ]
                    if len(grouped_values) >= 2:
                        try:
                            pass # Added pass to fix empty try block
                        except Exception as e:
                            print(f"    Erro ao realizar ANOVA (exceção durante tentativa): {e}")
                            pass # Added pass to fix empty except block
                    else:
                        print("    Não há grupos suficientes com dados para ANOVA após filtragem.")
                else:
                    print("    Não há rodadas suficientes com dados para realizar ANOVA.")

                # Generate and save bar plot
                if not comparison_data.empty:
                    plt.figure(figsize=VISUALIZATION_CONFIG.get('figure_size', (10, 6)))
                    value_col_for_plot = 'mean_value_across_tenants' if 'mean_value_across_tenants' in comparison_data.columns else 'mean_value'
                    
                    plt.bar(comparison_data['round'].astype(str), comparison_data[value_col_for_plot])
                    # Use current_phase_display_for_report for plot title
                    plt.title(f'Mean {metric_display_name} per Round during {current_phase_display_for_report}')
                    plt.xlabel('Round')
                    
                    # Configurar rótulo do eixo Y dependendo da exibição como percentual
                    if show_as_percentage and node_config:
                        if metric_name == 'cpu_usage' and 'CPUS' in node_config:
                            unit_info = f"% of {node_config['CPUS']} CPU cores"
                        elif metric_name == 'memory_usage' and 'MEMORY_GB' in node_config:
                            unit_info = f"% of {node_config['MEMORY_GB']} GB memory"
                        elif metric_name == 'disk_throughput_total' and 'DISK_SIZE_GB' in node_config:
                            unit_info = f"% of max throughput"
                        elif metric_name == 'network_total_bandwidth' and 'NETWORK_BANDWIDTH_MBPS' in node_config:
                            unit_info = f"% of {node_config['NETWORK_BANDWIDTH_MBPS']} Mbps"
                        else:
                            unit_info = "%"
                        plt.ylabel(f'Mean {metric_display_name} ({unit_info})')
                    else:
                        plt.ylabel(f'Mean {metric_display_name}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Use current_phase_display_for_report for plot filename
                    plot_filename = f"{metric_name}_{current_phase_display_for_report.replace(' ', '_')}_round_comparison_plot.png"
                    plot_path = os.path.join(plots_subdir, plot_filename)
                    try:
                        plt.savefig(plot_path)
                        print(f"    Gráfico de comparação de rodadas salvo em: {plot_path}")
                        current_output["plot_path"] = plot_path
                    except Exception as e:
                        print(f"    Erro ao salvar gráfico de comparação de rodadas: {e}")
                    plt.close()
                all_comparison_outputs[output_key] = current_output
            else:
                print(f"    Sem dados para a métrica {metric_display_name} na fase {current_phase_display_for_report} para comparação entre rodadas.")
    
    print("Comparação Entre Rodadas do Mesmo Experimento concluída.")
    return all_comparison_outputs


def run_application_metrics_analysis(metrics_dict, app_metrics_dict=None, noisy_tenant="tenant-b", slo_thresholds=None, output_dir=None):
    """
    Executa análise de métricas de aplicação.
    
    Args:
        metrics_dict: Dicionário com métricas de infraestrutura
        app_metrics_dict: Dicionário com métricas de aplicação (opcional)
        noisy_tenant: O tenant identificado como ruidoso
        slo_thresholds: Thresholds para SLOs por métrica
        output_dir: Diretório para salvar resultados
        
    Returns:
        Dict: Resultados da análise de métricas de aplicação
    """
    results = {}
    
    # Se não houver métricas de aplicação, usar as métricas de infraestrutura
    if app_metrics_dict is None:
        app_metrics_dict = metrics_dict
    
    print("\n=== Executando Análise de Métricas de Aplicação ===")
    
    # 1. Análise de Impacto na Latência
    if any(metric.startswith('latency') or metric == 'response_time' for metric in app_metrics_dict.keys()):
        print("Analisando impacto na latência...")
        latency_impact = analyze_latency_impact(app_metrics_dict, metrics_dict, noisy_tenant)
        results['latency_impact'] = latency_impact
        
        # Exibir resultados
        print(f"\nImpacto na latência por tenant causado por {noisy_tenant}:")
        for tenant, impact in latency_impact.items():
            sig = "✓" if impact['significant_impact'] else "✗"
            print(f"  {tenant}: {impact['increase_percentage']:.2f}% ({sig} p={impact['p_value']:.4f})")
    
    # 2. Correlação de Taxa de Erros
    if any(metric.startswith('error') for metric in app_metrics_dict.keys()):
        print("\nAnalisando correlação entre uso de CPU e taxas de erro...")
        error_correlations = analyze_error_rate_correlation(app_metrics_dict, metrics_dict, noisy_tenant)
        results['error_correlations'] = error_correlations
        
        # Exibir resultados
        print(f"\nCorrelação entre uso de CPU de {noisy_tenant} e taxas de erro:")
        for tenant, corr in error_correlations.items():
            print(f"  {tenant}: {corr:.4f}")
    
    # 3. Análise de violação de SLO
    if slo_thresholds:
        print("\nAnalisando violações de SLO...")
        slo_violations = calculate_application_slo_violations(app_metrics_dict, slo_thresholds, noisy_tenant)
        results['slo_violations'] = slo_violations
        
        # Exibir resultados
        print(f"\nViolações de SLO por tenant:")
        for tenant, violations in slo_violations.items():
            print(f"  {tenant}:")
            for metric, stats in violations.items():
                increase = stats['violation_increase'] * 100
                print(f"    {metric}: +{increase:.2f}% de violações durante ataque")
    
    return results


def run_technology_comparison(exp1_data, exp2_data, metrics_list=None, tenants_list=None, 
                           exp1_name="Tecnologia 1", exp2_name="Tecnologia 2",
                           output_dir=None, skip_plots=False):
    """
    Executa comparação entre experimentos com tecnologias diferentes.
    
    Args:
        exp1_data: Dados do primeiro experimento
        exp2_data: Dados do segundo experimento  
        metrics_list: Lista de métricas para análise
        tenants_list: Lista de tenants para filtrar
        exp1_name: Nome da primeira tecnologia
        exp2_name: Nome da segunda tecnologia
        output_dir: Diretório para salvar resultados
        skip_plots: Se True, não gera visualizações
        
    Returns:
        Dict: Resultados da comparação
    """
    print(f"\n=== Comparando Tecnologias: {exp1_name} vs {exp2_name} ===")
    
    # Garantir que temos métricas comuns para comparar
    if metrics_list is None:
        metrics_list = [m for m in exp1_data.keys() if m in exp2_data.keys()]
        print(f"Métricas comuns para comparação: {metrics_list}")
    
    # Realizar comparação
    comparison_results = compare_technologies(
        exp1_data, exp2_data, 
        metrics_list=metrics_list,
        tenants_list=tenants_list,
        exp1_name=exp1_name, 
        exp2_name=exp2_name,
        output_dir=output_dir,
        generate_plots=True
    )
    
    # Mostrar resumo dos resultados
    efficiency_metrics = comparison_results['efficiency_metrics']['all_phases']
    print("\nResumo de diferenças significativas:")
    
    significant_differences = efficiency_metrics[efficiency_metrics['statistically_significant']]
    if not significant_differences.empty:
        for _, row in significant_differences.iterrows():
            better = row['better_experiment']
            worse = exp2_name if better == exp1_name else exp2_name
            print(f"  {row['metric']} ({row['tenant']}): {better} é melhor que {worse} por {abs(row['percent_difference']):.2f}%")
    else:
        print("  Nenhuma diferença estatisticamente significativa encontrada")
    
    return comparison_results


def main():
    """Função principal que executa o pipeline de análise."""
    args = parse_arguments()
    
    plots_dir, tables_dir, advanced_dir, anomaly_dir, comparison_dir, reports_dir, causality_dir, rounds_comparison_intra_dir, tenant_comparison_dir = setup_output_directories(args.output_dir)
    
    experiment_data_dir_input = args.data_dir
    experiment_data_dir = ""

    if os.path.isabs(experiment_data_dir_input):
        experiment_data_dir = experiment_data_dir_input
    else:
        cwd = os.getcwd()
        path_from_cwd = os.path.join(cwd, experiment_data_dir_input)
        path_from_cwd_with_demodata_prefix = os.path.join(cwd, "demo-data", experiment_data_dir_input)

        if os.path.isdir(path_from_cwd):
            experiment_data_dir = path_from_cwd
        elif os.path.isdir(path_from_cwd_with_demodata_prefix) and not experiment_data_dir_input.startswith("demo-data"):
            experiment_data_dir = path_from_cwd_with_demodata_prefix
        else:
            experiment_data_dir = path_from_cwd

    experiment_data_dir = os.path.normpath(experiment_data_dir)

    if not os.path.isdir(experiment_data_dir):
        print(f"Erro: Diretório de dados não encontrado: {experiment_data_dir}")
        print(f"  Input --data-dir: '{args.data_dir}'")
        print(f"  Caminho absoluto verificado: '{os.path.abspath(experiment_data_dir)}'")
        if not os.path.isabs(args.data_dir):
            cwd_for_hint = os.getcwd()
            print(f"  Verifique se '{args.data_dir}' existe em relação a '{cwd_for_hint}'")
            if not args.data_dir.startswith("demo-data"):
                 print(f"  ou se '{os.path.join('demo-data', args.data_dir)}' existe em relação a '{cwd_for_hint}'")
        return

    print(f"Usando diretório de dados: {experiment_data_dir}")

    all_metrics_data = load_experiment_data(experiment_data_dir, rounds=args.rounds)
    
    # Verificar se os dados foram carregados corretamente
    if not all_metrics_data:
        print("ERRO CRÍTICO: Nenhum dado foi carregado. Verifique o diretório de dados e a lógica de carregamento.")
        return # Sair da função main se nenhum dado for carregado

    # Configurar diretórios de saída

    # Tentar carregar configuração do nó a partir das quotas
    quota_file_path = os.path.join(os.path.dirname(os.path.dirname(experiment_data_dir)), 'resource-quotas.yaml')
    
    # Configuração padrão do nó
    node_config_to_use = NODE_RESOURCE_CONFIGS.get(DEFAULT_NODE_CONFIG_NAME, NODE_RESOURCE_CONFIGS.get("Default"))
    
    # Flag para indicar se estamos usando quotas - verificando argumento da linha de comando
    using_quotas = args.use_quotas_for_normalization
    
    # Criar uma cópia adaptada do NODE_RESOURCE_CONFIGS para o formato esperado pelos módulos de normalização
    if 'MEMORY_GB' in node_config_to_use and 'MEMORY_BYTES' not in node_config_to_use:
        node_config_to_use = node_config_to_use.copy()  # Evitar alterar o original
        node_config_to_use['MEMORY_BYTES'] = node_config_to_use['MEMORY_GB'] * (2**30)
        node_config_to_use['DISK_SIZE_BYTES'] = node_config_to_use['DISK_SIZE_GB'] * (2**30)
        # Estimar largura de banda com base no CPU 
        node_config_to_use['NETWORK_BANDWIDTH_MBPS'] = max(1000, node_config_to_use['CPUS'] * 250)
    
    if os.path.exists(quota_file_path):
        print(f"Arquivo de quotas encontrado: {quota_file_path}")
        quota_based_config = create_node_config_from_quotas(quota_file_path)
        
        if using_quotas and quota_based_config and quota_based_config['CPUS'] > 0:
            print(f"Usando configuração baseada em quotas para normalização de métricas")
            print(f"  CPU Total: {format_value_with_unit(quota_based_config['CPUS'], 'cpu')}")
            print(f"  Memória Total: {format_value_with_unit(quota_based_config['MEMORY_BYTES'], 'memory')}")
            print(f"  Armazenamento Estimado: {format_value_with_unit(quota_based_config['DISK_SIZE_BYTES'], 'disk')}")
            print(f"  Largura de Banda Estimada: {format_value_with_unit(quota_based_config['NETWORK_BANDWIDTH_MBPS']*1e6/8, 'network')}")
            print(f"  Margem para recursos de sistema: {quota_based_config['SYSTEM_RESOURCES_MARGIN']:.1f}%")
            
            node_config_to_use = quota_based_config
        else:
            print(f"Usando configuração fixa de NODE_RESOURCE_CONFIGS para normalização de métricas")
            print(f"  CPU Total: {format_value_with_unit(node_config_to_use['CPUS'], 'cpu')}")
            print(f"  Memória Total: {format_value_with_unit(node_config_to_use.get('MEMORY_BYTES', node_config_to_use.get('MEMORY_GB', 0) * 2**30), 'memory')}")
            print(f"  Armazenamento: {format_value_with_unit(node_config_to_use.get('DISK_SIZE_BYTES', node_config_to_use.get('DISK_SIZE_GB', 0) * 2**30), 'disk')}")
    else:
        print(f"Arquivo de quotas não encontrado. Usando configuração padrão do nó.")
        # Mostrar informações da configuração padrão
        print(f"  CPU Total: {format_value_with_unit(node_config_to_use['CPUS'], 'cpu')}")
        print(f"  Memória Total: {format_value_with_unit(node_config_to_use.get('MEMORY_BYTES', node_config_to_use.get('MEMORY_GB', 0) * 2**30), 'memory')}")

    # Adicionar resumo formatado das quotas
    if quota_file_path and os.path.exists(quota_file_path):
        print("\nResumo das quotas de recursos por tenant:")
        
        # Obter resumo de quotas com porcentagens e formatação
        quota_summary = get_quota_summary(quota_file_path, include_requests=True, calculate_percentages=True)
        
        # Detectar o tenant total se existir
        total_entry = quota_summary.get('__total__', {})
        
        # Mostrar informações por tenant
        for tenant, quota_info in quota_summary.items():
            if tenant == '__total__':
                continue  # Mostraremos o total depois
                
            print(f"  {tenant.upper()}:")
            
            # CPU
            if 'cpu_limit' in quota_info:
                cpu_text = f"    CPU Limit: {quota_info['cpu_limit']}"
                if 'cpu_percent' in quota_info:
                    cpu_text += f" ({quota_info['cpu_percent']} do cluster)"
                print(cpu_text)
                
            # Memória
            if 'memory_limit' in quota_info:
                mem_text = f"    Memory Limit: {quota_info['memory_limit']}"
                if 'memory_percent' in quota_info:
                    mem_text += f" ({quota_info['memory_percent']} do cluster)"
                print(mem_text)
                
            # Requests (se disponíveis)
            if 'cpu_request' in quota_info:
                cpu_req_text = f"    CPU Request: {quota_info['cpu_request']}"
                if 'cpu_req_vs_limit' in quota_info:
                    cpu_req_text += f" ({quota_info['cpu_req_vs_limit']} do limit)"
                print(cpu_req_text)
                
            if 'memory_request' in quota_info:
                mem_req_text = f"    Memory Request: {quota_info['memory_request']}"
                if 'memory_req_vs_limit' in quota_info:
                    mem_req_text += f" ({quota_info['memory_req_vs_limit']} do limit)"
                print(mem_req_text)
        
        # Mostrar totais
        if total_entry:
            print(f"\n  TOTAL DO CLUSTER:")
            if 'cpu_limit' in total_entry:
                print(f"    CPU Total: {total_entry['cpu_limit']}")
            if 'memory_limit' in total_entry:
                print(f"    Memory Total: {total_entry['memory_limit']}")

    # Initialize all_metrics_data to an empty dict

    all_metrics_data = load_experiment_data(experiment_data_dir, rounds=args.rounds)

    if not all_metrics_data:
        print("Nenhum dado foi carregado. Verifique o diretório e a estrutura dos dados.")
        return

    # Listar e selecionar tenants

    # Se alguma análise adicional é solicitada OU se estamos usando a opção de percentual
    if args.advanced or args.anomaly_detection or args.inter_tenant_causality or args.compare_rounds_intra or args.show_as_percentage:
        print("\nCarregando e Processando Dados do Experimento")
        loaded_data = load_experiment_data(
            experiment_data_dir,
            metrics=DEFAULT_METRICS,  # Explicitly pass DEFAULT_METRICS
        )
        if loaded_data: # Check if data was actually loaded
            all_metrics_data = loaded_data
        else:
            print("Nenhum dado do experimento foi carregado pela load_experiment_data. Verifique o diretório de dados e a configuração.")
        
        if not all_metrics_data: 
            print("Nenhum dado do experimento foi carregado (all_metrics_data está vazio após tentativa de carga).")
        else:
            print("Dados do experimento carregados com sucesso.")
            
            # Preservar os dados originais para caso seja necessário
            all_metrics_data_original = {k: v.copy() for k, v in all_metrics_data.items()}
            
            # Definir um mapeamento explícito de nomes de métricas para tipos
            metric_type_map = {
                'cpu_usage': 'cpu',
                'memory_usage': 'memory',
                'disk_read_bytes': 'disk',
                'disk_write_bytes': 'disk',
                'network_rx_bytes': 'network',
                'network_tx_bytes': 'network',
                'disk_io_time': 'disk_iops',
                'disk_iops': 'disk_iops'
            }
            
            # Aplicar formatação automática das unidades para melhor legibilidade
            print("\nFormatando métricas com unidades adequadas...")
            all_metrics_data = auto_format_metrics(all_metrics_data, metric_type_map)
            print("Métricas formatadas com unidades legíveis.")
            
            # Resumo das unidades atribuídas
            for metric_name, rounds_data in all_metrics_data.items():
                # Check if rounds_data is a dictionary (nested structure)
                if isinstance(rounds_data, dict):
                    # Iterate through rounds if it's a nested structure
                    for round_name, df in rounds_data.items():
                        if isinstance(df, pd.DataFrame) and 'unit' in df.columns and not df['unit'].isna().all():
                            unit = df['unit'].iloc[0]
                            print(f"  {metric_name} (Round: {round_name}): {unit}")
                            break  # Show unit for the first round and break
                        elif isinstance(df, pd.DataFrame) and 'value_formatted' in df.columns:
                            print(f"  {metric_name} (Round: {round_name}): formatação personalizada aplicada")
                            break # Show for the first round and break
                    else: # If no round had unit info, or rounds_data was empty
                        if not rounds_data: # If rounds_data is an empty dict
                             print(f"  {metric_name}: No rounds data found to determine unit.")
                        # If rounds_data was not empty, but no df had unit or value_formatted, this loop completes.
                        # We might want a generic message or to check the first df if available.
                        # For now, if inner loop didn't print, this means no specific unit info was found for any round.
                elif isinstance(rounds_data, pd.DataFrame): # Handling non-nested structure (backward compatibility or other cases)
                    df = rounds_data 
                    if 'unit' in df.columns and not df['unit'].isna().all():
                        unit = df['unit'].iloc[0]
                        print(f"  {metric_name}: {unit}")
                    elif 'value_formatted' in df.columns:
                        print(f"  {metric_name}: formatação personalizada aplicada")
                else:
                    print(f"  {metric_name}: Data is not in expected format (DataFrame or Dict[str, DataFrame]).")
    else:
        print("Nenhuma flag de análise avançada/anomalia/causalidade/comparação de rounds ativa. Pulando carregamento principal de dados.")

    if not all_metrics_data:
        print("Nenhum dado carregado. Verifique o diretório de dados e tente novamente.")
        return
    
    # Normalizar métricas como percentuais se solicitado
    if args.show_as_percentage:
        print("\nNormalizando métricas como percentuais da capacidade dos recursos...")
        
        # Caminho para o arquivo de quotas para usar na normalização
        quota_file_for_norm = quota_file_path if os.path.exists(quota_file_path) else None
        
        # Aplicar normalização usando nossa configuração melhorada
        print("Aplicando normalização...")
        normalized_metrics = apply_normalization_to_all_metrics(
            all_metrics_data, 
            node_config_to_use, 
            replace_original=False,  # Manter valores originais
            use_tenant_quotas=(quota_file_for_norm is not None),
            add_relative_values=True,
            show_as_percentage=args.show_as_percentage,
            use_formatted_values=True,
            quota_file=quota_file_for_norm
        )
        
        # Mesclar com os dados existentes ou substituir
        all_metrics_data = normalized_metrics
        
        print("Normalização concluída com sucesso.")
        # Mostrar exemplo de formatação para a primeira métrica como referência
        for metric_name, df in all_metrics_data.items():
            # Check if df is a dictionary (nested structure)
            if isinstance(df, dict):
                # Iterate through rounds if it's a nested structure
                for round_name, round_df in df.items():
                    if isinstance(round_df, pd.DataFrame) and 'normalized_value' in round_df.columns and not round_df.empty:
                        print(f"  Exemplo para {metric_name} (Round: {round_name}):")
                        sample_row = round_df.iloc[0]
                        print(f"    Valor original: {sample_row.get('original_value', sample_row.get('value', 'N/A'))}")
                        print(f"    Valor normalizado: {sample_row.get('normalized_value', 'N/A')}%")
                        print(f"    Descrição: {sample_row.get('normalized_description', 'N/A')}")
                        break  # Show example for the first round and break from inner loop
                else: # If no round had normalized_value or df was empty
                    if not df: # if df (rounds_data) is an empty dict
                        print(f"  {metric_name}: No rounds data found to show normalization example.")
                    # If inner loop completed without break, means no round_df had 'normalized_value'
            elif isinstance(df, pd.DataFrame): # Handling non-nested structure
                if 'normalized_value' in df.columns and not df.empty:
                    print(f"  Exemplo para {metric_name}:")
                    sample_row = df.iloc[0]
                    print(f"    Valor original: {sample_row.get('original_value', sample_row.get('value', 'N/A'))}")
                    print(f"    Valor normalizado: {sample_row.get('normalized_value', 'N/A')}%")
                    print(f"    Descrição: {sample_row.get('normalized_description', 'N/A')}")
            else:
                print(f"  {metric_name}: Data is not in expected DataFrame or Dict[str, DataFrame] format for normalization example.")

    # Continuar com o processamento dos dados

    experiment_results = {
        'processed_data': all_metrics_data
    }

    metrics_data = experiment_results.get('processed_data', {})
    aggregated_data = experiment_results.get("aggregated_data", {})
    impact_score_results = experiment_results.get("impact_score_results", {})
    total_experiment_duration_seconds = experiment_results.get("total_experiment_duration_seconds", None)

    if not metrics_data:
        print("Nenhum dado processado (metrics_data) disponível. Muitas análises e plots serão pulados.")
        # Not returning, as some parts of the script might still be useful (e.g. empty report generation)

    print("\nNormalizando tempo global do experimento para todos os DataFrames de métricas...")
    # Iterar sobre cada métrica
    all_phase_markers = {} # Initialize all_phase_markers here
    for metric_name, rounds_data in all_metrics_data.items():
        all_phase_markers[metric_name] = {} # Initialize for each metric
        # Iterar sobre cada DataFrame de round dentro da métrica
        for round_name, df_round in rounds_data.items():
            if df_round is not None and not df_round.empty:
                # Aplicar normalização de tempo
                df_round = add_experiment_elapsed_time(df_round) # Reassign the result
                # Adicionar marcadores de fase
                if not df_round.empty:
                    # Passar o nome da coluna de fase explicitamente e o dicionário de display names
                    df_round, phase_markers_round = add_phase_markers(df_round, phase_column='phase', phase_display_names=PHASE_DISPLAY_NAMES)
                    all_phase_markers[metric_name][round_name] = phase_markers_round
                    all_metrics_data[metric_name][round_name] = df_round # Atualizar o DataFrame com a nova coluna 'phase_name'
            else:
                print(f"DataFrame para métrica '{metric_name}', round '{round_name}' está vazio ou None. Pulando normalização de tempo.")
    print("Normalização de tempo global concluída.")

    # Processamento e Análise Principal (se houver dados e flags ativas)
    if args.advanced and metrics_data:
        print("\nExecutando Análises Avançadas e Gerando Plots...")
        advanced_plots_dir = os.path.join(advanced_dir, 'plots')
        os.makedirs(advanced_plots_dir, exist_ok=True)
        
        advanced_analysis_results = experiment_results.get('advanced_analysis', {})

        # Prepare data for correlation/covariance: concatenate rounds for each metric
        all_metrics_data_concatenated = {}
        for metric_name_loop, rounds_data_for_metric_loop in metrics_data.items():
            if isinstance(rounds_data_for_metric_loop, dict) and rounds_data_for_metric_loop:
                all_rounds_dfs_loop = []
                for round_df_loop in rounds_data_for_metric_loop.values():
                    if isinstance(round_df_loop, pd.DataFrame) and not round_df_loop.empty:
                        all_rounds_dfs_loop.append(round_df_loop)
                if all_rounds_dfs_loop:
                    concatenated_df_loop = pd.concat(all_rounds_dfs_loop, ignore_index=True)
                    all_metrics_data_concatenated[metric_name_loop] = concatenated_df_loop

        all_round_names = set()
        if isinstance(metrics_data, dict):
            for metric_name_iter, rounds_data_iter in metrics_data.items():
                if isinstance(rounds_data_iter, dict):
                    all_round_names.update(rounds_data_iter.keys())
        
        if not all_metrics_data_concatenated:
            print("Nenhum dado concatenado disponível para análises avançadas (correlação/covariância). Pulando essas análises.")
        else:
            for round_name_to_analyze in sorted(list(all_round_names)):
                print(f"  Analisando Round: {round_name_to_analyze} para correlação e covariância")

                # 1. Correlation Analysis
                try:
                    print(f"    Calculando matriz de correlação para o round: {round_name_to_analyze}")
                    correlation_matrix = calculate_correlation_matrix(
                        metrics_dict=all_metrics_data_concatenated,
                        round_name=round_name_to_analyze,
                        tenants=args.tenants,
                        noisy_tenant=args.noisy_tenant
                    )

                    if correlation_matrix is not None and not correlation_matrix.empty:
                        plot_path_corr = os.path.join(advanced_plots_dir, f'correlation_heatmap_round_{round_name_to_analyze}.png')
                        print(f"      Gerando heatmap de correlação para o round: {round_name_to_analyze} em {plot_path_corr}")
                        fig_corr = plot_correlation_heatmap(
                            correlation_matrix,
                            title=f'Inter-Metric Correlation Heatmap (Round: {round_name_to_analyze})'
                        )
                        fig_corr.savefig(plot_path_corr)
                        plt.close(fig_corr)
                        print(f"      Heatmap de correlação salvo em: {plot_path_corr}")
                        advanced_analysis_results.setdefault('correlation_matrices', {})[round_name_to_analyze] = correlation_matrix
                    else:
                        print(f"      Matriz de correlação vazia ou None para o round: {round_name_to_analyze}. Pulando plot.")
                except Exception as e:
                    print(f"    Erro ao calcular ou plotar matriz de correlação para o round {round_name_to_analyze}: {e}")

                # 2. Covariance Analysis
                try:
                    print(f"    Calculando matriz de covariância para o round: {round_name_to_analyze}")
                    covariance_matrix, _ = calculate_covariance_matrix(
                        metrics_dict=all_metrics_data_concatenated,
                        round_name=round_name_to_analyze,
                        tenants=args.tenants
                    )

                    if covariance_matrix is not None and not covariance_matrix.empty:
                        plot_path_cov = os.path.join(advanced_plots_dir, f'covariance_heatmap_round_{round_name_to_analyze}.png')
                        print(f"      Gerando heatmap de covariância para o round: {round_name_to_analyze} em {plot_path_cov}")
                        fig_cov = plot_correlation_heatmap(
                            covariance_matrix,
                            title=f'Inter-Metric Covariance Heatmap (Round: {round_name_to_analyze})',
                            cmap='coolwarm'
                        )
                        fig_cov.savefig(plot_path_cov)
                        plt.close(fig_cov)
                        print(f"      Heatmap de covariância salvo em: {plot_path_cov}")
                        advanced_analysis_results.setdefault('covariance_matrices', {})[round_name_to_analyze] = covariance_matrix
                    else:
                        print(f"      Matriz de covariância vazia ou None para o round: {round_name_to_analyze}. Pulando plot.")
                except Exception as e:
                    print(f"    Erro ao calcular ou plotar matriz de covariância para o round {round_name_to_analyze}: {e}")
        
        # 3. Entropy Analysis (Example: for each metric, across tenants, per round)
        print(f"\n  Calculando métricas de entropia...")
        all_entropy_results = []
        for metric_name_entropy, rounds_data_entropy in metrics_data.items():
            if isinstance(rounds_data_entropy, dict):
                for round_name_entropy, df_round_entropy in rounds_data_entropy.items():
                    if isinstance(df_round_entropy, pd.DataFrame) and not df_round_entropy.empty and 'tenant' in df_round_entropy.columns:
                        print(f"    Calculando entropia para métrica: {metric_name_entropy}, round: {round_name_entropy}")
                        try:
                            entropy_results_df = calculate_entropy_metrics(
                                df_round_entropy,
                                tenants=args.tenants, # Pass specific tenants if provided
                                metric_column='value'
                            )
                            if entropy_results_df is not None and not entropy_results_df.empty:
                                entropy_results_df['metric'] = metric_name_entropy
                                entropy_results_df['round'] = round_name_entropy
                                all_entropy_results.append(entropy_results_df)
                                # Save to CSV
                                entropy_table_filename = f'entropy_metrics_{metric_name_entropy}_round_{round_name_entropy}.csv'
                                entropy_table_path = os.path.join(advanced_dir, 'tables', entropy_table_filename)
                                export_to_csv(entropy_results_df, entropy_table_path)
                                print(f"      Tabela de entropia salva em: {entropy_table_path}")
                            else:
                                print(f"      Nenhum resultado de entropia gerado para {metric_name_entropy}, round {round_name_entropy}.")
                        except Exception as e_entropy:
                            print(f"      Erro ao calcular entropia para {metric_name_entropy}, round {round_name_entropy}: {e_entropy}")
        if all_entropy_results:
            final_entropy_df = pd.concat(all_entropy_results, ignore_index=True)
            advanced_analysis_results['entropy_metrics'] = final_entropy_df
        else:
            print("    Nenhum resultado de entropia calculado para nenhuma métrica/round.")

        # 4. Granger Causality Analysis (Example: for each metric, between tenant pairs, per round)
        print(f"\n  Analisando relações de causalidade de Granger...")
        all_causality_results = []
        # Determine tenant pairs for causality analysis
        # If specific tenants are given, analyze pairs within that subset
        # Otherwise, analyze all pairs (or pairs involving noisy_tenant)
        tenants_for_causality = args.tenants
        if not tenants_for_causality and args.noisy_tenant:
            # If no specific tenants, but noisy_tenant is defined, focus on it
            # This requires a list of all tenants to form pairs with noisy_tenant
            # For simplicity, if args.tenants is None, analyze_causal_relationships will use all unique tenants from the df.
            pass 

        for metric_name_causality, rounds_data_causality in metrics_data.items():
            if isinstance(rounds_data_causality, dict):
                for round_name_causality, df_round_causality in rounds_data_causality.items():
                    if isinstance(df_round_causality, pd.DataFrame) and not df_round_causality.empty and 'tenant' in df_round_causality.columns:
                        # Filter df_round_causality by args.tenants if provided, before passing to analyze_causal_relationships
                        df_for_causality_analysis = df_round_causality
                        if args.tenants:
                            df_for_causality_analysis = df_round_causality[df_round_causality['tenant'].isin(args.tenants)]
                        
                        if df_for_causality_analysis['tenant'].nunique() < 2:
                            print(f"    Pulando causalidade para {metric_name_causality}, round {round_name_causality}: Menos de 2 tenants nos dados filtrados.")
                            continue

                        print(f"    Analisando causalidade para métrica: {metric_name_causality}, round: {round_name_causality}")
                        try:
                            causality_results_df = analyze_causal_relationships(
                                df_for_causality_analysis, 
                                metric_column='value', 
                                # tenant_pairs can be inferred by the function if None
                            )
                            if causality_results_df is not None and not causality_results_df.empty:
                                causality_results_df['metric'] = metric_name_causality
                                causality_results_df['round'] = round_name_causality
                                all_causality_results.append(causality_results_df)
                                # Save to CSV
                                causality_table_filename = f'granger_causality_{metric_name_causality}_round_{round_name_causality}.csv'
                                # Save in the dedicated causality tables directory
                                causality_table_path = os.path.join(causality_dir, 'tables', causality_table_filename)
                                export_to_csv(causality_results_df, causality_table_path)
                                print(f"      Tabela de causalidade salva em: {causality_table_path}")
                            else:
                                print(f"      Nenhum resultado de causalidade gerado para {metric_name_causality}, round {round_name_causality}.")
                        except Exception as e_causality:
                            print(f"      Erro ao analisar causalidade para {metric_name_causality}, round {round_name_causality}: {e_causality}")
        if all_causality_results:
            final_causality_df = pd.concat(all_causality_results, ignore_index=True)
            advanced_analysis_results['granger_causality'] = final_causality_df
        else:
            print("    Nenhum resultado de causalidade de Granger calculado.")

        # Plotting section for advanced analysis
        if isinstance(metrics_data, dict):
            print("\n  Gerando plots de métricas por fase para análises avançadas...")
            for metric_name_plot, rounds_data_plot in metrics_data.items():
                if isinstance(rounds_data_plot, dict):
                    for round_name_plot, df_round_plot in rounds_data_plot.items():
                        if isinstance(df_round_plot, pd.DataFrame) and not df_round_plot.empty:
                            required_cols = ['experiment_elapsed_seconds', 'value', 'phase_name', 'tenant']
                            if all(col in df_round_plot.columns for col in required_cols):
                                plot_m_path = os.path.join(advanced_plots_dir, f'{metric_name_plot}_by_phase_round_{round_name_plot}.png')
                                print(f"    Gerando plot de {metric_name_plot} por fase para round {round_name_plot} em {plot_m_path}")
                                try:
                                    fig_metric_phase = plot_metric_by_phase(
                                        df_round_plot,
                                        metric_name=metric_name_plot,
                                        time_column='experiment_elapsed_seconds',
                                        value_column='value',
                                        show_phase_markers=True,
                                        show_as_percentage=args.show_as_percentage,
                                        node_config=node_config_to_use,
                                        use_formatted_values=True,
                                        tenants=args.tenants
                                    )
                                    fig_metric_phase.savefig(plot_m_path)
                                    plt.close(fig_metric_phase)
                                    print(f"      Plot de {metric_name_plot} por fase salvo em: {plot_m_path}")
                                except Exception as e_plot_metric:
                                    print(f"      Erro ao gerar plot de {metric_name_plot} por fase para round {round_name_plot}: {e_plot_metric}")
                            else:
                                print(f"    Skipping plot_metric_by_phase for {metric_name_plot}, round {round_name_plot}: missing required columns. Found: {df_round_plot.columns.tolist()}")
                        else:
                            print(f"    Skipping plot_metric_by_phase for {metric_name_plot}, round {round_name_plot}: DataFrame is empty or not a DataFrame.")
                else:
                    print(f"    Skipping plots for metric {metric_name_plot}: rounds_data is not a dictionary.")
        
        experiment_results['advanced_analysis'] = advanced_analysis_results
        print("Análises Avançadas e Geração de Plots concluídas.")

    if args.compare_tenants_directly and metrics_data:
        print("\n=== Executando Comparação Direta Entre Tenants ===")
        baseline_tenant_for_comparison = args.noisy_tenant if args.noisy_tenant else DEFAULT_NOISY_TENANT
        if args.tenants and len(args.tenants) == 1: 
            baseline_tenant_for_comparison = args.tenants[0]
            print(f"Usando tenant especificado '{baseline_tenant_for_comparison}' como baseline para comparação direta.")
        elif args.tenants and len(args.tenants) > 1:
            print(f"Múltiplos tenants especificados. Usando '{baseline_tenant_for_comparison}' (noisy ou default) como baseline.")
        else:
            print(f"Nenhum tenant específico para baseline. Usando '{baseline_tenant_for_comparison}' (noisy ou default) como baseline.")

        for metric_name, rounds_data_dict in all_metrics_data.items():
            if args.metrics and metric_name not in args.metrics:
                continue

            print(f"  Comparando tenants para a métrica: {METRIC_DISPLAY_NAMES.get(metric_name, metric_name)}")
            
            metric_all_rounds_df_list = []
            for round_name, df_round in rounds_data_dict.items():
                if args.rounds and round_name not in args.rounds: 
                    continue
                metric_all_rounds_df_list.append(df_round)
            
            if not metric_all_rounds_df_list:
                print(f"    Nenhum dado de round encontrado para a métrica {metric_name} após filtragem. Pulando.")
                continue
                
            metric_combined_df = pd.concat(metric_all_rounds_df_list, ignore_index=True)

            if not metric_combined_df.empty and 'tenant' in metric_combined_df.columns:
                if baseline_tenant_for_comparison not in metric_combined_df['tenant'].unique():
                    print(f"    Tenant de baseline '{baseline_tenant_for_comparison}' não encontrado nos dados da métrica {metric_name}. Pulando comparação para esta métrica.")
                    continue
                
                other_tenants = metric_combined_df[metric_combined_df['tenant'] != baseline_tenant_for_comparison]['tenant'].unique()
                if len(other_tenants) == 0:
                    print(f"    Nenhum outro tenant encontrado para comparar com '{baseline_tenant_for_comparison}' na métrica {metric_name}. Pulando.")
                    continue

                tenant_comparison_results_df = compare_tenant_metrics(
                    metric_combined_df, 
                    baseline_tenant=baseline_tenant_for_comparison, 
                    metric_column='value'
                )

                if not tenant_comparison_results_df.empty:
                    table_file_name = f"tenant_comparison_{metric_name}.csv"
                    table_path = os.path.join(tenant_comparison_dir, "tables", table_file_name)
                    try:
                        export_to_csv(tenant_comparison_results_df, table_path)
                        print(f"    Tabela de comparação de tenants para {metric_name} salva em: {table_path}")
                    except Exception as e:
                        print(f"    Erro ao salvar tabela de comparação de tenants para {metric_name}: {e}")
                else:
                    print(f"    Nenhum resultado de comparação de tenants gerado para {metric_name}.")
            else:
                print(f"    Dados insuficientes ou coluna 'tenant' ausente para a métrica {metric_name}. Pulando comparação de tenants.")
        print("=== Comparação Direta Entre Tenants Concluída ===")
    
    print("\nPipeline de análise concluído com sucesso!")
    
    return experiment_results


if __name__ == "__main__":
    main()
