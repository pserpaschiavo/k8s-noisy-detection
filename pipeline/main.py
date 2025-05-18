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
import seaborn as sns
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
from pipeline.analysis.tenant_analysis import calculate_inter_tenant_correlation_per_metric, calculate_inter_tenant_covariance_per_metric
from pipeline.analysis.noisy_tenant_detection import identify_noisy_tenant
from pipeline.analysis.experiment_comparison import (
    load_multiple_experiments,
    preprocess_experiments,
    calculate_statistics_summary,
    compare_distributions,
    summarize_anomalies,
    compare_experiment_phases
)
from pipeline.analysis.rounds_aggregation import aggregate_metrics_across_rounds

from pipeline.visualization.plots import (plot_metric_by_phase, plot_phase_comparison,
                                plot_tenant_impact_heatmap, plot_recovery_effectiveness,
                                plot_impact_score_barplot, plot_impact_score_trend,
                                create_heatmap, plot_multivariate_anomalies, plot_correlation_heatmap,
                                plot_entropy_heatmap, plot_entropy_top_pairs_barplot)

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
    parser.add_argument('--data-dir-comparison', type=str, nargs='+',
                        help='Diretório(s) adicional(is) para comparar múltiplos experimentos. Usado com --compare-experiments.')
    parser.add_argument('--comparison-names', type=str, nargs='+',
                        help='Nomes para os experimentos de comparação. Deve corresponder ao número de diretórios em --data-dir-comparison.')
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
    parser.add_argument('--entropy-plot-type', type=str, default='all',
                        choices=['heatmap', 'barplot', 'all'],
                        help='Tipo de gráfico para resultados de entropia: heatmap, barplot (top N), ou all.')
    parser.add_argument('--compare-experiments-rounds', type=str, nargs='+',
                        help='Round(s) específico(s) a usar para --compare-experiments')
    parser.add_argument('--compare-experiments-tenants', type=str, nargs='+',
                        help='Tenant(s) específico(s) a usar para --compare-experiments')
    
    return parser.parse_args()


def setup_output_directories(output_dir):
    """Configura diretórios de saída."""
    plots_dir = os.path.join(output_dir, 'plots')
    tables_dir = os.path.join(output_dir, 'tables')
    advanced_dir = os.path.join(output_dir, 'advanced')
    comparison_dir = os.path.join(output_dir, 'comparisons')
    reports_dir = os.path.join(output_dir, 'reports')
    causality_dir = os.path.join(output_dir, 'causality')
    rounds_comparison_intra_dir = os.path.join(output_dir, 'rounds_comparison_intra')
    tenant_comparison_dir = os.path.join(output_dir, 'tenant_comparison') # New directory
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(advanced_dir, exist_ok=True)
    os.makedirs(os.path.join(advanced_dir, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(advanced_dir, 'plots'), exist_ok=True) # Ensure advanced plots directory is created
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(causality_dir, exist_ok=True)
    os.makedirs(os.path.join(causality_dir, 'tables'), exist_ok=True)  # Ensure causality tables directory is created
    os.makedirs(os.path.join(causality_dir, 'plots'), exist_ok=True)  # Ensure causality plots directory is created
    os.makedirs(rounds_comparison_intra_dir, exist_ok=True)
    os.makedirs(os.path.join(rounds_comparison_intra_dir, 'plots'), exist_ok=True)
    os.makedirs(tenant_comparison_dir, exist_ok=True) # Create new directory
    os.makedirs(os.path.join(tenant_comparison_dir, 'tables'), exist_ok=True) # Create tables subdirectory
    
    return plots_dir, tables_dir, advanced_dir, comparison_dir, reports_dir, causality_dir, rounds_comparison_intra_dir, tenant_comparison_dir


def consolidate_experiment_metrics_across_rounds(
    metrics_data_for_experiment: dict, 
    rounds_to_consider: list | None,
    aggregator_func, 
    experiment_name_for_log: str
) -> dict:
    """Consolidates metrics across rounds for a given experiment."""
    print(f"    Consolidating metrics for experiment: {experiment_name_for_log}...")
    data_for_aggregation_input = {} 

    for metric_name, rounds_dict in metrics_data_for_experiment.items():
        if not isinstance(rounds_dict, dict):
            print(f"      Metric {metric_name} for {experiment_name_for_log} is not in round-dictionary format. Assuming pre-aggregated or single source.")
            if isinstance(rounds_dict, pd.DataFrame): 
                data_for_aggregation_input[metric_name] = rounds_dict
            continue

        dfs_to_concat_for_metric = []
        for round_name, df_round in rounds_dict.items():
            if rounds_to_consider and round_name not in rounds_to_consider:
                continue
            if not isinstance(df_round, pd.DataFrame):
                print(f"        Data for round {round_name} of metric {metric_name} in {experiment_name_for_log} is not a DataFrame. Skipping.")
                continue
            df_round_copy = df_round.copy()
            df_round_copy['round'] = round_name 
            dfs_to_concat_for_metric.append(df_round_copy)

        if not dfs_to_concat_for_metric:
            print(f"      No relevant round data for metric {metric_name} in {experiment_name_for_log} after filtering.")
            continue
        
        combined_df_for_metric = pd.concat(dfs_to_concat_for_metric, ignore_index=True)
        data_for_aggregation_input[metric_name] = combined_df_for_metric

    if not data_for_aggregation_input:
        print(f"    No metrics data prepared for aggregation for experiment {experiment_name_for_log}.")
        return {}

    print(f"    Calling aggregator for {experiment_name_for_log} with {len(data_for_aggregation_input)} metrics.")
    aggregated_result = aggregator_func(data_for_aggregation_input, value_column='value')
    print(f"    Consolidation complete for {experiment_name_for_log}.")
    return aggregated_result


def main():
    """Função principal que executa o pipeline de análise."""
    args = parse_arguments()
    
    plots_dir, tables_dir, advanced_dir, comparison_dir, reports_dir, causality_dir, rounds_comparison_intra_dir, tenant_comparison_dir = setup_output_directories(args.output_dir)
    
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
        quota_summary = get_quota_summary(quota_file_path, include_requests=True, calculate_percentages=True)
        total_entry = quota_summary.get('__total__', {})
        for tenant, quota_info in quota_summary.items():
            if tenant == '__total__':
                continue
            print(f"  {tenant.upper()}:")
            if 'cpu_limit' in quota_info:
                cpu_text = f"    CPU Limit: {quota_info['cpu_limit']}"
                if 'cpu_percent' in quota_info:
                    cpu_text += f" ({quota_info['cpu_percent']} do cluster)"
                print(cpu_text)
            if 'memory_limit' in quota_info:
                mem_text = f"    Memory Limit: {quota_info['memory_limit']}"
                if 'memory_percent' in quota_info:
                    mem_text += f" ({quota_info['memory_percent']} do cluster)"
                print(mem_text)
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
        if total_entry:
            print(f"\n  TOTAL DO CLUSTER:")
            if 'cpu_limit' in total_entry:
                print(f"    CPU Total: {total_entry['cpu_limit']}")
            if 'memory_limit' in total_entry:
                print(f"    Memory Total: {total_entry['memory_limit']}")

    # UNIFIED DATA LOADING AND INITIAL PROCESSING
    print("\nCarregando e Processando Dados do Experimento...")
    all_metrics_data = load_experiment_data(
        experiment_data_dir,
        metrics=args.metrics if args.metrics else DEFAULT_METRICS,
        rounds=args.rounds
    )

    if not all_metrics_data:
        print("ERRO CRÍTICO: Nenhum dado foi carregado. Verifique o diretório de dados, métricas e rounds especificados.")
        return 

    print("Dados do experimento carregados com sucesso.")
            
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
            
    print("\nFormatando métricas com unidades adequadas...")
    all_metrics_data = auto_format_metrics(all_metrics_data, metric_type_map)
    print("Métricas formatadas com unidades legíveis.")
            
    # Print summary of formatted units
    for metric_name, rounds_data in all_metrics_data.items():
        if isinstance(rounds_data, dict):
            for round_name, df in rounds_data.items():
                if isinstance(df, pd.DataFrame) and 'unit' in df.columns and not df['unit'].isna().all():
                    unit = df['unit'].iloc[0]
                    print(f"  {metric_name} (Round: {round_name}): {unit}")
                    break 
                elif isinstance(df, pd.DataFrame) and 'value_formatted' in df.columns:
                    print(f"  {metric_name} (Round: {round_name}): formatação personalizada aplicada")
                    break
            else: 
                if not rounds_data:
                     print(f"  {metric_name}: No rounds data found to determine unit.")
        elif isinstance(rounds_data, pd.DataFrame): 
            df = rounds_data 
            if 'unit' in df.columns and not df['unit'].isna().all():
                unit = df['unit'].iloc[0]
                print(f"  {metric_name}: {unit}")
            elif 'value_formatted' in df.columns:
                print(f"  {metric_name}: formatação personalizada aplicada")
        else:
            print(f"  {metric_name}: Data is not in expected format (DataFrame or Dict[str, DataFrame]).")

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
            if isinstance(df, dict):
                for round_name, round_df in df.items():
                    if isinstance(round_df, pd.DataFrame) and 'normalized_value' in round_df.columns and not round_df.empty:
                        print(f"  Exemplo para {metric_name} (Round: {round_name}):")
                        sample_row = round_df.iloc[0]
                        print(f"    Valor original: {sample_row.get('original_value', sample_row.get('value', 'N/A'))}")
                        print(f"    Valor normalizado: {sample_row.get('normalized_value', 'N/A')}%")
                        print(f"    Descrição: {sample_row.get('normalized_description', 'N/A')}")
                        break 
            elif isinstance(df, pd.DataFrame): 
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
    if args.compare_experiments:
        print("\n=== Comparação Entre Experimentos ===")
        
        # Carregar dados do experimento base (já formatado anteriormente no pipeline)
        print(f"Carregando dados do experimento base: {args.data_dir}")
        base_experiment_metrics_original = experiment_results['processed_data'] # This is {metric: {round: df}}
        
        print("Consolidando métricas para o experimento base (se múltiplos rounds especificados)...")
        base_experiment_metrics_consolidated = consolidate_experiment_metrics_across_rounds(
            base_experiment_metrics_original,
            args.compare_experiments_rounds, # Use rounds specified for comparison
            aggregate_metrics_across_rounds,
            "Base Experiment"
        )
        
        base_experiment_data = {
            'base_experiment': {
                'metrics': base_experiment_metrics_consolidated, # Use CONSOLIDATED metrics
                'info': {'name': 'Base Experiment'},
                'path': args.data_dir
            }
        }

        # Carregar dados dos experimentos de comparação (métricas ainda em formato raw)
        comparison_raw_experiments_data = {}
        if args.data_dir_comparison:
            print(f"Carregando dados dos experimentos para comparação: {args.data_dir_comparison}")
            num_comparison_dirs = len(args.data_dir_comparison)
            num_comparison_names = len(args.comparison_names) if args.comparison_names else 0

            if num_comparison_names != num_comparison_dirs:
                raise ValueError(
                    "O número de --comparison-names deve corresponder ao número de --data-dir-comparison."
                )

            for i, comp_dir in enumerate(args.data_dir_comparison):
                exp_name = args.comparison_names[i]
                print(f"  Carregando {exp_name} de {comp_dir}")
                comp_metrics_data = load_experiment_data(
                    comp_dir, 
                    metrics=args.metrics if args.metrics else DEFAULT_METRICS, 
                    rounds=args.compare_experiments_rounds # Load only relevant rounds if specified
                )
                comparison_raw_experiments_data[exp_name] = {
                    'metrics': comp_metrics_data, # Raw metrics {metric: {round: df}}
                    'info': {'name': exp_name},
                    'path': comp_dir
                }
        
        # Formatar e Consolidar métricas para os experimentos de comparação
        print("\nFormatando e Consolidando métricas para experimentos de comparação...")
        formatted_and_consolidated_comparison_experiments_data = {}
        for exp_name, original_exp_data_content in comparison_raw_experiments_data.items():
            print(f"  Processando experimento de comparação: {exp_name}")
            
            new_exp_data_content = {key: value for key, value in original_exp_data_content.items()}
            metrics_to_format_and_consolidate = original_exp_data_content['metrics']

            print(f"    Formatando métricas para {exp_name}...")
            formatted_metrics = auto_format_metrics( # Returns {metric: {round: df}}
                metrics_to_format_and_consolidate,
                metric_type_map
            )
            
            print(f"    Consolidando métricas para {exp_name} (se múltiplos rounds especificados)...")
            consolidated_metrics_for_comp_exp = consolidate_experiment_metrics_across_rounds(
                formatted_metrics, # Input {metric: {round: df}}
                args.compare_experiments_rounds, # Use same rounds filter
                aggregate_metrics_across_rounds,
                exp_name
            )
            
            new_exp_data_content['metrics'] = consolidated_metrics_for_comp_exp # Store CONSOLIDATED metrics
            formatted_and_consolidated_comparison_experiments_data[exp_name] = new_exp_data_content
            print(f"  Métricas formatadas e consolidadas armazenadas para {exp_name}.")

        all_experiments_data = {**base_experiment_data, **formatted_and_consolidated_comparison_experiments_data}
        
        print(f"Pré-processando experimentos para comparação. Métricas: {args.metrics if args.metrics else DEFAULT_METRICS}")
        processed_comparison_experiments = preprocess_experiments(
            all_experiments_data, # Passar o all_experiments_data agora corretamente formatado
            metrics_of_interest=args.metrics if args.metrics else DEFAULT_METRICS,
            rounds_filter=args.compare_experiments_rounds, 
            tenants_filter=args.compare_experiments_tenants 
        )

        if not processed_comparison_experiments:
            print("Nenhum dado processado para comparação de experimentos. Pulando.")
            return

        # 1. Calcular Estatísticas Resumidas
        print("\nCalculando estatísticas resumidas entre experimentos (dados consolidados)...")
        stats_summary = calculate_statistics_summary(
            processed_comparison_experiments,
            metrics=args.metrics if args.metrics else DEFAULT_METRICS,
            group_by=['tenant'] if args.compare_experiments_tenants else None 
        )
        for metric_name, summary_df in stats_summary.items():
            if not summary_df.empty:
                summary_filename = f"comparison_stats_summary_consolidated_{metric_name}.csv" # MODIFIED
                summary_path = os.path.join(comparison_dir, summary_filename)
                export_to_csv(summary_df, summary_path)
                print(f"  Resumo estatístico consolidado para {metric_name} salvo em: {summary_path}")
                print(convert_df_to_markdown(summary_df.head()))

        # 2. Comparar Distribuições
        print("\nComparando distribuições de métricas entre experimentos (dados consolidados)...")
        for metric_name in args.metrics if args.metrics else DEFAULT_METRICS:
            print(f"  Comparando distribuições consolidadas para a métrica: {metric_name}")
            plot_data, test_results = compare_distributions(
                processed_comparison_experiments,
                metric=metric_name,
                tenants_filter=args.compare_experiments_tenants, 
                rounds_filter=args.compare_experiments_rounds # This filter is for preprocess_experiments, consolidation already happened
            )
            if test_results:
                dist_comp_filename = f"comparison_distribution_test_consolidated_{metric_name}.csv" # MODIFIED
                dist_comp_path = os.path.join(comparison_dir, dist_comp_filename)
                test_results_df = pd.DataFrame.from_dict(test_results, orient='index')
                export_to_csv(test_results_df, dist_comp_path)
                print(f"    Resultados do teste de distribuição consolidada para {metric_name} salvos em: {dist_comp_path}")
                print(convert_df_to_markdown(test_results_df.head()))

            if plot_data and plot_data['series']:
                plt.figure(figsize=VISUALIZATION_CONFIG.get('figure_size', (10, 6)))
                sns.boxplot(data=plot_data['series'])
                plt.xticks(ticks=range(len(plot_data['labels'])), labels=plot_data['labels'], rotation=45, ha="right")
                
                base_title = f"Distribuição Consolidada de {METRIC_DISPLAY_NAMES.get(metric_name, metric_name)} Entre Experimentos" # MODIFIED
                title_suffix = []
                if args.compare_experiments_rounds: # Info about which rounds were part of consolidation
                    title_suffix.append(f"Rounds Consolidados: {', '.join(args.compare_experiments_rounds)}")
                else:
                    title_suffix.append("Rounds Consolidados: Todos") # If no specific rounds, all were consolidated
                if args.compare_experiments_tenants:
                    title_suffix.append(f"Tenants: {', '.join(args.compare_experiments_tenants)}")
                
                if title_suffix:
                    plt.title(f"{base_title}\n({'; '.join(title_suffix)})")
                else:
                    plt.title(base_title)
                    
                plot_filename = f"comparison_distribution_plot_consolidated_{metric_name}.png" # MODIFIED
                plot_path = os.path.join(comparison_dir, plot_filename)
                try:
                    plt.tight_layout()
                    plt.savefig(plot_path)
                    print(f"    Gráfico de comparação de distribuição consolidada para {metric_name} salvo em: {plot_path}")
                except Exception as e:
                    print(f"    Erro ao salvar gráfico de comparação de distribuição consolidada: {e}")
                plt.close()
        print("Comparação Entre Experimentos (com consolidação de rounds) concluída.")

    print("\nPipeline de análise concluído com sucesso!")
    
    return experiment_results


if __name__ == "__main__":
    main()
