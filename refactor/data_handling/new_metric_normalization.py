"""
Módulo para normalização de métricas de recursos.

Este módulo fornece funções para converter métricas brutas em percentuais
em relação aos recursos totais do cluster ou aos limites definidos nos manifestos de quotas.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple

# Importar parser de quotas
try:
    from .quota_parser import (
        get_tenant_quotas, create_node_config_from_quotas, 
        get_best_unit_for_value, format_value_with_unit,
        convert_to_best_unit, get_formatted_quota_values,
        get_quota_summary
    )
except ImportError:
    # Caminho alternativo para importação quando executado como script
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.data_processing.quota_parser import (
        get_tenant_quotas, create_node_config_from_quotas,
        get_best_unit_for_value, format_value_with_unit,
        convert_to_best_unit, get_formatted_quota_values,
        get_quota_summary
    )


def normalize_metrics_by_node_capacity(metrics_dict: Dict[str, pd.DataFrame],
                                     node_config: Dict[str, float],
                                     use_tenant_quotas: bool = True,
                                     quota_file: str = None,
                                     add_relative_values: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Normaliza métricas brutas para percentuais do recurso total disponível ou dos limites do tenant.
    
    Args:
        metrics_dict: Dicionário com DataFrames de métricas
        node_config: Configuração do nó com capacidades totais dos recursos
        use_tenant_quotas: Se True, normaliza contra os limites específicos dos tenants
        quota_file: Caminho para arquivo de quotas (opcional)
        add_relative_values: Se True, adiciona colunas com valores relativos (% dos limites)
        
    Returns:
        Dicionário com métricas normalizadas
    """
    normalized_metrics = {}
    
    # Carregar quotas de tenants se solicitado
    tenant_quotas = {}
    if use_tenant_quotas:
        tenant_quotas = get_tenant_quotas(quota_file)
    
    # Normalizar CPU (assumindo que valores brutos estão em cores)
    if 'cpu_usage' in metrics_dict and 'CPUS' in node_config:
        df = metrics_dict['cpu_usage'].copy()
        
        # Aplicar normalização por tenant quando use_tenant_quotas=True
        if use_tenant_quotas:
            # Aplicar normalização por tenant
            for tenant, group in df.groupby('tenant'):
                namespace = tenant  # Assumindo que o nome do tenant é o mesmo do namespace
                # Obter limite específico do tenant ou usar valor global
                tenant_cpu_limit = tenant_quotas.get(namespace, {}).get('cpu_limit', 0)
                
                if tenant_cpu_limit > 0:
                    # Normalizar contra o limite específico do tenant (valores em cores)
                    tenant_mask = df['tenant'] == tenant
                    
                    # Converter para percentual do limite do tenant
                    if add_relative_values:
                        df.loc[tenant_mask, 'normalized_value'] = df.loc[tenant_mask, 'value'] * 100 / tenant_cpu_limit
                    
                    # Formatar o limite do tenant de forma legível para a descrição
                    cpu_value, cpu_unit = convert_to_best_unit(tenant_cpu_limit, 'cpu')
                    formatted_limit = f"{cpu_value:.2f} {cpu_unit}"
                    
                    # Adicionar informações detalhadas sobre a normalização
                    if add_relative_values:
                        df.loc[tenant_mask, 'normalized_description'] = f"% of {formatted_limit} (quota)"
                    
                    # Preservar os valores das quotas em diferentes formatos para uso posterior
                    df.loc[tenant_mask, 'quota_limit'] = tenant_cpu_limit
                    df.loc[tenant_mask, 'quota_limit_formatted'] = formatted_limit
                    df.loc[tenant_mask, 'quota_limit_cores'] = tenant_cpu_limit
                    df.loc[tenant_mask, 'quota_limit_millicores'] = tenant_cpu_limit * 1000
                    
                    # Adicionar porcentagens em relação à capacidade total do nó
                    df.loc[tenant_mask, 'quota_percent_of_node'] = tenant_cpu_limit * 100 / node_config['CPUS']
                else:
                    # Fallback para normalização global quando o tenant não tem quota definida
                    tenant_mask = df['tenant'] == tenant
                    
                    if add_relative_values:
                        df.loc[tenant_mask, 'normalized_value'] = df.loc[tenant_mask, 'value'] * 100 / node_config['CPUS']
                    
                    # Formatar o limite global de forma legível
                    cpu_value, cpu_unit = convert_to_best_unit(node_config['CPUS'], 'cpu')
                    formatted_limit = f"{cpu_value:.2f} {cpu_unit}"
                    
                    if add_relative_values:
                        df.loc[tenant_mask, 'normalized_description'] = f"% of {formatted_limit} (node)"
                    
                    # Adicionar o valor do limite para uso em outras partes do código
                    df.loc[tenant_mask, 'quota_limit'] = node_config['CPUS']
                    df.loc[tenant_mask, 'quota_limit_formatted'] = formatted_limit
                    df.loc[tenant_mask, 'quota_limit_cores'] = node_config['CPUS']
                    df.loc[tenant_mask, 'quota_limit_millicores'] = node_config['CPUS'] * 1000
                    df.loc[tenant_mask, 'quota_percent_of_node'] = 100  # 100% do nó
        else:
            # Normalização global para todos os tenants
            total_cpu_cores = node_config['CPUS']
            
            if add_relative_values:
                df['normalized_value'] = df['value'] * 100 / total_cpu_cores
            
            # Formatar o limite global de forma legível
            cpu_value, cpu_unit = convert_to_best_unit(total_cpu_cores, 'cpu')
            formatted_limit = f"{cpu_value:.2f} {cpu_unit}"
            
            if add_relative_values:
                df['normalized_description'] = f"% of {formatted_limit}"
            
            # Adicionar o valor do limite para uso em outras partes do código
            df['quota_limit'] = total_cpu_cores
            df['quota_limit_formatted'] = formatted_limit
            df['quota_limit_cores'] = total_cpu_cores
            df['quota_limit_millicores'] = total_cpu_cores * 1000
            df['quota_percent_of_node'] = 100  # 100% do nó
        
        # Adicionar informações adicionais
        df['value_cores'] = df['value']  # Preservar valor em cores
        
        # Adicionar outras representações úteis
        df['value_millicores'] = df['value'] * 1000  # Converter para milicores
        
        # Formatar valores em diferentes unidades para uso posterior
        df['value_formatted'] = df.apply(
            lambda row: format_value_with_unit(row['value'], 'cpu'), axis=1
        )
        
        if add_relative_values:
            df['unit'] = '%'
            
        df['metric_type'] = 'cpu'
        normalized_metrics['cpu_usage'] = df
    
    # Normalizar memória (assumindo que valores brutos estão em bytes)
    if 'memory_usage' in metrics_dict and 'MEMORY_BYTES' in node_config:
        df = metrics_dict['memory_usage'].copy()
        
        # Aplicar normalização por tenant quando use_tenant_quotas=True
        if use_tenant_quotas:
            # Aplicar normalização por tenant
            for tenant, group in df.groupby('tenant'):
                namespace = tenant  # Assumindo que o nome do tenant é o mesmo do namespace
                # Obter limite específico do tenant ou usar valor global
                tenant_memory_limit = tenant_quotas.get(namespace, {}).get('memory_limit', 0)
                
                if tenant_memory_limit > 0:
                    # Normalizar contra o limite específico do tenant
                    tenant_mask = df['tenant'] == tenant
                    
                    # Converter para percentual do limite do tenant
                    if add_relative_values:
                        df.loc[tenant_mask, 'normalized_value'] = df.loc[tenant_mask, 'value'] * 100 / tenant_memory_limit
                    
                    # Formatar o limite do tenant de forma legível
                    mem_value, mem_unit = convert_to_best_unit(tenant_memory_limit, 'memory')
                    formatted_limit = f"{mem_value:.2f} {mem_unit}"
                    
                    # Adicionar informações detalhadas sobre a normalização
                    if add_relative_values:
                        df.loc[tenant_mask, 'normalized_description'] = f"% of {formatted_limit} (quota)"
                    
                    # Preservar os valores originais em diferentes unidades para uso posterior
                    df.loc[tenant_mask, 'quota_limit'] = tenant_memory_limit
                    df.loc[tenant_mask, 'quota_limit_formatted'] = formatted_limit
                    df.loc[tenant_mask, 'quota_limit_bytes'] = tenant_memory_limit
                    df.loc[tenant_mask, 'quota_limit_kib'] = tenant_memory_limit / (2**10)
                    df.loc[tenant_mask, 'quota_limit_mib'] = tenant_memory_limit / (2**20)
                    df.loc[tenant_mask, 'quota_limit_gib'] = tenant_memory_limit / (2**30)
                    
                    # Adicionar porcentagens em relação à capacidade total do nó
                    df.loc[tenant_mask, 'quota_percent_of_node'] = tenant_memory_limit * 100 / node_config['MEMORY_BYTES']
                else:
                    # Fallback para normalização global quando o tenant não tem quota definida
                    tenant_mask = df['tenant'] == tenant
                    
                    if add_relative_values:
                        df.loc[tenant_mask, 'normalized_value'] = df.loc[tenant_mask, 'value'] * 100 / node_config['MEMORY_BYTES']
                    
                    # Formatar o limite global de forma legível
                    mem_value, mem_unit = convert_to_best_unit(node_config['MEMORY_BYTES'], 'memory')
                    formatted_limit = f"{mem_value:.2f} {mem_unit}"
                    
                    if add_relative_values:
                        df.loc[tenant_mask, 'normalized_description'] = f"% of {formatted_limit} (node)"
                    
                    # Adicionar o valor do limite para uso em outras partes do código
                    df.loc[tenant_mask, 'quota_limit'] = node_config['MEMORY_BYTES']
                    df.loc[tenant_mask, 'quota_limit_formatted'] = formatted_limit
                    df.loc[tenant_mask, 'quota_limit_bytes'] = node_config['MEMORY_BYTES']
                    df.loc[tenant_mask, 'quota_limit_kib'] = node_config['MEMORY_BYTES'] / (2**10)
                    df.loc[tenant_mask, 'quota_limit_mib'] = node_config['MEMORY_BYTES'] / (2**20)
                    df.loc[tenant_mask, 'quota_limit_gib'] = node_config['MEMORY_BYTES'] / (2**30)
                    df.loc[tenant_mask, 'quota_percent_of_node'] = 100  # 100% do nó
        else:
            # Normalização global para todos os tenants
            total_memory = node_config['MEMORY_BYTES']
            
            if add_relative_values:
                df['normalized_value'] = df['value'] * 100 / total_memory
            
            # Formatar o limite global de forma legível
            mem_value, mem_unit = convert_to_best_unit(total_memory, 'memory')
            formatted_limit = f"{mem_value:.2f} {mem_unit}"
            
            if add_relative_values:
                df['normalized_description'] = f"% of {formatted_limit}"
            
            # Adicionar o valor do limite para uso em outras partes do código
            df['quota_limit'] = total_memory
            df['quota_limit_formatted'] = formatted_limit
            df['quota_limit_bytes'] = total_memory
            df['quota_limit_kib'] = total_memory / (2**10)
            df['quota_limit_mib'] = total_memory / (2**20)
            df['quota_limit_gib'] = total_memory / (2**30)
            df['quota_percent_of_node'] = 100  # 100% do nó
        
        # Adicionar informações adicionais em diferentes unidades para uso posterior
        df['value_bytes'] = df['value']  # Preservar valor original em bytes
        df['value_kib'] = df['value'] / (2**10)
        df['value_mib'] = df['value'] / (2**20)
        df['value_gib'] = df['value'] / (2**30)
        
        # Formatar valores em unidades legíveis para uso posterior
        df['value_formatted'] = df.apply(
            lambda row: format_value_with_unit(row['value'], 'memory'), axis=1
        )
        
        if add_relative_values:
            df['unit'] = '%'
            
        df['metric_type'] = 'memory'
        normalized_metrics['memory_usage'] = df
    
    return normalized_metrics


def apply_normalization_to_all_metrics(metrics_dict: Dict[str, pd.DataFrame], 
                                     node_config: Dict[str, float], 
                                     replace_original: bool = False,
                                     use_tenant_quotas: bool = True,
                                     show_as_percentage: bool = False,
                                     use_readable_units: bool = True,
                                     add_relative_values: bool = True,
                                     quota_file: str = None) -> Dict[str, pd.DataFrame]:
    """
    Aplica normalização a todas as métricas e opcionalmente substitui os valores originais.
    
    Args:
        metrics_dict: Dicionário com DataFrames de métricas
        node_config: Configuração do nó com capacidades totais dos recursos
        replace_original: Se True, substitui a coluna 'value' pelos valores normalizados
        use_tenant_quotas: Se True, normaliza contra os limites específicos dos tenants
        show_as_percentage: Se True, mostra valores como percentuais
        use_readable_units: Se True, converte valores para unidades mais legíveis
        add_relative_values: Se True, adiciona colunas com valores relativos (% dos limites)
        quota_file: Caminho para arquivo de quotas (opcional)
        
    Returns:
        Dicionário com métricas processadas
    """
    processed_metrics = {}
    
    # Primeiro passo: Normalizar contra capacidades ou quotas
    normalized = normalize_metrics_by_node_capacity(
        metrics_dict, 
        node_config, 
        use_tenant_quotas=use_tenant_quotas,
        quota_file=quota_file,
        add_relative_values=add_relative_values
    )
    
    # Segundo passo: Aplicar unidades legíveis se solicitado
    if use_readable_units:
        for metric_name, df in normalized.items():
            processed_df = df.copy()
            
            if replace_original and 'normalized_value' in df.columns:
                # Substituir valores originais pelos normalizados
                processed_df['original_value'] = processed_df['value'].copy()
                processed_df['value'] = processed_df['normalized_value']
                
                if show_as_percentage:
                    processed_df['unit'] = '%'
            
            # Preservar no dicionário de saída
            processed_metrics[metric_name] = processed_df
    else:
        # Se não for para usar unidades legíveis, apenas copiar os DataFrames normalizados
        processed_metrics = {k: v.copy() for k, v in normalized.items()}
        
        if replace_original:
            # Substituir valores originais pelos normalizados quando solicitado
            for metric_name, df in processed_metrics.items():
                if 'normalized_value' in df.columns:
                    df['original_value'] = df['value'].copy()
                    df['value'] = df['normalized_value']
                    
                    if show_as_percentage:
                        df['unit'] = '%'
    
    # Processar métricas que não foram tratadas na normalização (como disk e network)
    for metric_name, df in metrics_dict.items():
        if metric_name not in processed_metrics:
            if use_readable_units:
                # Determinar o tipo de métrica com base no nome
                if 'disk' in metric_name.lower():
                    metric_type = 'disk'
                elif 'network' in metric_name.lower() or 'bandwidth' in metric_name.lower():
                    metric_type = 'network'
                else:
                    metric_type = None
                
                # Se conseguiu determinar o tipo, formatar com unidades adequadas
                if metric_type:
                    processed_df = df.copy()
                    
                    # Adicionar formatação legível
                    processed_df['value_formatted'] = processed_df['value'].apply(
                        lambda x: format_value_with_unit(x, metric_type)
                    )
                    
                    # Adicionar valores em diferentes unidades comuns para uso posterior
                    if metric_type == 'disk':
                        processed_df['value_bytes'] = processed_df['value']
                        processed_df['value_kb'] = processed_df['value'] / (2**10)
                        processed_df['value_mb'] = processed_df['value'] / (2**20)
                        processed_df['value_gb'] = processed_df['value'] / (2**30)
                        processed_df['metric_type'] = 'disk'
                        
                    elif metric_type == 'network':
                        processed_df['value_bytes'] = processed_df['value']
                        processed_df['value_kb'] = processed_df['value'] / (2**10)
                        processed_df['value_mb'] = processed_df['value'] / (2**20)
                        processed_df['value_gb'] = processed_df['value'] / (2**30)
                        processed_df['metric_type'] = 'network'
                    
                    processed_metrics[metric_name] = processed_df
                else:
                    # Se não conseguiu determinar o tipo, apenas copiar os dados
                    processed_metrics[metric_name] = df.copy()
            else:
                # Se não for para usar unidades legíveis, apenas copiar os dados
                processed_metrics[metric_name] = df.copy()
    
    return processed_metrics


def auto_format_metrics(metrics_dict: Dict[str, Dict[str, pd.DataFrame]],
                        metric_type_map: Dict[str, str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Formata automaticamente as métricas para exibição com as melhores unidades.
    Esta função não normaliza, apenas converte para unidades mais legíveis.
    
    Args:
        metrics_dict: Dicionário aninhado (nome_metrica -> nome_rodada -> DataFrame) de métricas.
                      The structure is metric_name (str) -> round_name (str) -> pd.DataFrame.
        metric_type_map: Mapeamento opcional de nomes de métricas para seus tipos.
        
    Returns:
        Dicionário aninhado com métricas formatadas, preserving the input structure.
    """
    formatted_outer = {}
    
    # Mapeamento padrão de sufixos para tipos
    default_type_map = {
        'cpu': ['cpu', 'cores', 'processor'],
        'memory': ['memory', 'mem', 'ram'],
        'disk': ['disk', 'storage', 'volume', 'pv', 'io_', 'iops'],
        'network': ['network', 'bandwidth', 'throughput', 'rx', 'tx']
    }
    
    for metric_name, rounds_dict in metrics_dict.items():
        formatted_inner = {}
        for round_name, df in rounds_dict.items():
            # Ensure df is a DataFrame before processing
            if not isinstance(df, pd.DataFrame):
                # If not a DataFrame, carry it over as is (e.g. if it's None or already processed)
                # Consider logging a warning here if this case is unexpected.
                formatted_inner[round_name] = df 
                continue

            if df.empty:
                formatted_inner[round_name] = df.copy()
                continue
            
            formatted_df = df.copy()
            
            # Determinar o tipo de métrica
            metric_type = None
            
            # Usar o mapa fornecido pelo usuário, se disponível
            if metric_type_map and metric_name in metric_type_map:
                metric_type = metric_type_map[metric_name]
            else:
                # Detecção automática baseada no nome da métrica
                metric_name_lower = metric_name.lower()
                for type_key, keywords in default_type_map.items():
                    if any(keyword in metric_name_lower for keyword in keywords):
                        metric_type = type_key
                        break
                
                # Casos especiais para subtipos
                if metric_type == 'disk' and any(kw in metric_name_lower for kw in ['iops', 'io_']):
                    metric_type = 'disk_iops'
            
            # Se não conseguir detectar o tipo, mantém o valor original
            if not metric_type:
                formatted_inner[round_name] = formatted_df
                continue
            
            # Verificar se já existe uma coluna 'unit' - nesse caso, preservamos os valores existentes
            if 'unit' in formatted_df.columns and not formatted_df['unit'].isna().all():
                formatted_inner[round_name] = formatted_df
                continue
            
            # Calcular estatísticas para determinar a melhor unidade
            if 'value' in formatted_df:
                # Usar o 75º percentil em vez da média para melhor representar os valores típicos
                typical_value = formatted_df['value'].quantile(0.75)
                if pd.isna(typical_value) or typical_value == 0:
                    # Fallback para média se o percentil for nulo ou zero
                    typical_value = formatted_df['value'].mean()
                
                # Processar apenas se typical_value é um número válido
                if pd.notna(typical_value):
                    converted_value, unit = convert_to_best_unit(typical_value, metric_type)
                    
                    # Define conversion_factor, ensuring converted_value is not zero and is a number
                    if pd.notna(converted_value) and converted_value != 0:
                        conversion_factor = typical_value / converted_value
                    else:
                        conversion_factor = 1.0 # Default to 1.0 if no valid conversion
                    
                    formatted_df['original_value'] = formatted_df['value'].copy()
                    
                    # Apply conversion if factor is not zero
                    if conversion_factor != 0:
                        formatted_df['value'] = formatted_df['value'] / conversion_factor
                    # Else: values remain as original_value (scaled by 1.0 effectively)
                    
                    formatted_df['unit'] = unit
                    formatted_df['metric_type'] = metric_type
                    
                    # Adicionar formato legível para cada valor individual
                    formatted_df['value_formatted'] = formatted_df['original_value'].apply(
                        lambda x: format_value_with_unit(x, metric_type)
                    )
                    
                    # Adicionar valor da conversão para facilitar manipulações posteriores
                    formatted_df['conversion_factor'] = conversion_factor
                # else: if typical_value is NaN, no formatting changes are applied to 'value', 'unit', etc.
                # formatted_df remains a copy of the original df in terms of these columns.
            
            formatted_inner[round_name] = formatted_df
        
        formatted_outer[metric_name] = formatted_inner
    
    return formatted_outer
