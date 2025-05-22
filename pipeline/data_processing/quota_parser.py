"""
Módulo para analisar manifestos de Resource Quotas do Kubernetes.

Este módulo fornece funções para extrair informações de recursos de manifestos
de quotas e utilizá-las para normalização adequada das métricas.
"""

import os
import yaml
import re
from typing import Dict, Optional, Any, Tuple, Union


def parse_quantity(quantity_str: str) -> float:
    """
    Converte uma string de quantidade no formato do Kubernetes para um valor numérico.
    
    Args:
        quantity_str: String de quantidade (ex: "100m", "2Gi", "500Mi")
        
    Returns:
        float: Valor numérico convertido para a unidade base
    """
    if not quantity_str or not isinstance(quantity_str, str):
        return 0.0
    
    # CPU pode ser especificado em milicores com sufixo "m"
    if quantity_str.endswith('m'):
        # Converte milicores para cores (divide por 1000)
        return float(quantity_str[:-1]) / 1000

    # Memória e armazenamento com sufixos de unidade
    suffixes = {
        'Ei': 2**60, 'Pi': 2**50, 'Ti': 2**40, 'Gi': 2**30, 'Mi': 2**20, 'Ki': 2**10,  # Binários
        'E': 10**18, 'P': 10**15, 'T': 10**12, 'G': 10**9, 'M': 10**6, 'K': 10**3      # Decimais
    }
    
    for suffix, multiplier in suffixes.items():
        if quantity_str.endswith(suffix):
            return float(quantity_str[:-len(suffix)]) * multiplier
            
    # Nenhum sufixo, tenta converter diretamente
    try:
        return float(quantity_str)
    except ValueError:
        return 0.0


def get_best_unit_for_value(value: float, metric_type: str) -> Tuple[float, str]:
    """
    Determina a melhor unidade para representar um valor com base no tipo de métrica.
    
    Args:
        value: Valor numérico a ser convertido
        metric_type: Tipo de métrica ('cpu', 'memory', 'disk', 'network')
        
    Returns:
        Tupla (valor_convertido, unidade)
    """
    if value is None or not isinstance(value, (int, float)):
        return 0.0, ''
    
    if metric_type == 'cpu':
        if value < 0.01:
            # Valores muito pequenos em millicores
            return value * 1000, 'millicores'
        elif value < 0.1:
            # Menos que 0.1 cores = mostrar em milicores
            return value * 1000, 'm'
        else:
            # Mostrar em cores
            return value, 'cores'
    
    elif metric_type == 'memory':
        if value < 2**10:  # < 1 KiB
            return value, 'B'
        elif value < 2**20:  # < 1 MiB
            return value / (2**10), 'KiB'
        elif value < 2**30:  # < 1 GiB
            return value / (2**20), 'MiB'
        else:
            return value / (2**30), 'GiB'
    
    elif metric_type in ['disk', 'storage']:
        # Para armazenamento, use unidades binárias (KiB, MiB, GiB)
        if value < 2**10:  # < 1 KiB
            return value, 'B'
        elif value < 2**20:  # < 1 MiB
            return value / (2**10), 'KiB'
        elif value < 2**30:  # < 1 GiB
            return value / (2**20), 'MiB'
        else:
            return value / (2**30), 'GiB'
    
    elif metric_type in ['disk_iops', 'io']:
        # Para IOPS, sem unidade específica além de operações/s
        if value < 1000:
            return value, 'IOPS'
        else:
            return value / 1000, 'kIOPS'
    
    elif metric_type in ['network', 'bandwidth']:
        # Para taxas de transferência, use unidades decimais (KB/s, MB/s, GB/s)
        if value < 1024:  # < 1 KB/s
            return value, 'B/s'
        elif value < 1024**2:  # < 1 MB/s
            return value / 1024, 'KB/s'
        elif value < 1024**3:  # < 1 GB/s
            return value / (1024**2), 'MB/s'
        else:
            return value / (1024**3), 'GB/s'
    
    # Caso genérico, apenas retornar o valor sem unidade
    return value, ''


def format_value_with_unit(value: float, metric_type: str = None, 
                          custom_format: str = '{:.2f} {}',
                          auto_detect_unit: bool = True,
                          preserve_small_values: bool = True) -> str:
    """
    Formata um valor numérico com a unidade apropriada.
    
    Args:
        value: Valor numérico a ser formatado
        metric_type: Tipo de métrica ('cpu', 'memory', 'disk', 'network')
        custom_format: String de formato personalizado
        auto_detect_unit: Se True, detecta automaticamente a melhor unidade
        preserve_small_values: Se True, valores pequenos não são arredondados para zero
        
    Returns:
        String formatada com valor e unidade
    """
    if value is None:
        return "N/A"
    
    if metric_type is None:
        return custom_format.format(value, '')
    
    if auto_detect_unit:
        converted_value, unit = get_best_unit_for_value(value, metric_type)
        
        # Para valores muito pequenos, usar notação científica para evitar arredondamentos para zero
        if preserve_small_values and converted_value < 0.01 and converted_value > 0:
            return f"{converted_value:.2e} {unit}"
        
        return custom_format.format(converted_value, unit)
    
    # Formatos específicos para tipos de métricas sem auto-detecção
    if metric_type == 'cpu':
        # Valor bruto em cores, formatação direta
        if value < 0.01:
            return f"{value * 1000:.2f} millicores"
        return custom_format.format(value, 'cores')
    
    elif metric_type == 'memory':
        # Valor bruto em bytes, formatação direta
        return custom_format.format(value / (2**30), 'GiB')
    
    elif metric_type in ['disk', 'network']:
        # Valor bruto, formatação direta
        return custom_format.format(value / (2**20), 'MiB')
    
    # Default para outros tipos
    return custom_format.format(value, metric_type)


def convert_to_best_unit(value: float, metric_type: str) -> Tuple[float, str]:
    """
    Converte um valor para a unidade mais legível com base no tipo de métrica.
    
    Args:
        value: Valor numérico a ser convertido
        metric_type: Tipo de métrica ('cpu', 'memory', 'disk', 'network')
        
    Returns:
        Tupla (valor_convertido, unidade)
    """
    return get_best_unit_for_value(value, metric_type)


def load_resource_quotas(yaml_file: str) -> Dict[str, Dict[str, float]]:
    """
    Carrega as quotas de recurso de um arquivo YAML.
    
    Args:
        yaml_file: Caminho para o arquivo YAML contendo ResourceQuotas
        
    Returns:
        Dict: Dicionário mapeando namespaces para seus limites de recursos
    """
    quotas = {}
    
    try:
        with open(yaml_file, 'r') as f:
            # O arquivo YAML pode conter vários documentos separados por "---"
            documents = yaml.safe_load_all(f)
            
            for doc in documents:
                if not doc:
                    continue
                    
                # Verificar se é um ResourceQuota
                if doc.get('kind') == 'ResourceQuota':
                    namespace = doc.get('metadata', {}).get('namespace', 'default')
                    
                    # Extrair limites
                    hard_limits = doc.get('spec', {}).get('hard', {})
                    
                    if namespace not in quotas:
                        quotas[namespace] = {}
                    
                    # Processar CPU
                    if 'limits.cpu' in hard_limits:
                        quotas[namespace]['cpu_limit'] = parse_quantity(hard_limits['limits.cpu'])
                    if 'requests.cpu' in hard_limits:
                        quotas[namespace]['cpu_request'] = parse_quantity(hard_limits['requests.cpu'])
                        
                    # Processar Memória
                    if 'limits.memory' in hard_limits:
                        quotas[namespace]['memory_limit'] = parse_quantity(hard_limits['limits.memory'])
                    if 'requests.memory' in hard_limits:
                        quotas[namespace]['memory_request'] = parse_quantity(hard_limits['requests.memory'])
                    
        return quotas
    except Exception as e:
        print(f"Erro ao carregar quotas: {e}")
        return {}


def get_tenant_quotas(quota_file: str = None) -> Dict[str, Dict[str, float]]:
    """
    Obtém as quotas de recurso para cada tenant.
    
    Args:
        quota_file: Caminho para o arquivo de quotas (opcional)
        
    Returns:
        Dict: Dicionário mapeando tenants para seus limites de recursos
    """
    if not quota_file:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        quota_file = os.path.join(base_dir, 'resource-quotas.yaml')
    
    if not os.path.exists(quota_file):
        print(f"Arquivo de quotas não encontrado: {quota_file}")
        return {}
    
    namespace_quotas = load_resource_quotas(quota_file)
    tenant_quotas = {}
    
    # Converter de namespaces para tenant-X
    for namespace, quotas in namespace_quotas.items():
        tenant_quotas[namespace] = quotas
    
    return tenant_quotas


def get_formatted_quota_values(namespace: str, quota_file: str = None, 
                              include_requests: bool = False) -> Dict[str, str]:
    """
    Obtém os valores formatados de quotas para um namespace específico.
    
    Args:
        namespace: O namespace para obter as quotas
        quota_file: Caminho para o arquivo de quotas (opcional)
        include_requests: Se True, inclui também os valores de 'requests'
        
    Returns:
        Dict: Dicionário com valores formatados em unidades legíveis
    """
    quotas = get_tenant_quotas(quota_file)
    if namespace not in quotas:
        return {}
    
    tenant_quota = quotas[namespace]
    formatted = {}
    
    # Formatar CPU
    if 'cpu_limit' in tenant_quota:
        cpu_value, cpu_unit = convert_to_best_unit(tenant_quota['cpu_limit'], 'cpu')
        formatted['cpu_limit'] = f"{cpu_value:.2f} {cpu_unit}"
    
    # Formatar Memória
    if 'memory_limit' in tenant_quota:
        mem_value, mem_unit = convert_to_best_unit(tenant_quota['memory_limit'], 'memory')
        formatted['memory_limit'] = f"{mem_value:.2f} {mem_unit}"
    
    # Incluir requests se solicitado
    if include_requests:
        if 'cpu_request' in tenant_quota:
            req_cpu_value, req_cpu_unit = convert_to_best_unit(tenant_quota['cpu_request'], 'cpu')
            formatted['cpu_request'] = f"{req_cpu_value:.2f} {req_cpu_unit}"
        
        if 'memory_request' in tenant_quota:
            req_mem_value, req_mem_unit = convert_to_best_unit(tenant_quota['memory_request'], 'memory')
            formatted['memory_request'] = f"{req_mem_value:.2f} {req_mem_unit}"
    
    return formatted


def create_node_config_from_quotas(quota_file: str = None) -> Dict[str, float]:
    """
    Cria uma configuração de nó baseada nas quotas totais.
    Esta função soma todas as quotas de recursos e adiciona
    margens para estimar a capacidade total do nó.
    
    Args:
        quota_file: Caminho para o arquivo de quotas (opcional)
        
    Returns:
        Dict: Configuração do nó com capacidades totais dos recursos
    """
    tenant_quotas = get_tenant_quotas(quota_file)
    
    # Calcular totais somando todos os limites de recursos
    total_cpu_limit = sum(quota.get('cpu_limit', 0) for quota in tenant_quotas.values())
    total_memory_limit = sum(quota.get('memory_limit', 0) for quota in tenant_quotas.values())
    
    # Adicionar margem para recursos de sistema (geralmente 10-20%)
    # Isso reflete que um nó reserva recursos para o sistema e outros componentes
    system_margin = 0.2  # 20% de margem
    estimated_node_cpu = total_cpu_limit / (1 - system_margin)
    estimated_node_memory = total_memory_limit / (1 - system_margin)
    
    # Converter memória para diferentes unidades
    memory_bytes = estimated_node_memory
    memory_kb = memory_bytes / (2**10)
    memory_mb = memory_bytes / (2**20)
    memory_gb = memory_bytes / (2**30)
    
    # Estimar métricas de armazenamento
    # Regra geral: ~10x memória total para armazenamento permanente, ~2-4x para scratch
    disk_size_bytes = memory_bytes * 10
    disk_size_gb = disk_size_bytes / (2**30)
    
    # Para I/O, estimar com base no número de cores
    # Regra geral: ~100 IOPS por core para cargas normais
    disk_iops = max(500, total_cpu_limit * 100)
    disk_bandwidth_mbps = max(100, disk_iops * 0.256)  # ~256 KB por operação I/O
    
    # Estimar largura de banda de rede baseada no CPU e memória
    # Regra geral: ~125-250 Mbps por core para cargas normais
    bandwidth_factor = 250  # Mbps por core
    network_bandwidth_mbps = max(1000, total_cpu_limit * bandwidth_factor)
    
    # Construir a configuração do nó com todos os valores
    node_config = {
        # Recursos de CPU
        'CPUS': estimated_node_cpu,
        'TOTAL_CPU_CORES': estimated_node_cpu,
        'CPU_CORES_PER_TENANT': total_cpu_limit / len(tenant_quotas) if tenant_quotas else 0,
        
        # Recursos de memória
        'MEMORY_BYTES': memory_bytes,
        'MEMORY_KB': memory_kb,
        'MEMORY_MB': memory_mb,
        'MEMORY_GB': memory_gb,
        
        # Recursos de armazenamento
        'DISK_SIZE_BYTES': disk_size_bytes,
        'DISK_SIZE_GB': disk_size_gb,
        'DISK_IOPS': disk_iops,
        'DISK_BANDWIDTH_MBPS': disk_bandwidth_mbps,
        
        # Recursos de rede
        'NETWORK_BANDWIDTH_MBPS': network_bandwidth_mbps,
        'NETWORK_BANDWIDTH_GBPS': network_bandwidth_mbps / 1000,
        
        # Metadados
        'TENANT_COUNT': len(tenant_quotas),
        'GENERATED_FROM_QUOTAS': True,
        'SYSTEM_RESOURCES_MARGIN': system_margin * 100  # em percentual
    }
    
    return node_config


def get_quota_summary(quota_file: str = None, include_requests: bool = False,
                   calculate_percentages: bool = True,
                   use_markdown: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Gera um resumo de todas as quotas com formatação de unidades para facilitar a leitura.
    
    Args:
        quota_file: Caminho para o arquivo de quotas (opcional)
        include_requests: Se True, inclui também os valores de 'requests'
        calculate_percentages: Se True, calcula a proporção de recursos de cada tenant
        use_markdown: Se True, formata o resumo para exibição em markdown
        
    Returns:
        Dict: Resumo formatado das quotas
    """
    quotas = get_tenant_quotas(quota_file)
    summary = {}
    
    # Calcular totais para percentuais
    total_cpu_limit = sum(quota.get('cpu_limit', 0) for quota in quotas.values())
    total_memory_limit = sum(quota.get('memory_limit', 0) for quota in quotas.values())
    
    for namespace, quota in quotas.items():
        namespace_summary = get_formatted_quota_values(namespace, quota_file, include_requests)
        
        # Adicionar valores brutos para cálculos
        raw_values = {}
        if 'cpu_limit' in quota:
            raw_values['cpu_limit_raw'] = quota['cpu_limit']
        if 'memory_limit' in quota:
            raw_values['memory_limit_raw'] = quota['memory_limit']
        
        if include_requests:
            if 'cpu_request' in quota:
                raw_values['cpu_request_raw'] = quota['cpu_request']
            if 'memory_request' in quota:
                raw_values['memory_request_raw'] = quota['memory_request']
        
        # Adicionar percentuais em relação ao total quando solicitado
        if calculate_percentages:
            if 'cpu_limit' in quota and total_cpu_limit > 0:
                cpu_percent = (quota['cpu_limit'] / total_cpu_limit) * 100
                namespace_summary['cpu_percent'] = f"{cpu_percent:.1f}%"
                raw_values['cpu_percent_raw'] = cpu_percent
            
            if 'memory_limit' in quota and total_memory_limit > 0:
                memory_percent = (quota['memory_limit'] / total_memory_limit) * 100
                namespace_summary['memory_percent'] = f"{memory_percent:.1f}%"
                raw_values['memory_percent_raw'] = memory_percent
        
        # Adicionar proporção entre request e limit
        if include_requests:
            if 'cpu_request' in quota and 'cpu_limit' in quota and quota['cpu_limit'] > 0:
                req_pct = (quota['cpu_request'] / quota['cpu_limit']) * 100
                namespace_summary['cpu_req_vs_limit'] = f"{req_pct:.0f}%"
                
            if 'memory_request' in quota and 'memory_limit' in quota and quota['memory_limit'] > 0:
                req_pct = (quota['memory_request'] / quota['memory_limit']) * 100
                namespace_summary['memory_req_vs_limit'] = f"{req_pct:.0f}%"
        
        # Formatar para markdown se solicitado
        if use_markdown:
            format_dict = {}
            for key, value in namespace_summary.items():
                if key.endswith('_percent'):
                    # Destacar percentuais do total
                    format_dict[key] = f"**{value}**"
                elif key.endswith('_req_vs_limit'):
                    # Destacar proporções request/limit
                    format_dict[key] = f"*{value}*"
                else:
                    format_dict[key] = value
            namespace_summary.update(format_dict)
            
        # Mesclar os valores brutos ao resumo
        namespace_summary.update(raw_values)
        summary[namespace] = namespace_summary
        
    # Adicionar totais ao resumo
    if total_cpu_limit > 0 or total_memory_limit > 0:
        totals = {}
        
        if total_cpu_limit > 0:
            cpu_value, cpu_unit = convert_to_best_unit(total_cpu_limit, 'cpu')
            totals['cpu_limit'] = f"{cpu_value:.2f} {cpu_unit}"
            totals['cpu_limit_raw'] = total_cpu_limit
        
        if total_memory_limit > 0:
            mem_value, mem_unit = convert_to_best_unit(total_memory_limit, 'memory')
            totals['memory_limit'] = f"{mem_value:.2f} {mem_unit}"
            totals['memory_limit_raw'] = total_memory_limit
        
        # Formatar para markdown se solicitado
        if use_markdown:
            for key, value in totals.items():
                if isinstance(value, str) and not key.endswith('_raw'):
                    totals[key] = f"**{value}**"
        
        summary['__total__'] = totals
    
    return summary
