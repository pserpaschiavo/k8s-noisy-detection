"""
Módulo consolidado para normalização de dados.

Este módulo fornece funções para:
- Normalização de métricas de recursos (conversão para percentuais)
- Normalização temporal (tempo decorrido, timestamps)
- Processamento e formatação de dados experimentais
- Formatação inteligente de unidades com detecção automática
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from ..utils.common import logger
from ..utils.metric_formatter import MetricFormatter, detect_and_convert_units


# ==================== NORMALIZAÇÃO E FORMATAÇÃO DE MÉTRICAS ====================

def normalize_cpu_usage(df: pd.DataFrame, total_cpu_cores: int) -> pd.DataFrame:
    """
    Normaliza o uso de CPU em relação ao total de cores disponíveis.
    
    Args:
        df: DataFrame com dados de CPU
        total_cpu_cores: Número total de cores de CPU no cluster
        
    Returns:
        DataFrame com CPU normalizada em percentual
    """
    result = df.copy()
    
    if 'cpu_usage' in result.columns:
        result['cpu_usage_percent'] = (result['cpu_usage'] / total_cpu_cores) * 100
        
    if 'cpu_requests' in result.columns:
        result['cpu_requests_percent'] = (result['cpu_requests'] / total_cpu_cores) * 100
        
    if 'cpu_limits' in result.columns:
        result['cpu_limits_percent'] = (result['cpu_limits'] / total_cpu_cores) * 100
        
    return result


def normalize_memory_usage(df: pd.DataFrame, total_memory_gb: float) -> pd.DataFrame:
    """
    Normaliza o uso de memória em relação ao total disponível.
    
    Args:
        df: DataFrame com dados de memória
        total_memory_gb: Total de memória em GB no cluster
        
    Returns:
        DataFrame com memória normalizada em percentual
    """
    result = df.copy()
    
    if 'memory_usage' in result.columns:
        result['memory_usage_percent'] = (result['memory_usage'] / total_memory_gb) * 100
        
    if 'memory_requests' in result.columns:
        result['memory_requests_percent'] = (result['memory_requests'] / total_memory_gb) * 100
        
    if 'memory_limits' in result.columns:
        result['memory_limits_percent'] = (result['memory_limits'] / total_memory_gb) * 100
        
    return result


def normalize_metrics(df: pd.DataFrame, cluster_config: Dict) -> pd.DataFrame:
    """
    Aplica normalização completa de métricas de recursos.
    
    Args:
        df: DataFrame com métricas brutas
        cluster_config: Configuração do cluster com totais de recursos
        
    Returns:
        DataFrame com todas as métricas normalizadas
    """
    result = df.copy()
    
    # CPU normalization
    if 'total_cpu_cores' in cluster_config:
        result = normalize_cpu_usage(result, cluster_config['total_cpu_cores'])
        
    # Memory normalization  
    if 'total_memory_gb' in cluster_config:
        result = normalize_memory_usage(result, cluster_config['total_memory_gb'])
        
    logger.info(f"Métricas normalizadas para {len(result)} registros")
    return result


# ==================== NORMALIZAÇÃO TEMPORAL ====================

def add_elapsed_time(df: pd.DataFrame, group_by: List[str] = ['round', 'phase']) -> pd.DataFrame:
    """
    Adiciona colunas de tempo decorrido desde o início de cada grupo.
    
    Args:
        df: DataFrame com dados do experimento
        group_by: Colunas para agrupar ao calcular tempo inicial
        
    Returns:
        DataFrame com colunas adicionais de tempo decorrido
    """
    result = df.copy()
    
    if 'datetime' not in result.columns:
        logger.warning("Coluna 'datetime' não encontrada para cálculo de tempo decorrido")
        return result
    
    # Encontrar o timestamp inicial para cada grupo
    start_times = result.groupby(group_by)['datetime'].min().reset_index()
    start_times = start_times.rename(columns={'datetime': 'start_time'})
    
    # Mesclar com o DataFrame original
    result = result.merge(start_times, on=group_by, how='left')
    
    # Calcular tempo decorrido em segundos
    result['elapsed_seconds'] = (result['datetime'] - result['start_time']).dt.total_seconds()
    result['elapsed_minutes'] = result['elapsed_seconds'] / 60
    
    # Remover coluna temporária
    result = result.drop('start_time', axis=1)
    
    logger.info(f"Tempo decorrido calculado para {len(result)} registros")
    return result


def normalize_timestamps(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Normaliza timestamps para datetime objects.
    
    Args:
        df: DataFrame com timestamps
        timestamp_col: Nome da coluna de timestamp
        
    Returns:
        DataFrame com timestamps normalizados
    """
    result = df.copy()
    
    if timestamp_col in result.columns:
        result['datetime'] = pd.to_datetime(result[timestamp_col])
        result = result.sort_values('datetime')
        
    logger.info(f"Timestamps normalizados para {len(result)} registros")
    return result


def add_time_features(df: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
    """
    Adiciona features temporais derivadas (hora, dia da semana, etc.).
    
    Args:
        df: DataFrame com coluna datetime
        datetime_col: Nome da coluna datetime
        
    Returns:
        DataFrame com features temporais adicionais
    """
    result = df.copy()
    
    if datetime_col in result.columns:
        result['hour'] = result[datetime_col].dt.hour
        result['day_of_week'] = result[datetime_col].dt.dayofweek
        result['minute'] = result[datetime_col].dt.minute
        
    return result


# ==================== UTILITÁRIOS DE PROCESSAMENTO ====================

def clean_metrics_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa dados de métricas removendo valores inválidos.
    
    Args:
        df: DataFrame com métricas
        
    Returns:
        DataFrame limpo
    """
    result = df.copy()
    
    # Remover valores negativos de recursos
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if any(keyword in col.lower() for keyword in ['cpu', 'memory', 'usage']):
            result = result[result[col] >= 0]
    
    # Remover NaN em colunas críticas
    critical_cols = ['datetime', 'tenant', 'namespace']
    for col in critical_cols:
        if col in result.columns:
            result = result.dropna(subset=[col])
    
    logger.info(f"Dados limpos: {len(result)} registros restantes")
    return result


def aggregate_by_time_window(df: pd.DataFrame, window: str = '1min', 
                           agg_func: str = 'mean') -> pd.DataFrame:
    """
    Agrega dados por janela temporal.
    
    Args:
        df: DataFrame com timestamp
        window: Janela temporal (ex: '1min', '5min', '1h')
        agg_func: Função de agregação ('mean', 'max', 'sum')
        
    Returns:
        DataFrame agregado
    """
    if 'datetime' not in df.columns:
        logger.error("Coluna 'datetime' necessária para agregação temporal")
        return df
    
    # Definir colunas numéricas para agregação
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Agrupar por janela temporal e agregar
    result = df.set_index('datetime').groupby(pd.Grouper(freq=window))[numeric_cols].agg(agg_func).reset_index()
    
    logger.info(f"Dados agregados em janelas de {window}: {len(result)} pontos")
    return result


# ==================== FUNÇÕES DE ALTO NÍVEL ====================

def full_normalization_pipeline(df: pd.DataFrame, cluster_config: Dict, 
                              time_window: Optional[str] = None) -> pd.DataFrame:
    """
    Pipeline completo de normalização de dados.
    
    Args:
        df: DataFrame bruto
        cluster_config: Configuração do cluster
        time_window: Janela temporal para agregação (opcional)
        
    Returns:
        DataFrame completamente normalizado
    """
    logger.info("Iniciando pipeline de normalização completo")
    
    # 1. Limpeza inicial
    result = clean_metrics_data(df)
    
    # 2. Normalização temporal
    result = normalize_timestamps(result)
    result = add_elapsed_time(result)
    result = add_time_features(result)
    
    # 3. Normalização de métricas
    result = normalize_metrics(result, cluster_config)
    
    # 4. Agregação temporal (opcional)
    if time_window:
        result = aggregate_by_time_window(result, time_window)
    
    logger.info(f"Pipeline de normalização concluído: {len(result)} registros finais")
    return result


def normalize_metrics_intelligent(df: pd.DataFrame, 
                                cluster_config: Dict,
                                preserve_original: bool = True,
                                auto_format: bool = True) -> pd.DataFrame:
    """
    Aplica normalização inteligente com detecção automática de unidades.
    
    Esta função substitui as normalizações hard-coded por um sistema inteligente
    que preserva dados originais e aplica formatação contextual.
    
    Args:
        df: DataFrame com métricas brutas
        cluster_config: Configuração do cluster com totais de recursos
        preserve_original: Se deve preservar os valores originais
        auto_format: Se deve aplicar formatação automática de unidades
        
    Returns:
        DataFrame com métricas normalizadas e formatadas
    """
    if df.empty:
        logger.warning("DataFrame vazio fornecido para normalização")
        return df.copy()
    
    result = df.copy()
    formatter = MetricFormatter()
    
    # Identificar colunas de métricas no DataFrame
    metric_columns = []
    
    # Procurar por colunas que podem conter métricas
    possible_metric_cols = ['cpu_usage', 'memory_usage', 'disk_throughput_total', 
                           'network_total_bandwidth', 'disk_io_total', 'value']
    
    for col in possible_metric_cols:
        if col in result.columns:
            metric_columns.append(col)
    
    # Se há uma coluna 'metric' ou similar, usar para determinar tipos
    metric_name_col = None
    for col in ['metric', 'metric_name', 'metric_type']:
        if col in result.columns:
            metric_name_col = col
            break
    
    if metric_name_col and len(metric_columns) > 0:
        # Processar por grupos de métricas
        for metric_name, group in result.groupby(metric_name_col):
            for metric_col in metric_columns:
                if metric_col in group.columns:
                    # Aplicar formatação inteligente
                    if auto_format:
                        formatted_group = formatter.format_dataframe(
                            group, str(metric_name), metric_col, preserve_original
                        )
                        # Atualizar resultado com dados formatados
                        mask = result[metric_name_col] == metric_name
                        for new_col in formatted_group.columns:
                            if new_col not in result.columns:
                                result[new_col] = None
                            result.loc[mask, new_col] = formatted_group[new_col].values
    else:
        # Processar colunas individuais se não há agrupamento por métrica
        for metric_col in metric_columns:
            if auto_format:
                # Inferir nome da métrica da coluna
                inferred_metric_name = metric_col.replace('_', ' ').title()
                formatted_df = formatter.format_dataframe(
                    result, inferred_metric_name, metric_col, preserve_original
                )
                # Mesclar colunas formatadas
                for new_col in formatted_df.columns:
                    if new_col not in result.columns:
                        result[new_col] = formatted_df[new_col]
    
    # Aplicar normalizações específicas baseadas na configuração do cluster
    if auto_format and cluster_config:
        result = _apply_cluster_normalization(result, cluster_config)
    
    logger.info(f"Normalized {len(metric_columns)} metric columns with intelligent formatting")
    return result


def _apply_cluster_normalization(df: pd.DataFrame, cluster_config: Dict) -> pd.DataFrame:
    """
    Aplica normalizações específicas baseadas na configuração do cluster.
    
    Args:
        df: DataFrame com métricas formatadas
        cluster_config: Configuração do cluster
        
    Returns:
        DataFrame com normalizações adicionais aplicadas
    """
    result = df.copy()
    
    # Normalizar CPU se disponível
    if 'cpu_cores' in cluster_config or 'total_cpu_cores' in cluster_config:
        total_cores = cluster_config.get('cpu_cores', cluster_config.get('total_cpu_cores'))
        if total_cores and 'cpu_usage' in result.columns:
            result = normalize_cpu_usage(result, total_cores)
    
    # Normalizar memória se disponível
    if 'memory_gb' in cluster_config or 'total_memory_gb' in cluster_config:
        total_memory = cluster_config.get('memory_gb', cluster_config.get('total_memory_gb'))
        if total_memory and 'memory_usage' in result.columns:
            result = normalize_memory_usage(result, total_memory)
    
    return result


def fix_hardcoded_conversions(metrics_dict: Dict[str, pd.DataFrame],
                             problematic_metrics: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Corrige conversões hard-coded problemáticas em múltiplas métricas.
    
    Esta função identifica e corrige métricas que podem ter sido incorretamente
    convertidas por divisões hard-coded como (1024 * 1024).
    
    Args:
        metrics_dict: Dicionário com métricas {nome_metrica: DataFrame}
        problematic_metrics: Lista de métricas conhecidas com problemas
        
    Returns:
        Dicionário com métricas corrigidas
    """
    if problematic_metrics is None:
        problematic_metrics = ['memory_usage', 'disk_throughput_total', 'network_total_bandwidth']
    
    formatter = MetricFormatter()
    corrected_metrics = {}
    
    for metric_name, df in metrics_dict.items():
        try:
            if metric_name in problematic_metrics:
                logger.warning(f"Applying correction for potentially problematic metric: {metric_name}")
                
                # Adicionar flags de aviso sobre possíveis problemas
                df_corrected = df.copy()
                if 'value' in df_corrected.columns:
                    df_corrected['_conversion_warning'] = (
                        "This metric may have been incorrectly converted. "
                        "Check if values seem too small (may need *1024*1024)"
                    )
                    df_corrected['_original_suspected_unit'] = 'MB (from hard-coded conversion)'
                
                # Aplicar formatação inteligente
                df_formatted = formatter.format_dataframe(df_corrected, metric_name)
                corrected_metrics[metric_name] = df_formatted
            else:
                # Aplicar formatação inteligente normal
                df_formatted = formatter.format_dataframe(df, metric_name)
                corrected_metrics[metric_name] = df_formatted
                
        except Exception as e:
            logger.error(f"Error correcting metric {metric_name}: {e}")
            corrected_metrics[metric_name] = df.copy()  # Preserva original em caso de erro
    
    return corrected_metrics


def validate_metric_conversions(df: pd.DataFrame, metric_name: str) -> Dict[str, any]:
    """
    Valida se as conversões de unidades de uma métrica estão corretas.
    
    Args:
        df: DataFrame com a métrica
        metric_name: Nome da métrica
        
    Returns:
        Dicionário com informações de validação
    """
    validation_result = {
        'metric_name': metric_name,
        'has_issues': False,
        'issues': [],
        'recommendations': [],
        'statistics': {}
    }
    
    if 'value' not in df.columns or df.empty:
        validation_result['issues'].append("No 'value' column found or empty DataFrame")
        validation_result['has_issues'] = True
        return validation_result
    
    values = df['value'].dropna()
    if len(values) == 0:
        validation_result['issues'].append("No valid values found")
        validation_result['has_issues'] = True
        return validation_result
    
    # Estatísticas básicas
    validation_result['statistics'] = {
        'count': len(values),
        'mean': values.mean(),
        'median': values.median(),
        'min': values.min(),
        'max': values.max(),
        'std': values.std()
    }
    
    # Verificar se valores são suspeitosamente pequenos para o tipo de métrica
    formatter = MetricFormatter()
    metric_type = formatter.detect_metric_type(metric_name)
    
    median_val = values.median()
    
    if metric_type == 'memory' and median_val < 100:
        validation_result['issues'].append(
            f"Memory values seem unusually small (median: {median_val:.2f}). "
            "May have been incorrectly converted to MB."
        )
        validation_result['recommendations'].append(
            "Consider checking if original values were in bytes and incorrectly divided by 1024*1024"
        )
        validation_result['has_issues'] = True
    
    if metric_type == 'disk' and median_val < 10:
        validation_result['issues'].append(
            f"Disk throughput values seem unusually small (median: {median_val:.2f}). "
            "May have been incorrectly converted."
        )
        validation_result['has_issues'] = True
    
    if metric_type == 'network' and median_val < 1:
        validation_result['issues'].append(
            f"Network values seem unusually small (median: {median_val:.2f}). "
            "May have been incorrectly converted."
        )
        validation_result['has_issues'] = True
    
    # Verificar se há valores Zero ou negativos em excesso
    zero_negative_count = len(values[values <= 0])
    if zero_negative_count > len(values) * 0.1:  # Mais de 10%
        validation_result['issues'].append(
            f"High percentage of zero/negative values: {zero_negative_count/len(values)*100:.1f}%"
        )
        validation_result['has_issues'] = True
    
    return validation_result
