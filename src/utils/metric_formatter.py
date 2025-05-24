"""
Módulo para formatação inteligente e detecção automática de unidades para métricas.

Este módulo fornece funcionalidades para:
- Detecção automática de unidades baseada na magnitude dos dados
- Conversão inteligente para unidades mais legíveis
- Preservação dos dados originais
- Formatação contextual baseada no tipo de métrica
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import re
import logging

# Configure logging
logger = logging.getLogger(__name__)


class MetricFormatter:
    """Classe para formatação inteligente de métricas com detecção automática de unidades."""
    
    def __init__(self):
        """Inicializa o formatador com definições de unidades."""
        # Unidades binárias (base 1024)
        self.binary_memory_units = [
            ('B', 1),
            ('KiB', 1024),
            ('MiB', 1024**2),
            ('GiB', 1024**3),
            ('TiB', 1024**4),
        ]
        
        # Unidades decimais (base 1000) 
        self.decimal_units = [
            ('B', 1),
            ('KB', 1000),
            ('MB', 1000**2),
            ('GB', 1000**3),
            ('TB', 1000**4),
        ]
        
        # Unidades de throughput/velocidade
        self.throughput_units = [
            ('B/s', 1),
            ('KB/s', 1000),
            ('MB/s', 1000**2),
            ('GB/s', 1000**3),
        ]
        
        # Unidades de rede (bits per second)
        self.network_units = [
            ('bps', 1),
            ('Kbps', 1000),
            ('Mbps', 1000**2),
            ('Gbps', 1000**3),
        ]
        
        # Padrões de métricas
        self.metric_patterns = {
            'memory': ['memory', 'mem', 'ram', 'cache'],
            'disk': ['disk', 'storage', 'volume', 'io', 'throughput'],
            'network': ['network', 'bandwidth', 'rx', 'tx', 'receive', 'transmit'],
            'cpu': ['cpu', 'processor', 'core'],
        }
    
    def detect_metric_type(self, metric_name: str) -> str:
        """
        Detecta o tipo de métrica baseado no nome.
        
        Args:
            metric_name: Nome da métrica
            
        Returns:
            Tipo de métrica detectado
        """
        name_lower = metric_name.lower()
        
        for metric_type, patterns in self.metric_patterns.items():
            if any(pattern in name_lower for pattern in patterns):
                return metric_type
        
        return 'generic'
    
    def detect_original_unit(self, values: pd.Series, metric_type: str) -> Tuple[str, float]:
        """
        Detecta a unidade original baseada na magnitude dos valores.
        
        Args:
            values: Série de valores
            metric_type: Tipo de métrica
            
        Returns:
            Tupla (unidade_detectada, fator_de_conversao_para_base)
        """
        # Remove valores nulos e zero
        clean_values = values.dropna()
        clean_values = clean_values[clean_values > 0]
        
        if len(clean_values) == 0:
            return 'unknown', 1.0
        
        # Usa percentil 75 para representar valores típicos
        typical_value = clean_values.quantile(0.75)
        median_value = clean_values.median()
        ref_value = typical_value if typical_value > 0 else median_value
        
        if metric_type == 'memory':
            return self._detect_memory_unit(ref_value)
        elif metric_type == 'disk':
            return self._detect_throughput_unit(ref_value)
        elif metric_type == 'network':
            return self._detect_network_unit(ref_value)
        else:
            return self._detect_generic_unit(ref_value)
    
    def _detect_memory_unit(self, ref_value: float) -> Tuple[str, float]:
        """Detecta unidade de memória baseada no valor de referência."""
        # Assume unidades binárias para memória
        for unit, factor in reversed(self.binary_memory_units):
            if ref_value >= factor * 0.5:  # Use 0.5 como threshold
                return unit, factor
        return 'B', 1
    
    def _detect_throughput_unit(self, ref_value: float) -> Tuple[str, float]:
        """Detecta unidade de throughput baseada no valor de referência."""
        # Assume unidades decimais para throughput
        for unit, factor in reversed(self.throughput_units):
            if ref_value >= factor * 0.5:
                return unit, factor
        return 'B/s', 1
    
    def _detect_network_unit(self, ref_value: float) -> Tuple[str, float]:
        """Detecta unidade de rede baseada no valor de referência."""
        # Para métricas de rede, pode ser bytes ou bits
        # Heurística: valores muito grandes provavelmente são bits
        if ref_value > 1000000:  # > 1M, provavelmente bits
            for unit, factor in reversed(self.network_units):
                if ref_value >= factor * 0.5:
                    return unit, factor
            return 'bps', 1
        else:  # Provavelmente bytes
            for unit, factor in reversed(self.throughput_units):
                if ref_value >= factor * 0.5:
                    return unit, factor
            return 'B/s', 1
    
    def _detect_generic_unit(self, ref_value: float) -> Tuple[str, float]:
        """Detecta unidade genérica baseada na magnitude."""
        if ref_value >= 1e12:
            return 'T', 1e12
        elif ref_value >= 1e9:
            return 'G', 1e9
        elif ref_value >= 1e6:
            return 'M', 1e6
        elif ref_value >= 1e3:
            return 'K', 1e3
        else:
            return '', 1
    
    def get_best_display_unit(self, values: pd.Series, metric_type: str) -> Tuple[str, float]:
        """
        Determina a melhor unidade para exibição baseada nos dados.
        
        Args:
            values: Série de valores
            metric_type: Tipo de métrica
            
        Returns:
            Tupla (melhor_unidade, fator_de_conversao)
        """
        clean_values = values.dropna()
        clean_values = clean_values[clean_values > 0]
        
        if len(clean_values) == 0:
            return 'unknown', 1.0
        
        # Usa percentil 75 para representar valores típicos após conversão
        typical_value = clean_values.quantile(0.75)
        
        if metric_type == 'memory':
            return self._get_best_memory_unit(typical_value)
        elif metric_type == 'disk':
            return self._get_best_throughput_unit(typical_value)
        elif metric_type == 'network':
            return self._get_best_network_unit(typical_value)
        else:
            return self._get_best_generic_unit(typical_value)
    
    def _get_best_memory_unit(self, typical_value: float) -> Tuple[str, float]:
        """Determina a melhor unidade de memória para exibição."""
        for unit, factor in reversed(self.binary_memory_units):
            converted_value = typical_value / factor
            if converted_value >= 1.0:  # Valor >= 1 na nova unidade
                return unit, factor
        return 'B', 1
    
    def _get_best_throughput_unit(self, typical_value: float) -> Tuple[str, float]:
        """Determina a melhor unidade de throughput para exibição."""
        for unit, factor in reversed(self.throughput_units):
            converted_value = typical_value / factor
            if converted_value >= 1.0:
                return unit, factor
        return 'B/s', 1
    
    def _get_best_network_unit(self, typical_value: float) -> Tuple[str, float]:
        """Determina a melhor unidade de rede para exibição."""
        # Decide entre bits e bytes baseado na magnitude
        if typical_value > 1000000:  # Provavelmente bits
            for unit, factor in reversed(self.network_units):
                converted_value = typical_value / factor
                if converted_value >= 1.0:
                    return unit, factor
            return 'bps', 1
        else:  # Provavelmente bytes
            for unit, factor in reversed(self.throughput_units):
                converted_value = typical_value / factor
                if converted_value >= 1.0:
                    return unit, factor
            return 'B/s', 1
    
    def _get_best_generic_unit(self, typical_value: float) -> Tuple[str, float]:
        """Determina a melhor unidade genérica para exibição."""
        if typical_value >= 1e12:
            return 'T', 1e12
        elif typical_value >= 1e9:
            return 'G', 1e9
        elif typical_value >= 1e6:
            return 'M', 1e6
        elif typical_value >= 1e3:
            return 'K', 1e3
        else:
            return '', 1
    
    def format_dataframe(self, df: pd.DataFrame, metric_name: str, 
                        value_col: str = 'value',
                        preserve_original: bool = True) -> pd.DataFrame:
        """
        Formata DataFrame com detecção automática de unidades e conversão inteligente.
        
        Args:
            df: DataFrame com dados da métrica
            metric_name: Nome da métrica
            value_col: Nome da coluna com valores
            preserve_original: Se deve preservar os valores originais
            
        Returns:
            DataFrame formatado com colunas adicionais
        """
        if df.empty or value_col not in df.columns:
            logger.warning(f"DataFrame vazio ou coluna '{value_col}' não encontrada para métrica '{metric_name}'")
            return df.copy()
        
        df_formatted = df.copy()
        
        # Preservar dados originais se solicitado
        if preserve_original and 'original_value' not in df_formatted.columns:
            df_formatted['original_value'] = df_formatted[value_col].copy()
        
        # Detectar tipo de métrica
        metric_type = self.detect_metric_type(metric_name)
        df_formatted['metric_type'] = metric_type
        
        # Detectar unidade original
        original_unit, original_factor = self.detect_original_unit(df_formatted[value_col], metric_type)
        df_formatted['detected_original_unit'] = original_unit
        df_formatted['original_unit_factor'] = original_factor
        
        # Determinar melhor unidade para exibição
        display_unit, display_factor = self.get_best_display_unit(df_formatted[value_col], metric_type)
        
        # Aplicar conversão para melhor unidade de exibição
        df_formatted[value_col] = df_formatted[value_col] / display_factor
        df_formatted['original_unit'] = original_unit
        df_formatted['display_unit'] = display_unit
        df_formatted['formatted_value'] = df_formatted.apply(
            lambda row: self._format_value_with_unit(row[value_col], row['display_unit']),
            axis=1
        )
        
        # Metadados para conversão
        df_formatted['conversion_info'] = f"Original: {original_unit} -> Display: {display_unit}"
        
        logger.info(f"Formatted metric '{metric_name}': {original_unit} -> {display_unit} (type: {metric_type})")
        
        return df_formatted
    
    def _format_value_with_unit(self, value: float, unit: str) -> str:
        """Formata valor com unidade de forma humanizada."""
        if pd.isna(value):
            return 'N/A'
        
        # Determinar número de casas decimais baseado na magnitude
        if abs(value) >= 100:
            decimal_places = 1
        elif abs(value) >= 10:
            decimal_places = 2
        else:
            decimal_places = 3
        
        return f"{value:.{decimal_places}f} {unit}"
    
    def format_multiple_metrics(self, metrics_dict: Dict[str, pd.DataFrame],
                               value_col: str = 'value',
                               preserve_original: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Formata múltiplas métricas de uma vez.
        
        Args:
            metrics_dict: Dicionário com métricas
            value_col: Nome da coluna com valores
            preserve_original: Se deve preservar valores originais
            
        Returns:
            Dicionário com métricas formatadas
        """
        formatted_metrics = {}
        
        for metric_name, df in metrics_dict.items():
            try:
                formatted_df = self.format_dataframe(
                    df, metric_name, value_col, preserve_original
                )
                formatted_metrics[metric_name] = formatted_df
            except Exception as e:
                logger.error(f"Erro ao formatar métrica '{metric_name}': {e}")
                formatted_metrics[metric_name] = df.copy()  # Retorna original em caso de erro
        
        return formatted_metrics


def detect_and_convert_units(df: pd.DataFrame, metric_name: str, 
                           value_col: str = 'value') -> pd.DataFrame:
    """
    Função utilitária para detecção automática e conversão de unidades.
    
    Args:
        df: DataFrame com dados
        metric_name: Nome da métrica
        value_col: Nome da coluna com valores
        
    Returns:
        DataFrame com unidades detectadas e convertidas
    """
    formatter = MetricFormatter()
    return formatter.format_dataframe(df, metric_name, value_col)


def remove_hardcoded_conversions(df: pd.DataFrame, 
                                metric_name: str,
                                problematic_metrics: List[str] = None) -> pd.DataFrame:
    """
    Remove conversões hard-coded problemáticas e aplica formatação inteligente.
    
    Args:
        df: DataFrame possivelmente com conversões problemáticas
        metric_name: Nome da métrica
        problematic_metrics: Lista de métricas que podem ter conversões problemáticas
        
    Returns:
        DataFrame com formatação corrigida
    """
    if problematic_metrics is None:
        problematic_metrics = ['memory_usage', 'disk_throughput_total', 'network_total_bandwidth']
    
    # Se esta métrica estava na lista de métricas com conversões problemáticas,
    # assumir que os valores podem ter sido incorretamente convertidos
    if metric_name in problematic_metrics:
        logger.warning(f"Métrica '{metric_name}' pode ter conversões hard-coded problemáticas")
        
        # Adicionar flag indicando possível problema
        df = df.copy()
        df['potential_conversion_issue'] = True
        df['conversion_warning'] = "This metric may have been incorrectly converted from bytes to MB"
    
    # Aplicar formatação inteligente
    formatter = MetricFormatter()
    return formatter.format_dataframe(df, metric_name)


# Funções de conveniência para casos específicos
def format_memory_metric(df: pd.DataFrame, metric_name: str = "memory_usage") -> pd.DataFrame:
    """Formata especificamente métricas de memória."""
    formatter = MetricFormatter()
    return formatter.format_dataframe(df, metric_name)


def format_throughput_metric(df: pd.DataFrame, metric_name: str = "throughput") -> pd.DataFrame:
    """Formata especificamente métricas de throughput."""
    formatter = MetricFormatter()
    return formatter.format_dataframe(df, metric_name)


def format_network_metric(df: pd.DataFrame, metric_name: str = "network_bandwidth") -> pd.DataFrame:
    """Formata especificamente métricas de rede."""
    formatter = MetricFormatter()
    return formatter.format_dataframe(df, metric_name)
