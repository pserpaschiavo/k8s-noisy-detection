# filepath: /home/phil/Projects/k8s-noisy-detection/pipeline/config.py
"""
Arquivo de configuração para o pipeline de análise de dados do experimento de noisy neighbors.
"""
import os  # Adicionado para os.path

# Configurações gerais
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "demo-data", "demo-experiment-3-rounds")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Configurações de métricas
DEFAULT_METRICS = [
    "cpu_usage",
    "memory_usage",
    "disk_throughput_total",
    "network_total_bandwidth"
]

# Configurações de visualização
VISUALIZATION_CONFIG = {
    "dpi": 300,
    "figure_width": 10,
    "figure_height": 6,
    "font_size": 12,
    "title_size": 16,
    "label_size": 14,
    "legend_size": 12
}

# Mapeamento de cores para tenants - Usando paleta "Paired" do Seaborn (colorblind-friendly)
TENANT_COLORS = {
    'tenant-a':      '#a6cee3',  # Light Blue
    'tenant-b':      '#1f78b4',  # Dark Blue
    'tenant-c':      '#b2df8a',  # Light Green
    'tenant-d':      '#33a02c',  # Dark Green
    'ingress-nginx': '#fb9a99',  # Light Red
    'unknown':       '#e31a1c',  # Dark Red
    'waiting':       '#fdbf6f',  # Light Orange
    'active':        '#ff7f00',  # Dark Orange
    # Cores adicionais da paleta "Paired" se necessário:
    # '#cab2d6',  # Light Purple
    # '#6a3d9a',  # Dark Purple
    # '#ffff99',  # Light Yellow (usar com cautela)
    # '#b15928'   # Brown
}

# Tenant configurado como gerador de ruído (noisy neighbor)
DEFAULT_NOISY_TENANT = 'tenant-b'

# Formatted names for metrics (for graph titles and tables)
METRIC_DISPLAY_NAMES = {
    "cpu_usage": "CPU Usage",
    "cpu_usage_variability": "CPU Usage Variability",
    "memory_usage": "Memory Usage",
    "disk_io_total": "Disk I/O Operations (ops/s)",
    "disk_throughput_total": "Disk Throughput (MB/s)",
    "network_receive": "Network Receive Traffic (MB/s)",
    "network_transmit": "Network Transmit Traffic (MB/s)",
    "network_total_bandwidth": "Total Network Bandwidth (MB/s)",
    "network_packet_rate": "Network Packet Rate (packets/s)",
    "network_error_rate": "Network Error Rate (%)",
    "network_dropped": "Dropped Packets (packets/s)",
    "pod_restarts": "Pod Restarts",
    "pod_ready_age": "Pod Uptime (minutes)",
    "oom_kills": "OOM Kill Events"
}

# Formatted names for phases
PHASE_DISPLAY_NAMES = {
    "1 - Baseline": "Baseline",
    "2 - Attack": "Attack",
    "3 - Recovery": "Recovery"
}

# Configurações para análise estatística
STATISTICAL_CONFIG = {
    "significance_level": 0.05,
    "effect_size_thresholds": {
        "small": 0.2,
        "medium": 0.5,
        "large": 0.8
    },
    "outlier_z_threshold": 3.0
}

# Configurações para exportação de tabelas
TABLE_EXPORT_CONFIG = {
    "float_format": ".2f",
    "include_index": False,
    "longtable": False
}

# Configurações para agregação de dados
AGGREGATION_CONFIG = {
    "aggregation_keys": ["tenant", "phase"],  # Elementos pelos quais os dados serão agrupados
    "elements_to_aggregate": None,  # Lista específica de elementos para focar, ex: ["tenant-a", "ingress-nginx"]. None para todos.
    "metrics_for_aggregation": DEFAULT_METRICS # Métricas a serem consideradas na agregação. Usa DEFAULT_METRICS se None.
}

# Configurações dos Limites de Recursos do Nó
NODE_RESOURCE_CONFIGS = {
    "Default": {
        "CPUS": 8,
        "MEMORY_GB": 16,
        "DISK_SIZE_GB": 40
    },
    "Limited": {
        "CPUS": 4,
        "MEMORY_GB": 8,
        "DISK_SIZE_GB": 30
    }
}

DEFAULT_NODE_CONFIG_NAME = "Default"  # Configuração padrão do nó se não especificada

# Configuration defaults for Impact Score Calculation
IMPACT_CALCULATION_DEFAULTS = {
    "baseline_phase_name": "Baseline",  # Match actual column name from pivot_df
    "attack_phase_name": "Attack",      # Match actual column name from pivot_df
    "recovery_phase_name": "Recovery",  # Match actual column name from pivot_df
    "metrics_for_impact_score": ["cpu_usage", "memory_usage", "network_total_bandwidth"]
}

# Defines attributes for metrics, e.g., whether higher values are better.
# This is imported and used directly by advanced_analysis.py for impact score calculation.
METRICS_CONFIG = {
    "cpu_usage": {"higher_is_better": False},
    "memory_usage": {"higher_is_better": False},
    "disk_throughput_total": {"higher_is_better": True},
    "network_total_bandwidth": {"higher_is_better": True},
    # Ensure all metrics listed in IMPACT_CALCULATION_DEFAULTS["metrics_for_impact_score"]
    # and any other metrics potentially used for impact scores are defined here.
}

# Weights for metrics in the Impact Score.
# This is imported and used directly by advanced_analysis.py.
IMPACT_SCORE_WEIGHTS = {
    "cpu_usage": 0.4,
    "memory_usage": 0.3,
    "network_total_bandwidth": 0.3
    # Ensure these correspond to metrics that might be included in the score.
}

# Configuration for Inter-Tenant Causality Analysis
DEFAULT_CAUSALITY_MAX_LAG = 5
DEFAULT_CAUSALITY_THRESHOLD_P_VALUE = 0.05
DEFAULT_METRICS_FOR_CAUSALITY = [
    "cpu_usage",
    "memory_usage",
    "network_total_bandwidth"
]

# Colors for metrics in causality graph (can be expanded)
# Usando cores mais formais para publicações acadêmicas
CAUSALITY_METRIC_COLORS = {
    "cpu_usage": "#4472C4",      # Azul formal
    "memory_usage": "#ED7D31",   # Laranja formal
    "network_total_bandwidth": "#70AD47", # Verde formal
    "disk_throughput_total": "#5B9BD5",  # Azul claro formal
    "pod_restarts": "#7030A0",  # Roxo formal
    "oom_kills": "#C00000"      # Vermelho formal
}
