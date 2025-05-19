"""
Configuration file for the data analysis pipeline of the noisy neighbors experiment.
"""

# General settings
DEFAULT_DATA_DIR = "/home/phil/Projects/k8s-noisy-detection/demo-data/demo-experiment-3-rounds"
DEFAULT_OUTPUT_DIR = "output"

# Metric settings
DEFAULT_METRICS = [
    "cpu_usage",
    "memory_usage",
    "network_total_bandwidth"
]

# Visualization settings
VISUALIZATION_CONFIG = {
    "dpi": 300,
    "figure_width": 10,
    "figure_height": 6,
    "font_size": 12,
    "title_size": 16,
    "label_size": 14,
    "legend_size": 12
}

# Color mapping for tenants - Using Seaborn's "Paired" palette (colorblind-friendly)
TENANT_COLORS = {
    'tenant-a':      '#a6cee3',  # Light Blue
    'tenant-b':      '#1f78b4',  # Dark Blue
    'tenant-c':      '#b2df8a',  # Light Green
    'tenant-d':      '#33a02c',  # Dark Green
    'ingress-nginx': '#fb9a99',  # Light Red
    'unknown':       '#e31a1c',  # Dark Red
    'waiting':       '#fdbf6f',  # Light Orange
    'active':        '#ff7f00',  # Dark Orange
    # Additional colors from the "Paired" palette if needed:
    # '#cab2d6',  # Light Purple
    # '#6a3d9a',  # Dark Purple
    # '#ffff99',  # Light Yellow (use with caution)
    # '#b15928'   # Brown
}

# Tenant configured as a noisy neighbor
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

# Settings for statistical analysis
STATISTICAL_CONFIG = {
    "significance_level": 0.05,
    "effect_size_thresholds": {
        "small": 0.2,
        "medium": 0.5,
        "large": 0.8
    },
    "outlier_z_threshold": 3.0
}

# Settings for table export
TABLE_EXPORT_CONFIG = {
    "float_format": ".2f",
    "include_index": False,
    "longtable": False
}

# Settings for data aggregation
AGGREGATION_CONFIG = {
    "aggregation_keys": ["tenant", "phase"],  # Elements by which data will be grouped
    "elements_to_aggregate": None,  # Specific list of elements to focus on, e.g., ["tenant-a", "ingress-nginx"]. None for all.
    "metrics_for_aggregation": DEFAULT_METRICS # Metrics to consider in aggregation. Uses DEFAULT_METRICS if None.
}

# Node Resource Limit Settings
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

DEFAULT_NODE_CONFIG_NAME = "Default"  # Default node configuration if not specified

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
CAUSALITY_FIGURE_SIZE = (12, 10)  # Default figure size for causality graphs (width, height)

# Colors for metrics in causality graph (can be expanded)
# Using more formal colors for academic publications
CAUSALITY_METRIC_COLORS = {
    "cpu_usage": "#4472C4",      # Formal Blue
    "memory_usage": "#ED7D31",   # Formal Orange
    "network_total_bandwidth": "#70AD47", # Formal Green
    "disk_throughput_total": "#5B9BD5",  # Formal Light Blue
    "pod_restarts": "#7030A0",  # Formal Purple
    "oom_kills": "#C00000"      # Formal Red
}
