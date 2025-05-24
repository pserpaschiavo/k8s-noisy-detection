"""
Common utilities and centralized imports for the k8s-noisy-detection project.

This module provides centralized imports and utility functions to reduce
redundancy across the codebase and ensure consistent configuration.
"""

import logging
import warnings
from typing import Dict, List, Optional, Union, Any

# Core scientific computing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Additional scientific libraries
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Configure pandas for better display
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Configure seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Suppress common warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


# ==================== UTILITY CLASSES ====================

class DataValidationError(Exception):
    """Exception raised for data validation errors."""
    pass


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass


class AnalysisError(Exception):
    """Exception raised for analysis errors."""
    pass


# ==================== UTILITY FUNCTIONS ====================

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    Validate a DataFrame for basic requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if valid
        
    Raises:
        DataValidationError: If validation fails
    """
    if df is None or df.empty:
        raise DataValidationError("DataFrame is None or empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"DataFrame validated: {len(df)} rows, {len(df.columns)} columns")
    return True


def safe_divide(numerator: Union[float, np.ndarray], 
                denominator: Union[float, np.ndarray], 
                default: float = 0.0) -> Union[float, np.ndarray]:
    """
    Safely divide two numbers/arrays, handling division by zero.
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default: Default value for division by zero
        
    Returns:
        Result of division or default value
    """
    if isinstance(denominator, (int, float)):
        return numerator / denominator if denominator != 0 else default
    else:
        return np.divide(numerator, denominator, 
                        out=np.full_like(numerator, default), 
                        where=denominator!=0)


def ensure_directory(path: str) -> str:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Absolute path to the directory
    """
    import os
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def format_bytes(bytes_value: Union[int, float], decimal_places: int = 2) -> str:
    """
    Format bytes value to human readable string.
    
    Args:
        bytes_value: Value in bytes
        decimal_places: Number of decimal places
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.{decimal_places}f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.{decimal_places}f} PB"


def get_memory_usage() -> str:
    """
    Get current memory usage of the process.
    
    Returns:
        Formatted memory usage string
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return format_bytes(memory_info.rss)


def timer_decorator(func):
    """
    Decorator to time function execution.
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    
    return wrapper


# ==================== CONSTANTS ====================

# Common date/time formats
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Common statistical thresholds
CORRELATION_THRESHOLD = 0.7
P_VALUE_THRESHOLD = 0.05
OUTLIER_Z_THRESHOLD = 3.0

# Color palettes for consistent visualization
DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

TENANT_COLOR_PALETTE = {
    'default': '#1f77b4',
    'tenant_a': '#ff7f0e', 
    'tenant_b': '#2ca02c',
    'tenant_c': '#d62728',
    'tenant_d': '#9467bd'
}

# Export commonly used objects
__all__ = [
    # Core libraries
    'pd', 'np', 'plt', 'sns', 'stats',
    # ML libraries  
    'StandardScaler', 'MinMaxScaler', 'PCA', 'FastICA', 'TSNE',
    # Logging
    'logger',
    # Exceptions
    'DataValidationError', 'ConfigurationError', 'AnalysisError',
    # Utility functions
    'validate_dataframe', 'safe_divide', 'ensure_directory', 
    'format_bytes', 'get_memory_usage', 'timer_decorator',
    # Constants
    'DATE_FORMAT', 'DATETIME_FORMAT', 'TIMESTAMP_FORMAT',
    'CORRELATION_THRESHOLD', 'P_VALUE_THRESHOLD', 'OUTLIER_Z_THRESHOLD',
    'DEFAULT_COLORS', 'TENANT_COLOR_PALETTE'
]