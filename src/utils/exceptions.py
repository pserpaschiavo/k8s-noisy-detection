"""
Custom exception classes for the k8s-noisy-detection project.

This module provides standardized exception handling across the entire
codebase, making error handling more consistent and informative.
"""

import logging
from typing import Optional, Dict, Any


class K8sNoisyDetectionError(Exception):
    """Base exception class for all k8s-noisy-detection errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        
        # Log the error automatically
        logger = logging.getLogger(__name__)
        logger.error(f"[{self.error_code or 'UNKNOWN'}] {self.message}")
        if self.context:
            logger.error(f"Error context: {self.context}")


class DataLoadingError(K8sNoisyDetectionError):
    """Raised when data loading operations fail."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 metric_name: Optional[str] = None):
        context = {}
        if file_path:
            context['file_path'] = file_path
        if metric_name:
            context['metric_name'] = metric_name
            
        super().__init__(message, "DATA_LOADING_ERROR", context)


class DataValidationError(K8sNoisyDetectionError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, data_type: Optional[str] = None,
                 validation_rule: Optional[str] = None):
        context = {}
        if data_type:
            context['data_type'] = data_type
        if validation_rule:
            context['validation_rule'] = validation_rule
            
        super().__init__(message, "DATA_VALIDATION_ERROR", context)


class AnalysisError(K8sNoisyDetectionError):
    """Raised when analysis operations fail."""
    
    def __init__(self, message: str, analysis_type: Optional[str] = None,
                 metric_name: Optional[str] = None, phase: Optional[str] = None):
        context = {}
        if analysis_type:
            context['analysis_type'] = analysis_type
        if metric_name:
            context['metric_name'] = metric_name
        if phase:
            context['phase'] = phase
            
        super().__init__(message, "ANALYSIS_ERROR", context)


class VisualizationError(K8sNoisyDetectionError):
    """Raised when visualization operations fail."""
    
    def __init__(self, message: str, plot_type: Optional[str] = None,
                 output_path: Optional[str] = None):
        context = {}
        if plot_type:
            context['plot_type'] = plot_type
        if output_path:
            context['output_path'] = output_path
            
        super().__init__(message, "VISUALIZATION_ERROR", context)


class ConfigurationError(K8sNoisyDetectionError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: Optional[str] = None,
                 config_file: Optional[str] = None):
        context = {}
        if config_key:
            context['config_key'] = config_key
        if config_file:
            context['config_file'] = config_file
            
        super().__init__(message, "CONFIGURATION_ERROR", context)


class MetricFormattingError(K8sNoisyDetectionError):
    """Raised when metric formatting operations fail."""
    
    def __init__(self, message: str, metric_type: Optional[str] = None,
                 unit_detection: Optional[str] = None):
        context = {}
        if metric_type:
            context['metric_type'] = metric_type
        if unit_detection:
            context['unit_detection'] = unit_detection
            
        super().__init__(message, "METRIC_FORMATTING_ERROR", context)


class CausalityAnalysisError(AnalysisError):
    """Raised when SEM or causality analysis fails."""
    
    def __init__(self, message: str, model_spec: Optional[str] = None,
                 sem_error: Optional[str] = None):
        context = {'model_spec': model_spec} if model_spec else {}
        if sem_error:
            context['sem_error'] = sem_error
            
        super().__init__(message, "SEM_CAUSALITY", context=context)


class RootCauseAnalysisError(AnalysisError):
    """Raised when root cause analysis fails."""
    
    def __init__(self, message: str, order: Optional[int] = None,
                 tenant_names: Optional[list] = None):
        context = {}
        if order:
            context['analysis_order'] = order
        if tenant_names:
            context['tenant_count'] = len(tenant_names)
            
        super().__init__(message, "ROOT_CAUSE_ANALYSIS", context=context)


class MultivariateAnalysisError(AnalysisError):
    """Raised when multivariate analysis (PCA/ICA) fails."""
    
    def __init__(self, message: str, method: Optional[str] = None,
                 n_components: Optional[int] = None):
        context = {}
        if method:
            context['method'] = method
        if n_components:
            context['n_components'] = n_components
            
        super().__init__(message, "MULTIVARIATE_ANALYSIS", context=context)


# Error handling utilities

def handle_critical_error(error: Exception, operation: str) -> None:
    """
    Handle critical errors that should terminate the application.
    
    Args:
        error: The exception that occurred
        operation: Description of the operation that failed
    """
    logger = logging.getLogger(__name__)
    
    if isinstance(error, K8sNoisyDetectionError):
        logger.critical(f"Critical error during {operation}: {error.message}")
        if error.context:
            logger.critical(f"Error context: {error.context}")
    else:
        logger.critical(f"Critical error during {operation}: {str(error)}")
        logger.critical(f"Error type: {type(error).__name__}")
    
    # In a real application, you might want to send notifications,
    # cleanup resources, or perform other critical error handling
    

def handle_recoverable_error(error: Exception, operation: str, 
                           default_return=None) -> Any:
    """
    Handle recoverable errors that allow the application to continue.
    
    Args:
        error: The exception that occurred
        operation: Description of the operation that failed
        default_return: Default value to return on error
        
    Returns:
        The default_return value or None
    """
    logger = logging.getLogger(__name__)
    
    if isinstance(error, K8sNoisyDetectionError):
        logger.warning(f"Recoverable error during {operation}: {error.message}")
        if error.context:
            logger.warning(f"Error context: {error.context}")
    else:
        logger.warning(f"Recoverable error during {operation}: {str(error)}")
        logger.warning(f"Error type: {type(error).__name__}")
    
    return default_return


def validate_data_requirements(df, required_columns: list, 
                             operation: str) -> None:
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        operation: Description of the operation requiring the data
        
    Raises:
        DataValidationError: If validation fails
    """
    if df is None or df.empty:
        raise DataValidationError(
            f"Empty or None DataFrame provided for {operation}",
            data_type="DataFrame",
            validation_rule="non_empty"
        )
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns for {operation}: {missing_columns}",
            data_type="DataFrame",
            validation_rule="required_columns"
        )
    
    # Check for sufficient data
    if len(df) < 2:
        raise DataValidationError(
            f"Insufficient data for {operation}: need at least 2 rows, got {len(df)}",
            data_type="DataFrame",
            validation_rule="minimum_rows"
        )
