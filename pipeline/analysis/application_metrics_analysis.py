"""
Module for analyzing application-level metrics in noisy neighbor scenarios.

This module extends the analysis beyond infrastructure metrics,
incorporating application metrics such as latency, error rate, and throughput.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats

def analyze_latency_impact(app_metrics: Dict[str, pd.DataFrame], 
                         infrastructure_metrics: Dict[str, pd.DataFrame],
                         noisy_tenant: str) -> Dict[str, Dict[str, float]]:
    """
    Analyzes the impact of the noisy tenant on the application latencies of other tenants.
    
    Args:
        app_metrics: Dictionary with application metrics (latency, errors, etc.).
        infrastructure_metrics: Dictionary with infrastructure metrics.
        noisy_tenant: The tenant identified as noisy.
        
    Returns:
        Dictionary with latency impact analysis for each tenant.
    """
    impact_analysis = {}
    
    if 'latency' not in app_metrics:
        return impact_analysis
        
    latency_df = app_metrics['latency']
    
    # Separate by phases
    baseline_df = latency_df[latency_df['phase'].str.contains('Baseline', case=False, na=False)]
    attack_df = latency_df[latency_df['phase'].str.contains('Attack', case=False, na=False)]
    
    # Calculate impact per tenant
    for tenant in latency_df['tenant'].unique():
        if tenant == noisy_tenant:
            continue
            
        # Latency in baseline
        tenant_baseline_latency = baseline_df[baseline_df['tenant'] == tenant]['value'].mean()
        
        # Latency in attack phase
        tenant_attack_latency = attack_df[attack_df['tenant'] == tenant]['value'].mean()
        
        # Calculate percentage change
        if tenant_baseline_latency > 0:
            latency_increase_pct = ((tenant_attack_latency - tenant_baseline_latency) / tenant_baseline_latency) * 100
        else:
            latency_increase_pct = 0 if tenant_attack_latency == tenant_baseline_latency else float('inf') # Handle division by zero or baseline of zero
            
        # Perform statistical significance test (e.g., t-test)
        baseline_values = baseline_df[baseline_df['tenant'] == tenant]['value'].dropna()
        attack_values = attack_df[attack_df['tenant'] == tenant]['value'].dropna()

        p_value = float('nan') # Default if test cannot be run
        if len(baseline_values) >= 2 and len(attack_values) >= 2: # t-test requires at least 2 samples per group
            t_stat, p_value = stats.ttest_ind(
                baseline_values,
                attack_values,
                equal_var=False, # Welch's t-test, does not assume equal variance
                nan_policy='omit' # Ignore NaNs if any slip through (though dropna should handle)
            )
        
        impact_analysis[tenant] = {
            'baseline_latency': tenant_baseline_latency,
            'attack_latency': tenant_attack_latency,
            'increase_percentage': latency_increase_pct,
            'p_value': p_value,
            'significant_impact': p_value < 0.05 # Assuming alpha = 0.05
        }
    
    return impact_analysis

def analyze_error_rate_correlation(app_metrics: Dict[str, pd.DataFrame],
                                 infrastructure_metrics: Dict[str, pd.DataFrame],
                                 noisy_tenant: str) -> Dict[str, float]:
    """
    Analyzes the correlation between the noisy tenant's resource utilization
    and error rates in other tenants.
    
    Args:
        app_metrics: Dictionary with application metrics.
        infrastructure_metrics: Dictionary with infrastructure metrics.
        noisy_tenant: The tenant identified as noisy.
        
    Returns:
        Dictionary with correlations between resource usage and error rates.
    """
    correlations = {}
    
    if 'error_rate' not in app_metrics or 'cpu_usage' not in infrastructure_metrics:
        print("Warning: 'error_rate' or 'cpu_usage' not found in metrics. Skipping error rate correlation.")
        return correlations
        
    error_df = app_metrics['error_rate']
    cpu_df = infrastructure_metrics['cpu_usage']
    
    # Filter data for the noisy tenant
    noisy_cpu_data = cpu_df[cpu_df['tenant'] == noisy_tenant]
    
    # For each tenant, calculate correlation between noisy tenant's CPU and their error rate
    for tenant in error_df['tenant'].unique():
        if tenant == noisy_tenant:
            continue
            
        tenant_error_data = error_df[error_df['tenant'] == tenant]
        
        # Align time indices for correlation
        # Ensure 'datetime' column exists and is in datetime format
        if 'datetime' in noisy_cpu_data.columns and 'datetime' in tenant_error_data.columns:
            noisy_cpu_data['datetime'] = pd.to_datetime(noisy_cpu_data['datetime'])
            tenant_error_data['datetime'] = pd.to_datetime(tenant_error_data['datetime'])

            # Create time series indexed by datetime
            cpu_series = noisy_cpu_data.set_index('datetime')['value']
            error_series = tenant_error_data.set_index('datetime')['value']
            
            # Resample to a common frequency (e.g., 1 minute) and align
            # This handles missing data points by forward-filling, then back-filling
            # Adjust frequency as needed based on data granularity
            resample_freq = '1T' 
            aligned_df = pd.DataFrame({'cpu': cpu_series, 'error': error_series})
            aligned_df_resampled = aligned_df.resample(resample_freq).mean().ffill().bfill()
            
            # Calculate correlation if enough data points
            if not aligned_df_resampled.empty and len(aligned_df_resampled) > 1:
                correlation = aligned_df_resampled['cpu'].corr(aligned_df_resampled['error'])
                correlations[tenant] = correlation
            else:
                correlations[tenant] = np.nan # Not enough data or no overlap
        else:
            print(f"Warning: 'datetime' column missing for tenant {tenant} or noisy tenant. Skipping correlation.")
            correlations[tenant] = np.nan

    return correlations

def calculate_application_slo_violations(app_metrics: Dict[str, pd.DataFrame],
                                       slo_thresholds: Dict[str, float],
                                       noisy_tenant: str) -> Dict[str, Dict[str, float]]:
    """
    Calculates SLO (Service Level Objectives) violations caused by the noisy tenant.
    
    Args:
        app_metrics: Dictionary with application metrics.
        slo_thresholds: Dictionary with SLO thresholds per metric.
        noisy_tenant: The tenant identified as noisy.
        
    Returns:
        Dictionary with SLO violation analysis per tenant.
    """
    slo_analysis = {}
    
    for metric_name, threshold in slo_thresholds.items():
        if metric_name not in app_metrics:
            print(f"Warning: Metric {metric_name} for SLO check not found in app_metrics. Skipping.")
            continue
            
        df_metric = app_metrics[metric_name]
        
        # Analyze by phase
        baseline_df = df_metric[df_metric['phase'].str.contains('Baseline', case=False, na=False)]
        attack_df = df_metric[df_metric['phase'].str.contains('Attack', case=False, na=False)]
        
        for tenant in df_metric['tenant'].unique():
            if tenant == noisy_tenant:
                continue
                
            # Calculate violations in baseline
            tenant_baseline_data = baseline_df[baseline_df['tenant'] == tenant]
            baseline_violations = len(tenant_baseline_data[tenant_baseline_data['value'] > threshold])
            baseline_total_points = len(tenant_baseline_data)
            baseline_violation_rate = baseline_violations / baseline_total_points if baseline_total_points > 0 else 0
            
            # Calculate violations in attack phase
            tenant_attack_data = attack_df[attack_df['tenant'] == tenant]
            attack_violations = len(tenant_attack_data[tenant_attack_data['value'] > threshold])
            attack_total_points = len(tenant_attack_data)
            attack_violation_rate = attack_violations / attack_total_points if attack_total_points > 0 else 0
            
            # Store results
            if tenant not in slo_analysis:
                slo_analysis[tenant] = {}
                
            slo_analysis[tenant][metric_name] = {
                'baseline_violation_rate': baseline_violation_rate,
                'attack_violation_rate': attack_violation_rate,
                'violation_increase': attack_violation_rate - baseline_violation_rate,
                'violation_increase_factor': attack_violation_rate / baseline_violation_rate if baseline_violation_rate > 0 else (float('inf') if attack_violation_rate > 0 else 0)
            }
    
    return slo_analysis
