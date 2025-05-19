"""
Module for analysis between different phases of the noisy neighbor experiment.

This module provides functions to analyze and compare metrics between the
different phases (Baseline, Attack, Recovery) of the experiment.
"""

import pandas as pd
import numpy as np
from scipy import stats


def compare_phases_ttest(df, metric_column='value', alpha=0.05):
    """
    Compares metrics between different phases using Student's t-test.
    
    Args:
        df (DataFrame): DataFrame with metric data.
        metric_column (str): Column with metric values.
        alpha (float): Significance level for the statistical test.
        
    Returns:
        DataFrame: DataFrame with results of the statistical tests.
    """
    # List to store results
    results = []
    
    # For each tenant and round
    for (tenant, round_name), group in df.groupby(['tenant', 'round']):
        phases = sorted(group['phase'].unique())
        
        # Compare each pair of phases
        for i, phase1 in enumerate(phases):
            for phase2 in phases[i+1:]:
                # Data from the two phases
                data1 = group[group['phase'] == phase1][metric_column].dropna()
                data2 = group[group['phase'] == phase2][metric_column].dropna()
                
                if len(data1) < 2 or len(data2) < 2:
                    continue  # Skip if not enough data
                
                # Calculate statistics
                mean1, mean2 = data1.mean(), data2.mean()
                std1, std2 = data1.std(), data2.std()
                
                # T-test for two independent samples
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(data1) - 1) * std1**2 + (len(data2) - 1) * std2**2) / 
                                     (len(data1) + len(data2) - 2))
                cohen_d = abs(mean1 - mean2) / pooled_std if pooled_std != 0 else np.nan
                
                # Add results
                results.append({
                    'tenant': tenant,
                    'round': round_name,
                    'phase1': phase1,
                    'phase2': phase2,
                    'mean1': mean1,
                    'mean2': mean2,
                    'std1': std1,
                    'std2': std2,
                    'diff': mean2 - mean1,
                    'percent_diff': ((mean2 - mean1) / mean1) * 100 if mean1 != 0 else np.nan,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'effect_size': cohen_d,
                    'effect_magnitude': 'Large' if cohen_d > 0.8 else 
                                        'Medium' if cohen_d > 0.5 else 
                                        'Small' if cohen_d > 0.2 else 'Negligible'
                })
    
    return pd.DataFrame(results)


def analyze_recovery_effectiveness(df, metric_column='value', 
                                  baseline_phase='1 - Baseline', 
                                  attack_phase='2 - Attack', 
                                  recovery_phase='3 - Recovery'):
    """
    Analyzes the effectiveness of recovery after the attack phase.
    
    Args:
        df (DataFrame): DataFrame with metric data.
        metric_column (str): Column with metric values.
        baseline_phase (str): Name of the baseline phase.
        attack_phase (str): Name of the attack phase.
        recovery_phase (str): Name of the recovery phase.
        
    Returns:
        DataFrame: DataFrame with recovery analysis.
    """
    # List to store results
    results = []
    
    # For each tenant and round
    for (tenant, round_name), group in df.groupby(['tenant', 'round']):
        # Get data for each phase
        baseline_data = group[group['phase'] == baseline_phase][metric_column]
        attack_data = group[group['phase'] == attack_phase][metric_column]
        recovery_data = group[group['phase'] == recovery_phase][metric_column]
        
        if len(baseline_data) == 0 or len(attack_data) == 0 or len(recovery_data) == 0:
            continue  # Skip if any phase has no data
        
        # Calculate statistics
        baseline_mean = baseline_data.mean()
        attack_mean = attack_data.mean()
        recovery_mean = recovery_data.mean()
        
        # Calculate degradation and recovery indices
        degradation = attack_mean - baseline_mean
        recovery_delta = recovery_mean - attack_mean
        
        # Calculate recovery percentage
        if degradation != 0:
            # Recovery percentage relative to the degradation experienced
            # If degradation was positive (metric increased), recovery_delta should be negative to recover.
            # If degradation was negative (metric decreased), recovery_delta should be positive.
            # We want to see how much of the absolute degradation was reversed.
            recovery_percent = (-recovery_delta / degradation) * 100 
        else:
            # If there was no degradation, recovery is not applicable in this context,
            # or could be considered 100% if recovery_mean is close to baseline_mean.
            # For simplicity, assign NaN or handle as a special case based on requirements.
            if recovery_mean == baseline_mean:
                recovery_percent = 100.0
            elif attack_mean == baseline_mean and recovery_mean != baseline_mean:
                 recovery_percent = 0.0 # No degradation, but recovery changed from baseline
            else:
                recovery_percent = np.nan
        
        # Difference of recovery value relative to baseline
        baseline_diff = recovery_mean - baseline_mean
        baseline_diff_percent = (baseline_diff / baseline_mean) * 100 if baseline_mean != 0 else np.nan
        
        # Add results
        results.append({
            'tenant': tenant,
            'round': round_name,
            'baseline_mean': baseline_mean,
            'attack_mean': attack_mean,
            'recovery_mean': recovery_mean,
            'degradation': degradation,
            'degradation_percent': (degradation / baseline_mean) * 100 if baseline_mean != 0 else np.nan,
            'recovery_delta': recovery_delta,
            'recovery_percent': recovery_percent,
            'baseline_diff': baseline_diff,
            'baseline_diff_percent': baseline_diff_percent,
            'full_recovery': abs(baseline_diff_percent) < 5 if not np.isnan(baseline_diff_percent) else False
        })
    
    return pd.DataFrame(results)
