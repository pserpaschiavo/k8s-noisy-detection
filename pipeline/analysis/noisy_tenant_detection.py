"""
Module for automatic detection of the noisy tenant (noisy neighbor).

This module provides functions to automatically identify which tenant
is likely the generator of interference on other tenants, without needing
to specify the noisy tenant beforehand.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler

def detect_noisy_tenant_from_correlation(correlation_matrix, tenant_metrics):
    """
    Detects the most likely noisy tenant based on the correlation matrix.
    A noisy tenant usually has negative correlations with other tenants
    for metrics like CPU and memory.
    
    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix between tenant metrics.
        tenant_metrics (List[str]): List of strings in the format "metric_tenant".
        
    Returns:
        Dict[str, float]: Dictionary with scores for each tenant, higher = more likely to be noisy.
    """
    tenant_scores = {}
    
    # Return an empty dictionary if the correlation matrix is None or empty
    if correlation_matrix is None or correlation_matrix.empty:
        return tenant_scores
        
    # Check if tenant_metrics is a valid list
    if tenant_metrics is None or len(tenant_metrics) == 0:
        return tenant_scores
    
    # Map tenants directly from the correlation matrix columns
    tenants = []
    
    for col in correlation_matrix.columns:
        if 'tenant-' in col:
            parts = col.split('tenant-')
            if len(parts) >= 2:
                tenant_name = f"tenant-{parts[1]}"
                if tenant_name not in tenants:
                    tenants.append(tenant_name)
    
    # If we didn't find tenants in the "tenant-X" format, use the columns themselves
    if not tenants:
        tenants = list(correlation_matrix.columns)
    
    for tenant in tenants:
        tenant_score = 0
        count = 0
        
        # Define the metrics we want to analyze
        metrics_to_analyze = ['cpu_usage', 'memory_usage', 'network_total_bandwidth', 'disk_throughput_total']

        # For each metric we are analyzing
        for metric in metrics_to_analyze:
            tenant_metric = f"{metric}_{tenant}"
            
            if tenant_metric not in correlation_matrix.columns:
                continue
                
            # Check correlations with other tenants for this metric
            for other_tenant in tenants:
                if other_tenant == tenant:
                    continue
                    
                other_metric = f"{metric}_{other_tenant}"
                if other_metric not in correlation_matrix.columns:
                    continue
                
                # Negative correlation in CPU/memory means that when one increases, the other decreases
                # This is indicative of a noisy tenant affecting others
                correlation = correlation_matrix.loc[tenant_metric, other_metric]
                
                # A negative value indicates a possible noisy neighbor relationship
                if correlation < 0:
                    # The more negative the correlation, the higher the score
                    tenant_score += abs(correlation)
                count += 1
        
        # Normalize by the number of comparisons
        if count > 0:
            tenant_scores[tenant] = tenant_score / count
        else:
            tenant_scores[tenant] = 0
    
    return tenant_scores


def detect_noisy_tenant_from_causality(causality_results):
    """
    Detects the most likely noisy tenant based on Granger causality tests.
    A noisy tenant generally causes changes in other tenants more than it is affected by them.
    
    Args:
        causality_results (List[Dict]): Results of causality tests between tenants.
        
    Returns:
        Dict[str, float]: Dictionary with scores for each tenant, higher = more likely to be noisy.
    """
    # If we have no causality results, return an empty dictionary
    if causality_results is None or not causality_results:
        return {}
        
    # Check the data type, if it's a string, return an empty dictionary
    if isinstance(causality_results, str):
        return {}
        
    # Count how many times each tenant is identified as a causer
    # and how many times it is identified as affected
    tenant_causality_scores = {}
    
    for result in causality_results:
        # Check if result is a dictionary
        if not isinstance(result, dict):
            continue
            
        # Use get() for dictionaries and access directly if it's an object with attributes
        try:
            if hasattr(result, 'get'):
                source = result.get('source_tenant')
                target = result.get('target_tenant')
                p_value = result.get('min_p_value')
            else:
                source = result.source_tenant if hasattr(result, 'source_tenant') else None
                target = result.target_tenant if hasattr(result, 'target_tenant') else None
                p_value = result.min_p_value if hasattr(result, 'min_p_value') else None
        except Exception:
            # If there's any problem, skip this result
            continue
        
        if not source or not target or p_value is None:
            continue
            
        # Initialize scores if necessary
        if source not in tenant_causality_scores:
            tenant_causality_scores[source] = {
                'caused_others': 0,
                'affected_by_others': 0,
                'total_relations': 0
            }
        
        if target not in tenant_causality_scores:
            tenant_causality_scores[target] = {
                'caused_others': 0,
                'affected_by_others': 0,
                'total_relations': 0
            }
        
        # Check if there is a significant causal relationship (p < 0.05)
        if p_value < 0.05:
            # Increment the score of the causing tenant
            tenant_causality_scores[source]['caused_others'] += 1
            # Increment the counter of the affected tenant
            tenant_causality_scores[target]['affected_by_others'] += 1
        
        # Increment the total counter of analyzed relations
        tenant_causality_scores[source]['total_relations'] += 1
        tenant_causality_scores[target]['total_relations'] += 1
    
    # Calculate final score for each tenant
    tenant_scores = {}
    for tenant, scores in tenant_causality_scores.items():
        # A noisy tenant should cause more than be affected
        if scores['total_relations'] > 0:
            # Ratio between causing and being affected
            causality_ratio = scores['caused_others'] / max(1, scores['affected_by_others'])
            # We multiply by the quantity of identified causal relations
            tenant_scores[tenant] = causality_ratio * scores['caused_others']
        else:
            tenant_scores[tenant] = 0
    
    return tenant_scores


def detect_noisy_tenant_from_anomalies(anomaly_results, phase_df):
    """
    Detects the most likely noisy tenant based on the quantity and severity
    of detected anomalies, especially during the attack phase.
    
    Args:
        anomaly_results (pd.DataFrame): DataFrame with anomaly detection results.
        phase_df (pd.DataFrame): DataFrame with experiment phase information.
        
    Returns:
        Dict[str, float]: Dictionary with scores for each tenant, higher = more likely to be noisy.
    """
    tenant_scores = {}
    
    # Check if anomaly_results is a DataFrame
    if anomaly_results is None or not isinstance(anomaly_results, pd.DataFrame):
        return tenant_scores
        
    # Check if the DataFrame has the necessary columns
    required_columns = ['tenant', 'is_anomaly_if', 'is_anomaly_lof']
    if not all(col in anomaly_results.columns for col in required_columns):
        return tenant_scores
        
    # Check if we have the 'phase' column or if we need to add it based on phase_df
    if 'phase' not in anomaly_results.columns:
        # If we don't have the phase in the anomalies DataFrame, we can try to analyze by tenant only
        # and check which tenants have more anomalies in general
        
        # Group by tenant and count anomalies
        tenant_anomalies = anomaly_results.groupby('tenant')[['is_anomaly_if', 'is_anomaly_lof']].sum().reset_index()
        
        # Calculate score based on the quantity of anomalies
        for _, row in tenant_anomalies.iterrows():
            tenant = row['tenant']
            # Sum the different types of anomalies
            total_anomalies = row['is_anomaly_if'] + row['is_anomaly_lof']
            tenant_scores[tenant] = total_anomalies
            
        return tenant_scores
    
    # If we have the phase, filter only the attack phase
    attack_phase = anomaly_results[anomaly_results['phase'].str.contains('Attack', case=False, na=False)]
    
    if len(attack_phase) == 0:
        return tenant_scores
    
    # Group by tenant and count anomalies
    tenant_anomalies = attack_phase.groupby('tenant')[['is_anomaly_if', 'is_anomaly_lof']].sum().reset_index()
    
    # Calculate score based on the quantity of anomalies
    for _, row in tenant_anomalies.iterrows():
        tenant = row['tenant']
        # Sum the different types of anomalies
        total_anomalies = row['is_anomaly_if'] + row['is_anomaly_lof']
        tenant_scores[tenant] = total_anomalies
    
    # Normalize scores
    if tenant_scores:
        max_score = max(tenant_scores.values())
        if max_score > 0:
            tenant_scores = {k: v / max_score for k, v in tenant_scores.items()}
    
    return tenant_scores


def detect_noisy_tenant_from_impact(impact_scores):
    """
    Detects the most likely noisy tenant based on the impact caused on other tenants.
    
    Args:
        impact_scores (Dict): Dictionary with impact scores between tenants.
        
    Returns:
        Dict[str, float]: Dictionary with scores for each tenant, higher = more likely to be noisy.
    """
    tenant_scores = {}
    
    for tenant, impacts in impact_scores.items():
        # Calculate the average impact caused on other tenants
        impact_values = [v for k, v in impacts.items() if k != tenant]
        if impact_values:
            tenant_scores[tenant] = sum(impact_values) / len(impact_values)
        else:
            tenant_scores[tenant] = 0
    
    return tenant_scores


def combine_detection_results(correlation_scores, causality_scores, anomaly_scores, impact_scores, weights=None):
    """
    Combines the results of different detection methods to obtain a final score.
    
    Args:
        correlation_scores (Dict[str, float]): Scores based on correlation.
        causality_scores (Dict[str, float]): Scores based on causality.
        anomaly_scores (Dict[str, float]): Scores based on anomalies.
        impact_scores (Dict[str, float]): Scores based on impact.
        weights (Dict[str, float]): Weights for each method (optional).
        
    Returns:
        Dict[str, float]: Final combined scores.
        Dict[str, Dict[str, float]]: Detailed scores by method.
    """
    # Define default weights if not provided
    if weights is None:
        weights = {
            'correlation': 0.2,
            'causality': 0.3,
            'anomaly': 0.2,
            'impact': 0.3
        }
    
    # Map all tenant names to the standard format (e.g., "tenant-a", "tenant-b")
    # this helps consolidate tenants that may appear with different prefixes in different methods
    def normalize_tenant_name(tenant_name):
        # If the name is already in "tenant-X" format, return it
        if tenant_name.startswith("tenant-"):
            return tenant_name
            
        # If the name has a prefix and then "tenant-X", extract only "tenant-X"
        if "_tenant-" in tenant_name:
            parts = tenant_name.split("_tenant-")
            return f"tenant-{parts[1]}"
            
        # If we can't normalize, return the original name
        return tenant_name
    
    # Normalize tenant names in the scores
    normalized_correlation_scores = {normalize_tenant_name(k): v for k, v in correlation_scores.items()}
    normalized_causality_scores = {normalize_tenant_name(k): v for k, v in causality_scores.items()}
    normalized_anomaly_scores = {normalize_tenant_name(k): v for k, v in anomaly_scores.items()}
    normalized_impact_scores = {normalize_tenant_name(k): v for k, v in impact_scores.items()}
    
    # Collect all unique tenants after normalization
    all_tenants = set()
    all_tenants.update(normalized_correlation_scores.keys())
    all_tenants.update(normalized_causality_scores.keys())
    all_tenants.update(normalized_anomaly_scores.keys())
    all_tenants.update(normalized_impact_scores.keys())
    
    # Normalize scores within each method
    methods = {
        'correlation': normalized_correlation_scores,
        'causality': normalized_causality_scores,
        'anomaly': normalized_anomaly_scores,
        'impact': normalized_impact_scores
    }
    
    normalized_scores = {}
    for method, scores in methods.items():
        if not scores:
            continue
        
        # Find the highest score to normalize
        max_score = max(scores.values()) if scores else 1
        
        # Normalize scores to [0, 1]
        if max_score > 0:
            normalized_scores[method] = {k: v / max_score for k, v in scores.items()}
        else:
            normalized_scores[method] = scores
    
    # Combine weighted scores
    final_scores = {}
    detailed_scores = {}
    
    for tenant in all_tenants:
        weighted_sum = 0
        detailed = {}
        
        for method, weight in weights.items():
            if method in normalized_scores and tenant in normalized_scores[method]:
                score = normalized_scores[method][tenant]
                weighted_sum += score * weight
                detailed[method] = score
            else:
                detailed[method] = 0
        
        final_scores[tenant] = weighted_sum
        detailed_scores[tenant] = detailed
    
    return final_scores, detailed_scores


def identify_noisy_tenant(
    metrics_dict,
    causality_results=None,
    anomaly_results=None,
    impact_scores=None,
    round_name='round-1',
    weights=None,
    real_tenants=None
):
    """
    Automatically identifies which tenant is likely the "noisy neighbor"
    based on multiple analysis criteria.
    
    Args:
        metrics_dict (Dict): Dictionary with DataFrames for each metric.
        causality_results (List[Dict]): Causality analysis results (optional).
        anomaly_results (pd.DataFrame): Anomaly detection results (optional).
        impact_scores (Dict): Impact scores between tenants (optional).
        round_name (str): Round to be analyzed.
        weights (Dict[str, float]): Weights for each detection method.
        real_tenants (List[str]): List of actual tenant names in the environment.
        
    Returns:
        str: Name of the tenant identified as the likely noisy neighbor.
        Dict: Final scores for each tenant.
        Dict: Detailed scores by detection method.
    """
    from pipeline.analysis.tenant_analysis import calculate_correlation_matrix
    
    try:
        # Calculate correlation matrix
        correlation_matrix = calculate_correlation_matrix(metrics_dict, round_name=round_name)
        
        # Extract metric/tenant names from matrix columns
        tenant_metrics = list(correlation_matrix.columns)
        
        # Detect noisy tenant using different methods
        correlation_scores = detect_noisy_tenant_from_correlation(correlation_matrix, tenant_metrics)
    except Exception as e:
        print(f"  Error calculating correlation: {e}")
        correlation_scores = {}
    
    # Initialize scores for optional methods
    causality_scores = {}
    anomaly_scores = {}
    impact_scores_processed = {}
    
    # Add causality results if available
    try:
        if causality_results:
            causality_scores = detect_noisy_tenant_from_causality(causality_results)
    except Exception as e:
        print(f"  Error processing causality data: {e}")
        causality_scores = {}
    
    # Add anomaly results if available
    try:
        if anomaly_results is not None:
            # We also need the DataFrame with phases
            # Let's extract phase information from the first metrics DataFrame
            first_metric_df = next(iter(metrics_dict.values()))
            phase_df = first_metric_df[['datetime', 'phase']].drop_duplicates() if 'phase' in first_metric_df.columns else None
            
            anomaly_scores = detect_noisy_tenant_from_anomalies(anomaly_results, phase_df)
    except Exception as e:
        print(f"  Error processing anomaly data: {e}")
        anomaly_scores = {}
    
    # Add impact results if available
    try:
        if impact_scores:
            impact_scores_processed = detect_noisy_tenant_from_impact(impact_scores)
    except Exception as e:
        print(f"  Error processing impact scores: {e}")
        impact_scores_processed = {}
    
    # Combine all results
    final_scores, detailed_scores = combine_detection_results(
        correlation_scores, 
        causality_scores, 
        anomaly_scores, 
        impact_scores_processed,
        weights
    )
    
    # Identify the tenant with the highest final score
    if final_scores:
        # If we have the list of real_tenants, filter the results to show only real tenants
        if real_tenants:
            filtered_scores = {k: v for k, v in final_scores.items() if k in real_tenants or any(k.endswith(t) for t in real_tenants)}
            if filtered_scores:
                # If we still have scores after filtering, use only real tenants
                sorted_tenants = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
                most_likely_noisy_tenant = sorted_tenants[0][0]
                # If the name contains a prefix, let's try to extract only the real tenant
                if not most_likely_noisy_tenant.startswith('tenant-') and 'tenant-' in most_likely_noisy_tenant:
                    # Extract tenant-X from the name
                    for real_tenant in real_tenants:
                        if real_tenant in most_likely_noisy_tenant:
                            most_likely_noisy_tenant = real_tenant
                            break
            else:
                # If we don't have scores for real tenants, use all scores
                sorted_tenants = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
                most_likely_noisy_tenant = sorted_tenants[0][0]
        else:
            # Sort from highest to lowest score
            sorted_tenants = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            most_likely_noisy_tenant = sorted_tenants[0][0]
            # If the identified tenant has a format like "metric_tenant", try to extract only the tenant
            if '_tenant-' in most_likely_noisy_tenant:
                most_likely_noisy_tenant = 'tenant-' + most_likely_noisy_tenant.split('_tenant-')[1]
    else:
        most_likely_noisy_tenant = None
    
    return most_likely_noisy_tenant, final_scores, detailed_scores
