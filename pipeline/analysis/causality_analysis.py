"""
Module for performing causality analysis using Transfer Entropy, Convergent Cross Mapping, and Granger Causality.
"""
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
import matplotlib.pyplot as plt
from pyinform.transferentropy import transfer_entropy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from skccm import CCM
from skccm.utilities import train_test_split

def perform_inter_tenant_causality_analysis(
    data, metrics_for_causality, noisy_tenant, other_tenants, 
    max_lag=5, test='ssr_chi2test', significance_level=0.05, 
    verbose=False, min_observations=30,
    time_col='datetime', value_col='value', tenant_col='tenant', 
    phase_col='phase', round_col='round', current_round='round-1', phase_filter=None
):
    """
    Performs inter-tenant Granger causality analysis.
    (Original function from advanced_analysis.py, adapted and enhanced)
    """
    results = []
    if data.empty:
        if verbose:
            print("Input data for causality analysis is empty. Skipping.")
        return pd.DataFrame(results)

    # Ensure 'datetime' is the index if it exists and is not already
    if time_col in data.columns and data.index.name != time_col:
        data = data.set_index(time_col)
    elif data.index.name != time_col: # If time_col is not in columns and not index name
        if verbose:
            print(f"Time column '{time_col}' not found in data. Causality analysis might be incorrect.")
        if not isinstance(data.index, pd.DatetimeIndex):
            if verbose:
                print("Index is not a DatetimeIndex. Cannot proceed with causality analysis without a proper time column.")
            return pd.DataFrame(results)

    for metric in metrics_for_causality:
        metric_data_frames = {}
        all_tenants_for_metric = [noisy_tenant] + other_tenants

        for tenant_name_loop in all_tenants_for_metric: # Renamed to avoid conflict with tenant_col parameter
            tenant_metric_data = data[
                (data[tenant_col] == tenant_name_loop) &
                (data['metric_name'] == metric)
            ]
            
            if phase_filter and phase_col in tenant_metric_data.columns:
                tenant_metric_data = tenant_metric_data[tenant_metric_data[phase_col] == phase_filter]

            if round_col in tenant_metric_data.columns:
                 tenant_metric_data = tenant_metric_data[tenant_metric_data[round_col] == current_round]

            ts = tenant_metric_data[[value_col]].copy()
            ts.rename(columns={value_col: tenant_name_loop}, inplace=True)
            
            if not ts.empty:
                metric_data_frames[tenant_name_loop] = ts

        if not metric_data_frames or len(metric_data_frames) < 2:
            if verbose:
                print(f"Not enough tenant data for metric {metric} (found {len(metric_data_frames)}) to perform causality analysis. Skipping.")
            continue
        
        combined_ts = pd.concat(metric_data_frames.values(), axis=1)
        combined_ts.ffill(inplace=True)
        combined_ts.bfill(inplace=True)
        # No dropna here on combined_ts yet, will be handled per pair

        # Check overall observations after initial ffill/bfill, but before pair-specific dropna
        # This is a general check; pair-specific checks are more critical below.
        if combined_ts.dropna().shape[0] < min_observations:
            if verbose:
                print(f"Not enough observations for metric {metric} after initial alignment and NaN handling ({combined_ts.dropna().shape[0]} < {min_observations}). Skipping.")
            # Optionally, add a general error entry for the metric here if desired
            continue
        
        if noisy_tenant not in combined_ts.columns:
            if verbose:
                print(f"Noisy tenant '{noisy_tenant}' not found in combined data for metric {metric}. Skipping causality tests for this metric.")
            # Record error for all pairs involving noisy_tenant for this metric
            for target_tenant_loop in other_tenants: # Renamed to avoid conflict
                 results.append({
                    'metric': metric, 'source_tenant': noisy_tenant, 'target_tenant': target_tenant_loop,
                    'lag': np.nan, 'p_value': np.nan, 'significant': False, 
                    'direction': f"{noisy_tenant} -> {target_tenant_loop}", 'error': f"Noisy tenant '{noisy_tenant}' not found"
                })
                 results.append({
                    'metric': metric, 'source_tenant': target_tenant_loop, 'target_tenant': noisy_tenant,
                    'lag': np.nan, 'p_value': np.nan, 'significant': False,
                    'direction': f"{target_tenant_loop} -> {noisy_tenant}", 'error': f"Noisy tenant '{noisy_tenant}' not found"
                })
            continue

        for target_tenant in other_tenants:
            error_message_skip = None
            if target_tenant not in combined_ts.columns:
                error_message_skip = f"Target tenant '{target_tenant}' not found in combined data for metric {metric}"
                if verbose:
                    print(error_message_skip)
                results.append({
                    'metric': metric, 'source_tenant': noisy_tenant, 'target_tenant': target_tenant,
                    'lag': np.nan, 'p_value': np.nan, 'significant': False,
                    'direction': f"{noisy_tenant} -> {target_tenant}", 'error': error_message_skip
                })
                results.append({
                    'metric': metric, 'source_tenant': target_tenant, 'target_tenant': noisy_tenant,
                    'lag': np.nan, 'p_value': np.nan, 'significant': False,
                    'direction': f"{target_tenant} -> {noisy_tenant}", 'error': error_message_skip
                })
                continue

            # Test causality from noisy_tenant to target_tenant
            causal_lag_nt_tt = np.nan
            p_value_nt_tt = np.nan
            error_nt_tt = None
            try:
                pair_data_nt_tt = combined_ts[[target_tenant, noisy_tenant]].dropna()
                if pair_data_nt_tt.shape[0] < min_observations:
                    raise ValueError(f"Not enough observations for pair ({target_tenant}, {noisy_tenant}) after dropna: {pair_data_nt_tt.shape[0]} < {min_observations}")

                gc_test_result_nt_tt = grangercausalitytests(
                    pair_data_nt_tt, maxlag=max_lag, verbose=False
                )
                
                p_values_at_lags_nt_tt = {}
                for lag_val in range(1, max_lag + 1):
                    if lag_val in gc_test_result_nt_tt:
                         # test result is a tuple, first element contains dict of test stats
                        test_stats = gc_test_result_nt_tt[lag_val][0]
                        if test in test_stats:
                             p_values_at_lags_nt_tt[lag_val] = test_stats[test][1] # p-value is the second element
                
                if p_values_at_lags_nt_tt:
                    causal_lag_nt_tt = min(p_values_at_lags_nt_tt, key=p_values_at_lags_nt_tt.get)
                    p_value_nt_tt = p_values_at_lags_nt_tt[causal_lag_nt_tt]
                else:
                    error_nt_tt = f"Granger test (noisy -> target) returned no valid p-values for test '{test}'"
            except Exception as e:
                error_nt_tt = str(e)
                if verbose:
                    print(f"Error in Granger causality test ({noisy_tenant} -> {target_tenant}) for {metric}: {e}")
            
            significant_nt_tt = (p_value_nt_tt < significance_level) if pd.notna(p_value_nt_tt) else False
            results.append({
                'metric': metric, 'source_tenant': noisy_tenant, 'target_tenant': target_tenant,
                'lag': causal_lag_nt_tt, 'p_value': p_value_nt_tt, 'significant': significant_nt_tt,
                'direction': f"{noisy_tenant} -> {target_tenant}", 'error': error_nt_tt
            })
            if verbose and not error_nt_tt and pd.notna(p_value_nt_tt):
                print(f"Granger causality ({noisy_tenant} -> {target_tenant}) for {metric} at lag {causal_lag_nt_tt}: p-value={p_value_nt_tt:.4f} (Significant: {significant_nt_tt})")

            # Test causality from target_tenant to noisy_tenant (reverse direction)
            causal_lag_tt_nt = np.nan
            p_value_tt_nt = np.nan
            error_tt_nt = None
            try:
                pair_data_tt_nt = combined_ts[[noisy_tenant, target_tenant]].dropna()
                if pair_data_tt_nt.shape[0] < min_observations:
                    raise ValueError(f"Not enough observations for pair ({noisy_tenant}, {target_tenant}) after dropna: {pair_data_tt_nt.shape[0]} < {min_observations}")

                gc_test_result_tt_nt = grangercausalitytests(
                    pair_data_tt_nt, maxlag=max_lag, verbose=False
                )
                p_values_at_lags_tt_nt = {}
                for lag_val in range(1, max_lag + 1):
                    if lag_val in gc_test_result_tt_nt:
                        test_stats = gc_test_result_tt_nt[lag_val][0]
                        if test in test_stats:
                            p_values_at_lags_tt_nt[lag_val] = test_stats[test][1]

                if p_values_at_lags_tt_nt:
                    causal_lag_tt_nt = min(p_values_at_lags_tt_nt, key=p_values_at_lags_tt_nt.get)
                    p_value_tt_nt = p_values_at_lags_tt_nt[causal_lag_tt_nt]
                else:
                    error_tt_nt = f"Granger test (target -> noisy) returned no valid p-values for test '{test}'"
            except Exception as e:
                error_tt_nt = str(e)
                if verbose:
                    print(f"Error in Granger causality test ({target_tenant} -> {noisy_tenant}) for {metric}: {e}")

            significant_tt_nt = (p_value_tt_nt < significance_level) if pd.notna(p_value_tt_nt) else False
            results.append({
                'metric': metric, 'source_tenant': target_tenant, 'target_tenant': noisy_tenant,
                'lag': causal_lag_tt_nt, 'p_value': p_value_tt_nt, 'significant': significant_tt_nt,
                'direction': f"{target_tenant} -> {noisy_tenant}", 'error': error_tt_nt
            })
            if verbose and not error_tt_nt and pd.notna(p_value_tt_nt):
                print(f"Granger causality ({target_tenant} -> {noisy_tenant}) for {metric} at lag {causal_lag_tt_nt}: p-value={p_value_tt_nt:.4f} (Significant: {significant_tt_nt})")

    return pd.DataFrame(results)

# visualize_causal_graph has been moved to pipeline/visualization/plots.py

def save_causality_results_to_csv(causality_results_df, output_path):
    """
    Saves the causality analysis results to a CSV file.
    """
    try:
        causality_results_df.to_csv(output_path, index=False)
        print(f"Causality results saved to {output_path}")
    except Exception as e:
        print(f"Error saving causality results to CSV: {e}")

def calculate_transfer_entropy(series1: pd.Series | np.ndarray, series2: pd.Series | np.ndarray, k: int = 1) -> float:
    """
    Calculates the Transfer Entropy from series2 to series1.
    This measures the amount of information flowing from series2 to series1.

    Args:
        series1: The target time series (pandas Series or NumPy array).
        series2: The source time series (pandas Series or NumPy array).
        k: The history length (embedding dimension).

    Returns:
        The calculated Transfer Entropy.
    """
    if len(series1) != len(series2):
        raise ValueError("Series must have the same length for Transfer Entropy calculation.")
    if len(series1) < k + 1:
        # pyinform requires len(series) > k
        # print(f"Warning: Series length ({len(series1)}) must be greater than k ({k}). Returning NaN.")
        return np.nan

    s1 = np.asarray(series1)
    s2 = np.asarray(series2)

    # Discretize data if it's continuous. PyInform works best with discrete data.
    scaler = MinMaxScaler()
    s1_scaled = scaler.fit_transform(s1.reshape(-1, 1)).flatten()
    s2_scaled = scaler.fit_transform(s2.reshape(-1, 1)).flatten()
    
    n_bins = 3 
    s1_discrete = pd.cut(s1_scaled, bins=n_bins, labels=False, include_lowest=True, duplicates='drop')
    s2_discrete = pd.cut(s2_scaled, bins=n_bins, labels=False, include_lowest=True, duplicates='drop')
    
    if np.isnan(s1_discrete).any() or np.isnan(s2_discrete).any():
        return np.nan
    
    # Ensure data is integer type for pyinform
    s1_discrete = s1_discrete.astype(int)
    s2_discrete = s2_discrete.astype(int)

    try:
        te = transfer_entropy(s1_discrete, s2_discrete, k=k)
    except Exception as e:
        te = np.nan
    return te

def calculate_convergent_cross_mapping(series1: pd.Series | np.ndarray, series2: pd.Series | np.ndarray, embed_dim: int = 2, tau: int = 1, lib_len: int | None = None, pred_len: int | None = None, random_state: int | None = None):
    """
    Calculates Convergent Cross Mapping (CCM) scores between two time series.
    CCM assesses causality by measuring if the historical record of one variable
    can reliably estimate the state of another.

    Args:
        series1: The first time series (pandas Series or NumPy array).
        series2: The second time series (pandas Series or NumPy array).
        embed_dim: Embedding dimension for state space reconstruction.
        tau: Time delay for state space reconstruction.
        lib_len: Length of the library (training) segment. If None, uses 75% of the data.
        pred_len: Length of the prediction (testing) segment. If None, uses the remainder after lib_len.
        random_state: Seed for reproducibility if train_test_split involves random sampling.

    Returns:
        A dictionary with CCM scores:
        - "ccm_s1_xmaps_s2": Score indicating how well s1 can be predicted from s2's library.
        - "ccm_s2_xmaps_s1": Score indicating how well s2 can be predicted from s1's library.
    """
    s1_np = np.asarray(series1).flatten()
    s2_np = np.asarray(series2).flatten()

    if len(s1_np) != len(s2_np):
        raise ValueError("Series must have the same length for CCM calculation.")
    
    min_len_required = embed_dim * tau + 1
    if len(s1_np) < min_len_required:
        print(f"Warning: Series length ({len(s1_np)}) is too short for the given embedding dimension ({embed_dim}) and tau ({tau}). Needs at least {min_len_required}. Returning NaN.")
        return {"ccm_s1_xmaps_s2": np.nan, "ccm_s2_xmaps_s1": np.nan}

    # Determine library and prediction lengths
    if lib_len is None:
        lib_len = int(0.75 * len(s1_np))
    
    if pred_len is None:
        pred_len = len(s1_np) - lib_len

    if lib_len + pred_len > len(s1_np):
        print(f"Warning: Sum of lib_len ({lib_len}) and pred_len ({pred_len}) exceeds series length ({len(s1_np)}). Adjusting pred_len. Returning NaN.")
        return {"ccm_s1_xmaps_s2": np.nan, "ccm_s2_xmaps_s1": np.nan}
    
    if lib_len <= 0 or pred_len <= 0:
        print(f"Warning: lib_len ({lib_len}) or pred_len ({pred_len}) is not positive. Returning NaN.")
        return {"ccm_s1_xmaps_s2": np.nan, "ccm_s2_xmaps_s1": np.nan}

    try:
        # CCM: Does s2 cause s1? (s1_np is X, s2_np is Y)
        ccm_s1_s2 = CCM(X=s1_np, Y=s2_np, tau=tau, E=embed_dim, L=lib_len)
        score_s1_xmap_s2 = ccm_s1_s2.score()

        # CCM: Does s1 cause s2? (s2_np is X, s1_np is Y)
        ccm_s2_s1 = CCM(X=s2_np, Y=s1_np, tau=tau, E=embed_dim, L=lib_len)
        score_s2_xmap_s1 = ccm_s2_s1.score()
        
        final_score_s1_xmap_s2 = np.mean(score_s1_xmap_s2) if score_s1_xmap_s2 is not None else np.nan
        final_score_s2_xmap_s1 = np.mean(score_s2_xmap_s1) if score_s2_xmap_s1 is not None else np.nan

    except Exception as e:
        print(f"Error during CCM calculation: {e}")
        final_score_s1_xmap_s2 = np.nan
        final_score_s2_xmap_s1 = np.nan
        
    return {
        "ccm_s1_xmaps_s2": final_score_s1_xmap_s2,
        "ccm_s2_xmaps_s1": final_score_s2_xmap_s1
    }
