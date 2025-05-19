"""
Module for generating tables in LaTeX and CSV formats for the noisy neighbors experiment.

This module provides functions to create well-formatted tables with the results
of the experiment, suitable for academic publications.
"""

import pandas as pd
import numpy as np
import os
from pipeline.config import TABLE_EXPORT_CONFIG  # Import TABLE_EXPORT_CONFIG


def format_float_columns(df, float_format='.2f'):
    """
    Formats float columns in a DataFrame.
    
    Args:
        df (DataFrame): DataFrame to be formatted
        float_format (str): Format to be applied (e.g., '.2f', '.3f')
        
    Returns:
        DataFrame: DataFrame with formatted columns
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # For each column, check if it is float
    for col in result.columns:
        if result[col].dtype in [np.float32, np.float64]:
            result[col] = result[col].map(lambda x: f"{x:{float_format}}" if not pd.isna(x) else "")
    
    return result


def convert_df_to_markdown(df, float_format=None, index=None):
    """
    Converts a DataFrame to a Markdown formatted table.
    
    Args:
        df (DataFrame): DataFrame to be converted
        float_format (str, optional): Format for float values. Uses config if None.
        index (bool, optional): Whether to include the index in the table. Uses config if None.
        
    Returns:
        str: String containing the Markdown formatted table
    """
    # Use values from TABLE_EXPORT_CONFIG if not provided
    final_float_format = float_format if float_format is not None else TABLE_EXPORT_CONFIG.get('float_format', '.2f')
    final_index = index if index is not None else TABLE_EXPORT_CONFIG.get('include_index', False)

    # Format float columns before converting
    formatted_df = format_float_columns(df, final_float_format)
    
    # Convert to Markdown
    markdown_table = formatted_df.to_markdown(index=final_index)
    
    return markdown_table


def export_to_latex(df, caption, label, filename, float_format=None, index=None, 
                   column_format=None, escape=True, longtable=None):
    """
    Exports a DataFrame to a formatted LaTeX table.
    
    Args:
        df (DataFrame): DataFrame to be exported
        caption (str): Table caption
        label (str): Identifier for cross-referencing
        filename (str): Output filename
        float_format (str, optional): Format for float values. Uses config if None.
        index (bool, optional): Whether to include the index in the table. Uses config if None.
        column_format (str): Custom column format for LaTeX
        escape (bool): Whether to escape special characters
        longtable (bool, optional): Whether to use the longtable environment. Uses config if None.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    # Use values from TABLE_EXPORT_CONFIG if not provided
    final_float_format_str = float_format if float_format is not None else TABLE_EXPORT_CONFIG.get('float_format', '.2f')
    final_index = index if index is not None else TABLE_EXPORT_CONFIG.get('include_index', False)
    final_longtable = longtable if longtable is not None else TABLE_EXPORT_CONFIG.get('longtable', False)

    # Format float columns to string BEFORE to_latex
    formatted_df = format_float_columns(df, final_float_format_str)
    
    # Export to LaTeX
    latex_table = formatted_df.to_latex(
        index=final_index,
        caption=caption,
        label=label,
        float_format=None,  # Floats are already formatted strings
        column_format=column_format,
        escape=escape,
        longtable=final_longtable
    )
    
    # Add packages and additional formatting
    latex_preamble = """\\documentclass{article}
\\usepackage{booktabs}
\\usepackage{siunitx}
\\usepackage{caption}
\\usepackage{longtable}
\\usepackage{multirow}
\\usepackage{array}
\\usepackage{colortbl}
\\begin{document}
"""
    latex_end = "\\end{document}"
    
    with open(filename, 'w') as f:
        f.write(latex_preamble)
        f.write(latex_table)
        f.write(latex_end)
    
    print(f"Table exported as LaTeX to {filename}")


def export_to_csv(df, filename, float_format=None):
    """
    Exports a DataFrame to a formatted CSV file.
    
    Args:
        df (DataFrame): DataFrame to be exported
        filename (str): Output filename
        float_format (str, optional): Format for float values. Uses config if None.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Use value from TABLE_EXPORT_CONFIG if not provided
    final_float_format = float_format if float_format is not None else TABLE_EXPORT_CONFIG.get('float_format', '.2f')

    # Format float columns before exporting
    formatted_df = format_float_columns(df, final_float_format)
    
    # Export to CSV
    formatted_df.to_csv(filename, index=False)
    
    print(f"Table exported as CSV to {filename}")


def create_phase_comparison_table(df, metric_name, phases=None, tenants=None, 
                                 value_column='mean', std_column='std'):
    """
    Creates a formatted table comparing metrics between phases for each tenant.
    
    Args:
        df (DataFrame): DataFrame with aggregated data
        metric_name (str): Name of the metric for the table title
        phases (list): List of phases to include (None = all)
        tenants (list): List of tenants to include (None = all)
        value_column (str): Name of the column with mean values
        std_column (str): Name of the column with standard deviations
        
    Returns:
        DataFrame: Formatted DataFrame for export
    """
    # Filter data if necessary
    data = df.copy()
    if phases:
        data = data[data['phase'].isin(phases)]
    if tenants:
        data = data[data['tenant'].isin(tenants)]
    
    # Create a formatted table
    formatted_table = pd.DataFrame()
    
    # For each tenant, add a row
    for tenant, tenant_data in data.groupby('tenant'):
        row = {'Tenant': tenant}
        
        # For each phase, add mean ± standard deviation
        for phase, phase_data in tenant_data.groupby('phase'):
            if len(phase_data) > 0:
                mean_val = phase_data[value_column].iloc[0]
                std_val = phase_data[std_column].iloc[0] if std_column in phase_data.columns else 0
                
                # Format as "mean ± std_dev"
                row[phase] = f"{mean_val:.2f} ± {std_val:.2f}"
        
        # Add row to DataFrame
        formatted_table = pd.concat([formatted_table, pd.DataFrame([row])], ignore_index=True)
    
    return formatted_table


def create_impact_summary_table(impact_df, metric_name, round_column='round', 
                              tenant_column='tenant', impact_column='impact_percent'):
    """
    Creates a summary table of the percentage impact on each tenant during the attack phase.
    
    Args:
        impact_df (DataFrame): DataFrame with calculated impact per tenant
        metric_name (str): Name of the metric for the table title
        round_column (str): Name of the column with round names
        tenant_column (str): Name of the column with tenant names
        impact_column (str): Name of the column with percentage impact values
        
    Returns:
        DataFrame: Formatted DataFrame for export
    """
    # Create a pivot table with rounds in columns and tenants in rows
    pivot_table = impact_df.pivot_table(
        index=tenant_column,
        columns=round_column,
        values=impact_column
    )
    
    # Add mean impact across rounds
    pivot_table['Mean'] = pivot_table.mean(axis=1) # Changed 'Média' to 'Mean'
    
    # Format the table for export
    formatted_table = pivot_table.copy()
    
    for col in formatted_table.columns:
        formatted_table[col] = formatted_table[col].apply(lambda x: f"{x:.2f}%" if not pd.isna(x) else "")
    
    return formatted_table


def create_causality_results_table(causal_links_df,
                                   source_col='Source Tenant',
                                   source_metric_col='Source Metric',
                                   target_col='Target Tenant',
                                   target_metric_col='Target Metric',
                                   lag_col='Lag (Granger)',
                                   p_value_col='P-Value (Granger)'):
    """
    Creates a formatted table from a DataFrame of causal links.

    Args:
        causal_links_df (pd.DataFrame): DataFrame containing the causal links.
                                        Expected columns: 'Source Tenant', 'Source Metric', 
                                                        'Target Tenant', 'Target Metric', 
                                                        'Lag (Granger)', 'P-Value (Granger)'.
        source_col (str): Name for the source tenant column in the output table.
        source_metric_col (str): Name for the source metric column in the output table.
        target_col (str): Name for the target tenant column in the output table.
        target_metric_col (str): Name for the target metric column in the output table.
        lag_col (str): Name for the lag column in the output table.
        p_value_col (str): Name for the p-value column in the output table.

    Returns:
        pd.DataFrame: A DataFrame formatted for export, with renamed columns
                      and formatted p-values.
    """
    if causal_links_df.empty:
        # Return an empty DataFrame with expected column names if no links are found
        return pd.DataFrame(columns=[
            source_col, source_metric_col, target_col, target_metric_col,
            lag_col, p_value_col
        ])

    # Select and rename columns
    table_df = causal_links_df[[
        'Source Tenant', 'Source Metric', 'Target Tenant', 'Target Metric',
        'Lag (Granger)', 'P-Value (Granger)'
    ]].copy()

    table_df.rename(columns={
        'Source Tenant': source_col,
        'Source Metric': source_metric_col,
        'Target Tenant': target_col,
        'Target Metric': target_metric_col,
        'Lag (Granger)': lag_col,
        'P-Value (Granger)': p_value_col
    }, inplace=True)

    # Format P-Value
    if p_value_col in table_df.columns:
        table_df[p_value_col] = table_df[p_value_col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    # Sort by P-Value and then by Lag
    table_df.sort_values(by=[p_value_col, lag_col], ascending=[True, True], inplace=True)
    
    return table_df
