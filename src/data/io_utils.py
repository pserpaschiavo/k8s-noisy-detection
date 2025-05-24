import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Moved from pipeline.visualization.table_generator
# Helper function, also used by the old export_to_latex
def format_float_columns(df, float_format='.2f'):
    """
    Formata colunas de ponto flutuante em um DataFrame.
    
    Args:
        df (DataFrame): DataFrame a ser formatado
        float_format (str): Formato a ser aplicado (ex: '.2f', '.3f')
        
    Returns:
        DataFrame: DataFrame com colunas formatadas
    """
    result = df.copy()
    
    # Inner function for robust formatting with debugging
    def _formatter(x_val, fmt_str):
        if pd.notna(x_val):
            try:
                # Check and fix the format string - ensure it's a valid float format
                # Most common format is '.4f' or similar - ensure it starts with a dot if not
                # and has a valid format specifier at the end
                fixed_fmt = fmt_str
                if not fmt_str.startswith('.') and not fmt_str.startswith('%'):
                    fixed_fmt = '.' + fmt_str
                    
                # Ensure format ends with a type specification (f, g, e, etc.)
                if not any(fixed_fmt.endswith(c) for c in 'fFeEgGdiouxXcrs%'):
                    fixed_fmt = fixed_fmt + 'f'
                
                # Now use the fixed format string with the float value
                return format(float(x_val), fixed_fmt)
            except ValueError as e:
                # For debugging, print the problematic value and format string
                print(f"DEBUG: format_float_columns ValueError: val='{x_val}' (type {type(x_val)}), fmt='{fmt_str}' (type {type(fmt_str)}), error: {e}")
                # Return the original value as a safe fallback
                return str(x_val)
            except TypeError as e:
                print(f"DEBUG: format_float_columns TypeError: val='{x_val}' (type {type(x_val)}), fmt='{fmt_str}' (type {type(fmt_str)}), error: {e}")
                return str(x_val)
            except Exception as e:
                # Catch any other formatting errors we might encounter
                print(f"DEBUG: format_float_columns unexpected error: {e}")
                return str(x_val)
        return ""

    for col in result.columns:
        if result[col].dtype in [np.float32, np.float64, object]: # Include object to attempt conversion
            result[col] = result[col].map(lambda x: _formatter(x, float_format))
    
    return result

# Moved from pipeline.visualization.table_generator
def export_to_csv(df, filename, float_format='.2f'): # Modified to use a default float_format directly
    """
    Exporta um DataFrame para um arquivo CSV formatado.
    
    Args:
        df (DataFrame): DataFrame a ser exportado
        filename (str): Nome do arquivo de saída
        float_format (str, optional): Formato para valores de ponto flutuante. Default '.2f'.
    """
    # Verify float_format is valid
    valid_float_format = float_format
    
    # Ensure it starts with a dot if it's supposed to be a standard format like .2f
    if not valid_float_format.startswith('.') and not valid_float_format.startswith('%'):
        valid_float_format = '.' + valid_float_format
    
    # Make sure it has a format type at the end
    if not any(valid_float_format.endswith(c) for c in 'fFeEgGdiouxXcrs%'):
        valid_float_format = valid_float_format + 'f'
    
    try:    
        # Garantir que o diretório existe
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Formatar float columns antes de exportar
        # Using the provided float_format or the default '.2f'
        formatted_df = format_float_columns(df, valid_float_format) 
        
        # Exportar para CSV
        formatted_df.to_csv(filename, index=False)
        
        print(f"Tabela exportada como CSV para {filename}")
    except IsADirectoryError:
        print(f"Error: '{filename}' is a directory, not a file. Please provide a valid file path.")
    except PermissionError:
        print(f"Error: Permission denied when writing to '{filename}'. Check your file permissions.")
    except Exception as e:
        print(f"Error exporting CSV to {filename}: {e}")

# Added to centralize CSV saving for causality results
def save_causality_results_to_csv(causality_results_df, output_path, float_format='.4f'): # Added float_format for p-values
    """
    Saves the causality analysis results to a CSV file.
    Uses the main export_to_csv function with specific formatting for causality data.
    """
    df_to_save = causality_results_df.copy()

    if 'p_value' in df_to_save.columns:
        df_to_save['p_value'] = df_to_save['p_value'].map(
            lambda x: format(float(x), float_format) if pd.notna(x) else ""
        )
    if 'lag' in df_to_save.columns:
        # Check if the column is numeric-like before attempting float conversion and formatting
        if pd.api.types.is_numeric_dtype(df_to_save['lag'].dropna()):
             df_to_save['lag'] = df_to_save['lag'].map(
                 lambda x: format(float(x), ".0f") if pd.notna(x) else ""
             )

    export_to_csv(df_to_save, output_path, float_format=float_format)

def save_figure(figure, output_path, filename, dpi=300, bbox_inches='tight', **kwargs):
    """
    Saves a Matplotlib figure to a file.

    Handles directory creation and provides default high-quality save options.

    Args:
        figure (matplotlib.figure.Figure or pyplot module): The figure object or plt module itself.
        output_path (str): The directory where the figure should be saved.
        filename (str): The name of the file (e.g., 'my_plot.png').
        dpi (int): Dots per inch for the saved figure.
        bbox_inches (str): Bounding box adjustment (e.g., 'tight').
        **kwargs: Additional keyword arguments to pass to savefig().
    """
    if not output_path or not filename:
        print("Error: output_path and filename must be provided to save_figure.")
        return

    full_path = os.path.join(output_path, filename)
    os.makedirs(output_path, exist_ok=True)

    try:
        if hasattr(figure, 'savefig'): # If it's a Figure object
            figure.savefig(full_path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        elif figure is plt: # If it's the pyplot module
            plt.savefig(full_path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        else:
            print(f"Error: Unsupported figure type passed to save_figure: {type(figure)}")
            return
        print(f"Figure saved to {full_path}")
    except Exception as e:
        print(f"Error saving figure {full_path}: {e}")
