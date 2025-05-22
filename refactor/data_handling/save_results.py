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
    # Fazer uma cópia para não modificar o original
    result = df.copy()
    
    # Para cada coluna, verificar se é ponto flutuante
    for col in result.columns:
        if result[col].dtype in [np.float32, np.float64]:
            # Ensure x is a Python float for robust formatting, handles 0-d arrays
            result[col] = result[col].map(lambda x: f"{float(x):{float_format}}" if pd.notna(x) else "")
    
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
    # Garantir que o diretório existe
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Formatar float columns antes de exportar
    # Using the provided float_format or the default '.2f'
    formatted_df = format_float_columns(df, float_format) 
    
    # Exportar para CSV
    formatted_df.to_csv(filename, index=False)
    
    print(f"Tabela exportada como CSV para {filename}")

# Added to centralize CSV saving for causality results
def save_causality_results_to_csv(causality_results_df, output_path, float_format='.4f'): # Added float_format for p-values
    """
    Saves the causality analysis results to a CSV file.
    Uses the main export_to_csv function with specific formatting for causality data.
    """
    # Ensure the directory exists (handled by export_to_csv)
    # Format p_value if it exists, and any other specific formatting needs for causality results.
    # The main export_to_csv handles general float formatting, but p-values might need higher precision.
    df_to_save = causality_results_df.copy()

    # Example: Ensure p_value is formatted with more precision if it exists
    # This is a bit redundant if export_to_csv handles all float columns with its float_format.
    # However, if different columns need different float precisions, this is where it would be handled.
    # For now, we assume export_to_csv's float_format is sufficient or can be adjusted.
    # If 'p_value' needs specific formatting different from other floats, that logic would go here.
    # For instance, if export_to_csv uses .2f, but p_value needs .4f:
    if 'p_value' in df_to_save.columns:
        df_to_save['p_value'] = df_to_save['p_value'].map(lambda x: f"{x:{float_format}}" if pd.notna(x) else "")
    if 'lag' in df_to_save.columns: # Lag is often integer, but ensure it's handled if float
        if df_to_save['lag'].dtype in [np.float32, np.float64]:
             df_to_save['lag'] = df_to_save['lag'].map(lambda x: f"{x:.0f}" if pd.notna(x) else "")

    # Call the main CSV export function
    # The float_format in export_to_csv will apply to any remaining float columns
    # that weren't specifically handled above.
    export_to_csv(df_to_save, output_path, float_format=float_format) # Pass the specific float_format for p-values
    # Original print from causality_analysis: print(f"Causality results saved to {output_path}")
    # This is now handled by export_to_csv.

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
