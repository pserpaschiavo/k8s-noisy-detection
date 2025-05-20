import os
import pandas as pd

def get_experiment_data_dir(data_dir_input: str) -> str:
    """
    Resolves the absolute path to the experiment data directory.
    It checks if the input is an absolute path, or constructs it
    relative to the current working directory, also trying a 'demo-data' prefix.
    """
    experiment_data_dir = ""
    if os.path.isabs(data_dir_input):
        experiment_data_dir = data_dir_input
    else:
        cwd = os.getcwd()
        path_from_cwd = os.path.join(cwd, data_dir_input)
        # Attempt to join with "demo-data" if not already prefixed and path_from_cwd doesn't exist
        path_from_cwd_with_demodata_prefix = os.path.join(cwd, "demo-data", data_dir_input)

        if os.path.isdir(path_from_cwd):
            experiment_data_dir = path_from_cwd
        elif not data_dir_input.startswith("demo-data") and os.path.isdir(path_from_cwd_with_demodata_prefix):
            experiment_data_dir = path_from_cwd_with_demodata_prefix
        else:
            # Default to path_from_cwd if the demo-data prefixed one also doesn't exist or wasn't applicable
            experiment_data_dir = path_from_cwd
            
    return os.path.normpath(experiment_data_dir)

def add_experiment_elapsed_time(df, time_col: str = 'timestamp'):
    """
    Adds an 'elapsed_time_s' column to the DataFrame based on the time_col.
    The elapsed time is calculated in seconds from the first timestamp in the DataFrame.
    If time_col is not present or df is empty, returns the original DataFrame.
    
    This function handles both DataFrame and Series objects.
    """
    # Handle the case when df is None
    if df is None:
        print(f"Warning: Input is None. Cannot add elapsed time.")
        return df

    # Handle the case when df is a Series
    if isinstance(df, pd.Series):
        print(f"Warning: Input is a Series, not a DataFrame. Cannot add elapsed time.")
        return df
    
    # Handle the case when df is a DataFrame but empty or missing the time column
    if df.empty:
        print(f"Warning: DataFrame is empty. Cannot add elapsed time.")
        return df
        
    if time_col not in df.columns:
        print(f"Warning: '{time_col}' not found in DataFrame. Cannot add elapsed time.")
        return df

    # Convert timestamp column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except Exception as e:
            print(f"Warning: Could not convert column '{time_col}' to datetime: {e}. Cannot add elapsed time.")
            return df
    
    # Make a copy of the DataFrame to avoid modifying the original
    df = df.copy()
    
    # Sort by timestamp
    df.sort_values(by=time_col, inplace=True)
    
    # Calculate elapsed time in seconds from the first timestamp
    start_time = df[time_col].iloc[0]
    df['elapsed_time_s'] = (df[time_col] - start_time).dt.total_seconds()
    
    return df

def add_phase_markers(df: pd.DataFrame, phase_column: str = 'phase', phase_display_names: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    Identifies phase changes and returns the DataFrame with phase information
    and a dictionary of phase markers (start and end times or indices).
    Placeholder implementation.
    """
    if df is None or df.empty or phase_column not in df.columns:
        print(f"Warning: DataFrame is empty or phase column '{phase_column}' not found. Cannot add phase markers.")
        return df, {}

    # This is a very basic placeholder. A real implementation would identify
    # actual start/end times or indices of phases.
    phase_markers = {}
    if phase_display_names is None:
        phase_display_names = {}

    unique_phases = df[phase_column].unique()
    for phase in unique_phases:
        phase_name = phase_display_names.get(phase, phase)
        phase_indices = df[df[phase_column] == phase].index
        if not phase_indices.empty:
            phase_markers[phase_name] = {'start_index': phase_indices.min(), 'end_index': phase_indices.max()}
        else:
            phase_markers[phase_name] = {'start_index': None, 'end_index': None}
            
    print(f"Placeholder: add_phase_markers called for column '{phase_column}'. Markers: {phase_markers}")
    return df, phase_markers

