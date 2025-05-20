\
import os

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

