#!/usr/bin/env python3
import os
import sys
import pandas as pd
import time

print("Script starting")
print(f"Python version: {sys.version}")
print(f"Current dir: {os.getcwd()}")

try:
    print("Importing module...")
    from pipeline.data_processing.data_loader import load_experiment_data
    print("Module imported successfully")
except Exception as e:
    print(f"Error importing module: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("Checking directory structure...")
    data_dir = '/home/phil/Projects/k8s-noisy-detection/demo-data/demo-experiment-3-rounds'
    print(f'Using data directory: {data_dir}')
    print(f'Directory exists: {os.path.exists(data_dir)}')
    if os.path.exists(data_dir):
        print(f'Directory contents: {os.listdir(data_dir)}')
        
        # Check first round
        round_dir = os.path.join(data_dir, 'round-1')
        print(f'Round dir exists: {os.path.exists(round_dir)}')
        if os.path.exists(round_dir):
            print(f'Round dir contents: {os.listdir(round_dir)}')
    else:
        print("Data directory does not exist!")

    # Try loading a single CSV file to check
    print("\nTrying to load a single CSV file...")
    try:
        csv_path = os.path.join(data_dir, 'round-1', '1 - Baseline', 'tenant-a', 'cpu_usage.csv')
        print(f'CSV path: {csv_path}')
        print(f'CSV file exists: {os.path.exists(csv_path)}')
        if os.path.exists(csv_path):
            print("Reading CSV with pandas...")
            df = pd.read_csv(csv_path)
            print(f"CSV read successfully: {df.shape} rows and columns")
            print(f"CSV columns: {df.columns.tolist()}")
            print(f"First few rows:\n{df.head()}")
    except Exception as e:
        print(f"Error reading CSV: {e}")

    print("\nStarting full data loading...")
    try:
        print("Calling load_experiment_data...")
        start_time = time.time()
        metrics_data = load_experiment_data(data_dir)
        end_time = time.time()
        print(f"Data loading completed in {end_time - start_time:.2f} seconds")
        print(f'Metrics loaded: {list(metrics_data.keys())}')
    except Exception as e:
        print(f"Error during load_experiment_data: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"General error: {e}")
    import traceback
    traceback.print_exc()
    
print("Script finished")
