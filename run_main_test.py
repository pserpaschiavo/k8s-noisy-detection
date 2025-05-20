#!/usr/bin/env python3
"""
Test script to run the main pipeline with the fixed data loader
"""

import os
import sys
import subprocess

def run_pipeline():
    """Run the main pipeline with basic arguments"""
    print("Testing main pipeline with fixed data loader")
    
    # Set the path to the demo data
    data_dir = "demo-data/demo-experiment-3-rounds"
    
    # Verify the data directory exists
    if not os.path.exists(data_dir):
        data_dir = os.path.join(os.getcwd(), "demo-data/demo-experiment-3-rounds")
        if not os.path.exists(data_dir):
            print(f"Error: Data directory not found at {data_dir}")
            return 1
    
    print(f"Using data directory: {os.path.abspath(data_dir)}")
    
    # Run the main pipeline with minimal options
    cmd = [
        "python", "-m", "pipeline.main",
        "--data-dir", data_dir,
        "--output-dir", "test_output",
        "--skip-advanced-analyses"  # Skip advanced analyses to make the test faster
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"Pipeline completed with return code: {result.returncode}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed with error: {e}")
        return e.returncode

if __name__ == "__main__":
    sys.exit(run_pipeline())
