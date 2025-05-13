#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test main.py with tenant comparison feature
"""
import os
import sys
import logging
import subprocess

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logging.info("Testing main.py with tenant comparison feature")

# Set up environment variables
os.environ['PYTHONPATH'] = os.path.abspath(os.path.dirname(__file__))

# Run main.py with tenant comparison option
cmd = [
    "python", "main.py",
    "--experiment", "2025-05-11/16-58-00/default-experiment-1",
    "--tenant-comparison",
    "--colorblind-friendly",  # Usar o modo amigável para daltônicos
    "--output", "/home/phil/Projects/k8s-noisy-lab-data-pipe/analysis_pipeline/results/main_tenant_test",
    "--components", "tenant-a", "tenant-b", "tenant-c", "tenant-d"
]

logging.info(f"Running command: {' '.join(cmd)}")
try:
    result = subprocess.run(cmd, 
                          capture_output=True, 
                          text=True,
                          cwd=os.path.dirname(os.path.abspath(__file__)))
    
    logging.info(f"Command exit code: {result.returncode}")
    
    if result.stdout:
        logging.info(f"Command output:\n{result.stdout}")
    
    if result.stderr:
        logging.warning(f"Command error output:\n{result.stderr}")
    
    # Check the output files
    output_dir = "/home/phil/Projects/k8s-noisy-lab-data-pipe/analysis_pipeline/results/main_tenant_test"
    tenant_dir = os.path.join(output_dir, "plots/tenant_comparison")
    
    if os.path.exists(tenant_dir):
        files = os.listdir(tenant_dir)
        logging.info(f"Found {len(files)} files in {tenant_dir}: {files}")
    else:
        logging.error(f"Tenant comparison directory not created: {tenant_dir}")
except Exception as e:
    logging.error(f"Error running main.py: {e}")
    sys.exit(1)

logging.info("Test completed")
