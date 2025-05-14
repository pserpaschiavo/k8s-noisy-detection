#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd

print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")

try:
    import ruptures
    print(f"Ruptures: {ruptures.__version__}")
    print("Ruptures is available")
except ImportError as e:
    print(f"Error importing ruptures: {e}")
    sys.exit(1)

print("Creating test data")
# Create synthetic test data
np.random.seed(42)
x = np.concatenate([np.random.normal(0, 1, 50), np.random.normal(5, 1, 50)])
print(f"Data shape: {x.shape}")

print("Running ruptures algorithm")
# Test ruptures functionality
algo = ruptures.Pelt(model="rbf").fit(x.reshape(-1, 1))
change_points = algo.predict(pen=10)
print(f"Detected change points: {change_points}")

print("Test successful!")
