#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '.')

# Redirect output to a file to capture any hanging issues
import io
from contextlib import redirect_stdout, redirect_stderr

output_buffer = io.StringIO()
error_buffer = io.StringIO()

try:
    with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
        print("=== TESTE DE PIPELINE COMPLETO ===")
        
        # Import test
        from src.utils.metric_formatter import detect_and_convert_units, MetricFormatter
        print("✓ MetricFormatter imported")
        
        # Real data test
        import pandas as pd
        df = pd.read_csv('demo-data/demo-experiment-1-round/round-1/1 - Baseline/tenant-a/memory_usage.csv')
        print(f"✓ Data loaded: {len(df)} records")
        
        # Format test
        result = detect_and_convert_units(df.head(10), 'memory_usage')
        print(f"✓ Formatting applied")
        print(f"Original: {df['value'].head(3).tolist()}")
        print(f"Converted: {result['value'].head(3).tolist()}")
        print(f"Unit: {result['display_unit'].iloc[0]}")
        
        # Integration test
        from src.data.loader import load_experiment_data
        print("✓ Loader imported")
        
        print("🎉 ALL TESTS PASSED!")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Write results to file
with open('/tmp/test_results.txt', 'w') as f:
    f.write("STDOUT:\n")
    f.write(output_buffer.getvalue())
    f.write("\nSTDERR:\n")
    f.write(error_buffer.getvalue())

print("Test completed - check /tmp/test_results.txt")
