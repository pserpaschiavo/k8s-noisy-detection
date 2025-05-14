#!/usr/bin/env python3
import sys
import os

print("Python path:", sys.path)
print("Current directory:", os.getcwd())
print("Script location:", __file__)

try:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    print("Updated Python path:", sys.path)
    from analysis_pipeline.tenant_degradation_analysis import TenantDegradationAnalyzer
    print("Successfully imported TenantDegradationAnalyzer from analysis_pipeline package")
except ImportError as e:
    print("Import error (package):", e)
    try:
        from tenant_degradation_analysis import TenantDegradationAnalyzer
        print("Successfully imported TenantDegradationAnalyzer directly")
    except ImportError as e:
        print("Import error (direct):", e)

try:
    output_dir = "../output/debug_test"
    os.makedirs(output_dir, exist_ok=True)
    analyzer = TenantDegradationAnalyzer(output_dir)
    print(f"Successfully created analyzer with output dir: {output_dir}")
    print(f"Analyzer paths: plots_dir={analyzer.plots_dir}, results_dir={analyzer.results_dir}")
except Exception as e:
    print("Error creating analyzer:", e)
    
print("Script completed")
