#!/usr/bin/env python3
"""
Validation script to test the cleaned up k8s-noisy-detection codebase.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.abspath('.'))

def test_imports():
    """Test all critical imports."""
    print("🔍 Testing imports...")
    
    try:
        # Test config
        from src.config import DEFAULT_METRICS, TENANT_COLORS, VISUALIZATION_CONFIG
        print("✅ Config imported successfully")
        print(f"   - {len(DEFAULT_METRICS)} default metrics")
        print(f"   - {len(TENANT_COLORS)} tenant colors")
        
        # Test common utilities
        from src.utils.common import pd, np, plt, logger
        print("✅ Common utilities imported successfully")
        
        # Test data modules
        from src.data.loader import load_experiment_data, list_available_tenants
        from src.data.normalization import normalize_cpu_usage, full_normalization_pipeline
        from src.data.io_utils import export_to_csv
        print("✅ Data modules imported successfully")
        
        # Test analysis modules
        from src.analysis.descriptive_statistics import calculate_descriptive_statistics
        from src.analysis.correlation_covariance import calculate_inter_tenant_correlation_per_metric
        from src.analysis.similarity import calculate_pairwise_distance_correlation
        print("✅ Analysis modules imported successfully")
        
        # Test visualization
        from src.visualization.plots import plot_correlation_heatmap, set_publication_style
        print("✅ Visualization modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality with sample data."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        sample_data = pd.DataFrame({
            'tenant': ['A', 'A', 'B', 'B'] * 5,
            'cpu_usage': np.random.rand(20),
            'memory_usage': np.random.rand(20),
            'timestamp': pd.date_range('2024-01-01', periods=20, freq='1min')
        })
        
        from src.data.normalization import normalize_cpu_usage, clean_metrics_data
        
        # Test normalization
        normalized = normalize_cpu_usage(sample_data, total_cpu_cores=4)
        print("✅ CPU normalization works")
        
        # Test data cleaning
        cleaned = clean_metrics_data(sample_data)
        print("✅ Data cleaning works")
        
        # Test descriptive statistics
        from src.analysis.descriptive_statistics import calculate_descriptive_statistics
        stats = calculate_descriptive_statistics(sample_data, ['cpu_usage', 'memory_usage'])
        print("✅ Descriptive statistics calculation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_structure():
    """Test that the directory structure is correct."""
    print("\n📁 Testing directory structure...")
    
    expected_dirs = [
        'src',
        'src/analysis',
        'src/data', 
        'src/utils',
        'src/visualization',
        'tests',
        'demo-data'
    ]
    
    expected_files = [
        'src/main.py',
        'src/config.py',
        'src/utils/common.py',
        'requirements.txt',
        'setup.py'
    ]
    
    missing_dirs = [d for d in expected_dirs if not os.path.isdir(d)]
    missing_files = [f for f in expected_files if not os.path.isfile(f)]
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ Directory structure is correct")
    return True


def main():
    """Run all validation tests."""
    print("🚀 k8s-noisy-detection Cleanup Validation")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_imports,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Cleanup was successful!")
        print("\n✨ The codebase is now optimized and ready for use!")
    else:
        print("⚠️  Some tests failed. Manual review needed.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
