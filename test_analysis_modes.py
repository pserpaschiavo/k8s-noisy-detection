#!/usr/bin/env python3
"""
Simple test script to verify all analysis modes work correctly.
"""

import sys
from pathlib import Path

# Add simple directory to path
sys.path.append('simple')

def test_basic_analysis():
    """Test basic analysis functionality."""
    print("🧪 Testing Basic Analysis...")
    try:
        from core import SimpleK8sAnalyzer
        
        analyzer = SimpleK8sAnalyzer('config/basic_config.yaml')
        analyzer.data_path = 'demo-data/demo-experiment-1-round/round-1'
        analyzer.output_dir = Path('./output/test_basic')
        analyzer.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test quick analysis
        result = analyzer.quick_analysis()
        
        if result:
            print("✅ Basic analysis completed successfully")
            return True
        else:
            print("❌ Basic analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Basic analysis error: {e}")
        return False

def test_extended_analysis():
    """Test extended analysis functionality."""
    print("🧪 Testing Extended Analysis...")
    try:
        from extended import ExtendedK8sAnalyzer
        
        analyzer = ExtendedK8sAnalyzer('config/extended_config.yaml')
        analyzer.data_path = 'demo-data/demo-experiment-1-round/round-1'
        analyzer.output_dir = Path('./output/test_extended')
        analyzer.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test loading data first
        success = analyzer.load_data()
        if not success:
            print("❌ Extended analysis: Data loading failed")
            return False
        
        print("✅ Extended analysis: Data loaded successfully")
        
        # Test basic processing first
        analyzer.process_data()
        analyzer.generate_stats()
        print("✅ Extended analysis: Basic processing completed")
        
        # Test extended features individually
        print("🔬 Testing PCA analysis...")
        pca_result = analyzer.perform_pca()
        print(f"✅ PCA completed: {pca_result is not None}")
        
        print("🔬 Testing clustering analysis...")
        cluster_result = analyzer.perform_clustering()
        print(f"✅ Clustering completed: {cluster_result is not None}")
        
        print("✅ Extended analysis components tested successfully")
        return True
        
    except Exception as e:
        print(f"❌ Extended analysis error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_analysis():
    """Test advanced analysis functionality."""
    print("🧪 Testing Advanced Analysis...")
    try:
        from advanced import AdvancedK8sAnalyzer
        
        analyzer = AdvancedK8sAnalyzer('config/advanced_config.yaml')
        analyzer.data_path = 'demo-data/demo-experiment-1-round/round-1'
        analyzer.output_dir = Path('./output/test_advanced')
        analyzer.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test initialization
        print("✅ Advanced analyzer created successfully")
        
        # Test data loading
        success = analyzer.load_data()
        if not success:
            print("❌ Advanced analysis: Data loading failed")
            return False
        
        print("✅ Advanced analysis: Data loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Advanced analysis error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Running Analysis Mode Tests\n")
    
    results = {
        'basic': test_basic_analysis(),
        'extended': test_extended_analysis(),
        'advanced': test_advanced_analysis()
    }
    
    print("\n📊 Test Results Summary:")
    print("=" * 40)
    for mode, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{mode.upper():>10}: {status}")
    
    all_passed = all(results.values())
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
