#!/usr/bin/env python3
"""
Minimal test to isolate metric formatter issues.
"""

print("Starting minimal test...")

try:
    import pandas as pd
    print("✓ Pandas imported")
    
    import numpy as np
    print("✓ Numpy imported")
    
    import sys
    sys.path.insert(0, '.')
    print("✓ Path configured")
    
    # Try importing just the module without instantiating
    import src.utils.metric_formatter as mf
    print("✓ Module imported")
    
    # Try getting the class
    MetricFormatter = mf.MetricFormatter
    print("✓ Class referenced")
    
    # Try instantiating
    formatter = MetricFormatter()
    print("✓ Class instantiated")
    
    # Test simple functionality
    metric_type = formatter.detect_metric_type('memory_usage')
    print(f"✓ Method called successfully: {metric_type}")
    
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
