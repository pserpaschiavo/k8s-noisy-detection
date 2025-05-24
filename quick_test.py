#!/usr/bin/env python3
"""
Teste rápido e simples para validar o MetricFormatter.
"""

import sys
sys.path.insert(0, '.')

try:
    print("🔄 Testando imports...")
    from src.utils.metric_formatter import MetricFormatter, detect_and_convert_units
    import pandas as pd
    print("✓ Imports OK")

    print("🔄 Testando funcionamento básico...")
    df = pd.DataFrame({
        'value': [1048576, 2097152], 
        'timestamp': pd.date_range('2024-01-01', periods=2, freq='1min')
    })
    
    result = detect_and_convert_units(df, 'memory_usage')
    print("✓ Formatação básica OK")
    print(f"Valores convertidos: {result['value'].tolist()}")
    print(f"Unidade: {result['display_unit'].iloc[0]}")
    
    print("🔄 Testando integração com loader...")
    from src.data.loader import load_experiment_data
    print("✓ Loader import OK")
    
    print("🎉 TODOS OS TESTES PASSARAM!")
    
except Exception as e:
    print(f"❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
