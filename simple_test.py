#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')
import pandas as pd

print("=== TESTE DIRETO DO METRIC FORMATTER ===")

# Test 1: Import
try:
    from src.utils.metric_formatter import detect_and_convert_units
    print("✓ Import OK")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: Load real data file
try:
    df = pd.read_csv('demo-data/demo-experiment-1-round/round-1/1 - Baseline/tenant-a/memory_usage.csv')
    print(f"✓ Dados carregados: {len(df)} registros")
    print(f"Valores originais (primeiros 3): {df['value'].head(3).tolist()}")
except Exception as e:
    print(f"❌ Erro ao carregar dados: {e}")
    exit(1)

# Test 3: Apply formatting
try:
    result = detect_and_convert_units(df, 'memory_usage')
    print("✓ Formatação aplicada")
    print(f"Unidade detectada: {result['display_unit'].iloc[0]}")
    print(f"Valores convertidos (primeiros 3): {result['value'].head(3).tolist()}")
    if 'formatted_value' in result.columns:
        print(f"Valores formatados: {result['formatted_value'].head(3).tolist()}")
except Exception as e:
    print(f"❌ Erro na formatação: {e}")
    exit(1)

print("🎉 TESTE CONCLUÍDO COM SUCESSO!")
