#!/usr/bin/env python3
"""
Teste com dados reais do demo-data.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from src.utils.metric_formatter import detect_and_convert_units
from src.data.loader import load_experiment_data

print("🔄 Carregando dados reais...")

try:
    # Load real demo data
    data = load_experiment_data(
        "demo-data/demo-experiment-1-round",
        selected_metrics=['memory_usage', 'disk_throughput_total', 'network_total_bandwidth']
    )
    
    print("✓ Dados carregados com sucesso!")
    
    # Check what we got
    for metric_name, rounds_data in data.items():
        print(f"\n📊 Métrica: {metric_name}")
        
        for round_name, phases_data in rounds_data.items():
            for phase_name, df in phases_data.items():
                if df is not None and not df.empty:
                    print(f"  {round_name}/{phase_name}: {len(df)} registros")
                    
                    # Check for formatting columns
                    if 'display_unit' in df.columns:
                        unit = df['display_unit'].iloc[0]
                        values = df['value'].head(3).tolist()
                        print(f"    ✓ Formatado: {values} ({unit})")
                    else:
                        values = df['value'].head(3).tolist()
                        print(f"    - Original: {values}")
                    break  # Just check first phase
            break  # Just check first round
    
    print("\n🎉 TESTE COM DADOS REAIS CONCLUÍDO!")
    
except Exception as e:
    print(f"❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
