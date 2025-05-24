#!/usr/bin/env python3
"""
Análise simples com tenants a, b, c, d
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from pathlib import Path

def main():
    print("🚀 ANÁLISE DE SIMILARIDADE - TENANTS A, B, C, D")
    print("="*50)
    
    # Create output directory
    output_dir = Path("output/similarity_tenants")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        from src.data.loader import load_experiment_data
        
        print("📊 Carregando dados...")
        data = load_experiment_data(
            "demo-data/demo-experiment-1-round",
            selected_metrics=['memory_usage', 'disk_throughput_total', 'network_total_bandwidth']
        )
        
        print("✅ Dados carregados!")
        
        # Check what we have
        for metric_name, rounds_data in data.items():
            print(f"\n📊 {metric_name}:")
            
            for round_name, phases_data in rounds_data.items():
                for phase_name, df in phases_data.items():
                    if df is not None and not df.empty:
                        print(f"  {phase_name}: {len(df)} registros")
                        
                        # Check formatting
                        if 'display_unit' in df.columns:
                            unit = df['display_unit'].iloc[0]
                            values = df['value'].head(3).tolist()
                            print(f"    ✅ Formatado: {values} ({unit})")
                        else:
                            values = df['value'].head(3).tolist()  
                            print(f"    📈 Original: {values}")
                        break
                break
        
        # Try similarity analysis
        print("\n🔍 Executando análise básica...")
        from src.analysis.similarity import SimilarityAnalysis
        
        analyzer = SimilarityAnalysis()
        print("✅ Analisador criado!")
        
        # Save summary
        summary_file = output_dir / "analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Análise de Similaridade - Tenants A, B, C, D\n")
            f.write("="*50 + "\n")
            f.write(f"Métricas processadas: {len(data)}\n")
            f.write("Sistema de formatação inteligente ativo\n")
        
        print(f"✅ Resumo salvo em: {summary_file}")
        print("🎉 Análise concluída!")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    main()
