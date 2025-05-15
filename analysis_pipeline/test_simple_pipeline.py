#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de teste direto para o módulo de análise causal corrigido
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    """Test main function"""
    print("Testando o módulo de análise causal corrigido...")
    
    # Criar séries de teste
    print("Criando dados de teste...")
    n = 100
    time_index = pd.date_range(start='2025-01-01', periods=n, freq='H')
    
    # Série X influencia Y com lag=2
    x = np.sin(np.linspace(0, 8*np.pi, n)) + np.random.normal(0, 0.2, n)
    y = np.roll(x, 2) + np.random.normal(0, 0.5, n)
    
    # Criar DataFrames
    df_x = pd.DataFrame({'value': x}, index=time_index)
    df_y = pd.DataFrame({'value': y}, index=time_index)
    
    # Pasta de saída
    output_dir = Path('/home/phil/Projects/k8s-noisy-detection/analysis_pipeline/test_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Importar o módulo causal corrigido
    try:
        sys.path.append('/home/phil/Projects/k8s-noisy-detection/analysis_pipeline')
        from analysis_pipeline.causal_fixed import CausalAnalysisFixed
        
        print("Módulo importado com sucesso!")
        
        # Criar instância do analisador
        causal_analyzer = CausalAnalysisFixed(output_dir)
        print("Analisador criado com sucesso!")
        
        # Executar teste de causalidade
        print("\nExecutando teste de causalidade Toda-Yamamoto...")
        result = causal_analyzer.toda_yamamoto_causality_test(
            x_series=df_x['value'],
            y_series=df_y['value'],
            max_lag=5,
            alpha=0.05
        )
        
        print("\nResultados do teste de causalidade:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        # Criar dados simulando estrutura do pipeline
        print("\nCriando estrutura simulada de dados por fase...")
        phase_data = {
            'baseline': {
                'component1': {
                    'metric1': df_x,
                    'metric2': df_y
                },
                'component2': {
                    'metric1': df_x * 1.5,
                    'metric2': df_y * 0.8
                }
            },
            'attack': {
                'component1': {
                    'metric1': df_x * 2,
                    'metric2': df_y * 2
                },
                'component2': {
                    'metric1': df_x * 3,
                    'metric2': df_y * 1.5
                }
            }
        }
        
        # Executar análise causal completa
        print("\nExecutando análise causal completa...")
        results_df = causal_analyzer.run_causal_analysis(
            phase_data=phase_data,
            method='toda-yamamoto',
            save_results=True
        )
        
        if not results_df.empty:
            print("\nResultados da análise causal completa:")
            print(results_df.head())
            print(f"\nTotal de resultados: {len(results_df)}")
        else:
            print("\nNenhum resultado obtido na análise causal.")
        
    except Exception as e:
        print(f"Erro ao executar o teste: {str(e)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())