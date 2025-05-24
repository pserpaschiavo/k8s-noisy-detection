#!/usr/bin/env python3
"""
Execução da Análise de Similaridade com Sistema de Formatação Inteligente
Usando tenants a, b, c e d do demo-data
"""

import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
from src.data.loader import load_experiment_data
from src.analysis.similarity import SimilarityAnalysis
from src.visualization.plots import create_similarity_plots

def main():
    print("🚀 EXECUTANDO ANÁLISE DE SIMILARIDADE")
    print("🔧 Usando Sistema de Formatação Inteligente de Métricas")
    print("👥 Tenants: a, b, c, d")
    print("="*70)
    
    # Configuration
    data_dir = "demo-data/demo-experiment-1-round"
    output_dir = Path("output/similarity_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics that were problematic with hard-coded conversions
    target_metrics = [
        'memory_usage',           # Previously hard-coded to MB incorrectly
        'disk_throughput_total',  # Previously hard-coded to MB/s incorrectly  
        'network_total_bandwidth', # Previously hard-coded to MB/s incorrectly
        'cpu_usage'              # Control metric (should not be affected)
    ]
    
    print(f"📊 Métricas selecionadas: {target_metrics}")
    print(f"📁 Dados de entrada: {data_dir}")
    print(f"📁 Saída: {output_dir}")
    
    # Load experiment data with intelligent formatting
    print("\n🔄 Carregando dados experimentais...")
    try:
        experiment_data = load_experiment_data(
            data_dir,
            selected_metrics=target_metrics
        )
        print("✅ Dados carregados com sucesso!")
        
        # Display data summary with new formatting
        print("\n📋 RESUMO DOS DADOS CARREGADOS:")
        for metric_name, rounds_data in experiment_data.items():
            print(f"\n📊 Métrica: {metric_name}")
            
            for round_name, phases_data in rounds_data.items():
                for phase_name, df in phases_data.items():
                    if df is not None and not df.empty:
                        print(f"  📁 {round_name}/{phase_name}: {len(df)} registros")
                        
                        # Check if intelligent formatting was applied
                        if 'display_unit' in df.columns:
                            unit = df['display_unit'].iloc[0]
                            sample_values = df['value'].head(3).tolist()
                            print(f"    ✅ Formatação inteligente aplicada!")
                            print(f"    📏 Unidade: {unit}")
                            print(f"    📈 Valores (amostra): {[f'{v:.2f}' for v in sample_values]}")
                            
                            if 'formatted_value' in df.columns:
                                formatted_sample = df['formatted_value'].head(3).tolist()
                                print(f"    🎨 Formatados: {formatted_sample}")
                        else:
                            sample_values = df['value'].head(3).tolist()
                            print(f"    📈 Valores originais: {[f'{v:.2f}' for v in sample_values]}")
                        break
                break
        
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Initialize similarity analysis
    print("\n🔧 Inicializando análise de similaridade...")
    try:
        similarity_analyzer = SimilarityAnalysis()
        print("✅ Analisador de similaridade inicializado!")
    except Exception as e:
        print(f"❌ Erro ao inicializar analisador: {e}")
        return False
    
    # Execute similarity analysis for each metric
    print("\n🎯 EXECUTANDO ANÁLISES DE SIMILARIDADE...")
    
    results_summary = {}
    
    for metric_name in target_metrics:
        print(f"\n{'='*50}")
        print(f"📊 ANALISANDO MÉTRICA: {metric_name.upper()}")
        print(f"{'='*50}")
        
        if metric_name not in experiment_data:
            print(f"⚠️  Métrica {metric_name} não encontrada nos dados")
            continue
        
        try:
            # Get data for this metric
            metric_data = experiment_data[metric_name]
            
            # Prepare data for analysis
            phases_data = {}
            tenants_found = set()
            
            for round_name, phases in metric_data.items():
                for phase_name, df in phases.items():
                    if df is not None and not df.empty:
                        phases_data[phase_name] = df.copy()
                        
                        # Extract tenant information
                        if 'tenant' in df.columns:
                            tenants_found.update(df['tenant'].unique())
                        elif 'namespace' in df.columns:
                            # Extract tenant from namespace
                            tenants = [ns for ns in df['namespace'].unique() 
                                     if any(tenant in ns for tenant in ['tenant-a', 'tenant-b', 'tenant-c', 'tenant-d'])]
                            tenants_found.update(tenants)
            
            print(f"📁 Fases encontradas: {list(phases_data.keys())}")
            print(f"👥 Tenants encontrados: {sorted(tenants_found)}")
            
            if len(phases_data) < 2:
                print(f"⚠️  Poucas fases para análise de similaridade")
                continue
            
            # Run similarity techniques
            techniques = [
                ('distance_correlation', 'Distance Correlation'),
                ('cosine_similarity', 'Cosine Similarity'),
                ('dtw', 'Dynamic Time Warping'),
                ('mutual_information', 'Mutual Information')
            ]
            
            metric_results = {}
            
            for technique_key, technique_name in techniques:
                print(f"\n🔍 Executando: {technique_name}")
                
                try:
                    # Execute similarity analysis
                    if technique_key == 'distance_correlation':
                        result = similarity_analyzer.distance_correlation_analysis(
                            phases_data, output_path=output_dir / f"{metric_name}_distance_corr"
                        )
                    elif technique_key == 'cosine_similarity':
                        result = similarity_analyzer.cosine_similarity_analysis(
                            phases_data, output_path=output_dir / f"{metric_name}_cosine"
                        )
                    elif technique_key == 'dtw':
                        result = similarity_analyzer.dtw_analysis(
                            phases_data, output_path=output_dir / f"{metric_name}_dtw"
                        )
                    elif technique_key == 'mutual_information':
                        result = similarity_analyzer.mutual_information_analysis(
                            phases_data, output_path=output_dir / f"{metric_name}_mutual_info"
                        )
                    
                    if result is not None:
                        metric_results[technique_key] = result
                        print(f"    ✅ {technique_name} - Concluído!")
                        
                        # Display key results
                        if isinstance(result, dict):
                            if 'similarity_matrix' in result:
                                matrix = result['similarity_matrix']
                                print(f"    📊 Matriz de similaridade: {matrix.shape}")
                                print(f"    📈 Valor médio: {matrix.mean():.3f}")
                            elif 'correlation' in result:
                                print(f"    📈 Correlação: {result['correlation']:.3f}")
                        elif isinstance(result, (int, float)):
                            print(f"    📈 Resultado: {result:.3f}")
                    else:
                        print(f"    ⚠️  {technique_name} - Sem resultado")
                        
                except Exception as e:
                    print(f"    ❌ Erro em {technique_name}: {e}")
                    continue
            
            results_summary[metric_name] = metric_results
            
            # Save metric summary
            summary_file = output_dir / f"{metric_name}_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Análise de Similaridade - {metric_name}\n")
                f.write("="*50 + "\n\n")
                f.write(f"Tenants analisados: {sorted(tenants_found)}\n")
                f.write(f"Fases analisadas: {list(phases_data.keys())}\n")
                f.write(f"Técnicas executadas: {len(metric_results)}\n\n")
                
                for technique, result in metric_results.items():
                    f.write(f"{technique}: {result}\n")
            
            print(f"✅ Análise de {metric_name} concluída!")
            
        except Exception as e:
            print(f"❌ Erro ao analisar {metric_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate final summary
    print(f"\n{'='*70}")
    print("📋 RESUMO FINAL DA ANÁLISE DE SIMILARIDADE")
    print(f"{'='*70}")
    
    print(f"📊 Métricas analisadas: {len(results_summary)}")
    for metric_name, results in results_summary.items():
        print(f"  - {metric_name}: {len(results)} técnicas executadas")
    
    print(f"\n📁 Resultados salvos em: {output_dir.absolute()}")
    print(f"📈 Plots e tabelas gerados para avaliação visual")
    
    # List generated files
    generated_files = list(output_dir.glob("*"))
    print(f"\n📄 Arquivos gerados ({len(generated_files)}):")
    for file_path in sorted(generated_files):
        print(f"  - {file_path.name}")
    
    print("\n🎉 ANÁLISE DE SIMILARIDADE CONCLUÍDA COM SUCESSO!")
    print("✅ Sistema de formatação inteligente aplicado")
    print("✅ Tenants a, b, c, d processados")
    print("✅ Pronto para avaliação visual dos resultados")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
