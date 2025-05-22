#!/usr/bin/env python3
"""
Script de teste completo para todas as técnicas de análise causal implementadas.
Testa Transfer Entropy, CCM, Granger Causality e SEM.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from refactor.analysis_modules.causality import (
    # Transfer Entropy
    calculate_transfer_entropy,
    calculate_pairwise_transfer_entropy,
    plot_transfer_entropy_heatmap,
    plot_transfer_entropy_network,
    
    # CCM
    calculate_pairwise_ccm,
    plot_ccm_convergence,
    summarize_ccm_results,
    plot_ccm_causality_heatmap,
    
    # Granger Causality
    calculate_pairwise_granger_causality,
    plot_granger_causality_heatmap,
    plot_granger_causality_network,
    
    # SEM
    perform_sem_analysis,
    plot_sem_path_diagram,
    plot_sem_fit_indices,
    
    # Comparison
    compare_causal_analysis_methods
)

def generate_test_data(n_samples=200, noise_level=0.1):
    """
    Gera dados sintéticos com relações causais conhecidas para teste.
    
    X1 -> X2 -> X3 (cadeia causal)
    X4 é independente (controle)
    """
    np.random.seed(42)
    
    # Série temporal base
    time = np.linspace(0, 10, n_samples)
    
    # X1: série independente com tendência
    X1 = np.sin(time) + 0.5 * np.cos(2*time) + noise_level * np.random.randn(n_samples)
    
    # X2: causado por X1 com delay
    X2 = np.zeros(n_samples)
    for i in range(1, n_samples):
        X2[i] = 0.7 * X1[i-1] + 0.3 * X2[i-1] + noise_level * np.random.randn()
    
    # X3: causado por X2 com delay
    X3 = np.zeros(n_samples)
    for i in range(1, n_samples):
        X3[i] = 0.8 * X2[i-1] + 0.2 * X3[i-1] + noise_level * np.random.randn()
    
    # X4: independente (controle)
    X4 = np.random.randn(n_samples) * 0.5
    
    # Criar DataFrame
    df = pd.DataFrame({
        'time': time,
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4
    })
    
    return df

def test_transfer_entropy():
    """Testa a implementação de Transfer Entropy."""
    print("="*60)
    print("TESTANDO TRANSFER ENTROPY")
    print("="*60)
    
    # Gerar dados de teste
    df = generate_test_data(n_samples=150)
    
    try:
        # Teste 1: Transfer Entropy básico
        print("\n1. Testando calculate_transfer_entropy()...")
        te_value = calculate_transfer_entropy(df['X1'].values, df['X2'].values, k=1)
        print(f"   TE(X1 -> X2): {te_value:.4f}")
        
        te_value_reverse = calculate_transfer_entropy(df['X2'].values, df['X1'].values, k=1)
        print(f"   TE(X2 -> X1): {te_value_reverse:.4f}")
        
        # Teste 2: Transfer Entropy pareado
        print("\n2. Testando calculate_pairwise_transfer_entropy()...")
        te_results = calculate_pairwise_transfer_entropy(df, time_col='time')
        
        print(f"   Encontradas {len(te_results)} relações:")
        for key, result in list(te_results.items())[:6]:  # Mostrar apenas as primeiras 6
            print(f"   {result['direction']}: TE = {result['transfer_entropy']:.4f}")
        
        # Teste 3: Visualizações
        print("\n3. Testando visualizações...")
        output_dir = "/tmp/test_causality"
        os.makedirs(output_dir, exist_ok=True)
        
        # Heatmap
        plot_transfer_entropy_heatmap(
            te_results,
            title="Test Transfer Entropy Heatmap",
            output_dir=output_dir,
            filename="te_heatmap_test.png",
            threshold=0.001
        )
        print("   Heatmap criado: /tmp/test_causality/te_heatmap_test.png")
        
        # Network
        plot_transfer_entropy_network(
            te_results,
            title="Test Transfer Entropy Network",
            output_dir=output_dir,
            filename="te_network_test.png",
            threshold=0.001
        )
        print("   Network criado: /tmp/test_causality/te_network_test.png")
        
        print("\n✅ Transfer Entropy: TODOS OS TESTES PASSARAM!")
        return True
        
    except Exception as e:
        print(f"\n❌ Transfer Entropy: ERRO - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ccm():
    """Testa a implementação de CCM."""
    print("\n" + "="*60)
    print("TESTANDO CONVERGENT CROSS MAPPING (CCM)")
    print("="*60)
    
    # Gerar dados de teste
    df = generate_test_data(n_samples=100)  # CCM precisa de menos dados
    
    try:
        print("\n1. Testando calculate_pairwise_ccm()...")
        ccm_results = calculate_pairwise_ccm(df, time_col='time', num_predictions=50)
        
        print(f"   Encontrados {len(ccm_results)} pares de variáveis")
        for pair, results_df in list(ccm_results.items())[:3]:
            print(f"   Par {pair}: {len(results_df)} resultados")
        
        print("\n2. Testando visualizações CCM...")
        output_dir = "/tmp/test_causality"
        
        # Convergence plot
        plot_ccm_convergence(
            ccm_results,
            title="Test CCM Convergence",
            output_dir=output_dir,
            filename="ccm_convergence_test.png"
        )
        print("   Convergence plot criado: /tmp/test_causality/ccm_convergence_test.png")
        
        # Summarize results
        ccm_summary = summarize_ccm_results(ccm_results, min_library_size=20)
        print(f"   CCM summary matrix shape: {ccm_summary.shape}")
        
        # Heatmap
        plot_ccm_causality_heatmap(
            ccm_summary,
            title="Test CCM Heatmap",
            output_dir=output_dir,
            filename="ccm_heatmap_test.png"
        )
        print("   Heatmap criado: /tmp/test_causality/ccm_heatmap_test.png")
        
        print("\n✅ CCM: TODOS OS TESTES PASSARAM!")
        return True
        
    except Exception as e:
        print(f"\n❌ CCM: ERRO - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_granger():
    """Testa a implementação de Granger Causality."""
    print("\n" + "="*60)
    print("TESTANDO GRANGER CAUSALITY")
    print("="*60)
    
    # Gerar dados de teste
    df = generate_test_data(n_samples=100)
    
    try:
        print("\n1. Testando calculate_pairwise_granger_causality()...")
        granger_results = calculate_pairwise_granger_causality(df, time_col='time')
        
        print("   Resultados obtidos:")
        print(f"   - P-values matrix shape: {granger_results['p_values'].shape}")
        print(f"   - F-statistics matrix shape: {granger_results['f_statistics'].shape}")
        
        # Mostrar algumas relações significativas
        p_values = granger_results['p_values']
        significant = (p_values < 0.05) & (p_values > 0)
        print(f"   - Relações significativas (p < 0.05): {significant.sum().sum()}")
        
        print("\n2. Testando visualizações Granger...")
        output_dir = "/tmp/test_causality"
        
        # Heatmap
        plot_granger_causality_heatmap(
            granger_results,
            title="Test Granger Heatmap",
            output_dir=output_dir,
            filename="granger_heatmap_test.png"
        )
        print("   Heatmap criado: /tmp/test_causality/granger_heatmap_test.png")
        
        # Network
        plot_granger_causality_network(
            granger_results,
            title="Test Granger Network",
            output_dir=output_dir,
            filename="granger_network_test.png"
        )
        print("   Network criado: /tmp/test_causality/granger_network_test.png")
        
        print("\n✅ Granger Causality: TODOS OS TESTES PASSARAM!")
        return True
        
    except Exception as e:
        print(f"\n❌ Granger Causality: ERRO - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sem():
    """Testa a implementação de SEM."""
    print("\n" + "="*60)
    print("TESTANDO STRUCTURAL EQUATION MODELING (SEM)")
    print("="*60)
    
    # Gerar dados de teste
    df = generate_test_data(n_samples=200)
    
    try:
        print("\n1. Testando perform_sem_analysis()...")
        
        # Modelo simples: X2 ~ X1, X3 ~ X2
        model_spec = """
        X2 ~ X1
        X3 ~ X2
        """
        
        sem_results = perform_sem_analysis(
            df[['X1', 'X2', 'X3', 'X4']], 
            model_spec
        )
        
        print("   SEM model fitted successfully!")
        print(f"   Estimates shape: {sem_results['estimates'].shape}")
        print("   Model fit statistics:")
        for key, value in sem_results['stats'].items():
            if isinstance(value, (int, float)):
                print(f"   - {key}: {value:.4f}")
        
        print("\n2. Testando visualizações SEM...")
        output_dir = "/tmp/test_causality"
        
        # Path diagram
        plot_sem_path_diagram(
            sem_results,
            title="Test SEM Path Diagram",
            output_dir=output_dir,
            filename="sem_path_test.png"
        )
        print("   Path diagram criado: /tmp/test_causality/sem_path_test.png")
        
        # Fit indices
        plot_sem_fit_indices(
            sem_results,
            title="Test SEM Fit Indices",
            output_dir=output_dir,
            filename="sem_fit_test.png"
        )
        print("   Fit indices plot criado: /tmp/test_causality/sem_fit_test.png")
        
        print("\n✅ SEM: TODOS OS TESTES PASSARAM!")
        return True
        
    except Exception as e:
        print(f"\n❌ SEM: ERRO - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comparison():
    """Testa a função de comparação entre métodos."""
    print("\n" + "="*60)
    print("TESTANDO COMPARAÇÃO ENTRE MÉTODOS")
    print("="*60)
    
    # Gerar dados de teste
    df = generate_test_data(n_samples=150)
    
    try:
        print("\n1. Testando compare_causal_analysis_methods()...")
        
        # SEM model specification
        sem_model = """
        X2 ~ X1
        X3 ~ X2
        """
        
        comparison_results = compare_causal_analysis_methods(
            df, 
            time_col='time',
            output_dir="/tmp/test_causality",
            sem_model_spec=sem_model
        )
        
        print("   Comparison completed!")
        print("   Results summary:")
        for method, results in comparison_results.items():
            if results:
                print(f"   - {method}: ✅ Results available")
            else:
                print(f"   - {method}: ❌ No results")
        
        print("   Comparison plot criado: /tmp/test_causality/causal_analysis_comparison.png")
        
        print("\n✅ COMPARAÇÃO: TODOS OS TESTES PASSARAM!")
        return True
        
    except Exception as e:
        print(f"\n❌ COMPARAÇÃO: ERRO - {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Executa todos os testes."""
    print("INICIANDO TESTES COMPLETOS DO MÓDULO CAUSALITY")
    print("=" * 80)
    
    # Lista de testes
    tests = [
        ("Transfer Entropy", test_transfer_entropy),
        ("CCM", test_ccm),
        ("Granger Causality", test_granger),
        ("SEM", test_sem),
        ("Comparison", test_comparison)
    ]
    
    results = {}
    
    # Executar cada teste
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ ERRO CRÍTICO em {test_name}: {e}")
            results[test_name] = False
    
    # Relatório final
    print("\n" + "="*80)
    print("RELATÓRIO FINAL DOS TESTES")
    print("="*80)
    
    total_tests = len(tests)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"{test_name:20s}: {status}")
    
    print(f"\nRESUMO: {passed_tests}/{total_tests} testes passaram")
    
    if passed_tests == total_tests:
        print("\n🎉 TODOS OS TESTES PASSARAM! O módulo causality.py está funcionando corretamente.")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} teste(s) falharam. Verifique os erros acima.")
    
    print(f"\nArquivos de teste salvos em: /tmp/test_causality/")

if __name__ == "__main__":
    main()
