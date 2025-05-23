#!/usr/bin/env python3
"""
Script para testar se todas as importações estão funcionando corretamente.
"""
import sys
import os

# Ajusta o path para importar módulos da raiz do projeto
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from new_config import TENANT_COLORS, METRICS_CONFIG
    print("✓ Importação de new_config bem-sucedida")
    
    from data_handling.loader import load_experiment_data
    print("✓ Importação de data_handling.loader bem-sucedida")
    
    from data_handling.save_results import export_to_csv
    print("✓ Importação de data_handling.save_results bem-sucedida")
    
    from analysis_modules.multivariate_exploration import perform_pca, perform_ica
    print("✓ Importação de analysis_modules.multivariate_exploration bem-sucedida")
    
    from analysis_modules.descritive_statistics import calculate_descriptive_statistics
    print("✓ Importação de analysis_modules.descritive_statistics bem-sucedida")
    
    from analysis_modules.correlation_covariance import calculate_inter_tenant_correlation_per_metric
    print("✓ Importação de analysis_modules.correlation_covariance bem-sucedida")
    
    from analysis_modules.causality import perform_sem_analysis
    print("✓ Importação de analysis_modules.causality bem-sucedida")
    
    from analysis_modules.root_cause import RootCauseAnalyzer
    print("✓ Importação de analysis_modules.root_cause bem-sucedida")
    
    from analysis_modules.similarity import calculate_pairwise_distance_correlation
    print("✓ Importação de analysis_modules.similarity bem-sucedida")
    
    from visualization.new_plots import plot_correlation_heatmap
    print("✓ Importação de visualization.new_plots bem-sucedida")
    
    print("\nTodas as importações estão funcionando corretamente!")
    
except Exception as e:
    print(f"ERRO: {e}")
    import traceback
    print(traceback.format_exc())
