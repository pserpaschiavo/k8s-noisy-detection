#!/usr/bin/env python3
"""
Script para depurar importações no new_main.py
"""
import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    # Importar cada módulo separadamente para encontrar o problema
    print("Importando argparse...")
    import argparse
    
    print("Importando logging...")
    import logging
    
    print("Importando new_config...")
    from new_config import TENANT_COLORS, METRICS_CONFIG
    
    print("Importando data_handling.loader...")
    from data_handling.loader import load_experiment_data
    
    print("Importando data_handling.save_results...")
    from data_handling.save_results import export_to_csv
    
    print("Importando analysis_modules.multivariate_exploration...")
    from analysis_modules.multivariate_exploration import (
        perform_pca, perform_ica, get_top_features_per_component
    )
    
    print("Importando KPCA...")
    from analysis_modules.multivariate_exploration import perform_kpca
    
    print("Importando t-SNE...")
    from analysis_modules.multivariate_exploration import perform_tsne
    
    print("Importando analysis_modules.descritive_statistics...")
    from analysis_modules.descritive_statistics import calculate_descriptive_statistics
    
    print("Importando analysis_modules.correlation_covariance...")
    from analysis_modules.correlation_covariance import calculate_inter_tenant_correlation_per_metric, calculate_inter_tenant_covariance_per_metric
    
    print("Importando analysis_modules.causality...")
    from analysis_modules.causality import perform_sem_analysis, plot_sem_path_diagram, plot_sem_fit_indices
    
    print("Importando analysis_modules.root_cause...")
    from analysis_modules.root_cause import RootCauseAnalyzer, perform_complete_root_cause_analysis
    
    print("Importando analysis_modules.similarity...")
    from analysis_modules.similarity import (
        calculate_pairwise_distance_correlation,
        calculate_pairwise_cosine_similarity,
        calculate_pairwise_mutual_information,
        plot_distance_correlation_heatmap,
        plot_mutual_information_heatmap
    )
    
    print("Configurando aliases...")
    plot_cosine_similarity_heatmap = plot_distance_correlation_heatmap
    
    print("Importando visualization.new_plots...")
    from visualization.new_plots import (
        plot_correlation_heatmap, plot_covariance_heatmap, plot_scatter_comparison,
        plot_pca_explained_variance, plot_pca_biplot, plot_pca_loadings_heatmap,
        plot_ica_components_heatmap, plot_ica_scatter,
        plot_descriptive_stats_boxplot, plot_descriptive_stats_lineplot
    )
    
    print("Todas as importações foram bem-sucedidas!")
    
except Exception as e:
    print(f"ERRO na importação: {e}")
    import traceback
    print(traceback.format_exc())
