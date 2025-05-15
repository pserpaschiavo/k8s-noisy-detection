#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de exemplo para visualização de degradação entre inquilinos.
Este script demonstra como utilizar todas as visualizações de degradação implementadas.

Uso:
    python3 visualize_degradation.py --data_dir ../analysis/2025-05-13_22-52-07 --output_dir ../output/visualizations
"""

import argparse
import logging
import sys
import pandas as pd
from pathlib import Path
from analysis_pipeline.data_loader import DataLoader
from analysis_pipeline.tenant_analysis import TenantAnalyzer

def main():
    # Configuração de logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Analisar argumentos
    parser = argparse.ArgumentParser(description='Gerar visualizações de degradação entre inquilinos')
    parser.add_argument('--data_dir', required=True, help='Diretório com os dados da análise')
    parser.add_argument('--output_dir', required=True, help='Diretório para salvar as visualizações')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        logging.error(f"Diretório de dados não encontrado: {data_dir}")
        sys.exit(1)
    
    # Criar diretório de saída se não existir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Carregar dados
    logging.info(f"Carregando dados de {data_dir}...")
    
    # Obter nome do experimento do diretório (último componente do caminho)
    # Exemplo: /home/phil/Projects/k8s-noisy-detection/analysis/2025-05-13_22-52-07
    experiment_name = data_dir.name
    logging.info(f"Nome do experimento detectado: {experiment_name}")
    
    try:
        data_loader = DataLoader(data_dir.parent, experiment_name)
        data = data_loader.load_all_data()
    except Exception as e:
        # Se falhar com a abordagem padrão, usar uma abordagem alternativa
        # para diretórios de análise
        logging.info(f"Tentando carregamento alternativo: {e}")
        
        # Implementação simplificada para carregar dados diretamente do diretório de análise
        data = {}
        for phase_dir in data_dir.glob("*"):
            if phase_dir.is_dir() and not phase_dir.name.startswith('.'):
                phase_name = phase_dir.name.replace("_", " ")
                data[phase_name] = {}
                
                # Para cada tenant no diretório de fase
                for tenant_dir in phase_dir.glob("*"):
                    if tenant_dir.is_dir():
                        tenant_name = tenant_dir.name
                        data[phase_name][tenant_name] = {}
                        
                        # Para cada arquivo CSV de métrica
                        for metric_file in tenant_dir.glob("*.csv"):
                            metric_name = metric_file.stem
                            try:
                                df = pd.read_csv(metric_file)
                                data[phase_name][tenant_name][metric_name] = df
                            except Exception as e:
                                logging.error(f"Erro ao ler {metric_file}: {e}")
                                
        logging.info(f"Carregamento alternativo concluído. Fases encontradas: {list(data.keys())}")
    
    # Verificar fases disponíveis
    phases = list(data.keys())
    logging.info(f"Fases disponíveis: {phases}")
    
    # Identificar inquilinos para análise (tenants)
    if phases:
        phase_data = data[phases[0]]
        # Encontrar os inquilinos (normalmente começam com "tenant-")
        tenants = [comp for comp in phase_data.keys() if comp.startswith("tenant-")]
        logging.info(f"Inquilinos identificados: {tenants}")
    else:
        logging.error("Nenhuma fase encontrada nos dados.")
        sys.exit(1)
    
    # Definir métricas de interesse
    metrics_of_interest = [
        'cpu_usage', 
        'memory_usage', 
        'disk_io_total', 
        'network_receive', 
        'network_transmit', 
        'network_total_bandwidth'
    ]
    logging.info(f"Métricas de interesse: {metrics_of_interest}")
    
    # Criar analisador de degradação
    analyzer = TenantDegradationAnalyzer(output_dir)
    
    # Gerar todas as visualizações
    logging.info("Gerando visualizações de degradação...")
    vis_results = analyzer.generate_all_visualizations(
        data=data,
        phases=phases,
        metrics_of_interest=metrics_of_interest,
        tenants=tenants,
        output_subdir='degradation_visualizations'
    )
    
    logging.info(f"Visualizações geradas: {len(vis_results)} fases processadas.")
    logging.info(f"Todas as visualizações foram salvas em {output_dir}/degradation_visualizations")
    
    # Exibir informações sobre fontes de degradação
    if 'degradation_sources' in vis_results:
        logging.info("\nPossíveis fontes de degradação identificadas:")
        for metric, results in vis_results['degradation_sources'].items():
            if 'likely_degradation_sources' in results and results['likely_degradation_sources']:
                sources = [s['tenant'] for s in results['likely_degradation_sources']]
                logging.info(f"  - Métrica {metric}: {', '.join(sources)}")
            else:
                logging.info(f"  - Métrica {metric}: Nenhuma fonte clara identificada")
    
    logging.info("\nNota: Consulte os relatórios detalhados nos arquivos degradation_report_*.txt")

if __name__ == "__main__":
    main()
