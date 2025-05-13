#!/bin/bash
# Script para testar as funcionalidades de visualização amigável para daltônicos

echo "===== Testando Visualizações Amigáveis para Daltônicos ====="
echo "1. Executando o módulo de comparação de inquilinos standalone"
cd /home/phil/Projects/k8s-noisy-lab-data-pipe/analysis_pipeline
python tenant_comparison_module.py --output ./results/test_colorblind_standalone

echo ""
echo "2. Executando o pipeline principal com comparação de inquilinos"
python main.py --experiment "2025-05-11/16-58-00/default-experiment-1" \
               --round "round-1" \
               --tenant-comparison \
               --colorblind-friendly \
               --output ./results/test_colorblind_main

echo ""
echo "3. Gerando comparação entre visualizações padrão e amigáveis para daltônicos"
python test_colorblind_comparison.py

echo ""
echo "===== Teste concluído ====="
echo "Visualizações geradas em:"
echo "- ./results/test_colorblind_standalone/tenant_comparison/"
echo "- ./results/test_colorblind_main/plots/tenant_comparison/"
echo "- ./results/colorblind_comparison/"
