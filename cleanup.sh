#!/bin/bash
# Script para limpeza do repositório k8s-noisy-detection

echo "Iniciando limpeza do repositório..."

# 1. Remover arquivos .pyc
echo "Removendo arquivos .pyc..."
find . -name "*.pyc" -type f -delete

# 2. Remover diretórios __pycache__
echo "Removendo diretórios __pycache__..."
find . -name "__pycache__" -type d -exec rm -rf {} +

# 3. Remover arquivos temporários
echo "Removendo arquivos temporários..."
find . -name "*.tmp" -type f -delete
find . -name ".DS_Store" -type f -delete

echo "Limpeza concluída!"
