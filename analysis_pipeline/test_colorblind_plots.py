#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script para verificar visualmente os plots gerados com a nova paleta acessível para daltônicos.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
from pathlib import Path
import sys

def display_plots(directory):
    """
    Displays all PNG images in the specified directory
    """
    print(f"Exibindo imagens do diretório: {directory}")
    
    # Encontra todos os arquivos PNG
    png_files = list(sorted(Path(directory).glob("*.png")))
    
    if not png_files:
        print(f"Nenhum arquivo PNG encontrado em {directory}")
        return
    
    print(f"Encontrados {len(png_files)} arquivos de imagem:")
    for f in png_files:
        print(f"- {f.name}")
    
    # Para cada imagem PNG, a exibe em uma janela
    fig = plt.figure(figsize=(12, 10))
    
    for i, img_path in enumerate(png_files):
        plt.subplot(2, 2, i+1)
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.title(img_path.stem)
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle("Visualizações com Paleta Amigável para Daltônicos", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()

if __name__ == "__main__":
    # Diretório padrão para procurar as imagens
    default_dir = Path(__file__).parent / "results" / "tenant_comparison" / "tenant_comparison"
    
    # Permite especificar um diretório diferente como argumento
    dir_path = sys.argv[1] if len(sys.argv) > 1 else default_dir
    
    if not Path(dir_path).exists():
        print(f"Diretório não encontrado: {dir_path}")
        sys.exit(1)
        
    display_plots(dir_path)
