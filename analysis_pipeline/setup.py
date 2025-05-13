#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de instalação de dependências para o pipeline de análise
"""

import sys
import subprocess
import platform

def check_python_version():
    """Verifica se a versão do Python é compatível."""
    if sys.version_info < (3, 6):
        print("ERRO: Este pipeline requer Python 3.6 ou superior.")
        sys.exit(1)
    else:
        print(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detectado. ✓")

def install_dependencies():
    """Instala as dependências necessárias."""
    # Pacotes básicos para análise de dados
    basic_packages = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy",
        "statsmodels"
    ]
    
    # Pacotes para análise avançada de séries temporais
    ts_packages = [
        "nolds",        # para análise de entropia
        "pyinform",     # para teoria da informação e medidas de complexidade
        "ruptures",     # para detecção de change points
        "scikit-learn", # para clustering e detecção de anomalias
        "pymoo"         # para otimização multi-objetivo (usado em alguns métodos avançados)
    ]
    
    # Pacotes para visualização acadêmica
    viz_packages = [
        "adjustText",   # para ajustes automáticos em textos em gráficos
        "palettable",   # paletas de cores para publicações acadêmicas
        "latex"         # para melhor integração com LaTeX
    ]
    
    # Todos os pacotes
    all_packages = basic_packages + ts_packages + viz_packages
    
    print("Instalando dependências...")
    
    # Primeiro instala os pacotes básicos (prioridade)
    for package in basic_packages:
        print(f"Instalando {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} instalado com sucesso. ✓")
        except subprocess.CalledProcessError:
            print(f"ERRO: Falha ao instalar {package}.")
            return False
    
    print("\nInstalando pacotes para análise avançada...")
    
    # Instala pacotes para análises avançadas
    for package in ts_packages:
        print(f"Instalando {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} instalado com sucesso. ✓")
        except subprocess.CalledProcessError:
            print(f"AVISO: Falha ao instalar {package}. Algumas funcionalidades avançadas podem não estar disponíveis.")
            continue
    
    print("\nInstalando pacotes para visualização de qualidade acadêmica...")
    
    # Instala pacotes para visualização
    for package in viz_packages:
        print(f"Instalando {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} instalado com sucesso. ✓")
        except subprocess.CalledProcessError:
            print(f"AVISO: Falha ao instalar {package}. Algumas opções visuais podem estar limitadas.")
            continue
    
    print("\nTodas as dependências principais foram instaladas com sucesso!")
    return True

def main():
    """Função principal."""
    print("=== Setup do Pipeline de Análise para K8s Noisy Neighbours Lab ===\n")
    
    check_python_version()
    
    # Verificar sistema operacional
    system = platform.system()
    print(f"Sistema operacional detectado: {system}")
    
    # Verificação adicional para LaTeX (necessário para exportação de alta qualidade)
    if system == "Linux":
        try:
            subprocess.check_call(["which", "pdflatex"], stdout=subprocess.DEVNULL)
            print("LaTeX detectado. Exportação para documentos acadêmicos disponível. ✓")
        except subprocess.CalledProcessError:
            print("LaTeX não detectado. Para exportação acadêmica completa, considere instalar texlive:")
            print("  sudo apt-get install texlive-latex-extra texlive-fonts-recommended")
    elif system == "Darwin":  # macOS
        try:
            subprocess.check_call(["which", "pdflatex"], stdout=subprocess.DEVNULL)
            print("LaTeX detectado. Exportação para documentos acadêmicos disponível. ✓")
        except subprocess.CalledProcessError:
            print("LaTeX não detectado. Para exportação acadêmica completa, considere instalar MacTeX:")
            print("  brew cask install mactex")
    elif system == "Windows":
        try:
            subprocess.check_call(["where", "pdflatex"], stdout=subprocess.DEVNULL)
            print("LaTeX detectado. Exportação para documentos acadêmicos disponível. ✓")
        except subprocess.CalledProcessError:
            print("LaTeX não detectado. Para exportação acadêmica completa, considere instalar MiKTeX ou TeX Live.")
    
    if install_dependencies():
        print("\n=== Setup concluído com sucesso! ===")
        print("Você já pode executar o pipeline com:")
        print("  python main.py")
        
        # Instruções para análises avançadas
        print("\nPara executar análises estatísticas avançadas, use as opções:")
        print("  python main.py --advanced-analysis")
        print("  python main.py --distribution-analysis")
        print("  python main.py --anomaly-detection iforest")
        print("  python main.py --change-point-detection")
        print("  python main.py --clustering")
        print("  python main.py --recovery-analysis")
        print("\nExemplo de pipeline completo:")
        print("  python main.py --advanced-analysis --distribution-analysis --anomaly-detection iforest --change-point-detection")
        return 0
    else:
        print("\n=== Erro durante o setup ===")
        print("Verifique os erros acima e tente novamente.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
