#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de instalação de dependências para o pipeline de análise
"""

import sys
import subprocess
import platform
import os
import shutil

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
    
    # Dependências críticas para compilação
    build_packages = [
        "setuptools",
        "wheel",
        "cython",
        "pip>=21.0.0"  # Versão mais recente do pip
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
    
    # Primeiro atualiza o pip e instala ferramentas de compilação
    print("Atualizando pip e ferramentas de build...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("Pip atualizado com sucesso. ✓")
    except subprocess.CalledProcessError:
        print("AVISO: Falha ao atualizar pip. Continuando com a versão existente.")
    
    # Instala dependências de build
    for package in build_packages:
        print(f"Instalando ferramenta de build {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} instalado com sucesso. ✓")
        except subprocess.CalledProcessError:
            print(f"AVISO: Falha ao instalar {package}. Algumas instalações podem falhar.")
    
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
        
        # Tratamento especial para o pacote ruptures que requer compilação
        if package == "ruptures":
            try:
                # Tenta instalar com compilação
                print("Instalando ruptures (pode exigir compilação)...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"{package} instalado com sucesso. ✓")
            except subprocess.CalledProcessError:
                print(f"AVISO: Falha ao instalar {package} com o método padrão.")
                print("Tentando com abordagem alternativa (pré-compilado)...")
                try:
                    # Tenta instalar uma versão pré-compilada (wheel) se disponível
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--only-binary", "ruptures", "ruptures"])
                    print(f"{package} instalado com sucesso (versão pré-compilada). ✓")
                except subprocess.CalledProcessError:
                    print(f"AVISO: Falha ao instalar {package}.")
                    print("A detecção de change points não estará disponível.")
                    print("Para resolver manualmente:")
                    print("  - Em sistemas Debian/Ubuntu: sudo apt-get install python3-dev")
                    print("  - Em sistemas Fedora: sudo dnf install python3-devel")
                    print("  - Em sistemas CentOS/RHEL: sudo yum install python3-devel")
                    print("Depois execute: pip install ruptures")
                    continue
        else:
            # Para outros pacotes, instala normalmente
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

def install_system_dependencies():
    """Instala as dependências do sistema necessárias para os pacotes Python."""
    system = platform.system()
    
    print("\nVerificando dependências de sistema necessárias para compilação...")
    
    if system == "Linux":
        # Detectar a distribuição Linux
        distro = ""
        package_manager = ""
        
        # Verificar se é uma distribuição baseada em Debian/Ubuntu
        if os.path.exists("/etc/debian_version"):
            distro = "Debian/Ubuntu"
            package_manager = "apt"
            install_cmd = ["sudo", "apt-get", "install", "-y", "python3-dev", "build-essential", "gcc"]
            
        # Verificar se é Fedora
        elif os.path.exists("/etc/fedora-release"):
            distro = "Fedora"
            package_manager = "dnf"
            install_cmd = ["sudo", "dnf", "install", "-y", "python3-devel", "gcc"]
            
        # Verificar se é CentOS/RHEL
        elif os.path.exists("/etc/redhat-release"):
            distro = "CentOS/RHEL"
            package_manager = "yum"
            install_cmd = ["sudo", "yum", "install", "-y", "python3-devel", "gcc", "gcc-c++"]
            
        # Verificar se é Arch Linux
        elif os.path.exists("/etc/arch-release"):
            distro = "Arch Linux"
            package_manager = "pacman"
            install_cmd = ["sudo", "pacman", "-S", "--noconfirm", "python", "base-devel", "gcc"]
            
        # Se não conseguiu identificar a distribuição específica
        else:
            print("Não foi possível identificar sua distribuição Linux específica.")
            print("Por favor, instale manualmente os pacotes de desenvolvimento Python e GCC:")
            print("  - Para Debian/Ubuntu: sudo apt-get install python3-dev build-essential gcc")
            print("  - Para Fedora: sudo dnf install python3-devel gcc")
            print("  - Para CentOS/RHEL: sudo yum install python3-devel gcc gcc-c++")
            print("  - Para Arch Linux: sudo pacman -S python base-devel gcc")
            return True
        
        print(f"Distribuição detectada: {distro} (usando {package_manager})")
        
        # Verificar se o sudo está disponível
        has_sudo = shutil.which("sudo") is not None
        if not has_sudo:
            print("Comando 'sudo' não encontrado. Tentando instalar sem privilégios de superusuário...")
            install_cmd = install_cmd[1:]  # Remove 'sudo' do comando
            
        # Verificar se o gerenciador de pacotes está disponível
        if shutil.which(package_manager) is None:
            print(f"AVISO: Gerenciador de pacotes '{package_manager}' não encontrado.")
            print("Você precisará instalar manualmente os pacotes de desenvolvimento Python e GCC.")
            return True
            
        # Instalar as dependências do sistema
        print(f"Instalando pacotes de desenvolvimento Python e compiladores para {distro}...")
        try:
            subprocess.check_call(install_cmd)
            print("Dependências do sistema instaladas com sucesso. ✓")
            return True
        except subprocess.CalledProcessError:
            print(f"AVISO: Falha ao instalar algumas dependências do sistema.")
            print("Alguns pacotes Python que requerem compilação podem falhar.")
            return True
        except Exception as e:
            print(f"ERRO ao instalar dependências do sistema: {e}")
            print("Você pode precisar instalar manualmente os pacotes de desenvolvimento Python e GCC.")
            return True
            
    elif system == "Darwin":  # macOS
        # Verificar se o Homebrew está instalado
        if shutil.which("brew") is None:
            print("Homebrew não detectado. Para instalar as dependências no macOS, considere instalar o Homebrew:")
            print("  /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            print("Em seguida, instale as ferramentas de desenvolvimento:")
            print("  brew install python gcc")
        else:
            print("Instalando dependências do sistema com Homebrew...")
            try:
                subprocess.check_call(["brew", "install", "python", "gcc"])
                print("Dependências do sistema instaladas com sucesso. ✓")
            except subprocess.CalledProcessError:
                print("Falha ao instalar algumas dependências do sistema com Homebrew.")
                print("Alguns pacotes Python que requerem compilação podem falhar.")
    
    elif system == "Windows":
        print("No Windows, recomendamos instalar as seguintes ferramentas:")
        print("1. Visual C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        print("2. Certifique-se de que o Python instalado tenha a opção 'Download debug binaries'")
        print("   selecionada durante a instalação.")
    
    return True

def main():
    """Função principal."""
    print("=== Setup do Pipeline de Análise para K8s Noisy Neighbours Lab ===\n")
    
    check_python_version()
    
    # Verificar sistema operacional
    system = platform.system()
    print(f"Sistema operacional detectado: {system}")
    
    # Instalar dependências do sistema necessárias para compilação
    install_system_dependencies()
    
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
        
        print("\nSe você estiver tendo problemas com o módulo 'ruptures', execute:")
        print("  python setup.py --install-ruptures")
        return 0
    else:
        print("\n=== Erro durante o setup ===")
        print("Verifique os erros acima e tente novamente.")
        return 1

def install_only_ruptures():
    """Instala apenas o pacote ruptures e suas dependências."""
    # Instalar dependências do sistema
    install_system_dependencies()
    
    # Atualizar pip
    print("Atualizando pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("Pip atualizado com sucesso. ✓")
    except subprocess.CalledProcessError:
        print("AVISO: Falha ao atualizar pip. Continuando com a versão existente.")
    
    # Instalar pacotes necessários para build
    build_packages = ["setuptools", "wheel", "cython"]
    for package in build_packages:
        print(f"Instalando {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} instalado com sucesso. ✓")
        except subprocess.CalledProcessError:
            print(f"AVISO: Falha ao instalar {package}.")
    
    # Instalar ruptures
    print("Instalando ruptures...")
    try:
        # Tenta instalar com compilação
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ruptures"])
        print("ruptures instalado com sucesso. ✓")
        return True
    except subprocess.CalledProcessError:
        print("AVISO: Falha ao instalar ruptures com o método padrão.")
        print("Tentando com abordagem alternativa (pré-compilado)...")
        try:
            # Tenta instalar uma versão pré-compilada (wheel) se disponível
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--only-binary", "ruptures", "ruptures"])
            print("ruptures instalado com sucesso (versão pré-compilada). ✓")
            return True
        except subprocess.CalledProcessError:
            print("ERRO: Falha ao instalar ruptures.")
            print("Para resolver manualmente:")
            print("  - Em sistemas Debian/Ubuntu: sudo apt-get install python3-dev")
            print("  - Em sistemas Fedora: sudo dnf install python3-devel")
            print("  - Em sistemas CentOS/RHEL: sudo yum install python3-devel")
            print("Depois execute: pip install ruptures")
            return False

if __name__ == "__main__":
    # Verificar se foi solicitada apenas a instalação do ruptures
    if len(sys.argv) > 1 and sys.argv[1] == "--install-ruptures":
        sys.exit(0 if install_only_ruptures() else 1)
    else:
        sys.exit(main())
