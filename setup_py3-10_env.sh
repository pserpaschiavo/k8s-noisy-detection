#!/bin/bash
# Script para criar e ativar um ambiente Python 3.10 para o pipeline de análise

echo "Configurando ambiente Python 3.10 para o pipeline de análise..."

# Verificar se Python 3.10 está instalado
if ! command -v python3.10 &> /dev/null
then
    echo "Python 3.10 não encontrado. Tentando instalar..."
    
    # Detectar o sistema operacional
    if command -v apt &> /dev/null
    then
        echo "Sistema baseado em Debian/Ubuntu detectado"
        echo "Executando: sudo apt install -y python3.10 python3.10-dev python3.10-venv"
        sudo apt install -y python3.10 python3.10-dev python3.10-venv
    elif command -v dnf &> /dev/null
    then
        echo "Sistema baseado em Fedora/RHEL detectado"
        echo "Executando: sudo dnf install -y python3.10 python3.10-devel"
        sudo dnf install -y python3.10 python3.10-devel
    else
        echo "Sistema operacional não suportado automaticamente."
        echo "Por favor, instale Python 3.10 manualmente e execute este script novamente."
        exit 1
    fi
fi

# Verificar se o ambiente virtual já existe
if [ ! -d ".venv310" ]
then
    echo "Criando ambiente virtual com Python 3.10..."
    python3.10 -m venv .venv310
else
    echo "Ambiente virtual .venv310 já existe."
fi

# Ativar o ambiente
echo "Ativando ambiente virtual..."
source .venv310/bin/activate

# Verificar a versão do Python ativada
python_version=$(python --version)
echo "Versão Python ativa: $python_version"

if [[ $python_version != *"3.10"* ]]
then
    echo "AVISO: A versão ativa não é Python 3.10! Pode haver problemas de compatibilidade."
fi

# Instalar dependências
echo "Instalando dependências..."
pip install -r requirements.txt

echo ""
echo "====================================================="
echo "Ambiente configurado! Para usar, execute:"
echo "source .venv310/bin/activate"
echo ""
echo "Para executar o pipeline, use:"
echo "python -m analysis_pipeline.main [argumentos]"
echo "====================================================="
