# Kubernetes Noisy Neighbors Lab - Data Analysis Pipeline

![Pipeline Status](https://img.shields.io/badge/Pipeline-Operational-green)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-orange)

## Visão Geral

O **Kubernetes Noisy Neighbors Lab** é um ambiente experimental controlado projetado para estudar o problema de "vizinhos barulhentos" (noisy neighbors) em clusters Kubernetes multi-inquilinos. Este repositório contém um pipeline completo de análise de dados que processa, analisa e visualiza métricas coletadas durante experimentos de contenção de recursos e interferência entre workloads.

## Objetivo do Projeto

O principal objetivo deste projeto é quantificar e caracterizar o impacto de cargas de trabalho agressivas em termos de recursos (CPU, memória, E/S de disco, rede) sobre outros inquilinos em um cluster Kubernetes compartilhado. O pipeline permite:

1. Analisar o comportamento de diferentes componentes durante condições normais e de ataque
2. Identificar correlações entre métricas e impactos entre inquilinos
3. Quantificar estatisticamente os efeitos de interferência
4. Detectar anomalias e pontos de mudança significativos nos dados
5. Gerar visualizações e resultados estatísticos com qualidade para publicações acadêmicas

## Requisitos do Ambiente

> **⚠️ Importante:** Este pipeline requer Python 3.10
> 
> O pipeline utiliza bibliotecas científicas (numpy, scipy, ruptures, etc.) que podem apresentar incompatibilidades com versões mais recentes do Python (como 3.13).

### Configuração do Ambiente

```bash
# Instalar Python 3.10 (se necessário)
sudo dnf install -y python3.10 python3.10-devel  # Fedora/RHEL
# OU
sudo apt install -y python3.10 python3.10-dev    # Ubuntu/Debian

# Criar ambiente virtual com Python 3.10
python3.10 -m venv .venv310

# Ativar ambiente virtual
source .venv310/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

## Estrutura do Repositório

```
k8s-noisy-lab-data-pipe/
├── analysis_pipeline/      # Pipeline principal de análise
│   ├── advanced_analysis.py      # Módulo de análises estatísticas avançadas
│   ├── correlation_analysis.py   # Análises de correlação entre métricas
│   ├── data_loader.py            # Carregador de dados com suporte a múltiplos rounds
│   ├── main.py                   # Ponto de entrada principal do pipeline
│   ├── setup.py                  # Script de instalação de dependências
│   ├── stats_summary.py          # Estatísticas básicas e comparações
│   ├── time_series_analysis.py   # Análises específicas para séries temporais 
│   ├── visualizations.py         # Gerador de visualizações
│   └── README.md                 # Documentação detalhada do pipeline
├── analysis/               # Resultados de análises gerados pelo pipeline  
└── results/                # Dados brutos dos experimentos
```

## Funcionalidades Principais

O pipeline de análise oferece diversas funcionalidades avançadas para processamento e análise:

### 1. Carregamento e Integração de Dados

- **Carregamento Hierárquico**: Organiza dados por fase (baseline, ataque, recuperação), componente e métrica
- **Combinação de Múltiplos Rounds**: Permite combinar dados de múltiplas rodadas experimentais para análises robustas
- **Métodos de Combinação**: Suporta combinação por média, mediana, mínimo ou máximo
- **Alinhamento Temporal**: Sincroniza séries temporais para análises comparativas precisas

### 2. Análise Estatística Básica

- **Estatísticas Descritivas**: Calcula média, mediana, desvio padrão, quartis, etc.
- **Testes de Estacionariedade**: Implementa testes ADF e KPSS para séries temporais
- **Comparação entre Fases**: Utiliza testes estatísticos para comparar métricas entre fases distintas
- **Análise de Efeitos**: Quantifica o impacto percentual e tamanho do efeito de eventos de interferência

### 3. Análise de Séries Temporais

- **Cross-Correlação**: Identifica relações entre séries temporais com diferentes defasagens
- **Causalidade de Granger**: Avalia estatisticamente relações de causalidade entre métricas
- **Análise de Entropia**: Quantifica a complexidade e regularidade das séries temporais
- **Decomposição de Séries**: Separa componentes de tendência, sazonalidade e resíduos

### 4. Análises Avançadas

- **Detecção de Change Points**: Identifica automaticamente pontos de mudança significativa nos dados
- **Análise de Distribuições**: Ajusta e compara dados com distribuições teóricas
- **Detecção de Anomalias**: Implementa algoritmos como Isolation Forest, Z-score e IQR
- **Clustering de Métricas**: Agrupa métricas relacionadas usando técnicas de aprendizado não supervisionado
- **Análise de Recuperação**: Quantifica tempos e padrões de recuperação após eventos de interferência

### 5. Visualizações e Exportações

- **Visualizações Automáticas**: Gera gráficos para análise rápida de todas as métricas relevantes
- **Comparações Visuais**: Facilita comparações entre fases e componentes
- **Exportação Acadêmica**: Produz resultados em formatos adequados para publicações (CSV, LaTeX, PNG, PDF)

## Como Usar

### Requisitos

- Python 3.6 ou superior
- Dependências listadas em `analysis_pipeline/setup.py`

### Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/k8s-noisy-lab-data-pipe.git
   cd k8s-noisy-lab-data-pipe
   ```

2. Instale as dependências:
   ```bash
   cd analysis_pipeline
   python setup.py
   ```

### Exemplo de Uso Básico

Para executar uma análise básica:

```bash
cd analysis_pipeline
python main.py --experiment "2025-05-11/16-58-00/default-experiment-1" --round "round-1"
```

### Exemplo de Análise Avançada

Para realizar análises estatísticas avançadas:

```bash
python main.py --experiment "2025-05-11/16-58-00/default-experiment-1" \
               --round "round-1" \
               --advanced-analysis \
               --distribution-analysis \
               --anomaly-detection iforest \
               --change-point-detection \
               --clustering \
               --recovery-analysis
```

### Análise de Múltiplos Rounds

Para combinar dados de múltiplas rodadas experimentais:

```bash
python main.py --experiment "2025-05-11/16-58-00/default-experiment-1" \
               --round round-1 round-2 round-3 \
               --combine-rounds \
               --combine-method mean
```

## Parâmetros Principais

| Parâmetro | Descrição |
|-----------|-----------|
| `--experiment` | Caminho do experimento relativo à pasta results/ |
| `--round` | Rodada(s) a analisar (ex: "round-1" ou múltiplas: "round-1 round-2") |
| `--combine-rounds` | Ativa combinação de dados de múltiplas rodadas |
| `--combine-method` | Método para combinar rodadas ('mean', 'median', 'min', 'max') |
| `--phases` | Fases a analisar (padrão: "1 - Baseline" "2 - Attack" "3 - Recovery") |
| `--output` | Diretório de saída para resultados |
| `--metrics-of-interest` | Lista de métricas-chave para análise detalhada |
| `--components` | Lista de componentes para análise |
| `--advanced-analysis` | Ativa análises estatísticas avançadas |
| `--distribution-analysis` | Realiza análises de distribuição e ajustes |
| `--anomaly-detection` | Método para detecção de anomalias ('iforest', 'zscore', 'iqr') |
| `--change-point-detection` | Ativa detecção de pontos de mudança |
| `--clustering` | Realiza clustering de métricas relacionadas |
| `--recovery-analysis` | Analisa métricas de recuperação após eventos |

## Estrutura de Saída

Os resultados são organizados da seguinte forma:

```
analysis/YYYY-MM-DD_HH-MM-SS/
├── plots/                        # Visualizações básicas
│   ├── [fases]/                  # Gráficos por fase
│   ├── correlations/             # Análises de correlação
│   └── comparacao_fases/         # Comparações entre fases
├── stats_results/                # Resultados estatísticos em CSV e LaTeX
├── advanced_analysis/            # Resultados de análises avançadas
│   ├── advanced_plots/           # Visualizações avançadas 
│   └── advanced_results/         # Resultados tabulares avançados
└── analysis_config.txt           # Configuração usada na análise
```

## Contribuição

Contribuições são bem-vindas! Para contribuir:

1. Faça um fork do repositório
2. Crie um branch para sua feature (`git checkout -b feature/sua-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para o branch (`git push origin feature/sua-feature`)
5. Abra um Pull Request

## Licença

Este projeto é licenciado sob a licença MIT - veja o arquivo LICENSE para mais detalhes.

## Citação

Se você utilizar este pipeline em sua pesquisa, por favor cite:

```
@software{k8s_noisy_lab,
  author = {SCHIAVO, P. S.},
  title = {Kubernetes Noisy Neighbors Lab - Data Analysis Pipeline},
  year = {2025},
  url = {https://github.com/pserpaschiavo/k8s-noisy-lab-data-pipe}
}
```
