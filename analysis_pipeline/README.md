# Pipeline de Análise para Kubernetes Noisy Neighbours Lab

Este pipeline realiza análise automatizada dos dados coletados nos experimentos do Kubernetes Noisy Neighbours Lab, incluindo análises estatísticas, de séries temporais, visualizações e análises estatísticas avançadas com qualidade para publicação acadêmica.

## Funcionalidades

O pipeline oferece as seguintes funcionalidades:

### 1. Carregamento de Dados

- Carregamento recursivo de métricas de todos os inquilinos (tenants) e componentes
- Organização hierárquica por fase, componente e métrica
- Alinhamento temporal de séries para análises comparativas
- **Combinação de múltiplos rounds** para análises mais robustas, calculando médias, medianas, mínimos ou máximos

### 2. Análise Estatística

- Estatísticas descritivas (média, mediana, desvio padrão, etc.)
- Testes de estacionariedade (ADF e KPSS)
- Detecção de anomalias baseada em Z-score
- Comparação estatística entre fases (teste-t, tamanho do efeito)

### 3. Análise de Séries Temporais

- **Análise de Cross-Correlação**: Identifica correlações considerando diferentes defasagens temporais
- **Análise de Lag**: Determina o atraso ótimo entre eventos relacionados
- **Causalidade de Granger**: Avalia estatisticamente se uma série temporal causa outra
- **Análise de Entropia**: Quantifica a complexidade e regularidade das séries

### 4. Análise de Correlação

### 5. Visualizações

- Geração automática de diversos tipos de gráficos (séries temporais, distribuições, boxplots)
- Comparações visuais entre fases experimentais
- **Comparação entre inquilinos (tenants)** com visualização do impacto entre diferentes fases
- **Visualizações acessíveis** com suporte para daltônicos, utilizando paletas e padrões distintos
- Exportação em alta qualidade para publicação acadêmica

- Matrizes de correlação entre métricas
- Identificação de correlações fortes
- Visualização de pares de métricas altamente correlacionadas

### 5. Visualizações

- Gráficos de séries temporais para cada métrica
- Distribuições e boxplots
- Comparações entre fases (linha, boxplot, violino)
- Heatmaps de correlação

### 6. Detecção de Mudanças Significativas

- Identificação automática de métricas com variações significativas durante ataques
- Quantificação do tamanho do efeito e mudança percentual

### 7. Análises Estatísticas Avançadas

- **Decomposição de Séries Temporais**: Separação de tendência, sazonalidade e resíduos
- **Detecção de Change Points**: Identificação automática de pontos de mudança significativa nos dados
- **Análise de Distribuições**: Ajuste e comparação com distribuições teóricas (normal, lognormal, etc.)
- **Detecção de Anomalias**: Identificação de pontos anômalos usando algoritmos avançados
- **Clustering de Métricas**: Agrupamento automático de métricas relacionadas
- **Análise de Recuperação**: Quantificação de tempos e níveis de recuperação após eventos

## Como Executar

Para executar o pipeline completo:

```bash
cd analysis_pipeline
python main.py --experiment "2025-05-11/16-58-00/default-experiment-1" --round "round-1"
```

### Parâmetros Básicos

- `--experiment`: Caminho do experimento relativo à pasta results/ (ex: "2025-05-11/16-58-00/default-experiment-1")
- `--round`: Rodada(s) a analisar (ex: "round-1" ou múltiplas rodadas "round-1 round-2 round-3")
- `--combine-rounds`: Combinar dados de múltiplas rodadas para a análise
- `--combine-method`: Método para combinar os dados das rodadas ('mean', 'median', 'min', 'max')
- `--phases`: Fases a analisar (padrão: "1 - Baseline" "2 - Attack" "3 - Recovery")
- `--output`: Diretório de saída (padrão: results/analysis/YYYY-MM-DD_HH-MM-SS/)
- `--skip-plots`: Pula a geração de visualizações (para análise mais rápida)
- `--skip-advanced`: Pula análises avançadas de séries temporais (mais rápido)
- `--metrics-of-interest`: Lista de métricas-chave para comparação entre fases
- `--components`: Lista de componentes para análise

### Parâmetros para Análises Avançadas

- `--advanced-analysis`: Realiza análises estatísticas avançadas (decomposição de séries temporais)
- `--distribution-analysis`: Realiza análises de distribuição e ajustes a distribuições teóricas
- `--anomaly-detection METHOD`: Realiza detecção de anomalias usando método especificado ('iforest', 'zscore', 'iqr')
- `--change-point-detection`: Executa detecção de pontos de mudança nas séries temporais
- `--clustering`: Realiza clustering de métricas para identificar padrões relacionados
- `--recovery-analysis`: Analisa métricas de recuperação após ataques

### Exemplos de Uso

#### Análise Básica
```bash
python main.py --experiment "2025-05-11/16-58-00/default-experiment-1" \
               --round "round-1" \
               --metrics-of-interest cpu_usage memory_usage network_total_bandwidth \
               --components tenant-a tenant-b ingress-nginx
```

#### Análise de Múltiplos Rounds
```bash
python main.py --experiment "2025-05-11/16-58-00/default-experiment-1" \
               --round round-1 round-2 round-3 \
               --combine-rounds \
               --combine-method mean
```

#### Análise com Detecção de Change Points
```bash
python main.py --experiment "2025-05-11/16-58-00/default-experiment-1" \
               --round "round-1" \
               --change-point-detection
```

#### Análise Completa com Todas as Ferramentas Avançadas
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

#### Visualização de Comparação entre Inquilinos (Tenant Comparison)
```bash
python main.py --experiment "2025-05-11/16-58-00/default-experiment-1" \
               --round "round-1" \
               --tenant-comparison
```

#### Visualização Acessível para Daltônicos
O pipeline suporta visualizações amigáveis para daltônicos que usam combinações de cores específicas, padrões de linhas diferentes e texturas distintas para fases. Por padrão, esse modo está ativado em todas as visualizações.

```bash
# Para ativar manualmente se necessário (já é o padrão)
python tenant_comparison_module.py --colorblind-friendly
```

Para visualizar os plots gerados:
```bash
python test_colorblind_plots.py
```

### Características das Visualizações Acessíveis:

- **Paleta amigável para daltônicos**: Cores selecionadas para serem distinguíveis mesmo por quem possui deficiência na visão de cores (protanopia, deuteranopia e tritanopia)
- **Padrões de linha distintos**: Combinação de estilos de linha diferentes (sólido, tracejado, pontilhado)
- **Marcadores diferenciados**: Utilização de diferentes formas de marcadores para cada inquilino
- **Texturas de fundo**: Padrões de hachura para diferenciar as fases do experimento
- **Alto contraste**: Rótulos com melhor contraste e posicionamento
- **Legendas duplas**: Exibição separada de legendas para inquilinos e fases

## Estrutura de Saída

Os resultados são organizados da seguinte forma:

```
results/analysis/YYYY-MM-DD_HH-MM-SS/
├── plots/                           # Visualizações básicas
│   ├── 1_-_Baseline/                # Gráficos da fase baseline
│   ├── 2_-_Attack/                  # Gráficos da fase de ataque
│   ├── 3_-_Recovery/                # Gráficos da fase de recuperação
│   ├── correlations/                # Análises de correlação
│   ├── tenant_comparison/           # Comparações entre inquilinos
│   └── comparacao_fases/            # Comparações entre fases
├── stats_results/                   # Resultados estatísticos em CSV e LaTeX
│   ├── significant_changes.csv      # Métricas com mudanças significativas
│   ├── phase_comparison_*.csv       # Comparações estatísticas entre fases
│   ├── *_stats.csv                  # Estatísticas descritivas
│   ├── granger_*.csv                # Resultados de causalidade de Granger
│   └── entropy_*.csv                # Resultados de análise de entropia
├── advanced_analysis/               # Resultados de análises estatísticas avançadas
│   ├── advanced_plots/              # Visualizações avançadas
│   │   ├── time_series/             # Decomposição, anomalias e recuperação
│   │   ├── distributions/           # Análises e ajustes de distribuição
│   │   ├── changepoints/            # Detecção de pontos de mudança
│   │   └── multivariate/            # Clustering e análises multivariadas
│   ├── advanced_results/            # Resultados tabulares em CSV e LaTeX
│   └── advanced_analysis_summary.txt # Resumo das análises avançadas
├── analysis_config.txt              # Configuração usada na análise
└── analysis_pipeline.log            # Log de execução
```

Quando múltiplos rounds são combinados, o diretório de saída inclui o sufixo `_combined` ou `_combined_[método]` para indicar o método utilizado na combinação.

## Módulos

O pipeline contém os seguintes módulos:

- `data_loader.py`: Carregamento recursivo de dados e combinação de múltiplos rounds
- `stats_summary.py`: Análises estatísticas básicas
- `time_series_analysis.py`: Análises de séries temporais
- `correlation_analysis.py`: Análises de correlação entre métricas
- `visualizations.py`: Geração de gráficos e visualizações
- `advanced_analysis.py`: Análises estatísticas avançadas para publicação acadêmica
- `main.py`: Ponto de entrada do pipeline
- `setup.py`: Instalação de dependências

## Requisitos

Para instalar todas as dependências necessárias:

```bash
python setup.py
```

Alternativamente, você pode instalar manualmente as dependências principais:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels ruptures scikit-learn nolds pyinform
```

## Análises Avançadas Disponíveis

### Decomposição de Séries Temporais

Separa uma série temporal em componentes de tendência, sazonalidade e resíduos, permitindo identificar:
- Tendências de longo prazo no comportamento dos sistemas
- Padrões cíclicos e sazonais nas métricas
- Ruído aleatório vs. componentes sistemáticos

### Detecção de Change Points

Identifica automaticamente pontos nas séries temporais onde ocorrem mudanças significativas:
- Detecção precisa do início dos efeitos de "noisy neighbours"
- Quantificação da latência entre início do ataque e impacto observado
- Identificação de múltiplos regimes de comportamento durante as fases

### Análise de Distribuições

Examina a distribuição estatística dos dados e ajusta distribuições teóricas:
- Identificação de componentes com distribuições não-normais
- Ajuste a distribuições conhecidas (normal, lognormal, exponencial, etc.)
- Quantificação do ajuste usando testes estatísticos

### Detecção de Anomalias

Identifica pontos de dados anômalos em relação ao padrão esperado:
- Isolation Forest para detecção de outliers multidimensionais
- Métodos baseados em Z-Score para valores extremos
- Detecção baseada em IQR (Intervalo Interquartil) para robustez

### Clustering de Métricas

Agrupa métricas relacionadas para encontrar padrões comuns:
- K-means clustering de métricas correlacionadas
- Redução de dimensionalidade via PCA para visualização
- Importância relativa das métricas para diferentes clusters

### Análise de Recuperação

Quantifica como o sistema se recupera após eventos disruptivos:
- Tempo de recuperação após cessação do ataque
- Percentual de recuperação em relação ao baseline
- Análise do nível de estabilidade no período pós-ataque

## Exportação para Publicações Acadêmicas

Todos os resultados são exportados em formatos adequados para publicações:
- Gráficos em PNG de alta resolução (300 dpi)
- Versões em PDF vetoriais para inclusão em LaTeX
- Tabelas em formato CSV para processamento adicional
- Tabelas em formato LaTeX para inclusão direta em artigos
- Estatísticas completas documentadas para metodologia

## Exemplos de Uso Avançado

### Análise Rápida (sem Plots):
```bash
python main.py --skip-plots
```

### Combinando Rounds com Mediana (mais robustez contra outliers):
```bash
python main.py --round round-1 round-2 round-3 --combine-rounds --combine-method median
```

### Análise Completa para Publicação Acadêmica:
```bash
python main.py --advanced-analysis --distribution-analysis --anomaly-detection iforest --change-point-detection --clustering --recovery-analysis
```

### Análise Focada em Recuperação:
```bash
python main.py --recovery-analysis --metrics-of-interest cpu_usage memory_usage response_time
```

### Detecção de Anomalias em Componentes Específicos:
```bash
python main.py --anomaly-detection iforest --components tenant-a tenant-b
```
