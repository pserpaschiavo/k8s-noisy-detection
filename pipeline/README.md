# Pipeline de Análise de Dados para Experimento de Noisy Neighbors em Kubernetes

Este pipeline fornece uma estrutura completa para análise de dados coletados em experimentos de detecção de "vizinhos barulhentos" (noisy neighbors) em clusters Kubernetes.

## Características

- **Processamento de dados flexível**: Permite selecionar tenants, métricas e fases específicas para análise
- **Análise inter-fases e inter-tenants**: Compara o comportamento entre diferentes tenants e fases do experimento
- **Visualizações de qualidade acadêmica**: Gráficos formatados para publicações em periódicos e conferências
- **Exportação de resultados tabulares**: Gera tabelas em formatos LaTeX e CSV para inclusão em publicações
- **Notebooks didáticos**: Fornece notebooks com explicações detalhadas para cada etapa da análise

## Estrutura do Projeto

```
pipeline/
├── data_processing/       # Módulos para carregamento e pré-processamento de dados
│   ├── consolidation.py   # Funções para consolidar dados dos CSVs
│   ├── cleaning.py        # Funções para limpeza de dados
│   ├── aggregation.py     # Funções para agregação por tenant/fase
│   └── time_normalization.py # Normalização temporal
├── analysis/              # Módulos para análise de dados
│   ├── phase_analysis.py  # Análise inter-fases
│   ├── tenant_analysis.py # Análise inter-tenants
│   ├── noisy_tenant_detection.py # Detecção automática de tenant ruidoso
│   ├── comparative.py     # Análises comparativas
│   └── anomaly_detection.py # Detecção de anomalias
├── visualization/         # Módulos para visualização
│   ├── plots.py           # Funções de plotagem de alta qualidade
│   ├── publication_styles.py # Estilos para publicação acadêmica
│   └── table_generator.py # Geração de tabelas LaTeX e CSV
├── notebooks/             # Notebooks didáticos
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_tenant_comparisons.ipynb
│   ├── 04_phase_impact_analysis.ipynb
│   └── 05_anomaly_detection.ipynb
├── config.py              # Configurações do pipeline
└── main.py                # Script principal
```

## Instalação

1. Clone este repositório:
```bash
git clone <repository-url>
cd k8s-noisy-detection
```

2. Execute o script de configuração para instalar dependências e criar a estrutura de diretórios:
```bash
python setup_pipeline.py
```

## Uso

### Executando o pipeline completo

Para executar o pipeline com as configurações padrão:
```bash
python pipeline/main.py
```

### Opções de linha de comando

O pipeline aceita vários argumentos para personalizar a análise:

```bash
python pipeline/main.py --data-dir <caminho-dados> --output-dir <caminho-saída> --tenants tenant-a tenant-b --metrics cpu_usage memory_usage
```

Opções disponíveis:
- `--data-dir`: Diretório com os dados do experimento
- `--output-dir`: Diretório para salvar os resultados
- `--tenants`: Lista de tenants específicos para analisar
- `--noisy-tenant`: Tenant específico que gera ruído (por padrão: tenant-b)
- `--auto-detect-noisy`: Detectar automaticamente qual tenant é o gerador de ruído
- `--metrics`: Lista de métricas específicas para analisar
- `--phases`: Lista de fases específicas para analisar
- `--rounds`: Lista de rounds específicos para analisar
- `--skip-plots`: Pular geração de gráficos
- `--skip-tables`: Pular geração de tabelas

### Usando os notebooks

Para uma análise interativa e didática, você pode usar os notebooks Jupyter fornecidos:

```bash
cd k8s-noisy-detection
jupyter notebook pipeline/notebooks/
```

## Detecção Automática de Tenant Ruidoso

O pipeline inclui a capacidade de detectar automaticamente qual tenant está agindo como "noisy neighbor" (vizinho barulhento), sem a necessidade de especificá-lo manualmente. Esta detecção é baseada em quatro análises complementares:

1. **Análise de Correlação**: Identifica tenants que apresentam correlações negativas com outros tenants (quando um consome mais recursos, os outros têm menos disponíveis)
2. **Análise de Causalidade**: Utiliza testes de causalidade de Granger para identificar tenants que "causam" mudanças nos outros
3. **Detecção de Anomalias**: Identifica qual tenant apresenta mais comportamentos anômalos durante a fase de ataque
4. **Análise de Impacto**: Avalia o impacto de cada tenant nos outros em termos de degradação de desempenho

Para usar esta funcionalidade, basta adicionar o parâmetro `--auto-detect-noisy` ao executar o pipeline:

```bash
python pipeline/main.py --data-dir <caminho-dados> --output-dir <caminho-saída> --auto-detect-noisy
```

## Notebooks Didáticos

1. **Data Preprocessing**: Explica o processo de carregamento e pré-processamento dos dados
2. **Exploratory Analysis**: Demonstra análise exploratória inicial das métricas coletadas
3. **Tenant Comparisons**: Compara o comportamento entre diferentes tenants
4. **Phase Impact Analysis**: Analisa o impacto das diferentes fases do experimento
5. **Anomaly Detection**: Demonstra técnicas para detectar comportamentos anômalos

## Exemplos de Visualizações

O pipeline gera diversos tipos de visualizações:

- Gráficos de séries temporais com métricas para diferentes tenants
- Comparações de métricas entre fases (Baseline vs Attack vs Recovery)
- Heatmaps de impacto mostrando como o tenant barulhento afeta os outros
- Gráficos de efetividade da recuperação após a fase de ataque

## Tabelas para Publicações

O pipeline exporta tabelas formatadas tanto para LaTeX (para inclusão direta em papers) quanto para CSV (para análises adicionais):

- Tabelas de comparação entre fases para cada tenant
- Resumo estatístico do impacto do tenant barulhento
- Análise de efetividade da recuperação

## Contribuições

Contribuições são bem-vindas! Por favor, siga as diretrizes de contribuição do projeto.

## Licença

Este projeto é licenciado sob a licença MIT.
