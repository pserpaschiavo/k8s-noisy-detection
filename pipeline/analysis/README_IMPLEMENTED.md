# Módulos Implementados

Este documento descreve os módulos que foram recentemente implementados no pipeline de análise.

## Análise de Métricas de Aplicação

O módulo `application_metrics_analysis.py` foi integrado ao pipeline principal para permitir análise de métricas em nível de aplicação como latência, taxa de erros e throughput. Este módulo oferece uma visão mais profunda do impacto de "noisy neighbors" no desempenho das aplicações.

### Funcionalidades

- **Análise de Impacto na Latência**: Calcula o aumento percentual na latência de cada tenant causado pelo tenant ruidoso
- **Correlação de Taxa de Erros**: Mede a correlação entre o uso de recursos do tenant ruidoso e a taxa de erros nos outros tenants
- **Análise de violação de SLO**: Avalia violações de Service Level Objectives causadas pela interferência do tenant ruidoso

### Como usar

```bash
python pipeline/main.py --app-metrics-analysis [--slo-thresholds path/to/thresholds.json]
```

## Comparação de Tecnologias

O módulo `technology_comparison.py` permite comparar diferentes experimentos realizados com tecnologias distintas de containerização ou isolamento (como Docker vanilla vs. Kata Containers).

### Funcionalidades

- **Normalização de Métricas**: Normaliza métricas entre experimentos para permitir comparação justa
- **Cálculo de Eficiência Relativa**: Quantifica o desempenho relativo entre diferentes tecnologias
- **Visualização Comparativa**: Gera visualizações comparando o comportamento das métricas entre experimentos
- **Análise Estatística**: Realiza testes estatísticos para determinar se as diferenças são significativas

### Como usar

```bash
python pipeline/main.py --compare-technologies --compare-dir /path/to/second/experiment [--technology-names "Docker" "Kata"]
```

## Integração com o Pipeline

Ambos os módulos estão totalmente integrados ao pipeline principal, permitindo:

1. Análise isolada ou conjunta com outros módulos
2. Exportação de resultados nos formatos padrão (CSV, LaTeX, Markdown)
3. Inclusão em relatórios gerados automaticamente
4. Customização via parâmetros de linha de comando
