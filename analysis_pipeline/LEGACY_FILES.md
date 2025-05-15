# Arquivos Legados para Retrocompatibilidade

Este diretório contém alguns arquivos que faziam parte do pipeline original e são mantidos por motivos de compatibilidade:

## Arquivos para Compatibilidade

- **time_series_analysis.py**: Implementação original de análise de séries temporais
- **correlation_analysis.py**: Implementação original de análise de correlação
- **stats_summary.py**: Implementação original de análise estatística
- **tenant_degradation_analysis.py**: Implementação original de análise de degradação de inquilinos (ainda usado pelo visualize_degradation.py)
- **advanced_analysis.py**: Implementação original de análises avançadas

## Arquivos Movidos para legacy_modules/

- **causal_analysis.py**: Implementação original de análise de causalidade (substituída por causal_fixed.py)
- **causal_analysis_integrated.py**: Versão integrada da análise causal (substituída)
- **fix_causal_analysis.py**: Scripts de correção da análise causal (não mais necessários)
- **fixed_causal_analysis.py**: Versão corrigida da análise causal (substituída)
- **test_causal_analysis.py**: Testes para o módulo original de análise causal
- **tenant_comparison_module.py**: Módulo de comparação entre inquilinos (funcionalidade incorporada em tenant_analysis.py)
- **main_new.py**: Versão intermediária do pipeline principal

## Motivos para Manter

1. Alguns desses arquivos são importados por módulos integrados na nova estrutura
2. Manter retrocompatibilidade para projetos que usam a API anterior
3. Preservar funcionalidades que ainda não foram completamente migradas para os novos módulos

## Plano para o Futuro

No futuro, temos a intenção de:

1. Migrar completamente todas as funcionalidades para os novos módulos integrados
2. Deprecar gradualmente os arquivos legados com avisos apropriados
3. Remover os arquivos legados quando não forem mais necessários

## Recomendações para Desenvolvedores

- Para novos desenvolvimentos, use os módulos integrados:
  - **metrics_analysis.py**
  - **phase_analysis.py**
  - **tenant_analysis.py**
  - **suggestion_engine.py**
  - **pipeline_manager.py**
- Evite adicionar novas funcionalidades aos módulos legados
- Se precisar usar uma funcionalidade que só existe em um módulo legado, considere migrá-la para os módulos integrados

## Arquivos Removidos

Os seguintes arquivos foram removidos:

- `main.py` (pipeline original) - Substituído pelas novas versões integradas
- `main_new.py` (versão intermediária) - Funcionalidades incorporadas nas versões atuais

Estas funcionalidades foram substituídas por:
- `main_integrated.py` - Versão recomendada com integração completa
- `main_updated.py` - Versão intermediária com melhor organização
- `run_pipeline.py` - Script de seleção para escolher a versão apropriada do pipeline
