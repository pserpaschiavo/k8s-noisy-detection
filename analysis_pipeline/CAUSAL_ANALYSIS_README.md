## Análise Causal - Notas Técnicas

Este documento contém informações técnicas importantes sobre o módulo de análise causal (`causal_analysis.py`).

### Requisitos de Ambiente

O módulo de análise causal requer **Python 3.10** devido às seguintes razões:

1. As funções de detecção de pontos de mudança (`change_point_impact_analysis`) se baseiam na biblioteca `ruptures`, que apresenta problemas de compatibilidade com Python 3.13
2. Outras bibliotecas científicas como `numpy`, `scipy`, e `statsmodels` podem ter comportamento inconsistente em versões mais recentes do Python

### Como Resolver Problemas de Compatibilidade

Se você encontrar erros como:
```
AttributeError: module 'numpy' has no attribute '...'
ImportError: cannot import name '...' from 'ruptures'
```

Siga as seguintes etapas:

1. Instale o Python 3.10:
   ```bash
   sudo dnf install -y python3.10 python3.10-devel  # Fedora/RHEL
   # OU
   sudo apt install -y python3.10 python3.10-dev    # Ubuntu/Debian
   ```

2. Crie um ambiente virtual dedicado:
   ```bash
   python3.10 -m venv .venv310
   source .venv310/bin/activate
   ```

3. Instale as dependências:
   ```bash
   pip install ruptures numpy pandas matplotlib scipy statsmodels pyinform seaborn scikit-learn nolds
   ```

4. Execute seu script a partir deste ambiente:
   ```bash
   python your_script.py
   ```

### Testes

Para verificar se o ambiente está corretamente configurado, execute o script de teste:

```bash
source .venv310/bin/activate
python test_change_point.py
```

Você deve ver uma saída indicando a detecção bem-sucedida de pontos de impacto.
