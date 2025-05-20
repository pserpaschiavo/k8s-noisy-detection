import pandas as pd
import numpy as np
from scipy import stats # Para testes estatísticos como t-test

# Você pode definir um limiar de significância global no seu config.py
# Por exemplo: SIGNIFICANCE_THRESHOLD = 0.05

def aggregate_analysis_results(
    df_individual_results: pd.DataFrame,
    grouping_cols: list = ['experiment_name', 'phase_label', 'source_metric', 'target_metric'],
    metric_cols: list = ['te_value', 'cosine_similarity_value', 'pearson_corr', 'dtw_distance'],
    p_value_cols: dict = {'te_value': 'te_p_value', 'pearson_corr': 'pearson_p'}, # Mapeia métrica para sua coluna de p-valor
    significance_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Agrega os resultados de análises individuais (por round/fase/setup)
    calculando estatísticas descritivas e contagem de significância.

    Args:
        df_individual_results (pd.DataFrame): DataFrame contendo os resultados de análise para
                                              cada round/fase/setup (ex: 'round_number', 'phase_label',
                                              'experiment_name', 'te_value', 'te_p_value', etc.).
        grouping_cols (list): Lista de colunas pelas quais agrupar os resultados
                              (ex: ['experiment_name', 'phase_label', 'source_metric', 'target_metric']).
        metric_cols (list): Lista de colunas contendo os valores das métricas de análise
                            (ex: 'te_value', 'cosine_similarity_value').
        p_value_cols (dict): Dicionário mapeando o nome da coluna da métrica para o nome da sua coluna de p-valor.
                              Usado para calcular a contagem de resultados significativos.
        significance_threshold (float): Limiar de p-valor para considerar um resultado estatisticamente significativo.

    Returns:
        pd.DataFrame: Um DataFrame agregado com médias, medianas, desvios padrão e contagens
                      de resultados significativos para as métricas especificadas.
    """
    
    # Agrupamento básico para médias, medianas, desvios padrão
    agg_dict = {col: ['mean', 'median', 'std'] for col in metric_cols}
    df_aggregated = df_individual_results.groupby(grouping_cols).agg(agg_dict)
    
    # Renomeia as colunas de forma mais clara
    df_aggregated.columns = ['_'.join(col).strip() for col in df_aggregated.columns.values]
    df_aggregated.reset_index(inplace=True)

    # Adiciona a contagem de resultados significativos para cada métrica
    for metric, p_col in p_value_cols.items():
        if p_col in df_individual_results.columns:
            # Cria uma coluna temporária para identificar resultados significativos
            df_individual_results[f'{metric}_is_significant'] = df_individual_results[p_col] < significance_threshold
            
            # Agrega contando os True (significativos) e o total para calcular a porcentagem
            sig_counts = df_individual_results.groupby(grouping_cols)[f'{metric}_is_significant'].agg(
                significant_count='sum', total_count='count'
            ).reset_index()
            
            sig_counts[f'{metric}_percent_significant'] = (sig_counts['significant_count'] / sig_counts['total_count']) * 100
            
            # Mescla de volta ao DataFrame agregado principal
            df_aggregated = pd.merge(df_aggregated, sig_counts[grouping_cols + [f'{metric}_percent_significant']], on=grouping_cols, how='left')

    return df_aggregated


def compare_phases_statistically(
    df_aggregated_results: pd.DataFrame,
    metric_col: str,
    phase_baseline: str = 'baseline',
    phase_attack: str = 'attack',
    grouping_cols_for_test: list = ['experiment_name', 'source_metric', 'target_metric']
) -> pd.DataFrame:
    """
    Realiza testes estatísticos (ex: t-test) para comparar uma métrica entre duas fases (ex: baseline vs attack).

    Args:
        df_aggregated_results (pd.DataFrame): DataFrame com os resultados individuais de todos os rounds.
        metric_col (str): Nome da coluna da métrica a ser testada (ex: 'te_value').
        phase_baseline (str): Rótulo da fase considerada baseline.
        phase_attack (str): Rótulo da fase a ser comparada com a baseline.
        grouping_cols_for_test (list): Colunas para agrupar os dados ANTES de realizar o teste,
                                        para comparar o mesmo cenário em diferentes fases.

    Returns:
        pd.DataFrame: DataFrame com os resultados dos testes estatísticos (t-statistic, p-value).
    """
    test_results = []

    # Iterar por cada cenário único (ex: 'Vanilla_cpu_usage_tenantA_p99_latency_tenantB')
    for name, group in df_aggregated_results.groupby(grouping_cols_for_test):
        data_baseline = group[group['phase_label'] == phase_baseline][metric_col].dropna()
        data_attack = group[group['phase_label'] == phase_attack][metric_col].dropna()

        if len(data_baseline) < 2 or len(data_attack) < 2: # Minimo de 2 amostras para t-test
            warnings.warn(f"Dados insuficientes para teste em cenário {name}. Pulando.")
            continue

        # Realiza o teste t de Student para duas amostras independentes
        # (Assumindo variâncias desiguais 'False' ou 'auto', dependendo do seu rigor)
        # Você pode considerar 'related' para ttest_rel se os rounds são pareados
        t_statistic, p_value = stats.ttest_ind(data_attack, data_baseline, equal_var=False) # Welch's t-test

        result = {
            'metric_analyzed': metric_col,
            'phase_compared': f"{phase_attack} vs {phase_baseline}",
            't_statistic': t_statistic,
            'p_value': p_value
        }
        # Adiciona as colunas de agrupamento ao resultado
        for i, col_name in enumerate(grouping_cols_for_test):
            result[col_name] = name[i] if isinstance(name, tuple) else name

        test_results.append(result)

    return pd.DataFrame(test_results)


# --- Exemplo de uso (para testar o módulo diretamente) ---
if __name__ == "__main__":
    print("Executando exemplos do módulo analytics/aggregation.py.")

    # Criar um DataFrame de resultados individuais mock (simulando a saída do loop do main.py)
    mock_data = []
    experiment_names = ['Vanilla', 'Kata']
    phases = ['baseline', 'attack', 'recovery']
    rounds_per_setup = 10
    
    for exp_name in experiment_names:
        for phase in phases:
            for r_num in range(1, rounds_per_setup + 1):
                # Simular TE values: Baseline baixo, Attack alto no Vanilla, Attack médio no Kata
                if phase == 'baseline':
                    te = np.random.uniform(0.05, 0.15)
                    pearson = np.random.uniform(0.1, 0.3)
                elif phase == 'attack' and exp_name == 'Vanilla':
                    te = np.random.uniform(0.7, 0.9)
                    pearson = np.random.uniform(0.6, 0.8)
                elif phase == 'attack' and exp_name == 'Kata':
                    te = np.random.uniform(0.3, 0.5)
                    pearson = np.random.uniform(0.4, 0.6)
                else: # Recovery
                    te = np.random.uniform(0.1, 0.25)
                    pearson = np.random.uniform(0.2, 0.4)

                # Simular p-values (menores para TE/Pearson altos)
                te_p = np.random.uniform(0.001, 0.01) if te > 0.5 else np.random.uniform(0.1, 0.5)
                pearson_p = np.random.uniform(0.001, 0.01) if pearson > 0.5 else np.random.uniform(0.1, 0.5)

                mock_data.append({
                    'experiment_name': exp_name,
                    'phase_label': phase,
                    'round_number': f'R{r_num}',
                    'source_metric': 'CPU_A',
                    'target_metric': 'Latency_B',
                    'te_value': te,
                    'te_p_value': te_p,
                    'pearson_corr': pearson,
                    'pearson_p': pearson_p,
                    'cosine_similarity_value': np.random.uniform(0.5, 0.9) if phase == 'attack' else np.random.uniform(0.1, 0.4)
                })
    
    df_mock_individual_results = pd.DataFrame(mock_data)
    print("DataFrame de resultados individuais mock:")
    print(df_mock_individual_results.head())
    print("\n")

    # --- Testando a função aggregate_analysis_results ---
    print("Agregando resultados...")
    df_aggregated = aggregate_analysis_results(
        df_mock_individual_results,
        grouping_cols=['experiment_name', 'phase_label', 'source_metric', 'target_metric'],
        metric_cols=['te_value', 'pearson_corr', 'cosine_similarity_value'],
        p_value_cols={'te_value': 'te_p_value', 'pearson_corr': 'pearson_p'},
        significance_threshold=0.05
    )
    print("DataFrame de resultados agregados:")
    print(df_aggregated)
    print("\n")

    # --- Testando a função compare_phases_statistically ---
    print("Realizando testes estatísticos de comparação de fases (Attack vs Baseline)...")
    df_statistical_comparison = compare_phases_statistically(
        df_mock_individual_results,
        metric_col='te_value',
        phase_baseline='baseline',
        phase_attack='attack',
        grouping_cols_for_test=['experiment_name', 'source_metric', 'target_metric']
    )
    print("Resultados dos testes estatísticos:")
    print(df_statistical_comparison)
    print("\n")
    print("Fim dos exemplos de agregação.")