# main.py (Seu orquestrador principal do pipeline)

import os
import pandas as pd

# Tentativa de importar módulos do projeto
try:
    from pipeline import config
    from pipeline.analysis import correlation_analysis
    from pipeline.analysis import causality_analysis
    from pipeline.visualization import plots
    from pipeline.data_processing import consolidation as loader
    # Os módulos/objetos 'preprocessor', e 'tables' precisam ser definidos ou importados.
    # Exemplo:
    # from pipeline.data_processing import data_preprocessor as preprocessor
    # from pipeline.visualization import table_generator as tables
except ImportError:
    print("AVISO: Falha ao importar módulos do projeto. Verifique os caminhos e a estrutura do projeto.")
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from pipeline import config
    from pipeline.analysis import correlation_analysis
    from pipeline.analysis import causality_analysis
    from pipeline.visualization import plots
    from pipeline.data_processing import consolidation as loader
    # Ainda precisaria definir 'preprocessor', 'tables'

# --- Placeholder para módulos/funções não resolvidas ---
class MockPreprocessor:
    def resample_and_interpolate(self, df, target_frequency, interpolation_method): return df
    def handle_outliers(self, df): return df

class MockTables:
    def generate_summary_table_of_causality(self, df): return df

preprocessor = MockPreprocessor()
tables_utils = MockTables()
# --- Fim dos Placeholders ---

def run_analysis_pipeline(base_data_path: str):
    print("--- 1. INÍCIO DO PIPELINE DE ANÁLISE ---")

    # 1. ESTAÇÃO DE CARREGAMENTO (data_loader.py)
    print(f"1. Carregando dados brutos a partir de: {base_data_path}")
    raw_loaded_data = loader.load_all_experiment_data(base_data_path)

    if raw_loaded_data.empty:
        print("Nenhum dado carregado. Encerrando pipeline.")
        return

    # 2. ESTAÇÃO DE PRÉ-PROCESSAMENTO (data_preprocessor.py)
    print("2. Pré-processando e limpando os dados...")
    processed_data = preprocessor.resample_and_interpolate(raw_loaded_data,
                                                         target_frequency=f"{config.SCRAPE_INTERVAL_S}s",
                                                         interpolation_method=config.INTERPOLATION_METHOD)
    processed_data = preprocessor.handle_outliers(processed_data)
    
    # 3. ESTAÇÃO DE ANÁLISE (analytics/*.py)
    print("3. Executando análises de correlação, similaridade e causalidade...")
    
    all_calculated_results = []
    
    unique_segments = processed_data[['experiment_run_id', 'round_number', 'phase_label']].drop_duplicates()

    for _, row in unique_segments.iterrows():
        exp_run_id = row['experiment_run_id']
        round_num = row['round_number']
        phase_lbl = row['phase_label']
        
        segment_df = processed_data[
            (processed_data['experiment_run_id'] == exp_run_id) &
            (processed_data['round_number'] == round_num) &
            (processed_data['phase_label'] == phase_lbl)
        ]
        
        if segment_df.empty:
            continue

        print(f"   Analisando segmento: Run={exp_run_id}, Round={round_num}, Fase={phase_lbl}")
        
        try:
            if 'value' in segment_df.columns and not segment_df.empty:
                if len(segment_df) >= 2:
                    tenant_A_cpu = pd.Series(segment_df['value'].head(10).fillna(0).values, name="tenant_A_cpu")
                    tenant_B_latency = pd.Series(segment_df['value'].tail(10).fillna(0).values, name="tenant_B_latency")
                else:
                    print(f"     Dados insuficientes no segmento para análise {exp_run_id}, {round_num}, {phase_lbl}")
                    continue
            else:
                print(f"     Coluna 'value' não encontrada ou segmento vazio para {exp_run_id}, {round_num}, {phase_lbl}")
                continue

            if not tenant_A_cpu.empty and not tenant_B_latency.empty and len(tenant_A_cpu) > 1 and len(tenant_B_latency) > 1:
                pearson_corr, pearson_p = correlation_analysis.calculate_pearson_correlation(tenant_A_cpu, tenant_B_latency)
                
                te_value = causality_analysis.calculate_transfer_entropy(
                    [tenant_A_cpu.values, tenant_B_latency.values],
                    k=config.TE_K_LAG,
                    l=config.TE_L_LAG
                )
                te_p_value = None
                
                all_calculated_results.append({
                    'experiment_run_id': exp_run_id,
                    'round_number': round_num,
                    'phase_label': phase_lbl,
                    'source_metric': 'cpu_usage_tenantA',
                    'target_metric': 'p99_latency_tenantB',
                    'pearson_corr': pearson_corr,
                    'pearson_p': pearson_p,
                    'te_value': te_value,
                    'te_p_value': te_p_value
                })
            else:
                print(f"     Dados insuficientes para análise de um par em {exp_run_id}, {round_num}, {phase_lbl} após extração.")

        except Exception as e:
            print(f"     Erro na análise de segmento {exp_run_id}, {round_num}, {phase_lbl}: {e}")

    final_analysis_results_df = pd.DataFrame(all_calculated_results)
    if final_analysis_results_df.empty:
        print("Nenhum resultado de análise gerado.")
        return

    # 4. ESTAÇÃO DE AGREGAÇÃO E VISUALIZAÇÃO (visualization/*.py e tables.py)
    print("4. Agregando resultados e gerando visualizações/tabelas...")
    
    if not final_analysis_results_df.empty:
        avg_te_by_phase = final_analysis_results_df.groupby('phase_label')['te_value'].mean()
        print("\nMédia da TE por Fase (em todos os rounds):")
        print(avg_te_by_phase)

        # plots.plot_boxplot_causality_across_rounds(
        #     final_analysis_results_df,
        #     metric_pair=('cpu_usage_tenantA', 'p99_latency_tenantB'), 
        #     output_filename="te_boxplot_by_phase.png"
        # )

        # summary_table_df = tables_utils.generate_summary_table_of_causality(final_analysis_results_df)
        # print("\nResumo dos Resultados de Causalidade:")
        # print(summary_table_df)
        # if summary_table_df is not None and not summary_table_df.empty:
        #    summary_table_df.to_csv("causality_summary.csv", index=False)
    else:
        print("Nenhum resultado de análise para agregar ou visualizar.")

    print("\n--- PIPELINE DE ANÁLISE CONCLUÍDO COM SUCESSO! ---")

# --- Execução do Pipeline ---
# Comentado pois a execução principal deve ser via pipeline/main.py
# if __name__ == "__main__":
#     os.makedirs('output', exist_ok=True) 
#     run_analysis_pipeline(base_data_path='results')