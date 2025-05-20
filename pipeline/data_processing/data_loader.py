# main.py (Seu orquestrador principal do pipeline)

import os
import pandas as pd

# Tentativa de importar módulos do projeto
try:
    from pipeline import config
    from pipeline.analysis import correlation_analysis
    from pipeline.analysis import causality_analysis
    from pipeline.visualization import plots
    # from pipeline.data_processing import consolidation as loader # Commented out
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
    # from pipeline.data_processing import consolidation as loader # Commented out
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

def load_experiment_data(base_data_path, specific_metrics=None, specific_rounds=None, specific_phases=None, specific_tenants=None, node_config_name=None, time_col='timestamp', value_col='value', specific_metrics_map=None):
    """
    Loads, processes, and aggregates experiment data from a directory structure.
    
    Args:
        base_data_path: Path to the experiment data directory
        specific_metrics: List of metric names to load (optional)
        specific_rounds: List of round names to load (optional)
        specific_phases: List of phase names to load (optional)
        specific_tenants: List of tenant names to load (optional)
        node_config_name: Name of the node configuration to use (optional)
        time_col: Name of the timestamp column (default: 'timestamp')
        value_col: Name of the value column (default: 'value')
        specific_metrics_map: Dictionary mapping metric names to their file names (optional)
    """
    print(f"Starting data loading from: {base_data_path}")
    
    # Check if base_data_path exists and is a directory
    if not os.path.isdir(base_data_path):
        print(f"Error: Base data path does not exist or is not a directory: {base_data_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Checking if it's a relative path in demo-data...")
        
        # Try to resolve as a relative path from demo-data
        demo_data_path = os.path.join(os.getcwd(), "demo-data", base_data_path)
        if os.path.isdir(demo_data_path):
            print(f"Found data directory in demo-data: {demo_data_path}")
            base_data_path = demo_data_path
        else:
            print(f"Could not find data directory at {demo_data_path} either")
            return {}, {}
    
    print(f"Using data directory: {base_data_path}")
    
    # If specific_metrics_map is provided, use it to map metrics to filenames
    if specific_metrics_map and specific_metrics:
        metric_filenames = []
        for metric_name in specific_metrics:
            metric_filename = specific_metrics_map.get(metric_name, metric_name)
            metric_filenames.append(metric_filename)
        specific_metrics = metric_filenames
    
    raw_loaded_data = {}
    
    # List available rounds
    available_rounds = [d for d in os.listdir(base_data_path) if os.path.isdir(os.path.join(base_data_path, d))]
    if specific_rounds:
        available_rounds = [r for r in available_rounds if r in specific_rounds]
    
    if not available_rounds:
        print(f"Warning: No valid rounds found in {base_data_path}")
        print(f"Available directories: {os.listdir(base_data_path)}")
        return {}, {}
    
    print(f"Found rounds: {available_rounds}")
    
    # Process each round
    for round_name in available_rounds:
        round_path = os.path.join(base_data_path, round_name)
        raw_loaded_data[round_name] = {}
        
        # List available phases
        available_phases = [d for d in os.listdir(round_path) if os.path.isdir(os.path.join(round_path, d))]
        if specific_phases:
            available_phases = [p for p in available_phases if p in specific_phases]
        
        if not available_phases:
            print(f"Warning: No valid phases found in round {round_name}")
            continue
        
        print(f"Processing round: {round_name}, found phases: {available_phases}")
        
        # Process each phase
        for phase_name in available_phases:
            phase_path = os.path.join(round_path, phase_name)
            raw_loaded_data[round_name][phase_name] = {}
            
            # List available tenants
            available_tenants = [d for d in os.listdir(phase_path) if os.path.isdir(os.path.join(phase_path, d))]
            if specific_tenants:
                available_tenants = [t for t in available_tenants if t in specific_tenants]
            
            if not available_tenants:
                print(f"Warning: No valid tenants found in phase {phase_name}")
                continue
            
            print(f"  Processing phase: {phase_name}, found tenants: {available_tenants}")
            
            # Process each tenant
            for tenant_name in available_tenants:
                tenant_path = os.path.join(phase_path, tenant_name)
                raw_loaded_data[round_name][phase_name][tenant_name] = {}
                
                # List available metrics (CSV files)
                all_files = os.listdir(tenant_path)
                csv_files = [f for f in all_files if f.endswith('.csv')]
                
                if not csv_files:
                    print(f"    Warning: No CSV files found for tenant {tenant_name}")
                    continue
                
                # Filter metrics if specific ones are requested
                filtered_csv_files = csv_files
                if specific_metrics:
                    filtered_csv_files = []
                    for metric in specific_metrics:
                        metric_file = f"{metric}.csv"
                        if metric_file in csv_files:
                            filtered_csv_files.append(metric_file)
                
                if not filtered_csv_files:
                    print(f"    Warning: No matching metric files for tenant {tenant_name}")
                    if specific_metrics:
                        print(f"    Requested metrics: {specific_metrics}")
                        print(f"    Available metrics: {[os.path.splitext(f)[0] for f in csv_files]}")
                    continue
                
                print(f"    Processing tenant: {tenant_name}, found metrics: {[os.path.splitext(f)[0] for f in filtered_csv_files]}")
                
                # Process each metric file
                for csv_file in filtered_csv_files:
                    metric_name = os.path.splitext(csv_file)[0]
                    file_path = os.path.join(tenant_path, csv_file)
                    
                    try:
                        # Read CSV file
                        print(f"      Reading file: {file_path}")
                        df = pd.read_csv(file_path)
                        
                        # Validate columns
                        if time_col not in df.columns:
                            print(f"      Error: Missing time column '{time_col}' in {file_path}")
                            print(f"      Available columns: {df.columns.tolist()}")
                            continue
                        
                        if value_col not in df.columns:
                            print(f"      Error: Missing value column '{value_col}' in {file_path}")
                            print(f"      Available columns: {df.columns.tolist()}")
                            continue
                        
                        # Handle timestamp parsing
                        try:
                            # First try with specified format (YYYYMMDD_HHMMSS)
                            df[time_col] = pd.to_datetime(df[time_col], format="%Y%m%d_%H%M%S")
                            print(f"      Successfully parsed timestamps with format '%Y%m%d_%H%M%S'")
                        except ValueError:
                            try:
                                # Try another common format with quotes
                                df[time_col] = pd.to_datetime(df[time_col].str.replace('"', ''), format="%Y%m%d_%H%M%S")
                                print(f"      Successfully parsed timestamps after removing quotes")
                            except (ValueError, AttributeError):
                                # Fall back to automatic parsing
                                print(f"      Warning: Using automatic datetime parsing for {file_path}")
                                df[time_col] = pd.to_datetime(df[time_col])
                        
                        # Sort by timestamp
                        df.sort_values(by=time_col, inplace=True)
                        
                        # Store the DataFrame
                        raw_loaded_data[round_name][phase_name][tenant_name][metric_name] = df
                        print(f"      Successfully loaded {len(df)} rows for {metric_name}")
                        
                    except Exception as e:
                        print(f"      Error processing {file_path}: {str(e)}")
                        continue
    
    # Check if any data was loaded
    data_loaded = False
    for round_data in raw_loaded_data.values():
        for phase_data in round_data.values():
            for tenant_data in phase_data.values():
                if tenant_data:  # If there's at least one metric
                    data_loaded = True
                    break
            if data_loaded:
                break
        if data_loaded:
            break
    
    if not data_loaded:
        print("No data was loaded. Check the specified data directory, metrics, and rounds.")
        return {}, {}
    
    # Process the raw data into a standard format for the pipeline
    metrics_data = {}
    
    # For each round
    for round_name, round_data in raw_loaded_data.items():
        # For each phase
        for phase_name, phase_data in round_data.items():
            # For each tenant
            for tenant_name, tenant_data in phase_data.items():
                # For each metric
                for metric_name, df in tenant_data.items():
                    if metric_name not in metrics_data:
                        metrics_data[metric_name] = []
                    
                    # Add round, phase, tenant info to dataframe
                    df = df.copy()
                    df['round'] = round_name
                    df['phase'] = phase_name
                    df['tenant'] = tenant_name
                    
                    # Append to metrics_data
                    metrics_data[metric_name].append(df)
    
    # Combine all dataframes for each metric
    for metric_name in metrics_data:
        if metrics_data[metric_name]:
            metrics_data[metric_name] = pd.concat(metrics_data[metric_name], ignore_index=True)
            print(f"Processed metric: {metric_name}, data points: {len(metrics_data[metric_name])}")
        else:
            metrics_data[metric_name] = pd.DataFrame()
            print(f"No data for metric: {metric_name}")
    
    return metrics_data

def list_available_metrics(base_data_path):
    """
    Scans the data directory structure to find available metrics.
    
    The directory structure is expected to be:
    base_data_path/round-X/phase-Y/tenant-Z/*.csv
    where the CSV files represent different metrics.
    
    Returns a list of available metric names based on CSV files found.
    """
    print(f"Scanning for available metrics in: {base_data_path}")
    metrics = set()
    
    # Check if base_data_path exists and is a directory
    if not os.path.isdir(base_data_path):
        print(f"Warning: Base data path does not exist or is not a directory: {base_data_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Checking if it's a relative path in demo-data...")
        
        # Try to resolve as a relative path from demo-data
        demo_data_path = os.path.join(os.getcwd(), "demo-data", base_data_path)
        if os.path.isdir(demo_data_path):
            print(f"Found data directory in demo-data: {demo_data_path}")
            base_data_path = demo_data_path
        else:
            print(f"Could not find data directory at {demo_data_path} either")
            from pipeline.config import DEFAULT_METRICS
            print(f"Falling back to default metrics: {DEFAULT_METRICS}")
            return DEFAULT_METRICS
    
    # Explore the directory structure - rounds
    for round_name in os.listdir(base_data_path):
        round_path = os.path.join(base_data_path, round_name)
        if not os.path.isdir(round_path):
            continue
            
        # Phases within this round
        for phase_name in os.listdir(round_path):
            phase_path = os.path.join(round_path, phase_name)
            if not os.path.isdir(phase_path):
                continue
                
            # Tenants within this phase
            for tenant_name in os.listdir(phase_path):
                tenant_path = os.path.join(phase_path, tenant_name)
                if not os.path.isdir(tenant_path):
                    continue
                    
                # Get all CSV files = metrics
                for file_name in os.listdir(tenant_path):
                    if file_name.endswith('.csv'):
                        # Extract metric name from filename (without extension)
                        metric_name = os.path.splitext(file_name)[0]
                        metrics.add(metric_name)
    
    metrics_list = sorted(list(metrics))
    print(f"Found {len(metrics_list)} metrics: {metrics_list}")
    
    # If no metrics found, use DEFAULT_METRICS from config
    if not metrics_list:
        print("No metrics found in directory structure. Using default metrics from config.")
        from pipeline.config import DEFAULT_METRICS
        metrics_list = DEFAULT_METRICS
        
    return metrics_list

def run_analysis_pipeline(base_data_path: str):
    print("--- 1. INÍCIO DO PIPELINE DE ANÁLISE ---")

    # 1. ESTAÇÃO DE CARREGAMENTO (data_loader.py)
    print(f"1. Carregando dados brutos a partir de: {base_data_path}")
    raw_loaded_data = load_experiment_data(base_data_path)

    if not raw_loaded_data:
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