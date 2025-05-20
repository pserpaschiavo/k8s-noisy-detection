import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os # Para criar diretórios de saída
import networkx as nx # Added for visualize_causal_graph
import warnings # Added to fix undefined warnings object

# --- Configurações Globais de Plotagem ---
# Isso ajuda a garantir que todos os seus gráficos tenham uma aparência consistente e profissional
sns.set_theme(style="whitegrid", palette="viridis") # Estilo padrão do seaborn
plt.rcParams['figure.figsize'] = (12, 7) # Tamanho padrão da figura
plt.rcParams['font.size'] = 12 # Tamanho da fonte padrão
plt.rcParams['axes.labelsize'] = 14 # Tamanho da fonte dos rótulos dos eixos
plt.rcParams['axes.titlesize'] = 16 # Tamanho da fonte dos títulos
plt.rcParams['xtick.labelsize'] = 12 # Tamanho da fonte dos ticks do eixo X
plt.rcParams['ytick.labelsize'] = 12 # Tamanho da fonte dos ticks do eixo Y
plt.rcParams['legend.fontsize'] = 12 # Tamanho da fonte da legenda
plt.rcParams['lines.linewidth'] = 2 # Espessura das linhas
plt.rcParams['grid.alpha'] = 0.7 # Transparência da grade

OUTPUT_DIR = 'output/plots' # Diretório padrão para salvar os gráficos

# Garante que o diretório de saída exista
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Funções de Plotagem ---

def save_plot(fig: plt.Figure, filename: str):
    """Função auxiliar para salvar o plot no diretório de saída."""
    # If filename is already an absolute path or contains path separators, 
    # use it directly. Otherwise, join with OUTPUT_DIR
    if os.path.isabs(filename) or os.sep in filename:
        filepath = filename
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    else:
        filepath = os.path.join(OUTPUT_DIR, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    fig.tight_layout() # Ajusta o layout para evitar cortes de rótulos
    fig.savefig(filepath, dpi=300, bbox_inches='tight') # Salva em alta resolução
    plt.close(fig) # Fecha a figura para liberar memória
    print(f"Gráfico salvo em: {filepath}")

# --- 1. Plots da Fase de Triagem e Pré-análise ---

def plot_correlation_heatmap(df_correlation_matrix: pd.DataFrame, title: str = "Matriz de Correlação entre Métricas", output_filename: str = "correlation_heatmap.png"):
    """
    Gera um heatmap de uma matriz de correlação.

    Args:
        df_correlation_matrix (pd.DataFrame): Matriz de correlação (ex: resultado de df.corr()).
        title (str): Título do gráfico.
        output_filename (str): Nome do arquivo para salvar o plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax,
                cbar_kws={'label': 'Coeficiente de Correlação'})
    ax.set_title(title)
    save_plot(fig, output_filename)


def plot_ccf(series_a: pd.Series, series_b: pd.Series, ccf_results: dict, scrape_interval_s: int = 5, output_filename: str = "ccf_plot.png"):
    """
    Gera um plot da Função de Correlação Cruzada (FCC).

    Args:
        series_a (pd.Series): A primeira série temporal (origem).
        series_b (pd.Series): A segunda série temporal (destino).
        ccf_results (dict): Dicionário com 'lags' e 'correlation_values' (e opcionalmente 'conf_interval_upper', 'conf_interval_lower').
        scrape_interval_s (int): Intervalo de coleta das métricas em segundos.
        output_filename (str): Nome do arquivo para salvar o plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Converte os lags de número de amostras para segundos
    time_lags_s = [lag * scrape_interval_s for lag in ccf_results['lags']]
    
    ax.plot(time_lags_s, ccf_results['correlation_values'], marker='o', linestyle='-')
    
    # Adiciona bandas de confiança se disponíveis
    if 'conf_interval_upper' in ccf_results and 'conf_interval_lower' in ccf_results:
        ax.axhline(ccf_results['conf_interval_upper'], color='red', linestyle='--', label='Intervalo de Confiança (95%)')
        ax.axhline(ccf_results['conf_interval_lower'], color='red', linestyle='--')
    
    ax.set_title(f"Função de Correlação Cruzada: {series_a.name} vs {series_b.name}")
    ax.set_xlabel("Lag (segundos)")
    ax.set_ylabel("Coeficiente de Correlação")
    ax.grid(True)
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.8) # Linha no lag 0
    ax.legend()
    save_plot(fig, output_filename)


# --- 2. Plots da Fase de Análise de Similaridade de Padrões ---

def plot_dtw_path(series_a: pd.Series, series_b: pd.Series, dtw_path: list, output_filename: str = "dtw_path_plot.png"):
    """
    Gera um plot do caminho de alinhamento ótimo do DTW.

    Args:
        series_a (pd.Series): A primeira série temporal.
        series_b (pd.Series): A segunda série temporal.
        dtw_path (list): Lista de tuplas (idx_a, idx_b) representando o caminho de alinhamento.
        output_filename (str): Nome do arquivo para salvar o plot.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plotar as séries originais
    # As séries devem ter um índice de tempo real (timestamp) para que o tempo decorrido funcione
    # Garantir que o índice seja um datetime para calcular o tempo decorrido
    if not isinstance(series_a.index, pd.DatetimeIndex):
        warnings.warn("O índice da série A não é DatetimeIndex. O eixo X pode não ser tempo decorrido.")
        x_a = np.arange(len(series_a))
    else:
        start_time_a = series_a.index.min()
        x_a = (series_a.index - start_time_a).total_seconds()
        ax1.set_xlabel('Tempo Decorrido (segundos)')

    if not isinstance(series_b.index, pd.DatetimeIndex):
        warnings.warn("O índice da série B não é DatetimeIndex. O eixo X pode não ser tempo decorrido.")
        x_b = np.arange(len(series_b))
    else:
        start_time_b = series_b.index.min()
        x_b = (series_b.index - start_time_b).total_seconds()
        ax2.set_xlabel('Tempo Decorrido (segundos)')


    ax1.plot(x_a, series_a.values, label=series_a.name, color='blue')
    ax2.plot(x_b, series_b.values, label=series_b.name, color='orange')
    
    ax1.set_ylabel(series_a.name)
    ax2.set_ylabel(series_b.name)
    ax1.legend()
    ax2.legend()
    
    # Plotar o caminho de empenamento
    # Assume que dtw_path contém índices das séries originais
    for (idx_a, idx_b) in dtw_path:
        # Pega os valores de tempo correspondentes aos índices
        time_a = x_a[idx_a]
        time_b = x_b[idx_b]
        ax1.plot([time_a, time_b], [series_a.values[idx_a], series_b.values[idx_b]], 
                 'gray', linestyle=':', linewidth=0.5, alpha=0.5, transform=fig.transFigure) # Isso não está 100% correto para coordenadas do plot
        # Uma forma mais robusta é plotar linhas entre as séries diretamente:
        ax1.plot([time_a, time_b], [series_a.values[idx_a], series_b.values[idx_b]], color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    save_plot(fig, output_filename)


def plot_aligned_time_series(series_a_aligned: pd.Series, series_b_aligned: pd.Series, output_filename: str = "aligned_time_series_plot.png"):
    """
    Gera um plot das duas séries temporais após o alinhamento DTW.

    Args:
        series_a_aligned (pd.Series): A primeira série temporal alinhada.
        series_b_aligned (pd.Series): A segunda série temporal alinhada.
        output_filename (str): Nome do arquivo para salvar o plot.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # As séries alinhadas podem ter um novo índice ou serem de tamanhos diferentes
    # O ideal é que o processo de alinhamento retorne séries que podem ser plotadas juntas
    # Aqui, assumimos que elas têm um índice comum ou um índice que representa o "tempo alinhado"
    
    # Calcular tempo decorrido se o índice for datetime
    if isinstance(series_a_aligned.index, pd.DatetimeIndex):
        start_time = series_a_aligned.index.min()
        x_axis = (series_a_aligned.index - start_time).total_seconds()
        ax.set_xlabel("Tempo Alinhado (segundos)")
    else:
        x_axis = np.arange(len(series_a_aligned))
        ax.set_xlabel("Ponto de Dados Alinhado")


    ax.plot(x_axis, series_a_aligned.values, label=series_a_aligned.name, color='blue', alpha=0.8)
    ax.plot(x_axis, series_b_aligned.values, label=series_b_aligned.name, color='orange', alpha=0.8, linestyle='--')
    
    ax.set_title(f"Séries Temporais Alinhadas por DTW: {series_a_aligned.name} vs {series_b_aligned.name}")
    ax.set_ylabel("Valor da Métrica")
    ax.legend()
    ax.grid(True)
    save_plot(fig, output_filename)


def plot_dtw_distance_heatmap(df_dtw_distances: pd.DataFrame, title: str = "Matriz de Distâncias DTW", output_filename: str = "dtw_distance_heatmap.png"):
    """
    Gera um heatmap de uma matriz de distâncias DTW.

    Args:
        df_dtw_distances (pd.DataFrame): Matriz de distâncias DTW.
        title (str): Título do gráfico.
        output_filename (str): Nome do arquivo para salvar o plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    # Usar cmap='Blues_r' ou similar para que distâncias menores (mais similares) sejam mais escuras
    sns.heatmap(df_dtw_distances, annot=True, cmap='viridis_r', fmt=".2f", linewidths=.5, ax=ax,
                cbar_kws={'label': 'Distância DTW (menor = mais similar)'})
    ax.set_title(title)
    save_plot(fig, output_filename)


def plot_time_series_with_cosine_similarity(series_a: pd.Series, series_b: pd.Series, cosine_similarity_score: float, title: str = "Time Series with Cosine Similarity", output_filename: str = "time_series_cosine_similarity.png"):
    """
    Plots two time series and displays their cosine similarity score.

    Args:
        series_a (pd.Series): The first time series.
        series_b (pd.Series): The second time series.
        cosine_similarity_score (float): The cosine similarity score between the two series.
        title (str): Title of the plot.
        output_filename (str): Filename for saving the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Determine x-axis based on index type
    if isinstance(series_a.index, pd.DatetimeIndex) and isinstance(series_b.index, pd.DatetimeIndex):
        x_a = (series_a.index - series_a.index.min()).total_seconds() if not series_a.index.empty else []
        x_b = (series_b.index - series_b.index.min()).total_seconds() if not series_b.index.empty else []
        ax.set_xlabel("Tempo Decorrido (segundos)")
    else:
        x_a = np.arange(len(series_a))
        x_b = np.arange(len(series_b))
        ax.set_xlabel("Ponto de Dados")

    ax.plot(x_a, series_a.values, label=series_a.name or 'Series A', color='blue', alpha=0.8)
    ax.plot(x_b, series_b.values, label=series_b.name or 'Series B', color='orange', alpha=0.8, linestyle='--')
    
    ax.set_title(title)
    ax.set_ylabel("Valor da Métrica")
    ax.legend(loc='upper left')
    
    # Display cosine similarity score on the plot
    ax.text(0.05, 0.95, f"Cosine Similarity: {cosine_similarity_score:.4f}",
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
            
    ax.grid(True)
    save_plot(fig, output_filename)


def plot_cosine_similarity_heatmap(similarity_matrix: pd.DataFrame, title: str = "Heatmap de Similaridade de Cossenos", output_filename: str = "cosine_similarity_heatmap.png"):
    """
    Gera um heatmap de uma matriz de similaridade de cossenos.

    Args:
        similarity_matrix (pd.DataFrame): Matriz de similaridade de cossenos.
        title (str): Título do gráfico.
        output_filename (str): Nome do arquivo para salvar o plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, ax=ax,
                cbar_kws={'label': 'Similaridade de Cossenos'})
    ax.set_title(title)
    save_plot(fig, output_filename)


def plot_time_varying_cosine_similarity(df_time_varying_similarity: pd.DataFrame, series_a_name: str = 'Series A', series_b_name: str = 'Series B', output_filename: str = "time_varying_cosine_similarity.png", phase_start_time: pd.Timestamp = None):
    """
    Plota a similaridade de cossenos variante no tempo entre duas séries.

    Args:
        df_time_varying_similarity (pd.DataFrame): DataFrame com colunas 'timestamp' (ou índice de tempo)
                                                  e 'cosine_similarity'.
        series_a_name (str): Nome da primeira série.
        series_b_name (str): Nome da segunda série.
        output_filename (str): Nome do arquivo para salvar o plot.
        phase_start_time (pd.Timestamp): Timestamp de início da fase para calcular tempo decorrido.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    if 'timestamp' in df_time_varying_similarity.columns and phase_start_time is not None and isinstance(df_time_varying_similarity['timestamp'].iloc[0], pd.Timestamp):
        x_axis = (df_time_varying_similarity['timestamp'] - phase_start_time).dt.total_seconds()
        ax.set_xlabel("Tempo Decorrido desde o Início da Fase (segundos)")
    elif isinstance(df_time_varying_similarity.index, pd.DatetimeIndex) and phase_start_time is not None:
        x_axis = (df_time_varying_similarity.index - phase_start_time).total_seconds()
        ax.set_xlabel("Tempo Decorrido desde o Início da Fase (segundos)")
    elif 'timestamp' in df_time_varying_similarity.columns:
        x_axis = df_time_varying_similarity['timestamp']
        ax.set_xlabel("Timestamp")
    else: # Assume o índice é o eixo x (ex: window number)
        x_axis = df_time_varying_similarity.index
        ax.set_xlabel("Janela de Tempo / Ponto de Dados")


    ax.plot(x_axis, df_time_varying_similarity['cosine_similarity'], label=f'Cosine Similarity ({series_a_name} vs {series_b_name})')
    ax.set_title(f"Similaridade de Cossenos Variante no Tempo: {series_a_name} vs {series_b_name}")
    ax.set_ylabel("Similaridade de Cossenos")
    ax.set_ylim(-1.05, 1.05) # Cosine similarity ranges from -1 to 1
    ax.grid(True)
    ax.legend()
    save_plot(fig, output_filename)


# --- 3. Plots da Fase de Análise de Causalidade ---

def visualize_causal_graph(causality_results_df, output_path, title='Causal Graph', significance_level=0.05):
    """
    Visualizes the causal relationships as a directed graph.
    """
    output_filename = os.path.basename(output_path)
    
    if causality_results_df.empty:
        print("Causality results are empty. Cannot generate graph.")
        return

    G = nx.DiGraph()
    
    if 'p_value' not in causality_results_df.columns:
        print("Column 'p_value' not found in causality_results_df. Cannot filter by significance.")
        significant_results = causality_results_df
    else:
        significant_results = causality_results_df[causality_results_df['p_value'] < significance_level]

    if significant_results.empty:
        print(f"No significant causal relationships found at p < {significance_level}. Graph will be empty or show no edges.")
        if 'source_tenant' in causality_results_df.columns and 'target_tenant' in causality_results_df.columns:
            all_tenants_in_results = pd.unique(causality_results_df[['source_tenant', 'target_tenant']].values.ravel('K'))
            for tenant in all_tenants_in_results:
                if tenant and pd.notna(tenant): G.add_node(str(tenant))
        else:
            print("Columns 'source_tenant' or 'target_tenant' not found. Cannot add nodes for empty graph.")
    else:
        for _, row in significant_results.iterrows():
            source = str(row['source_tenant'])
            target = str(row['target_tenant'])
            metric = str(row.get('metric', 'N/A'))
            p_value = row['p_value']
            
            G.add_node(source)
            G.add_node(target)
            
            label = f"{metric}\\n(p={p_value:.3f})"
            current_weight = 1 / (p_value + 1e-6)

            if G.has_edge(source, target):
                if p_value < G[source][target].get('p_value', float('inf')):
                    G[source][target]['label'] = label
                    G[source][target]['p_value'] = p_value
                    G[source][target]['weight'] = current_weight
            else:
                G.add_edge(source, target, label=label, p_value=p_value, weight=current_weight)

    if not G.nodes():
        print("No nodes to draw in the causal graph.")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3000, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
    
    edges = G.edges(data=True)
    if edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u,v) for u,v,d in edges],
                               arrowstyle='-|>', arrowsize=20, edge_color='gray', alpha=0.6, node_size=3000)
        edge_labels = {(u,v): d['label'] for u,v,d in edges if 'label' in d}
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=8, label_pos=0.3)

    ax.set_title(title, fontsize=15)
    ax.axis('off')
    save_plot(fig, output_filename)

def plot_causality_heatmap(df_causality_matrix: pd.DataFrame, title: str = "Matriz de Causalidade (TE/CCM)", output_filename: str = "causality_heatmap.png"):
    """
    Gera um heatmap de uma matriz de valores de causalidade (TE, CCM, Granger).

    Args:
        df_causality_matrix (pd.DataFrame): Matriz de causalidade (linhas: fontes, colunas: destinos).
        title (str): Título do gráfico.
        output_filename (str): Nome do arquivo para salvar o plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_causality_matrix, annot=True, cmap='plasma', fmt=".2f", linewidths=.5, ax=ax,
                cbar_kws={'label': 'Força da Causalidade'})
    ax.set_title(title)
    save_plot(fig, output_filename)

def plot_time_varying_causality(df_time_varying_causality: pd.DataFrame, source_col: str = 'source_metric', target_col: str = 'target_metric', causality_value_col: str = 'te_value', output_filename: str = "time_varying_causality.png", phase_start_time: pd.Timestamp = None):
    """
    Gera um plot da causalidade (TE/CCM) ao longo do tempo.

    Args:
        df_time_varying_causality (pd.DataFrame): DataFrame com colunas 'timestamp', 'te_value' (ou valor CCM).
        source_col (str): Nome da métrica/tenant fonte.
        target_col (str): Nome da métrica/tenant destino.
        causality_value_col (str): Nome da coluna com o valor da causalidade (ex: 'te_value').
        output_filename (str): Nome do arquivo para salvar o plot.
        phase_start_time (pd.Timestamp): Timestamp de início da fase para calcular tempo decorrido.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    if phase_start_time is not None and isinstance(df_time_varying_causality['timestamp'].iloc[0], pd.Timestamp):
        x_axis = (df_time_varying_causality['timestamp'] - phase_start_time).dt.total_seconds()
        ax.set_xlabel("Tempo Decorrido (segundos)")
    else:
        x_axis = df_time_varying_causality['timestamp']
        ax.set_xlabel("Timestamp")

    ax.plot(x_axis, df_time_varying_causality[causality_value_col], label=f'{source_col} -> {target_col}')
    ax.set_title(f"Causalidade ao Longo do Tempo: {source_col} para {target_col}")
    ax.set_ylabel(f"{causality_value_col.replace('_', ' ').title()}")
    ax.grid(True)
    ax.legend()
    save_plot(fig, output_filename)

# --- Plots de Análise de Covariância/Correlação ---
def plot_covariance_matrix(covariance_matrix, output_filename, title='Covariance Matrix', cmap='coolwarm', annot=True, fmt=".2f"):
    """
    Plots a covariance matrix as a heatmap.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(covariance_matrix, annot=annot, fmt=fmt, cmap=cmap, linewidths=.5, ax=ax)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45, ha='right')
    ax.tick_params(axis='y', rotation=0)
    save_plot(fig, output_filename)

# --- Plots para Comparação Cross-Round/Cross-Setup ---

def plot_boxplot_causality_across_rounds_and_setups(
    df_aggregated_results: pd.DataFrame, 
    metric_pair: tuple, 
    causality_value_col: str = 'te_value',
    title: str = "Distribuição da Causalidade por Fase e Configuração de Isolamento",
    output_filename: str = "boxplot_causality_comparison.png"
):
    """
    Gera um boxplot para comparar a distribuição de valores de causalidade
    (TE/CCM) através de diferentes fases e/ou setups de isolamento.

    Args:
        df_aggregated_results (pd.DataFrame): DataFrame com os resultados de causalidade
                                              de cada round/fase/setup (ex: 'te_value', 'phase_label', 'experiment_name').
        metric_pair (tuple): Par de métricas (source, target) para filtrar se o DF contém múltiplos pares.
        causality_value_col (str): Nome da coluna com o valor da causalidade (ex: 'te_value').
        title (str): Título do gráfico.
        output_filename (str): Nome do arquivo para salvar o plot.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    df_filtered = df_aggregated_results

    if 'experiment_name' in df_filtered.columns:
        df_filtered['phase_setup'] = df_filtered['phase_label'] + ' (' + df_filtered['experiment_name'] + ')'
        x_col = 'phase_setup'
    else:
        x_col = 'phase_label'

    sns.boxplot(x=x_col, y=causality_value_col, data=df_filtered, ax=ax)
    sns.stripplot(x=x_col, y=causality_value_col, data=df_filtered, color=".3", size=4, jitter=True, ax=ax)

    ax.set_title(f"{title}\n({metric_pair[0]} -> {metric_pair[1]})")
    ax.set_xlabel("Fase e Configuração de Isolamento")
    ax.set_ylabel(f"{causality_value_col.replace('_', ' ').title()}")
    ax.tick_params(axis='x', rotation=45)
    save_plot(fig, output_filename)


def plot_bar_chart_slo_violations(df_slo_violations: pd.DataFrame, output_filename: str = "slo_violations_bar_chart.png"):
    """
    Gera um gráfico de barras para comparar a frequência de violação de SLOs
    por setup de isolamento.

    Args:
        df_slo_violations (pd.DataFrame): DataFrame com colunas 'isolation_setup' e 'violation_percentage'.
                                          Ex: {'isolation_setup': ['Vanilla', 'NP', 'Kata'], 'violation_percentage': [90, 40, 10]}
        output_filename (str): Nome do arquivo para salvar o plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='isolation_setup', y='violation_percentage', data=df_slo_violations, ax=ax, palette='coolwarm')
    ax.set_title("Porcentagem de Violações de SLO por Configuração de Isolamento")
    ax.set_xlabel("Configuração de Isolamento")
    ax.set_ylabel("Porcentagem de Violações de SLO (%)")
    ax.set_ylim(0, 100)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    save_plot(fig, output_filename)


def plot_violinplot_causality_across_rounds_and_setups(
    df_aggregated_results: pd.DataFrame, 
    source_col: str, 
    target_col: str, 
    causality_value_col: str = 'te_value',
    title: str = "Distribuição da Causalidade por Fase e Configuração de Isolamento",
    output_filename: str = "violinplot_causality_comparison.png"
):
    """
    Gera um violin plot para comparar a distribuição de valores de causalidade
    (TE/CCM) através de diferentes fases e/ou setups de isolamento.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    df_filtered = df_aggregated_results # Assumindo que já está filtrado ou pronto para uso
    
    x_col = 'phase_label'
    if 'experiment_name' in df_filtered.columns:
        df_filtered['phase_setup_display'] = df_filtered['phase_label'] + ' (' + df_filtered['experiment_name'] + ')'
        x_col = 'phase_setup_display'

    # AQUI ESTÁ A MUDANÇA: sns.violinplot em vez de sns.boxplot
    sns.violinplot(x=x_col, y=causality_value_col, data=df_filtered, ax=ax, palette="colorblind", inner="quartile") 
    # inner="quartile" adiciona as linhas dos quartis e mediana dentro do violino, combinando o melhor dos dois mundos.
    # sns.stripplot(x=x_col, y=causality_value_col, data=df_filtered, color=".3", size=4, jitter=True, ax=ax) # Se quiser ver os pontos individuais

    ax.set_title(f"{title}\n({source_col} -> {target_col})")
    ax.set_xlabel("Fase e Configuração de Isolamento")
    ax.set_ylabel(f"{causality_value_col.replace('_', ' ').title()}")
    ax.tick_params(axis='x', rotation=45, ha='right')
    save_plot(fig, output_filename)


# --- Exemplo de uso (para testar o módulo diretamente) ---
if __name__ == "__main__":
    print("Executando exemplos de plots. Verifique a pasta 'output/plots'.")

    data_corr = np.random.rand(5, 5)
    df_corr = pd.DataFrame(data_corr, columns=[f'M{i}' for i in range(5)], index=[f'M{i}' for i in range(5)])
    np.fill_diagonal(df_corr.values, 1)
    plot_correlation_heatmap(df_corr, output_filename="example_correlation_heatmap.png")

    timestamps = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='5S'))
    series_a = pd.Series(np.random.rand(100).cumsum(), index=timestamps, name='CPU_TenantA')
    series_b = pd.Series(np.random.rand(100).cumsum() + np.roll(series_a.values, 3) * 0.5, index=timestamps, name='Latency_TenantB')
    
    mock_ccf_results = {
        'lags': list(range(-10, 11)),
        'correlation_values': np.random.rand(21) * 0.8 - 0.4,
        'conf_interval_upper': 0.3,
        'conf_interval_lower': -0.3
    }
    mock_ccf_results['correlation_values'][13] = 0.7

    plot_ccf(series_a, series_b, mock_ccf_results, scrape_interval_s=5, output_filename="example_ccf_plot.png")

    plot_aligned_time_series(series_a.iloc[:90].copy(), series_b.iloc[:90].copy(), output_filename="example_aligned_time_series_plot.png")

    data_dtw = np.random.rand(4, 4) * 10
    df_dtw = pd.DataFrame(data_dtw, columns=[f'S{i}' for i in range(4)], index=[f'S{i}' for i in range(4)])
    np.fill_diagonal(df_dtw.values, 0)
    plot_dtw_distance_heatmap(df_dtw, output_filename="example_dtw_distance_heatmap.png")

    data_causality = np.random.rand(3, 3)
    df_causality = pd.DataFrame(data_causality, columns=[f'T{i}' for i in range(3)], index=[f'T{i}' for i in range(3)])
    plot_causality_heatmap(df_causality, output_filename="example_causality_heatmap.png")

    df_tv_causality = pd.DataFrame({
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01 10:00:00', periods=50, freq='30S')),
        'te_value': np.sin(np.linspace(0, 4*np.pi, 50)) * 0.4 + 0.5
    })
    plot_time_varying_causality(
        df_tv_causality, 'CPU_TenantA', 'Latency_TenantB', 'te_value',
        output_filename="example_time_varying_causality.png",
        phase_start_time=df_tv_causality['timestamp'].min()
    )

    mock_agg_results = pd.DataFrame({
        'experiment_run_id': [f'run_{i}' for i in range(10)] * 3,
        'round_number': [f'R{i}' for i in range(1, 11)] * 3,
        'phase_label': ['baseline'] * 10 + ['attack'] * 10 + ['recovery'] * 10,
        'experiment_name': ['Vanilla'] * 15 + ['Kata'] * 15,
        'source_metric': ['cpu_usage_tenantA'] * 30,
        'target_metric': ['p99_latency_tenantB'] * 30,
        'te_value': np.random.rand(30) * 0.2 + 0.1
    })
    mock_agg_results.loc[(mock_agg_results['phase_label'] == 'attack') & (mock_agg_results['experiment_name'] == 'Vanilla'), 'te_value'] += np.random.rand(5) * 0.6 + 0.4
    mock_agg_results.loc[(mock_agg_results['phase_label'] == 'attack') & (mock_agg_results['experiment_name'] == 'Kata'), 'te_value'] += np.random.rand(5) * 0.3 + 0.1

    plot_boxplot_causality_across_rounds_and_setups(
        mock_agg_results, ('cpu_usage_tenantA', 'p99_latency_tenantB'), 
        output_filename="example_boxplot_comparison.png",
        title="TE (CPU A -> Latency B) por Fase e Setup"
    )

    df_slo_violations = pd.DataFrame({
        'isolation_setup': ['Vanilla', 'Vanilla c/NP', 'Kata Containers', 'vCluster'],
        'violation_percentage': [95, 60, 20, 5]
    })
    plot_bar_chart_slo_violations(df_slo_violations, output_filename="example_slo_violations.png")

    df_cosine_similarity = pd.DataFrame({
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01 10:00:00', periods=50, freq='30S')),
        'cosine_similarity': np.cos(np.linspace(0, 4*np.pi, 50)) * 0.4 + 0.5
    })
    plot_time_varying_cosine_similarity(
        df_cosine_similarity, 'CPU_TenantA', 'Latency_TenantB',
        output_filename="example_time_varying_cosine_similarity.png",
        phase_start_time=df_cosine_similarity['timestamp'].min()
    )

    similarity_matrix = np.random.rand(5, 5)
    df_similarity = pd.DataFrame(similarity_matrix, columns=[f'M{i}' for i in range(5)], index=[f'M{i}' for i in range(5)])
    plot_cosine_similarity_heatmap(df_similarity, output_filename="example_cosine_similarity_heatmap.png")

    print("\nTodos os exemplos de plots gerados na pasta 'output/plots'.")